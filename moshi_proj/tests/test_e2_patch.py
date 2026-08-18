import ast
import asyncio
import contextlib
import logging
import os
import textwrap
import time
from types import SimpleNamespace

from serving_exp.patches import e2_forced_ret as e2


@contextlib.contextmanager
def _capture_backstop_logs():
    """Attach a real logging.Handler to the "backstop" logger (the one
    `_bs_log` in the hook writes to) and yield the list of captured
    messages, in order."""
    caught = []

    class _Handler(logging.Handler):
        def emit(self, record):
            caught.append(record.getMessage())

    backstop_logger = logging.getLogger("backstop")
    handler = _Handler()
    backstop_logger.addHandler(handler)
    backstop_logger.setLevel(logging.INFO)
    try:
        yield caught
    finally:
        backstop_logger.removeHandler(handler)


def test_replacement_contains_anchor_prefix():
    assert e2.REPLACEMENT.startswith(e2.ANCHOR)


def test_replacement_is_valid_python():
    ast.parse("if True:\n" + textwrap.indent(textwrap.dedent(e2.REPLACEMENT), "    "))


def test_single_token_only():
    # the hook only ever writes the rag token onto the sampled slot — no
    # scripted greeting text, no encode() call building a token sequence.
    assert e2.REPLACEMENT.count("rag_token_id") >= 1
    assert "encode(" not in e2.REPLACEMENT


def test_env_gated_and_logged():
    assert "MOSHI_BACKSTOP_FORCE" in e2.REPLACEMENT
    assert "[Backstop] engaged" in e2.REPLACEMENT


def test_question_gate_present():
    assert 'endswith("?")' in e2.REPLACEMENT


def test_no_direct_trigger_call():
    # E2 forces the model's OWN fire path via the hook; it must never call
    # rag_manager.trigger directly the way e1's timer does — that would be a
    # second, uncoordinated injection path.
    assert "rag_manager.trigger" not in e2.REPLACEMENT


def test_apply_simulation_no_duplicate_loops():
    # Stock _stt_recv_loop block containing anchors.md section D's line.
    stock_stt = (
        "    async def _stt_recv_loop(self):\n"
        "        async for msg in self.stt:\n"
        "            if True:\n"
        "                await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))\n"
    )
    patched_stt = stock_stt.replace(e2.ANCHOR, e2.REPLACEMENT, 1)
    assert patched_stt != stock_stt  # anchor matched
    assert "MOSHI_BACKSTOP_FORCE" in patched_stt
    assert (
        patched_stt.count(
            "await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))"
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Runtime-semantics tests: extract REPLACEMENT's appended block (everything
# after the anchor line), exec it as `async def _run(self, msg): ...` against
# a fake Channel/server, and drive it under real asyncio. Style matches
# tests/test_e1_patch.py's runtime section.
# ---------------------------------------------------------------------------

RAG_ID = 4242


class _FakeLog:
    def __init__(self):
        self.calls = []

    def info(self, msg):
        self.calls.append(msg)


class _FakeLmModel:
    rag_token_id = RAG_ID


class _FakeLmGen:
    def __init__(self):
        self.lm_model = _FakeLmModel()
        self.on_text_hook = None


class _FakeRunner:
    def __init__(self):
        self.lm_gen = _FakeLmGen()


class _FakeServer:
    def __init__(self):
        self.runner = _FakeRunner()


class _FakeTurnManager:
    def get_context(self):
        return {}


class _FakeChannel:
    def __init__(self, slot_idx=0):
        self._log = _FakeLog()
        self.server = _FakeServer()
        self.turn_manager = _FakeTurnManager()
        self._handle_reference_text = lambda *a, **kw: None
        self.slot_idx = slot_idx
        self._task_group = None  # set by the test to a real asyncio.TaskGroup


class _FakeTok:
    """Wraps a raw int so `.item()` matches the real torch scalar API."""

    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _FakeTensor:
    """List-like batch text-token tensor: __getitem__/__setitem__ + .item()."""

    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, i):
        return _FakeTok(self._values[i])

    def __setitem__(self, i, v):
        self._values[i] = v

    def __len__(self):
        return len(self._values)


def _build_run():
    """Extract everything in REPLACEMENT after the anchor line, dedent it,
    and wrap it as `async def _run(self, msg): ...` so it can be executed
    directly against a fake `self`."""
    appended = e2.REPLACEMENT[len(e2.ANCHOR) :]
    body = textwrap.indent(textwrap.dedent(appended), "    ")
    src = "async def _run(self, msg):\n" + body
    ns = {"asyncio": asyncio, "os": os}
    exec(compile(src, "<e2_timer_runtime>", "exec"), ns)
    return ns["_run"]


def test_runtime_engage_after_delay_sets_pending_force_and_logs(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "0.01")
    run = _build_run()

    async def scenario():
        fake = _FakeChannel()
        msg = SimpleNamespace(text="Is this real?")
        async with asyncio.TaskGroup() as tg:
            fake._task_group = tg
            await run(fake, msg)
        return fake

    fake = asyncio.run(scenario())
    # Armed with a TTL deadline (not a bare True sentinel) so a stale force
    # can't outlive its window and hijack a later turn/session.
    deadline = fake.server._e2_force.get(fake.slot_idx)
    assert deadline is not None
    assert isinstance(deadline, float)
    assert deadline > time.monotonic()  # still within its TTL right now
    assert fake._log.calls.count("[Backstop] engaged") == 1


def test_runtime_native_fire_suppresses_forcing(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "0.02")
    run = _build_run()

    async def scenario():
        fake = _FakeChannel()
        msg = SimpleNamespace(text="Is this real?")
        async with asyncio.TaskGroup() as tg:
            fake._task_group = tg
            await run(fake, msg)
            # the model fires natively on its own before the timer expires
            hook = fake.server.runner.lm_gen.on_text_hook
            hook(_FakeTensor([RAG_ID]))
        return fake

    fake = asyncio.run(scenario())
    assert fake.slot_idx not in fake.server._e2_force
    assert "[Backstop] engaged" not in fake._log.calls


class _FakeTaskGroup:
    """Stand-in for asyncio.TaskGroup that never blocks on close — this test
    only needs the hook installed as a side effect of arming the timer, not
    the (arbitrarily long) delayed check itself to run to completion."""

    def __init__(self):
        self.tasks = []

    def create_task(self, coro):
        t = asyncio.ensure_future(coro)
        self.tasks.append(t)
        return t


def test_hook_consumes_on_padding_token_only(monkeypatch):
    # Consume-on-pad: the force lands in the quiet beat before an answer
    # (token == 3, the padding token), never on a word token — this is what
    # makes it stall-proof (a stalled model emitting only padding still gets
    # served) and speech-safe (mid-sentence word tokens are never hijacked).
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "999")
    fake = _FakeChannel()
    lm_model = fake.server.runner.lm_gen.lm_model

    async def scenario():
        msg = SimpleNamespace(text="Is this real?")
        run = _build_run()
        tg = _FakeTaskGroup()
        fake._task_group = tg
        # Arm via the real code path so on_text_hook is installed exactly as
        # production does it, then override state to the pending case.
        await run(fake, msg)
        # Mutate the SAME dict the hook's closure captured (production never
        # rebinds self.server._e2_force, only mutates it in place via
        # `getattr(self.server, "_e2_force", {})`) — rebinding here would
        # desync the test from the hook's closed-over reference. Use a
        # real future deadline (TTL semantics), not a bare True sentinel.
        fake.server._e2_force[0] = time.monotonic() + 10.0
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())
    hook = fake.server.runner.lm_gen.on_text_hook
    assert hook is not None

    # A word token (> 3) with a pending force must pass through untouched —
    # her speech is never hijacked mid-word.
    tok = _FakeTensor([7])
    hook(tok)
    assert tok._values[0] == 7
    assert 0 in fake.server._e2_force  # still pending, not consumed or expired

    # Another special token (0/1/2) also passes through — only ==3 consumes.
    tok = _FakeTensor([1])
    hook(tok)
    assert tok._values[0] == 1
    assert 0 in fake.server._e2_force

    # The padding token (== 3) is overwritten with rag_token_id and the
    # pending entry is cleared.
    tok = _FakeTensor([3])
    hook(tok)
    assert tok._values[0] == lm_model.rag_token_id
    assert 0 not in fake.server._e2_force

    # A subsequent padding token is left untouched — one-shot only.
    tok = _FakeTensor([3])
    hook(tok)
    assert tok._values[0] == 3


def test_hook_does_not_rewrite_past_ttl_deadline(monkeypatch):
    # A force armed with an already-expired deadline must never rewrite a
    # token, even the padding token — and the stale entry must be removed so
    # it doesn't linger. Installs the hook via the real REPLACEMENT code path
    # (not a hand-rolled copy) so this exercises the actual implementation.
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "999")
    run = _build_run()
    fake = _FakeChannel()

    async def scenario():
        tg = _FakeTaskGroup()
        fake._task_group = tg
        msg = SimpleNamespace(text="Is this real?")
        await run(fake, msg)  # installs on_text_hook as a side effect
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())
    fake.server._e2_force[fake.slot_idx] = time.monotonic() - 1.0  # expired
    hook = fake.server.runner.lm_gen.on_text_hook

    tok = _FakeTensor([3])  # padding, would have been rewritten if live
    hook(tok)
    assert tok._values[0] == 3  # untouched — deadline had already passed
    assert fake.slot_idx not in fake.server._e2_force  # stale entry removed


def test_runtime_native_fire_while_force_pending_clears_and_suppresses(monkeypatch):
    # Regression: a native fire arriving WHILE a force is pending must clear
    # the pending entry and record the native flag — not leave the pending
    # entry to fire a duplicate forced <ret> on the very next padding frame
    # (the E1 "screech mode" this whole design exists to avoid).
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "999")
    run = _build_run()
    fake = _FakeChannel()

    async def scenario():
        tg = _FakeTaskGroup()
        fake._task_group = tg
        msg = SimpleNamespace(text="Is this real?")
        await run(fake, msg)  # installs on_text_hook as a side effect
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())
    # Arm a fresh (unexpired) pending force directly, as the delayed check
    # would once it engages.
    fake.server._e2_force[fake.slot_idx] = time.monotonic() + 10.0
    hook = fake.server.runner.lm_gen.on_text_hook

    # The model fires natively while the force is still pending.
    tok = _FakeTensor([RAG_ID])
    hook(tok)
    assert tok._values[0] == RAG_ID  # already a ret — must not be touched
    assert fake.slot_idx not in fake.server._e2_force  # pending entry cleared
    # v4: no more boolean flag — a timestamp, recorded just now.
    ts = fake.server._e2_native_ts.get(fake.slot_idx)
    assert ts is not None
    assert 0.0 <= time.monotonic() - ts < 1.0

    # The next padding frame must NOT be rewritten — no duplicate ret.
    tok = _FakeTensor([3])
    hook(tok)
    assert tok._values[0] == 3


def test_ttl_env_respected(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "0.01")
    monkeypatch.setenv("MOSHI_BACKSTOP_TTL", "0.5")
    run = _build_run()

    async def scenario():
        fake = _FakeChannel()
        msg = SimpleNamespace(text="Is this real?")
        async with asyncio.TaskGroup() as tg:
            fake._task_group = tg
            await run(fake, msg)
        return fake

    before = time.monotonic()
    fake = asyncio.run(scenario())
    after = time.monotonic()
    deadline = fake.server._e2_force[fake.slot_idx]
    # deadline was set to monotonic() + 0.5 at engage time, somewhere between
    # `before` and `after` (the delay + scheduling window around it).
    assert before + 0.5 <= deadline <= after + 0.5 + 0.05


def test_ttl_malformed_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_TTL", "oops")
    ns = {"_os": os}
    exec(compile(textwrap.dedent(e2._TTL_PARSE), "<ttl_parse>", "exec"), ns)
    assert ns["_ttl"] == 8.0


def test_runtime_stale_force_cleared_by_new_question(monkeypatch):
    # A force armed for Q1 (still within its TTL) must not survive into Q2 —
    # question-time clearing pops any pending force before rearming.
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "999")
    run = _build_run()

    async def scenario():
        fake = _FakeChannel()
        tg = _FakeTaskGroup()
        fake._task_group = tg
        msg1 = SimpleNamespace(text="First question?")
        await run(fake, msg1)
        # Simulate Q1's timer having engaged and armed a live (unexpired)
        # force, still pending because Q1's word token never arrived.
        fake.server._e2_force[fake.slot_idx] = time.monotonic() + 10.0
        msg2 = SimpleNamespace(text="Second question?")
        await run(fake, msg2)  # question-time clearing must pop Q1's force
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        return fake

    fake = asyncio.run(scenario())
    hook = fake.server.runner.lm_gen.on_text_hook
    # Q1's stale force must be gone; Q2's next padding frame is NOT
    # hijacked by it (Q2's own timer hasn't engaged yet at this point).
    tok = _FakeTensor([3])
    hook(tok)
    assert tok._values[0] == 3


def test_runtime_malformed_delay_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "oops")
    ns = {"_os": os}
    exec(compile(textwrap.dedent(e2._DELAY_PARSE), "<delay_parse>", "exec"), ns)
    assert ns["_delay"] == 2.5


# ---------------------------------------------------------------------------
# v4: FIX 1 — mid-question native fires count (timestamp, not flag)
# ---------------------------------------------------------------------------


def test_new_env_vars_present():
    assert "MOSHI_BACKSTOP_LOOKBACK" in e2.REPLACEMENT
    assert "MOSHI_BACKSTOP_REFRACTORY" in e2.REPLACEMENT


def test_attribution_log_strings_present():
    assert "forced ret consumed" in e2.REPLACEMENT
    assert "native fire observed" in e2.REPLACEMENT


def test_runtime_mid_question_native_fire_counts(monkeypatch):
    # Field bug: a native <ret> fired WHILE the user is still speaking —
    # before the terminal "?" ever arrives to arm the timer — used to be
    # erased by arming's flag reset, causing a duplicate forced fire. v4
    # replaces the flag with a timestamp that arming never clears.
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "999")
    run = _build_run()

    async def install_and_fire():
        fake = _FakeChannel()
        tg = _FakeTaskGroup()
        fake._task_group = tg
        msg1 = SimpleNamespace(text="First question?")
        await run(fake, msg1)  # installs the hook (huge delay; never engages)
        # Simulate a native fire landing mid-utterance, before any "?" of
        # THIS question has arrived — exactly what the hook does when the
        # model samples its own rag token off the wire.
        hook = fake.server.runner.lm_gen.on_text_hook
        hook(_FakeTensor([RAG_ID]))
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        return fake

    fake = asyncio.run(install_and_fire())
    ts_before = fake.server._e2_native_ts[fake.slot_idx]
    assert ts_before is not None

    # Now the "?" for this question arrives and arms a fresh timer — arming
    # must NOT clear the native ts recorded a moment ago.
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "0.01")

    async def arm_and_wait():
        msg2 = SimpleNamespace(text="Is this real?")
        async with asyncio.TaskGroup() as tg2:
            fake._task_group = tg2
            await run(fake, msg2)
        return fake

    fake = asyncio.run(arm_and_wait())
    assert fake.server._e2_native_ts[fake.slot_idx] == ts_before  # untouched by arming
    assert fake.slot_idx not in fake.server._e2_force  # suppressed — never engaged
    assert "[Backstop] engaged" not in fake._log.calls


def test_runtime_stale_native_ts_does_not_suppress(monkeypatch):
    # A native fire older than the lookback window must NOT suppress this
    # question's backstop — only a RECENT native fire counts.
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "0.01")
    monkeypatch.setenv("MOSHI_BACKSTOP_LOOKBACK", "0.05")
    run = _build_run()

    async def scenario():
        fake = _FakeChannel()
        msg = SimpleNamespace(text="Is this real?")
        async with asyncio.TaskGroup() as tg:
            fake._task_group = tg
            await run(fake, msg)  # installs hook + dicts
            fake.server._e2_native_ts[fake.slot_idx] = time.monotonic() - 5.0
        return fake

    fake = asyncio.run(scenario())
    assert fake.server._e2_force.get(fake.slot_idx) is not None
    assert "[Backstop] engaged" in fake._log.calls


def test_lookback_malformed_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_LOOKBACK", "oops")
    ns = {"_os": os}
    exec(compile(textwrap.dedent(e2._LOOKBACK_PARSE), "<lookback_parse>", "exec"), ns)
    assert ns["_lookback"] == 8.0


# ---------------------------------------------------------------------------
# v4: FIX 2 — refractory window on arming
# ---------------------------------------------------------------------------


def test_runtime_refractory_skips_arming(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "0.01")
    monkeypatch.setenv("MOSHI_BACKSTOP_REFRACTORY", "60")
    run = _build_run()

    async def scenario():
        fake = _FakeChannel()
        tg = _FakeTaskGroup()
        fake._task_group = tg
        msg0 = SimpleNamespace(text="Warm up?")
        await run(fake, msg0)  # installs hook + dicts, arms (no prior consumption)
        # Simulate the backstop having just consumed a forced ret.
        fake.server._e2_forced_ts[fake.slot_idx] = time.monotonic()
        msg = SimpleNamespace(text="Is this real?")
        await run(fake, msg)  # arming attempt within the refractory window
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        return fake

    fake = asyncio.run(scenario())
    assert "[Backstop] refractory skip" in fake._log.calls
    assert fake.slot_idx not in fake.server._e2_force  # no pending/engaged force from the skipped arm
    assert len(fake._task_group.tasks) == 1  # only msg0's check task — msg's arm was skipped


def test_runtime_refractory_does_not_block_native_fires(monkeypatch):
    # Native fires are never blocked by the refractory window — it only
    # gates the backstop's OWN arming.
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "999")
    run = _build_run()
    fake = _FakeChannel()

    async def scenario():
        tg = _FakeTaskGroup()
        fake._task_group = tg
        msg = SimpleNamespace(text="Is this real?")
        await run(fake, msg)  # installs hook
        fake.server._e2_forced_ts[fake.slot_idx] = time.monotonic()  # inside refractory
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())
    hook = fake.server.runner.lm_gen.on_text_hook
    tok = _FakeTensor([RAG_ID])
    hook(tok)
    assert tok._values[0] == RAG_ID  # untouched, native fire always passes through
    assert fake.server._e2_native_ts.get(fake.slot_idx) is not None  # still recorded


def test_refractory_malformed_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_REFRACTORY", "oops")
    ns = {"_os": os}
    exec(compile(textwrap.dedent(e2._REFRACTORY_PARSE), "<refractory_parse>", "exec"), ns)
    assert ns["_refractory"] == 6.0


# ---------------------------------------------------------------------------
# v4: FIX 3 — attribution logging in the hook
# ---------------------------------------------------------------------------


def _install_hook_with_pending(monkeypatch, delay="999"):
    """Shared setup for the attribution tests below: install the hook (via
    the real REPLACEMENT code path) and arm a live pending force on slot 0,
    cancelling the scheduled check so it never fires on its own. Returns the
    fake channel."""
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", delay)
    run = _build_run()
    fake = _FakeChannel()

    async def scenario():
        tg = _FakeTaskGroup()
        fake._task_group = tg
        msg = SimpleNamespace(text="Is this real?")
        await run(fake, msg)
        fake.server._e2_force[fake.slot_idx] = time.monotonic() + 10.0
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())
    return fake


def test_hook_logs_attribution_on_consume_and_native_fire(monkeypatch):
    fake = _install_hook_with_pending(monkeypatch)
    hook = fake.server.runner.lm_gen.on_text_hook

    with _capture_backstop_logs() as caught:
        hook(_FakeTensor([3]))  # consumes the pending force

    assert any("forced ret consumed" in m for m in caught)
    # Tightened (v4.1 regression pin): a forced consumption must NOT also
    # be mislabeled as a native fire — the rewritten token (now == rag_id,
    # no longer pending) used to fall through into the second (no-pending)
    # loop and get logged/stamped a second time as a distinct native event.
    assert not any("native fire observed" in m for m in caught)


def test_forced_consumption_does_not_stamp_native_ts(monkeypatch):
    # (a) Forced consumption produces ONLY "forced ret consumed" — no
    # "native fire observed" — and _e2_native_ts is left untouched.
    fake = _install_hook_with_pending(monkeypatch)
    hook = fake.server.runner.lm_gen.on_text_hook
    assert fake.slot_idx not in fake.server._e2_native_ts  # nothing recorded yet

    with _capture_backstop_logs() as caught:
        hook(_FakeTensor([3]))  # consumes the pending force

    assert sum("forced ret consumed" in m for m in caught) == 1
    assert not any("native fire observed" in m for m in caught)
    assert fake.slot_idx not in fake.server._e2_native_ts  # still untouched
    assert fake.server._e2_forced_ts.get(fake.slot_idx) is not None  # consumption WAS stamped


def test_native_while_pending_logs_and_stamps_exactly_once(monkeypatch):
    # (b) A native fire while a force is pending must log "native fire
    # observed" exactly once and stamp _e2_native_ts exactly once — not
    # once from the pending-clear branch and again from the no-pending scan
    # (the rewritten... well here the token is left untouched, but the slot
    # leaves `_p` either way, so without the `_handled` guard it would
    # double-match).
    fake = _install_hook_with_pending(monkeypatch)
    hook = fake.server.runner.lm_gen.on_text_hook

    with _capture_backstop_logs() as caught:
        tok = _FakeTensor([RAG_ID])
        hook(tok)

    assert tok._values[0] == RAG_ID  # untouched — already a ret
    assert sum("native fire observed" in m for m in caught) == 1
    assert not any("forced ret consumed" in m for m in caught)
    assert fake.slot_idx not in fake.server._e2_force  # pending cleared
    assert fake.server._e2_native_ts.get(fake.slot_idx) is not None


def test_genuine_no_pending_native_fire_logs_and_stamps(monkeypatch):
    # (c) An ordinary native fire with no pending force at all (the common
    # case — the model answers on its own before any backstop timer ever
    # engages) must still log and stamp normally; the _handled guard must
    # not suppress this legitimate case.
    monkeypatch.setenv("MOSHI_BACKSTOP_FORCE", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "999")
    run = _build_run()
    fake = _FakeChannel()

    async def scenario():
        tg = _FakeTaskGroup()
        fake._task_group = tg
        msg = SimpleNamespace(text="Is this real?")
        await run(fake, msg)  # installs the hook; no pending force yet (delay=999)
        for t in tg.tasks:
            t.cancel()
        for t in tg.tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    asyncio.run(scenario())
    hook = fake.server.runner.lm_gen.on_text_hook
    assert fake.slot_idx not in fake.server._e2_force  # confirm: no pending

    with _capture_backstop_logs() as caught:
        tok = _FakeTensor([RAG_ID])
        hook(tok)

    assert tok._values[0] == RAG_ID
    assert sum("native fire observed" in m for m in caught) == 1
    assert fake.server._e2_native_ts.get(fake.slot_idx) is not None
