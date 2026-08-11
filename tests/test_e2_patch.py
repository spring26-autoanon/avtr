import ast
import asyncio
import os
import textwrap
import time
from types import SimpleNamespace

from serving_exp.patches import e2_forced_ret as e2


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


def test_hook_consumes_pending_force_exactly_once(monkeypatch):
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

    # A non-word token (<= 3) must not be consumed.
    tok = _FakeTensor([2])
    hook(tok)
    assert tok._values[0] == 2
    assert 0 in fake.server._e2_force  # still pending, not consumed or expired

    # First word token (> 3) is overwritten with rag_token_id and the pending
    # entry is cleared.
    tok = _FakeTensor([100])
    hook(tok)
    assert tok._values[0] == lm_model.rag_token_id
    assert 0 not in fake.server._e2_force

    # A subsequent word token is left untouched — one-shot only.
    tok = _FakeTensor([200])
    hook(tok)
    assert tok._values[0] == 200


def test_hook_does_not_rewrite_past_ttl_deadline(monkeypatch):
    # A force armed with an already-expired deadline must never rewrite a
    # token, even a word token — and the stale entry must be removed so it
    # doesn't linger. Installs the hook via the real REPLACEMENT code path
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

    tok = _FakeTensor([100])  # a word token, would have been rewritten if live
    hook(tok)
    assert tok._values[0] == 100  # untouched — deadline had already passed
    assert fake.slot_idx not in fake.server._e2_force  # stale entry removed


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
    # Q1's stale force must be gone; Q2's first word token is NOT hijacked.
    tok = _FakeTensor([100])
    hook(tok)
    assert tok._values[0] == 100


def test_runtime_malformed_delay_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "oops")
    ns = {"_os": os}
    exec(compile(textwrap.dedent(e2._DELAY_PARSE), "<delay_parse>", "exec"), ns)
    assert ns["_delay"] == 2.5
