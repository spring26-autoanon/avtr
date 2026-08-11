import ast
import asyncio
import os
import textwrap
from types import SimpleNamespace

from serving_exp.patches import e1_silent_backstop as e1


def test_flag_replacement_contains_anchor_prefix():
    assert e1.REPLACEMENT_FLAG.startswith(e1.ANCHOR_FLAG)


def test_timer_replacement_contains_anchor_prefix():
    assert e1.REPLACEMENT_TIMER.startswith(e1.ANCHOR_TIMER)


def test_flag_replacement_is_valid_python():
    ast.parse(
        "if True:\n" + textwrap.indent(textwrap.dedent(e1.REPLACEMENT_FLAG), "    ")
    )


def test_timer_replacement_is_valid_python():
    ast.parse(
        "if True:\n" + textwrap.indent(textwrap.dedent(e1.REPLACEMENT_TIMER), "    ")
    )


def test_backstop_is_env_gated_and_logged():
    assert "MOSHI_BACKSTOP" in e1.REPLACEMENT_TIMER
    assert "[Backstop] engaged" in e1.REPLACEMENT_TIMER


def test_question_gate_present():
    assert 'endswith("?")' in e1.REPLACEMENT_TIMER


def test_fired_flag_set_in_flag_patch():
    assert "self._backstop_fired = True" in e1.REPLACEMENT_FLAG


def test_apply_simulation_no_duplicate_loops():
    # Stock fire-detection block, verbatim from anchors.md section B.
    stock_fire = (
        "            # Text / RAG triggers.\n"
        "            if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:\n"
        '                self._log.info("[RAG] model emitted RAG token, triggering reference generation")\n'
        "                assert self._task_group is not None\n"
        '                await self._send_turn_outputs([("[RET]", "model")])\n'
        "                await self.stt.flush()\n"
        "                await self.rag_manager.trigger(\n"
        "                    task_group=self._task_group,\n"
        "                    wait_steps=self.server.stt_wait_steps,\n"
        "                    handle_reference_fn=self._handle_reference_text,\n"
        "                    context_provider=self.turn_manager.get_context,\n"
        "                )\n"
    )
    patched_fire = stock_fire.replace(e1.ANCHOR_FLAG, e1.REPLACEMENT_FLAG, 1)
    assert patched_fire != stock_fire  # anchor matched
    assert patched_fire.count("self._backstop_fired = True") == 1
    for line in (
        "if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:",
        '[RAG] model emitted RAG token, triggering reference generation',
        "assert self._task_group is not None",
        'await self._send_turn_outputs([("[RET]", "model")])',
        "await self.stt.flush()",
        "await self.rag_manager.trigger(",
    ):
        assert patched_fire.count(line) == 1

    # Stock _stt_recv_loop block containing anchors.md section D's line.
    stock_stt = (
        "    async def _stt_recv_loop(self):\n"
        "        async for msg in self.stt:\n"
        "            if True:\n"
        "                await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))\n"
    )
    patched_stt = stock_stt.replace(e1.ANCHOR_TIMER, e1.REPLACEMENT_TIMER, 1)
    assert patched_stt != stock_stt  # anchor matched
    assert patched_stt.count("[Backstop] engaged") == 1
    assert patched_stt.count("rag_manager.trigger(") == 1
    assert (
        patched_stt.count(
            "await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))"
        )
        == 1
    )


# ---------------------------------------------------------------------------
# Runtime-semantics tests: extract REPLACEMENT_TIMER's appended block (i.e.
# everything after the anchor line) and actually execute it under real
# asyncio against a fake Channel, instead of only asserting on the source
# string. No box access needed — this is pure stdlib.
# ---------------------------------------------------------------------------


class _FakeLog:
    def __init__(self):
        self.calls = []

    def info(self, msg):
        self.calls.append(msg)


class _FakeRagManager:
    def __init__(self):
        self.calls = []

    async def trigger(self, **kw):
        self.calls.append(kw)


class _FakeTurnManager:
    def get_context(self):
        return {}


class _FakeChannel:
    def __init__(self):
        self._log = _FakeLog()
        self.rag_manager = _FakeRagManager()
        self.turn_manager = _FakeTurnManager()
        self.server = object()
        self._handle_reference_text = lambda *a, **kw: None
        self._task_group = None  # set by the test to a real asyncio.TaskGroup


def _build_run():
    """Extract everything in REPLACEMENT_TIMER after the anchor line, dedent
    it, and wrap it as `async def _run(self, msg): ...` so it can be executed
    directly against a fake `self`."""
    appended = e1.REPLACEMENT_TIMER[len(e1.ANCHOR_TIMER):]
    body = textwrap.indent(textwrap.dedent(appended), "    ")
    src = "async def _run(self, msg):\n" + body
    ns = {"asyncio": asyncio, "os": os}
    exec(compile(src, "<e1_timer_runtime>", "exec"), ns)
    return ns["_run"]


def test_runtime_backstop_engages_on_unfired_question(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP", "1")
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
    assert len(fake.rag_manager.calls) == 1
    assert fake.rag_manager.calls[0]["wait_steps"] == 0
    assert fake._log.calls.count("[Backstop] engaged") == 1
    assert fake._backstop_fired is True


def test_runtime_stale_generation_noop(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "0.05")
    run = _build_run()

    async def scenario():
        fake = _FakeChannel()
        msg1 = SimpleNamespace(text="First question?")
        msg2 = SimpleNamespace(text="Second question?")
        async with asyncio.TaskGroup() as tg:
            fake._task_group = tg
            await run(fake, msg1)
            await asyncio.sleep(0.001)
            await run(fake, msg2)
        return fake

    fake = asyncio.run(scenario())
    assert len(fake.rag_manager.calls) == 1


def test_runtime_native_fire_suppresses(monkeypatch):
    monkeypatch.setenv("MOSHI_BACKSTOP", "1")
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "0.01")
    run = _build_run()

    async def scenario():
        fake = _FakeChannel()
        msg = SimpleNamespace(text="Is this real?")
        async with asyncio.TaskGroup() as tg:
            fake._task_group = tg
            await run(fake, msg)
            # simulate a native <ret> fire arriving before the timer expires
            fake._backstop_fired = True
        return fake

    fake = asyncio.run(scenario())
    assert fake.rag_manager.calls == []


def test_runtime_malformed_delay_survives(monkeypatch):
    # The delay is parsed EARLY (synchronously, before the check task is
    # created), guarded by try/except — a typo'd env var must fall back to
    # the 2.5s default instead of raising inside the TaskGroup task (which
    # would propagate as an ExceptionGroup and tear down every sibling loop).
    monkeypatch.setenv("MOSHI_BACKSTOP_DELAY", "oops")
    ns = {"_os": os}
    exec(compile(textwrap.dedent(e1._DELAY_PARSE), "<delay_parse>", "exec"), ns)
    assert ns["_delay"] == 2.5
