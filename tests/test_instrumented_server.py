"""
Tests for scripts/instrumented_server.py's SessionLog/_ChannelState/
_push_instrumentation — pure turn-tracking logic with no moshi/torch
dependency (those imports are lazy, inside the _patch_* functions and
main(), so this module imports cleanly without moshi-rag installed).

Does NOT test five of the six monkeypatches (_patch_server_state,
_patch_channel, _patch_rag_manager_trigger, _patch_turn_manager,
_patch_rag_manager_background_task) — those attach to real moshi-rag
classes that aren't available here, and mocking them would just encode
assumptions about their behavior rather than verify against it. That's
what VM verification is for (see specs/moshirag-evals-requirements.md's
Demo section, item 8 of the work plan).

_patch_rag_manager_get_reference_text IS tested below, unlike the other
five — it attaches to a plain, duck-typed RAGManager-shaped object and
imports no moshi-rag module at all (only core/retrieval_backend.py's
format_context()), so it's genuinely exercisable without a real moshi-rag
install, same discipline as core/model_interface.py's _TimedInferenceJob
tests. _build_retrieval_backend_for_demo() is tested for the same reason —
pure core.config/core.retrieval_backend, no moshi import.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from scripts.instrumented_server import (
    SessionLog,
    _ChannelState,
    _build_retrieval_backend_for_demo,
    _patch_rag_manager_get_reference_text,
    _push_instrumentation,
)


class _FakeSession:
    """Minimal stand-in for SessionLog, for testing _ChannelState in isolation."""

    def __init__(self):
        self.written = []
        self.raw_written = []

    def now_iso(self):
        return "2026-01-01T00:00:00Z"

    def write_turn(self, record):
        self.written.append(record)

    def write_raw_event(self, event, *, turn_index, **fields):
        self.raw_written.append({"event": event, "turn_index": turn_index, **fields})


def _fake_channel():
    channel = AsyncMock()
    channel.ws.send_bytes = AsyncMock()
    return channel


# ── SessionLog ────────────────────────────────────────────────────────────


def test_session_log_writes_session_start(tmp_path):
    session = SessionLog(tmp_path)
    session.write_session_start(
        checkpoint="base",
        retrieval_backend={"model": "gemini-3.5-flash", "base_url": "https://example.com"},
        rag_timeout_s=8.0,
        stt_mode="local",
    )

    lines = (tmp_path / "turns.jsonl").read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["type"] == "session_start"
    assert record["session_id"] == tmp_path.name
    assert record["checkpoint"] == "base"
    assert record["rag_timeout_s"] == 8.0
    assert record["stt_mode"] == "local"
    assert record["retrieval_backend"]["model"] == "gemini-3.5-flash"


def test_session_log_write_turn_appends_compact_jsonl(tmp_path):
    session = SessionLog(tmp_path)
    session.write_turn({"type": "turn", "turn_index": 1})
    session.write_turn({"type": "turn", "turn_index": 2})

    lines = (tmp_path / "turns.jsonl").read_text().splitlines()
    assert len(lines) == 2
    # Each line is exactly one compact JSON value — no embedded newlines.
    assert "\n" not in lines[0]
    assert json.loads(lines[0])["turn_index"] == 1
    assert json.loads(lines[1])["turn_index"] == 2


def test_session_log_writes_raw_events_to_a_separate_file(tmp_path):
    session = SessionLog(tmp_path)
    session.write_raw_event("user_text", turn_index=1, text="hi")
    session.write_raw_event("model_text", turn_index=1, text="hello")

    # Goes to raw_events.jsonl, not turns.jsonl — the two are deliberately
    # separate files (see SessionLog's own docstring for why). turns.jsonl
    # still exists (SessionLog.__init__ opens it unconditionally) but stays
    # empty since write_turn was never called.
    assert (tmp_path / "turns.jsonl").read_text() == ""
    lines = (tmp_path / "raw_events.jsonl").read_text().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["type"] == "raw_event"
    assert record["event"] == "user_text"
    assert record["turn_index"] == 1
    assert record["text"] == "hi"
    assert "t_rel_s" in record


def test_session_log_lookup_maps(tmp_path):
    session = SessionLog(tmp_path)
    channel = _fake_channel()
    channel.rag_manager = object()
    channel.turn_manager = object()

    state = session.register_channel(channel)

    assert session.state_for_channel(channel) is state
    assert session.state_for_rag_manager(channel.rag_manager) is state
    assert session.state_for_turn_manager(channel.turn_manager) is state
    assert session.state_for_channel(object()) is None


def test_unregister_channel_clears_all_three_lookups(tmp_path):
    """Regression test for a real GPU OOM found during VM verification:
    without this cleanup, every closed channel (including every automatic
    reconnect Queue.tsx's retry-on-failure makes) stayed permanently
    referenced by SessionLog for the life of the server process — pinning
    each connection's deepcopied Mimi model (held via Channel.stt) in GPU
    memory forever."""
    session = SessionLog(tmp_path)
    channel = _fake_channel()
    channel.rag_manager = object()
    channel.turn_manager = object()
    session.register_channel(channel)

    session.unregister_channel(channel)

    assert session.state_for_channel(channel) is None
    assert session.state_for_rag_manager(channel.rag_manager) is None
    assert session.state_for_turn_manager(channel.turn_manager) is None


def test_unregister_channel_never_registered_is_a_no_op(tmp_path):
    session = SessionLog(tmp_path)
    channel = _fake_channel()
    channel.rag_manager = object()
    channel.turn_manager = object()

    session.unregister_channel(channel)  # must not raise


# ── _ChannelState: turn boundaries ──────────────────────────────────────────


def test_utterance_end_starts_a_new_turn_with_accumulated_text():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_user_text("What's ")
        state.on_user_text("the capital of France?")
        state.on_utterance_end()

        assert state.turn_index == 1
        assert session.written == []  # nothing flushed yet — this is the first turn
        assert state._pending["user_question_text"] == "What's the capital of France?"

    asyncio.run(run())


def test_second_utterance_end_flushes_the_first_turn():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_user_text("first question")
        state.on_utterance_end()
        state.on_user_text("second question")
        state.on_utterance_end()

        assert len(session.written) == 1
        assert session.written[0]["turn_index"] == 1
        assert session.written[0]["user_question_text"] == "first question"
        assert state.turn_index == 2
        assert state._pending["user_question_text"] == "second question"

    asyncio.run(run())


def test_model_text_accumulates_into_the_pending_turn():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_user_text("first question")
        state.on_utterance_end()  # turn 1 now pending
        state.on_model_text("It's ")
        state.on_model_text("Paris.")
        state.on_user_text("second question")
        state.on_utterance_end()  # flushes turn 1

        assert session.written[0]["user_question_text"] == "first question"
        assert session.written[0]["model_response_text"] == "It's Paris."

    asyncio.run(run())


def test_model_text_before_any_turn_is_a_no_op():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_model_text("Ready for a new idea!")  # session-opening greeting

        assert session.written == []

    asyncio.run(run())


def test_flushed_turn_record_has_no_private_fields():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_utterance_end()
        state.on_utterance_end()  # flushes turn 1

        record = session.written[0]
        assert all(not k.startswith("_") for k in record)

    asyncio.run(run())


def test_flush_final_writes_pending_turn_once():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_utterance_end()
        state.flush_final()
        state.flush_final()  # second call must be a no-op, not a duplicate write

        assert len(session.written) == 1

    asyncio.run(run())


def test_flush_final_with_no_pending_turn_is_a_no_op():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.flush_final()

        assert session.written == []

    asyncio.run(run())


# ── _ChannelState: <ret>/retrieval logging (no pending-turn state at all —
# see _ChannelState's own docstring for why) and ttfat ──────────────────────


def test_ret_triggered_writes_a_raw_event_and_pushes_a_live_update():
    """on_ret_triggered no longer mutates any pending-turn state — it just
    logs the raw fact and pushes a best-effort live update. Reattribution
    to the "correct" turn happens entirely in
    scripts/summarize_demo_session.py, post-session (see
    instrumented_server.py's module docstring and _ChannelState's own
    docstring for why)."""

    async def run():
        session = _FakeSession()
        channel = _fake_channel()
        state = _ChannelState(session, channel)

        state.on_utterance_end()  # turn 1
        state.on_ret_triggered()
        state.on_ret_triggered()  # a second trigger, still just logged as-is

        events = [ev for ev in session.raw_written if ev["event"] == "ret_triggered"]
        assert len(events) == 2
        assert all(ev["turn_index"] == 1 for ev in events)
        # No pending-turn mutation at all — turns.jsonl carries no
        # retrieval fields (see on_utterance_end's pending template).
        assert "rag_triggered" not in state._pending
        assert "rag_trigger_count" not in state._pending

    asyncio.run(run())


def test_retrieval_complete_writes_full_data_to_the_raw_event():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_utterance_end()  # turn 1
        state.on_retrieval_complete(
            context="ctx", reference_text="ref", asr_wait_s=0.12, context_injection_s=0.02, api_call_s=0.71
        )

        events = [ev for ev in session.raw_written if ev["event"] == "retrieval_complete"]
        assert len(events) == 1
        ev = events[0]
        assert ev["turn_index"] == 1
        assert ev["retrieval_context"] == "ctx"
        assert ev["retrieved_reference_text"] == "ref"
        breakdown = ev["retrieval_breakdown_s"]
        assert breakdown["asr_wait_s"] == 0.12
        assert breakdown["api_call_s"] == 0.71
        assert breakdown["context_injection_s"] == 0.02
        assert breakdown["total_s"] == pytest.approx(0.85)

    asyncio.run(run())


def test_retrieval_complete_tags_with_whatever_turn_index_is_live_no_reattribution():
    """Confirmed against real moshi-rag source (channel.py's _output_loop):
    a <ret> trigger's turn_index tag can be one turn behind reality, since
    the model often reacts before our own VAD-based boundary catches up
    (see instrumented_server.py's module docstring). Earlier rounds tried
    to correct this live, in-process; now on_retrieval_complete
    deliberately does *not* try — it just tags with self.turn_index,
    live, even across an intervening turn boundary. Correcting this is
    entirely scripts/summarize_demo_session.py's job, done post-hoc."""

    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_utterance_end()  # turn 1
        state.on_ret_triggered()  # tagged turn 1
        state.on_utterance_end()  # turn 2 — boundary races ahead before retrieval resolves
        state.on_retrieval_complete(
            context="ctx", reference_text="ref", asr_wait_s=0.1, context_injection_s=0.0, api_call_s=0.5
        )

        events = [ev for ev in session.raw_written if ev["event"] == "retrieval_complete"]
        assert events[0]["turn_index"] == 2  # live tag, not turn 1

    asyncio.run(run())


def test_retrieval_complete_never_mutates_the_pending_turn():
    """turns.jsonl's own turn records carry no retrieval fields at all —
    on_retrieval_complete must never touch self._pending."""

    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_utterance_end()
        pending_before = dict(state._pending)
        state.on_retrieval_complete(
            context="ctx", reference_text="ref", asr_wait_s=0.1, context_injection_s=0.0, api_call_s=0.5
        )

        assert state._pending == pending_before

    asyncio.run(run())


def test_retrieval_complete_with_no_pending_turn_still_logs():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_retrieval_complete(
            context="ctx", reference_text="ref", asr_wait_s=0.1, context_injection_s=0.0, api_call_s=0.5
        )

        events = [ev for ev in session.raw_written if ev["event"] == "retrieval_complete"]
        assert len(events) == 1
        assert events[0]["turn_index"] == 0

    asyncio.run(run())


def test_first_audio_records_ttfat_once_only():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_utterance_end()
        state.on_first_audio()
        first_value = state._pending["ttfat_s"]
        assert first_value is not None

        state.on_first_audio()  # a later frame in the same turn
        assert state._pending["ttfat_s"] == first_value  # unchanged, not overwritten

    asyncio.run(run())


def test_model_text_triggers_first_audio_automatically():
    """The real fix (found validating latency.ttfat on the batch path, then
    applied here by analogy): ttfat_s's first-audio signal now comes from
    the first genuine (non-pad) model text token via on_model_text, not a
    separate pcm-presence check — see the module docstring's
    TurnManager.handle_spoken_text bullet for why."""
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_utterance_end()
        assert state._pending["ttfat_s"] is None

        state.on_model_text("Paris.")

        assert state._pending["ttfat_s"] is not None
        assert any(ev["event"] == "first_audio" for ev in session.raw_written)

    asyncio.run(run())


def test_first_audio_before_any_turn_is_a_no_op():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_first_audio()  # no pending turn — must not raise

    asyncio.run(run())


# ── _ChannelState: raw_events.jsonl stream ──────────────────────────────────


def test_raw_events_recorded_for_a_full_turn_in_chronological_order():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_user_text("What's the capital of France?")
        state.on_utterance_end()
        state.on_ret_triggered()
        state.on_retrieval_complete(
            context="ctx", reference_text="ref", asr_wait_s=0.1, context_injection_s=0.0, api_call_s=0.5
        )
        state.on_model_text("It's Paris.")
        state.on_first_audio()

        kinds = [ev["event"] for ev in session.raw_written]
        assert kinds == [
            "user_text",
            "utterance_end",
            "ret_triggered",
            "retrieval_complete",
            "model_text",
            "first_audio",
        ]
        # user_text arrives before on_utterance_end increments turn_index,
        # so it's tagged with the *upcoming* turn's predecessor (0) — every
        # event from utterance_end onward belongs to turn 1.
        assert session.raw_written[0]["turn_index"] == 0
        assert all(ev["turn_index"] == 1 for ev in session.raw_written[1:])

    asyncio.run(run())


def test_raw_event_ret_triggered_recorded_even_with_no_pending_turn():
    """An out-of-turn/unprompted trigger is exactly what the raw stream
    exists to surface, unlike the aggregated turns.jsonl record which has
    nowhere to put it."""

    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_ret_triggered()  # no pending turn

        assert session.written == []  # nothing in the aggregated log
        assert [ev["event"] for ev in session.raw_written] == ["ret_triggered"]

    asyncio.run(run())


def test_raw_event_carries_relevant_fields():
    async def run():
        session = _FakeSession()
        state = _ChannelState(session, _fake_channel())

        state.on_utterance_end()
        state.on_user_text("hi")
        state.on_model_text("hello")
        state.on_retrieval_complete(
            context="ctx", reference_text="ref", asr_wait_s=0.1, context_injection_s=0.0, api_call_s=0.5
        )
        state.on_first_audio()

        by_event = {ev["event"]: ev for ev in session.raw_written}
        assert by_event["user_text"]["text"] == "hi"
        assert by_event["model_text"]["text"] == "hello"
        assert by_event["retrieval_complete"]["retrieval_context"] == "ctx"
        assert by_event["retrieval_complete"]["retrieved_reference_text"] == "ref"
        assert by_event["retrieval_complete"]["retrieval_breakdown_s"]["total_s"] == pytest.approx(0.6)
        assert "ttfat_s" in by_event["first_audio"]

    asyncio.run(run())


# ── _push_instrumentation ──────────────────────────────────────────────────


def test_push_instrumentation_sends_expected_payload():
    async def run():
        channel = _fake_channel()
        _push_instrumentation(channel, 3, {"ttfat_s": 0.42})
        await asyncio.sleep(0)  # let the scheduled task run

        channel.ws.send_bytes.assert_called_once()
        (sent,), _ = channel.ws.send_bytes.call_args
        assert sent[:1] == b"\x04"
        payload = json.loads(sent[1:])
        assert payload == {"kind": "instrumentation", "turn_index": 3, "fields": {"ttfat_s": 0.42}}

    asyncio.run(run())


def test_push_instrumentation_swallows_send_failures():
    async def run():
        channel = _fake_channel()
        channel.ws.send_bytes.side_effect = RuntimeError("connection closed")

        _push_instrumentation(channel, 1, {"rag_triggered": True})
        await asyncio.sleep(0)  # must not raise / crash the event loop

    asyncio.run(run())


# ── _patch_rag_manager_get_reference_text ───────────────────────────────────


def _fake_rag_manager():
    rag_manager = MagicMock()
    rag_manager._history = []
    return rag_manager


def test_patch_rag_manager_get_reference_text_routes_through_backend(tmp_path):
    session = SessionLog(tmp_path)
    backend = MagicMock()
    backend.retrieve.return_value = ("Paris is the capital of France.", 0.02)
    backend.context_formatting = {}
    rag_manager = _fake_rag_manager()

    _patch_rag_manager_get_reference_text(session, rag_manager, backend)
    context, ref_text, elapsed, label = asyncio.run(
        rag_manager.get_reference_text("user: what is the capital of France?")
    )

    assert context == "user: what is the capital of France?"
    assert ref_text == "Paris is the capital of France."
    assert label == "RetrievalBackend"
    backend.retrieve.assert_called_once_with(
        "user: what is the capital of France?", history=rag_manager._history
    )


def test_patch_rag_manager_get_reference_text_appends_to_history(tmp_path):
    session = SessionLog(tmp_path)
    backend = MagicMock()
    backend.retrieve.return_value = ("some reference", 0.01)
    backend.context_formatting = {}
    rag_manager = _fake_rag_manager()

    _patch_rag_manager_get_reference_text(session, rag_manager, backend)
    asyncio.run(rag_manager.get_reference_text("user: hello"))

    assert rag_manager._history == [(1, "some reference")]


def test_patch_rag_manager_get_reference_text_skips_history_append_on_empty_context(tmp_path):
    session = SessionLog(tmp_path)
    backend = MagicMock()
    backend.retrieve.return_value = ("", 0.0)
    backend.context_formatting = {}
    rag_manager = _fake_rag_manager()

    _patch_rag_manager_get_reference_text(session, rag_manager, backend)
    asyncio.run(rag_manager.get_reference_text("no recognizable turn prefix"))

    assert rag_manager._history == []


def test_patch_rag_manager_get_reference_text_sets_context_injection_s(tmp_path):
    """This is what restores a genuine context_injection_s measurement now
    that the former LLMReferenceGenerator.process_reference_text patch is
    gone — session.last_context_injection_s is what
    _patch_rag_manager_background_task reads for retrieval_breakdown_s."""
    session = SessionLog(tmp_path)
    backend = MagicMock()
    backend.retrieve.return_value = ("ref", 0.01)
    backend.context_formatting = {}
    rag_manager = _fake_rag_manager()

    assert session.last_context_injection_s is None
    _patch_rag_manager_get_reference_text(session, rag_manager, backend)
    asyncio.run(rag_manager.get_reference_text("user: hi"))

    assert session.last_context_injection_s is not None
    assert session.last_context_injection_s >= 0.0


def test_patch_rag_manager_get_reference_text_backend_failure_returns_empty(tmp_path):
    session = SessionLog(tmp_path)
    backend = MagicMock()
    backend.retrieve.side_effect = RuntimeError("boom")
    backend.context_formatting = {}
    rag_manager = _fake_rag_manager()

    _patch_rag_manager_get_reference_text(session, rag_manager, backend)
    context, ref_text, elapsed, label = asyncio.run(rag_manager.get_reference_text("user: hi"))

    assert ref_text == ""
    assert rag_manager._history == []


def test_patch_rag_manager_get_reference_text_respects_context_formatting(tmp_path):
    """context_formatting on the backend instance is what makes
    core/retrieval_backend.py's format_context() drops actually apply here
    too — a leading moshi: turn is dropped, leaving one real turn (so
    num_turns == 1 and the history append proceeds)."""
    session = SessionLog(tmp_path)
    backend = MagicMock()
    backend.retrieve.return_value = ("ref", 0.01)
    backend.context_formatting = {"drop_leading_incomplete_turn": True}
    rag_manager = _fake_rag_manager()

    _patch_rag_manager_get_reference_text(session, rag_manager, backend)
    asyncio.run(rag_manager.get_reference_text("moshi: hi there\nuser: what's up"))

    assert rag_manager._history == [(1, "ref")]


def test_patch_rag_manager_get_reference_text_threads_history_into_retrieve(tmp_path):
    """rag_manager._history is a mutable list appended to *after* retrieve()
    is called — capture a copy at call time, or the later append would also
    (wrongly) show up in what retrieve() appears to have been called with."""
    session = SessionLog(tmp_path)
    captured_history: list = []

    def _fake_retrieve(context, history=None):
        captured_history.append(list(history) if history is not None else None)
        return ("new ref", 0.01)

    backend = MagicMock()
    backend.retrieve.side_effect = _fake_retrieve
    backend.context_formatting = {}
    rag_manager = _fake_rag_manager()
    rag_manager._history.append((1, "prior ref"))

    _patch_rag_manager_get_reference_text(session, rag_manager, backend)
    asyncio.run(rag_manager.get_reference_text("user: follow-up question"))

    assert captured_history == [[(1, "prior ref")]]


# ── _build_retrieval_backend_for_demo ───────────────────────────────────────


def test_build_retrieval_backend_for_demo_requires_demo_config(monkeypatch):
    monkeypatch.delenv("DEMO_CONFIG", raising=False)
    with pytest.raises(SystemExit):
        _build_retrieval_backend_for_demo()


def test_build_retrieval_backend_for_demo_builds_from_real_config(tmp_path, monkeypatch):
    from core.retrieval_backend import GeminiAPIBackend

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-test")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n  checkpoint: local/whatever\n  retrieval:\n"
        "    enabled: true\n    backend: gemini_api\n    latency_gate_ms: 5000\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    monkeypatch.setenv("DEMO_CONFIG", str(cfg))

    backend, display = _build_retrieval_backend_for_demo()

    assert isinstance(backend, GeminiAPIBackend)
    # gemini_api was promoted to gemini-3.5-flash-lite 2026-07-28 — see
    # configs/retrieval_backends.yaml's own comment on this entry.
    assert display == {"name": "gemini_api", "type": "gemini_api", "model": "gemini-3.5-flash-lite"}


def test_build_retrieval_backend_for_demo_disabled_retrieval_returns_null_backend(tmp_path, monkeypatch):
    from core.retrieval_backend import NullBackend

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n  checkpoint: local/whatever\n  retrieval:\n    enabled: false\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    monkeypatch.setenv("DEMO_CONFIG", str(cfg))

    backend, display = _build_retrieval_backend_for_demo()

    assert isinstance(backend, NullBackend)
    assert display == {"name": "null", "type": "null_backend"}


# ── _generation_overrides_for_demo ──────────────────────────────────────────


def test_generation_overrides_for_demo_requires_demo_config(monkeypatch):
    from scripts.instrumented_server import _generation_overrides_for_demo

    monkeypatch.delenv("DEMO_CONFIG", raising=False)
    with pytest.raises(SystemExit):
        _generation_overrides_for_demo()


def test_generation_overrides_for_demo_falls_back_to_defaults(tmp_path, monkeypatch):
    from scripts.instrumented_server import _generation_overrides_for_demo

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n  checkpoint: local/whatever\n  retrieval:\n    enabled: false\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    monkeypatch.setenv("DEMO_CONFIG", str(cfg))

    temp_text, top_k_text = _generation_overrides_for_demo()

    assert (temp_text, top_k_text) == (0.7, 25)


def test_generation_overrides_for_demo_reads_config_overrides(tmp_path, monkeypatch):
    from scripts.instrumented_server import _generation_overrides_for_demo

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "model:\n  checkpoint: local/whatever\n  retrieval:\n    enabled: false\n"
        "  generation:\n    temp_text: 0.9\n    top_k_text: 50\n"
        "evals: []\noutput_dir: ./out/\n"
    )
    monkeypatch.setenv("DEMO_CONFIG", str(cfg))

    temp_text, top_k_text = _generation_overrides_for_demo()

    assert (temp_text, top_k_text) == (0.9, 50)
