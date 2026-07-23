import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.model_interface import (
    ModelInterface,
    MoshiRAGAdapter,
    StubModelAdapter,
    _DEFAULT_GENERATION,
    _TimedInferenceJob,
    _clean_model_text,
    _silent_wav,
)
from core.retrieval_backend import NullBackend


def test_silent_wav_is_valid():
    wav = _silent_wav(duration_s=0.1)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert len(wav) > 44  # header + data


# ── _clean_model_text ──────────────────────────────────────────────────────────


def test_clean_model_text_replaces_word_boundary_marker_with_space():
    pieces = ["▁Hmm", ",", "▁that", "'", "s", "▁a", "▁good", "▁question", "."]
    assert _clean_model_text(pieces) == "Hmm, that's a good question."


def test_clean_model_text_drops_pad_tokens():
    pieces = ["<pad>", "<pad>", "▁Hi", "<pad>", "."]
    assert _clean_model_text(pieces) == "Hi."


def test_clean_model_text_drops_byte_fallback_tokens():
    # Reproduces the exact reported case: a leading byte-fallback artifact
    # token before real text begins.
    pieces = ["<0x00>", "▁Hmm", ",", "▁that", "'", "s", "▁a", "▁good", "▁question", "."]
    assert _clean_model_text(pieces) == "Hmm, that's a good question."


def test_clean_model_text_strips_leading_space_from_word_boundary_marker():
    assert _clean_model_text(["▁Paris"]) == "Paris"


def test_clean_model_text_empty_input():
    assert _clean_model_text([]) == ""


def test_stub_transcribe_returns_string():
    m = StubModelAdapter()
    result = m.transcribe(b"\x00" * 1600)
    assert isinstance(result, str)
    assert len(result) > 0


def test_stub_respond_return_types():
    m = StubModelAdapter()
    audio_out, text_out, metadata = m.respond(b"\x00" * 1600)
    assert isinstance(audio_out, bytes)
    assert isinstance(text_out, str)
    assert isinstance(metadata, dict)


def test_stub_respond_audio_is_wav():
    m = StubModelAdapter()
    audio_out, _, _ = m.respond(b"")
    assert audio_out[:4] == b"RIFF"


def test_stub_respond_metadata_fields():
    m = StubModelAdapter()
    _, _, metadata = m.respond(b"")
    assert "ttfat_s" in metadata
    assert "first_audio_token_ts" in metadata
    assert isinstance(metadata["ttfat_s"], float)
    assert metadata["first_audio_token_ts"] > 0


def test_stub_respond_calls_retrieval_backend():
    mock_backend = MagicMock()
    mock_backend.retrieve.return_value = ("reference text", 0.05)

    m = StubModelAdapter(retrieval_backend=mock_backend)
    _, _, metadata = m.respond(b"")

    mock_backend.retrieve.assert_called_once()
    assert metadata["retrieval_text"] == "reference text"
    assert metadata["retrieval_latency_s"] == 0.05
    assert metadata["retrieval_context"] == m._TRANSCRIPTION


def test_stub_respond_no_backend():
    m = StubModelAdapter(retrieval_backend=None)
    _, _, metadata = m.respond(b"")
    assert metadata["retrieval_text"] == ""
    assert metadata["retrieval_latency_s"] == 0.0
    assert metadata["retrieval_context"] == ""


def test_stub_respond_with_null_backend():
    m = StubModelAdapter(retrieval_backend=NullBackend())
    _, _, metadata = m.respond(b"")
    assert metadata["retrieval_text"] == ""


def test_stub_is_model_interface():
    assert isinstance(StubModelAdapter(), ModelInterface)


# ── MoshiRAGAdapter generation dict ─────────────────────────────────────────
# _load_models() needs a real GPU/moshi environment — patched to a no-op so
# these exercise only the config-merging behavior, not model loading.


def test_moshi_rag_adapter_generation_defaults(monkeypatch):
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    adapter = MoshiRAGAdapter("some/checkpoint")
    assert adapter._generation == _DEFAULT_GENERATION


def test_moshi_rag_adapter_generation_overrides_merge_over_defaults(monkeypatch):
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    adapter = MoshiRAGAdapter("some/checkpoint", generation={"rag_timeout": 12.0})
    assert adapter._generation["rag_timeout"] == 12.0
    # Every other field keeps its default — a config omitting a field
    # doesn't need to specify all of them.
    assert adapter._generation["cfg_coef"] == _DEFAULT_GENERATION["cfg_coef"]
    assert adapter._generation["tail_silence_steps"] == _DEFAULT_GENERATION["tail_silence_steps"]


def test_moshi_rag_adapter_generation_none_uses_defaults(monkeypatch):
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    adapter = MoshiRAGAdapter("some/checkpoint", generation=None)
    assert adapter._generation == _DEFAULT_GENERATION


# ── MoshiRAGAdapter.respond() degenerate-silence retry/flag ─────────────────
# _respond_once() itself needs a real GPU/moshi environment — patched here to
# fake single-call outcomes so respond()'s retry/flagging wrapper (see its
# docstring — the WebQ/LlamaQ "zero engagement" investigation in CLAUDE.md)
# is exercisable without one.


def _make_adapter(monkeypatch, once_results):
    """once_results: list of (text_out, rag_trigger_count) for successive _respond_once() calls."""
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    adapter = MoshiRAGAdapter("some/checkpoint")
    calls = iter(once_results)

    def _fake_respond_once(self, audio_in):
        text_out, rag_trigger_count = next(calls)
        return b"RIFF....WAVEfmt ", text_out, {"rag_trigger_count": rag_trigger_count}

    monkeypatch.setattr(MoshiRAGAdapter, "_respond_once", _fake_respond_once)
    return adapter


def test_respond_normal_turn_not_flagged_no_retry(monkeypatch):
    adapter = _make_adapter(monkeypatch, [("Paris.", 0)])
    _, text_out, metadata = adapter.respond(b"")
    assert text_out == "Paris."
    assert metadata["degenerate_silence"] is False
    assert metadata["degenerate_silence_retried"] is False


def test_respond_rag_triggered_empty_text_not_flagged(monkeypatch):
    # rag_trigger_count > 0 alone doesn't count as degenerate, even if the
    # visible text happens to be empty at that exact point in the trace.
    adapter = _make_adapter(monkeypatch, [("", 1)])
    _, _, metadata = adapter.respond(b"")
    assert metadata["degenerate_silence"] is False


def test_respond_degenerate_silence_retries_once(monkeypatch):
    adapter = _make_adapter(monkeypatch, [("", 0), ("", 0)])
    _, text_out, metadata = adapter.respond(b"")
    assert text_out == ""
    assert metadata["degenerate_silence_retried"] is True
    assert metadata["degenerate_silence"] is True


def test_respond_degenerate_silence_recovers_on_retry(monkeypatch):
    adapter = _make_adapter(monkeypatch, [("", 0), ("Paris.", 0)])
    _, text_out, metadata = adapter.respond(b"")
    assert text_out == "Paris."
    assert metadata["degenerate_silence_retried"] is True
    assert metadata["degenerate_silence"] is False


# ── _TimedInferenceJob retrieval diagnostics ──────────────────────────────────


def _make_timed_job(backend):
    """A minimal fake base_job — only rag_manager.get_reference_text needs
    to be a settable attribute for _patch_rag_manager() to work."""
    base_job = MagicMock()
    return _TimedInferenceJob(base_job, backend)


def test_patch_rag_manager_records_context_and_reference_text():
    backend = MagicMock()
    backend.retrieve.return_value = ("Paris is the capital of France.", 0.01)

    timed = _make_timed_job(backend)
    patched = timed._job.rag_manager.get_reference_text

    context, ref_text, elapsed, label = asyncio.run(patched("user: what is the capital of France?"))

    assert context == "user: what is the capital of France?"
    assert ref_text == "Paris is the capital of France."
    assert timed.retrieval_context == "user: what is the capital of France?"
    assert timed.retrieval_latency_s == 0.01
    assert label == "RetrievalBackend"


def test_patch_rag_manager_logs_context_and_result(caplog):
    backend = MagicMock()
    backend.retrieve.return_value = ("reference text", 0.02)
    timed = _make_timed_job(backend)
    patched = timed._job.rag_manager.get_reference_text

    with caplog.at_level("INFO", logger="core.model_interface"):
        asyncio.run(patched("some question context"))

    assert any("[RetrievalBackend]" in r.message for r in caplog.records)
    assert any("some question context" in r.message for r in caplog.records)
    assert any("reference text" in r.message for r in caplog.records)


def test_patch_rag_manager_null_backend_context_visible_with_empty_result():
    timed = _make_timed_job(NullBackend())
    patched = timed._job.rag_manager.get_reference_text

    context, ref_text, elapsed, label = asyncio.run(patched("what year was the eiffel tower built?"))

    assert timed.retrieval_context == "what year was the eiffel tower built?"
    assert ref_text == ""


def test_patch_rag_manager_backend_failure_still_records_context():
    backend = MagicMock()
    backend.retrieve.side_effect = RuntimeError("boom")
    timed = _make_timed_job(backend)
    patched = timed._job.rag_manager.get_reference_text

    context, ref_text, elapsed, label = asyncio.run(patched("context that triggered a failure"))

    assert timed.retrieval_context == "context that triggered a failure"
    assert ref_text == ""


# ── _patch_output_loop — no _doing_retrieval gate, immediate conditioning ──────


def _make_output_loop_job() -> MagicMock:
    """A fake InferenceJob with just enough real asyncio primitives
    (_shutdown_event, _pcm_one_step_cv) for one _output_loop iteration to
    run for real, everything else mocked."""
    job = MagicMock()
    job._task_group = MagicMock()
    job._shutdown_event = asyncio.Event()
    job._pcm_one_step_cv = asyncio.Condition()
    job.trace = {}
    job.model_text = []
    job.user_text = []
    job._user_id_buffer = []
    job.step_index = 0
    job.max_tail_silence = None  # disables the unrelated tail-silence finalize path
    job.stop_on_end_of_input = False
    job._model_pcm_chunks = []
    job.turn_manager.stt_wait_steps = 6  # must be a real int: int(job.turn_manager.stt_wait_steps)
    job.server.runner.lm_gen.lm_model.rag_token_id = 999
    job.rag_manager = MagicMock()
    job.rag_manager.trigger = AsyncMock()
    job._async_update_reference = AsyncMock()

    rag_output = MagicMock(text_token=999, pcm=None)

    async def _get_once():
        job._shutdown_event.set()  # so the loop exits after this one iteration
        return rag_output

    job.output_queue.get = AsyncMock(side_effect=_get_once)
    return job


def test_patch_output_loop_replaces_output_loop():
    timed = _make_timed_job(MagicMock())
    assert asyncio.iscoroutinefunction(timed._job._output_loop)


def test_patch_output_loop_never_touches_doing_retrieval():
    """The whole point of this patch: Channel (the real production
    WebSocket server class) has no _doing_retrieval gate at all, so neither
    should our replacement _output_loop."""
    job = _make_output_loop_job()
    job._doing_retrieval = "sentinel-should-stay-untouched"

    _TimedInferenceJob(job, MagicMock())
    asyncio.run(job._output_loop())

    assert job._doing_retrieval == "sentinel-should-stay-untouched"
    job.rag_manager.trigger.assert_awaited_once()
    assert job.trace["rag_trigger_step"] == 0
    assert job.trace["rag_trigger_count"] == 1


def test_patch_output_loop_counts_multiple_rag_triggers_in_one_turn():
    """The diagnostic this investigation needed: distinguishing "<ret> fired
    once" from "<ret> fired repeatedly, re-cancelling retrieval each time"
    (RAGManager.trigger() cancels any pending task on every call) —
    trace["rag_trigger_step"] alone can't tell these apart since it's
    silently overwritten on every occurrence."""
    job = _make_output_loop_job()
    calls = {"n": 0}
    rag_output = MagicMock(text_token=999, pcm=None)

    async def _get_three_times():
        calls["n"] += 1
        if calls["n"] >= 3:
            job._shutdown_event.set()
        return rag_output

    job.output_queue.get = AsyncMock(side_effect=_get_three_times)

    _TimedInferenceJob(job, MagicMock())
    asyncio.run(job._output_loop())

    assert job.trace["rag_trigger_count"] == 3
    assert job.rag_manager.trigger.await_count == 3


def test_patch_output_loop_logs_and_reraises_trigger_exception(caplog):
    """rag_manager.trigger() should be near-impossible to raise per its own
    source — if it ever does, we want the full traceback surfaced, not a
    silently swallowed/hung turn."""
    job = _make_output_loop_job()
    job.rag_manager.trigger = AsyncMock(side_effect=RuntimeError("boom"))

    _TimedInferenceJob(job, MagicMock())
    with caplog.at_level("ERROR", logger="core.model_interface"):
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(job._output_loop())

    assert any("rag_manager.trigger() raised" in r.message for r in caplog.records)


def test_patch_output_loop_handle_reference_fn_applies_conditioning_immediately():
    """No step-index deferral (unlike the old _catch_reference_text /
    _retrieval_done_step handoff) — matches Channel's _handle_reference_text,
    called directly from RAGManager's own background task."""
    job = _make_output_loop_job()

    _TimedInferenceJob(job, MagicMock())
    asyncio.run(job._output_loop())

    handle_reference_fn = job.rag_manager.trigger.await_args.kwargs["handle_reference_fn"]
    asyncio.run(handle_reference_fn("Paris is the capital of France.", lm_label="test-lm"))

    assert job.trace["reference_text"] == "Paris is the capital of France."
    job._async_update_reference.assert_awaited_once_with("Paris is the capital of France.")
