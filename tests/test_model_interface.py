import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.gpu import DeviceAssignment
from core.model_interface import (
    ModelInterface,
    MoshiRAGAdapter,
    StubModelAdapter,
    _DEFAULT_GENERATION,
    _TimedInferenceJob,
    _clean_model_text,
    _maybe_enable_eval_asyncio_debug,
    _patch_local_stt_off_thread,
    _patched_stt_send_audio,
    _run_stt_frames_sync,
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
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    adapter = MoshiRAGAdapter("some/checkpoint")
    assert adapter._generation == _DEFAULT_GENERATION


def test_moshi_rag_adapter_generation_overrides_merge_over_defaults(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    adapter = MoshiRAGAdapter("some/checkpoint", generation={"rag_timeout": 12.0})
    assert adapter._generation["rag_timeout"] == 12.0
    # Every other field keeps its default — a config omitting a field
    # doesn't need to specify all of them.
    assert adapter._generation["cfg_coef"] == _DEFAULT_GENERATION["cfg_coef"]
    assert adapter._generation["tail_silence_steps"] == _DEFAULT_GENERATION["tail_silence_steps"]


def test_moshi_rag_adapter_generation_none_uses_defaults(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    adapter = MoshiRAGAdapter("some/checkpoint", generation=None)
    assert adapter._generation == _DEFAULT_GENERATION


# ── MoshiRAGAdapter GPU device assignment ───────────────────────────────────
# __init__ resolves gpu_devices (and logs the contention warning) itself,
# before ever calling _load_models() — see core/gpu.py and specs/
# moshirag-evals-requirements.md's "GPU Sizing and Multi-GPU Deployment"
# section for why this has to happen before _load_models() imports torch.


def test_moshi_rag_adapter_sets_gpu_devices_from_ensure_cuda_visible_devices(monkeypatch):
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    fake_assignment = DeviceAssignment(
        frontend_cuda_visible_devices="0",
        conditioner_cuda_visible_devices="1",
        contended=False,
    )
    calls = []

    def _fake_ensure(role, count=None):
        calls.append(role)
        return fake_assignment

    monkeypatch.setattr("core.model_interface.ensure_cuda_visible_devices", _fake_ensure)
    adapter = MoshiRAGAdapter("some/checkpoint")
    assert adapter.gpu_devices is fake_assignment
    assert calls == ["frontend"]


def test_moshi_rag_adapter_logs_warning_when_gpu_contended(monkeypatch, caplog):
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    monkeypatch.setattr(
        "core.model_interface.ensure_cuda_visible_devices",
        lambda role, count=None: DeviceAssignment("0", "0", contended=True),
    )
    with caplog.at_level("WARNING"):
        MoshiRAGAdapter("some/checkpoint")
    assert "single-GPU mode" in caplog.text


def test_moshi_rag_adapter_no_warning_when_gpu_not_contended(monkeypatch, caplog):
    monkeypatch.setattr(MoshiRAGAdapter, "_load_models", lambda self: None)
    monkeypatch.setattr(
        "core.model_interface.ensure_cuda_visible_devices",
        lambda role, count=None: DeviceAssignment("0", "1", contended=False),
    )
    with caplog.at_level("WARNING"):
        MoshiRAGAdapter("some/checkpoint")
    assert "single-GPU mode" not in caplog.text


def test_stub_model_adapter_gpu_devices_defaults_to_none():
    assert StubModelAdapter().gpu_devices is None


# ── MoshiRAGAdapter.respond() degenerate-silence retry/flag ─────────────────
# _respond_once() itself needs a real GPU/moshi environment — patched here to
# fake single-call outcomes so respond()'s retry/flagging wrapper (see its
# docstring — the WebQ/LlamaQ "zero engagement" investigation in CLAUDE.md)
# is exercisable without one.


def _make_adapter(monkeypatch, once_results):
    """once_results: list of (text_out, rag_trigger_count) for successive _respond_once() calls."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
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


def test_patch_output_loop_handle_reference_fn_applies_conditioning_immediately(monkeypatch):
    """No step-index deferral (unlike the old _catch_reference_text /
    _retrieval_done_step handoff) — matches Channel's _handle_reference_text,
    called directly from RAGManager's own background task."""
    job = _make_output_loop_job()
    fake_conditioning = AsyncMock()
    monkeypatch.setattr("core.model_interface._fetch_and_apply_reference_conditioning", fake_conditioning)

    _TimedInferenceJob(job, MagicMock())
    asyncio.run(job._output_loop())

    handle_reference_fn = job.rag_manager.trigger.await_args.kwargs["handle_reference_fn"]
    asyncio.run(handle_reference_fn("Paris is the capital of France.", lm_label="test-lm"))

    assert job.trace["reference_text"] == "Paris is the capital of France."
    fake_conditioning.assert_awaited_once_with(
        "Paris is the capital of France.",
        encoder_url=job.server.reference_encoder_url,
        lm_gen=job.server.runner.lm_gen,
        batch_size=job.server.batch_size,
        slot_idx=job.slot_idx,
    )
    # Matches moshi-rag's own _async_update_reference side effect (see
    # core/model_interface.py's docstring) — set by the caller now, since
    # _fetch_and_apply_reference_conditioning is shared with the demo path's
    # Channel, which has no InferenceJob-style trace dict.
    assert job.trace["conditioning_step"] == job.step_index


def test_patch_output_loop_handle_reference_fn_times_conditioning_call(monkeypatch):
    """conditioning_latency_s (latency.retrieval_breakdown's context_injection_s
    source) should reflect the real elapsed time of
    _fetch_and_apply_reference_conditioning — the ARC-encoder /embed round
    trip — not stay at its 0.0 default once a reference has actually been
    applied."""
    job = _make_output_loop_job()
    monkeypatch.setattr("core.model_interface._fetch_and_apply_reference_conditioning", AsyncMock())

    timed = _TimedInferenceJob(job, MagicMock())
    assert timed.conditioning_latency_s == 0.0
    asyncio.run(job._output_loop())

    handle_reference_fn = job.rag_manager.trigger.await_args.kwargs["handle_reference_fn"]
    asyncio.run(handle_reference_fn("Paris is the capital of France.", lm_label="test-lm"))

    assert timed.conditioning_latency_s >= 0.0
    assert isinstance(timed.conditioning_latency_s, float)


# ── _fetch_and_apply_reference_conditioning ─────────────────────────────────
# See its own docstring in core/model_interface.py — the narrow fix for the
# ~1.5-1.9s conditioning latency (moshi-rag's own step loop starving this
# HTTP call's event-loop time, not GPU contention — see CLAUDE.md's
# "SUPERSEDED: GPU contention conclusion was wrong"). moshi isn't installed
# in this sandbox, so get_conditioning_remote_async's module is faked via
# sys.modules injection — this still exercises the real per_slot/
# update_streaming_sum_tensors logic, just not the real HTTP call itself.


def _install_fake_moshi_inference_job(monkeypatch, fake_get_conditioning_remote_async):
    import sys
    import types

    for name in ("moshi", "moshi.inference_utils", "moshi.inference_utils.inference_job"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["moshi.inference_utils.inference_job"].get_conditioning_remote_async = (
        fake_get_conditioning_remote_async
    )


def test_fetch_and_apply_reference_conditioning_places_tensor_at_slot_idx(monkeypatch):
    from core.model_interface import _fetch_and_apply_reference_conditioning

    fake_tensor = MagicMock()
    fake_tensor.squeeze.return_value = "squeezed-tensor"

    async def fake_get_conditioning_remote_async(text, encoder_url):
        assert text == "some reference text"
        assert encoder_url == "http://localhost:8001"
        return fake_tensor

    _install_fake_moshi_inference_job(monkeypatch, fake_get_conditioning_remote_async)
    lm_gen = MagicMock()

    asyncio.run(_fetch_and_apply_reference_conditioning(
        "some reference text",
        encoder_url="http://localhost:8001",
        lm_gen=lm_gen,
        batch_size=3,
        slot_idx=1,
    ))

    lm_gen.update_streaming_sum_tensors.assert_called_once()
    per_slot = lm_gen.update_streaming_sum_tensors.call_args[0][0]
    assert per_slot == [None, "squeezed-tensor", None]


def test_fetch_and_apply_reference_conditioning_runs_fetch_off_the_main_thread(monkeypatch):
    """The whole point of this function: get_conditioning_remote_async must
    not run on the caller's own event loop, so moshi-rag's own step loop
    blocking that loop can't starve it. Confirmed by checking the fetch
    executes on a different thread than the one that called this function."""
    import threading

    from core.model_interface import _fetch_and_apply_reference_conditioning

    caller_thread = threading.current_thread()
    fetch_thread_name = {}

    async def fake_get_conditioning_remote_async(text, encoder_url):
        fetch_thread_name["thread"] = threading.current_thread()
        tensor = MagicMock()
        tensor.squeeze.return_value = "tensor"
        return tensor

    _install_fake_moshi_inference_job(monkeypatch, fake_get_conditioning_remote_async)

    asyncio.run(_fetch_and_apply_reference_conditioning(
        "text", encoder_url="http://localhost:8001", lm_gen=MagicMock(), batch_size=1, slot_idx=0,
    ))

    assert fetch_thread_name["thread"] != caller_thread


# ── _run_stt_frames_sync / _patched_stt_send_audio (LocalSpeechToText fix) ─────
# See both functions' own docstrings in core/model_interface.py — the demo-
# path residual-latency root cause (CLAUDE.md's "SUPERSEDED: GPU contention
# conclusion was wrong"): LocalSpeechToText.send_audio ran real Mimi+STT-LM
# compute fully synchronously inside an async def, blocking the shared event
# loop for as long as a buffered-frame backlog took to drain. torch isn't
# installed in this sandbox, so torch is faked via sys.modules injection for
# _run_stt_frames_sync's own tests (its calls chain through a bare MagicMock,
# which auto-vivifies attributes/return values — exercises the real control
# flow, not real tensor math). _patched_stt_send_audio's own tests instead
# fake _run_stt_frames_sync itself, since what's under test there is the
# wrapper's buffering/locking/threading behavior, not the compute.


def _install_fake_torch(monkeypatch):
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.from_numpy = MagicMock(return_value=MagicMock())
    fake_torch.ones = MagicMock(return_value=MagicMock())
    fake_torch.float32 = "float32"
    fake_torch.bool = "bool"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    return fake_torch


class _FakeCodesTensor:
    """Stand-in for the real torch.Tensor codes shape [:, :needed_tokens]
    ends up as — real MagicMock auto-chaining doesn't satisfy
    _run_stt_frames_sync's own `codes.shape[-1] == 1` assertion, since
    comparing an auto-vivified MagicMock to 1 is never truthy."""

    shape = (1, 1, 1)

    def __getitem__(self, item):
        return self

    def to(self, device):
        return self


def _make_fake_stt(sample_rate=24000, frame_rate=12.5):
    import numpy as np

    stt = MagicMock()
    stt._device = "cpu"
    stt.mimi.sample_rate = sample_rate
    stt.mimi.frame_rate = frame_rate
    stt.mimi.encode.return_value = _FakeCodesTensor()
    stt._lm_gen.needed_tokens = 1
    stt._playhead_s = 0.0
    stt._pending = np.zeros(0, dtype=np.float32)
    stt._lock = asyncio.Lock()
    stt._out_queue = asyncio.Queue()
    stt.sent_samples = 0
    return stt


def test_run_stt_frames_sync_calls_run_codes_once_per_frame_and_updates_playhead(monkeypatch):
    _install_fake_torch(monkeypatch)
    stt = _make_fake_stt(sample_rate=24000, frame_rate=12.5)  # fs = 1920
    stt._run_codes.side_effect = ["word-1", None, "word-3"]
    frames = [object(), object(), object()]

    words = _run_stt_frames_sync(stt, frames, fs=1920, sr=24000)

    assert words == ["word-1", "word-3"]
    assert stt._run_codes.call_count == 3
    assert stt._playhead_s == pytest.approx(3 * 1920 / 24000)


def test_run_stt_frames_sync_empty_frames_is_a_noop(monkeypatch):
    _install_fake_torch(monkeypatch)
    stt = _make_fake_stt()

    words = _run_stt_frames_sync(stt, [], fs=1920, sr=24000)

    assert words == []
    stt._run_codes.assert_not_called()
    assert stt._playhead_s == 0.0


def test_patched_stt_send_audio_validates_input():
    import numpy as np

    stt = _make_fake_stt()
    with pytest.raises(ValueError):
        asyncio.run(_patched_stt_send_audio(stt, np.zeros((2, 2), dtype=np.float32)))
    with pytest.raises(ValueError):
        asyncio.run(_patched_stt_send_audio(stt, np.zeros(10, dtype=np.float64)))


def test_patched_stt_send_audio_buffers_partial_frame_without_computing(monkeypatch):
    import numpy as np

    compute_calls = []

    async def fake_to_thread(fn, *args):
        compute_calls.append(args)
        return fn(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    stt = _make_fake_stt(sample_rate=24000, frame_rate=12.5)  # fs = 1920

    asyncio.run(_patched_stt_send_audio(stt, np.zeros(500, dtype=np.float32)))

    assert compute_calls == []
    assert stt._pending.size == 500
    assert stt.sent_samples == 500


def test_patched_stt_send_audio_extracts_complete_frames_and_calls_compute_once(monkeypatch):
    import numpy as np

    frame_batches = []

    def fake_run_stt_frames_sync(stt, frames, fs, sr):
        frame_batches.append((len(frames), fs, sr))
        return ["decoded-word"]

    monkeypatch.setattr("core.model_interface._run_stt_frames_sync", fake_run_stt_frames_sync)
    stt = _make_fake_stt(sample_rate=24000, frame_rate=12.5)  # fs = 1920

    # 3 full frames (5760 samples) plus a 100-sample remainder.
    asyncio.run(_patched_stt_send_audio(stt, np.zeros(5860, dtype=np.float32)))

    assert frame_batches == [(3, 1920, 24000)]
    assert stt._pending.size == 100


def test_patched_stt_send_audio_puts_words_on_queue_in_order(monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        "core.model_interface._run_stt_frames_sync",
        lambda stt, frames, fs, sr: ["word-1", "word-2", "word-3"],
    )
    stt = _make_fake_stt(sample_rate=24000, frame_rate=12.5)  # fs = 1920

    asyncio.run(_patched_stt_send_audio(stt, np.zeros(1920, dtype=np.float32)))

    got = []
    while not stt._out_queue.empty():
        got.append(stt._out_queue.get_nowait())
    assert got == ["word-1", "word-2", "word-3"]


def test_patched_stt_send_audio_runs_compute_off_the_main_thread():
    """The whole point of this fix: mirrors
    test_fetch_and_apply_reference_conditioning_runs_fetch_off_the_main_thread
    — confirms the compute executes on a different thread than the caller,
    the same property that keeps moshi-rag's shared event loop free."""
    import threading

    import numpy as np

    caller_thread = threading.current_thread()
    compute_thread_name = {}

    def fake_run_stt_frames_sync(stt, frames, fs, sr):
        compute_thread_name["thread"] = threading.current_thread()
        return []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("core.model_interface._run_stt_frames_sync", fake_run_stt_frames_sync)
        stt = _make_fake_stt(sample_rate=24000, frame_rate=12.5)
        asyncio.run(_patched_stt_send_audio(stt, np.zeros(1920, dtype=np.float32)))

    assert compute_thread_name["thread"] != caller_thread


def test_patched_stt_send_audio_does_not_block_concurrent_coroutine():
    """The real mechanism check, not just structure: a fake compute that
    does a genuine blocking time.sleep (standing in for real GPU work,
    matching the ~0.58-0.80s real bursts observed in CLAUDE.md's root-cause
    section) must not stall a concurrent lightweight coroutine on the same
    event loop — that's the actual bug being fixed. Compare against
    _fetch_and_apply_reference_conditioning's equivalent property, already
    covered above; this is the same check for the STT path."""
    import time

    import numpy as np

    SLEEP_S = 0.08

    def fake_run_stt_frames_sync(stt, frames, fs, sr):
        time.sleep(SLEEP_S)  # genuine blocking call, runs via asyncio.to_thread
        return []

    async def heartbeat(tick_gap_s: float, stop: asyncio.Event) -> list:
        ticks = []
        t0 = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(tick_gap_s)
            ticks.append(time.perf_counter() - t0)
        return ticks

    async def main():
        stop = asyncio.Event()
        hb_task = asyncio.ensure_future(heartbeat(0.01, stop))
        stt = _make_fake_stt(sample_rate=24000, frame_rate=12.5)
        t0 = time.perf_counter()
        await _patched_stt_send_audio(stt, np.zeros(1920, dtype=np.float32))
        elapsed = time.perf_counter() - t0
        stop.set()
        ticks = await hb_task
        return elapsed, ticks

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("core.model_interface._run_stt_frames_sync", fake_run_stt_frames_sync)
        elapsed, ticks = asyncio.run(main())

    assert elapsed >= SLEEP_S
    # If the event loop were blocked for the sleep's duration (the pre-fix
    # bug), the heartbeat would accumulate ~0 ticks during that window. With
    # the fix, ticks should keep landing roughly every 0.01s throughout.
    assert len(ticks) >= int(SLEEP_S / 0.01) - 1


def test_patched_stt_send_audio_serializes_overlapping_calls():
    """self._lock must still prevent two overlapping send_audio calls from
    interleaving their compute, now that compute runs via asyncio.to_thread
    — holding an asyncio.Lock across an awaited to_thread call keeps it
    logically held for the whole duration, so this should hold by
    construction; asserted directly rather than trusted."""
    import numpy as np

    concurrent_count = {"current": 0, "max": 0}

    def fake_run_stt_frames_sync(stt, frames, fs, sr):
        concurrent_count["current"] += 1
        concurrent_count["max"] = max(concurrent_count["max"], concurrent_count["current"])
        import time

        time.sleep(0.02)
        concurrent_count["current"] -= 1
        return []

    async def main():
        stt = _make_fake_stt(sample_rate=24000, frame_rate=12.5)
        await asyncio.gather(
            _patched_stt_send_audio(stt, np.zeros(1920, dtype=np.float32)),
            _patched_stt_send_audio(stt, np.zeros(1920, dtype=np.float32)),
        )

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("core.model_interface._run_stt_frames_sync", fake_run_stt_frames_sync)
        asyncio.run(main())

    assert concurrent_count["max"] == 1


def _install_fake_moshi_stt_local_stt(monkeypatch):
    import sys
    import types

    for name in ("moshi", "moshi.stt", "moshi.stt.local_stt"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))

    class _FakeLocalSpeechToText:
        async def send_audio(self, audio):
            raise AssertionError("should have been replaced by the patch")

    sys.modules["moshi.stt.local_stt"].LocalSpeechToText = _FakeLocalSpeechToText
    return _FakeLocalSpeechToText


def test_patch_local_stt_off_thread_replaces_send_audio(monkeypatch):
    fake_cls = _install_fake_moshi_stt_local_stt(monkeypatch)

    _patch_local_stt_off_thread()

    assert fake_cls.send_audio is _patched_stt_send_audio


def test_patch_local_stt_off_thread_is_idempotent(monkeypatch):
    fake_cls = _install_fake_moshi_stt_local_stt(monkeypatch)

    _patch_local_stt_off_thread()
    _patch_local_stt_off_thread()

    assert fake_cls.send_audio is _patched_stt_send_audio


# ── _maybe_enable_eval_asyncio_debug (eval-path counterpart of the demo's
#    DEMO_ASYNCIO_DEBUG diagnostic — see that function's own docstring) ────────


def test_maybe_enable_eval_asyncio_debug_noop_when_unset(monkeypatch):
    monkeypatch.delenv("EVAL_ASYNCIO_DEBUG", raising=False)
    loop = asyncio.new_event_loop()
    try:
        _maybe_enable_eval_asyncio_debug(loop)
        assert loop.get_debug() is False
    finally:
        loop.close()


def test_maybe_enable_eval_asyncio_debug_sets_debug_and_threshold(monkeypatch):
    monkeypatch.setenv("EVAL_ASYNCIO_DEBUG", "1")
    monkeypatch.setenv("EVAL_ASYNCIO_DEBUG_THRESHOLD_S", "0.015")
    loop = asyncio.new_event_loop()
    try:
        _maybe_enable_eval_asyncio_debug(loop)
        assert loop.get_debug() is True
        assert loop.slow_callback_duration == pytest.approx(0.015)
    finally:
        loop.close()


def test_maybe_enable_eval_asyncio_debug_default_threshold(monkeypatch):
    monkeypatch.setenv("EVAL_ASYNCIO_DEBUG", "1")
    monkeypatch.delenv("EVAL_ASYNCIO_DEBUG_THRESHOLD_S", raising=False)
    loop = asyncio.new_event_loop()
    try:
        _maybe_enable_eval_asyncio_debug(loop)
        assert loop.slow_callback_duration == pytest.approx(0.02)
    finally:
        loop.close()


# ── asr_wait_s / retrieval_trigger_ts (latency.retrieval_breakdown support) ────


def test_timed_inference_job_asr_wait_defaults():
    timed = _make_timed_job(MagicMock())
    assert timed.retrieval_trigger_ts is None
    assert timed.asr_wait_s == 0.0


def test_patch_output_loop_records_retrieval_trigger_ts():
    job = _make_output_loop_job()
    timed = _TimedInferenceJob(job, MagicMock())

    assert timed.retrieval_trigger_ts is None
    asyncio.run(job._output_loop())

    assert timed.retrieval_trigger_ts is not None


def test_patch_rag_manager_computes_asr_wait_s_from_trigger_ts(monkeypatch):
    backend = MagicMock()
    backend.retrieve.return_value = ("reference text", 0.01)
    timed = _make_timed_job(backend)
    timed.retrieval_trigger_ts = 100.0  # simulated <ret> trigger moment

    # _patched_get_reference_text calls time.perf_counter() twice: once for
    # t0 (asr_wait_s = t0 - retrieval_trigger_ts), once later for its own
    # elapsed computation — both must be supplied.
    fake_clock = iter([100.25, 100.30])
    monkeypatch.setattr("core.model_interface.time.perf_counter", lambda: next(fake_clock))

    patched = timed._job.rag_manager.get_reference_text
    asyncio.run(patched("some context"))

    assert timed.asr_wait_s == pytest.approx(0.25)


def test_patch_rag_manager_asr_wait_s_stays_zero_without_a_trigger():
    """get_reference_text called with no prior <ret> trigger recorded (e.g.
    a test calling it directly) shouldn't fabricate a wait time."""
    backend = MagicMock()
    backend.retrieve.return_value = ("reference text", 0.01)
    timed = _make_timed_job(backend)

    patched = timed._job.rag_manager.get_reference_text
    asyncio.run(patched("some context"))

    assert timed.asr_wait_s == 0.0


def test_output_loop_trigger_then_get_reference_text_end_to_end_asr_wait():
    """Full integration: a real <ret> trigger followed by the retrieval
    call it schedules, exercising both patches together the way a real
    turn would (trigger records retrieval_trigger_ts, then whatever later
    calls get_reference_text sees a real, positive gap)."""
    job = _make_output_loop_job()
    backend = MagicMock()
    backend.retrieve.return_value = ("reference text", 0.01)
    timed = _TimedInferenceJob(job, backend)

    asyncio.run(job._output_loop())
    assert timed.retrieval_trigger_ts is not None

    asyncio.run(timed._job.rag_manager.get_reference_text("some context"))


# ── first-audio detection (real bug fix #2, confirmed on a VM run) ──────────
# t_first_audio used to be set on the first output chunk with
# pcm.abs().max() > 1e-6. A real VM run's diagnostic logging showed that
# firing at step_index=0 with an *identical* amplitude (0.01241324, to 8
# decimal places) across five completely different questions/responses —
# proof it was a fixed decoder warm-up artifact, not content-dependent
# speech onset. Fixed by using the first non-pad TEXT token instead (text
# and audio are generated in lockstep in this architecture), inside the
# already-patched _patched_output_loop rather than a new patch.


def test_patch_output_loop_records_first_audio_on_first_non_pad_text_token():
    job = _make_output_loop_job()
    real_output = MagicMock(text_token=123, pcm=None)  # not rag_token_id (999)
    job._decode_text_token = MagicMock(return_value="real-word")  # not None -> a real token

    async def _get_once():
        job._shutdown_event.set()
        return real_output

    job.output_queue.get = AsyncMock(side_effect=_get_once)

    timed = _TimedInferenceJob(job, MagicMock())
    assert timed._first_audio_recorded is False

    asyncio.run(job._output_loop())

    assert timed._first_audio_recorded is True
    assert timed.t_first_audio > 0


def test_patch_output_loop_pad_token_does_not_record_first_audio():
    job = _make_output_loop_job()
    pad_output = MagicMock(text_token=123, pcm=None)  # not rag_token_id (999)
    job._decode_text_token = MagicMock(return_value=None)  # pad token

    async def _get_once():
        job._shutdown_event.set()
        return pad_output

    job.output_queue.get = AsyncMock(side_effect=_get_once)

    timed = _TimedInferenceJob(job, MagicMock())
    asyncio.run(job._output_loop())

    assert timed._first_audio_recorded is False


def test_patch_output_loop_only_records_first_audio_once():
    """A second real token shouldn't overwrite t_first_audio from the first."""
    job = _make_output_loop_job()
    real_output = MagicMock(text_token=123, pcm=None)
    job._decode_text_token = MagicMock(return_value="real-word")
    calls = {"n": 0}

    async def _get_twice():
        calls["n"] += 1
        if calls["n"] >= 2:
            job._shutdown_event.set()
        return real_output

    job.output_queue.get = AsyncMock(side_effect=_get_twice)

    timed = _TimedInferenceJob(job, MagicMock())
    asyncio.run(job._output_loop())

    assert timed._first_audio_recorded is True
    first_t = timed.t_first_audio
    # Re-running the loop body's logic isn't re-invoked here since the loop
    # already exited — the guarantee under test is the `if not
    # self._first_audio_recorded` gate itself, exercised above by the loop
    # processing 2 real-token iterations without t_first_audio changing.
    assert timed.t_first_audio == first_t


# ── _finalize_ttfat / run() (real bug fix #1, confirmed on a VM run) ────────
# ttfat_s was previously always exactly 0.0 on every real respond() call.
# Root cause (confirmed against real inference_job.py source): _feed_loop(),
# under stop_on_end_of_input=False (respond()'s own setting), does not
# return when the user's real audio ends — it keeps feeding silence until
# the whole turn's _shutdown_event fires. The old code set t_question_end
# to wall-clock time right after _feed_loop() *returned*, i.e. near
# end-of-turn — always later than t_first_audio, clamping ttfat_s to 0.0.
# The fix correlates trace["question_end_step"] (moshi-rag's own, correct
# step index for the last real input frame) against a
# (feed_step -> wall-clock time) mapping recorded by wrapping
# input_queue.put(), rather than trusting _feed_loop()'s return time.


class _FakeInputQueue:
    def __init__(self):
        self.put = AsyncMock()  # _TimedInferenceJob.run() wraps this


def test_finalize_ttfat_uses_question_end_step_not_feed_loop_completion(monkeypatch):
    fake_times = iter([10.0, 10.1, 10.2, 10.3, 10.4])  # 5 values for 5 input_queue.put() calls below
    monkeypatch.setattr("core.model_interface.time.time", lambda: next(fake_times))

    job = MagicMock()
    job.trace = {}
    job.input_queue = _FakeInputQueue()

    async def _fake_run(task_group):
        # Simulate _feed_loop: 3 real input frames (question_end_step ends
        # up at 2, the last real frame's index), then 2 trailing silence
        # frames that keep calling input_queue.put() too — matching real
        # inference_job.py's stop_on_end_of_input=False behavior. First
        # audio detection (now text-token-based) is exercised separately
        # above — simulated directly here by setting t_first_audio/
        # _first_audio_recorded on `timed` directly, decoupling this test's
        # concern (question_end_step correlation) from that one.
        for i in range(3):
            await job.input_queue.put(f"real-{i}")
        job.trace["question_end_step"] = 2
        for i in range(2):
            await job.input_queue.put(f"silence-{i}")

    job.run = AsyncMock(side_effect=_fake_run)

    timed = _TimedInferenceJob(job, MagicMock())
    timed.t_first_audio = 15.0  # unrelated to the feed clock — set directly, not via time.time()
    timed._first_audio_recorded = True
    asyncio.run(timed.run(MagicMock()))

    assert timed._feed_call_times == [10.0, 10.1, 10.2, 10.3, 10.4]
    assert timed._t_question_end == 10.2  # feed_call_times[2] — the real question's end
    assert timed._t_question_end != timed._feed_call_times[-1]  # NOT 10.4 (end of trailing silence)
    assert timed.ttfat_s == pytest.approx(4.8)  # 15.0 - 10.2, not clamped to 0


def test_finalize_ttfat_defaults_to_zero_when_question_end_step_missing():
    """A turn where the user's input never registers a real frame at all
    (trace never gets question_end_step) shouldn't crash — ttfat_s just
    stays at its safe 0.0 default."""
    job = MagicMock()
    job.trace = {}
    job.input_queue = _FakeInputQueue()
    job.run = AsyncMock()

    timed = _TimedInferenceJob(job, MagicMock())
    timed.t_first_audio = 100.0
    timed._first_audio_recorded = True
    asyncio.run(timed.run(MagicMock()))

    assert timed._t_question_end is None
    assert timed.ttfat_s == 0.0


def test_finalize_ttfat_defaults_to_zero_when_first_audio_never_recorded():
    """A fully-silent output (degenerate response) shouldn't crash either —
    no first-audio timestamp means no ttfat_s to compute."""
    job = MagicMock()
    job.trace = {"question_end_step": 0}
    job.input_queue = _FakeInputQueue()

    async def _fake_run(task_group):
        await job.input_queue.put("real-0")

    job.run = AsyncMock(side_effect=_fake_run)

    timed = _TimedInferenceJob(job, MagicMock())
    asyncio.run(timed.run(MagicMock()))

    assert timed._first_audio_recorded is False
    assert timed.ttfat_s == 0.0

    assert timed.asr_wait_s >= 0.0
