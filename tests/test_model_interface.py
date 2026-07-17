import asyncio
from unittest.mock import MagicMock

from core.model_interface import ModelInterface, StubModelAdapter, _TimedInferenceJob, _clean_model_text, _silent_wav
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
