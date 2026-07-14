from unittest.mock import MagicMock
from core.model_interface import ModelInterface, StubModelAdapter, _silent_wav
from core.retrieval_backend import NullBackend


def test_silent_wav_is_valid():
    wav = _silent_wav(duration_s=0.1)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert len(wav) > 44  # header + data


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


def test_stub_respond_no_backend():
    m = StubModelAdapter(retrieval_backend=None)
    _, _, metadata = m.respond(b"")
    assert metadata["retrieval_text"] == ""
    assert metadata["retrieval_latency_s"] == 0.0


def test_stub_respond_with_null_backend():
    m = StubModelAdapter(retrieval_backend=NullBackend())
    _, _, metadata = m.respond(b"")
    assert metadata["retrieval_text"] == ""


def test_stub_is_model_interface():
    assert isinstance(StubModelAdapter(), ModelInterface)
