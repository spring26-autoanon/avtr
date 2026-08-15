from unittest.mock import MagicMock, patch

import pytest

from core.tts import (
    TTS_REGISTRY,
    GeminiTTSBackend,
    build_tts_backend,
    describe_tts_backend,
    synthesize_cached,
)


def _mock_client(pcm_bytes=b"raw-pcm-bytes"):
    client = MagicMock()
    response = MagicMock()
    response.candidates[0].content.parts[0].inline_data.data = pcm_bytes
    client.models.generate_content.return_value = response
    return client


# ── GeminiTTSBackend.synthesize ──────────────────────────────────────────────


def test_gemini_tts_backend_success_returns_wav_bytes():
    backend = GeminiTTSBackend(model="gemini-3.1-flash-tts-preview", voice="Kore", _client=_mock_client())
    wav_bytes = backend.synthesize("What is 2 + 2?")
    assert wav_bytes.startswith(b"RIFF")
    assert b"raw-pcm-bytes" in wav_bytes


def test_gemini_tts_backend_passes_model_and_voice():
    client = _mock_client()
    backend = GeminiTTSBackend(model="gemini-3.1-flash-tts-preview", voice="Kore", _client=client)
    backend.synthesize("hello")

    _, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-3.1-flash-tts-preview"
    assert kwargs["contents"] == "hello"
    speech_config = kwargs["config"].speech_config
    assert speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"


def test_gemini_tts_backend_retries_on_transient_failure():
    client = MagicMock()
    response = MagicMock()
    response.candidates[0].content.parts[0].inline_data.data = b"eventual-pcm"
    client.models.generate_content.side_effect = [RuntimeError("transient"), RuntimeError("transient"), response]

    backend = GeminiTTSBackend(model="m", voice="Kore", _client=client)
    with patch("core.tts.time.sleep"):
        wav_bytes = backend.synthesize("hello")

    assert b"eventual-pcm" in wav_bytes
    assert client.models.generate_content.call_count == 3


def test_gemini_tts_backend_raises_after_max_retries():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("persistent error")

    backend = GeminiTTSBackend(model="m", voice="Kore", _client=client)
    with patch("core.tts.time.sleep"):
        with pytest.raises(RuntimeError, match="persistent error"):
            backend.synthesize("hello")

    assert client.models.generate_content.call_count == 3


def test_gemini_tts_backend_retries_on_empty_audio():
    client = MagicMock()
    empty_response = MagicMock()
    empty_response.candidates[0].content.parts[0].inline_data.data = b""
    real_response = MagicMock()
    real_response.candidates[0].content.parts[0].inline_data.data = b"real-pcm"
    client.models.generate_content.side_effect = [empty_response, empty_response, real_response]

    backend = GeminiTTSBackend(model="m", voice="Kore", _client=client)
    with patch("core.tts.time.sleep"):
        wav_bytes = backend.synthesize("hello")

    assert b"real-pcm" in wav_bytes
    assert client.models.generate_content.call_count == 3


def test_gemini_tts_backend_from_config():
    backend = GeminiTTSBackend.from_config({"type": "gemini_tts", "model": "gemini-3.1-flash-tts-preview", "voice": "Kore"})
    assert backend.model == "gemini-3.1-flash-tts-preview"
    assert backend.voice == "Kore"


# ── TTS_REGISTRY / build_tts_backend ─────────────────────────────────────────


def test_tts_registry_has_known_types():
    assert TTS_REGISTRY["gemini_tts"] is GeminiTTSBackend


def test_build_tts_backend_gemini_tts():
    backend = build_tts_backend("gemini_tts", {"type": "gemini_tts", "model": "gemini-3.1-flash-tts-preview", "voice": "Kore"})
    assert isinstance(backend, GeminiTTSBackend)
    assert backend.model == "gemini-3.1-flash-tts-preview"
    assert backend.voice == "Kore"


def test_build_tts_backend_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown TTS backend type"):
        build_tts_backend("mystery", {"type": "not_a_real_type"})


# ── describe_tts_backend ──────────────────────────────────────────────────────


def test_describe_tts_backend_includes_model_and_voice():
    result = describe_tts_backend("gemini_tts", {"type": "gemini_tts", "model": "gemini-3.1-flash-tts-preview", "voice": "Kore"})
    assert result == {"name": "gemini_tts", "type": "gemini_tts", "model": "gemini-3.1-flash-tts-preview", "voice": "Kore"}


def test_describe_tts_backend_omits_missing_fields():
    result = describe_tts_backend("kokoro", {"type": "kokoro"})
    assert result == {"name": "kokoro", "type": "kokoro"}
    assert "model" not in result
    assert "voice" not in result


def test_describe_tts_backend_never_includes_api_key_env():
    result = describe_tts_backend(
        "gemini_tts", {"type": "gemini_tts", "model": "gemini-3.1-flash-tts-preview", "api_key_env": "GEMINI_API_KEY"}
    )
    assert "api_key_env" not in result
    assert "GEMINI_API_KEY" not in str(result)


# ── synthesize_cached ─────────────────────────────────────────────────────────


class _FakeBackend:
    def __init__(self):
        self.calls = 0

    def synthesize(self, text: str) -> bytes:
        self.calls += 1
        return f"wav-for-{text}".encode()


def test_synthesize_cached_calls_backend_and_writes_cache(tmp_path):
    backend = _FakeBackend()
    backend_def = {"type": "gemini_tts", "model": "m", "voice": "Kore"}

    result = synthesize_cached("hello", backend, "gemini_tts", backend_def, tmp_path)

    assert result == b"wav-for-hello"
    assert backend.calls == 1
    assert list(tmp_path.glob("*.wav"))


def test_synthesize_cached_reuses_cache_without_recalling_backend(tmp_path):
    backend = _FakeBackend()
    backend_def = {"type": "gemini_tts", "model": "m", "voice": "Kore"}

    first = synthesize_cached("hello", backend, "gemini_tts", backend_def, tmp_path)
    second = synthesize_cached("hello", backend, "gemini_tts", backend_def, tmp_path)

    assert first == second
    assert backend.calls == 1  # only synthesized once


def test_synthesize_cached_different_text_different_cache_entry(tmp_path):
    backend = _FakeBackend()
    backend_def = {"type": "gemini_tts", "model": "m", "voice": "Kore"}

    synthesize_cached("hello", backend, "gemini_tts", backend_def, tmp_path)
    synthesize_cached("world", backend, "gemini_tts", backend_def, tmp_path)

    assert backend.calls == 2
    assert len(list(tmp_path.glob("*.wav"))) == 2


def test_synthesize_cached_different_backend_def_different_cache_entry(tmp_path):
    """Changing voice/model invalidates the old cache entry naturally,
    rather than silently reusing stale audio."""
    backend = _FakeBackend()

    synthesize_cached("hello", backend, "gemini_tts", {"type": "gemini_tts", "model": "m", "voice": "Kore"}, tmp_path)
    synthesize_cached("hello", backend, "gemini_tts", {"type": "gemini_tts", "model": "m", "voice": "Puck"}, tmp_path)

    assert backend.calls == 2
    assert len(list(tmp_path.glob("*.wav"))) == 2
