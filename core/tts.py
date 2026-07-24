import abc
import hashlib
import json
import logging
import os
import time
from pathlib import Path

from core.model_interface import _pcm16_to_wav

logger = logging.getLogger(__name__)

# Matches core/model_interface.py's _SAMPLE_RATE — MoshiRAG's own sample
# rate. Gemini's TTS output is natively 24kHz/mono/16-bit PCM too, so no
# resampling is needed between the two.
_SAMPLE_RATE = 24000


class TTSBackend(abc.ABC):
    @abc.abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Return a complete mono WAV file's bytes for the given text."""

    @classmethod
    def from_config(cls, backend_def: dict) -> "TTSBackend":
        """
        Construct an instance from a resolved configs/tts_backends.yaml
        entry. Default: no-arg construction — override for backends that
        need fields out of backend_def (see GeminiTTSBackend.from_config).
        """
        return cls()


class GeminiTTSBackend(TTSBackend):
    def __init__(self, model: str, voice: str, api_key: str | None = None, _client=None):
        self.model = model
        self.voice = voice
        if _client is not None:
            self._client = _client
        else:
            from google import genai
            self._client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    @classmethod
    def from_config(cls, backend_def: dict) -> "GeminiTTSBackend":
        return cls(model=backend_def["model"], voice=backend_def["voice"])

    def synthesize(self, text: str) -> bytes:
        """
        Calls Gemini's TTS-capable generate_content path
        (response_modalities=["AUDIO"] + a speech_config naming a prebuilt
        voice) — the same client.models.generate_content() call shape
        core/llm_judge.py/core/retrieval_backend.py already use, not
        Google's newer client.interactions.create() API. See
        specs/moshirag-evals-requirements.md's core/tts.py section for why
        that choice was made deliberately, not by omission.
        """
        from google.genai import types

        last_exc: Exception | None = None
        for attempt in range(3):
            if attempt:
                wait = 2 ** (attempt - 1)  # 1s then 2s
                logger.warning("Gemini TTS retry %d/3, waiting %ds", attempt + 1, wait)
                time.sleep(wait)

            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                            )
                        ),
                    ),
                )
                pcm = response.candidates[0].content.parts[0].inline_data.data
            except Exception as exc:
                logger.warning("Gemini TTS call failed: %s", exc)
                last_exc = exc
                continue

            if not pcm:
                logger.warning("Gemini TTS response had no audio content — retrying")
                last_exc = RuntimeError("Gemini TTS returned no audio content")
                continue

            return _pcm16_to_wav(pcm, _SAMPLE_RATE)

        raise last_exc  # type: ignore[misc]


TTS_REGISTRY: dict[str, type[TTSBackend]] = {
    "gemini_tts": GeminiTTSBackend,
}


def build_tts_backend(name: str, backend_def: dict) -> TTSBackend:
    """
    Factory: builds a TTSBackend instance from a resolved
    configs/tts_backends.yaml entry, keyed by its `type` field — mirrors
    core/retrieval_backend.py's build_retrieval_backend().
    """
    backend_type = backend_def.get("type")
    cls = TTS_REGISTRY.get(backend_type)
    if cls is None:
        raise ValueError(
            f"Unknown TTS backend type {backend_type!r} for backend {name!r} — "
            f"known types: {sorted(TTS_REGISTRY)}"
        )
    return cls.from_config(backend_def)


def describe_tts_backend(name: str, backend_def: dict) -> dict:
    """
    Display-safe summary of a resolved backend definition — mirrors
    core/retrieval_backend.py's describe_backend(). Never includes a secret
    value (api_key_env only names an env var, never its value).
    """
    fields: dict = {"name": name, "type": backend_def.get("type", "")}
    for key in ("model", "voice"):
        if key in backend_def:
            fields[key] = backend_def[key]
    return fields


def _cache_key(backend_def: dict, text: str) -> str:
    # Only the fields that actually affect the resulting audio — a change to
    # any of these should naturally start populating a new cache namespace,
    # not require manual invalidation.
    payload = json.dumps(
        {"type": backend_def.get("type"), "model": backend_def.get("model"), "voice": backend_def.get("voice")},
        sort_keys=True,
    )
    return hashlib.sha256((payload + "\x00" + text).encode("utf-8")).hexdigest()


def synthesize_cached(
    text: str, backend: TTSBackend, backend_name: str, backend_def: dict, cache_dir: Path
) -> bytes:
    """
    The reusable TTS scaffold: returns cached audio for (backend, text) if
    present, otherwise synthesizes, caches, and returns it. Any eval (or
    future custom eval) needing synthesized question audio calls this
    directly rather than writing its own caching logic — see
    knowledge/gsm8k.py for the first real caller.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_cache_key(backend_def, text)}.wav"
    if cache_path.exists():
        return cache_path.read_bytes()

    logger.info("Synthesizing new TTS audio via %r: %r", backend_name, text[:60])
    wav_bytes = backend.synthesize(text)

    # Atomic write (temp file + rename) — a killed process mid-write can't
    # leave a corrupt cache entry a future run would otherwise trust.
    tmp_path = cache_path.with_suffix(".wav.tmp")
    tmp_path.write_bytes(wav_bytes)
    tmp_path.replace(cache_path)
    return wav_bytes
