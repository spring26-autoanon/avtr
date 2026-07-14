import abc
import argparse
import asyncio
import logging
import struct
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from core.retrieval_backend import NullBackend, RetrievalBackend

logger = logging.getLogger(__name__)

# MoshiRAG outputs 24kHz audio
_SAMPLE_RATE = 24000


def _silent_wav(duration_s: float = 0.5, sample_rate: int = _SAMPLE_RATE) -> bytes:
    """Minimal valid mono 16-bit PCM WAV of silence."""
    n_samples = int(duration_s * sample_rate)
    pcm = b"\x00\x00" * n_samples
    data_size = len(pcm)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_size)
        + pcm
    )


class ModelInterface(abc.ABC):
    @abc.abstractmethod
    def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes to text."""

    @abc.abstractmethod
    def respond(self, audio_in: bytes) -> tuple[bytes, str, dict]:
        """
        Process an audio input and return (audio_out, text_out, metadata).

        metadata must include at minimum:
          ttfat_s              — time to first audio token in seconds
          first_audio_token_ts — absolute timestamp of first audio token (time.time())
        """


class StubModelAdapter(ModelInterface):
    """
    No-GPU stand-in for MoshiRAGAdapter.

    Returns canned audio + text so the full eval pipeline can be exercised
    locally. Calls the retrieval backend (if provided) to exercise the RAG path.
    """

    _TRANSCRIPTION = "stub transcription: what is the capital of France?"
    _RESPONSE_TEXT = "stub response: the capital of France is Paris."
    _TTFAT_S = 0.02  # simulated first-token latency

    def __init__(self, retrieval_backend: RetrievalBackend | None = None):
        self.retrieval_backend = retrieval_backend

    def transcribe(self, audio: bytes) -> str:
        return self._TRANSCRIPTION

    def respond(self, audio_in: bytes) -> tuple[bytes, str, dict]:
        t0 = time.time()

        ref_text = ""
        retrieval_latency_s = 0.0
        if self.retrieval_backend is not None:
            ref_text, retrieval_latency_s = self.retrieval_backend.retrieve(
                self._TRANSCRIPTION
            )

        metadata = {
            "ttfat_s": self._TTFAT_S,
            "first_audio_token_ts": t0 + self._TTFAT_S,
            "retrieval_text": ref_text,
            "retrieval_latency_s": retrieval_latency_s,
        }

        return _silent_wav(), self._RESPONSE_TEXT, metadata


# ── MoshiRAGAdapter ───────────────────────────────────────────────────────────
# All GPU deps (torch, moshi) are lazy-imported inside methods so this module
# can be imported on local dev machines without a GPU environment.


def _float32_pcm_to_wav(pcm: "np.ndarray", sample_rate: int) -> bytes:  # type: ignore[name-defined]
    """Convert float32 numpy PCM array to 16-bit mono WAV bytes."""
    import numpy as np
    pcm = np.clip(pcm, -1.0, 1.0)
    samples = (pcm * 32767).astype(np.int16)
    data = samples.tobytes()
    return (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )


class _TimedInferenceJob:
    """
    Thin wrapper around InferenceJob that:
      1. Intercepts the <ret> retrieval trigger to call our RetrievalBackend
         and record latency (feeds reference text back into moshi-rag).
      2. Tracks wall-clock timestamps for TTFAT computation.

    Subclassing via monkey-patching avoids forking the moshi-rag package.
    """

    def __init__(self, base_job, retrieval_backend: RetrievalBackend):
        self._job = base_job
        self._retrieval_backend = retrieval_backend
        self.ttfat_s: float = 0.0
        self.t_first_audio: float = 0.0
        self.retrieval_latency_s: float = 0.0
        self._t_question_end: float | None = None
        self._first_audio_recorded = False

        # Patch the job's RAGManager trigger to route through our backend
        if retrieval_backend is not None and not isinstance(retrieval_backend, NullBackend):
            self._patch_rag_manager()

    def _patch_rag_manager(self):
        """Replace the job's retrieval call with our GeminiAPIBackend."""
        backend = self._retrieval_backend
        original_trigger = self._job.rag_manager.trigger

        async def _patched_trigger(context: str, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                ref_text, latency = backend.retrieve(context)
                self.retrieval_latency_s = latency
            except Exception as exc:
                logger.warning("RetrievalBackend.retrieve() failed: %s", exc)
                ref_text, latency = "", 0.0
            # Feed the reference text back through the original trigger mechanism
            await original_trigger(context, *args, override_reference=ref_text, **kwargs)

        self._job.rag_manager.trigger = _patched_trigger

    async def run(self, task_group):
        """Delegate to the underlying InferenceJob, hooking timing."""
        # Wrap _feed_loop to record t_question_end
        original_feed = self._job._feed_loop

        async def _timed_feed():
            await original_feed()
            self._t_question_end = time.time()

        self._job._feed_loop = _timed_feed

        # Wrap _output_loop to record t_first_audio
        original_output = self._job._output_loop

        async def _timed_output():
            await original_output()

        # Hook into output_queue consumption to detect first audio
        original_queue_get = self._job.output_queue.get

        async def _watching_get():
            out = await original_queue_get()
            if not self._first_audio_recorded and out is not None:
                import numpy as np
                pcm = getattr(out, "pcm", None)
                if pcm is not None and pcm.abs().max().item() > 1e-6:
                    self.t_first_audio = time.time()
                    self._first_audio_recorded = True
                    if self._t_question_end is not None:
                        self.ttfat_s = max(0.0, self.t_first_audio - self._t_question_end)
            return out

        self._job.output_queue.get = _watching_get
        await self._job.run(task_group)

    @property
    def trace(self):
        return self._job.trace

    @property
    def _model_pcm_chunks(self):
        return self._job._model_pcm_chunks

    @property
    def slot_idx(self):
        return self._job.slot_idx

    @slot_idx.setter
    def slot_idx(self, v):
        self._job.slot_idx = v

    @property
    def stt(self):
        return getattr(self._job, "stt", None)

    @stt.setter
    def stt(self, v):
        self._job.stt = v


class MoshiRAGAdapter(ModelInterface):
    """
    Wraps the moshi-rag inference pipeline for use in the eval framework.

    Requires the GPU environment (uv sync --extra gpu) and a valid checkpoint.
    All moshi imports are deferred to init so this module is importable locally.

    Retrieval:
      - If retrieval_backend is NullBackend or None: moshi-rag's RAG path is
        still active (predicts <ret>) but uses its built-in LLM retrieval
        configured via REFERENCE_ENCODER_URL env var.
      - If retrieval_backend is GeminiAPIBackend: our backend intercepts the
        trigger and routes retrieval through Gemini, so latency is measured
        by retrieval_breakdown eval hooks on that backend instance.

    TTFAT is computed as wall-clock delta from user audio end → first non-silent
    audio token emitted. In batch offline inference this is driven by the model's
    processing latency (~1 inference step at 12.5Hz ≈ 80ms minimum).
    """

    def __init__(self, checkpoint_path: str, retrieval_backend: RetrievalBackend | None = None):
        self.checkpoint_path = checkpoint_path
        self.retrieval_backend = retrieval_backend or NullBackend()
        self._load_models()

    def _load_models(self) -> None:
        import torch
        from moshi.inference_utils import load_models
        from moshi.inference_utils.utils import get_reference_encoder_url
        from moshi.server import ServerState
        from moshi.stt.local_stt import LocalSpeechToText

        # Build the args namespace that load_models() expects
        args = argparse.Namespace(
            device="cuda:0",
            cfg_coef=1.0,
            batch_size=1,
            hf_repo=None,
            moshi_weight=self.checkpoint_path,
            mimi_weight=None,
            tokenizer=None,
            config=None,
            dtype=torch.bfloat16,
            init_active_speaker="user",
            stt_wait_time=0.5,
            rag_timeout=2.0,
            max_reference_tokens=64,
            vad_window_size=4,
            vad_threshold=0.5,
            power_threshold=-65,
        )

        logger.info("Loading MoshiRAG models from %s", self.checkpoint_path)
        self._mimi, self._text_tokenizer, self._lm_gen = load_models(args)
        self._device = args.device
        self._args = args

        try:
            reference_encoder_url = get_reference_encoder_url()
        except Exception:
            reference_encoder_url = None
            logger.warning("REFERENCE_ENCODER_URL not set — retrieval conditioning disabled")

        self._state = ServerState(
            mimi=self._mimi,
            text_tokenizer=self._text_tokenizer,
            lm_gen=self._lm_gen,
            reference_encoder_url=reference_encoder_url,
            stt_wait_time=args.stt_wait_time,
            gradium_stt=False,
            device=args.device,
            rag_timeout=args.rag_timeout,
            max_reference_tokens=args.max_reference_tokens,
            batch_size=args.batch_size,
            vad_window_size=args.vad_window_size,
            vad_threshold=args.vad_threshold,
            init_active_speaker=args.init_active_speaker,
            power_threshold=args.power_threshold,
        )
        logger.info("Warming up MoshiRAG model")
        self._state.warmup()
        self._stt_template = LocalSpeechToText(self._mimi)
        logger.info("MoshiRAG ready")

    def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes via the streaming ASR component."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            wav_path = Path(f.name)
        try:
            stt = deepcopy(self._stt_template)
            return asyncio.run(stt.transcribe_file(wav_path))
        finally:
            wav_path.unlink(missing_ok=True)

    def respond(self, audio_in: bytes) -> tuple[bytes, str, dict]:
        """
        Run one inference turn.

        Writes audio_in to a temp WAV, runs InferenceJob to completion,
        returns (audio_out_wav, inner_monologue_text, timing_metadata).
        """
        import numpy as np
        from moshi.inference_utils.inference_job import InferenceJob

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
            fin.write(audio_in)
            in_path = Path(fin.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fout:
            out_path = Path(fout.name)

        try:
            stt = deepcopy(self._stt_template)
            state = self._state
            retrieval_backend = self.retrieval_backend

            async def _run() -> _TimedInferenceJob:
                step_task = asyncio.create_task(state._step_loop(), name="step-loop")
                raw_job = InferenceJob(
                    state,
                    in_path,
                    out_path,
                    stop_on_end_of_input=True,
                )
                timed = _TimedInferenceJob(raw_job, retrieval_backend)
                slot_idx = await state.wait_acquire_slot(raw_job)
                timed.slot_idx = slot_idx
                timed.stt = stt
                try:
                    async with asyncio.TaskGroup() as tg:
                        await timed.run(tg)
                finally:
                    if timed.slot_idx >= 0:
                        await state.release_slot(timed.slot_idx)
                step_task.cancel()
                try:
                    await step_task
                except asyncio.CancelledError:
                    pass
                return timed

            job = asyncio.run(_run())

            # Reconstruct audio WAV from accumulated PCM chunks
            sample_rate = int(self._state.runner.mimi.sample_rate)
            if job._model_pcm_chunks:
                pcm = np.concatenate(job._model_pcm_chunks).astype(np.float32)
                audio_out = _float32_pcm_to_wav(pcm, sample_rate)
            else:
                audio_out = _silent_wav(duration_s=0.1, sample_rate=sample_rate)

            trace = job.trace
            inner_text = "".join(trace.get("model_text", []))

            metadata = {
                "ttfat_s": job.ttfat_s,
                "first_audio_token_ts": job.t_first_audio,
                "retrieval_text": trace.get("reference_text", ""),
                "retrieval_latency_s": job.retrieval_latency_s,
                "rag_triggered": "rag_trigger_step" in trace,
                "rag_trigger_step": trace.get("rag_trigger_step"),
                "question_end_step": trace.get("question_end_step"),
            }

            return audio_out, inner_text, metadata

        finally:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            out_path.with_suffix(".wav").unlink(missing_ok=True)
