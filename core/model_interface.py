import abc
import argparse
import asyncio
import logging
import os
import re
import struct
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import AsyncIterator

from core.retrieval_backend import NullBackend, RetrievalBackend

logger = logging.getLogger(__name__)

# MoshiRAG outputs 24kHz audio
_SAMPLE_RATE = 24000

# Name of the conditioner subset loaded from config.json — matches the
# --conditioner flag the now-removed server_conditioner.py sidecar used.
_CONDITIONER_NAME = "reference_with_time"

# Consecutive silent (<pad>) model-text steps after question_end_step before
# InferenceJob._output_loop decides the model has finished responding and
# finalizes the job (see _live_feed_loop). run_inference.py's own CLI default
# for the equivalent flag is None (unused) even in its streaming branch, so
# there's no reference value to copy — this is a judgment call, not a
# measured constant. Mimi/Moshi steps at 12.5Hz, so 25 steps ≈ 2s of
# continuous silence. Tune if real audio shows premature/late cutoffs.
_TAIL_SILENCE_STEPS = 25

# moshi-rag's ServerState unconditionally constructs its own built-in
# LLMReferenceGenerator (reads LLM_BASE_URL/LLM_API_KEY/LLM_MODEL_NAME env
# vars via moshi.llm.client.LLMClient), independent of our RetrievalBackend
# abstraction. LLM_BASE_URL is read via a strict os.environ[...] lookup at
# *construction* time, so ServerState() raises KeyError immediately if it's
# unset — and ServerState.warmup() -> reference_generator.warmup() does make
# one real API call against it (confirmed: this crashed once already on a
# stale model name). But at actual retrieval time, moshi's own
# generate_reference_text() is never reached — _TimedInferenceJob
# ._patch_rag_manager patches RAGManager.get_reference_text directly, one
# level below where LLMReferenceGenerator would otherwise be called. Default
# these below (setdefault, so an explicit .env override still wins) rather
# than requiring manual .env setup.
_DEFAULT_LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_DEFAULT_LLM_MODEL_NAME = "gemini-3.5-flash"

_BYTE_FALLBACK_TOKEN_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def _clean_model_text(pieces: list[str]) -> str:
    """
    Detokenize trace["model_text"] into readable text.

    inference_job.py's own _output_loop computes a cleaned, space-normalized
    version of each token for its internal turn-taking/display logic
    (_decode_text_token does text.replace("▁", " ")) but appends the *raw*
    id_to_piece() output to the trace instead of that cleaned value — so the
    trace (and anything built from it, like respond()'s returned text) is
    literal SentencePiece pieces: word-start markers as "▁" and occasional
    byte-fallback pieces like "<0x00>" for bytes with no direct vocab entry.
    Reproduces the same "▁" -> " " replacement here, plus drops byte-fallback
    pieces entirely (never meaningful spoken content, same category as the
    "<pad>" filtering this already needs).
    """
    words = [
        p.replace("▁", " ")
        for p in pieces
        if p != "<pad>" and not _BYTE_FALLBACK_TOKEN_RE.match(p)
    ]
    return "".join(words).strip()


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
            "retrieval_context": self._TRANSCRIPTION if self.retrieval_backend is not None else "",
            "retrieval_text": ref_text,
            "retrieval_latency_s": retrieval_latency_s,
        }

        return _silent_wav(), self._RESPONSE_TEXT, metadata

    async def respond_stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[tuple[bytes, str, dict]]:
        """
        No-GPU streaming stand-in for MoshiRAGAdapter.respond_stream(), used so
        demo/server.py can be exercised (e.g. in tests) without a GPU. For each
        incoming raw-PCM chunk, echoes back a same-length silent chunk. Calls
        the retrieval backend once per stream, not per chunk.
        """
        ref_text = ""
        retrieval_latency_s = 0.0
        if self.retrieval_backend is not None:
            ref_text, retrieval_latency_s = self.retrieval_backend.retrieve(
                self._TRANSCRIPTION
            )

        async for chunk in audio_chunks:
            n_samples = max(1, len(chunk) // 2)  # 16-bit PCM -> 2 bytes/sample
            audio_out = b"\x00\x00" * n_samples
            metadata = {
                "ttfat_s": self._TTFAT_S,
                "first_audio_token_ts": time.time(),
                "retrieval_text": ref_text,
                "retrieval_latency_s": retrieval_latency_s,
            }
            yield audio_out, self._RESPONSE_TEXT, metadata


# ── MoshiRAGAdapter ───────────────────────────────────────────────────────────
# All GPU deps (torch, moshi) are lazy-imported inside methods so this module
# can be imported on local dev machines without a GPU environment.


def _pcm16_to_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    """Wrap raw 16-bit mono PCM bytes (as sent by the browser client) in a WAV header."""
    data_size = len(pcm_bytes)
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_size)
        + pcm_bytes
    )


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
        self.retrieval_context: str = ""
        self._t_question_end: float | None = None
        self._first_audio_recorded = False

        # Patch the job's RAGManager retrieval to route through our backend —
        # unconditionally, including NullBackend. The model predicts <ret>
        # tokens as an intrinsic trained behavior independent of our config,
        # and ServerState.__init__ unconditionally constructs moshi's own
        # LLMReferenceGenerator regardless of our RetrievalBackend choice
        # (see _DEFAULT_LLM_BASE_URL above). Skipping the patch for
        # NullBackend (the original condition here) does NOT disable
        # retrieval — it just means <ret> predictions fall through to
        # moshi's own unpatched, unlatency-gated internal retrieval instead,
        # silently hitting a real endpoint even when the config says
        # retrieval is off. NullBackend.retrieve() already does the right
        # thing (instant empty return, no network call), so patching it in
        # too is what actually makes retrieval.enabled: false true.
        if retrieval_backend is not None:
            self._patch_rag_manager()
            self._patch_catch_reference_text()

    def _patch_rag_manager(self):
        """
        Route retrieval through our RetrievalBackend instead of moshi's own
        LLMReferenceGenerator.

        RAGManager.trigger(task_group, wait_steps, handle_reference_fn,
        context_provider) orchestrates *when* retrieval happens (task
        scheduling, feeding the result into conditioning via
        handle_reference_fn) — that stays untouched. The actual retrieval
        fetch happens one level deeper, in RAGManager.get_reference_text
        (context), which normally calls
        self.reference_generator.generate_reference_text(...) — moshi's
        built-in Gemini-flavored retrieval, gated behind LLM_BASE_URL/
        LLM_API_KEY/LLM_MODEL_NAME (see _DEFAULT_LLM_BASE_URL above). We
        patch get_reference_text directly so trigger()'s own task/
        conditioning-injection orchestration is preserved unmodified —
        confirmed against moshi-rag's actual rag_manager.py source, not
        guessed; the original patch here (which patched trigger() itself,
        assuming a `context: str` positional first argument) had never
        actually been exercised until a real <ret> prediction fired for the
        first time, and its signature was wrong.
        """
        backend = self._retrieval_backend

        async def _patched_get_reference_text(context: str) -> tuple[str, str, float, str]:
            t0 = time.perf_counter()
            self.retrieval_context = context
            try:
                ref_text, latency = backend.retrieve(context)
                self.retrieval_latency_s = latency
            except Exception as exc:
                logger.warning("RetrievalBackend.retrieve() failed: %s", exc)
                ref_text, latency = "", 0.0
            elapsed = time.perf_counter() - t0
            # Distinct "[RetrievalBackend]" tag so this doesn't get confused
            # with moshi's own "[Reference] Triggering retrieval with
            # context_len=N snippet='...'" line — that one only shows a
            # truncated tail of context and never logs what came back.
            # Neither context nor ref_text is truncated here: seeing the full
            # exchange (including confirming NullBackend genuinely returned
            # empty, not silently truncated something real) is the point.
            logger.info(
                "[RetrievalBackend] context=%r -> reference_text=%r (backend_latency=%.3fs)",
                context, ref_text, latency,
            )
            return context, ref_text, elapsed, "RetrievalBackend"

        self._job.rag_manager.get_reference_text = _patched_get_reference_text

    def _patch_catch_reference_text(self):
        """
        Work around a race in moshi-rag's own step-loop bookkeeping —
        confirmed against inference_job.py's actual source, not guessed.

        When a <ret> token is predicted, _output_loop immediately sets
        self._retrieval_start_step = step_index + wait_steps and calls
        rag_manager.trigger(...). _retrieval_start_time (a *different*
        attribute) is only set later, via an exact equality check inside
        the per-step loop: `if self.step_index == self._retrieval_start_step`.
        RAGManager.trigger()'s own internal wait_steps timer is a separate,
        independent counting mechanism from that step_index check — nothing
        guarantees they stay in lockstep. If handle_reference_fn
        (InferenceJob._catch_reference_text) fires before that equality
        check has ever hit, self._retrieval_start_time is still None, and
        _catch_reference_text's second line is a bare `assert
        self._retrieval_start_time is not None` with no message — crashing
        the whole reference-generation task. moshi's own _background_task
        catches this into a one-line log with no traceback
        ("[Reference] Error generating reference: "), which is what
        surfaced this in the first place.

        UPDATE, confirmed via a step-index watchdog on a real run: this is
        NOT harmless bookkeeping — self._doing_retrieval staying True
        forever silently deadlocks the whole step loop (step_index stops
        advancing entirely, no exception, no timeout). _catch_reference_text
        sets self._retrieval_done_step = self.step_index + retrieval_steps
        as *another* future step target, checked via the same kind of exact
        step_index == ... equality comparison as _retrieval_start_step —
        and our own defaulting of _retrieval_start_time to "now" above makes
        retrieval_elapsed compute to ~0, so _retrieval_done_step resolves to
        essentially "whatever step it is right now" — a target the main
        step loop, running concurrently in the background, has very
        plausibly already advanced past by the time this callback finishes,
        for the exact same missed-equality-check reason as the first race.
        Rather than trust that mechanism a second time, set
        self._doing_retrieval = False ourselves once we know
        handle_reference_fn has actually completed.
        """
        original = self._job._catch_reference_text

        async def _safe_catch_reference_text(reference_text, lm_label: str = "") -> None:
            if self._job._retrieval_start_time is None:
                logger.warning(
                    "moshi's _retrieval_start_time was still None when "
                    "handle_reference_fn fired (known step-index race) — "
                    "defaulting to now rather than crashing retrieval"
                )
                self._job._retrieval_start_time = time.monotonic()
            await original(reference_text, lm_label)
            if self._job._doing_retrieval:
                logger.warning(
                    "moshi's _doing_retrieval was still True after "
                    "handle_reference_fn completed (known step-index race, "
                    "same root cause as _retrieval_start_time above) — "
                    "clearing it directly rather than deadlocking the step loop"
                )
                self._job._doing_retrieval = False

        self._job._catch_reference_text = _safe_catch_reference_text

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


async def _step_watchdog(raw_job) -> None:
    """
    Logs step_index every 3s so a stalled turn shows definitively whether the
    model's own step loop is still advancing — a true deadlock here raises no
    exception and hits no timeout on its own, confirmed on a real run.

    Also a safety net for a confirmed race: _patch_catch_reference_text's own
    fix (clearing _doing_retrieval right after handle_reference_fn completes)
    only works if _output_loop's step loop has already set _doing_retrieval =
    True by the time that callback runs. With NullBackend, retrieval is
    near-instant (no network call), so the callback can complete *before*
    _output_loop's own step_index == self._retrieval_start_step check ever
    fires — meaning _doing_retrieval gets set True only *after* our callback
    already checked and found it False, and nothing is left to ever clear it
    — a real run confirmed exactly this (our own "still True" warning never
    fired, yet step_index still froze with doing_retrieval stuck True).
    Whether the callback wins or loses that race depends on retrieval
    latency, not something a callback alone can reliably fix — so if
    step_index is stalled with doing_retrieval stuck True for two
    consecutive checks (6s+, well past any legitimate retrieval), clear it
    here regardless of why it got stuck.

    Shared by respond() and respond_stream(): confirmed on respond_stream()
    first, but both paths build a raw InferenceJob and drive it through
    _TimedInferenceJob.run() against the same moshi-rag step loop, so
    respond() was always exposed to the identical deadlock — it just hadn't
    been exercised against a real checkpoint yet. Confirmed on a real
    respond() run: stuck at step_index with doing_retrieval=True and no
    watchdog, hung indefinitely with no exception until this was added.
    """
    last_step = None
    stalled_while_retrieving = 0
    while True:
        await asyncio.sleep(3.0)
        current_step = raw_job.step_index
        advancing = current_step != last_step
        # Confirmed working via this exact log at INFO level across several
        # real runs — kept at DEBUG now so normal sessions aren't spammed
        # every 3s; the WARNING below, which fires only when it actually has
        # to intervene, stays visible at the default level.
        logger.debug(
            "[watchdog] step_index=%d (%s) doing_retrieval=%s slot_idx=%s",
            current_step,
            "advancing" if advancing else "STALLED",
            raw_job._doing_retrieval,
            raw_job.slot_idx,
        )
        if not advancing and raw_job._doing_retrieval:
            stalled_while_retrieving += 1
            if stalled_while_retrieving >= 2:
                logger.warning(
                    "[watchdog] step_index stalled at %d with doing_retrieval "
                    "stuck True for %ds — forcibly clearing (known race: "
                    "_output_loop can set this True after our own callback "
                    "already checked and finished, especially with "
                    "near-instant NullBackend retrieval)",
                    current_step, stalled_while_retrieving * 3,
                )
                raw_job._doing_retrieval = False
        else:
            stalled_while_retrieving = 0
        last_step = current_step


class MoshiRAGAdapter(ModelInterface):
    """
    Wraps the moshi-rag inference pipeline for use in the eval framework.

    Requires the GPU environment (uv sync --extra gpu) and a valid checkpoint.
    All moshi imports are deferred to init so this module is importable locally.

    Conditioning:
      - The ARC-Encoder (reference-text conditioner) is loaded in-process from
        the checkpoint via moshi.server_conditioner.EncoderService — no HTTP
        sidecar, no REFERENCE_ENCODER_URL. moshi-rag's InferenceJob normally
        fetches the conditioning tensor via an HTTP call to a separately
        launched server_conditioner.py process; that call site
        (get_conditioning_remote_async) is monkeypatched in _load_models() to
        call the local EncoderService directly instead. See
        _patch_inprocess_conditioning().

    Retrieval:
      - If retrieval_backend is NullBackend or None: moshi-rag's RAG path is
        still active (predicts <ret>) but produces no reference text.
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

    def _resolve_checkpoint_paths(self, checkpoint_dir: Path) -> dict[str, Path]:
        """
        Resolve the checkpoint directory's files: config.json and
        model.safetensors have fixed names; the Mimi weights and text
        tokenizer are versioned per training run (tokenizer-*.safetensors,
        *.model) and must be globbed — matching the naming the abandoned
        scripts/run_demo.sh used to derive before this in-process loading
        path replaced it.

        moshi.inference_utils.utils.load_models() unconditionally calls
        CheckpointInfo.from_hf_repo(hf_repo, moshi_weight, mimi_weight,
        tokenizer, config_path=config); hf_repo is only ever dereferenced for
        whichever of those four is left None. Passing all four as real local
        paths (as this method does) means hf_repo=None is safe and no
        network call happens.
        """
        config_path = checkpoint_dir / "config.json"
        moshi_weight_path = checkpoint_dir / "model.safetensors"
        mimi_candidates = sorted(checkpoint_dir.glob("tokenizer-*.safetensors"))
        tokenizer_candidates = sorted(checkpoint_dir.glob("*.model"))

        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found at {config_path}")
        if not moshi_weight_path.exists():
            raise FileNotFoundError(f"model.safetensors not found at {moshi_weight_path}")
        if not mimi_candidates:
            raise FileNotFoundError(
                f"no tokenizer-*.safetensors (Mimi weights) found in {checkpoint_dir}"
            )
        if not tokenizer_candidates:
            raise FileNotFoundError(f"no *.model (text tokenizer) found in {checkpoint_dir}")

        return {
            "config": config_path,
            "moshi_weight": moshi_weight_path,
            "mimi_weight": mimi_candidates[0],
            "tokenizer": tokenizer_candidates[0],
        }

    def _load_models(self) -> None:
        import torch
        from moshi.inference_utils import load_models
        from moshi.server import ServerState
        from moshi.stt.local_stt import LocalSpeechToText

        checkpoint_dir = Path(self.checkpoint_path)
        ckpt_paths = self._resolve_checkpoint_paths(checkpoint_dir)

        # Build the args namespace that load_models() expects
        args = argparse.Namespace(
            device="cuda:0",
            cfg_coef=1.0,
            batch_size=1,
            hf_repo=None,
            moshi_weight=str(ckpt_paths["moshi_weight"]),
            mimi_weight=str(ckpt_paths["mimi_weight"]),
            tokenizer=str(ckpt_paths["tokenizer"]),
            config=str(ckpt_paths["config"]),
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

        # Matches run_inference.py exactly: LocalSpeechToText gets its own
        # deepcopy of mimi, built before ServerState/warmup() ever touch the
        # original. Sharing self._mimi directly (or building this after
        # warmup()) crashes — ServerState's runner.warmup() puts the shared
        # mimi into a persistent streaming_forever state, and
        # LocalSpeechToText.__init__ asserts its mimi isn't already streaming.
        self._stt_template = LocalSpeechToText(deepcopy(self._mimi))

        self._arc_encoder = self._load_arc_encoder(args, ckpt_paths)
        self._patch_inprocess_conditioning()

        # See _DEFAULT_LLM_BASE_URL above — moshi's ServerState.__init__
        # requires these to exist even though its own LLM call is never
        # actually reached in our flow.
        os.environ.setdefault("LLM_BASE_URL", _DEFAULT_LLM_BASE_URL)
        os.environ.setdefault("LLM_MODEL_NAME", _DEFAULT_LLM_MODEL_NAME)
        os.environ.setdefault("LLM_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

        self._state = ServerState(
            mimi=self._mimi,
            text_tokenizer=self._text_tokenizer,
            lm_gen=self._lm_gen,
            # Vestigial after _patch_inprocess_conditioning(): the only code
            # that ever read this attribute (get_conditioning_remote_async,
            # patched below) now calls self._arc_encoder directly instead of
            # making an HTTP request, so no real URL is needed here.
            reference_encoder_url="in-process",
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
        logger.info("MoshiRAG ready")

    def _load_arc_encoder(self, args: argparse.Namespace, ckpt_paths: dict[str, Path]):
        """
        Load the ARC-Encoder (reference-text conditioner) in-process from the
        checkpoint directory — same config.json, model.safetensors, and
        conditioner name the abandoned server_conditioner.py sidecar used to
        load, but instantiated directly in this process instead of behind a
        FastAPI /embed route.
        """
        from moshi.server_conditioner import EncoderService

        logger.info("Loading ARC-Encoder in-process from %s", self.checkpoint_path)
        return EncoderService(
            config=str(ckpt_paths["config"]),
            moshi_weight=str(ckpt_paths["moshi_weight"]),
            conditioner=_CONDITIONER_NAME,
            device=args.device,
        )

    def _patch_inprocess_conditioning(self) -> None:
        """
        moshi-rag's InferenceJob fetches the RAG conditioning tensor via
        get_conditioning_remote_async(text, encoder_url), which POSTs to
        {encoder_url}/embed on a separately-running server_conditioner.py
        process. We run the ARC-Encoder in-process (self._arc_encoder), so
        replace that call site with one that calls it directly — no HTTP, no
        sidecar. Monkeypatched rather than forked, consistent with
        _TimedInferenceJob._patch_rag_manager above.
        """
        import torch

        import moshi.inference_utils.inference_job as inference_job_module

        encoder = self._arc_encoder

        def _encode_and_sync(text: str):
            result = encoder.encode(text)
            # encoder.encode() runs on a separate OS thread (via
            # asyncio.to_thread below) from the main generation loop's
            # thread. CUDA work dispatched from a different thread is not
            # automatically synchronized with the consuming thread's stream
            # — if this tensor crosses back into
            # _async_update_reference/update_streaming_sum_tensors before
            # its CUDA ops have actually completed, that's exactly the kind
            # of cross-thread stream-ordering issue that silently deadlocks
            # the main step loop rather than raising: no exception, no
            # timeout, generation just never produces another step. Force
            # completion here, inside this thread, before the tensor ever
            # crosses the thread boundary.
            torch.cuda.synchronize(encoder.device)
            return result

        async def _local_conditioning(text: str, encoder_url: str | None = None):
            # _async_update_reference (moshi's own caller) passes reference_text
            # through unconditionally, no special-casing for empty strings —
            # confirmed against inference_job.py's source. Pass it through
            # unmodified rather than substituting a placeholder: arc_encoder.py's
            # _get_condition has its own native handling for empty text (it
            # detects the empty case via the attention mask specifically to
            # apply a learnt padding embedding — learnt_padding is a real,
            # trained weight in this checkpoint's conditioner state dict, per
            # the paper's dropout-training description, section 4.2). A
            # substituted placeholder would bypass that intended path and go
            # through normal (meaningless, out-of-distribution) encoding
            # instead — likely worse, not safer.
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(_encode_and_sync, text), timeout=15.0
                )
            except asyncio.TimeoutError:
                logger.error(
                    "In-process ARC-Encoder conditioning timed out after 15s "
                    "for text=%r", text
                )
                raise
            except Exception:
                # This path has never been exercised with a real reference
                # string until recently. Whatever calls
                # get_conditioning_remote_async appears to swallow exceptions
                # into a bare log line with no traceback (same pattern as
                # _catch_reference_text's caller) — log the full traceback
                # ourselves before it disappears, since audio generation going
                # silent right around this point is exactly what an unhandled
                # exception here would look like from the outside.
                logger.exception(
                    "In-process ARC-Encoder conditioning failed for text=%r", text
                )
                raise

        inference_job_module.get_conditioning_remote_async = _local_conditioning

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
                    # UPDATE, confirmed against the real inference_job.py
                    # source (not the earlier, wrong assumption that
                    # stop_on_end_of_input=True + max_tail_silence=None
                    # "matches run_inference.py's defaults" — run_inference.py
                    # actually defaults stop_on_end_of_input to False, and its
                    # own argparse *requires* a real --max-consecutive-
                    # silence-frames value whenever it's False). With
                    # stop_on_end_of_input=True, _output_loop switches to a
                    # hard 1-second asyncio.wait_for() timeout on
                    # output_queue.get() the moment input feeding ends — and
                    # finalizes the WHOLE job the instant that fires, even
                    # mid-<ret>-wait (confirmed on a real run: "[Reference]
                    # Reference generation cancelled" logged right after
                    # "Started waiting for 6 steps", well before that wait
                    # elapsed). That produced near-empty responses across
                    # every question in a real smoke test — the model never
                    # got a real chance to speak. stop_on_end_of_input=False
                    # keeps feeding silence and calling output_queue.get()
                    # with NO timeout after input ends — termination instead
                    # comes from max_tail_silence's consecutive-<pad>-token
                    # check (_check_tail_silence), the same bounded mechanism
                    # respond_stream() already relies on. Not gating it
                    # behind something like _client_done here: unlike
                    # respond_stream()'s continuous multi-turn session,
                    # respond() is inherently one question in, one answer
                    # out — a natural pause SHOULD end the job.
                    stop_on_end_of_input=False,
                    use_gt_reference=False,
                    max_tail_silence=_TAIL_SILENCE_STEPS,
                    sidecar={},
                )
                timed = _TimedInferenceJob(raw_job, retrieval_backend)
                slot_idx = await state.wait_acquire_slot(raw_job)
                timed.slot_idx = slot_idx
                timed.stt = stt
                # See _step_watchdog's docstring: respond() drives the same
                # moshi-rag step loop as respond_stream() and is exposed to
                # the identical _doing_retrieval deadlock — confirmed on a
                # real respond() run (hung indefinitely on a <ret> trigger,
                # no exception, before this was added).
                watchdog_task = asyncio.create_task(_step_watchdog(raw_job), name="step-watchdog")
                try:
                    async with asyncio.TaskGroup() as tg:
                        await timed.run(tg)
                finally:
                    watchdog_task.cancel()
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
            inner_text = _clean_model_text(trace.get("model_text", []))

            metadata = {
                "ttfat_s": job.ttfat_s,
                "first_audio_token_ts": job.t_first_audio,
                # What was actually sent to RetrievalBackend.retrieve() —
                # previously invisible anywhere (moshi's own "[Reference]
                # Triggering retrieval..." log only shows a truncated tail).
                "retrieval_context": job.retrieval_context,
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

    async def respond_stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[tuple[bytes, str, dict]]:
        """
        Streaming variant of respond(), used by demo/server.py. Genuinely
        full-duplex: incoming raw-PCM chunks are fed into InferenceJob's
        input_queue frame-by-frame as they arrive — via a custom _feed_loop
        (_live_feed_loop below) swapped onto the job instance in place of
        InferenceJob's own, which only supports loading one complete wav file
        up front (confirmed against moshi-rag's source — see
        _live_feed_loop's docstring for exactly which internals this
        reconstructs). So the model can begin producing output while the
        user is still talking, rather than waiting for a full utterance.
        Output is also streamed: each PCM chunk is yielded to the caller as
        soon as it's produced, via a tee on output_queue.get().

        NOT YET VERIFIED AGAINST A REAL CHECKPOINT/GPU. This reconstructs
        InferenceJob's internal frame-feeding contract (StepInput,
        AudioProcessor.filter_by_power, input_queue, _wait_step_index_at_least,
        _shutdown_event, _feed_finished) from moshi-rag's published source,
        not a live install — treat as a first draft pending a real run. See
        scripts/verify_respond_stream.py.
        """
        import numpy as np
        import torch
        from moshi.inference_utils.inference_job import InferenceJob, StepInput

        sample_rate = int(self._state.runner.mimi.sample_rate)
        # Never actually read (see _live_feed_loop) — InferenceJob.__init__
        # does no file I/O, it just stores wav_path for _feed_loop to open
        # lazily, and we replace _feed_loop entirely.
        placeholder_in = _silent_wav(duration_s=0.01, sample_rate=sample_rate)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
            fin.write(placeholder_in)
            in_path = Path(fin.name)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fout:
            out_path = Path(fout.name)

        output_chunks: asyncio.Queue = asyncio.Queue()
        _DONE = object()

        async def _live_feed_loop(raw_job) -> None:
            """
            Replaces InferenceJob._feed_loop, adapted for a live audio_chunks
            source instead of a preloaded wav file, but otherwise mirroring
            its two-phase structure exactly:

            Phase 1 (real audio): assemble server.frame_size windows from
            audio_chunks as they arrive, push StepInput into input_queue,
            pace via _wait_step_index_at_least — same as the original. Each
            step updates trace["question_end_step"], same as the original.

            Phase 2 (tail silence, once audio_chunks is exhausted): keep
            feeding *silence* frames — never breaks out and returns on its
            own — so the model has room to keep responding. Crucially,
            question_end_step is NOT updated in this phase, exactly like the
            original's own post-wav-exhaustion loop: this freezes the window
            _check_tail_silence() scans in InferenceJob._output_loop, which
            is what actually detects the model has gone quiet and finalizes
            the job (sets _shutdown_event) — but only once raw_job._client_done
            is True (set below, on StopAsyncIteration); _check_tail_silence is
            monkeypatched in _run_job to return False otherwise, since it can
            misfire mid-conversation while the client is still connected (see
            that patch's comment). Without max_tail_silence being a real
            number in the first place, that detection never fires at all and
            the job hangs forever regardless of client_done — this loop only
            ever exits via the _shutdown_event/_wait_step_index_at_least
            checks below, same as the original.
            """
            device = raw_job.server.device
            frame = raw_job.server.frame_size
            max_stream_delay = max(raw_job.server.runner.lm_gen.delays_cuda).item()

            buffer = np.zeros(0, dtype=np.float32)
            client_done = False
            feed_step = 0
            is_first = True
            chunk_iter = audio_chunks.__aiter__()

            while True:
                if feed_step > 0 and not await raw_job._wait_step_index_at_least(
                    feed_step - max_stream_delay
                ):
                    return
                if raw_job._shutdown_event.is_set():
                    return

                if client_done:
                    frame_np = np.zeros(frame, dtype=np.float32)
                else:
                    while len(buffer) < frame and not client_done:
                        try:
                            chunk_bytes = await asyncio.wait_for(
                                chunk_iter.__anext__(), timeout=frame / sample_rate
                            )
                        except asyncio.TimeoutError:
                            break  # no real audio in time for this step — pad below
                        except StopAsyncIteration:
                            client_done = True
                            raw_job._client_done = True
                            break
                        incoming = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        buffer = np.concatenate([buffer, incoming])

                    if len(buffer) >= frame:
                        frame_np = buffer[:frame]
                        buffer = buffer[frame:]
                    else:
                        pad = np.zeros(frame - len(buffer), dtype=np.float32)
                        frame_np = np.concatenate([buffer, pad])
                        buffer = np.zeros(0, dtype=np.float32)
                    raw_job.trace["question_end_step"] = feed_step

                chunk = torch.from_numpy(frame_np).to(device=device, dtype=torch.float32)[None, None]
                filtered = raw_job.audio_processor.filter_by_power(chunk)
                await raw_job.input_queue.put(
                    StepInput(filtered_pcm=filtered[0], is_first=is_first, pcm=chunk[0])
                )
                await raw_job.stt.send_audio(frame_np.astype(np.float32))

                is_first = False
                feed_step += 1

        async def _run_job() -> None:
            stt = deepcopy(self._stt_template)
            state = self._state
            step_task = asyncio.create_task(state._step_loop(), name="step-loop")
            # stop_on_end_of_input is only read by InferenceJob's own
            # _feed_loop, which we replace below, so its value here is moot —
            # kept False since our loop, not this flag, decides when to stop.
            # use_gt_reference/sidecar match run_inference.py's own argparse
            # defaults (see the respond() call site above), but
            # max_tail_silence must be a real number here, unlike respond()'s
            # None — see _TAIL_SILENCE_STEPS and _live_feed_loop's docstring.
            raw_job = InferenceJob(
                state,
                in_path,
                out_path,
                stop_on_end_of_input=False,
                use_gt_reference=False,
                max_tail_silence=_TAIL_SILENCE_STEPS,
                sidecar={},
            )
            raw_job._feed_loop = lambda: _live_feed_loop(raw_job)
            raw_job._client_done = False  # set True by _live_feed_loop, see below

            # _check_tail_silence (gated by max_tail_silence=_TAIL_SILENCE_STEPS
            # above) exists to end ONE-SHOT jobs — e.g.
            # scripts/verify_respond_stream.py's finite wav file — once the
            # model's response naturally tails off after input ends. For this
            # live, continuous session it must NEVER end the job while the
            # client is still connected. Confirmed on a real run that it can
            # misfire mid-conversation: during the multi-second
            # doing_retrieval stall the watchdog below recovers from,
            # _live_feed_loop is ALSO blocked (it awaits the same frozen
            # step_index via _wait_step_index_at_least before feeding each
            # frame), so question_end_step stops advancing right alongside
            # the stall. Once step_index unfreezes, the model can resume
            # generating (and emit a burst of <pad> tokens) before
            # _live_feed_loop's own task is rescheduled and catches
            # question_end_step back up — a brief window where
            # _check_tail_silence's scan (stale question_end_step to now)
            # can see enough consecutive padding to end the whole session,
            # even though the client never disconnected. Only trust the
            # original check once client_done is actually True.
            original_check_tail_silence = raw_job._check_tail_silence

            def _guarded_check_tail_silence():
                if not raw_job._client_done:
                    return False
                return original_check_tail_silence()

            raw_job._check_tail_silence = _guarded_check_tail_silence

            timed = _TimedInferenceJob(raw_job, self.retrieval_backend)

            # Tee output_queue.get(): the normal consumer (InferenceJob's own
            # _output_loop, watched by _TimedInferenceJob for TTFAT) still
            # sees every item, but each chunk is *also* forwarded to our local
            # queue so we can yield it to the caller as it's produced.
            original_get = timed._job.output_queue.get

            async def _tee_get():
                out = await original_get()
                pcm = getattr(out, "pcm", None) if out is not None else None
                if pcm is not None:
                    await output_chunks.put(pcm.detach().cpu().float().numpy().reshape(-1))
                return out

            timed._job.output_queue.get = _tee_get

            watchdog_task = asyncio.create_task(_step_watchdog(raw_job), name="step-watchdog")

            slot_idx = await state.wait_acquire_slot(raw_job)
            timed.slot_idx = slot_idx
            timed.stt = stt
            try:
                async with asyncio.TaskGroup() as tg:
                    await timed.run(tg)
            finally:
                watchdog_task.cancel()
                if timed.slot_idx >= 0:
                    await state.release_slot(timed.slot_idx)
            step_task.cancel()
            try:
                await step_task
            except asyncio.CancelledError:
                pass

        async def _run_and_signal_done() -> None:
            try:
                await _run_job()
            finally:
                await output_chunks.put(_DONE)

        job_task = asyncio.create_task(_run_and_signal_done())

        try:
            while True:
                item = await output_chunks.get()
                if item is _DONE:
                    break
                pcm_i16 = (np.clip(item, -1.0, 1.0) * 32767).astype(np.int16)
                yield pcm_i16.tobytes(), "", {}
            await job_task
        finally:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            out_path.with_suffix(".wav").unlink(missing_ok=True)
