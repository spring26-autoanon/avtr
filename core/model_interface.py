import abc
import argparse
import asyncio
import logging
import os
import re
import struct
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

from core.gpu import DeviceAssignment, ensure_cuda_visible_devices
from core.retrieval_backend import NullBackend, RetrievalBackend

logger = logging.getLogger(__name__)

# MoshiRAG outputs 24kHz audio
_SAMPLE_RATE = 24000

# Consecutive silent (<pad>) model-text steps after question_end_step before
# InferenceJob._output_loop decides the model has finished responding and
# finalizes the job (see _live_feed_loop). run_inference.py's own CLI default
# for the equivalent flag is None (unused) even in its streaming branch, so
# there's no reference value to copy — this is a judgment call, not a
# measured constant. Mimi/Moshi steps at 12.5Hz, so 25 steps ≈ 2s of
# continuous silence. Tune if real audio shows premature/late cutoffs.
# Default for generation["tail_silence_steps"] — see _DEFAULT_GENERATION.
_TAIL_SILENCE_STEPS = 25

# Every generation-affecting InferenceJob parameter MoshiRAGAdapter builds,
# previously hardcoded directly into _load_models()'s argparse.Namespace —
# now a config["model"]["generation"] dict (see core/config.py,
# configs/baseline_*.yaml), merged over these defaults so a config omitting
# the block (or a bare StubModelAdapter/tiny run) keeps working unchanged.
# Field names match moshi.server's own CLI flag names (underscores vs.
# hyphens) exactly — confirmed directly against kyutai-labs/moshi-rag's
# server.py argparse block — so scripts/print_demo_env.py can translate
# this same dict into the demo's CLI flags with a pure name transform.
# batch_size/init_active_speaker are deliberately NOT here — see specs/
# moshirag-evals-requirements.md's "Generation parameters" section for why
# those two stay hardcoded/path-specific rather than shared config.
#
# temp_text/top_k_text are the one exception to the "matches a moshi.server
# CLI flag" rule above: moshi.server has no --temp-text/--top-k-text flag at
# all (confirmed via its real argparse block), so scripts/print_demo_env.py
# excludes both from its CLI-flags translation (_DEMO_EXCLUDED_GENERATION_FIELDS,
# same mechanism tail_silence_steps already uses, for a different reason —
# see that field's own comment below) and instead reads them directly out of
# DEMO_CONFIG for _patch_load_models_generation_overrides() to apply via
# monkeypatch. Defaults below (0.7/25) match LMGen's own hardcoded values
# exactly, so adding these two fields changes nothing until a config
# explicitly overrides them — see CLAUDE.md's pad-token-sampling-drift
# investigation ("Real root cause of demo response lag") for why these two
# are the ones worth exposing: sample_token() (moshi/utils/sampling.py) is
# plain temp/top-k/top-p multinomial sampling on text_logits with no
# repetition penalty or anti-pad bias of any kind, and temp_text/top_k_text
# are the only knobs that reach it — cfg_coef also reaches text_logits (see
# that field's own history above) but is structurally incompatible with the
# per-slot RAG conditioning this project depends on
# (lm.py: "assert self.cfg_coef == 1.0, Per-slot streaming_sum update
# requires cfg_coef == 1."), so it isn't a real second lever here.
_DEFAULT_GENERATION = {
    "cfg_coef": 1.0,
    "stt_wait_time": 0.5,
    # Was 2.0 — raised to match scripts/run_demo.sh's own fix: real
    # Gemini round-trip latency was directly observed at ~2.5-3s (see
    # CLAUDE.md's Phase 0 prodcheck findings), reliably longer than a 2.0s
    # budget. Same underlying defect, same fix, applied here too rather
    # than leaving evals exposed to it.
    "rag_timeout": 8.0,
    "max_reference_tokens": 64,
    "vad_window_size": 4,
    "vad_threshold": 0.5,
    "power_threshold": -65,
    "tail_silence_steps": _TAIL_SILENCE_STEPS,
    "temp_text": 0.7,
    "top_k_text": 25,
}

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
    # Populated by MoshiRAGAdapter; stays None for StubModelAdapter and any
    # other adapter with no real GPU story. See core/gpu.py and specs/
    # moshirag-evals-requirements.md's "GPU Sizing and Multi-GPU Deployment"
    # section.
    gpu_devices: DeviceAssignment | None = None

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


async def _fetch_and_apply_reference_conditioning(
    reference_text: str,
    *,
    encoder_url: str,
    lm_gen,
    batch_size: int,
    slot_idx: int,
) -> None:
    """
    Narrow, evidence-backed replacement for the conditioning-application
    tail moshi-rag's own InferenceJob._async_update_reference() and
    Channel._async_update_reference() each do identically (confirmed via
    inspect.getsource() against the real installed package and a direct
    clone of kyutai-labs/moshi-rag) — used by both core/model_interface.py's
    _TimedInferenceJob (respond()/evals) and scripts/instrumented_server.py's
    Channel patch (demo), so there's one implementation, not two that could
    drift.

    Why this exists — see CLAUDE.md's "SUPERSEDED: GPU contention conclusion
    was wrong" section for the full evidence chain: a real dual-GPU test
    proved the ~1.5-1.9s conditioning latency is NOT GPU compute contention
    (the conditioner's own GPU sat ~0% utilized throughout). Root cause,
    confirmed by reading moshi-rag's real `server.py` source: `ServerState.
    _step_loop`/`run_one_step` calls the real per-step model forward pass
    (`BatchRunner.run_step`) synchronously inside an `async def`, with no
    yield mid-step, blocking the single-threaded event loop for the real,
    logged duration of each step ("batched step ... took 95.3ms" warnings)
    — starving this exact conditioning HTTP call (`get_conditioning_remote_
    async`, a properly-awaited `httpx.AsyncClient` call with no blocking of
    its own) of the event-loop time it needs to complete promptly, even
    though the conditioner itself responds in ~42-100ms.

    That step-loop behavior is NOT a bug to patch around: `lm.py`'s
    `apply_pending_streaming_sum_condition` consumes the conditioning tensor
    one row per real-time step (`state.pending_streaming_sums[b]`, drained
    one entry per call), in lockstep with Mimi/Moshi's fixed 12.5Hz frame
    rate — `run_step`'s own `elapsed_ms >= 77` warning threshold confirms a
    hard real-time budget per step. Splitting `run_step` across threads or
    inserting yields mid-step would risk a real race on
    `pending_streaming_sums` and would fight a genuine real-time constraint
    the model's own audio pacing depends on — moshi-rag's `InferenceJob`/
    `ServerState`/`Channel`/`BatchRunner` step-loop logic is deliberately
    NOT touched here, per this project's own rule against reimplementing it.

    The fix instead: only the conditioning HTTP fetch — pure network I/O,
    zero GPU/model-state involvement — moves off the main event loop, via
    `asyncio.to_thread` running `get_conditioning_remote_async` to
    completion on its own throwaway event loop in a worker thread
    (`asyncio.run()` is safe here: a fresh worker thread has no existing
    loop). `update_streaming_sum_tensors` stays on the calling (main)
    thread, unchanged from upstream — confirmed cheap (~1ms: eval JSON's
    `context_injection_values` matched the HTTP-only "Received response"
    log timings to the millisecond, see CLAUDE.md), and it's the one piece
    of this that touches live, shared model state, so there's no
    correctness reason to move it off-thread too.
    """
    from moshi.inference_utils.inference_job import get_conditioning_remote_async

    def _run_in_new_loop():
        return asyncio.run(get_conditioning_remote_async(text=reference_text, encoder_url=encoder_url))

    t0 = time.perf_counter()
    condition_tensor = await asyncio.to_thread(_run_in_new_loop)
    logger.info(
        "[Reference] ARC encoding received in %.3fs (streaming_sum %s)",
        time.perf_counter() - t0,
        tuple(condition_tensor.shape),
    )
    per_slot = [None] * batch_size
    per_slot[slot_idx] = condition_tensor.squeeze(0)
    lm_gen.update_streaming_sum_tensors(per_slot)


def _patch_load_models_generation_overrides(temp_text: float, top_k_text: int) -> None:
    """
    Makes the front-end's own LMGen construction (inside moshi-rag's
    load_models(), moshi/inference_utils/utils.py) sample text tokens with
    temp_text/top_k_text from our own config instead of LMGen's hardcoded
    defaults (0.7/25) — see _DEFAULT_GENERATION's entries for these two
    fields, and CLAUDE.md's "Real root cause of demo response lag" section
    for why they're the one real lever onto the pad-token-sampling drift
    (sample_token() applies temp/top-k directly to text_logits, with no
    repetition penalty or anti-pad bias anywhere else in the pipeline).

    Patches the module-local name moshi.inference_utils.utils.LMGen — same
    technique as _patch_stt_no_cuda_graph()'s local_stt_mod.LMGen swap
    (subclass swapped in at the module-attribute level), not
    moshi.models.lm.LMGen globally. load_models()'s own `LMGen(...)` call
    resolves that bare name dynamically out of utils.py's own module
    globals every time it runs — not once, at definition time — so this
    reaches the call correctly regardless of which name(s) other modules
    used to reach the load_models *function* itself (moshi.server.py's own
    `from .inference_utils import load_models` binds a completely separate
    name for the function, in moshi.server's own namespace, at moshi.server's
    own import time; patching a re-exported function name like that would
    silently miss callers who already resolved it — see
    feedback_vm_instrumentation_patching's "patch submodule originals not
    re-exports" note). Patching the LMGen name actually referenced *inside*
    load_models()'s body sidesteps that class of bug entirely.

    Scoped to the front-end only: moshi.stt.local_stt.py imports its own,
    separate LMGen name into its own module globals and resolves that one
    inside LocalSpeechToText.__init__ — a different attribute, untouched by
    this patch. Confirmed via real source (same read that grounded
    _patch_stt_no_cuda_graph() above): the front-end and STT builds are two
    independent code paths that happen to construct the same underlying
    class, not one call site both funnel through.

    Subclasses and uses kwargs.setdefault(...), not a forced override or a
    wholesale __init__ replacement — a checkpoint whose own lm_gen_config
    ever specifies temp_text/top_k_text (spread into this same call as
    **kwargs ahead of our subclass's defaults) still wins. True for the
    real checkpoint as of this writing (confirmed empty lm_gen_config).

    Idempotent and process-global (like _patch_stt_no_cuda_graph()): safe to
    call once per process regardless of which entry point (this module's own
    _load_models(), or scripts/instrumented_server.py's apply_patches())
    runs first.
    """
    import moshi.inference_utils.utils as moshi_utils_mod

    if getattr(moshi_utils_mod.LMGen, "_generation_overrides_patched", False):
        return

    # print(), not logger.info() -- this runs from both entry points, and
    # scripts/instrumented_server.py's apply_patches() calls this before
    # moshi.server.main()'s own setup_logging(), so a logger.info() call
    # here would be silently swallowed the same way
    # _patch_server_state_step_pacing()'s docstring already documents for
    # that patch. Placed after the idempotency check (not before) so it
    # only ever states the value that will actually take effect -- this
    # exact ambiguity bit a real A/B test once already (2026-07-29): no way
    # to confirm after the fact whether a scratch config's temp_text
    # override had actually taken effect.
    print(
        f"[Generation] temp_text={temp_text} top_k_text={top_k_text} "
        "(front-end LMGen only -- see _patch_load_models_generation_overrides)",
        file=sys.stderr,
    )

    original_lm_gen_cls = moshi_utils_mod.LMGen

    class _GenerationOverrideLMGen(original_lm_gen_cls):
        _generation_overrides_patched = True

        def __init__(self, *args, **kwargs):
            kwargs.setdefault("temp_text", temp_text)
            kwargs.setdefault("top_k_text", top_k_text)
            super().__init__(*args, **kwargs)

    moshi_utils_mod.LMGen = _GenerationOverrideLMGen


def _warm_up_stt_exec_mask(stt) -> None:
    """
    Forces StreamingModule.set_exec_mask's CUDA graph capture to happen
    synchronously, up front, on this turn's freshly-deepcopy'd `stt`,
    instead of during the turn's first real per-frame call.

    Purely a latency-cleanliness measure: without it the *first* real
    per-frame call of every turn eats a capture's cost rather than a
    replay's, inflating that turn's own latency measurements for no
    reason. That matters most on the eval path, where a fresh `stt` is
    deepcopy'd per respond() call and `latency.ttfat` is measuring exactly
    that window.

    **This is the only survivor of the deleted "Option E" chain** (see the
    _load_models() comment on why the rest went, 2026-07-30). It was
    originally a correctness fix for a cross-thread CUDA-graph capture
    race; that hazard is gone with off-thread STT, and this is kept solely
    on the latency argument above. It is also the only piece of that chain
    that ever ran on the default path — it is called unconditionally, never
    gated by the old STT_OFF_THREAD toggle — so keeping it is what makes
    that deletion a genuine no-op on current behaviour. Whether it still
    earns its keep single-threaded is an open, separately-measurable
    question: it only pre-warms set_exec_mask's own graph, while STT's LM
    forward and Mimi encode graphs are still captured on first real use
    regardless.

    Calls each of stt.mimi.set_exec_mask/stt._lm_gen.set_exec_mask
    *twice*, not once — a real bug in an earlier version of this
    function, found via a real VM crash and confirmed by reading
    CUDAGraphed.__init__'s own default (`warmup_steps: int = 1`, never
    overridden at either construction site in moshi/modules/streaming.py):
    the *first* call to a fresh CUDAGraphed only consumes that warmup
    buffer and executes directly, uncaptured — the capture itself happens
    on the *second* call. Calling once (the original version of this
    function) only ever primed the buffer, leaving the real capture to
    happen on the first real per-frame call instead, exactly the
    concurrent window this function exists to avoid.

    Checked whether stt._lm_gen.set_exec_mask's own nested recursion
    (LMGen.set_exec_mask's set_exec_mask_callback calls a *different*,
    nested StreamingModule's set_exec_mask — see lm.py:549,717) needs its
    *own* separate warm-up accounting, since it's a distinct CUDAGraphed
    instance: traced through moshi-rag's real source and confirmed it
    doesn't — that nested call always happens *while* the outer
    lm_gen.set_exec_mask call already holds `_set_in_cuda_graph()`
    (entered before either its warmup or capture branch runs), so
    `in_cuda_graph()` is already True by the time the nested call's own
    CUDAGraphed.__call__ checks it, and it always takes the early-return,
    uncaptured path — never reaches its own warmup/capture machinery at
    all. Only the two *outer*, directly-called objects (mimi, _lm_gen)
    ever actually build a graph.

    Call site matters: must run after `stt = deepcopy(self._stt_template)`
    but strictly before `_run()`/`self._loop.run_until_complete(...)`
    starts. Confirmed genuinely safe via `_ensure_step_loop()`'s own
    source: the persistent step-loop task is only ever *scheduled*
    (asyncio.create_task, never awaited to run) until something next
    drives `self._loop` — so between turns, and up to this exact point in
    `_respond_once()`, nothing is executing concurrently on any thread.
    `stt.mimi`/`stt._lm_gen` are always a fresh deepcopy per turn (see
    _run_stt_frames_sync's docstring), so `_set_exec_mask_graphed` starts
    `None` again every single call — this must run every turn, not once
    per process.

    Real per-frame calls (moshi-rag's own local_stt.py:129,83) always use
    `torch.ones(1, device=..., dtype=torch.bool)` — batch_size=1, matching
    this module's hardcoded eval batch size — so warming with the exact
    same shape/dtype here means the graph captured now is the same one
    replayed later; CUDAGraphed's own `_match_values_copy_tensors` would
    reject a shape mismatch otherwise.
    """
    import torch

    mask = torch.ones(1, device=stt._device, dtype=torch.bool)
    stt.mimi.set_exec_mask(mask)
    stt.mimi.set_exec_mask(mask)
    stt._lm_gen.set_exec_mask(mask)
    stt._lm_gen.set_exec_mask(mask)


def _maybe_enable_eval_asyncio_debug(loop: "asyncio.AbstractEventLoop") -> None:
    """
    Eval-path counterpart to scripts/instrumented_server.py's
    _patch_event_loop_diagnostics() — same diagnostic (asyncio debug mode +
    a configurable slow_callback_duration, so any slow callback gets logged
    with a repr identifying it), same purpose (see what's occupying the
    event loop during a real gap window), but a much simpler hook here:
    _ensure_step_loop() builds its own loop directly via
    asyncio.new_event_loop() and drives it with loop.run_until_complete(),
    never passing it through asyncio.set_event_loop() at all — no
    asyncio.run() internals to intercept the way the demo path needed (see
    that function's own docstring for the real monkeypatching pitfall hit
    there: patching the asyncio.set_event_loop re-export instead of
    asyncio.events.set_event_loop, the name Python 3.11's asyncio.run()
    actually calls through, which silently never fired). No such pitfall
    here — just configure the loop object directly.

    Opt-in only (EVAL_ASYNCIO_DEBUG=1), off by default — debug mode adds
    real per-callback overhead that would skew the very latency numbers
    under investigation if left on for normal eval runs. Lets
    make smoke/gpu_diag_contended.sh-style runs (already fully scripted,
    no live conversation needed) check whether the LocalSpeechToText-off-
    thread fix behaves as expected on real hardware before spending a live
    demo round trip on it — see CLAUDE.md's "SUPERSEDED: GPU contention
    conclusion was wrong" section. Eval's own feed loop calls
    LocalSpeechToText.send_audio() too (InferenceJob's feed loop, the same
    method demo's Channel._recv_loop calls), so this exercises the same
    patched code path — under eval's self-paced (not network-paced) audio
    feeding, a real but different scenario, not a full substitute for
    confirming the demo-specific backlog magnitude collapses too.
    """
    if os.environ.get("EVAL_ASYNCIO_DEBUG") != "1":
        return
    threshold_s = float(os.environ.get("EVAL_ASYNCIO_DEBUG_THRESHOLD_S", "0.02"))
    loop.set_debug(True)
    loop.slow_callback_duration = threshold_s
    logger.warning(
        "[Diag] asyncio debug mode ON for this loop "
        "(slow_callback_duration=%.3fs, EVAL_ASYNCIO_DEBUG=1)",
        threshold_s,
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
        # Wall-clock time of the ARC-encoder round trip (job._async_update_
        # reference's get_conditioning_remote_async call) — the real
        # /embed HTTP call to the separate-process conditioner, distinct
        # from retrieval_latency_s (the text-retrieval LLM call above it in
        # the pipeline). Timed in _immediate_handle_reference_text below;
        # 0.0 if no <ret> ever fired this turn.
        self.conditioning_latency_s: float = 0.0
        self.retrieval_context: str = ""
        # Wall-clock (perf_counter) timestamp of the most recent <ret>
        # trigger, and the gap between it and get_reference_text() actually
        # being called — see _patch_output_loop()'s <ret> branch (sets
        # retrieval_trigger_ts) and _patch_rag_manager()'s
        # _patched_get_reference_text (computes asr_wait_s from it). This is
        # the real, same wait_steps-based delay RAGManager._background_task
        # (unpatched, shared by both InferenceJob and Channel) always
        # imposes between a <ret> trigger and grabbing the final context —
        # genuinely measurable here too, not an approximation, even though
        # respond() is a one-shot batch call with no live session. Like
        # rag_trigger_step, a second <ret> in the same turn overwrites this
        # before the first's get_reference_text call may have run — same
        # last-trigger-wins fidelity already accepted for rag_trigger_step,
        # not a new limitation introduced here.
        self.retrieval_trigger_ts: float | None = None
        self.asr_wait_s: float = 0.0
        self._t_question_end: float | None = None
        self._first_audio_recorded = False
        # (feed_step -> wall-clock time) for every _feed_loop() call to
        # input_queue.put() — see _finalize_ttfat()'s docstring.
        self._feed_call_times: list[float] = []

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
        # Diagnostic for the "model produces zero speech at all" investigation
        # (rag_trigger_count=0, response='' turns) — applies regardless of
        # retrieval_backend, since a no-retrieval control run needs it too.
        self._patch_audio_power_diagnostics()

        # One-time-per-turn log of what the "pad-like" token ids actually
        # mean, so pad_token_ids_seen (below) is interpretable rather than
        # just "0 vs 3" with no context — e.g. distinguishing genuine
        # padding from an end-of-word marker that (if it turns out to be
        # one of these ids) would only appear once the model's text stream
        # has some real internal structure, not during a truly degenerate/
        # stuck generation.
        try:
            lm_model = base_job.server.runner.lm_gen.lm_model
            logger.info(
                "[TokenIDs] text_padding_token_id=%s end_of_text_padding_id=%s "
                "text_initial_token_id=%s rag_token_id=%s",
                lm_model.text_padding_token_id,
                lm_model.end_of_text_padding_id,
                lm_model.text_initial_token_id,
                lm_model.rag_token_id,
            )
        except Exception:
            logger.exception("[TokenIDs] failed to read token id meanings")

        if retrieval_backend is not None:
            self._patch_rag_manager()
            self._patch_output_loop()

    def _patch_audio_power_diagnostics(self):
        """
        Diagnostic for turns where rag_trigger_count=0 and response='' —
        the model never predicts <ret> and never produces a single real
        token, even though STT (fed the *unfiltered* chunk directly in
        _feed_loop, one line before audio_processor.filter_by_power() is
        called on the copy that actually reaches the LM) transcribes the
        question correctly. Testing whether filter_by_power's power gate
        is occasionally zeroing an entire utterance before the LM ever sees
        it — which would explain correct STT + total LM silence together,
        and would be consistent with non-determinism (gate result depends
        on each specific recording's loudness, not question content).
        """
        job = self._job
        original_filter = job.audio_processor.filter_by_power
        state = {"frames": 0, "zeroed": 0}

        def _wrapped_filter_by_power(chunk):
            result = original_filter(chunk)
            rms_in = chunk.detach().float().pow(2).mean().sqrt().item()
            rms_out = result.detach().float().pow(2).mean().sqrt().item()
            state["frames"] += 1
            if rms_out <= 1e-8 and rms_in > 1e-6:
                state["zeroed"] += 1
                logger.info(
                    "[AudioPower] frame=%d GATED TO ZERO rms_in=%.6f rms_out=%.6f",
                    state["frames"], rms_in, rms_out,
                )
            else:
                logger.debug(
                    "[AudioPower] frame=%d rms_in=%.6f rms_out=%.6f",
                    state["frames"], rms_in, rms_out,
                )
            return result

        job.audio_processor.filter_by_power = _wrapped_filter_by_power
        self._audio_power_state = state

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
            if self.retrieval_trigger_ts is not None:
                self.asr_wait_s = max(0.0, t0 - self.retrieval_trigger_ts)
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

    def _patch_output_loop(self):
        """
        Replace InferenceJob._output_loop with a version matching Channel's
        (moshi-rag's real production WebSocket server class,
        inference_utils/channel.py) simpler, confirmed-working
        retrieval-trigger handling — removes InferenceJob's own "pause
        output during retrieval" mechanism entirely, rather than continuing
        to patch around its bugs one at a time.

        Root-caused by reading Channel directly, prompted by a user's real
        test of the actual unmodified prod path (moshi.server, which uses
        Channel) showing no unnatural pauses on retrieval triggers: Channel
        is a SEPARATE, PARALLEL implementation from InferenceJob within
        moshi-rag's own codebase (live WebSocket server vs. offline/batch
        inference), and it has NO self._doing_retrieval gate, no
        _retrieval_start_step, no _retrieval_done_step at all. When the
        model emits <ret>, Channel._output_loop calls
        rag_manager.trigger(...) and immediately loops back to `await
        self.output_queue.get()` for the next token — no pausing.
        RAGManager.trigger() (shared by both classes, confirmed against
        rag_manager.py's real source) is already fully async/non-blocking
        on its own — task_group.create_task(...), returns immediately; the
        actual retrieval and conditioning update happen in a background
        task, entirely independent of the output loop. So InferenceJob's
        _doing_retrieval gate was never load-bearing for "let retrieval
        happen in the background" — that already worked via RAGManager
        alone, unconditionally. The gate's only unique effect was pausing
        the output loop's own token-forwarding while waiting, via an exact
        `step_index == self._retrieval_start_step` equality check that
        doesn't reliably fire — a real, confirmed bug in this specific
        InferenceJob mechanism (previously patched around here and by
        _step_watchdog's self-healing clear), not something moshi-rag's
        own live server exercises at all, and not something we introduced.

        Replaces _output_loop wholesale (matching the existing
        _live_feed_loop/_feed_loop pattern) rather than hooking a smaller
        seam, since the gate is woven through the method: the top-of-loop
        wait, the step_index target computed at the <ret> branch, and the
        step-index-triggered deferred handle_reference_fn call at the
        bottom. Also matches Channel's IMMEDIATE conditioning application
        (handle_reference_fn called directly from RAGManager's own
        background task, no step-based deferral) rather than
        InferenceJob's _retrieval_done_step deferral — that deferral was
        never itself buggy (it compares with >=, not exact equality), but
        a hybrid (Channel's gate removal plus InferenceJob's deferral)
        would be a novel, untested combination neither class actually
        runs; matching Channel exactly is the configuration with real
        evidence behind it.

        Supersedes the former _patch_catch_reference_text (removed) —
        that patched InferenceJob._catch_reference_text, which this no
        longer calls at all (handle_reference_fn is replaced outright, not
        layered on top of it).
        """
        job = self._job

        async def _immediate_handle_reference_text(reference_text, lm_label: str = "") -> None:
            # Channel's _handle_reference_text does both of these in one
            # step (storing for its UI history and applying conditioning
            # immediately) — we do the same, combining what
            # _catch_reference_text (trace bookkeeping, for our own
            # metadata["retrieval_text"]) and _handle_reference_text
            # (the actual conditioning update) used to split across a
            # step-index-deferred handoff.
            job.trace["reference_text"] = reference_text or ""
            t0 = time.perf_counter()
            # _fetch_and_apply_reference_conditioning() replaces moshi-rag's
            # own job._async_update_reference() call that used to be here —
            # see that function's docstring for why (event-loop starvation
            # by moshi-rag's own real-time step loop, not GPU contention).
            await _fetch_and_apply_reference_conditioning(
                reference_text or "",
                encoder_url=job.server.reference_encoder_url,
                lm_gen=job.server.runner.lm_gen,
                batch_size=job.server.batch_size,
                slot_idx=job.slot_idx,
            )
            job.trace["conditioning_step"] = job.step_index
            self.conditioning_latency_s = time.perf_counter() - t0

        rag_trigger_count = 0
        # Diagnostic: _decode_text_token() treats any of {0,1,2,3} as "pad"
        # and collapses them all to the same "<pad>" display string — this
        # records which raw ids actually showed up, so a silent turn can be
        # told apart between "genuinely the single pad token, repeated" vs.
        # "some other/varying special token being emitted", which would
        # point at a different bug than plain non-engagement.
        pad_token_ids_seen: set[int] = set()
        # Raw ordered sequence (not just the set above) — the set alone
        # can't distinguish "degenerate, stuck repeating a single id from
        # step 0" from "the usual mix, just never transitioning to real
        # speech": e.g. [3,3,3,3,3,...] throughout vs [3,3,0,3,0,0,3,...].
        # Capped so a long, fully-silent tail-timeout turn doesn't blow up
        # the log line.
        pad_token_sequence: list[int] = []
        _PAD_SEQUENCE_CAP = 40

        async def _patched_output_loop() -> None:
            nonlocal rag_trigger_count
            assert job._task_group is not None
            while not job._shutdown_event.is_set():
                if job.stop_on_end_of_input and job._feed_finished.is_set():
                    try:
                        out = await asyncio.wait_for(job.output_queue.get(), timeout=1.0)
                    except TimeoutError:
                        await job._finalize()
                        return
                else:
                    out = await job.output_queue.get()

                if out.pcm is not None:
                    job._model_pcm_chunks.append(out.pcm.detach().cpu().float().numpy().reshape(-1))

                text_token = out.text_token

                if text_token == job.server.runner.lm_gen.lm_model.rag_token_id:
                    rag_trigger_count += 1
                    self.retrieval_trigger_ts = time.perf_counter()
                    job.trace["rag_trigger_step"] = job.step_index
                    job.trace["rag_trigger_count"] = rag_trigger_count
                    job.model_text.append(job.server.text_tokenizer.id_to_piece(text_token))  # type: ignore[arg-type]
                    # Channel logs this explicitly (channel.py: "[RAG] model
                    # emitted RAG token, triggering reference generation") —
                    # our patch didn't carry that over, so there was no direct
                    # way to tell how many times <ret> fired per turn versus
                    # just "at least once" (trace["rag_trigger_step"] gets
                    # silently overwritten on every occurrence).
                    logger.info(
                        "[RAG] model emitted RAG token (occurrence #%d this turn) "
                        "at step_index=%d, triggering reference generation",
                        rag_trigger_count, job.step_index,
                    )
                    try:
                        await job.rag_manager.trigger(
                            task_group=job._task_group,
                            wait_steps=int(job.turn_manager.stt_wait_steps),
                            handle_reference_fn=_immediate_handle_reference_text,
                            context_provider=job.turn_manager.get_context,
                        )
                    except Exception:
                        # rag_manager.trigger() itself should only ever raise
                        # RuntimeError if called outside its `async with`
                        # scope — everything past that point (the actual
                        # retrieval fetch) runs in a background task that
                        # already catches its own exceptions (see
                        # RAGManager._background_task). If *this* call raises,
                        # it's happening somewhere we don't expect; log the
                        # full traceback rather than let it propagate
                        # silently or crash the whole output loop.
                        logger.exception(
                            "job.rag_manager.trigger() raised — this should "
                            "be near-impossible per rag_manager.py's own "
                            "source; investigate directly rather than assume"
                        )
                        raise
                else:
                    decoded = job._decode_text_token(text_token)
                    job.turn_manager.handle_spoken_text(model_text=decoded)
                    if decoded is None:
                        pad_token_ids_seen.add(text_token)
                        job.trace["pad_token_ids_seen"] = sorted(pad_token_ids_seen)
                        if len(pad_token_sequence) < _PAD_SEQUENCE_CAP:
                            pad_token_sequence.append(text_token)
                            job.trace["pad_token_sequence"] = list(pad_token_sequence)
                        job.model_text.append("<pad>")
                    else:
                        job.model_text.append(job.server.text_tokenizer.id_to_piece(text_token))  # type: ignore[arg-type]
                        # First non-pad text token = first audio token, per
                        # _finalize_ttfat()'s docstring: text and audio are
                        # generated in lockstep, and this is a robust proxy
                        # where raw PCM amplitude thresholding was not — a
                        # real VM run showed pcm.abs().max() firing at
                        # step_index=0 with an *identical* value across
                        # every different question/response (0.01241324,
                        # to 8 decimal places), confirming it was a fixed
                        # decoder warm-up artifact, not content-dependent
                        # speech onset.
                        if not self._first_audio_recorded:
                            self.t_first_audio = time.time()
                            self._first_audio_recorded = True

                if job._user_id_buffer:
                    uid = job._user_id_buffer.popleft()
                    job.user_text.append(job.stt.text_tokenizer.id_to_piece(uid))  # type: ignore[arg-type]
                else:
                    job.user_text.append("<pad>")

                job.rag_manager.step()

                async with job._pcm_one_step_cv:
                    job.step_index += 1
                    job._pcm_one_step_cv.notify_all()

                if job.trace.get("question_end_step", -1) >= 0 and job.max_tail_silence is not None:
                    if job._check_tail_silence():
                        await job._finalize()
                        return

        job._output_loop = _patched_output_loop

    async def run(self, task_group):
        """Delegate to the underlying InferenceJob, hooking timing."""
        # Records a (feed_step -> wall-clock time) mapping for every
        # _feed_loop() call to input_queue.put() — see _finalize_ttfat()'s
        # docstring for why this replaces a previous, broken approach.
        # Wraps a single method, not _feed_loop's own logic — its behavior
        # is unchanged, only observed.
        original_put = self._job.input_queue.put

        async def _timed_put(step_input):
            self._feed_call_times.append(time.time())
            return await original_put(step_input)

        self._job.input_queue.put = _timed_put

        await self._job.run(task_group)
        self._finalize_ttfat()

    def _finalize_ttfat(self) -> None:
        """
        Computes ttfat_s once the job has fully finished, using
        trace["question_end_step"] (set by moshi-rag's own, unpatched
        _feed_loop() to the index of the last REAL — not padding — input
        frame) correlated against _feed_call_times (recorded in run()
        above).

        Replaces a previous, broken approach: _t_question_end used to be
        set to wall-clock time right after _feed_loop() *returned* — but
        confirmed against real inference_job.py source, under
        stop_on_end_of_input=False (respond()'s own deliberate setting —
        see _respond_once), _feed_loop() does NOT return when the user's
        real audio ends; it falls into an unconditional while loop feeding
        zero-padding silence until the whole turn's _shutdown_event fires
        (i.e. near end-of-turn, not end-of-question). Since first real
        audio output always precedes end-of-turn, the old computation had
        t_first_audio < t_question_end by construction, clamping ttfat_s to
        exactly 0.0 on every single call — confirmed on a real VM run, not
        a rounding artifact.

        _feed_loop's own feed_step counter (source of question_end_step)
        increments exactly once per input_queue.put() call, in both its
        real-input loop and its trailing-silence loop — so
        _feed_call_times[question_end_step] is the precise wall-clock
        moment the user's real question finished being fed, independent of
        whatever _feed_loop does afterward.

        t_first_audio itself comes from _patch_output_loop()'s first
        non-pad text token, not PCM amplitude — an earlier version watched
        output_queue for the first pcm.abs().max() > 1e-6 chunk, but a real
        VM run showed that firing at step_index=0 with an *identical*
        amplitude across every different question/response, confirming a
        fixed decoder warm-up artifact rather than genuine speech onset.
        """
        question_end_step = self._job.trace.get("question_end_step")
        if question_end_step is not None and 0 <= question_end_step < len(self._feed_call_times):
            self._t_question_end = self._feed_call_times[question_end_step]
        if self._first_audio_recorded and self._t_question_end is not None:
            self.ttfat_s = max(0.0, self.t_first_audio - self._t_question_end)

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

    UPDATE: _TimedInferenceJob._patch_output_loop() removed the
    self._doing_retrieval gate this watchdog's intervention branch below was
    built to unstick — root-caused instead of patched around further, see
    that method's docstring. raw_job._doing_retrieval should now be
    permanently False (nothing sets it True anymore), so the `if not
    advancing and raw_job._doing_retrieval:` branch is expected to be dead
    code — kept as-is for now, deliberately not removed, as a defensive
    safety net until the fix has been validated against a real checkpoint
    across enough retrieval triggers to be confident no *other* stall
    mechanism exists. Revisit once confirmed — the general step_index
    staleness logging above is still useful independent of this.

    Original rationale, kept for context: also a safety net for a confirmed
    race: the former _patch_catch_reference_text's own fix (clearing
    _doing_retrieval right after handle_reference_fn completes) only worked
    if _output_loop's step loop has already set _doing_retrieval =
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
      - MANDATORY separate-process conditioner. respond() requires a real,
        separately launched `python -m moshi.server_conditioner` process,
        pointed at via the REFERENCE_ENCODER_URL env var — raises at
        construction time if unset. There is no in-process ARC-Encoder
        loading path anymore.

        Backstory / why this is mandatory rather than opt-in: respond() used
        to have a confirmed bug — the first call on a MoshiRAGAdapter
        instance produced a correct answer, but every subsequent call went
        completely silent (zero non-<pad> tokens, no <ret> ever predicted)
        whenever retrieval was enabled. Three targeted hypotheses were ruled
        out with real diagnostic evidence first: stale streaming_sum state
        surviving into the next job (BatchRunner's own is_first-triggered
        reset was confirmed correctly clearing it), a cross-thread CUDA sync
        gap between the conditioning update and the next generation step (an
        explicit torch.cuda.synchronize() there made no difference), and
        per-call event-loop/step-task teardown (switching respond() to a
        persistent loop, same as run_inference.py's own structure, made no
        difference either). Pointing conditioning at a real separate-process
        conditioner instead measurably fixed it in two independent
        real-checkpoint tests. This was originally kept opt-in pending a
        "permanent fix" and a scoped spec update — see
        specs/moshirag-evals-requirements.md's "Architecture: Target
        moshi-rag's Production Server" section and CLAUDE.md's Phase 0/1
        findings: the broader pivot to targeting
        moshi-rag's real production architecture (which has never used
        in-process conditioning for *any* path, not just respond()) makes
        "always separate-process" the actual permanent fix rather than a
        workaround pending one. respond_stream() (the demo) was separately
        confirmed to never have had this bug even when in-process
        conditioning still existed as an option — moot now since the demo no
        longer runs through MoshiRAGAdapter at all (see CLAUDE.md).

    Retrieval:
      - If retrieval_backend is NullBackend or None: moshi-rag's RAG path is
        still active (predicts <ret>) but produces no reference text.
      - If retrieval_backend is GeminiAPIBackend: our backend intercepts the
        trigger and routes retrieval through Gemini, so latency is measured
        by retrieval_breakdown eval hooks on that backend instance.

    GPU device assignment:
      - Always claims physical GPU 0 (core/gpu.py's
        ensure_cuda_visible_devices("frontend")), resolved and applied to
        this process's own CUDA_VISIBLE_DEVICES before _load_models()
        imports torch. On the current single-A100 VM this is also where
        the conditioner runs, which is a confirmed source of GPU
        contention (see specs/moshirag-evals-requirements.md's "GPU Sizing
        and Multi-GPU Deployment" section) -- self.gpu_devices.contended
        is True in that case, and a warning is logged. Not carried further
        into respond()'s own metadata dict; read self.gpu_devices directly
        (evals/runner.py does, to tag conditioner_contended into the
        result JSON).

    TTFAT is computed as wall-clock delta from user audio end → first non-silent
    audio token emitted. In batch offline inference this is driven by the model's
    processing latency (~1 inference step at 12.5Hz ≈ 80ms minimum).
    """

    def __init__(
        self,
        checkpoint_path: str,
        retrieval_backend: RetrievalBackend | None = None,
        generation: dict | None = None,
    ):
        self.checkpoint_path = checkpoint_path
        self.retrieval_backend = retrieval_backend or NullBackend()
        # Any field omitted from `generation` falls back to the same
        # default that was previously hardcoded — see _DEFAULT_GENERATION.
        self._generation = {**_DEFAULT_GENERATION, **(generation or {})}
        # Must resolve before _load_models() imports torch -- CUDA_VISIBLE_DEVICES
        # has no effect on a process once its CUDA runtime has initialized.
        # See core/gpu.py and specs/moshirag-evals-requirements.md's "GPU
        # Sizing and Multi-GPU Deployment" section.
        self.gpu_devices = ensure_cuda_visible_devices("frontend")
        if self.gpu_devices.contended:
            logger.warning(
                "single-GPU mode: known conditioner contention (see specs/"
                "moshirag-evals-requirements.md's \"GPU Sizing and Multi-GPU "
                "Deployment\" section) -- do not trust retrieval-latency or "
                "grounding-dependent scores from this run"
            )
        self._load_models()
        # See _ensure_step_loop() — lazily created on first respond() call,
        # then kept alive for this adapter's whole lifetime.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._step_task: asyncio.Task | None = None
        # See respond()'s gap-timing log: wall-clock end of the previous
        # respond() call, used to measure the fully-synchronous gap (judge
        # call + next-row data loading, both blocking, run while self._loop
        # is not being driven at all) between calls — the current lead on
        # why some turns come back with rag_trigger_count=0 and response=''.
        self._last_respond_end_ts: float | None = None

    def _ensure_step_loop(self) -> None:
        """
        Start (once) a persistent event loop + step_task for respond(), and
        keep both alive across every subsequent respond() call on this
        adapter instance — matching run_inference.py's own structure, where
        the step loop is created once outside the per-job loop and stays
        running continuously across every wav file in the batch via
        asyncio.gather. respond() previously called asyncio.run(_run())
        fresh per question: a brand new event loop and a brand new
        state._step_loop() task, created and torn down every single time —
        a real, confirmed structural difference from the reference
        implementation. Investigating a reproduced bug (with retrieval
        enabled, only the first respond() call on an adapter instance ever
        produces a response — every subsequent call goes completely silent,
        zero non-<pad> tokens, no <ret> ever predicted) after two other
        targeted hypotheses were each ruled out with real diagnostic
        evidence: stale RAG conditioning surviving into the next job (ruled
        out — BatchRunner's own is_first-triggered reset was confirmed
        correctly clearing pending_streaming_sums before the second call);
        and a cross-thread CUDA ordering gap between the conditioning
        update and the next generation step (ruled out — an explicit
        torch.cuda.synchronize() there made no difference). This addresses
        the next remaining structural difference instead of another guess
        at CUDA timing.
        """
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        _maybe_enable_eval_asyncio_debug(self._loop)

        async def _start_step_loop() -> asyncio.Task:
            return asyncio.create_task(self._state._step_loop(), name="step-loop")

        self._step_task = self._loop.run_until_complete(_start_step_loop())

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

        # Build the args namespace that load_models() expects. Every
        # generation-affecting field below comes from self._generation (see
        # _DEFAULT_GENERATION) — device/batch_size/init_active_speaker stay
        # hardcoded, not config-driven (see that constant's own comment for
        # why those three are path-specific rather than shared).
        args = argparse.Namespace(
            # Always "cuda:0" for the MAIN model, deliberately not derived
            # from gpu_devices (__init__ already resolved and applied the
            # real physical GPU choice to this process's own
            # CUDA_VISIBLE_DEVICES before this method ever ran) -- whichever
            # physical GPU the front-end was assigned is always the first
            # one visible to this process, so it's always addressed as index
            # 0 from here. (Before 2026-07-30 a second physical GPU could
            # ALSO be visible, for the deleted Option E chain's STT pinning
            # -- see core/gpu.py's resolve_devices() for that history.)
            device="cuda:0",
            cfg_coef=self._generation["cfg_coef"],
            batch_size=1,
            hf_repo=None,
            moshi_weight=str(ckpt_paths["moshi_weight"]),
            mimi_weight=str(ckpt_paths["mimi_weight"]),
            tokenizer=str(ckpt_paths["tokenizer"]),
            config=str(ckpt_paths["config"]),
            dtype=torch.bfloat16,
            init_active_speaker="user",
            stt_wait_time=self._generation["stt_wait_time"],
            rag_timeout=self._generation["rag_timeout"],
            max_reference_tokens=self._generation["max_reference_tokens"],
            vad_window_size=self._generation["vad_window_size"],
            vad_threshold=self._generation["vad_threshold"],
            power_threshold=self._generation["power_threshold"],
        )

        # Must run before load_models() -- see
        # _patch_load_models_generation_overrides()'s own docstring for why
        # this can't be threaded through `args` like every other field
        # above (moshi.server has no --temp-text/--top-k-text CLI flag for
        # print_demo_env.py's translation to mirror, so the demo path reads
        # these two directly out of config instead — see
        # scripts/instrumented_server.py's apply_patches()).
        _patch_load_models_generation_overrides(
            temp_text=self._generation["temp_text"],
            top_k_text=self._generation["top_k_text"],
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
        # STT runs synchronously on the calling thread, exactly as upstream
        # moshi-rag wrote it. The "Option E" chain that used to run here
        # (off-thread STT via asyncio.to_thread, plus four patches guarding
        # the CUDA-graph hazards that introduced) was deleted 2026-07-30 —
        # see docs/demo-turn-onset-regression.md. It existed to mitigate an
        # audio backlog in Channel._recv_loop that turned out to be an
        # artifact of this repo's own step-pacing patch, not of synchronous
        # STT: with pacing off, measured per-step work is 33ms against an
        # 80ms budget and a live session held input-queue lag at 1 frame for
        # 58s straight with synchronous STT. git history has the chain if a
        # genuine need ever reappears.
        self._stt_template = LocalSpeechToText(deepcopy(self._mimi))

        # See the class docstring's "Conditioning" section — mandatory, not
        # an opt-in workaround. Confirmed root cause of the respond()-only
        # silent-response bug: in-process ARC-Encoder loading (the former
        # default) broke every respond() call after the first one whenever
        # retrieval was enabled. This also matches the real production
        # architecture — moshi.server itself always calls a separate
        # conditioner process; in-process conditioning was never how
        # upstream does this for any path.
        reference_encoder_url = os.environ.get("REFERENCE_ENCODER_URL")
        if not reference_encoder_url:
            raise RuntimeError(
                "REFERENCE_ENCODER_URL must be set. MoshiRAGAdapter.respond() "
                "requires a real, separate `python -m moshi.server_conditioner` "
                "process — matching the demo's architecture (see "
                "scripts/run_demo.sh) and the real moshi-rag production "
                "server, which never runs the ARC-Encoder in-process either. "
                "In-process loading was removed after being confirmed as the "
                "root cause of a silent-response bug on every respond() call "
                "after the first one (see this class's docstring). Start "
                "server_conditioner and set REFERENCE_ENCODER_URL to its "
                "address (e.g. http://localhost:8001) before running evals."
            )
        logger.info("Using separate-process conditioner at %s", reference_encoder_url)

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
        logger.info("MoshiRAG ready")

    def transcribe(self, audio: bytes) -> str:
        """Transcribe audio bytes via the streaming ASR component."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio)
            wav_path = Path(f.name)
        try:
            stt = deepcopy(self._stt_template)
            _warm_up_stt_exec_mask(stt)
            return asyncio.run(stt.transcribe_file(wav_path))
        finally:
            wav_path.unlink(missing_ok=True)

    def respond(self, audio_in: bytes) -> tuple[bytes, str, dict]:
        """
        Run one inference turn, with a single automatic retry and explicit
        flagging on "degenerate silence" (empty response, <ret> never
        predicted — see _respond_once).

        Background: two specific recordings in OpenAudioBench (WebQ/LlamaQ)
        reliably reproduce this on every call — confirmed via a dedicated
        diagnostic (see CLAUDE.md) that ruled out sample rate/resampling and
        loudness/gain as causes across controlled real-checkpoint runs; the
        model itself enters a genuine zero-engagement state for the whole
        turn (never a single non-pad token), not a pipeline artifact we can
        normalize away. Root cause (specific to those recordings' acoustic
        content) is still open. Retrying does NOT fix those two — they're
        deterministic — but costs one extra ~5s call and is worth keeping as
        a safety net for less-deterministic occurrences in datasets we
        haven't hit this on yet. What actually matters for scoring
        integrity is the flag: without it, a degenerate-silence turn is
        indistinguishable in results from "model genuinely got the question
        wrong," silently deflating accuracy on whatever subset happens to
        contain more of these recordings. metadata["degenerate_silence"]
        marks the case explicitly so callers (evals, transcripts) can
        surface it instead of scoring it blind.
        """
        audio_out, inner_text, metadata = self._respond_once(audio_in)
        degenerate = not inner_text.strip() and metadata["rag_trigger_count"] == 0
        metadata["degenerate_silence_retried"] = degenerate
        if degenerate:
            logger.warning("[respond] degenerate silence (empty response, no <ret>) — retrying once")
            audio_out, inner_text, metadata = self._respond_once(audio_in)
            degenerate = not inner_text.strip() and metadata["rag_trigger_count"] == 0
            metadata["degenerate_silence_retried"] = True
            if degenerate:
                logger.warning("[respond] degenerate silence persisted after retry")
        metadata["degenerate_silence"] = degenerate
        return audio_out, inner_text, metadata

    def _respond_once(self, audio_in: bytes) -> tuple[bytes, str, dict]:
        """
        Run one inference turn.

        Writes audio_in to a temp WAV, runs InferenceJob to completion,
        returns (audio_out_wav, inner_monologue_text, timing_metadata).
        """
        import numpy as np
        from moshi.inference_utils.inference_job import InferenceJob

        # Gap-timing diagnostic: the synchronous work between calls (the
        # judge's blocking API call, next-row/next-subset data loading) all
        # happens while self._loop is not being driven at all — testing
        # whether the two known "rag_trigger_count=0, response=''" turns
        # correlate with an unusually long gap here versus the 18 calls
        # that responded normally.
        call_start_ts = time.monotonic()
        if self._last_respond_end_ts is not None:
            logger.info(
                "[respond] gap since previous respond() call finished: %.3fs",
                call_start_ts - self._last_respond_end_ts,
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin:
            fin.write(audio_in)
            in_path = Path(fin.name)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fout:
            out_path = Path(fout.name)

        self._ensure_step_loop()

        try:
            stt = deepcopy(self._stt_template)
            _warm_up_stt_exec_mask(stt)
            state = self._state
            retrieval_backend = self.retrieval_backend

            async def _run() -> _TimedInferenceJob:
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
                    max_tail_silence=self._generation["tail_silence_steps"],
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
                return timed

            # Runs on the persistent loop from _ensure_step_loop() — the
            # step_task started there keeps running continuously across
            # every respond() call, not recreated per question.
            job = self._loop.run_until_complete(_run())

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
                # Real ARC-encoder /embed round trip — see
                # _TimedInferenceJob.conditioning_latency_s's docstring.
                # This is what actually gates when the model's spoken
                # response reflects the retrieved reference, distinct from
                # (and typically much larger than) retrieval_latency_s.
                "conditioning_latency_s": job.conditioning_latency_s,
                # Time between the <ret> trigger and get_reference_text()
                # actually being called — see _TimedInferenceJob.__init__'s
                # retrieval_trigger_ts/asr_wait_s docstring. 0.0 (the
                # __init__ default) when no <ret> ever fired this turn.
                "asr_wait_s": job.asr_wait_s,
                # BUG FIX: "rag_trigger_step" in trace was always True — moshi's
                # own InferenceJob.__init__ pre-populates trace with
                # {"rag_trigger_step": -1, ...} as a sentinel default, before
                # anything happens, so the key's mere presence never meant
                # <ret> actually fired. Confirmed on a real run: rag_triggered
                # showed True even when rag_trigger_count (below) was 0 and no
                # "[RAG] model emitted..."/"[Reference]" logging appeared at
                # all for that turn. rag_trigger_count is the accurate signal;
                # derive this flag from it instead of the sentinel-populated key.
                "rag_triggered": trace.get("rag_trigger_count", 0) > 0,
                "rag_trigger_step": trace.get("rag_trigger_step"),
                # How many times <ret> fired this turn, not just whether it
                # fired at least once — rag_trigger_step alone gets silently
                # overwritten on every occurrence, so "1" and "5" looked
                # identical before this.
                "rag_trigger_count": trace.get("rag_trigger_count", 0),
                "question_end_step": trace.get("question_end_step"),
                # See _patch_audio_power_diagnostics()/pad_token_ids_seen above.
                "audio_zeroed_frames": job._audio_power_state["zeroed"],
                "audio_total_frames": job._audio_power_state["frames"],
                "pad_token_ids_seen": trace.get("pad_token_ids_seen", []),
                "pad_token_sequence": trace.get("pad_token_sequence", []),
            }

            # One clear, greppable line per question — everything else about
            # whether <ret> fired is buried in dozens of moshi-internal log
            # lines ("[Reference] Triggering retrieval...", "[Buffer]
            # buffering model text...", etc). Search logs for "[respond]".
            logger.info(
                "[respond] rag_triggered=%s rag_trigger_count=%d ttfat=%.3fs "
                "audio_zeroed=%d/%d pad_ids=%s pad_seq=%s response=%r",
                metadata["rag_triggered"], metadata["rag_trigger_count"], metadata["ttfat_s"],
                metadata["audio_zeroed_frames"], metadata["audio_total_frames"],
                metadata["pad_token_ids_seen"], metadata["pad_token_sequence"], inner_text[:200],
            )

            return audio_out, inner_text, metadata

        finally:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)
            out_path.with_suffix(".wav").unlink(missing_ok=True)
            self._last_respond_end_ts = time.monotonic()

