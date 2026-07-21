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
            await job._async_update_reference(reference_text or "")

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

    TTFAT is computed as wall-clock delta from user audio end → first non-silent
    audio token emitted. In batch offline inference this is driven by the model's
    processing latency (~1 inference step at 12.5Hz ≈ 80ms minimum).
    """

    def __init__(self, checkpoint_path: str, retrieval_backend: RetrievalBackend | None = None):
        self.checkpoint_path = checkpoint_path
        self.retrieval_backend = retrieval_backend or NullBackend()
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
            # Was 2.0 — raised to match scripts/run_demo.sh's own fix: real
            # Gemini round-trip latency was directly observed at ~2.5-3s
            # (see CLAUDE.md's Phase 0 prodcheck findings), reliably longer
            # than a 2.0s budget. Same underlying defect, same fix, applied
            # here too rather than leaving evals exposed to it.
            rag_timeout=8.0,
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

