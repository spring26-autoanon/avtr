"""Instrumentation wrapper around moshi-rag's real `moshi.server` — see
specs/moshirag-evals-requirements.md's "Session instrumentation" section.

Forwards every CLI arg through to moshi.server.main() unchanged (that
function reads sys.argv itself via argparse; this script never touches
sys.argv). Before calling it, patches a handful of moshi-rag classes to
observe/log per-turn instrumentation and — one deliberate, bounded
exception to "logging only", see _patch_rag_manager_get_reference_text
below — to actually redirect retrieval through the same
core/retrieval_backend.py RetrievalBackend evals use. None of this touches
Channel.run()/_output_loop()/_recv_loop() or RAGManager.trigger()'s control
flow. Patches, in order of "how invasive":

  - ServerState.__init__            leaf wrap: capture rag_timeout/stt_mode,
                                     write the session_start record once
  - Channel.__init__                leaf wrap: per-connection state, plus
                                     installs the per-RAGManager-instance
                                     get_reference_text patch below, right
                                     where self.rag_manager becomes available
  - RAGManager.get_reference_text   per-instance replacement (NOT a wrap):
                                     routes real retrieval through the same
                                     RetrievalBackend evals use, bypassing
                                     moshi's native LLMReferenceGenerator
                                     entirely — see
                                     _patch_rag_manager_get_reference_text's
                                     own docstring for the full rationale
  - RAGManager.trigger              leaf wrap: detect <ret> -> rag_triggered
                                     (every real call site — Channel's own
                                     output loop and InferenceJob/respond()'s
                                     mirrored logic — only calls this when
                                     the model has just predicted the RAG
                                     token, so this is the correct signal;
                                     see that patch's own docstring for why
                                     _decode_text_token was tried first and
                                     is wrong)
  - TurnManager._update_active_speaker  leaf wrap: detect end-of-user-utterance
                                     (this is the ttfat clock start and the
                                     turn boundary)
  - TurnManager.handle_spoken_text  leaf wrap: also the ttfat clock *stop* —
                                     _ChannelState.on_model_text() (called
                                     here whenever model_text is not None,
                                     i.e. a genuine non-pad token) triggers
                                     on_first_audio(). An earlier version
                                     wrapped ServerState._deliver_step_row
                                     instead, firing ttfat_s off "pcm_out is
                                     not None" — the real batch-path
                                     equivalent of that check was confirmed,
                                     via a real VM run's diagnostic logging,
                                     to fire on step_index=0 with an
                                     amplitude *identical* across every
                                     different question/response (a fixed
                                     decoder artifact, not genuine speech
                                     onset — see core/model_interface.py's
                                     _finalize_ttfat docstring). Not directly
                                     re-confirmed against a live demo session
                                     as of this writing (unlike the batch
                                     path, which was), but the same
                                     underlying moshi-rag step mechanics
                                     apply to both, and this reuses a signal
                                     the demo already computes correctly for
                                     transcript purposes rather than
                                     introducing a new one
  - RAGManager._background_task     whole-method patch (same class of change
                                     as model_interface.py's _patch_output_loop):
                                     needs internal timestamps (asr_wait_s
                                     boundary, then the get_reference_text
                                     call) that aren't observable by wrapping
                                     the method from outside, so the method
                                     body below is a verbatim copy of the
                                     real one with three time.perf_counter()
                                     calls and one session callback added.
                                     context_injection_s now comes from the
                                     get_reference_text patch's own timed,
                                     redundant format_context() call (see
                                     that patch's docstring) rather than a
                                     separate LLMReferenceGenerator.process_reference_text
                                     patch — that patch is removed entirely,
                                     since process_reference_text is never
                                     reached once get_reference_text bypasses
                                     LLMReferenceGenerator for real retrieval.

Each _ChannelState hook (on_ret_triggered/on_retrieval_complete/
on_first_audio) also pushes the newly-known fields to the browser as they
become available, via _push_instrumentation() — an additive "metadata"
WebSocket message (kind-byte 4), the same generic message type the client
already uses for retrieval-backend capability announcements. See that
function's docstring for the exact payload shape.

Verified against kyutai-labs/moshi-rag @ main (`git clone`d directly,
2026-07-22) — moshi/server.py,
moshi/inference_utils/{channel,rag_manager,turn_manager}.py,
moshi/reference/llm_reference_generator.py. Re-diff this file's patches
against upstream if any of those change; RAGManager._background_task in
particular will silently drift since it's a full-body copy, not a wrap.

RAGManager.trigger's signature (`trigger(self, task_group, wait_steps=0,
handle_reference_fn=None, context_provider=None)`, called from
channel.py's _output_loop with all four as keywords) is confirmed against
that clone. That same source read explains why a `<ret>` trigger's
turn_index tag (recorded in raw_events.jsonl) is only ever a *live,
best-effort* value, not a trustworthy final answer: `_output_loop` has two
real `await` points (`_send_turn_outputs`, `stt.flush()`) between detecting
the RAG token and actually calling `trigger()`, during which the
concurrently-running `_stt_recv_loop` task can legitimately advance the
turn boundary first — so the RAG token can fire while our own
on_utterance_end() hasn't yet caught up to the user having already
finished the *next* turn's question. Earlier rounds tried to solve this
live, in-process, with a captured-target-plus-late-update-record scheme —
abandoned in favor of a strictly better fix: since
scripts/summarize_demo_session.py always runs *after* the full session is
over, it has complete knowledge of every trigger and every turn boundary
that raw_events.jsonl recorded, and can therefore correctly reattribute
each retrieval post-hoc — something the live process fundamentally cannot
do, since at write time it doesn't yet know whether a boundary is about to
follow. See that script's `compute_retrievals_from_raw_events()` for the
actual reattribution rule (validated against a real VM session with a
same-turn double-retrieval, which is why the rule is "only the *last*
trigger before a turn boundary reassigns forward," not "always shift by
one").

e2ekd_s/keyword_delay_s are intentionally not computed here (always null in
the turn record) — see the spec section above for why: evals/registry/
latency/e2ekd.py doesn't exist yet, and its methodology (MoshiRAG paper
Table 17 keyword-extraction prompt + nvidia/parakeet-tdt-0.6b-v2 onset
timing) must not be approximated ahead of that eval proving it out.

Writes two files per session (see SessionLog's own docstring for why):
  turns.jsonl        one record per completed turn — question, response,
                      ttfat_s only; deliberately carries no retrieval
                      fields at all (see the note above: those can only be
                      correctly attributed post-hoc, by
                      summarize_demo_session.py, not live)
  raw_events.jsonl    one record per raw user/model text chunk and RAG
                      event (ret_triggered/retrieval_complete carry the
                      full retrieval result — context, reference text,
                      breakdown — not just latency), un-turn-scoped, for
                      diagnosing <ret> firing too early/late/unprompted —
                      see scripts/summarize_demo_session.py's "[raw
                      stream]" section

Required environment variables (set by scripts/run_demo.sh):
  DEMO_SESSION_DIR   directory to write turns.jsonl/raw_events.jsonl into
                      (must already exist)
  DEMO_CHECKPOINT    checkpoint alias/name, for the session_start record
  DEMO_CONFIG        path to the same YAML config shape evals/runner.py
                      uses — loaded directly in Python (unlike
                      LLM_BASE_URL/LLM_MODEL_NAME/generation CLI flags,
                      which scripts/print_demo_env.py translates into shell
                      env/CLI args ahead of time) to build the one
                      RetrievalBackend instance _patch_rag_manager_get_reference_text
                      routes retrieval through
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _git_hash() -> str:
    # Mirrors evals/runner.py's _git_hash() — duplicated rather than imported
    # across the scripts/ <-> evals/ boundary for a 12-line subprocess call.
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _push_instrumentation(channel, turn_index: int, fields: dict) -> None:
    """Fire-and-forget an additive "metadata" WebSocket message carrying live
    instrumentation fields for one turn — reuses the existing kind-byte-4 /
    client `"metadata"` message type (client/src/protocol/types.ts already
    has `{type: "metadata", data: unknown}`; today it's only used for
    retrieval-backend capability announcements). `"kind": "instrumentation"`
    distinguishes ours from that existing payload shape, since both travel
    over the same generic message type.

    Called from several sync monkeypatched methods (_update_active_speaker,
    handle_spoken_text) that can't `await` directly, so this always schedules
    via asyncio.create_task — even from the async call sites (RAGManager.trigger,
    _background_task), for one uniform fire-and-forget path that never
    blocks or slows the real conversation on an instrumentation send.
    """
    payload = {"kind": "instrumentation", "turn_index": turn_index, "fields": fields}
    _push_metadata(channel, payload)


def _push_session_info(channel, retrieval_backend: dict) -> None:
    """Same additive "metadata" mechanism as _push_instrumentation, but for
    the one session-level (not per-turn) field: which retrieval backend this
    session is using. Sent once, right after a channel connects — a separate
    "kind" (not "instrumentation") so the client keeps this in its own state
    slot rather than it being overwritten/dropped by later per-turn merges.
    """
    _push_metadata(channel, {"kind": "instrumentation_session", "retrieval_backend": retrieval_backend})


def _push_metadata(channel, payload: dict) -> None:
    async def _send():
        try:
            await channel.ws.send_bytes(b"\x04" + json.dumps(payload).encode("utf-8"))
        except Exception:
            logger.exception("[Instrumentation] failed to push metadata to client")

    asyncio.create_task(_send())


class _ChannelState:
    """Per-connection turn-tracking state, owned by one Channel instance.

    Deliberately holds *no* retrieval-related state at all (no captured
    "which turn asked for this" reference, no flushed-tracking) — see the
    module docstring's note on why that reattribution problem is solved
    entirely in scripts/summarize_demo_session.py, post-session, rather
    than here. on_ret_triggered()/on_retrieval_complete() just log the raw
    facts with whatever turn_index is live at that instant and push a
    best-effort live update to the browser; nothing here needs to be
    "correct" in the final-answer sense, since nothing downstream (turns.jsonl,
    the browser) depends on it being so — the browser only ever shows
    "most recently known retrieval state," and turns.jsonl doesn't carry
    retrieval fields at all.
    """

    def __init__(self, session: "SessionLog", channel) -> None:
        self._session = session
        self.channel = channel
        self.turn_index = 0
        self._incoming_user_text: list[str] = []
        self._pending: dict | None = None

    def on_user_text(self, text: str) -> None:
        self._session.write_raw_event("user_text", turn_index=self.turn_index, text=text)
        self._incoming_user_text.append(text)

    def on_model_text(self, text: str) -> None:
        self._session.write_raw_event("model_text", turn_index=self.turn_index, text=text)
        # This is also ttfat_s's first-audio-frame signal: on_model_text is
        # only ever called with a genuine, non-pad decoded token (see
        # _patch_turn_manager's patched_handle_spoken_text, which filters
        # out model_text=None before calling here) — the same real-vs-pad
        # distinction core/model_interface.py's _finalize_ttfat uses on the
        # batch path, for the same reason: a raw "is there a PCM chunk at
        # all" check fires on essentially the first step of a turn
        # regardless of whether genuine speech has started yet.
        self.on_first_audio()
        # Unlike user text, the model's response to turn N arrives while
        # self._pending already *is* turn N's record (on_utterance_end for N
        # creates it before the model starts responding, and it isn't
        # flushed until turn N+1's utterance end) — so this appends directly
        # into the pending record instead of a separate accumulation buffer.
        if self._pending is not None:
            self._pending["model_response_text"] += text

    def on_utterance_end(self) -> None:
        """Called right as VAD confirms the user stopped speaking and the
        pending switch to the model turn is set — this is both the ttfat
        clock start and the turn boundary: whatever was pending from the
        previous turn is now complete and gets flushed."""
        self._flush_pending()
        self.turn_index += 1
        self._session.write_raw_event("utterance_end", turn_index=self.turn_index)
        self._pending = {
            "type": "turn",
            "turn_index": self.turn_index,
            "timestamp": self._session.now_iso(),
            "user_question_text": "".join(self._incoming_user_text).strip(),
            "model_response_text": "",
            "ttfat_s": None,
            "e2ekd_s": None,
            "keyword_delay_s": None,
            "_ttfat_start": time.perf_counter(),
        }
        self._incoming_user_text = []

    def on_ret_triggered(self) -> None:
        # Logged unconditionally (even with no pending turn) — an
        # unprompted/out-of-turn trigger is exactly the kind of event this
        # raw stream exists to surface, not one to silently drop. turn_index
        # here is a live, best-effort tag only — see the module docstring
        # and _ChannelState's own docstring for why the *correct* turn
        # attribution is computed later, from this raw event, not here.
        self._session.write_raw_event("ret_triggered", turn_index=self.turn_index)
        _push_instrumentation(self.channel, self.turn_index, {"rag_triggered": True})

    def on_retrieval_complete(
        self,
        *,
        context: str,
        reference_text: str,
        asr_wait_s: float,
        context_injection_s: float,
        api_call_s: float,
    ) -> None:
        breakdown = {
            "asr_wait_s": round(asr_wait_s, 4),
            "api_call_s": round(api_call_s, 4),
            "context_injection_s": round(context_injection_s, 4),
            "total_s": round(asr_wait_s + api_call_s + context_injection_s, 4),
        }
        # Same live/best-effort turn_index tagging as on_ret_triggered — the
        # full result (context/reference/breakdown) is logged unconditionally
        # regardless of which turn_index it lands under, since
        # summarize_demo_session.py reattributes by matching this event back
        # to its triggering ret_triggered event, not by trusting this tag.
        self._session.write_raw_event(
            "retrieval_complete",
            turn_index=self.turn_index,
            retrieval_context=context,
            retrieved_reference_text=reference_text,
            retrieval_breakdown_s=breakdown,
        )
        _push_instrumentation(
            self.channel,
            self.turn_index,
            {"retrieval_context": context, "retrieved_reference_text": reference_text, "retrieval_breakdown_s": breakdown},
        )

    def on_first_audio(self) -> None:
        if self._pending is None or self._pending["ttfat_s"] is not None:
            return  # already recorded for this turn
        self._pending["ttfat_s"] = round(time.perf_counter() - self._pending["_ttfat_start"], 4)
        self._session.write_raw_event("first_audio", turn_index=self.turn_index, ttfat_s=self._pending["ttfat_s"])
        _push_instrumentation(self.channel, self.turn_index, {"ttfat_s": self._pending["ttfat_s"]})

    def flush_final(self) -> None:
        """Called on channel close so the last turn isn't lost."""
        self._flush_pending()

    def _flush_pending(self) -> None:
        if self._pending is None:
            return
        record = {k: v for k, v in self._pending.items() if not k.startswith("_")}
        record["model_response_text"] = record["model_response_text"].strip()
        self._session.write_turn(record)
        self._pending = None


class SessionLog:
    """Owns turns.jsonl (per-turn, aggregated — easy to scan) and
    raw_events.jsonl (per-token/per-event, un-turn-scoped) for one demo
    session — both append-only, one compact JSON value per line (see spec's
    "Demo session log" section for why this is deliberately not
    pretty-printed).

    raw_events.jsonl exists because turns.jsonl's turn boundaries are
    themselves derived from VAD (on_utterance_end) and hide exactly the
    thing worth diagnosing when the model fires <ret> too early, too late,
    or unprompted: the real, un-bucketed order user speech, model speech,
    and RAG events actually arrived in. Every event carries `t_rel_s`
    (seconds since this SessionLog was constructed, from time.perf_counter()
    — monotonic and millisecond-precision, unlike now_iso()'s
    second-resolution wall-clock timestamp) so ordering/overlap is
    unambiguous even for events within the same second.

    It's also the *sole* source of truth for retrieval data — turns.jsonl's
    `turn` records carry no rag_triggered/retrieval_context/etc fields at
    all. Earlier attempts to keep retrieval fields on the turn record
    (correctly written when live, patched with a separate late-arriving
    update record when a turn boundary raced ahead of a still-resolving
    retrieval) worked but needed real live-process bookkeeping to get
    right. Since scripts/summarize_demo_session.py only ever runs after a
    session ends, it has complete knowledge no live code path can — every
    ret_triggered/retrieval_complete/utterance_end event, in order — so it
    can correctly attribute every retrieval to its real turn with a simple
    post-hoc rule instead. See that script's
    compute_retrievals_from_raw_events() for the rule itself.
    """

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self._t0 = time.perf_counter()
        self._path = session_dir / "turns.jsonl"
        self._fh = open(self._path, "a")
        self._raw_path = session_dir / "raw_events.jsonl"
        self._raw_fh = open(self._raw_path, "a")
        self._channel_states: dict[int, _ChannelState] = {}
        # RAGManager/TurnManager instances are per-channel but don't hold a
        # back-reference to their owning Channel, so Channel.__init__ (below)
        # registers both under the same _ChannelState.
        self._by_rag_manager_id: dict[int, _ChannelState] = {}
        self._by_turn_manager_id: dict[int, _ChannelState] = {}
        # Set by _patch_rag_manager_get_reference_text's timed, redundant
        # format_context() call, read by _patch_rag_manager_background_task
        # for retrieval_breakdown_s.context_injection_s. Both run inside the
        # same asyncio task (the get_reference_text call happens directly
        # inside _background_task's body), so — since this demo is
        # single-session by design (see spec's Explicitly Out of Scope) and
        # asyncio is single-threaded/cooperative — a plain module-level slot
        # is enough to hand the timing back without a contextvar. Would need
        # revisiting if concurrent multi-channel retrieval is ever supported.
        self.last_context_injection_s: float | None = None

    def now_iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def write_session_start(self, *, checkpoint: str, retrieval_backend: dict, rag_timeout_s: float, stt_mode: str) -> None:
        record = {
            "type": "session_start",
            "session_id": self.session_dir.name,
            "checkpoint": checkpoint,
            "git_hash": _git_hash(),
            "timestamp": self.now_iso(),
            "retrieval_backend": retrieval_backend,
            "rag_timeout_s": rag_timeout_s,
            "stt_mode": stt_mode,
        }
        self._write(record)

    def write_turn(self, record: dict) -> None:
        self._write(record)

    def write_raw_event(self, event: str, *, turn_index: int, **fields) -> None:
        record = {
            "type": "raw_event",
            "event": event,
            "timestamp": self.now_iso(),
            "t_rel_s": round(time.perf_counter() - self._t0, 3),
            "turn_index": turn_index,
            **fields,
        }
        self._raw_fh.write(json.dumps(record) + "\n")
        self._raw_fh.flush()

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def register_channel(self, channel) -> _ChannelState:
        state = _ChannelState(self, channel)
        self._channel_states[id(channel)] = state
        self._by_rag_manager_id[id(channel.rag_manager)] = state
        self._by_turn_manager_id[id(channel.turn_manager)] = state
        return state

    def unregister_channel(self, channel) -> None:
        """Must be called on channel close (Channel.__aexit__) — without this,
        every closed channel (including every automatic reconnect attempt
        Queue.tsx's retry-on-failure makes) stays permanently referenced here
        for the life of the server process. That pins everything reachable
        from it in memory too, most importantly the per-channel deepcopied
        Mimi model handle_chat() makes for STT (Channel.stt) — a real CUDA
        OOM observed during VM verification traced back to this leak.
        """
        self._channel_states.pop(id(channel), None)
        self._by_rag_manager_id.pop(id(channel.rag_manager), None)
        self._by_turn_manager_id.pop(id(channel.turn_manager), None)

    def state_for_turn_manager(self, turn_manager) -> _ChannelState | None:
        return self._by_turn_manager_id.get(id(turn_manager))

    def state_for_rag_manager(self, rag_manager) -> _ChannelState | None:
        return self._by_rag_manager_id.get(id(rag_manager))

    def state_for_channel(self, channel) -> _ChannelState | None:
        return self._channel_states.get(id(channel))


def _patch_server_state(session: SessionLog, checkpoint: str, retrieval_backend_display: dict) -> None:
    """
    retrieval_backend_display (core/retrieval_backend.py's describe_backend()
    output) is computed once in main() and passed in here — NOT read from
    LLM_MODEL_NAME/LLM_BASE_URL env vars, which are otherwise-vestigial
    (see scripts/print_demo_env.py's own docstring on why they're still set
    at all) and don't necessarily reflect the real backend answering
    retrieval once _patch_rag_manager_get_reference_text is active.
    """
    from moshi.server import ServerState

    original_init = ServerState.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        session.write_session_start(
            checkpoint=checkpoint,
            retrieval_backend=retrieval_backend_display,
            rag_timeout_s=self.rag_timeout,
            stt_mode="gradium" if self.gradium_stt else "local",
        )

    ServerState.__init__ = patched_init


def _patch_rag_manager_get_reference_text(session: SessionLog, rag_manager, retrieval_backend) -> None:
    """
    Per-instance replacement (not a wrap) of one RAGManager's
    get_reference_text — routes the demo's actual retrieval call through
    the same core/retrieval_backend.py RetrievalBackend evals use, ported
    near-verbatim from core/model_interface.py's _patch_rag_manager (same
    signature: context: str -> (context, ref_text, elapsed, backend_label)).

    Why a real behavioral patch, not just logging: moshi-rag's own
    LLMReferenceGenerator only ever offers a choice between two bundled
    canned prompt templates (original/simplified, via prompt_style) — there
    is no way to point it at our own prompt_template file or
    context_formatting settings without forking it. Routing the demo
    through the exact same RetrievalBackend instance (same registry-selected
    backend, same prompt file) evals use is the only way to get genuine
    prompt/backend parity between the two paths — not an equivalent
    mechanism, the literal same code path. See specs/
    moshirag-evals-requirements.md's "On prompt parity with the demo path"
    note for the full rationale and what's given up (moshi's native
    multi-provider live-switching UI, MOSHI_RETRIEVAL_LLMS_JSON — a
    documented future extension point on core/retrieval_backend.py's
    registry instead, not implemented here).

    RAGManager is a plain Python object, not the PyO3-native kind
    instance-patching can't touch (confirmed the hard way: an early attempt
    to wrap Channel.opus_writer.append_pcm directly failed with
    `AttributeError: ... attribute 'append_pcm' is read-only`, since
    sphn.OpusStreamWriter is a compiled/native extension type), and Channel
    constructs its own fresh RAGManager per
    connection exactly like InferenceJob does — confirmed directly against
    real inference_utils/channel.py and inference_utils/rag_manager.py
    source. Installed from _patch_channel's patched_init, right where
    self.rag_manager becomes available.

    Reference-history bookkeeping, mirroring moshi's own unpatched
    RAGManager.get_reference_text: after retrieval_backend.retrieve()
    returns, this calls core/retrieval_backend.py's format_context() a
    second, redundant time with the same inputs retrieve() already used
    internally — pure and stateless, so recomputing costs the same as the
    original call — purely to recover num_turns, then does
    rag_manager._history.append((num_turns, reference_text)) exactly like
    moshi's real source does. That redundant call is timed and reported as
    session.last_context_injection_s — the same field
    _patch_rag_manager_background_task already reads for
    retrieval_breakdown_s.context_injection_s, restoring a genuine
    measurement now that the former LLMReferenceGenerator.process_reference_text
    patch (removed — never reached anymore) no longer sets it.
    """
    from core.retrieval_backend import format_context

    async def _patched_get_reference_text(context: str) -> tuple[str, str, float, str]:
        t0 = time.perf_counter()
        try:
            ref_text, latency = retrieval_backend.retrieve(context, history=rag_manager._history)
        except Exception as exc:
            logger.warning("RetrievalBackend.retrieve() failed: %s", exc)
            ref_text, latency = "", 0.0
        elapsed = time.perf_counter() - t0

        t_fmt0 = time.perf_counter()
        context_formatting = getattr(retrieval_backend, "context_formatting", {})
        _, num_turns = format_context(context, rag_manager._history, **context_formatting)
        session.last_context_injection_s = time.perf_counter() - t_fmt0
        if num_turns > 0 and ref_text:
            rag_manager._history.append((num_turns, ref_text))

        # Distinct "[RetrievalBackend]" tag so this doesn't get confused
        # with moshi's own "[Reference] Triggering retrieval with
        # context_len=N snippet='...'" line — that one only shows a
        # truncated tail of context and never logs what came back.
        logger.info(
            "[RetrievalBackend] context=%r -> reference_text=%r (backend_latency=%.3fs)",
            context, ref_text, latency,
        )
        return context, ref_text, elapsed, "RetrievalBackend"

    rag_manager.get_reference_text = _patched_get_reference_text


def _patch_channel(session: SessionLog, retrieval_backend, retrieval_backend_display: dict) -> None:
    from moshi.inference_utils.channel import Channel

    original_init = Channel.__init__
    original_aexit = Channel.__aexit__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        session.register_channel(self)
        _patch_rag_manager_get_reference_text(session, self.rag_manager, retrieval_backend)

        _push_session_info(self, retrieval_backend=retrieval_backend_display)
        # ttfat_s's first-audio-frame signal comes from _ChannelState.on_model_text
        # (see _patch_turn_manager and _ChannelState.on_first_audio's own
        # docstring), not from wrapping self.opus_writer.append_pcm here —
        # that was tried first and confirmed broken during VM verification:
        # sphn.OpusStreamWriter is a compiled/native (PyO3) extension type
        # and doesn't support instance attribute assignment on its methods.

    async def patched_aexit(self, exc_type, exc, tb):
        state = session.state_for_channel(self)
        if state is not None:
            state.flush_final()
        try:
            return await original_aexit(self, exc_type, exc, tb)
        finally:
            # Always deregister, even if original_aexit itself raises — an
            # error tearing down a channel must not also leak its GPU-backed
            # state for the rest of the process's life.
            session.unregister_channel(self)

    Channel.__init__ = patched_init
    Channel.__aexit__ = patched_aexit


def _patch_rag_manager_trigger(session: SessionLog) -> None:
    """`<ret>` detection.

    First attempt patched Channel._decode_text_token to look for a literal
    "[RET]" return value — wrong, confirmed by VM verification:
    rag_triggered stayed False on every turn even when retrieval
    demonstrably happened (retrieval_context/retrieved_reference_text/
    retrieval_breakdown_s all populated via the separate
    RAGManager._background_task patch, which fired correctly and
    independently). Root cause: per core/model_interface.py's already-
    validated real-source read, the real output-loop logic branches
    `if text_token == rag_token_id: ... else: decoded =
    self._decode_text_token(text_token)` — the RAG token is intercepted
    *before* _decode_text_token is ever called, so it never sees that token
    and never returns "[RET]" at all.

    RAGManager.trigger() is the actual real call site the RAG token drives
    (confirmed directly against channel.py: `_output_loop`'s RAG-token
    branch calls `await self.rag_manager.trigger(task_group=self._task_group,
    wait_steps=self.server.stt_wait_steps, handle_reference_fn=...,
    context_provider=...)`) — every real invocation, in both Channel's own
    output loop and respond()/InferenceJob's mirrored logic in
    model_interface.py, only calls this when the model has just predicted
    the RAG token. Wrapping this leaf method is therefore a direct, correct
    signal instead of pattern-matching decoded text.

    One real subtlety confirmed by the same source read: `_output_loop` has
    two `await` points (`_send_turn_outputs`, `stt.flush()`) between
    detecting the RAG token and this trigger() call actually firing, during
    which the concurrently-running `_stt_recv_loop` task can legitimately
    advance the turn boundary first. See _ChannelState.on_ret_triggered's
    own docstring for how that's handled.
    """
    from moshi.inference_utils.rag_manager import RAGManager

    original = RAGManager.trigger

    async def patched(self, *args, **kwargs):
        state = session.state_for_rag_manager(self)
        if state is not None:
            state.on_ret_triggered()
        return await original(self, *args, **kwargs)

    RAGManager.trigger = patched


def _patch_turn_manager(session: SessionLog) -> None:
    from moshi.inference_utils.turn_manager import TurnManager

    original = TurnManager._update_active_speaker

    def patched(self):
        was_user = self.active_speaker == "user"
        new_speaker = original(self)
        # Check the *return value*, not self._pending_speaker after the
        # fact: _handle_pending_speaker_switch() (called inside the
        # original method) clears _pending_speaker back to None within the
        # same call whenever _wait_counter is already 0 (e.g. stt_wait_steps
        # == 0), so probing _pending_speaker afterward would miss that case
        # entirely. The return value is what the caller (handle_spoken_text)
        # actually uses to switch active_speaker, so it's the correct signal
        # in both the immediate (_wait_counter == 0) and delayed cases.
        if was_user and new_speaker == "model":
            state = session.state_for_turn_manager(self)
            if state is not None:
                state.on_utterance_end()
        return new_speaker

    original_handle_spoken_text = TurnManager.handle_spoken_text

    def patched_handle_spoken_text(self, model_text=None, user_text=None):
        state = session.state_for_turn_manager(self)
        if state is not None:
            if user_text is not None:
                state.on_user_text(user_text)
            if model_text is not None:
                state.on_model_text(model_text)
        return original_handle_spoken_text(self, model_text=model_text, user_text=user_text)

    TurnManager._update_active_speaker = patched
    TurnManager.handle_spoken_text = patched_handle_spoken_text


def _patch_rag_manager_background_task(session: SessionLog) -> None:
    """Whole-method patch, not a wrap — same class of change as
    core/model_interface.py's _patch_output_loop, and for the same reason:
    the internal asr_wait/api_call timing boundary this needs to measure
    isn't observable from outside the method. Body below is a verbatim copy
    of RAGManager._background_task as of the "Verified against" commit in
    this file's module docstring, plus three time.perf_counter() calls and
    one session.on_retrieval_complete() call — no other logic changed.
    """
    from moshi.inference_utils.rag_manager import RAGManager

    async def patched_background_task(self, handle_reference_fn, context_provider):
        logger.info("[Reference] Started new reference generation task in background")
        t_start = time.perf_counter()
        try:
            if self._wait_event is not None:
                await self._wait_event.wait()
                self._wait_event = None
            t_after_wait = time.perf_counter()
            logger.info("[Reference] Waiting ended (including zero wait_steps)")
            if context_provider is not None:
                context = context_provider()
            else:
                context = ""
                logger.warning("[Reference] No context provider supplied, generating reference with empty context")
            logger.info(
                f"[Reference] Triggering retrieval with context_len={len(context)} snippet='...{context[-200:]}'"
            )
            session.last_context_injection_s = None
            _, reference_text, _, lm_label = await self.get_reference_text(context)
            t_after_retrieval = time.perf_counter()
            if handle_reference_fn is not None:
                await handle_reference_fn(reference_text, lm_label)
            logger.info("[Reference] Background reference generation task completed")

            state = session.state_for_rag_manager(self)
            if state is not None:
                injection_s = session.last_context_injection_s or 0.0
                api_call_s = (t_after_retrieval - t_after_wait) - injection_s
                state.on_retrieval_complete(
                    context=context,
                    reference_text=reference_text,
                    asr_wait_s=t_after_wait - t_start,
                    context_injection_s=injection_s,
                    api_call_s=max(api_call_s, 0.0),
                )
        except asyncio.CancelledError:
            logger.info("[Reference] Reference generation cancelled")
            raise
        except Exception as e:
            logger.error(f"[Reference] Error generating reference: {e}")

    RAGManager._background_task = patched_background_task


def apply_patches(
    session: SessionLog,
    checkpoint: str,
    retrieval_backend,
    retrieval_backend_display: dict,
) -> None:
    _patch_server_state(session, checkpoint, retrieval_backend_display)
    _patch_channel(session, retrieval_backend, retrieval_backend_display)
    _patch_rag_manager_trigger(session)
    _patch_turn_manager(session)
    _patch_rag_manager_background_task(session)


def _build_retrieval_backend_for_demo() -> tuple[object, dict]:
    """
    Loads DEMO_CONFIG (same YAML shape evals/runner.py uses) and builds the
    one RetrievalBackend instance + describe_backend() display dict shared
    across every connection/turn in this server process — see module
    docstring and core/retrieval_backend.py's build_backend_from_retrieval_config().
    """
    config_path = os.environ.get("DEMO_CONFIG")
    if not config_path:
        print("DEMO_CONFIG must be set (run via scripts/run_demo.sh)", file=sys.stderr)
        sys.exit(1)

    from core.config import load_config
    from core.retrieval_backend import build_backend_from_retrieval_config, describe_backend

    config = load_config(config_path)
    retrieval_cfg = config.get("model", {}).get("retrieval", {})
    retrieval_backend = build_backend_from_retrieval_config(retrieval_cfg)

    backend_name = retrieval_cfg.get("backend")
    backend_def = retrieval_cfg.get("_resolved_backend")
    if backend_name and backend_def is not None:
        retrieval_backend_display = describe_backend(backend_name, backend_def)
    else:
        retrieval_backend_display = {"name": "null", "type": "null_backend"}

    return retrieval_backend, retrieval_backend_display


def main() -> None:
    session_dir_env = os.environ.get("DEMO_SESSION_DIR")
    if not session_dir_env:
        print("DEMO_SESSION_DIR must be set (run via scripts/run_demo.sh)", file=sys.stderr)
        sys.exit(1)
    session_dir = Path(session_dir_env)
    if not session_dir.is_dir():
        print(f"DEMO_SESSION_DIR does not exist: {session_dir}", file=sys.stderr)
        sys.exit(1)

    retrieval_backend, retrieval_backend_display = _build_retrieval_backend_for_demo()

    checkpoint = os.environ.get("DEMO_CHECKPOINT", "unknown")
    session = SessionLog(session_dir)
    apply_patches(session, checkpoint, retrieval_backend, retrieval_backend_display)

    import torch
    import moshi.server

    with torch.no_grad():
        moshi.server.main()


if __name__ == "__main__":
    main()
