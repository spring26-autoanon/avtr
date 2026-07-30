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
        # ttfat_s anchors, held outside self._pending because both are
        # established *before* the turn record exists — see
        # on_user_turn_vad()/on_utterance_end() for why.
        self._t_user_stopped: float | None = None
        self._t_first_model_token: float | None = None
        self._vad_stopped = False
        self._pad_stall_steps = 0

    def on_user_text(self, text: str) -> None:
        self._session.write_raw_event("user_text", turn_index=self.turn_index, text=text)
        self._incoming_user_text.append(text)

    def on_user_turn_vad(self, stopped: bool) -> None:
        """Called once per TurnManager._update_active_speaker evaluation while
        the *user* still holds the turn, with upstream's own ``vad_neg``
        verdict (see _patch_turn_manager's _vad_says_user_stopped).

        Arms ttfat_s's clock on each false->true transition, i.e. on each
        moment VAD newly concludes the user has stopped. Re-arming (rather
        than latching the first one) is deliberate and matters: a mid-question
        pause produces an earlier transition, and anchoring there would
        attribute the pause itself to response latency. Taking the *last*
        stop before the model speaks is what matches the paper's definition
        of TTFAT ("the delay between the end of a user's utterance and the
        moment the model generates the first audio token", §3.1).
        """
        if stopped and not self._vad_stopped:
            self._t_user_stopped = time.perf_counter()
            self._t_first_model_token = None
            self._pad_stall_steps = 0
        self._vad_stopped = stopped

    def on_awaiting_turn_step(self) -> None:
        """One model step elapsed while the user still holds the turn.

        Counts the pad stall (``pad_stall_steps``) — the same quantity
        scripts/count_pad_stalls.py derives from server.log's "LM buffer
        empty" lines, recorded per turn here so it is queryable from
        turns.jsonl without log parsing.

        Off by at most one step: the caller runs before
        _update_active_speaker for the same step, so ``_vad_stopped``
        reflects the previous step's verdict. Immaterial at 80ms resolution
        for a diagnostic counter, and not worth reordering the wrapper for.
        """
        if self._vad_stopped:
            self._pad_stall_steps += 1

    def on_model_text(self, text: str) -> None:
        # First genuine (non-pad) model token after VAD concluded the user
        # stopped — the "first audio token" side of ttfat_s. Recorded here
        # rather than in on_first_audio() because it happens stt_wait_steps
        # *before* on_utterance_end() creates the turn record it belongs to.
        if self._vad_stopped and self._t_first_model_token is None:
            self._t_first_model_token = time.perf_counter()
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
        """Called as the turn actually switches to the model — the turn
        boundary: whatever was pending from the previous turn is now complete
        and gets flushed.

        **This is no longer the ttfat clock start (changed 2026-07-30).** It
        used to be, and that was the bug: this fires only once
        TurnManager.model_text_buffer is non-empty, i.e. *after* the model has
        already produced its first real token, and then after a further
        stt_wait_steps hold. The resulting ttfat_s measured one step and read
        0.08-0.32s on sessions where the human waited 4-19s. See
        docs/demo-turn-onset-regression.md §6.1.

        ttfat_s is now computed here from the two anchors gathered earlier in
        the turn (on_user_turn_vad / on_model_text) — both are known by the
        time control reaches this point, which is why it can be resolved
        eagerly rather than waiting for another model token.

        The old quantity is preserved as ``turn_switch_to_first_token_s``
        (filled in by on_first_audio) rather than dropped: it measures
        something real — the stt_wait_steps + display-gating interval — and
        every pre-2026-07-30 session log carries it under the ttfat_s name, so
        keeping it under an honest name is what makes old and new sessions
        comparable instead of silently incomparable.
        """
        self._flush_pending()
        self.turn_index += 1
        ttfat_s = self._resolve_ttfat_s()
        pad_stall_steps = self._pad_stall_steps if self._t_user_stopped is not None else None
        self._session.write_raw_event(
            "utterance_end",
            turn_index=self.turn_index,
            ttfat_s=ttfat_s,
            pad_stall_steps=pad_stall_steps,
        )
        self._pending = {
            "type": "turn",
            "turn_index": self.turn_index,
            "timestamp": self._session.now_iso(),
            "user_question_text": "".join(self._incoming_user_text).strip(),
            "model_response_text": "",
            "ttfat_s": ttfat_s,
            "pad_stall_steps": pad_stall_steps,
            "turn_switch_to_first_token_s": None,
            "e2ekd_s": None,
            "keyword_delay_s": None,
            "_switch_at": time.perf_counter(),
        }
        self._incoming_user_text = []
        # ttfat_s is known now, so push it immediately rather than waiting for
        # the next model token (which is what the old on_first_audio path did).
        _push_instrumentation(self.channel, self.turn_index, {"ttfat_s": ttfat_s})
        self._t_user_stopped = None
        self._t_first_model_token = None
        self._vad_stopped = False
        self._pad_stall_steps = 0

    def _resolve_ttfat_s(self) -> float | None:
        """End of the user's utterance -> first genuine model token, per the
        paper's §3.1 TTFAT definition.

        Returns None only when VAD never concluded the user stopped during
        this turn (no anchor to measure from — e.g. the very first turn of a
        session that opens with ``--init-active-speaker model``).

        Returns 0.0 when the user *did* stop but the model had already begun
        speaking before that (a barge-in / backchannel that carried into the
        turn switch): TTFAT is zero-or-negative there, and the paper reports
        exactly 0.0 for both MoshiRAG and vanilla Moshi, so clamping matches
        both the definition and the published baseline.
        """
        if self._t_user_stopped is None:
            return None
        if self._t_first_model_token is None:
            return 0.0
        return round(max(0.0, self._t_first_model_token - self._t_user_stopped), 4)

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
        """Records ``turn_switch_to_first_token_s`` — the first genuine model
        token *after* the turn switched to the model.

        This is exactly what ``ttfat_s`` used to mean before 2026-07-30 (see
        on_utterance_end's docstring); the name now says so. It is the
        stt_wait_steps + display-gating interval, not response latency.
        """
        if self._pending is None or self._pending["turn_switch_to_first_token_s"] is not None:
            return  # already recorded for this turn
        elapsed = round(time.perf_counter() - self._pending["_switch_at"], 4)
        self._pending["turn_switch_to_first_token_s"] = elapsed
        self._session.write_raw_event(
            "first_audio",
            turn_index=self.turn_index,
            turn_switch_to_first_token_s=elapsed,
            ttfat_s=self._pending["ttfat_s"],
        )
        _push_instrumentation(self.channel, self.turn_index, {"turn_switch_to_first_token_s": elapsed})

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
        # Opened lazily by write_step_diag() so DEMO_QUEUE_DIAG=0 sessions
        # never create an empty step_diag.jsonl.
        self._step_diag_fh = None
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

    def write_step_diag(self, **fields) -> None:
        """One record per *sampled* model step, to its own file.

        Separate from raw_events.jsonl on purpose: at Mimi's 12.5Hz this is
        ~1500 steps for a two-minute session, which would drown the existing
        logs. Only written when DEMO_QUEUE_DIAG=1 (see
        _patch_step_queue_diagnostics), so the file is absent from ordinary
        sessions rather than empty.
        """
        if self._step_diag_fh is None:
            self._step_diag_fh = open(self.session_dir / "step_diag.jsonl", "a")
        record = {"t_rel_s": round(time.perf_counter() - self._t0, 3), **fields}
        self._step_diag_fh.write(json.dumps(record) + "\n")
        self._step_diag_fh.flush()

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


def _step_pacing_enabled() -> bool:
    """Gate for _patch_server_state_step_pacing().

    Extracted from that function purely so the default is testable: the
    patch body itself imports moshi.server, which isn't installed in the
    local test environment (see tests/test_instrumented_server.py's module
    docstring on why those patches are VM-verified rather than mocked).

    Default **disabled** as of 2026-07-30 — see that function's docstring
    for why the patch's original premise was wrong, and
    docs/demo-turn-onset-regression.md for the evidence. Mirrors how
    the since-deleted STT_OFF_THREAD toggle was flipped: the variable
    stays, opt-in rather than opt-out.
    """
    return os.environ.get("DEMO_STEP_PACING", "0") == "1"


def _input_queue_max_frames() -> int:
    """Cap for Channel.input_queue, in frames. 0 disables the bound entirely
    (upstream behaviour), for A/B.

    Default 2: `_gather_step_inputs()` consumes exactly one frame per step, so
    a cap of 1 would hold only the newest frame and drop on any scheduling
    jitter; 2 gives one frame of genuine slack. Measured sessions hold
    `lag_frames` at 1 in steady state, so at this cap mid-session drops should
    be ~0 — `input_dropped` in step_diag.jsonl is what confirms that.
    """
    try:
        return max(0, int(os.environ.get("DEMO_INPUT_QUEUE_MAX", "2")))
    except ValueError:
        return 2


class _DropOldestQueue(asyncio.Queue):
    """Channel.input_queue with a frame cap and drop-oldest overflow — B3.

    Why (docs/demo-turn-onset-fix-plan.md's B3 and D10 cycle 3): every session
    accumulates a 3.9-4.9s input backlog during model load and STT priming,
    while the client is already streaming. Draining it is not free — the step
    loop consumes at ~1.78x real time until it catches up, the browser's
    playback buffer cannot hold the surplus, and it discards **2.534s of audio,
    about 47% of the model's greeting** (measured, session
    2026-07-30T20-31-09Z). That is the largest single contributor to the D10
    artifact and the reason the opening phrase sounds rushed.

    Bounding the queue removes the backlog instead of draining it. Dropping
    those frames here costs nothing — the window precedes any user speech, and
    `Channel._recv_loop` feeds the STT *before* queueing, so the transcript and
    VAD still see every frame; only the front-end model's own input is
    thinned. Dropping them at the output instead destroys half the greeting.

    `put()` stays `async def` to match the single call site
    (`await self.input_queue.put(step_input)`) but never actually blocks: the
    parent is left unbounded and the cap is enforced here, so a full queue
    trims rather than applying backpressure to `_recv_loop`.

    **`is_first` is never lost.** The first frame of a session carries
    `is_first=True`, which is what makes `BatchRunner.run_step()` call
    `mimi.reset_streaming`/`lm_gen.reset_streaming`. Silently dropping it would
    leave the models in whatever state the previous connection left behind, so
    the flag is transferred onto the surviving frame instead. This is the one
    genuinely dangerous part of bounding this queue.
    """

    def __init__(self, max_frames: int) -> None:
        # Deliberately unbounded upstream; _cap is enforced in put() so that
        # overflow drops instead of blocking the receive loop.
        super().__init__()
        self._cap = max_frames
        self.dropped = 0

    async def put(self, item) -> None:
        if self._cap > 0:
            while self.qsize() >= self._cap:
                try:
                    stale = self.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - racing put/get
                    break
                self.dropped += 1
                if getattr(stale, "is_first", False):
                    # Carry the session-reset signal forward, see class docstring.
                    item.is_first = True
        self.put_nowait(item)


def _slot_backlog_rows(server, consumed: dict[int, int]) -> list[dict]:
    """Per-active-slot backlog snapshot. Pure function of the server object
    plus a caller-owned consumed-frames tally, so it is unit-testable against
    a duck-typed stand-in without a real moshi-rag install.

    Three independent numbers, deliberately — they localise *where* audio is
    piling up, which `qsize` alone cannot:

    - ``qsize``: frames sitting in Channel.input_queue right now.
    - ``frames_received``: frames the *live* path has seen, derived from
      ``LocalSpeechToText.sent_samples`` (incremented once per
      ``send_audio()`` call in ``Channel._recv_loop``, before the frame is
      queued). This is the only live-clock counter that already exists in
      upstream, and it is what makes the measurement independent of where
      buffering happens — if audio is backing up in the WebSocket or the
      Opus reader rather than in input_queue, ``qsize`` stays small while
      this number still tracks reality. Absent under ``--gradium-stt``
      (GradiumSpeechToText has no such attribute), hence the getattr.
    - ``frames_consumed``: frames the step loop has actually stepped for
      this slot, tallied by the _deliver_step_row wrapper.

    ``lag_frames = frames_received - frames_consumed`` is the headline
    figure: multiply by 80ms to get how far behind live audio the front-end
    model's perception is. See docs/demo-turn-onset-regression.md §8.
    """
    rows: list[dict] = []
    frame_size = getattr(server, "frame_size", 0) or 0
    for idx, occupant in enumerate(getattr(server, "slots", None) or []):
        if occupant is None:
            continue
        sent_samples = getattr(getattr(occupant, "stt", None), "sent_samples", None)
        received = None if (sent_samples is None or not frame_size) else int(sent_samples // frame_size)
        used = consumed.get(id(occupant), 0)
        rows.append(
            {
                "slot": idx,
                "qsize": occupant.input_queue.qsize(),
                "frames_received": received,
                "frames_consumed": used,
                "lag_frames": None if received is None else received - used,
                # Cumulative frames discarded by _DropOldestQueue (B3). Absent
                # (None) when the queue is upstream's unbounded asyncio.Queue,
                # i.e. DEMO_INPUT_QUEUE_MAX=0. Expected shape once B3 is on: a
                # jump of ~30-50 during startup, then flat -- steady-state
                # lag_frames is 1, so there should be nothing left to drop.
                "input_dropped": getattr(occupant.input_queue, "dropped", None),
            }
        )
    return rows


def _patch_step_queue_diagnostics(session: SessionLog) -> None:
    """Opt-in (``DEMO_QUEUE_DIAG=1``) per-step trace of input-queue backlog,
    real step timing, and raw text token ids — the measurement that
    docs/demo-turn-onset-regression.md §8 identifies as the one thing able to
    settle *why* the demo's turn onset is slow.

    The two candidate mechanisms make opposite predictions, and this patch
    distinguishes them in a single session:

    - **Input backlog** (leading hypothesis): ``lag_frames`` climbs to ~50
      during startup and never returns to ~0, growing through the session.
      The front-end model is perceiving user audio seconds late, so its
      turn-taking is correct behaviour applied to stale input.
    - **Genuine generation behaviour**: ``lag_frames`` sits at 0-2 throughout
      and the model really is declining to speak on time-aligned input. The
      ``text_tokens`` trace then becomes the actionable data, which is why it
      is collected here rather than left to a follow-up session.

    ``text_tokens`` folds in what the fix plan tracked separately as D1
    (PAD vs EPAD indistinguishable in logs, since
    ``Channel._decode_text_token`` maps ids 0-3 all to ``None``). It is free
    here: ``_deliver_step_row`` already receives ``text_token`` as a plain
    Python int — ``BatchRunner.run_step`` does the ``.item()`` sync itself —
    so recording it costs no extra GPU synchronisation. That mattered: any
    per-step device sync added by this patch would inflate the very step
    timings it exists to measure.

    Sampling: one record every ``DEMO_QUEUE_DIAG_EVERY`` steps (default 12,
    ≈1/s) to keep the file scannable, but ``text_tokens`` carries *every*
    token id since the previous record, so the token stream is complete
    regardless of the sampling rate.

    ``work_ms`` vs ``mean_period_ms``: this patch must be applied *before*
    _patch_server_state_step_pacing() so that it ends up the inner wrapper.
    ``work_ms`` is then the real ``run_one_step`` compute with any pacing
    sleep excluded, while ``mean_period_ms`` (measured between consecutive
    sampled records) includes it — i.e. the pair directly shows how much of
    the 80ms budget is work and how much is artificial floor.
    """
    if os.environ.get("DEMO_QUEUE_DIAG", "0") != "1":
        return

    from moshi.server import ServerState

    if getattr(ServerState, "_queue_diag_patched", False):
        return

    try:
        sample_every = max(1, int(os.environ.get("DEMO_QUEUE_DIAG_EVERY", "12")))
    except ValueError:
        sample_every = 12

    original_deliver = ServerState._deliver_step_row
    original_run_one_step = ServerState.run_one_step

    # Keyed by id(channel). One int per channel ever opened — unlike
    # SessionLog._channel_states (see unregister_channel's leak note) these
    # hold no reference to the channel itself, so a stale entry cannot pin a
    # deepcopied Mimi model in memory.
    consumed: dict[int, int] = {}
    pending_tokens: list[int] = []
    state = {"step": 0, "prev_t": None}

    def patched_deliver(self, occupant, *, text_token, pcm_out):
        consumed[id(occupant)] = consumed.get(id(occupant), 0) + 1
        pending_tokens.append(int(text_token))
        return original_deliver(self, occupant, text_token=text_token, pcm_out=pcm_out)

    async def patched_run_one_step(self):
        t0 = time.perf_counter()
        ran = await original_run_one_step(self)
        t1 = time.perf_counter()
        if not ran:
            return ran
        state["step"] += 1
        if state["step"] % sample_every:
            return ran
        prev_t = state["prev_t"]
        state["prev_t"] = t1
        tokens = list(pending_tokens)
        pending_tokens.clear()
        session.write_step_diag(
            step=state["step"],
            work_ms=round((t1 - t0) * 1000, 2),
            mean_period_ms=None if prev_t is None else round((t1 - prev_t) * 1000 / sample_every, 2),
            slots=_slot_backlog_rows(self, consumed),
            text_tokens=tokens,
        )
        return ran

    ServerState._deliver_step_row = patched_deliver
    ServerState.run_one_step = patched_run_one_step
    ServerState._queue_diag_patched = True
    # print(), not logger.info() — see _patch_server_state_step_pacing()'s
    # docstring: apply_patches() runs before moshi.server.main() configures
    # the root logger, so logger.info() here is silently swallowed.
    print(
        f"[QueueDiag] DEMO_QUEUE_DIAG=1 -- writing step_diag.jsonl every {sample_every} steps",
        file=sys.stderr,
    )


def _patch_server_state_step_pacing() -> None:
    """Adds real-time pacing to ServerState.run_one_step() — demo-only,
    does not touch BatchRunner.run_step() itself (shared with
    core/model_interface.py's eval path, which wants maximum batch
    throughput, not real-time pacing).

    **DEFAULT FLIPPED TO DISABLED (2026-07-30). The original rationale
    below is wrong, and this patch is the leading cause of the demo's
    turn-onset regression — see docs/demo-turn-onset-regression.md §4.3.**

    What the original rationale got wrong: it claimed nothing throttles
    output, but ServerState._gather_step_inputs() pulls with
    input_queue.get_nowait() and run_one_step() returns False when no
    channel has a frame ready (upstream source, re-read directly). The
    step loop is therefore *input-driven* — the client's own real-time
    audio upload is the throttle and Channel.input_queue is the buffer.
    The server cannot outrun the client, so the premise that it "ships
    audio strictly faster than real-time, continuously" does not hold.

    What this patch actually does, given that: it acts as a one-way latch.
    It never fires while the queue is empty (`ran` is False, so no sleep —
    harmless, and invisible in the healthy case). Once a backlog exists it
    caps drain at exactly the production rate, so the backlog can never
    shrink, and every step whose own compute exceeds 80ms ratchets it
    further up. Real per-step work measured 60-79ms against an 80ms
    budget; that 1-25% headroom was the only mechanism draining the
    3.9-4.9s backlog every session accumulates during warm-up/STT priming.
    Removing it turned a transient startup backlog into a permanent ~4s
    turn-onset floor that then grew to 8-10s over a session.

    Supporting correction: the client-side jitter-buffer overrun cited
    below as the motivation was measured with the *first, later-corrected*
    version of the liveBufferS field (CLAUDE.md records that correction
    itself). The corrected field reads 0ms for 99.3% of samples — recorded
    at the time as "genuinely healthy", but a playback buffer pinned at
    zero is a *starved* client, i.e. the opposite condition. The evidence
    that justified this patch has since been invalidated.

    **Partial rehabilitation (VM session 2, 2026-07-30):** the author of
    this patch saw something real. Measured with pacing off,
    `mean_period_ms` is ~66ms *while a backlog is draining* and snaps to
    ~80ms the instant the queue empties — so during a drain the server
    genuinely ships audio ~21% faster than real time, which is audible
    (independently reported as "slightly fast, metallic artifacts more
    noticeable", concentrated in the session's first ~25s). The mistake was
    scope, not observation: a transient confined to backlog drain was read
    as continuous and inherent, and suppressed by guaranteeing the drain
    never completes.

    Do not reintroduce pacing on the strength of that artifact, though:
    session 3's client-side data showed the audible metallic artifact is a
    *client* per-utterance playback-cap warm-up (drops cluster within
    ±0.31s of every model turn onset, `newMaxBufferMs` ratcheting 15->35ms),
    unrelated to pacing or to the drain. The corrected client buffer field
    `liveBufferS` stays a flat, healthy 70-81ms right through the drain
    window. So the drain's fast delivery is real server-side but has no
    demonstrated audible cost. See docs/demo-turn-onset-fix-plan.md's
    session-3 notes and D10.

    Original rationale, kept verbatim as the reasoning trail:

    Root cause this addresses (confirmed against real moshi-rag source,
    not guessed — see batch_runner.py's run_step(): it measures
    elapsed_ms and only *warns* past ~77ms, with no complementary sleep
    when a step finishes early): nothing anywhere in the pipeline paces
    output to real-time. _deliver_step_row() does an uncapped
    output_queue.put_nowait() and Channel ships it over the WebSocket
    immediately. On hardware fast enough to reliably beat the ~80ms
    real-time budget per step (this project's A100s routinely see
    ~50-95ms steps), the server ships audio strictly faster than
    real-time, continuously, with nothing downstream to throttle it back
    down. This is the confirmed mechanism behind the demo's chronic
    client-side jitter-buffer overrun (see
    project_demo_audio_quality_investigation memory / CLAUDE.md's "Demo
    audio quality" section) — a genuine average-rate mismatch, not
    ordinary jitter, which the client's own buffer can absorb but never
    stabilize against.

    Implementation: wraps (does not reimplement) run_one_step — times the
    real call, and if it completed faster than one frame's real-time
    duration (1 / mimi.frame_rate), sleeps the remainder before
    returning. When no step ran (ran=False, no active connections), no
    change — _step_loop's own existing idle-poll sleep(0.005) still
    applies untouched.

    Gate: DEMO_STEP_PACING=1 enables (default **disabled** since
    2026-07-30, was default-enabled) — kept rather than deleted so the
    regression can be reproduced on demand for the VM A/B described in
    docs/demo-turn-onset-fix-plan.md.
    """
    from moshi.server import ServerState

    # print(), not logger.info(): this function runs from apply_patches(),
    # called before moshi.server.main() reaches its own setup_logging()
    # call (server.py's main(), near the bottom) — the root logger has no
    # handler/level configured yet at this point, so logger.info() here
    # would be silently swallowed (confirmed live: this was the original,
    # buggy version, and its [Pacing] lines never appeared in a real VM
    # run's startup log). Every other logger.info() in this file lives
    # inside per-connection handlers that only run once a real session
    # starts, well after setup_logging() -- this function is the
    # exception, since it must run at startup, before any connection.
    if not _step_pacing_enabled():
        print(
            "[Pacing] DEMO_STEP_PACING=0 (default) -- real-time step pacing disabled "
            "(set DEMO_STEP_PACING=1 to reproduce the pre-2026-07-30 behaviour)",
            file=sys.stderr,
        )
        return

    if getattr(ServerState, "_step_pacing_patched", False):
        return

    original_run_one_step = ServerState.run_one_step

    async def patched_run_one_step(self) -> bool:
        step_start = time.monotonic()
        ran = await original_run_one_step(self)
        if ran:
            frame_period_s = 1.0 / self.runner.mimi.frame_rate
            remaining = frame_period_s - (time.monotonic() - step_start)
            if remaining > 0:
                await asyncio.sleep(remaining)
        return ran

    ServerState.run_one_step = patched_run_one_step
    ServerState._step_pacing_patched = True
    print(
        "[Pacing] DEMO_STEP_PACING=1 -- real-time step pacing ENABLED; this is the "
        "known-bad turn-onset configuration, see docs/demo-turn-onset-regression.md",
        file=sys.stderr,
    )


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
            # asyncio.to_thread, NOT a direct call: RetrievalBackend.retrieve()
            # is synchronous (`def`, not `async def`) and makes a real blocking
            # HTTP request to the retrieval LLM, so calling it directly from
            # this `async def` blocked the whole event loop -- including
            # ServerState._step_loop and Channel._recv_loop -- for its full
            # 0.45-0.63s duration, once per <ret>.
            #
            # Measured on demo/sessions/2026-07-30T22-38-51Z: all 4 step-period
            # hitches in the session (113-133ms mean over a 12-step window,
            # against an 80ms budget and a 33ms mean work_ms) contained a
            # retrieval_complete event, 4 for 4, and the extra wall time per
            # hitch matched api_call_s almost exactly (0.639 vs 0.630, 0.483 vs
            # 0.482, 0.398 vs 0.455, 0.481 vs 0.549). Those hitches starved the
            # browser's ~90ms playback buffer outright, producing one underrun
            # and one audible stall per turn -- the residual D10 artifact left
            # after B3.
            #
            # Self-inflicted, not upstream: moshi-rag's own
            # RAGManager.get_reference_text properly awaits
            # `self.reference_generator.generate_reference_text(...)`. This
            # patch replaced an awaited async call with a blocking one. Same
            # class of bug as _patch_server_state_step_pacing's, and the same
            # fix already used for the conditioning fetch (see
            # _patch_channel_conditioning).
            #
            # Safe off-thread: retrieve() reads self.context_formatting and
            # `history`, and does the HTTP call. It mutates nothing shared --
            # rag_manager._history is only appended to below, back on this
            # thread -- and RAGManager.trigger() cancels any pending task
            # before starting a new one, so calls are serialised.
            ref_text, latency = await asyncio.to_thread(
                retrieval_backend.retrieve, context, history=rag_manager._history
            )
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
    from moshi.stt.local_stt import LocalSpeechToText
    from core.model_interface import _warm_up_stt_exec_mask

    original_init = Channel.__init__
    original_aexit = Channel.__aexit__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # See _warm_up_stt_exec_mask()'s own docstring (eval-path
        # precedent) — same fix, same reasoning, applied here since
        # Channel.__init__ (moshi-rag's own upstream code, just called
        # above) constructs self.stt the same way core/model_interface.py
        # does, and this point — after __init__ returns, before this
        # channel's own recv_loop/step_loop tasks are created — is the
        # equivalent genuinely single-threaded safe window for the demo
        # path. Guarded by isinstance: unlike the eval path's
        # _load_models() (which hardcodes LocalSpeechToText), Channel.stt
        # can also be a GradiumSpeechToText (--stt gradium, moshi-rag's
        # own real branch, channel.py) — no local mimi/_lm_gen at all, so
        # calling this unconditionally would crash with an AttributeError
        # whenever remote STT is configured.
        if isinstance(self.stt, LocalSpeechToText):
            _warm_up_stt_exec_mask(self.stt)
        # B3: swap in the bounded, drop-oldest input queue. Safe to replace the
        # attribute here rather than patching asyncio.Queue: Channel.__init__
        # has just constructed it and nothing has been queued yet (this runs
        # before run()'s TaskGroup creates _recv_loop), and the only consumer,
        # ServerState._gather_step_inputs(), reaches it through this same
        # attribute. See _DropOldestQueue's docstring for why the bound exists.
        cap = _input_queue_max_frames()
        if cap > 0:
            self.input_queue = _DropOldestQueue(cap)
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


def _vad_says_user_stopped(turn_manager) -> bool:
    """Read-only mirror of TurnManager._update_active_speaker's own ``vad_neg``
    expression, plus its ``len(vad_history) < window_size`` early return.

    Copied rather than derived because upstream computes ``vad_neg`` in a local
    and never exposes it, and this repo does not reimplement TurnManager. Kept
    to a single expression so the duplication is auditable at a glance:

        vad_neg = all([value > self.threshold for value in self.vad_history])

    If a future moshi-rag changes that expression, ttfat_s silently anchors on
    the wrong condition — the signal would be ttfat_s going None or absurd on
    every turn while server.log still shows normal "LM buffer empty" runs.
    """
    history = turn_manager.vad_history
    if len(history) < turn_manager.window_size:
        return False
    return all(value > turn_manager.threshold for value in history)


def _patch_turn_manager(session: SessionLog) -> None:
    from moshi.inference_utils.turn_manager import TurnManager

    original = TurnManager._update_active_speaker

    def patched(self):
        was_user = self.active_speaker == "user"
        # ttfat_s's clock start. Sampled before calling through, so the verdict
        # is the one this evaluation is about to act on — see
        # _ChannelState.on_user_turn_vad for why the *last* stop wins.
        if was_user:
            state = session.state_for_turn_manager(self)
            if state is not None:
                state.on_user_turn_vad(_vad_says_user_stopped(self))
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
            # user_text is None <=> this call came from Channel._output_loop,
            # i.e. exactly one model step (it passes model_text only, possibly
            # None for a pad token). Channel._stt_recv_loop is the only other
            # caller and always passes user_text. That distinction is what
            # makes pad_stall_steps a true step count rather than a mix of
            # steps and STT words.
            if user_text is None and self.active_speaker == "user":
                state.on_awaiting_turn_step()
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


def _patch_channel_conditioning() -> None:
    """
    Whole-method replacement of Channel._async_update_reference — the demo
    path's exact counterpart to core/model_interface.py's
    _fetch_and_apply_reference_conditioning() fix for respond()/evals. See
    that function's docstring for the full root-cause chain (confirmed via
    a real dual-GPU test: the ~1.5-1.9s conditioning latency is moshi-rag's
    own ServerState._step_loop blocking the event loop synchronously per
    real-time step, starving the conditioning HTTP call — not GPU
    contention, and not fixable by touching the step loop itself, which is
    load-bearing for real-time per-step conditioning injection).

    Confirmed via a direct clone of kyutai-labs/moshi-rag that Channel's own
    _async_update_reference is the same get_conditioning_remote_async +
    update_streaming_sum_tensors pattern as InferenceJob's, driven by the
    same shared step loop — so the demo is exposed to the identical bug,
    not just respond()/evals. This is also the original bug: the live
    hallucination-feedback-loop session that kicked off this whole
    investigation ran through this exact code path.

    Reuses core/model_interface.py's shared helper rather than a second,
    independently-drifting implementation of the same fetch-and-apply
    sequence. Does not touch session.last_context_injection_s (still set
    from the older, separately-known-imprecise format_context() timing in
    _patch_rag_manager_get_reference_text — see that function's docstring)
    — Channel._handle_reference_text schedules _async_update_reference as a
    fire-and-forget background task, so by the time
    RAGManager._background_task reads last_context_injection_s to report
    retrieval_breakdown_s, this task's real timing hasn't necessarily
    completed yet; correcting that measurement for the demo path is a
    separate, unresolved problem, not addressed by this patch.
    """
    from moshi.inference_utils.channel import Channel
    from core.model_interface import _fetch_and_apply_reference_conditioning

    async def patched_async_update_reference(self, reference_text: str) -> None:
        await _fetch_and_apply_reference_conditioning(
            reference_text,
            encoder_url=self.server.reference_encoder_url,
            lm_gen=self.server.runner.lm_gen,
            batch_size=self.server.batch_size,
            slot_idx=self.slot_idx,
        )
        self._log.info("[Reference] updated streaming_sum condition on LM")

    Channel._async_update_reference = patched_async_update_reference


def _patch_event_loop_diagnostics() -> None:
    """
    Opt-in-only (DEMO_ASYNCIO_DEBUG=1) diagnostic for the conditioning-fetch
    residual-latency investigation (CLAUDE.md's "SUPERSEDED: GPU contention
    conclusion was wrong" section): a real demo session measured the
    post-fetch asyncio.to_thread handoff (loop.call_soon_threadsafe
    delivering the worker thread's already-finished result back to the
    awaiting coroutine) at 0.6-0.9s per trigger — ~6-9x the eval path's own
    ~30-100ms version of the identical gap — and unlike the eval path, the
    demo's gap doesn't correlate with any nearby `batched step` warning, so
    plain run_step() blocking doesn't obviously explain its size. This turns
    on asyncio's own debug mode on the real server event loop, which logs
    any callback exceeding slow_callback_duration together with a repr that
    identifies it, to see directly what's occupying the loop during a gap
    window instead of continuing to infer it from run_step()'s own warning
    line, which only covers one code path.

    Hooks asyncio.events.set_event_loop, NOT the asyncio.set_event_loop
    re-export — confirmed the hard way (first version of this patch hooked
    the wrong one and silently never fired on a real VM run, no error, just
    zero diagnostic output). On Python 3.11 asyncio.run() is implemented via
    asyncio.runners.Runner._lazy_init(), which does `from . import events`
    and calls events.set_event_loop(loop) directly — a name resolved through
    the events submodule at call time, never touching whatever
    `asyncio.set_event_loop` happens to be rebound to in the asyncio package
    namespace. `asyncio/__init__.py`'s `asyncio.set_event_loop` is a
    separate name binding to the same original function object at import
    time; rebinding one doesn't rebind the other. Patching
    asyncio.events.set_event_loop directly hits the actual call site
    asyncio.run() uses, verified against the real Runner code path (not just
    against a direct call to the same patched name, which is what silently
    passed the first, wrong version of this check).

    Off by default — meant for one targeted diagnostic session, not left on:
    debug mode adds real per-callback overhead that would skew the very
    latency numbers under investigation if enabled for normal demo/eval use.
    """
    if os.environ.get("DEMO_ASYNCIO_DEBUG") != "1":
        return

    threshold_s = float(os.environ.get("DEMO_ASYNCIO_DEBUG_THRESHOLD_S", "0.02"))
    original_set_event_loop = asyncio.events.set_event_loop

    def patched_set_event_loop(loop) -> None:
        original_set_event_loop(loop)
        if loop is not None:
            loop.set_debug(True)
            loop.slow_callback_duration = threshold_s
            logger.warning(
                "[Diag] asyncio debug mode ON for this loop "
                "(slow_callback_duration=%.3fs, DEMO_ASYNCIO_DEBUG=1)",
                threshold_s,
            )

    asyncio.events.set_event_loop = patched_set_event_loop
    asyncio.set_event_loop = patched_set_event_loop


def apply_patches(
    session: SessionLog,
    checkpoint: str,
    retrieval_backend,
    retrieval_backend_display: dict,
    temp_text: float,
    top_k_text: int,
) -> None:
    from core.model_interface import _patch_load_models_generation_overrides

    _patch_load_models_generation_overrides(temp_text=temp_text, top_k_text=top_k_text)
    _patch_event_loop_diagnostics()
    _patch_server_state(session, checkpoint, retrieval_backend_display)
    # Order matters: queue diagnostics must be the *inner* run_one_step
    # wrapper so its work_ms excludes any pacing sleep applied on top. See
    # _patch_step_queue_diagnostics()'s docstring.
    _patch_step_queue_diagnostics(session)
    _patch_server_state_step_pacing()
    _patch_channel(session, retrieval_backend, retrieval_backend_display)
    _patch_channel_conditioning()
    # STT runs synchronously on the event loop, as upstream moshi-rag wrote
    # it. The "Option E" chain that used to be applied here behind
    # STT_OFF_THREAD was deleted 2026-07-30 — see core/model_interface.py's
    # _load_models() comment and docs/demo-turn-onset-regression.md for why
    # the backlog it mitigated was self-inflicted.
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


def _generation_overrides_for_demo() -> tuple[float, int]:
    """
    Reads model.generation.temp_text/top_k_text from DEMO_CONFIG, merged
    over _DEFAULT_GENERATION the same way every other generation field is —
    see that dict's own comment in core/model_interface.py for why these
    two specifically bypass scripts/print_demo_env.py's CLI-flags mechanism
    (moshi.server has no matching CLI flag) and get read directly here
    instead. Pure/stateless like _build_retrieval_backend_for_demo() above
    — a second, independent parse of the same small YAML file is cheap and
    keeps each function self-contained.
    """
    config_path = os.environ.get("DEMO_CONFIG")
    if not config_path:
        print("DEMO_CONFIG must be set (run via scripts/run_demo.sh)", file=sys.stderr)
        sys.exit(1)

    from core.config import load_config
    from core.model_interface import _DEFAULT_GENERATION

    config = load_config(config_path)
    generation = {**_DEFAULT_GENERATION, **config.get("model", {}).get("generation", {})}
    return generation["temp_text"], generation["top_k_text"]


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
    temp_text, top_k_text = _generation_overrides_for_demo()

    checkpoint = os.environ.get("DEMO_CHECKPOINT", "unknown")
    session = SessionLog(session_dir)
    apply_patches(
        session, checkpoint, retrieval_backend, retrieval_backend_display, temp_text, top_k_text
    )

    import torch
    import moshi.server

    with torch.no_grad():
        moshi.server.main()


if __name__ == "__main__":
    main()
