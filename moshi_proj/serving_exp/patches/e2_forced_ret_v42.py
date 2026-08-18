"""v4.2: adds a session-start holdoff and a sentinel scaffold on top of the
frozen v4.1 hook below (v4.1 lives, unmodified, in e2_forced_ret.py — this
file is a full copy with deltas, not an import). The holdoff skips arming
entirely for MOSHI_BACKSTOP_HOLDOFF seconds after the first STT message of
the session ("[Backstop] holdoff skip"), stamped off a session-proxy
timestamp set before the question gate so it covers the whole session, not
just the first question. Native fires are never gated by the holdoff — only
this patch's own arming is. The sentinel (SENTINEL_ANCHOR/SENTINEL_REPLACEMENT)
declines injecting a reference the model reports as unavailable; it is a
scaffold until anchors.md section E is recorded, so both constants may be
None. MUTUALLY EXCLUSIVE with the other on_text_hook patches (e0/e2/e42):
apply at most one hook patch at a time — hooks overwrite each other.

E2: forced single-<ret> backstop. If the user asks a question and the
model never natively emits a <ret> (RAG) token for it, a delayed timer forces
ONE occurrence by overwriting this slot's token, on its next PADDING frame
(sampled token == 3), with the model's own rag_token_id — the model's native
fire-detection code (which this patch does NOT touch) then reacts to that
token exactly as it would to an organic fire: real context, real Gemini
call, real injection. Gated by MOSHI_BACKSTOP_FORCE=1. apply/revert on the
BOX.

Consume-on-pad, not consume-on-word: field testing at the target checkpoint
showed the original "next word token" rule deadlocking on exactly the turns
the backstop exists for — a stalled model emitting only padding never gives
it a word token, and a fast deflection spends its words before the force
even arms. Landing the forced token in a padding (silence) frame is also
in-distribution: trained <ret> markers sit in the quiet beat before an
answer. Word tokens (and the other special tokens 0/1/2) pass through
untouched while a force is pending, so her mid-sentence speech is never
hijacked — the force just waits for her next gap.

One anchored replacement (see serving_exp/patches/anchors.md, section D —
the same _stt_recv_loop line e1_silent_backstop.py's ANCHOR_TIMER uses).
Anchor B (native fire detection) is left untouched: E2 observes native fires
from inside its own on_text_hook instead of patching that site, so a single
hook is the one place that both (a) consumes a pending force and (b) records
that a native fire happened, without needing a second anchored edit.

MUTUALLY EXCLUSIVE with the other on_text_hook patches (e0/e2): apply at
most one at a time — hooks overwrite each other.

Usage on box:  python3 e2_forced_ret_v42.py apply
               python3 e2_forced_ret_v42.py revert
"""
import sys

try:
    from patchlib import PatchError, apply_patch, revert_patch
except ImportError:  # local test import; box runs from the patches dir
    from serving_exp.patches.patchlib import PatchError, apply_patch, revert_patch

TARGET = "/home/leenatantawy/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/channel.py"

TAG = "e42_force"
SENTINEL_TAG = "e42_sentinel"
_SENTINEL_MISSING_MSG = "SENTINEL_ANCHOR is None (anchors.md section E not recorded yet) — sentinel part skipped"

# anchors.md section D — _stt_recv_loop body line, verbatim (exact indentation).
ANCHOR = '''                await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))'''

# Parsed EARLY (synchronously, before the check task is created) and guarded —
# a malformed MOSHI_BACKSTOP_DELAY must degrade to the 2.5s default instead of
# raising inside the TaskGroup task, where a ValueError would propagate as an
# ExceptionGroup and tear down every sibling loop (recv/output/stt). Same
# fixed shape as e1_silent_backstop.py's _DELAY_PARSE.
_DELAY_PARSE = '''\
                    try:
                        _delay = float(_os.environ.get("MOSHI_BACKSTOP_DELAY", "2.5"))
                    except ValueError:
                        _delay = 2.5
'''

# Same early-parse, same-shape fallback as _DELAY_PARSE, for the pending
# force's TTL. Default 8.0s: at the target checkpoint the next padding frame
# after arming can legitimately be several seconds out, and the TTL now
# mainly guards the cross-session case (question-time clearing is the
# cross-question guard) rather than bounding a routine wait.
_TTL_PARSE = '''\
                    try:
                        _ttl = float(_os.environ.get("MOSHI_BACKSTOP_TTL", "8.0"))
                    except ValueError:
                        _ttl = 8.0
'''

# v4 FIX 1: how far back (seconds) a native <ret> fire still counts against
# this question at engage time. Field evidence showed a native fire that
# lands WHILE the user is still talking — before the terminal "?" that arms
# the timer even arrives — used to get erased by arming's flag reset,
# causing a duplicate forced fire. The fix replaces the boolean flag with a
# timestamp arming never touches (see REPLACEMENT below); this is that
# timestamp's lookback window. Same early-parse, same-shape fallback as
# _DELAY_PARSE/_TTL_PARSE.
_LOOKBACK_PARSE = '''\
                    try:
                        _lookback = float(_os.environ.get("MOSHI_BACKSTOP_LOOKBACK", "8.0"))
                    except ValueError:
                        _lookback = 8.0
'''

# v4 FIX 2: minimum gap (seconds) required since the backstop last actually
# consumed a forced ret (rewrote a pad) before it will arm a new one on this
# slot. Cumulative injections correlate with late-session audio degradation
# at the target checkpoint, so this bounds condition-swap density. Native
# fires are never blocked by this window — it only gates the backstop's own
# arming. Same early-parse, same-shape fallback as the others above.
_REFRACTORY_PARSE = '''\
                    try:
                        _refractory = float(_os.environ.get("MOSHI_BACKSTOP_REFRACTORY", "6.0"))
                    except ValueError:
                        _refractory = 6.0
'''

# v4.2: session-start holdoff — for this many seconds after the first STT
# message of the session (see the _e2_sess_ts stamp below), arming is
# skipped entirely, regardless of question content. Native fires are never
# gated by this window — only this patch's own arming is. Same early-parse,
# same-shape fallback as the others above.
_HOLDOFF_PARSE = '''\
                    try:
                        _holdoff = float(_os.environ.get("MOSHI_BACKSTOP_HOLDOFF", "12.0"))
                    except ValueError:
                        _holdoff = 12.0
'''

REPLACEMENT = ANCHOR + '''
                import os as _os
                import time as _time
                if not hasattr(self, "_e2_sess_ts"):
                    self._e2_sess_ts = _time.monotonic()
                if _os.environ.get("MOSHI_BACKSTOP_FORCE", "") == "1" and msg.text.rstrip().endswith("?"):
                    if not hasattr(self.server, "_e2_force"):
                        self.server._e2_force = {}
                        self.server._e2_native_ts = {}
                        self.server._e2_forced_ts = {}
                        _pending = self.server._e2_force
                        _lm_gen = self.server.runner.lm_gen
                        _rag_id = _lm_gen.lm_model.rag_token_id
                        _bs_log = __import__("logging").getLogger("backstop")
                        def _e2_hook(text_token, _p=_pending, _r=_rag_id, _s=self.server, _t=_time, _log=_bs_log):
                            # A pending entry is a TTL deadline, not a bare
                            # flag: past-deadline entries are dropped
                            # unconditionally (expire) so a stale force from
                            # an earlier turn/session can never hijack a
                            # later one. Within the TTL, the entry is
                            # consumed (and deleted) the first time the
                            # PADDING token (== 3) is sampled on that slot —
                            # word tokens (and specials 0/1/2) pass through
                            # untouched while a force is pending, so a
                            # forced ret always lands in silence, never
                            # mid-word. If the model fires NATIVELY while a
                            # force is still pending, the pending entry is
                            # cleared (no rewrite needed — it's already a
                            # ret) and the native fire is TIMESTAMPED (v4:
                            # not flagged — see _LOOKBACK_PARSE), so the
                            # backstop never lands a duplicate ret on the
                            # next padding frame.
                            # Slots this call already logged/stamped via the
                            # branches below — the second (no-pending) loop
                            # must skip them: a forced consumption rewrites
                            # the token TO _r and deletes the slot from _p,
                            # and a native-while-pending fire deletes the
                            # slot from _p too, so without this set BOTH
                            # would also match the second loop's "not in _p
                            # and == _r" test and get mislabeled/double
                            # counted as a second, distinct native fire.
                            _handled = set()
                            for _slot in list(_p.keys()):
                                _deadline = _p[_slot]
                                if _t.monotonic() >= _deadline:
                                    del _p[_slot]
                                elif int(text_token[_slot].item()) == _r:
                                    del _p[_slot]
                                    _s._e2_native_ts[_slot] = _t.monotonic()
                                    _log.info("[Backstop] native fire observed (slot %d)", _slot)
                                    _handled.add(_slot)
                                elif int(text_token[_slot].item()) == 3:
                                    text_token[_slot] = _r
                                    del _p[_slot]
                                    _s._e2_forced_ts[_slot] = _t.monotonic()
                                    _log.info("[Backstop] forced ret consumed (slot %d)", _slot)
                                    _handled.add(_slot)
                            # Native fires with NO pending force in flight —
                            # the common case, including a fire that lands
                            # mid-question, before the "?" that will arm a
                            # timer even arrives — are timestamped on every
                            # slot in the batch, not just slots the pending
                            # dict happens to know about. Slots already
                            # handled above (forced consumption OR
                            # native-while-pending) are skipped — otherwise a
                            # forced rewrite (now == _r, no longer in _p)
                            # would be mislabeled as a SECOND, distinct
                            # native fire, corrupting attribution and
                            # polluting the lookback with a phantom event.
                            for _slot in range(len(text_token)):
                                if _slot not in _p and _slot not in _handled and int(text_token[_slot].item()) == _r:
                                    _s._e2_native_ts[_slot] = _t.monotonic()
                                    _log.info("[Backstop] native fire observed (slot %d)", _slot)
                        _lm_gen.on_text_hook = _e2_hook
                    # A new question always clears any not-yet-consumed force
                    # from a previous question on this slot — this is the
                    # cross-question guard; the hook's own TTL expiry is the
                    # cross-session backstop. v4: this is the ONLY thing
                    # arming clears — native-fire timestamps are never
                    # touched here, so a fire observed moments ago (even
                    # mid-question, before this "?" arrived) still counts at
                    # engage time (FIX 1).
                    self.server._e2_force.pop(self.slot_idx, None)
                    self._backstop_gen = getattr(self, "_backstop_gen", 0) + 1
                    _gen = self._backstop_gen
''' + _DELAY_PARSE + _TTL_PARSE + _LOOKBACK_PARSE + _REFRACTORY_PARSE + _HOLDOFF_PARSE + '''
                    # v4 FIX 2: refractory — if the backstop itself consumed
                    # a forced ret on this slot too recently, skip arming a
                    # new one entirely (native fires are never gated by
                    # this; only our own arming is).
                    _forced_ts = getattr(self.server, "_e2_forced_ts", {}).get(self.slot_idx, 0.0)
                    if _time.monotonic() - self._e2_sess_ts < _holdoff:
                        self._log.info("[Backstop] holdoff skip")
                    elif _time.monotonic() - _forced_ts < _refractory:
                        self._log.info("[Backstop] refractory skip")
                    else:
                        async def _e2_backstop_check(_gen=_gen, _delay=_delay, _ttl=_ttl, _lookback=_lookback):
                            import asyncio as _aio
                            await _aio.sleep(_delay)
                            if self._backstop_gen != _gen:
                                return
                            # v4 FIX 1: engage iff no native fire landed on
                            # this slot within the lookback window — this
                            # covers a fire that happened mid-question,
                            # before arming, which the old flag design lost.
                            _native_ts = getattr(self.server, "_e2_native_ts", {}).get(self.slot_idx, 0.0)
                            if _time.monotonic() - _native_ts <= _lookback:
                                return
                            self._log.info("[Backstop] engaged")
                            self.server._e2_force = getattr(self.server, "_e2_force", {})
                            self.server._e2_force[self.slot_idx] = _time.monotonic() + _ttl

                        self._task_group.create_task(_e2_backstop_check())'''

# Sentinel: declines injecting a reference the generator itself marks as
# unavailable (the templates instruct Gemini to output exactly
# "Reference: no reference available" for live/unknowable data). The guard
# sits right after the received-reference log, before any UI/injection use,
# so a decline reference is logged but never encoded or injected — the
# model's own trained refusal then answers unassisted.
# Anchor recorded from operator grep 2026-08-11 (anchors.md section E):
# channel.py:162-173, the _handle_reference_text signature + docstring +
# preview block through the received-text log line.
SENTINEL_ANCHOR = '''    async def _handle_reference_text(self, reference_text: str | None, lm_label: str = ""):
        """Forward a freshly generated reference text to the UI and the LM.

        Args:
            reference_text: Generated reference text.
            lm_label: Display name of the LLM used for reference generation (sent to client UI).
        """
        if reference_text is None:
            preview, ref_len = "", 0
        else:
            preview, ref_len = reference_text[:120], len(reference_text)
        self._log.info(f"[Reference] received reference text (len={ref_len}) lm={lm_label!r}: '{preview}'")'''

SENTINEL_REPLACEMENT = SENTINEL_ANCHOR + '''
        if reference_text and "no reference available" in reference_text.lower():
            self._log.info("[Sentinel] decline reference — injection skipped")
            return'''


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("apply", "revert"):
        raise SystemExit("usage: e2_forced_ret_v42.py apply|revert")
    if sys.argv[1] == "apply":
        backup = apply_patch(TARGET, ANCHOR, REPLACEMENT, TAG)
        print(f"applied {TAG}; backup at {backup}")
        if SENTINEL_ANCHOR is None:
            print(_SENTINEL_MISSING_MSG)
        else:
            sentinel_backup = apply_patch(TARGET, SENTINEL_ANCHOR, SENTINEL_REPLACEMENT, SENTINEL_TAG)
            print(f"applied {SENTINEL_TAG}; backup at {sentinel_backup}")
    else:
        try:
            revert_patch(TARGET, SENTINEL_TAG)
            print(f"reverted {SENTINEL_TAG}")
        except PatchError as e:
            print(f"skip revert {SENTINEL_TAG}: {e}")
        try:
            revert_patch(TARGET, TAG)
            print(f"reverted {TAG}")
        except PatchError as e:
            print(f"skip revert {TAG}: {e}")


if __name__ == "__main__":
    main()
