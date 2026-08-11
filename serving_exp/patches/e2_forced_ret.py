"""E2: forced single-<ret> backstop. If the user asks a question and the
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

Usage on box:  python3 e2_forced_ret.py apply
               python3 e2_forced_ret.py revert
"""
import sys

try:
    from patchlib import apply_patch, revert_patch
except ImportError:  # local test import; box runs from the patches dir
    from serving_exp.patches.patchlib import apply_patch, revert_patch

TARGET = "/home/leenatantawy/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/channel.py"

TAG = "e2_force"

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

REPLACEMENT = ANCHOR + '''
                import os as _os
                import time as _time
                if _os.environ.get("MOSHI_BACKSTOP_FORCE", "") == "1" and msg.text.rstrip().endswith("?"):
                    if not hasattr(self.server, "_e2_force"):
                        self.server._e2_force = {}
                        self.server._e2_native_fired = {}
                        _pending = self.server._e2_force
                        _native = self.server._e2_native_fired
                        _lm_gen = self.server.runner.lm_gen
                        _rag_id = _lm_gen.lm_model.rag_token_id
                        def _e2_hook(text_token, _p=_pending, _n=_native, _r=_rag_id, _t=_time):
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
                            # mid-word.
                            for _slot in list(_p.keys()):
                                _deadline = _p[_slot]
                                if _t.monotonic() >= _deadline:
                                    del _p[_slot]
                                elif int(text_token[_slot].item()) == 3:
                                    text_token[_slot] = _r
                                    del _p[_slot]
                            for _slot in list(_n.keys()):
                                if _slot not in _p and int(text_token[_slot].item()) == _r:
                                    _n[_slot] = True
                        _lm_gen.on_text_hook = _e2_hook
                    # A new question always clears any not-yet-consumed force
                    # from a previous question on this slot — this is the
                    # cross-question guard; the hook's own TTL expiry is the
                    # cross-session backstop.
                    self.server._e2_force.pop(self.slot_idx, None)
                    self.server._e2_native_fired[self.slot_idx] = False
                    self._backstop_gen = getattr(self, "_backstop_gen", 0) + 1
                    _gen = self._backstop_gen
''' + _DELAY_PARSE + _TTL_PARSE + '''
                    async def _e2_backstop_check(_gen=_gen, _delay=_delay, _ttl=_ttl):
                        import asyncio as _aio
                        await _aio.sleep(_delay)
                        if self._backstop_gen == _gen and not getattr(self.server, "_e2_native_fired", {}).pop(self.slot_idx, False):
                            self._log.info("[Backstop] engaged")
                            self.server._e2_force = getattr(self.server, "_e2_force", {})
                            self.server._e2_force[self.slot_idx] = _time.monotonic() + _ttl

                    self._task_group.create_task(_e2_backstop_check())'''


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("apply", "revert"):
        raise SystemExit("usage: e2_forced_ret.py apply|revert")
    if sys.argv[1] == "apply":
        backup = apply_patch(TARGET, ANCHOR, REPLACEMENT, TAG)
        print(f"applied {TAG}; backup at {backup}")
    else:
        revert_patch(TARGET, TAG)
        print(f"reverted {TAG}")


if __name__ == "__main__":
    main()
