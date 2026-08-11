"""E1: silent-injection backstop. If the user asks a question and the model
never natively emits a <ret> (RAG) token for it, a delayed timer fires the
same generate-and-inject path a native fire uses, so a dropped retrieval
still gets a chance to land. Gated by MOSHI_BACKSTOP=1. apply/revert on the BOX.

Two anchored replacements (see serving_exp/patches/anchors.md, "Revised E1
patch shape"):
  - e1_flag:  channel.py's native fire-detection site (_output_loop) — records
    that a native fire happened this turn, so the backstop skips it.
  - e1_timer: channel.py's _stt_recv_loop — on a completed question, arms a
    delayed check that engages the backstop if no native fire happened.

Usage on box:  python3 e1_silent_backstop.py apply
               python3 e1_silent_backstop.py revert
"""
import sys

try:
    from patchlib import apply_patch, revert_patch
except ImportError:  # local test import; box runs from the patches dir
    from serving_exp.patches.patchlib import apply_patch, revert_patch

TARGET = "/home/leenatantawy/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/channel.py"

TAG_FLAG = "e1_flag"
TAG_TIMER = "e1_timer"

# anchors.md section B — native fire detection, verbatim (exact indentation).
ANCHOR_FLAG = '''            if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:
                self._log.info("[RAG] model emitted RAG token, triggering reference generation")'''

REPLACEMENT_FLAG = '''            if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:
                self._log.info("[RAG] model emitted RAG token, triggering reference generation")
                self._backstop_fired = True'''

# anchors.md section D — _stt_recv_loop body line, verbatim (exact indentation).
ANCHOR_TIMER = '''                await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))'''

# Parsed EARLY (synchronously, before the check task is created) and guarded —
# a malformed MOSHI_BACKSTOP_DELAY must degrade to the 2.5s default instead of
# raising inside the TaskGroup task, where a ValueError would propagate as an
# ExceptionGroup and tear down every sibling loop (recv/output/stt). Kept as
# its own constant so tests can exercise the fallback in isolation without
# waiting out a real delay.
_DELAY_PARSE = '''\
                    try:
                        _delay = float(_os.environ.get("MOSHI_BACKSTOP_DELAY", "2.5"))
                    except ValueError:
                        _delay = 2.5
'''

REPLACEMENT_TIMER = ANCHOR_TIMER + '''
                import os as _os
                if _os.environ.get("MOSHI_BACKSTOP", "") == "1" and msg.text.rstrip().endswith("?"):
                    self._backstop_fired = False
                    self._backstop_gen = getattr(self, "_backstop_gen", 0) + 1
                    _gen = self._backstop_gen
''' + _DELAY_PARSE + '''
                    async def _e1_backstop_check(_gen=_gen, _delay=_delay):
                        import asyncio as _aio
                        await _aio.sleep(_delay)
                        if self._backstop_gen == _gen and not getattr(self, "_backstop_fired", False):
                            self._log.info("[Backstop] engaged")
                            self._backstop_fired = True
                            # native fires are never suppressed: an emitted ret token must receive a note (fire-freeze otherwise)
                            await self.rag_manager.trigger(
                                task_group=self._task_group,
                                wait_steps=0,
                                handle_reference_fn=self._handle_reference_text,
                                context_provider=self.turn_manager.get_context,
                            )

                    self._task_group.create_task(_e1_backstop_check())'''


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("apply", "revert"):
        raise SystemExit("usage: e1_silent_backstop.py apply|revert")
    if sys.argv[1] == "apply":
        backup_flag = apply_patch(TARGET, ANCHOR_FLAG, REPLACEMENT_FLAG, TAG_FLAG)
        print(f"applied {TAG_FLAG}; backup at {backup_flag}")
        backup_timer = apply_patch(TARGET, ANCHOR_TIMER, REPLACEMENT_TIMER, TAG_TIMER)
        print(f"applied {TAG_TIMER}; backup at {backup_timer}")
    else:
        revert_patch(TARGET, TAG_TIMER)
        print(f"reverted {TAG_TIMER}")
        revert_patch(TARGET, TAG_FLAG)
        print(f"reverted {TAG_FLAG}")


if __name__ == "__main__":
    main()
