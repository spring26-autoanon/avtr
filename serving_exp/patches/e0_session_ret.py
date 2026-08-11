"""E0: at session start, force ONE synthetic <ret> token (model's own words
otherwise untouched) and inject a pre-seeded reference AFTER the synthetic
fire's empty-context generation has come and gone (delay), so the note stands.

Gated by MOSHI_E0_PRESEED (path to note text file). apply/revert on the BOX.

Usage on box:  python3 e0_session_ret.py apply
               python3 e0_session_ret.py revert
"""
import sys

try:
    from patchlib import apply_patch, revert_patch
except ImportError:  # local test import; box runs from the patches dir
    from serving_exp.patches.patchlib import apply_patch, revert_patch

TARGET = "/home/leenatantawy/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/channel.py"
TAG = "e0_session_ret"

# The TaskGroup opening inside Channel.run() — verified present in the stock
# file on 2026-08-10 (the pre-seed experiment used the same anchor).
ANCHOR = '''            async with asyncio.TaskGroup() as tg:
                self._task_group = tg'''

REPLACEMENT = '''            async with asyncio.TaskGroup() as tg:
                self._task_group = tg
                import os as _os
                _e0_note = _os.environ.get("MOSHI_E0_PRESEED", "")
                if _e0_note:
                    if _os.path.isfile(_e0_note):
                        _e0_note = open(_e0_note).read().strip()
                    _lm_gen = self.server.runner.lm_gen
                    if not hasattr(self.server, "_e0_pending"):
                        self.server._e0_pending = {}
                        _pending = self.server._e0_pending
                        _rag_id = _lm_gen.lm_model.rag_token_id
                        def _e0_hook(text_token, _p=_pending, _r=_rag_id):
                            for _slot in list(_p.keys()):
                                if int(text_token[_slot].item()) > 3:
                                    text_token[_slot] = _r
                                    del _p[_slot]
                        _lm_gen.on_text_hook = _e0_hook
                    self.server._e0_pending[self.slot_idx] = True
                    self._log.info("[E0] armed: one synthetic ret at first word frame")
                    async def _e0_delayed_preseed(_text=_e0_note):
                        import asyncio as _aio
                        await _aio.sleep(6.0)
                        self._log.info(f"[E0] injecting delayed preseed (len={len(_text)})")
                        await self._async_update_reference(_text)
                    tg.create_task(_e0_delayed_preseed())
                tg.create_task(self._recv_loop())'''


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("apply", "revert"):
        raise SystemExit("usage: e0_session_ret.py apply|revert")
    if sys.argv[1] == "apply":
        backup = apply_patch(TARGET, ANCHOR, REPLACEMENT, TAG)
        print(f"applied {TAG}; backup at {backup}")
    else:
        revert_patch(TARGET, TAG)
        print(f"reverted {TAG}")


if __name__ == "__main__":
    main()
