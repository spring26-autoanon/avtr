# channel.py anchor discovery (operator greps, 2026-08-11, wb-gpu-a1ultra2g)

Target file: `/home/leenatantawy/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/channel.py`
(stock at discovery time — all prior experimental patches reverted).

## A: switch-to-model site — NOT in channel.py; NOT NEEDED

`grep -n "Switching to model" $IU/*.py` matched nothing: the `[State] Switching to
model` log is a dynamic f-string inside the turn manager. Design consequence
(supersedes the plan's Task 4 assumption): the backstop does NOT hook the turn
switch. It triggers from the STT receive loop instead (anchor D), with a timer —
strictly channel.py-only surgery.

## B: native fire detection — channel.py:372-375 (inside `_output_loop`)

```python
            # Text / RAG triggers.
            if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:
                self._log.info("[RAG] model emitted RAG token, triggering reference generation")
                assert self._task_group is not None
                await self._send_turn_outputs([("[RET]", "model")])
                await self.stt.flush()
                await self.rag_manager.trigger(
                    task_group=self._task_group,
                    wait_steps=self.server.stt_wait_steps,
                    handle_reference_fn=self._handle_reference_text,
                    context_provider=self.turn_manager.get_context,
                )
```

The `fired` flag is set here (backstop patch inserts `self._backstop_fired = True`
after the log line).

## C: the generate-and-inject entry point — `rag_manager.trigger(...)`

Exact call signature (from the native fire path above; `async def trigger(` at
rag_manager.py:106): `await self.rag_manager.trigger(task_group=self._task_group,
wait_steps=..., handle_reference_fn=self._handle_reference_text,
context_provider=self.turn_manager.get_context)`. The backstop calls it with
`wait_steps=0` (the question is already complete when the backstop engages — no
reason to wait further). `trigger` internally creates the background task, builds
context via `context_provider()`, calls Gemini, and routes the reference through
`handle_reference_fn` → ARC encode → streaming_sum injection. Reuse verbatim.

## D: user transcript arrival — channel.py:354-357 (`_stt_recv_loop`)

```python
    async def _stt_recv_loop(self):
        async for msg in self.stt:
            ...
                await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))
```

`msg.text` is the incremental user transcript (word-level; the STT emits
punctuation, so a completed question's final message text ends with `?`).
Backstop insertion point: in this loop, when `msg.text.rstrip().endswith("?")`,
arm a one-shot delayed check (`asyncio.sleep(BACKSTOP_DELAY)`, default 2.5 s to
sit past the stt-wait window) that fires `rag_manager.trigger(...)` iff
`self._backstop_fired` is still False since the question; the flag resets when a
new user question arms the timer.

## Revised E1 patch shape (channel.py only, two anchored replacements)

1. Anchor B edit (tag `e1_flag`): after the `[RAG] model emitted` log line, add
   `self._backstop_fired = True`.
2. Anchor D edit (tag `e1_timer`): wrap the `_stt_recv_loop` body's send with the
   question-detection + delayed-trigger described above, gated by
   `MOSHI_BACKSTOP=1`, logging `[Backstop] engaged` at fire time.

The plan's Task 4 code sketch (turn-switch anchor + `_last_user_text()`) is
superseded by this shape; the `## ANCHORS.md` substitution markers resolve to the
Section B/C/D code above.
