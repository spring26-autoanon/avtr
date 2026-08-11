import ast
import textwrap

from serving_exp.patches import e1_silent_backstop as e1


def test_flag_replacement_contains_anchor_prefix():
    assert e1.REPLACEMENT_FLAG.startswith(e1.ANCHOR_FLAG)


def test_timer_replacement_contains_anchor_prefix():
    assert e1.REPLACEMENT_TIMER.startswith(e1.ANCHOR_TIMER)


def test_flag_replacement_is_valid_python():
    ast.parse(
        "if True:\n" + textwrap.indent(textwrap.dedent(e1.REPLACEMENT_FLAG), "    ")
    )


def test_timer_replacement_is_valid_python():
    ast.parse(
        "if True:\n" + textwrap.indent(textwrap.dedent(e1.REPLACEMENT_TIMER), "    ")
    )


def test_backstop_is_env_gated_and_logged():
    assert "MOSHI_BACKSTOP" in e1.REPLACEMENT_TIMER
    assert "[Backstop] engaged" in e1.REPLACEMENT_TIMER


def test_question_gate_present():
    assert 'endswith("?")' in e1.REPLACEMENT_TIMER


def test_fired_flag_set_in_flag_patch():
    assert "self._backstop_fired = True" in e1.REPLACEMENT_FLAG


def test_apply_simulation_no_duplicate_loops():
    # Stock fire-detection block, verbatim from anchors.md section B.
    stock_fire = (
        "            # Text / RAG triggers.\n"
        "            if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:\n"
        '                self._log.info("[RAG] model emitted RAG token, triggering reference generation")\n'
        "                assert self._task_group is not None\n"
        '                await self._send_turn_outputs([("[RET]", "model")])\n'
        "                await self.stt.flush()\n"
        "                await self.rag_manager.trigger(\n"
        "                    task_group=self._task_group,\n"
        "                    wait_steps=self.server.stt_wait_steps,\n"
        "                    handle_reference_fn=self._handle_reference_text,\n"
        "                    context_provider=self.turn_manager.get_context,\n"
        "                )\n"
    )
    patched_fire = stock_fire.replace(e1.ANCHOR_FLAG, e1.REPLACEMENT_FLAG, 1)
    assert patched_fire != stock_fire  # anchor matched
    assert patched_fire.count("self._backstop_fired = True") == 1
    for line in (
        "if text_token == self.server.runner.lm_gen.lm_model.rag_token_id:",
        '[RAG] model emitted RAG token, triggering reference generation',
        "assert self._task_group is not None",
        'await self._send_turn_outputs([("[RET]", "model")])',
        "await self.stt.flush()",
        "await self.rag_manager.trigger(",
    ):
        assert patched_fire.count(line) == 1

    # Stock _stt_recv_loop block containing anchors.md section D's line.
    stock_stt = (
        "    async def _stt_recv_loop(self):\n"
        "        async for msg in self.stt:\n"
        "            if True:\n"
        "                await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))\n"
    )
    patched_stt = stock_stt.replace(e1.ANCHOR_TIMER, e1.REPLACEMENT_TIMER, 1)
    assert patched_stt != stock_stt  # anchor matched
    assert patched_stt.count("[Backstop] engaged") == 1
    assert patched_stt.count("rag_manager.trigger(") == 1
    assert (
        patched_stt.count(
            "await self._send_turn_outputs(self.turn_manager.handle_spoken_text(user_text=msg.text))"
        )
        == 1
    )
