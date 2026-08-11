import ast
import textwrap
from serving_exp.patches import e0_session_ret as e0


def test_replacement_contains_anchor_prefix():
    assert e0.REPLACEMENT.startswith(e0.ANCHOR)


def test_replacement_is_valid_python():
    ast.parse("if True:\n" + textwrap.indent(textwrap.dedent(e0.REPLACEMENT), "    "))


def test_single_ret_only_no_greeting_forcing():
    assert "rag_token_id" in e0.REPLACEMENT
    assert "encode(" not in e0.REPLACEMENT  # no scripted text tokens


def test_env_gated_delayed_and_logged():
    assert "MOSHI_E0_PRESEED" in e0.REPLACEMENT
    assert "sleep(6.0)" in e0.REPLACEMENT
    assert "[E0] armed" in e0.REPLACEMENT and "[E0] injecting delayed preseed" in e0.REPLACEMENT
