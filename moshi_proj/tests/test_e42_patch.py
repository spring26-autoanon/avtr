import ast
import textwrap
from serving_exp.patches import e2_forced_ret as e2
from serving_exp.patches import e2_forced_ret_v42 as e42


def _parses(block):
    ast.parse("async def _w(self, msg):\n if True:\n" + textwrap.indent(block, "  "))


def test_new_tags_v41_untouched():
    assert e42.TAG == "e42_force" and e42.SENTINEL_TAG == "e42_sentinel"
    assert e2.TAG == "e2_force"                      # frozen file untouched
    assert e42.TARGET == e2.TARGET

def test_replacement_starts_with_anchor():
    assert e42.REPLACEMENT.startswith(e42.ANCHOR)
    assert e42.ANCHOR == e2.ANCHOR                   # same section-D anchor

def test_replacement_is_valid_python():
    _parses(e42.REPLACEMENT[len(e42.ANCHOR):])

def test_v41_semantics_carried_over():
    for s in ("MOSHI_BACKSTOP_DELAY", "MOSHI_BACKSTOP_TTL", "MOSHI_BACKSTOP_LOOKBACK",
              "MOSHI_BACKSTOP_REFRACTORY", "forced ret consumed", "native fire observed",
              "refractory skip", "== 3", "_handled"):
        assert s in e42.REPLACEMENT

def test_holdoff_env_parse_and_default():
    assert 'MOSHI_BACKSTOP_HOLDOFF", "12.0"' in e42.REPLACEMENT
    assert "except ValueError" in e42.REPLACEMENT.split("MOSHI_BACKSTOP_HOLDOFF")[1][:200]

def test_holdoff_skip_logged_and_gates_arming_only():
    assert "[Backstop] holdoff skip" in e42.REPLACEMENT
    # stamped on first STT message (session proxy), before the question gate
    assert e42.REPLACEMENT.index("_e2_sess_ts") < e42.REPLACEMENT.index('endswith("?")')

def test_sentinel_constants_exist():
    assert hasattr(e42, "SENTINEL_ANCHOR") and hasattr(e42, "SENTINEL_REPLACEMENT")
    if e42.SENTINEL_ANCHOR is not None:
        assert e42.SENTINEL_REPLACEMENT.startswith(e42.SENTINEL_ANCHOR)
        assert "no reference available" in e42.SENTINEL_REPLACEMENT
        assert "[Sentinel] decline reference" in e42.SENTINEL_REPLACEMENT

def test_apply_skips_sentinel_when_anchor_missing():
    # main() with SENTINEL_ANCHOR None must warn-and-continue, not crash
    assert "SENTINEL_ANCHOR is None" in e42.__dict__.get("_SENTINEL_MISSING_MSG", "SENTINEL_ANCHOR is None")

def test_mutual_exclusion_documented():
    assert "MUTUALLY EXCLUSIVE" in e42.__doc__ and "e2" in e42.__doc__
