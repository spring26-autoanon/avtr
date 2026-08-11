# Backstop Retrieval (streaming_sum) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve-time backstop that guarantees grounding on retrieval-worthy questions using the existing streaming_sum injection path, with an E1–E4 experiment matrix on existing checkpoints that also determines what the single remaining training run must fix.

**Architecture:** All new code lives in `serving_exp/` on branch `serving-experiments`. Box-side surgery on the fork venv's `channel.py` happens only through versioned, anchored patch scripts (apply/revert with backups). Every GPU step is a 🖥️ HANDOFF: the operator (user) runs paste-able blocks on wb-gpu-a1ultra2g and pastes output back; the implementer NEVER runs remote commands.

**Tech Stack:** Python 3.12 stdlib (patch scripts must run on the box with no new deps), pytest (local tests), the existing fork moshi 0.2.13 serving stack, Gemini reference backend, ARC encoder on :8001.

## Global Constraints

- The user runs every GPU/box command; hand paste-able blocks and wait for output (standing project rule).
- Encoder-safe kill only: `pkill -f "moshi.server --hf"` (never bare `moshi.server` — it matches the conditioner).
- No training runs in this plan; exactly one training run remains in project budget and is out of scope here.
- All verdict sessions on a quiet box (no training processes anywhere during E-sessions).
- Every box patch: anchored string replacement, `assert anchor in src`, backup written before write, revert script provided.
- Every session archived to `replay/auditions/logs/` and logged as a row in `serving_exp/results.md`. No unlogged experiments.
- Spec of record: `docs/superpowers/specs/2026-08-11-backstop-retrieval-design.md`.
- Serve configs fixed unless a task says otherwise: rank 64, `CUDA_VISIBLE_DEVICES=0`, port 8998, `REFERENCE_ENCODER_URL=http://localhost:8001`, `source ~/.moshi_env` before launch.

---

### Task 1: Workspace — branch, directories, protocol, results skeleton

**Files:**
- Create: `serving_exp/protocol.md`
- Create: `serving_exp/results.md`
- Create: `serving_exp/patches/.gitkeep`
- Create: `serving_exp/preseed_texts/.gitkeep`

**Interfaces:**
- Produces: branch `serving-experiments`; `serving_exp/results.md` row format used by every later task: `| date | experiment | checkpoint | scaling | stt-wait | patch | native fires | backstops | grounded/asked | notes | log path |`

- [ ] **Step 1: Create the branch**

```bash
cd /Users/leenatantawy/Downloads/capstone/moshi-finetune
git checkout -b serving-experiments
```

- [ ] **Step 2: Create the workspace files**

Write `serving_exp/protocol.md`:

```markdown
# Serving experiments protocol (spec: docs/superpowers/specs/2026-08-11-backstop-retrieval-design.md)

## Standard probe script (say verbatim; question ends the turn; ~5 s gaps)
1. "Hi, how are you?"            (health; never backstopped — not retrieval-worthy)
2. "Are you from Florida originally?"
3. "Where did you grow up?"
4. "What should I call you?"
5. "What do you do for work?"
6. "How many houseplants do you have?"
7. "How many tattoos do you have now?"
8. "What music do you listen to the most?"
9. "Can you tell me about the first World Cup?"
10. "What's the live price of Bitcoin right now?"   (decline probe — E4 focus)
Then 1 casual minute (voice/stalls).

## Per-question verdicts to record
F  = native fire; B = backstop engaged; G = answer grounded (tracks the note);
X  = wrong/confabulated; S = stall/no answer; D = correct decline.

## Session ritual
curl -s localhost:8001/health   -> encoder UP required
fresh private browser window per session; archive serve.log after every session:
cp ~/serve.log ~/moshi-finetune/replay/auditions/logs/<date>_<experiment>_<ckpt>.log
```

Write `serving_exp/results.md`:

```markdown
# Serving experiment results

| date | exp | ckpt | scale | stt-wait | patch | native fires | backstops | grounded/asked | notes | log |
|---|---|---|---|---|---|---|---|---|---|---|
```

- [ ] **Step 3: Commit**

```bash
mkdir -p serving_exp/patches serving_exp/preseed_texts
touch serving_exp/patches/.gitkeep serving_exp/preseed_texts/.gitkeep
git add serving_exp docs/superpowers/plans/2026-08-11-backstop-retrieval.md
git commit -m "serving-exp: workspace, protocol, results skeleton"
```

---

### Task 2: patchlib — anchored apply/revert with backups (TDD)

**Files:**
- Create: `serving_exp/patches/patchlib.py`
- Test: `tests/test_patchlib.py`

**Interfaces:**
- Produces: `apply_patch(path: str, anchor: str, replacement: str, tag: str) -> str` (returns backup path; raises `PatchError` if anchor missing, already applied — detected by `tag_marker(tag)` present — or file unreadable). `revert_patch(path: str, tag: str) -> None` (restores `<path>.bak_<tag>`). `tag_marker(tag: str) -> str` returns `f"# PATCH:{tag}"`; `apply_patch` appends `"  " + tag_marker(tag)` to the first line of `replacement` before writing, so applied-detection is self-contained. Patch scripts in Tasks 4/7 import these.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_patchlib.py
import pytest
from serving_exp.patches.patchlib import apply_patch, revert_patch, tag_marker, PatchError


def _write(tmp_path, text):
    p = tmp_path / "target.py"
    p.write_text(text)
    return str(p)


def test_apply_replaces_anchor_and_writes_backup(tmp_path):
    p = _write(tmp_path, "a = 1\nANCHOR\nb = 2\n")
    backup = apply_patch(p, "ANCHOR", "PATCHED", tag="e1")
    text = open(p).read()
    assert "PATCHED" in text and "ANCHOR" not in text
    assert tag_marker("e1") in text
    assert open(backup).read() == "a = 1\nANCHOR\nb = 2\n"


def test_apply_missing_anchor_raises_and_leaves_file(tmp_path):
    p = _write(tmp_path, "no anchor here\n")
    with pytest.raises(PatchError):
        apply_patch(p, "ANCHOR", "PATCHED", tag="e1")
    assert open(p).read() == "no anchor here\n"


def test_apply_twice_raises(tmp_path):
    p = _write(tmp_path, "ANCHOR\n")
    apply_patch(p, "ANCHOR", "PATCHED", tag="e1")
    with pytest.raises(PatchError):
        apply_patch(p, "PATCHED", "AGAIN", tag="e1")


def test_revert_restores_original(tmp_path):
    p = _write(tmp_path, "ANCHOR\n")
    apply_patch(p, "ANCHOR", "PATCHED", tag="e1")
    revert_patch(p, tag="e1")
    assert open(p).read() == "ANCHOR\n"


def test_revert_without_backup_raises(tmp_path):
    p = _write(tmp_path, "x\n")
    with pytest.raises(PatchError):
        revert_patch(p, tag="nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_patchlib.py -v`
Expected: FAIL / error with "No module named 'serving_exp'" (add `serving_exp/__init__.py` and `serving_exp/patches/__init__.py` in Step 3 so imports resolve).

- [ ] **Step 3: Implement**

```python
# serving_exp/patches/patchlib.py
"""Anchored, revertible surgery for box-side files (stdlib only — runs on the GPU box)."""
import shutil


class PatchError(RuntimeError):
    pass


def tag_marker(tag: str) -> str:
    return f"# PATCH:{tag}"


def apply_patch(path: str, anchor: str, replacement: str, tag: str) -> str:
    try:
        src = open(path).read()
    except OSError as e:
        raise PatchError(f"cannot read {path}: {e}") from e
    if tag_marker(tag) in src:
        raise PatchError(f"{path} already carries patch {tag!r}")
    if anchor not in src:
        raise PatchError(f"anchor not found in {path} — file drifted from expected")
    backup = f"{path}.bak_{tag}"
    shutil.copyfile(path, backup)
    first, sep, rest = replacement.partition("\n")
    marked = first + "  " + tag_marker(tag) + sep + rest
    open(path, "w").write(src.replace(anchor, marked, 1))
    return backup


def revert_patch(path: str, tag: str) -> None:
    backup = f"{path}.bak_{tag}"
    try:
        shutil.copyfile(backup, path)
    except OSError as e:
        raise PatchError(f"no backup for tag {tag!r} at {backup}: {e}") from e
```

Also create empty `serving_exp/__init__.py` and `serving_exp/patches/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_patchlib.py -v`
Expected: 5 passed. Then run the full suite (`python3 -m pytest tests/ -q`) — no regressions.

- [ ] **Step 5: Commit**

```bash
git add serving_exp/__init__.py serving_exp/patches/__init__.py serving_exp/patches/patchlib.py tests/test_patchlib.py
git commit -m "serving-exp: patchlib — anchored apply/revert with backups (TDD)"
```

---

### Task 3: 🖥️ HANDOFF — anchor discovery on the box's channel.py

**Files:**
- Create: `serving_exp/patches/anchors.md`

**Interfaces:**
- Produces: `anchors.md` recording, verbatim from the box: (A) the switch-to-model code block, (B) how the channel detects a ⟨ret⟩ emission, (C) the name/signature of the method the fire path calls to generate a reference (the "Triggering retrieval" path), (D) where the latest user transcript text is accessible on the channel object. Tasks 4 and 7 substitute these exact names into their patch code.

- [ ] **Step 1: Hand the operator this block and WAIT for pasted output**

```bash
CH=~/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/channel.py
grep -n "Switching to model" $CH
grep -n -B3 -A12 "Switching to model" $CH | head -40
grep -n "rag_token\|\[RET\]\|emitted RAG" $CH ~/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/*.py | head -10
grep -n -B3 -A15 "Triggering retrieval" ~/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/*.py | head -50
grep -n "stt\|transcript\|user_text" $CH | head -15
```

- [ ] **Step 2: Record findings**

Write `serving_exp/patches/anchors.md` with four sections (A–D above), each containing the pasted code verbatim plus one line naming the exact symbol Task 4 must call (e.g., "C: fire path calls `self.rag_manager.trigger_retrieval(...)` at rag_manager.py:NN — backstop calls the same method"). If (C) turns out to live outside `channel.py`, record the import path.

- [ ] **Step 3: Commit**

```bash
git add serving_exp/patches/anchors.md
git commit -m "serving-exp: channel.py anchor discovery for backstop patch"
```

---

### Task 4: E1 patch — silent injection backstop (no ⟨ret⟩ anywhere)

**Files:**
- Create: `serving_exp/patches/e1_silent_backstop.py`
- Test: `tests/test_e1_patch.py`

**Interfaces:**
- Consumes: `patchlib.apply_patch/revert_patch`; the four anchors from `serving_exp/patches/anchors.md` (substitute the two symbols marked `## ANCHORS.md` below with the discovered names before running Task 5).
- Produces: a CLI (`python3 e1_silent_backstop.py apply|revert`) run ON THE BOX; behavior gated by `MOSHI_BACKSTOP=1`; log signature `[Backstop] engaged` consumed by Task 6's scorer.

- [ ] **Step 1: Write the failing test (patch self-consistency)**

```python
# tests/test_e1_patch.py
import ast
import textwrap
from serving_exp.patches import e1_silent_backstop as e1


def test_replacement_contains_anchor_prefix():
    # replacement must begin with the anchor so the original behavior is preserved
    assert e1.REPLACEMENT.startswith(e1.ANCHOR)


def test_replacement_is_valid_python():
    ast.parse(textwrap.dedent(e1.REPLACEMENT))


def test_backstop_is_env_gated_and_logged():
    assert 'MOSHI_BACKSTOP' in e1.REPLACEMENT
    assert '[Backstop] engaged' in e1.REPLACEMENT


def test_question_gate_present():
    assert 'endswith("?")' in e1.REPLACEMENT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_e1_patch.py -v`
Expected: FAIL with "No module named ... e1_silent_backstop".

- [ ] **Step 3: Implement the patch script**

The ANCHOR below is the switch-to-model block recorded in anchors.md section A; the
two symbols marked `## ANCHORS.md` are placeholders for the *discovered* names from
sections C and D — substitute them from anchors.md (they are the only permitted
edits to this code). Logic: on switching to model, if the backstop is enabled, the
turn had no native fire, and the just-finished user text ends with "?", schedule the
same generate-and-inject path a native fire uses.

```python
# serving_exp/patches/e1_silent_backstop.py
"""E1: silent-injection backstop. apply/revert on the BOX (stdlib only).

Usage on box:  python3 e1_silent_backstop.py apply
               python3 e1_silent_backstop.py revert
"""
import sys
from patchlib import apply_patch, revert_patch

TARGET = "/home/leenatantawy/fork-venv/lib/python3.12/site-packages/moshi/inference_utils/channel.py"
TAG = "e1_backstop"

# Section A of anchors.md — the exact switch-to-model block, verbatim.
ANCHOR = '''self._log.info("[State] Switching to model")'''

REPLACEMENT = '''self._log.info("[State] Switching to model")
                import os as _os
                if _os.environ.get("MOSHI_BACKSTOP", "") == "1":
                    _fired = getattr(self, "_backstop_fired_this_turn", False)
                    _last_user = self._last_user_text()          ## ANCHORS.md D: expression yielding latest user utterance text
                    if (not _fired) and _last_user.strip().endswith("?"):
                        self._log.info("[Backstop] engaged")
                        self._task_group.create_task(
                            self.rag_manager.trigger_retrieval()  ## ANCHORS.md C: the fire path's generate+inject entry point
                        )
                    self._backstop_fired_this_turn = False'''


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("apply", "revert"):
        raise SystemExit("usage: e1_silent_backstop.py apply|revert")
    if sys.argv[1] == "apply":
        backup = apply_patch(TARGET, ANCHOR, REPLACEMENT, TAG)
        print(f"applied {TAG}; backup at {backup}")
    else:
        revert_patch(TARGET, TAG)
        print(f"reverted {TAG}")


if __name__ == "__main__":
    main()
```

Note for the implementer: `_backstop_fired_this_turn` must be SET where the channel
detects a native ⟨ret⟩ (anchors.md section B). If section B's detection site is in
`channel.py`, add a second anchor/replacement pair to this same script (same
apply/revert calls, tag `e1_backstop_b`) that inserts
`self._backstop_fired_this_turn = True` at that site; if detection lives in another
file, patch that file with the same mechanism. anchors.md governs.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_e1_patch.py -v`
Expected: 4 passed. (`test_replacement_is_valid_python` operates on the dedented block; if the discovered anchor's indentation makes dedent invalid, wrap the parse target as `"if True:\n" + textwrap.indent(...)` in the test — adjust the test, not the patch.)

- [ ] **Step 5: Commit**

```bash
git add serving_exp/patches/e1_silent_backstop.py tests/test_e1_patch.py
git commit -m "serving-exp: E1 silent-injection backstop patch (anchored, revertible)"
```

---

### Task 5: 🖥️ HANDOFF — E1 session on 4c2-700 and the adjacency verdict

**Files:**
- Modify: `serving_exp/results.md` (one row)
- Modify: `serving_exp/patches/anchors.md` (only if the box apply reveals drift)

**Interfaces:**
- Consumes: Task 4's patch script; Task 1's probe script.
- Produces: the E1 verdict recorded in results.md notes: `ADJACENCY-GATED` (notes used without ⟨ret⟩ → skip Task 7) or `TOKEN-GATED` (notes ignored → Task 7 required).

- [ ] **Step 1: Ship and apply (operator block; wait for output)**

```bash
rsync -avP serving_exp/patches/patchlib.py serving_exp/patches/e1_silent_backstop.py wb-gpu-a1ultra2g:~/patches/
```

```bash
cd ~/patches && python3 e1_silent_backstop.py apply
```

Expected: `applied e1_backstop; backup at ...`. On `PatchError: anchor not found`, run Task 3's greps again, update anchors.md and the patch, re-ship.

- [ ] **Step 2: Serve 4c2-700 with backstop armed (operator block)**

```bash
curl -s localhost:8001/health && echo " encoder UP" || echo "ENCODER DOWN - restart first"
pkill -f "moshi.server --hf"; sleep 3; fuser -k 8998/tcp 2>/dev/null; sleep 2
source ~/.moshi_env
export REFERENCE_ENCODER_URL=http://localhost:8001
export MOSHI_BACKSTOP=1
MOSHI_LORA_SCALING=2.0 MOSHI_LORA_RANK=64 \
MOSHI_LORA_WEIGHT=~/moshi-finetune/runs/moshika_rag_stage4c2/checkpoints/checkpoint_000700/consolidated/lora.safetensors \
CUDA_VISIBLE_DEVICES=0 setsid nohup ~/fork-venv/bin/python -m moshi.server \
  --hf-repo kyutai/moshika-rag-pytorch-bf16 --stt-wait-time 2.0 --port 8998 > ~/serve.log 2>&1 &
sleep 120 && tail -2 ~/serve.log
```

Watcher: `tail -f ~/serve.log | grep --line-buffered -E "Backstop|RAG token|Generated reference|snippet="`

- [ ] **Step 3: Run the standard probe script** (protocol.md), operator marks F/B/G/X/S/D per question. Archive: `cp ~/serve.log ~/moshi-finetune/replay/auditions/logs/2026-08-11_E1_4c2-700.log`

- [ ] **Step 4: Record the verdict**

Add the results.md row. Decision rule from the spec: notes used (G) on ≥ half of
backstopped (B) turns → write `ADJACENCY-GATED; E2 skipped` and mark Task 7 skipped.
Otherwise write `TOKEN-GATED; E2 required`.

- [ ] **Step 5: Commit**

```bash
git add serving_exp/results.md serving_exp/patches/anchors.md
git commit -m "serving-exp: E1 verdict on 4c2-700"
```

---

### Task 6: Backstop metric in the audition scorer (TDD)

**Files:**
- Modify: `scripts/audition_checkpoint.py` (the serve.log parser)
- Test: `tests/test_audition_checkpoint.py` (append two tests)

**Interfaces:**
- Consumes: log lines of the form `HH:MM:SS.ff INFO    [slot 0] [Backstop] engaged` (Task 4's signature; the parser must match on the substring `[Backstop] engaged` regardless of prefix).
- Produces: scorecard fields `backstops: int` and `backstop_rate: float | None` (backstops ÷ (native fires + backstops); `None` when denominator is 0), printed alongside the existing fire/stall counts.

- [ ] **Step 1: Write the failing tests** (follow the existing test file's fixture style — it builds log text inline and calls the existing parse entry point; match whatever that function is named in the file, alongside the existing 10+ tests)

```python
def test_backstop_lines_counted():
    log = (
        "12:00:01.00 INFO    [slot 0] [RAG] model emitted RAG token, triggering reference generation\n"
        "12:00:05.00 INFO    [slot 0] [Backstop] engaged\n"
        "12:00:09.00 INFO    [slot 0] [Backstop] engaged\n"
    )
    result = parse_log_text(log)  # use the module's existing parse entry point
    assert result["backstops"] == 2
    assert result["backstop_rate"] == 2 / 3


def test_backstop_rate_none_when_no_events():
    result = parse_log_text("12:00:01.00 INFO    nothing relevant\n")
    assert result["backstops"] == 0
    assert result["backstop_rate"] is None
```

- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest tests/test_audition_checkpoint.py -v` → the two new tests FAIL (KeyError `backstops`).

- [ ] **Step 3: Implement** — in the parser, count occurrences of the substring `[Backstop] engaged`; compute `backstop_rate = backstops / (fires + backstops)` when the denominator is positive else `None`; add both to the returned dict and to the human-readable scorecard print.

- [ ] **Step 4: Run the full suite** — `python3 -m pytest tests/ -q` → all pass, no regressions.

- [ ] **Step 5: Commit**

```bash
git add scripts/audition_checkpoint.py tests/test_audition_checkpoint.py
git commit -m "serving-exp: backstop count + rate in audition scorer"
```

---

### Task 7: E2 patch — single forced ⟨ret⟩ at the boundary (CONDITIONAL: only if Task 5 verdict is TOKEN-GATED)

**Files:**
- Create: `serving_exp/patches/e2_forced_ret.py`
- Test: `tests/test_e2_patch.py`
- Modify: `serving_exp/results.md` (one row, via its 🖥️ session)

**Interfaces:**
- Consumes: patchlib; anchors.md sections A/B; the known-unused `on_text_hook` on `lm_gen` (`self.server.runner.lm_gen.on_text_hook`, receives the per-frame batch text-token tensor `[B]`; in-place mutation propagates to depformer audio and cache — established in the 2026-08-10 forcing experiment).
- Produces: env-gated (`MOSHI_BACKSTOP_FORCE=1`) forced single ⟨ret⟩; same `[Backstop] engaged` log signature (Task 6's metric needs no change).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e2_patch.py
import ast, textwrap
from serving_exp.patches import e2_forced_ret as e2


def test_replacement_contains_anchor_prefix():
    assert e2.REPLACEMENT.startswith(e2.ANCHOR)


def test_single_token_only():
    # the queue must contain exactly the rag token — no greeting text, no extras
    assert e2.REPLACEMENT.count("rag_token_id") >= 1
    assert "encode(" not in e2.REPLACEMENT


def test_env_gated_and_logged():
    assert "MOSHI_BACKSTOP_FORCE" in e2.REPLACEMENT
    assert "[Backstop] engaged" in e2.REPLACEMENT
```

- [ ] **Step 2: Run to verify it fails** — `python3 -m pytest tests/test_e2_patch.py -v` → module missing.

- [ ] **Step 3: Implement** — same script shape as Task 4 (TARGET/TAG `e2_force`/ANCHOR from anchors.md section A/apply-revert CLI). REPLACEMENT: at switch-to-model on an unfired question turn (same gates as E1, env `MOSHI_BACKSTOP_FORCE`), log `[Backstop] engaged`, then install a one-shot forcing state consumed by an `on_text_hook` (installed once per server, same lazy-install pattern as the 2026-08-10 experiment): on the next frame where this slot's sampled token is a word token (`int(tok) > 3`), overwrite it with `self.server.runner.lm_gen.lm_model.rag_token_id` and clear the one-shot. The natural fire machinery then reacts to the emitted token (real context, real Gemini call) — no further patch code needed downstream.

- [ ] **Step 4: Run tests** — 3 passed; full suite green.

- [ ] **Step 5: Commit**

```bash
git add serving_exp/patches/e2_forced_ret.py tests/test_e2_patch.py
git commit -m "serving-exp: E2 forced single-ret backstop patch"
```

- [ ] **Step 6: 🖥️ HANDOFF session** — operator reverts E1 (`python3 e1_silent_backstop.py revert`), applies E2, serves 4c2-700 with `MOSHI_BACKSTOP_FORCE=1` (same serve block as Task 5 with the env swapped), runs the standard probes. **Audio gate first**: if the forced token audibly garbles her speech on the first backstopped turn, stop the session, record `FORCING AUDIO-BROKEN` in results.md — the spec's E2 failure branch (final training run must include ambient-reference clips). Otherwise complete probes, archive log, add results row, commit.

---

### Task 8: 🖥️ HANDOFF — E3 cross-checkpoint sweep of the winning mechanism

**Files:**
- Modify: `serving_exp/results.md` (three rows)

**Interfaces:**
- Consumes: the winning patch from Task 5/7 (applied on box); Task 6's scorer for post-session metrics.
- Produces: backstop-rate + interference data across the fire-propensity spectrum; the non-interference success criterion verdict.

- [ ] **Step 1: Session A — 4b-300 (fire-rich control).** Same serve block as Task 5 but `MOSHI_LORA_RANK=32` and `MOSHI_LORA_WEIGHT=~/moshi-finetune/runs/moshika_rag_stage4b/checkpoints/checkpoint_000300/consolidated/lora.safetensors`, stt-wait 2.0. Standard probes. The question this session answers: does the backstop suppress or duplicate native fires? (Success criterion 2: native fire count comparable to its known ~14-fires/5-min baseline.)
- [ ] **Step 2: Session B — 4c2-400.** Serve block with `checkpoint_000400`, rank 64, stt-wait 0.5 (its locked config). Standard probes.
- [ ] **Step 3: Session C — Stage 3 (bonus, only if its checkpoints are found).** Operator locates via `ls ~/moshi-finetune/runs/ | head` and `ls runs_backup/` on the Mac; if found, serve with its consolidated lora at rank 64 scaling 2.0, sessions capped at 90 seconds (its known 99 s ceiling), abbreviated probes (questions 2, 5, 6, 9 only). If not found in 10 minutes, record `Stage3: checkpoints not located; skipped` — do not hunt further.
- [ ] **Step 4: Run the scorer on each archived log** (`python3 scripts/audition_checkpoint.py --log <path>`), record three results.md rows including `backstop_rate`, commit:

```bash
git add serving_exp/results.md
git commit -m "serving-exp: E3 sweep — backstop across fire-propensity spectrum"
```

---

### Task 9: 🖥️ HANDOFF — E4 decline probe + sentinel mitigation

**Files:**
- Modify: `serving_exp/results.md` (one row)
- Modify (box, revertible): both reference templates via a patch script created here: `serving_exp/patches/e4_decline_sentinel.py`
- Test: `tests/test_e4_patch.py`

**Interfaces:**
- Consumes: winning backstop patch (armed); patchlib.
- Produces: verdict on success criterion 3 (declines survive) and the classifier go/no-go recorded in results.md.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_e4_patch.py
from serving_exp.patches import e4_decline_sentinel as e4


def test_sentinel_line_and_anchor():
    assert e4.SENTINEL in e4.REPLACEMENT
    assert e4.REPLACEMENT.startswith(e4.ANCHOR)


def test_sentinel_text_exact():
    assert e4.SENTINEL == "If the question asks for live, real-time, or unknowable data, output exactly: Reference: no reference available"
```

- [ ] **Step 2: Run to verify it fails**, then **Step 3: Implement** `e4_decline_sentinel.py`: patchlib-based apply/revert on BOTH template files (`~/fork-venv/lib/python3.12/site-packages/moshi/llm/reference_prompt_template.txt` and `..._simplified.txt`), ANCHOR = the shared rules line `- Simple punctuation only.` (present in both — verified 2026-08-10), REPLACEMENT = that line plus a new rule line containing SENTINEL. Plus, in the E1/E2 backstop replacement's injection path, the server-side half is behavioral, not patched: a reference whose text contains `no reference available` still injects (this is the *measured* condition — E4 part 1) — the skip-on-sentinel server change is only built if part 1 shows flattening (recorded as a follow-up, not in this plan).
- [ ] **Step 4: Tests pass; commit** (`git commit -m "serving-exp: E4 decline sentinel template patch"`).
- [ ] **Step 5: 🖥️ Session (two parts):** operator applies e4 patch, restarts server (template loads at startup), runs decline probes (protocol question 10 plus: "What's the exact weight of the moon?", "What's the score of the Tampa Bay game right now?") twice — once with backstop armed, once with `MOSHI_BACKSTOP` unset (control). Verdicts per question: D (declined) vs X (attempted answer). Record row + the classifier decision per spec ("declines survive → classifier deferred permanently"). Revert e4 patch if the sentinel confused non-decline references. Commit.

---

### Task 10: Synthesis — results, spec addendum, training-run recommendation

**Files:**
- Modify: `serving_exp/results.md` (summary section)
- Modify: `docs/superpowers/specs/2026-08-11-backstop-retrieval-design.md` (append "## Outcomes" section)

**Interfaces:**
- Consumes: every results.md row from Tasks 5–9.
- Produces: the final written recommendation for the single remaining training run (one of the three branches in the spec's "Implications" section, now selected by evidence).

- [ ] **Step 1: Write the summary** at the bottom of results.md: per-checkpoint backstop rates, the E1/E2 gating verdict, interference verdict, decline verdict, against the spec's four success criteria (state pass/fail for each with the measured number).
- [ ] **Step 2: Append "## Outcomes"** to the spec: 5–10 lines — which mechanism won, which success criteria passed, and the selected training-run branch with its success metric ("backstop rate on the E3 protocol driven from X% to target ≤Y%").
- [ ] **Step 3: Commit**

```bash
git add serving_exp/results.md docs/superpowers/specs/2026-08-11-backstop-retrieval-design.md
git commit -m "serving-exp: E1-E4 synthesis + final training run recommendation"
```

---

## Self-review notes

- Spec coverage: workspace → T1; patch tooling discipline → T2; E1 → T4+T5; E2 → T7 (conditional, matching spec); E3 → T8; E4 → T9; metric → T6; training implications → T10. Spec's "preseed_texts/" exists in T1 for future note-variant tests (none required by E1–E4; kept as workspace only).
- The two `## ANCHORS.md` symbols in Task 4 are the only deliberately deferred names in the plan; they are the *output* of Task 3 and cannot be known before the box grep — the plan makes their substitution an explicit, bounded step rather than a placeholder.
- Type consistency: `[Backstop] engaged` signature identical in T4, T6, T7; `backstops`/`backstop_rate` names identical in T6 and T8; patchlib API identical in T2, T4, T7, T9.
