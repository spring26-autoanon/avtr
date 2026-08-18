# Demo Sprint Implementation Plan (Thu 2026-08-07 → Sun 2026-08-10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two live demos Sunday — Track A (plain moshika + voice LoRA) and Track B (moshika-rag + voice LoRA with working retrieval) — with locked checkpoints, known-good question scripts, backup videos, and warm GPUs.

**Architecture:** This is the sprint umbrella. Tonight's data work is fully specified in `docs/superpowers/plans/2026-08-07-tonight-recording-pipeline.md` (Tasks 1–5 there); this plan sequences it, adds the one new code component (the audition harness), and carries the runbooks for provisioning, launches, Friday audition, Saturday serving optimization, and Sunday. Spec: `docs/superpowers/specs/2026-08-07-demo-sprint-design.md`.

**Tech Stack:** Python 3 (Mac system `python3`, box `~/fork-venv/bin/python` on wb-gpu-a1ultra2g, `uv` on wb-gpu-training), pytest, whisper_timestamped, Gemini `gemini-flash-latest`, jupyter keep-alive on both boxes.

## Global Constraints

- **The user runs every GPU command.** Steps marked 🖥️ **HANDOFF** stop, print the exact block, and wait for the user's pasted output. Never ssh, never launch training/annotate/precompute/servers.
- Boxes: `wb-gpu-training` (1×A100-80GB, uses `uv run`), `wb-gpu-a1ultra2g` (2×A100-80GB, **no repo .venv — always `~/fork-venv/bin/python`, never `uv run`**). Box repo path `/home/leenatantawy/moshi-finetune`.
- Both boxes are stockout-prone: long jobs run inside the user's **jupyter keep-alive** pattern (the Vertex AI 180-min idle timer watches Jupyter kernels, not terminals). Boxes stay up from tonight through Sunday; checkpoints rsync to the Mac as they land.
- ch0 = Danielle = `SPEAKER_MAIN`; manifest paths absolute; `finetune/data/prepared/` never appears in any manifest; `source .env` before Gemini calls; model `gemini-flash-latest`.
- Track B launch env flags **all default OFF**: `RAG_TOKEN_WEIGHT=25 MASK_UNLABELLED_RAG=1 RAG_DELAY=1 RAG_REF_DROPOUT=0.2`. Check the launch line before walking away.
- Tonight's recording is used **raw** — reject edited files (spec "Recording intake").
- `python3 -m pytest tests/ -q` green after every code task (171 now; counts below track additions).
- Serve-time rules: raw `lora.safetensors` (never padded), `REFERENCE_ENCODER_URL=http://localhost:8001` for Track B, `MOSHI_LORA_SCALING=2.0` default, local DSM STT (no `--gradium-stt`), kill the server between checkpoints, private browser window.

---

## Task 1: Persona guard (blocker for tonight's Pass B)

**Files:** exactly as written in `docs/superpowers/plans/2026-08-07-tonight-recording-pipeline.md` **Task 1** (test code and edits are fully specified there — execute that task verbatim).

**Interfaces:**
- Produces: kind `"persona"` recognised by `normalize`, counted by `turn_mix`, excluded from `retrieval_share`, never marked by `cut_retrieval_clips`.

- [ ] **Step 1:** Execute tonight-plan Task 1 Steps 1–7 (write `tests/test_persona_guard.py`, verify fail, prompt + `normalize` + `turn_mix` changes, verify 9 pass, full suite 180).
- [ ] **Step 2:** Commit: `git add scripts/segment_retrieval_audio.py tests/test_persona_guard.py && git commit -m "segmenter: persona kind so identity answers never carry a <RAG> marker"`

## Task 2: Audition protocol doc + session template

**Files:**
- Create: `docs/audition_protocol.md`
- Create: `replay/auditions/session_template.json`

**Interfaces:**
- Produces: the fixed question list (ids below, used verbatim by Task 3's `QUESTIONS`), and the session JSON schema consumed by `scripts/audition_checkpoint.py`:
  `{"checkpoint": str, "scaling": float, "session": int, "voice": int 1-5, "interrupt_yields": bool|null, "results": {qid: {"fired": bool|null, "passed": bool, "notes": str}}}`

- [ ] **Step 1: Write `docs/audition_protocol.md`** with exactly this content:

```markdown
# Audition protocol — same script every time, 3 sessions per checkpoint

Triggering is stochastic; one session proves nothing (the false "checkpoint 500
regression" came from single-session judging). Ask with THESE phrasings, same mic,
quiet room — trigger rate tracks user-speech intelligibility (spec F2).

| id | ask exactly | kind | passed means |
|---|---|---|---|
| q01 | "Hi, how are you?" | greeting | responds promptly, hands it back |
| q02 | "What's your name?" | persona | says Danielle; no ⟨ret⟩ |
| q03 | "What do you do for work?" | persona | Director of Digital Platforms / legal |
| q04 | "Can you tell me about the World Cup?" | factual | fires + answer matches reference |
| q05 | "Who won the first World Cup?" | factual | fires + Uruguay 1930 |
| q06 | "What's the capital of Australia?" | factual | fires + Canberra |
| q07 | "What's the weather in Dublin right now?" | decline | declines gracefully, keeps talking |
| q08 | "Do you like playing soccer?" | personal | answers as herself; no ⟨ret⟩ |
| q09 | "Did you ever do martial arts?" | persona | tae kwon do / wrestling content |
| q10 | "Do you have any plants?" | persona | 60+ / monstera content |
| q11 | "Do you have tattoos?" | persona | sleeve / travel-tattoo content |
| q12 | "Where are you from?" | persona | Brooklyn → Tampa |
| q13 | re-ask q05, interrupt mid-answer ("wait, which year?") | barge-in | yields, then picks back up → session `interrupt_yields` |

Per session: copy `replay/auditions/session_template.json` →
`replay/auditions/<checkpoint>_s<N>.json`, fill `fired`/`passed`/`notes` per question,
session `voice` 1–5 (does it sound like HER — filler, warmth, timing), `interrupt_yields`.
Keep the serve.log (`~/serve.log`) per session; the scorer reads it.

Score: `python3 scripts/audition_checkpoint.py --log serve.log \
  --sessions replay/auditions/<ckpt>_s1.json <ckpt>_s2.json <ckpt>_s3.json \
  --out-dir replay/auditions/<ckpt>/`

Verdict weighting (decision of record): stalls + voice/persona FIRST, trigger rate second.
Never trade criteria 3/7 (lead, sounds-like-her) for 1/2 (triggering) — scaling 1.5 made
that trade and it was wrong.
```

- [ ] **Step 2: Write `replay/auditions/session_template.json`:**

```json
{
  "checkpoint": "FILL_run_ckpt",
  "scaling": 2.0,
  "session": 1,
  "voice": 0,
  "interrupt_yields": null,
  "results": {
    "q01": {"fired": null, "passed": false, "notes": ""},
    "q02": {"fired": null, "passed": false, "notes": ""},
    "q03": {"fired": null, "passed": false, "notes": ""},
    "q04": {"fired": null, "passed": false, "notes": ""},
    "q05": {"fired": null, "passed": false, "notes": ""},
    "q06": {"fired": null, "passed": false, "notes": ""},
    "q07": {"fired": null, "passed": false, "notes": ""},
    "q08": {"fired": null, "passed": false, "notes": ""},
    "q09": {"fired": null, "passed": false, "notes": ""},
    "q10": {"fired": null, "passed": false, "notes": ""},
    "q11": {"fired": null, "passed": false, "notes": ""},
    "q12": {"fired": null, "passed": false, "notes": ""}
  }
}
```

- [ ] **Step 3: Commit:** `git add docs/audition_protocol.md replay/auditions/session_template.json && git commit -m "audition: fixed 13-question protocol + session template"`

## Task 3: Audition scorer — `scripts/audition_checkpoint.py`

**Files:**
- Create: `scripts/audition_checkpoint.py`
- Test: `tests/test_audition_checkpoint.py`

**Interfaces:**
- Consumes: session JSONs (Task 2 schema); serve.log text.
- Produces: `parse_log(text) -> dict` (keys `triggers`, `references`, `stall_lines`, `longest_stall_sec`, `round_trips`, `step_ms`); `score(sessions, log_metrics) -> dict`; `known_good(sessions) -> dict` with keys `demo`, `persona`, `avoid`; CLI writing `scorecard.md` + `scorecard.json` + `known_good.md`.
- ⚠️ The serve.log regexes are centralized in constants at the top and derived from the spec §6 grep patterns; **verify against a real serve.log at first Friday use** and fix the constants only.

- [ ] **Step 1: Write the failing tests** — `tests/test_audition_checkpoint.py`:

```python
"""Scorecard math and log parsing for the audition harness.

The verdict weighting encodes the decision of record: stalls + voice/persona first,
trigger rate second. known_good.md is the Sunday demo script, so the ≥2-of-3 rule is
what keeps a lucky single-session fire out of the live demo.
"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audition_checkpoint import parse_log, score, known_good, QUESTIONS  # noqa: E402

LOG = """\
2026-08-08 10:00:00,000 INFO batched step took 81.2 ms
2026-08-08 10:00:05,000 INFO [RAG] model emitted ret token
2026-08-08 10:00:06,900 INFO Generated reference: The first World Cup...
2026-08-08 10:00:20,000 WARNING LM buffer empty
2026-08-08 10:00:20,400 WARNING LM buffer empty
2026-08-08 10:00:22,100 WARNING LM buffer empty
2026-08-08 10:00:40,000 INFO batched step took 79.8 ms
"""


def _session(n, overrides=None, voice=4):
    results = {q["id"]: {"fired": False, "passed": True, "notes": ""} for q in QUESTIONS}
    for qid in ("q04", "q05", "q06"):
        results[qid] = {"fired": True, "passed": True, "notes": ""}
    for qid, r in (overrides or {}).items():
        results[qid] = r
    return {"checkpoint": "t", "scaling": 2.0, "session": n, "voice": voice,
            "interrupt_yields": True, "results": results}


def test_parse_log_counts():
    m = parse_log(LOG)
    assert m["triggers"] == 1 and m["references"] == 1 and m["stall_lines"] == 3


def test_longest_stall_groups_adjacent_lines():
    # 10:00:20.0 → 10:00:20.4 (gap .4 merges) → 10:00:22.1 (gap 1.7 splits)
    assert parse_log(LOG)["longest_stall_sec"] == 0.4


def test_round_trip_pairs_trigger_with_next_reference():
    assert parse_log(LOG)["round_trips"] == [1.9]


def test_step_timings_extracted():
    assert parse_log(LOG)["step_ms"] == [81.2, 79.8]


def test_no_timestamps_falls_back_to_none():
    m = parse_log("LM buffer empty\nLM buffer empty\n")
    assert m["stall_lines"] == 2 and m["longest_stall_sec"] is None


def test_score_trigger_and_persona_rates():
    s = score([_session(1)], parse_log(LOG))
    assert s["trigger_rate"] == 1.0
    assert s["persona_rate"] == 1.0
    assert s["spurious_fires"] == 0
    assert s["voice_mean"] == 4.0


def test_score_counts_spurious_fires_on_nonfactual():
    bad = {"q02": {"fired": True, "passed": False, "notes": "retrieved own name"}}
    assert score([_session(1, bad)], parse_log(LOG))["spurious_fires"] == 1


def test_known_good_requires_two_of_three_sessions():
    miss = {"q06": {"fired": False, "passed": False, "notes": ""}}
    kg = known_good([_session(1), _session(2, miss), _session(3, miss)])
    assert "q05" in kg["demo"] and "q06" not in kg["demo"]
    assert "q06" in kg["avoid"]


def test_known_good_factual_needs_fire_and_pass():
    conf = {"q04": {"fired": False, "passed": True, "notes": "confabulated correctly"}}
    kg = known_good([_session(1, conf), _session(2, conf), _session(3, conf)])
    assert "q04" in kg["avoid"]


def test_persona_questions_listed_separately():
    kg = known_good([_session(1), _session(2), _session(3)])
    assert "q02" in kg["persona"] and "q02" not in kg["demo"]
```

- [ ] **Step 2: Run to verify they fail** — `python3 -m pytest tests/test_audition_checkpoint.py -q` → FAIL (`ModuleNotFoundError: audition_checkpoint`).

- [ ] **Step 3: Write `scripts/audition_checkpoint.py`:**

```python
#!/usr/bin/env python3
"""Audition scorecard: serve.log + filled session JSONs → numbers + known_good.md.

Replaces single-session judgement calls (which produced the wrong "checkpoint 500
regression" verdict) with the same fixed questions scored the same way every time.
Verdict weighting is the decision of record: stalls + voice/persona first, triggering
second. known_good.md IS the Sunday demo script.

Log patterns come from spec §6 greps. If a real serve.log doesn't match, fix the
constants below — nowhere else.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

TRIGGER_PAT = re.compile(r"model emitted")
REFERENCE_PAT = re.compile(r"Generated reference")
STALL_PAT = re.compile(r"LM buffer empty")
STEP_PAT = re.compile(r"batched step.*?([\d.]+)\s*ms")
TS_PAT = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")
STALL_MERGE_GAP_SEC = 1.0  # stall lines closer than this are one stall event

QUESTIONS = [
    {"id": "q01", "kind": "greeting"},
    {"id": "q02", "kind": "persona"},
    {"id": "q03", "kind": "persona"},
    {"id": "q04", "kind": "factual"},
    {"id": "q05", "kind": "factual"},
    {"id": "q06", "kind": "factual"},
    {"id": "q07", "kind": "decline"},
    {"id": "q08", "kind": "personal"},
    {"id": "q09", "kind": "persona"},
    {"id": "q10", "kind": "persona"},
    {"id": "q11", "kind": "persona"},
    {"id": "q12", "kind": "persona"},
]
KINDS = {q["id"]: q["kind"] for q in QUESTIONS}


def _ts(line):
    m = TS_PAT.search(line)
    if not m:
        return None
    raw = m.group(1).replace(",", ".").replace("T", " ")
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in raw else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(raw, fmt)


def parse_log(text):
    triggers, references, stalls, step_ms = [], [], [], []
    n_trig = n_ref = n_stall = 0
    for line in text.splitlines():
        t = _ts(line)
        if TRIGGER_PAT.search(line):
            n_trig += 1
            triggers.append(t)
        if REFERENCE_PAT.search(line):
            n_ref += 1
            references.append(t)
        if STALL_PAT.search(line):
            n_stall += 1
            stalls.append(t)
        m = STEP_PAT.search(line)
        if m:
            step_ms.append(float(m.group(1)))

    longest = None
    ts_stalls = [t for t in stalls if t]
    if ts_stalls:
        longest, start, prev = 0.0, ts_stalls[0], ts_stalls[0]
        for t in ts_stalls[1:]:
            if (t - prev).total_seconds() > STALL_MERGE_GAP_SEC:
                longest = max(longest, (prev - start).total_seconds())
                start = t
            prev = t
        longest = round(max(longest, (prev - start).total_seconds()), 3)

    round_trips = []
    refs = [t for t in references if t]
    for t in [t for t in triggers if t]:
        nxt = next((r for r in refs if r >= t), None)
        if nxt:
            round_trips.append(round((nxt - t).total_seconds(), 3))

    return {"triggers": n_trig, "references": n_ref, "stall_lines": n_stall,
            "longest_stall_sec": longest, "round_trips": round_trips,
            "step_ms": step_ms}


def score(sessions, log_metrics):
    fired = asked = grounded = spurious = 0
    persona_ok = persona_n = decline_ok = decline_n = 0
    for s in sessions:
        for qid, r in s["results"].items():
            kind = KINDS.get(qid)
            if kind == "factual":
                asked += 1
                if r["fired"]:
                    fired += 1
                    grounded += bool(r["passed"])
            elif kind in ("persona", "personal", "greeting"):
                spurious += bool(r["fired"])
                if kind == "persona":
                    persona_n += 1
                    persona_ok += bool(r["passed"])
            elif kind == "decline":
                decline_n += 1
                decline_ok += bool(r["passed"])
    return {
        "checkpoint": sessions[0]["checkpoint"],
        "scaling": sessions[0]["scaling"],
        "sessions": len(sessions),
        "trigger_rate": round(fired / asked, 3) if asked else None,
        "grounded_rate": round(grounded / fired, 3) if fired else None,
        "spurious_fires": spurious,
        "persona_rate": round(persona_ok / persona_n, 3) if persona_n else None,
        "decline_rate": round(decline_ok / decline_n, 3) if decline_n else None,
        "voice_mean": round(sum(s["voice"] for s in sessions) / len(sessions), 2),
        "interrupt_yields": [s.get("interrupt_yields") for s in sessions],
        **log_metrics,
    }


def known_good(sessions):
    demo, persona, avoid = [], [], []
    need = 2  # of 3 sessions — keeps lucky one-off fires out of the live demo
    for q in QUESTIONS:
        qid, kind = q["id"], q["kind"]
        rs = [s["results"][qid] for s in sessions if qid in s["results"]]
        if kind == "factual":
            ok = sum(1 for r in rs if r["fired"] and r["passed"])
        else:
            ok = sum(1 for r in rs if r["passed"] and not r["fired"])
        bucket = persona if kind == "persona" else demo if kind in ("factual", "decline", "greeting", "personal") else demo
        (bucket if ok >= need else avoid).append(qid)
    return {"demo": [q for q in demo], "persona": persona, "avoid": avoid}


def _render(card, kg):
    lines = [f"# Scorecard — {card['checkpoint']} (scaling {card['scaling']})", ""]
    for k in ("trigger_rate", "grounded_rate", "spurious_fires", "persona_rate",
              "decline_rate", "voice_mean", "longest_stall_sec", "stall_lines",
              "round_trips", "step_ms", "interrupt_yields"):
        lines.append(f"- **{k}**: {card[k]}")
    lines += ["", "## Demo script (fired+grounded or clean in ≥2/3 sessions)",
              ", ".join(kg["demo"]) or "(none)",
              "", "## Persona (in character, no fire, ≥2/3)",
              ", ".join(kg["persona"]) or "(none)",
              "", "## DO NOT ASK IN THE DEMO", ", ".join(kg["avoid"]) or "(none)"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    sessions = [json.load(open(p)) for p in a.sessions]
    card = score(sessions, parse_log(Path(a.log).read_text(errors="replace")))
    kg = known_good(sessions)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scorecard.json").write_text(json.dumps({**card, "known_good": kg}, indent=2))
    (out / "scorecard.md").write_text(_render(card, kg))
    print(_render(card, kg))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests** — `python3 -m pytest tests/test_audition_checkpoint.py -q` → PASS, 11 tests.
- [ ] **Step 5: Full suite** — `python3 -m pytest tests/ -q` → PASS, 191 (180 after Task 1 + 11).
- [ ] **Step 6: Commit:** `git add scripts/audition_checkpoint.py tests/test_audition_checkpoint.py && git commit -m "audition: log parser + scorecard + known-good demo script emitter"`

## Task 4: Provision wb-gpu-training — 🖥️ HANDOFF

**Files:**
- Create: `docs/demo_sprint_runbook.md` (section 1; later tasks append sections 2–4)

**Interfaces:**
- Produces: a box able to run Track A: repo synced, `uv sync` done (pyproject pins the fork, moshi 0.2.13), HF login valid, moshika weights cached, Pass-A-ready data present.

- [ ] **Step 1: Write runbook section 1** in `docs/demo_sprint_runbook.md`:

```markdown
# Demo sprint runbook

## 1. Provision wb-gpu-training (Thu afternoon)

# Mac — push the repo state and the data Track A needs
rsync -avP --exclude runs --exclude runs_backup --exclude .git \
  --exclude finetune/data/datastereo --exclude finetune/data/prepared \
  /Users/leenatantawy/Downloads/capstone/moshi-finetune/ wb-gpu-training:~/moshi-finetune/

# Box — env (uv sync reads the fork-pinned pyproject)
cd ~/moshi-finetune && uv sync
uv run python -c "import moshi; print(moshi.__version__)"   # expect 0.2.13
huggingface-cli login    # must reach kyutai/moshika-pytorch-bf16
uv run python -c "from huggingface_hub import snapshot_download; \
snapshot_download('kyutai/moshika-pytorch-bf16')"           # prefetch ~8 GB

# Box — verify Track A's manifests resolve (paths are absolute /home/leenatantawy/...)
for f in finetune/data/prepared_dialogue/train.jsonl replay/retrieval_nomarkers/train.jsonl; do
  echo "$f: $(wc -l < ~/moshi-finetune/$f) rows"; done
ls finetune/data/prepared_dialogue/*.wav | head -2
ls replay/retrieval_nomarkers/*.wav | head -2

# Box — 60-second smoke: config loads, one forward+backward
sed 's/max_steps:.*/max_steps: 1/; s/do_ckpt:.*/do_ckpt: false/' \
  example/moshika_voice_max.yaml > /tmp/smoke.yaml
uv run torchrun --nproc-per-node 1 -m train /tmp/smoke.yaml
```

⚠️ If `uv sync` fails on setuptools or the box lacks uv, fall back to the ultra-box
pattern: create `~/fork-venv` and `pip install -e .` — see `docs/stage3_a100_runbook.md`
preamble for the interpreter-not-uv rule.

- [ ] **Step 2: 🖥️ HANDOFF** — give the user section 1; wait for pasted output. Gate: moshi 0.2.13 printed, both manifests non-zero rows, smoke step completes. If the box cannot start at all (stockout): fallback per spec risk table — all three runs sequence on wb-gpu-a1ultra2g (Track A GPU0 overnight, 4a GPU1 after precompute, 4b Friday morning); note it in the runbook and continue.
- [ ] **Step 3: Commit:** `git add docs/demo_sprint_runbook.md && git commit -m "runbook: wb-gpu-training provisioning + smoke"`

## Task 5: Tonight — recording intake + two-pass pipeline

**Files:** as specified in `docs/superpowers/plans/2026-08-07-tonight-recording-pipeline.md` Tasks 2–5 (execute them task-by-task; they contain every command, gate, and config edit).

**Interfaces:**
- Consumes: the raw Zoom per-participant files; Task 1's persona guard.
- Produces: `finetune/data/prepared_persona/all.jsonl` (Pass A), `replay/persona/train.jsonl` + `.ref.safetensors` (Pass B), configs wired 0.35/0.15/0.25/0.25, tests 199 green (191 + 8 config tests).

- [ ] **Step 1: Intake gates (before tonight-plan Task 2 Step 1).** Confirm with Danielle the files are **unedited** (no noise reduction / normalize / limiting / silence trimming / fades / concatenation). Then on the Mac:

```bash
python3 - <<'EOF'
import soundfile as sf, glob, sys
files = sorted(glob.glob('<download dir>/*'))
infos = []
for f in files:
    try:
        i = sf.info(f); infos.append((f, i)); print(f, i.samplerate, i.channels, f"{i.duration/60:.2f}min")
    except Exception as e:
        print(f, 'SKIP', e)
durs = [i.duration for _, i in infos if i.channels == 1]
if len(durs) >= 2 and abs(durs[0] - durs[1]) > 1.0:
    sys.exit("TRACKS DIFFER BY >1s — edited or desynced; get the raw Zoom files")
EOF
```

Gate: two mono files, durations within 1 s. Zoom exports m4a at 32 kHz — that's fine (the pairing step resamples to 24 kHz); **duration mismatch is the red flag**, it means per-track editing happened. If she also has takes of the 60 s overlap test, listen: both voices must survive on both tracks.

- [ ] **Step 2:** Execute tonight-plan **Task 2** (pair → channel-correlation gate → rsync **to both boxes** → 🖥️ manifest → 🖥️ annotate ch0 then ch1 → pull transcripts → verify non-empty → commit manifest).
- [ ] **Step 3: Launch Track A the moment ch0 transcripts are verified** — 🖥️ HANDOFF (do not wait for Pass B; Track A needs only Pass A). Runbook section 2, written now, appended to `docs/demo_sprint_runbook.md`:

```markdown
## 2. Launch lines (tonight — run each inside a jupyter keep-alive notebook cell)

# wb-gpu-training — Track A (5–7 h; watch first 20 steps; 50–60 GB expected; halve batch_size on OOM)
%%bash
cd ~/moshi-finetune
uv run torchrun --nproc-per-node 1 -m train example/moshika_voice_max.yaml 2>&1 | tee ~/trackA.log

# wb-gpu-a1ultra2g GPU0 — 4a (~3 h). ALL FOUR flags or you silently reproduce Stage 3.
%%bash
cd ~/moshi-finetune
RAG_TOKEN_WEIGHT=25 MASK_UNLABELLED_RAG=1 RAG_DELAY=1 RAG_REF_DROPOUT=0.2 \
CUDA_VISIBLE_DEVICES=0 ~/fork-venv/bin/python -m torch.distributed.run --nproc-per-node 1 \
  -m train example/moshika_rag_stage4a.yaml 2>&1 | tee ~/run4a.log

# wb-gpu-a1ultra2g GPU1 — 4b (AFTER precompute done and :8001 killed; same four flags)
%%bash
cd ~/moshi-finetune
RAG_TOKEN_WEIGHT=25 MASK_UNLABELLED_RAG=1 RAG_DELAY=1 RAG_REF_DROPOUT=0.2 \
CUDA_VISIBLE_DEVICES=1 ~/fork-venv/bin/python -m torch.distributed.run --nproc-per-node 1 \
  -m train example/moshika_rag_stage4b.yaml 2>&1 | tee ~/run4b.log

# Mac — checkpoint backup loop (run in its own terminal, leave it)
while true; do
  rsync -avP wb-gpu-training:~/moshi-finetune/runs/moshika_voice_max/checkpoints/ runs_backup/trackA/
  rsync -avP wb-gpu-a1ultra2g:~/moshi-finetune/runs/moshika_rag_stage4a/checkpoints/ runs_backup/stage4a/
  rsync -avP wb-gpu-a1ultra2g:~/moshi-finetune/runs/moshika_rag_stage4b/checkpoints/ runs_backup/stage4b/
  sleep 900
done
```

- [ ] **Step 4:** Execute tonight-plan **Task 3** (segment with persona guard → `turn_mix` gate: persona ≫ 30, decline 15–20, persona ≈ 0 means the guard failed → filter → cut → marker-placement checks → commit).
- [ ] **Step 5:** Execute tonight-plan **Task 4** (🖥️ encoder :8001 on GPU1 → 🖥️ precompute → 🖥️ **verify gate: marker count == tensor count — never proceed on failure** → pull tensors → 🖥️ clip manifest).
- [ ] **Step 6:** Execute tonight-plan **Task 5** (config tests → data lists 0.35/0.15/0.25/0.25 → full suite → 🖥️ path preflight on the box → commit).
- [ ] **Step 7: 🖥️ HANDOFF — launch 4a (GPU0), kill the encoder, launch 4b (GPU1)** using runbook section 2. Watch the first 20 steps of each. If the recording slipped past ~1 a.m.: launch 4a tonight, defer 4b to Friday morning; never skip the verify gate.
- [ ] **Step 8:** Record in `docs/moshi-rag-experiments.md`: launch times, data counts (markers, persona/decline/grounded turns), and the answer to the intake question **"were previous deliveries silence-trimmed?"** — if yes, annotate the 0.43–1.2 s lead measurements as editing artifacts. Commit.

## Task 6: Friday — audition and decide

**Files:**
- Create: `replay/auditions/<ckpt>_s<N>.json` per session, scorecards per checkpoint (via Task 3's CLI)
- Modify: `docs/moshi-rag-experiments.md` (results), `docs/demo_sprint_runbook.md` (section 3)

**Interfaces:**
- Consumes: `docs/audition_protocol.md`, `scripts/audition_checkpoint.py`, checkpoints in `runs/` + `runs_backup/`.
- Produces: locked-candidate list + each candidate's `known_good.md`; the 6 pm gate decision.

- [ ] **Step 1: Write runbook section 3 (serve blocks)** — append to `docs/demo_sprint_runbook.md`:

```markdown
## 3. Serving for auditions (Friday) — kill the server between checkpoints (holds ~43 GB)

# encoder — ultra GPU1, leave up all day
CUDA_VISIBLE_DEVICES=1 setsid nohup ~/fork-venv/bin/python -m moshi.reference_encoder \
  --port 8001 > ~/refenc.log 2>&1 &
sleep 30 && curl -s localhost:8001/health || tail -20 ~/refenc.log

# Track B server — ultra GPU0, one checkpoint at a time
source ~/.moshi_env
export REFERENCE_ENCODER_URL=http://localhost:8001
MOSHI_LORA_SCALING=2.0 \
MOSHI_LORA_WEIGHT=~/moshi-finetune/runs/<run>/checkpoints/checkpoint_<N>/consolidated/lora.safetensors \
CUDA_VISIBLE_DEVICES=0 ~/fork-venv/bin/python -m moshi.server \
  --hf-repo kyutai/moshika-rag-pytorch-bf16 --stt-wait-time 2.0 --port 8998 2>&1 | tee ~/serve.log

# Track A server — wb-gpu-training (no encoder, no REFERENCE_ENCODER_URL)
MOSHI_LORA_WEIGHT=~/moshi-finetune/runs/moshika_voice_max/checkpoints/checkpoint_<N>/consolidated/lora.safetensors \
CUDA_VISIBLE_DEVICES=0 uv run python -m moshi.server \
  --hf-repo kyutai/moshika-pytorch-bf16 --stt-wait-time 2.0 --port 8999 2>&1 | tee ~/serveA.log

# after each session: grep sanity (zero "acquired slot" = stale service worker, use a private window)
grep -c "acquired slot" ~/serve.log
```

- [ ] **Step 2: First real serve.log → validate the scorer's regex constants** (Task 3 interface note). Run `python3 scripts/audition_checkpoint.py --log serve.log --sessions <one session> --out-dir /tmp/regexcheck` and compare `triggers`/`stall_lines` against manual `grep -cE "RAG. model emitted"` / `grep -c "LM buffer empty"`. Fix the constants if they disagree; commit the fix.
- [ ] **Step 3: Coarse pass** (3 sessions each): Track A 1200/1600/2000; 4a 400/600/800; 4b 200/300/400. Refine one step around each winner if time allows. Voice under-imprint rule for Track A: prefer later checkpoints unless eval-by-ear degrades.
- [ ] **Step 4: 6 pm decision gate.** Compare the best Track B scorecard against the base-model control row (spec §6 table). Satisfactory (stalls < 3 s, voice ≥ 4, trigger ≥ base×0.75) → no more training; else pick ONE contingency and launch overnight:
  - voice under-imprinted → **4b-relaxed**: copy `example/moshika_rag_stage4b.yaml` → `_4b_relaxed.yaml`, set `optim.lr: 2.0e-6`, `max_steps: 800` (spec §2: 4b totals ~¼ of Stage 3 adaptation; relax before touching rank)
  - triggering weak → **4c** per spec §2 preference order (mined dialogue first)
  - noise-distraction dominant → **F3 noise-aug run** (plan written at decision time; recipe in spec F3)
  - a rerun benefiting from real leads → do **lead labelling** first (spec "Friday-optional"; plan written at decision time)
- [ ] **Step 5:** Log every scorecard row in `docs/moshi-rag-experiments.md`; commit sessions + scorecards + log entry.

## Task 7: Saturday — lock, optimize serving, rehearse, record backups

**Files:**
- Modify: `docs/demo_sprint_runbook.md` (section 4 — demo day)
- Create: `docs/demo_script.md`; backup videos (outside the repo)

**Interfaces:**
- Consumes: Friday's winners + `known_good.md` files.
- Produces: locked checkpoints (paths written in runbook §4), demo script, two backup videos, rehearsed setup.

- [ ] **Step 1: Audition any overnight run by noon** (same protocol) → **lock both checkpoints**; write the exact `MOSHI_LORA_WEIGHT` paths into runbook section 4.
- [ ] **Step 2: stt-wait step-down (F1).** On the locked Track B checkpoint, re-run 3 sessions at `--stt-wait-time 1.0`, then `0.5`, comparing `round_trips`, `grounded_rate`, and missed/garbled context in `Generated reference` lines. Keep the lowest setting whose grounded_rate matches 2.0's. Record round-trip numbers vs the paper's 1.5 s cliff in the experiments log.
- [ ] **Step 3: Scaling A/B (only if trigger_rate disappointed).** Same locked checkpoint, `MOSHI_LORA_SCALING=1.5`, 3 sessions. Decision rule (decision of record): 1.5 wins only if it improves triggering **without** dropping voice_mean or persona_rate — Stage 3 says it won't; the harness makes it minutes to confirm.
- [ ] **Step 4: Write `docs/demo_script.md`** from the winners' `known_good.md`: opening greeting, 2–3 persona questions from `persona`, 2 factual from `demo`, one decline, one barge-in — in an order that mirrors Part 5's drift (casual → factual → casual). List the `avoid` questions prominently. Same script for the video and the live demo.
- [ ] **Step 5: Record backup videos** of both demos (screen + audio capture of a full scripted session each). Verify playback before tearing anything down.
- [ ] **Step 6: Dress rehearsal on demo hardware:** headphones + directional mic + quiet room (F2/F3), private browser window, both servers from runbook §3/§4, full script end-to-end. Fix only setup issues — the model is locked.
- [ ] **Step 7: Write runbook section 4** (demo day): exact locked-checkpoint serve blocks (copy §3 with the locked paths and the chosen stt-wait), boot order (encoder → Track B server → Track A server → private-window check → one harness smoke session), and the fallback ladder (live → backup video → Track A only). Commit everything: `git add docs/demo_script.md docs/demo_sprint_runbook.md replay/auditions && git commit -m "demo: locked checkpoints, script, rehearsal + demo-day runbook"`
- [ ] **Step 8 (evening, optional):** one more run **only if** Saturday's audition exposed a fixable-by-data/hyperparameter fault — pick from Task 6 Step 4's list; boxes stay up regardless.

## Task 8: Sunday — demo

- [ ] **Step 1: 🖥️** Boot per runbook §4 ≥90 min before the slot; run one harness smoke session per track; `grep -c "acquired slot"` before trusting the browser.
- [ ] **Step 2:** Demo from `docs/demo_script.md`. If Track B misbehaves live: fall back per the runbook ladder (video, then Track A-only) — decided in advance, not improvised.
- [ ] **Step 3 (after):** Write `docs/HANDOFF-2026-08-10.md`: what was demoed, which checkpoints, scorecards, and what remains (lead labelling, noise-aug, dialogue mining status). Commit.

---

## Self-review notes

- Spec coverage: F1 → Task 7 Step 2; F2 → protocol + rehearsal; F3 → Task 6 Step 4 + rehearsal hardware; F4 → brief already updated (user texted Danielle); F5 → coarse pass includes `runs_backup` fallbacks implicitly via Task 6 Step 3 comparisons and the Task 8 fallback ladder; scaling A/B → Task 7 Step 3; lead labelling → Task 6 Step 4 (decision-gated); intake rules → Task 5 Step 1; keep-alive/stockout → Global Constraints + Task 5 Step 3 backup loop.
- Contingencies (noise-aug, lead labelling, 4c) are decision-gated and get their plan written at decision time — deliberately not speculatively planned here (YAGNI); their recipes live in the spec.
- Type consistency: session JSON schema identical in Tasks 2 and 3; `parse_log`/`score`/`known_good` names match between tests and implementation; runbook section numbers referenced consistently (1 provision, 2 launch, 3 serve, 4 demo day).
