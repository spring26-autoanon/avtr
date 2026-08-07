# Tonight's Recording Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn tonight's ~48-minute Danielle+Joshua recording into two datasets — an unmarked voice corpus both training tracks use, and a marked retrieval subset Track B uses — verified and wired into the configs, ready to launch overnight.

**Architecture:** Two passes over the same audio. **Pass A** treats the whole recording as long-form dialogue (pair → annotate → manifest), with no segmentation and no `<RAG>` markers — this is the voice/persona/turn-taking data. **Pass B** runs the existing Stage-3 retrieval pipeline over the same master to extract *only* the marked material (declines, and the factual turns that surface mid-chat in Parts 5 and 7). The same audio appearing in both passes is safe **only because of the loss mask** landed in `63caa2d`: unmarked windows now assert nothing about `⟨ret⟩`, so the long-form copy cannot contradict the marked clips. Before that change this design would have trained "marker here" and "no marker here" on identical audio.

**Tech Stack:** Python 3, `sphn`, `soundfile`, `whisper_timestamped` (GPU), Gemini `gemini-flash-latest`, `safetensors`, `pytest`.

## Global Constraints

- **The user runs every GPU command.** Steps marked **🖥️ HANDOFF** stop, print the exact block for the user to paste, and wait for their output. Never ssh to a box, never launch training, annotate, `precompute_references`, or `:8001`.
- Channel convention: **ch0 = Danielle = `SPEAKER_MAIN`**, ch1 = Joshua.
- Audio: **24 kHz, PCM_24, 2-channel** stereo out of `pair_dialogue_stereo.py`.
- Manifest paths must be **absolute** — `sphn.dataset_jsonl` resolves relative paths against the jsonl's own directory.
- **Never modify** `finetune/data/datastereo/` or `finetune/data/prepared/`. The latter is the silent-right-channel monologue corpus that caused the monologuing failure; it stays out of every manifest.
- `source .env` before any Gemini call (`GEMINI_API_KEY`). Model is `gemini-flash-latest` — `gemini-2.5-flash` 404s for this account.
- Box paths are `/home/leenatantawy/moshi-finetune/...`; Mac paths are repo-relative.
- Reference encoder is `http://localhost:8001`, started on the GPU that is *not* running training.
- Run `python3 -m pytest tests/ -q` after every code task; the suite is 171 tests and must stay green.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `scripts/segment_retrieval_audio.py` | Gemini windowed segmentation + per-turn labelling | **Modify** — add the `persona` kind |
| `scripts/cut_retrieval_clips.py` | Slice clips, insert `<RAG>` markers | Unchanged — `RETRIEVAL_KINDS` already excludes anything new |
| `tests/test_persona_guard.py` | Proves identity answers never get a marker | **Create** |
| `docs/tonight_recording_runbook.md` | The paste-able command sequence, with handoff points | **Create** |
| `example/moshika_voice_max.yaml` | Track A data list | **Modify** — add Pass A manifest |
| `example/moshika_rag_stage4a.yaml`, `..._stage4b.yaml` | Track B data lists | **Modify** — add Pass A + Pass B manifests |
| `tests/test_stage4_configs.py` | Config invariants | **Modify** — assert the new data wiring |

---

### Task 1: Persona guard in the segmenter

**Why this is a code task and not a runbook note.** The segmenter's prompt defines `grounded` as "she answers a question using specific external facts." Tonight's Part 1 is twenty minutes of her stating specific facts about herself — name, hometown, job, 60+ plants. Gemini will label those `grounded` and write reference passages for them. That would mark ~45 identity answers with `<RAG>`, training the model to fire a retrieval every time someone asks its name.

The decision of record (spec §3.2) is that **persona lives in the weights, not in retrieval** — routing identity through a trigger that fires ~50% of the time is strictly worse than the model simply knowing who it is. So identity answers must be labelled and then *not* marked.

`cut_retrieval_clips.py:26` already reads `RETRIEVAL_KINDS = ("grounded", "decline")`, so any kind outside that tuple produces no marker automatically. The work is to make `persona` a *recognised* kind rather than one that degrades to `smalltalk` — we want it counted separately so the runbook can sanity-check that Gemini actually found ~45 of them.

**Files:**
- Modify: `scripts/segment_retrieval_audio.py` (prompt block ~line 254, `turn_mix` line 343-353, `normalize` line 367-372)
- Test: `tests/test_persona_guard.py`

**Interfaces:**
- Consumes: `normalize(raw_segments, window_index)`, `turn_mix(segments)` from `scripts/segment_retrieval_audio.py`; `_retrieval_turns(segment)` and `RETRIEVAL_KINDS` from `scripts/cut_retrieval_clips.py`
- Produces: the string kind `"persona"`, valid in `normalize` output and counted by `turn_mix` under the key `"persona"`. `turn_mix`'s `retrieval_share` must **exclude** persona turns from the numerator.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_persona_guard.py`:

```python
"""Identity answers must never carry a <RAG> marker.

Tonight's Part 1 is ~20 min of Danielle stating specific facts about herself. The segmenter
prompt defines "grounded" as answering with specific external facts, so Gemini will label
those grounded and write reference passages — marking ~45 identity answers and teaching the
model to fire a retrieval whenever it is asked its name.

Decision of record (spec 3.2): persona lives in the weights, not in retrieval. A trigger that
fires ~50% of the time is a bad place to keep the model's own name.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from cut_retrieval_clips import _retrieval_turns          # noqa: E402
from segment_retrieval_audio import normalize, turn_mix   # noqa: E402


def _turn(kind, speaker="DANIELLE", start=0.0, end=1.0, reference="a passage"):
    return {"speaker": speaker, "start": start, "end": end,
            "kind": kind, "reference": reference, "text": "words"}


def test_persona_is_a_recognised_kind():
    """It must survive normalize, not degrade to smalltalk — we need to count it."""
    out = normalize([{"start": 0.0, "end": 2.0, "turns": [_turn("persona")]}], 0)
    assert out[0]["turns"][0]["kind"] == "persona"


def test_persona_turns_carry_no_reference():
    """A reference would be precomputed into a tensor with no marker to attach it to."""
    out = normalize([{"start": 0.0, "end": 2.0, "turns": [_turn("persona")]}], 0)
    assert out[0]["turns"][0]["reference"] is None


def test_persona_never_becomes_a_retrieval_turn():
    """The whole point: no <RAG> on 'What's your name?'."""
    segment = {"turns": [_turn("smalltalk", speaker="JOSHUA"),
                         _turn("persona", start=1.0, end=2.0)]}
    assert _retrieval_turns(segment) == []


def test_grounded_still_becomes_a_retrieval_turn():
    """Guard must not suppress real retrieval."""
    segment = {"turns": [_turn("smalltalk", speaker="JOSHUA"),
                         _turn("grounded", start=1.0, end=2.0)]}
    assert len(_retrieval_turns(segment)) == 1


def test_turn_mix_counts_persona_separately():
    segments = [{"turns": [_turn("persona"), _turn("persona"), _turn("grounded")]}]
    mix = turn_mix(segments)
    assert mix["persona"] == 2
    assert mix["grounded"] == 1


def test_persona_is_excluded_from_retrieval_share():
    """retrieval_share sets RAG_TOKEN_WEIGHT. Counting 45 persona turns as retrieval would
    inflate it and pick the wrong weight."""
    segments = [{"turns": [_turn("persona")] * 8 + [_turn("grounded")] * 2}]
    assert turn_mix(segments)["retrieval_share"] == 0.2


def test_unknown_kinds_still_degrade_to_smalltalk():
    """The existing safety net must survive the change."""
    out = normalize([{"start": 0.0, "end": 2.0, "turns": [_turn("weird")]}], 0)
    assert out[0]["turns"][0]["kind"] == "smalltalk"
    assert out[0]["turns"][0]["reference"] is None


def test_joshua_turns_are_never_persona():
    """Only her turns get a kind; his are always smalltalk."""
    out = normalize(
        [{"start": 0.0, "end": 2.0, "turns": [_turn("persona", speaker="JOSHUA")]}], 0)
    assert out[0]["turns"][0]["kind"] == "smalltalk"


def test_the_prompt_defines_the_persona_label():
    """Gemini cannot emit a label it was never told about."""
    src = (REPO / "scripts/segment_retrieval_audio.py").read_text()
    assert '"persona"' in src
    assert "about herself" in src.lower() or "about yourself" in src.lower()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_persona_guard.py -q`
Expected: FAIL — `test_persona_is_a_recognised_kind` asserts `"persona" == "smalltalk"` (the current `normalize` degrades it), and `test_turn_mix_counts_persona_separately` raises `KeyError`.

- [ ] **Step 3: Add the persona label to the prompt**

In `scripts/segment_retrieval_audio.py`, find the label definitions (~line 254) and replace:

```python
  "grounded"  - she answers a question using specific external facts.
  "decline"   - she says she does not know / cannot answer, instead of guessing.
  "smalltalk" - casual conversation needing no external facts.
```

with:

```python
  "grounded"  - she answers a question using specific external facts.
  "decline"   - she says she does not know / cannot answer, instead of guessing.
  "persona"   - she answers a question ABOUT HERSELF: her name, where she is from, her job,
                her tattoos, plants, martial arts, music taste, travel. These state specific
                facts, but they are facts about her own life, not retrieved knowledge.
                Label these "persona", never "grounded", and omit "reference".
  "smalltalk" - casual conversation needing no external facts.
```

- [ ] **Step 4: Accept `persona` in `normalize`**

In `normalize` (~line 367), change:

```python
            if kind not in ("grounded", "decline", "smalltalk"):
                kind = "smalltalk"
```

to:

```python
            if kind not in ("grounded", "decline", "persona", "smalltalk"):
                kind = "smalltalk"
```

and change the reference-stripping line below it:

```python
            if kind == "smalltalk":
                reference = None
```

to:

```python
            if kind in ("smalltalk", "persona"):
                reference = None
```

- [ ] **Step 5: Count persona in `turn_mix`**

Replace `turn_mix` (line 343-353) with:

```python
def turn_mix(segments):
    """Count HER turns by kind. retrieval_share sets RAG_TOKEN_WEIGHT (see the spec).

    `persona` turns state specific facts but are answered from the weights, not retrieval, so
    they are counted and then excluded from the share — otherwise tonight's ~45 identity
    answers would inflate it and pick the wrong weight.
    """
    counts = {"grounded": 0, "decline": 0, "persona": 0, "smalltalk": 0}
    for seg in segments:
        for t in seg["turns"]:
            if t["speaker"] != "DANIELLE":
                continue
            counts[t["kind"]] = counts.get(t["kind"], 0) + 1
    total = sum(counts.values())
    share = (counts["grounded"] + counts["decline"]) / total if total else 0.0
    return {**counts, "total": total, "retrieval_share": round(share, 4)}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_persona_guard.py -q`
Expected: PASS, 9 tests.

- [ ] **Step 7: Run the full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS, 180 tests (171 existing + 9 new).

- [ ] **Step 8: Commit**

```bash
git add scripts/segment_retrieval_audio.py tests/test_persona_guard.py
git commit -m "segmenter: persona kind so identity answers never carry a <RAG> marker"
```

---

### Task 2: Pass A — the unmarked voice corpus

Everything she recorded, as long-form stereo dialogue. No segmentation, no cutting, no markers, no reference tensors. This is what gives both tracks her voice, her filler, the persona content, the conversation openings and the overlap.

**Files:**
- Create: `finetune/data/prepared_persona/*.wav`, `*.json`, `*.ch1.json`, `all.jsonl`
- Uses: `scripts/pair_dialogue_stereo.py`, `scripts/build_manifest.py`, `annotate.py`

**Interfaces:**
- Consumes: the raw two-track recording from Danielle (her track + Joshua's track)
- Produces: `finetune/data/prepared_persona/all.jsonl` — absolute paths, used by Track A and both Track B runs

- [ ] **Step 1: Confirm what arrived**

```bash
ls -la <download dir>
python3 -c "
import soundfile as sf, sys, glob
for f in sorted(glob.glob('<download dir>/*')):
    try:
        i = sf.info(f); print(f'{f}  {i.samplerate}Hz  {i.channels}ch  {i.duration/60:.1f}min')
    except Exception as e: print(f'{f}  SKIP {e}')
"
```

Expected: two mono files, ~48 min each, same duration. If one is stereo or they differ in length by more than a second, stop — the pairing assumes aligned isolated tracks.

- [ ] **Step 2: Pair to stereo**

```bash
python3 scripts/pair_dialogue_stereo.py \
  --src <download dir> \
  --dst finetune/data/prepared_persona \
  --main-name danielle \
  --no-eval
```

`--main-name danielle` puts her on ch0. `--no-eval` because the eval set is already too small to be useful (spec: `do_eval: false` at batch 16).

- [ ] **Step 3: Verify the channel assignment before anything expensive**

```bash
python3 -c "
import soundfile as sf, numpy as np, glob
for f in sorted(glob.glob('finetune/data/prepared_persona/*.wav')):
    x, sr = sf.read(f)
    a = 20*np.log10(np.sqrt((x**2).mean(axis=0)) + 1e-12)
    print(f'{f}  {sr}Hz  {x.shape[1]}ch  ch0 {a[0]:.1f}dB  ch1 {a[1]:.1f}dB  '
          f'corr {np.corrcoef(x[:,0], x[:,1])[0,1]:+.3f}')
"
```

Expected: 24000 Hz, 2 channels, both channels well above -60 dB, correlation near zero. A correlation above ~0.5 means the tracks were not isolated and turn-taking data is worthless — stop and ask for a re-export.

- [ ] **Step 4: Upload to the training box**

```bash
rsync -avP finetune/data/prepared_persona/*.wav \
  wb-a1ultra1g:~/moshi-finetune/finetune/data/prepared_persona/
```

If Track A is running on `wb-gpu-training`, rsync there too — that box needs the same wavs.

- [ ] **Step 5: Build the manifest on the box** — 🖥️ **HANDOFF**

Give the user this block and wait:

```bash
~/fork-venv/bin/python scripts/build_manifest.py \
  --wav-dir finetune/data/prepared_persona \
  --out-dir finetune/data/prepared_persona \
  --no-eval
```

- [ ] **Step 6: Annotate both channels** — 🖥️ **HANDOFF**

Two sequential runs. ch0 is the training transcript; ch1 is needed by the segmenter in Task 3 so it can see Joshua's questions.

```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup ~/fork-venv/bin/python annotate.py \
  finetune/data/prepared_persona/all.jsonl -l --whisper_model medium --lang en \
  --channel 0 --out-suffix .json > ~/annotate_persona_ch0.log 2>&1 &

# after ch0 finishes:
CUDA_VISIBLE_DEVICES=1 setsid nohup ~/fork-venv/bin/python annotate.py \
  finetune/data/prepared_persona/all.jsonl -l --whisper_model medium --lang en \
  --channel 1 --out-suffix .ch1.json > ~/annotate_persona_ch1.log 2>&1 &
```

`-l` is required — without it `annotate.py` mis-handles the manifest. Roughly 5-10 min per channel for 48 min of audio on an A100.

- [ ] **Step 7: Pull the transcripts back**

```bash
rsync -avP wb-a1ultra1g:~/moshi-finetune/finetune/data/prepared_persona/'*.json' \
  finetune/data/prepared_persona/
```

- [ ] **Step 8: Verify every wav has a non-empty transcript**

```bash
python3 -c "
import json, glob, os
bad = []
for w in sorted(glob.glob('finetune/data/prepared_persona/*.wav')):
    for suf in ('.json', '.ch1.json'):
        p = w[:-4] + suf
        if not os.path.exists(p): bad.append(f'MISSING {p}'); continue
        a = json.load(open(p)).get('alignments', [])
        if not a: bad.append(f'EMPTY {p}')
        else: print(f'{os.path.basename(p)}: {len(a)} words')
print('\n'.join(bad) if bad else 'all transcripts present and non-empty')
"
```

Training crashes on a missing sibling `.json` — `interleaver.py:267-270` opens it unconditionally.

- [ ] **Step 9: Commit the manifest**

```bash
git add finetune/data/prepared_persona/all.jsonl
git commit -m "data: Pass A manifest for the 2026-08-07 persona recording"
```

Wavs and transcripts stay untracked; only the manifest is small enough to version.

---

### Task 3: Pass B — the marked retrieval subset

Run the Stage-3 pipeline over the *same* master audio, keeping only what should carry a `<RAG>`: the Part 2 declines, and the factual turns that surface mid-chat in Parts 5 and 7. Task 1's persona guard is what keeps Part 1 out of this.

**Files:**
- Create: `replay/persona_segments.jsonl`, `replay/persona_segments.filtered.jsonl`, `replay/persona_dropped.jsonl`, `replay/persona/*.wav|.json`, `replay/persona_manifest.jsonl`
- Uses: `scripts/segment_retrieval_audio.py`, `scripts/filter_retrieval_segments.py`, `scripts/cut_retrieval_clips.py`

**Interfaces:**
- Consumes: `finetune/data/prepared_persona/*.wav` + `.json` + `.ch1.json` from Task 2; the `persona` kind from Task 1
- Produces: `replay/persona_manifest.jsonl` with a `reference` field per retrieval turn, consumed by `precompute_references.py` in Task 4

- [ ] **Step 1: Segment with Gemini**

```bash
source .env
python3 scripts/segment_retrieval_audio.py \
  --ch0 finetune/data/prepared_persona/<master>.json \
  --ch1 finetune/data/prepared_persona/<master>.ch1.json \
  --out replay/persona_segments.jsonl \
  --model gemini-flash-latest
```

12-minute windows with 1-minute overlap, deduped. Watch the printed `turn_mix`.

- [ ] **Step 2: Sanity-check the mix against what she was asked to record**

```bash
python3 -c "
import json
segs = [json.loads(l) for l in open('replay/persona_segments.jsonl')]
from collections import Counter
c = Counter(t['kind'] for s in segs for t in s['turns'] if t['speaker'] == 'DANIELLE')
print(dict(c))
print('segments:', len(segs))
"
```

Expected shape, given the brief asks for ~45 persona exchanges, 15-20 declines, and ~11 factual turns across Parts 5 and 7:

- `persona` **≫ 30** — if this is near zero, the guard did not take and Part 1 is being marked. Stop.
- `decline` roughly 15-20
- `grounded` roughly 8-15 — these are the Part 5/7 factual turns, the most valuable new data
- `smalltalk` the remainder

- [ ] **Step 3: Filter the grounded turns on topic match**

```bash
python3 scripts/filter_retrieval_segments.py \
  --in replay/persona_segments.jsonl \
  --out replay/persona_segments.filtered.jsonl \
  --dropped replay/persona_dropped.jsonl \
  --model gemini-flash-latest
```

This is a **topic screen, not a completeness grader** — it drops a segment whose reference passage is about a different subject than her answer. An earlier version graded completeness and rejected 34% of good turns on Whisper transcription noise. Expect very few drops; if it rejects more than ~20%, read `replay/persona_dropped.jsonl` before accepting it.

- [ ] **Step 4: Cut the clips**

```bash
python3 scripts/cut_retrieval_clips.py \
  --master finetune/data/prepared_persona/<master>.wav \
  --alignments finetune/data/prepared_persona/<master>.json \
  --segments replay/persona_segments.filtered.jsonl \
  --outdir replay/persona \
  --manifest replay/persona_manifest.jsonl
```

`TAIL_SEC = 3.0` carries three seconds of Joshua's next turn past her last word. Without it 97% of clips ended within 0.5 s of her final word, which trained the empty-buffer stall.

- [ ] **Step 5: Verify markers landed only where intended**

```bash
python3 -c "
import json, glob
tot = 0
for f in sorted(glob.glob('replay/persona/*.json')):
    a = json.load(open(f)).get('alignments', [])
    n = sum(1 for w in a if w[0] == '<RAG>')
    tot += n
    if n: print(f'{f.split(\"/\")[-1]}: {n} marker(s)')
print('total markers:', tot)
"
```

Cross-check against Step 2: total markers should equal `grounded + decline` minus anything the filter dropped. A count far above that means persona answers are being marked.

- [ ] **Step 6: Spot-check three clips by ear and by transcript**

```bash
python3 -c "
import json, glob, random
for f in random.sample(sorted(glob.glob('replay/persona/*.json')), 3):
    a = json.load(open(f)).get('alignments', [])
    print('---', f)
    print(' '.join(w[0] for w in a)[:400])
"
```

Confirm each `<RAG>` sits immediately before the start of an answer, never mid-sentence. One earlier clip had four markers inside a single answer.

- [ ] **Step 7: Commit the manifests**

```bash
git add replay/persona_segments.jsonl replay/persona_segments.filtered.jsonl \
        replay/persona_dropped.jsonl replay/persona_manifest.jsonl
git commit -m "data: Pass B marked retrieval subset from the 2026-08-07 recording"
```

---

### Task 4: Reference tensors and the verify gate

**Files:**
- Create: `replay/persona/*.ref.safetensors`
- Uses: `scripts/precompute_references.py`, `scripts/verify_retrieval_clips.py`

**Interfaces:**
- Consumes: `replay/persona_manifest.jsonl` and `replay/persona/*.wav|.json` from Task 3
- Produces: one `.ref.safetensors` per marked clip, keys `reference_0..N-1`, loaded by `finetune/data/interleaver.py::_load_reference_tensors`

- [ ] **Step 1: Upload the clips**

```bash
rsync -avP replay/persona/ wb-a1ultra1g:~/moshi-finetune/replay/persona/
rsync -avP replay/persona_manifest.jsonl wb-a1ultra1g:~/moshi-finetune/replay/
```

- [ ] **Step 2: Start the reference encoder** — 🖥️ **HANDOFF**

```bash
CUDA_VISIBLE_DEVICES=1 setsid nohup ~/fork-venv/bin/python -m moshi.reference_encoder \
  --port 8001 > ~/refenc.log 2>&1 &
sleep 30 && curl -s localhost:8001/health || tail -20 ~/refenc.log
```

Put this on the GPU that is not running training. It is the ARC encoder (Llama-3.2-3B based), roughly 7 GB.

- [ ] **Step 3: Precompute the tensors** — 🖥️ **HANDOFF**

```bash
~/fork-venv/bin/python scripts/precompute_references.py \
  --manifest replay/persona_manifest.jsonl \
  --audiodir replay/persona \
  --url http://localhost:8001
```

The script preflight-probes the encoder before doing work, so a dead `:8001` fails immediately rather than after twenty minutes.

- [ ] **Step 4: Run the verify gate** — 🖥️ **HANDOFF**

```bash
~/fork-venv/bin/python scripts/verify_retrieval_clips.py --clips replay/persona
```

`check_clips()` asserts **marker count equals tensor count** for every clip. This is the gate that catches the failure mode `build_reference_condition` cannot: a clip with two `<RAG>` and one reference silently trains the second answer with no conditioning, teaching the model to answer factual questions from its own head. Do not proceed on a failure — re-run Task 3 Step 4.

- [ ] **Step 5: Pull the tensors back for the record**

```bash
rsync -avP wb-a1ultra1g:~/moshi-finetune/replay/persona/'*.ref.safetensors' replay/persona/
```

- [ ] **Step 6: Build the clip manifest on the box** — 🖥️ **HANDOFF**

```bash
~/fork-venv/bin/python scripts/build_manifest.py \
  --wav-dir replay/persona --out-dir replay/persona --no-eval
```

---

### Task 5: Wire into the configs and preflight

**Files:**
- Modify: `example/moshika_voice_max.yaml:22`, `example/moshika_rag_stage4a.yaml`, `example/moshika_rag_stage4b.yaml`
- Modify: `tests/test_stage4_configs.py`

⚠️ `test_all_data_paths_are_absolute` (line 96) splits `train_data` on `,` and asserts each
piece starts with `/`. The `:weight` suffix does not break that assertion, but any new test
that treats a piece as a bare path must strip the suffix with `src.rsplit(":", 1)[0]`.

**Interfaces:**
- Consumes: `finetune/data/prepared_persona/all.jsonl` (Task 2), `replay/persona/train.jsonl` (Task 4)
- Produces: launch-ready configs

- [ ] **Step 1: Write the failing config tests**

Append to `tests/test_stage4_configs.py`:

```python
def test_voice_track_includes_the_persona_recording_unmarked():
    """Pass A is the whole recording as long-form dialogue — persona, openings, overlap."""
    sources = cfg(VOICE)["data"]["train_data"]
    assert "prepared_persona" in sources


def test_voice_track_still_excludes_every_marked_directory():
    """<RAG> is token 4, an ordinary word on plain moshika. Pass B must not appear here."""
    sources = cfg(VOICE)["data"]["train_data"]
    assert "replay/persona/" not in sources
    assert "replay/retrieval/" not in sources


def test_rag_tracks_get_both_passes():
    """Pass A for voice (unmarked, masked by the loss), Pass B for the trigger."""
    for p in (S4A, S4B):
        sources = cfg(p)["data"]["train_data"]
        assert "prepared_persona" in sources, p.name
        assert "replay/persona/" in sources, p.name


def test_rag_tracks_exclude_the_marker_stripped_copy():
    """retrieval_nomarkers has no reference tensors — on the RAG track it is dead weight
    that duplicates audio already present in replay/retrieval/."""
    for p in (S4A, S4B):
        assert "retrieval_nomarkers" not in cfg(p)["data"]["train_data"], p.name


def test_rag_tracks_keep_explicit_sampling_weights():
    """These configs sample by weight. An unweighted source silently changes the mix and
    breaks 4a's single-variable property."""
    for p in (S4A, S4B):
        for src in cfg(p)["data"]["train_data"].split(","):
            assert ":" in src.rsplit("/", 1)[-1], (p.name, src)


def test_rag_sampling_weights_sum_to_one():
    for p in (S4A, S4B):
        w = [float(s.rsplit(":", 1)[1]) for s in cfg(p)["data"]["train_data"].split(",")]
        assert abs(sum(w) - 1.0) < 1e-6, (p.name, w)


def test_rag_tracks_hold_the_stage3_marked_unmarked_ratio():
    """0.5 marked / 0.5 unmarked, as Stage 3 had. Changing the mix is run 4c, not 4a."""
    for p in (S4A, S4B):
        marked = unmarked = 0.0
        for src in cfg(p)["data"]["train_data"].split(","):
            path, weight = src.rsplit(":", 1)
            if "retrieval_nomarkers" in path or "prepared_dialogue" in path \
                    or "prepared_persona" in path:
                unmarked += float(weight)
            else:
                marked += float(weight)
        assert abs(marked - 0.5) < 1e-6 and abs(unmarked - 0.5) < 1e-6, (p.name, marked)


def test_no_config_references_the_silent_channel_monologues():
    """finetune/data/prepared/ is the Option-A corpus that caused the monologuing failure."""
    for p in (VOICE, S4A, S4B):
        for src in cfg(p)["data"]["train_data"].split(","):
            assert "/prepared/" not in src, (p.name, src)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_stage4_configs.py -q`
Expected: FAIL — `prepared_persona` appears in no config yet.

- [ ] **Step 3: Update Track A's data list**

In `example/moshika_voice_max.yaml`, replace line 22's `train_data` value with:

```yaml
  train_data: "/home/leenatantawy/moshi-finetune/finetune/data/prepared_dialogue/train.jsonl,/home/leenatantawy/moshi-finetune/replay/retrieval_nomarkers/train.jsonl,/home/leenatantawy/moshi-finetune/finetune/data/prepared_persona/all.jsonl"
```

- [ ] **Step 4: Update both Track B data lists — keeping the sampling weights**

⚠️ **These configs use weighted sampling.** They currently read
`replay/retrieval/train.jsonl:0.5,.../prepared_dialogue/train.jsonl:0.5`. Dropping the
`:weight` suffixes is a silent failure: sampling shifts to whatever four unweighted sources
produce, and 4a stops being a single-variable run. Every source below keeps an explicit
weight, and they sum to 1.0.

| source | weight | marked? | why |
|---|---|---|---|
| `replay/retrieval` | 0.35 | yes | the existing 119 trigger positives |
| `replay/persona` | 0.15 | yes | tonight's real declines + casual-register facts |
| `prepared_dialogue` | 0.25 | no | 135 min of voice, two partners |
| `prepared_persona` | 0.25 | no | tonight's voice + identity |

This holds Stage 3's **0.5 marked / 0.5 unmarked** ratio, so 4a still isolates the mask and
delay rather than confounding them with a data-mix change (that is 4c's job). The persona
recording gets the same weight as the dialogue despite being roughly a third its length —
deliberate, because it is the only audio in the corpus containing her name.

In `example/moshika_rag_stage4a.yaml:19` and `example/moshika_rag_stage4b.yaml:21`, set
`train_data` to:

```yaml
  train_data: "/home/leenatantawy/moshi-finetune/replay/retrieval/train.jsonl:0.35,/home/leenatantawy/moshi-finetune/replay/persona/train.jsonl:0.15,/home/leenatantawy/moshi-finetune/finetune/data/prepared_dialogue/train.jsonl:0.25,/home/leenatantawy/moshi-finetune/finetune/data/prepared_persona/all.jsonl:0.25"
```

Leave `eval_data` alone — `do_eval` is false on both runs.

- [ ] **Step 5: Run the config tests**

Run: `python3 -m pytest tests/test_stage4_configs.py -q`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS, 188 tests.

- [ ] **Step 7: Preflight every data path exists on the box** — 🖥️ **HANDOFF**

```bash
for f in finetune/data/prepared_dialogue/train.jsonl \
         replay/retrieval/train.jsonl \
         replay/retrieval_nomarkers/train.jsonl \
         finetune/data/prepared_persona/all.jsonl \
         replay/persona/train.jsonl; do
  n=$(wc -l < ~/moshi-finetune/$f 2>/dev/null || echo MISSING)
  echo "$f: $n rows"
done
```

Every path in a config must exist with a non-zero row count. A missing manifest fails at step 1 of training, after the model has loaded — five minutes wasted, but only if someone is watching.

- [ ] **Step 8: Commit**

```bash
git add example/moshika_voice_max.yaml example/moshika_rag_stage4a.yaml \
        example/moshika_rag_stage4b.yaml tests/test_stage4_configs.py
git commit -m "configs: fold the 2026-08-07 recording into both tracks"
```

- [ ] **Step 9: Write the runbook**

Create `docs/tonight_recording_runbook.md` containing, in order: the Task 2-4 command blocks with their handoff markers, the three verification gates (channel correlation, marker count vs `turn_mix`, `verify_retrieval_clips`), and the two launch lines from spec §3.2b:

```bash
# Track A - plain moshika. No env vars: token 4 is an ordinary token here.
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 -m train example/moshika_voice_max.yaml

# Track B run 4a - all four switches. Every one defaults to OFF.
RAG_TOKEN_WEIGHT=25 MASK_UNLABELLED_RAG=1 RAG_DELAY=1 RAG_REF_DROPOUT=0.2 \
  CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 -m train example/moshika_rag_stage4a.yaml
```

End it with the standing rules: watch the first 20 steps of any launch (rank 128 at batch 16 is the only untested memory configuration; budget 50-60 GB of 80), back up checkpoints to the Mac as they land, and stop the box when idle.

```bash
git add docs/tonight_recording_runbook.md
git commit -m "docs: runbook for tonight's recording pipeline"
```

---

## Notes carried from the spec

**Why the same audio can appear in both passes.** Before `63caa2d`, a long-form unmarked window overlapping a marked clip would have trained contradictory signals on identical audio — "fire here" and "never fire here." The loss mask makes unmarked windows silent on the trigger, so the contradiction cannot arise. This plan depends on that commit; do not run it against an older checkout.

**Expect the delay fix to contribute little on this data.** Eq. 3 bounds `d'` by her lead, and at her measured 0.4-0.7 s leads no training example reaches the real 1.7-3.4 s serve latency. The brief now asks for a ~3 s runway, which is what would change this. A null delay result tomorrow is a data limit, not a broken implementation.

**`d_lead` is not labelled yet.** Every turn falls back to `DEFAULT_LEAD_SEC = 1.2`. Lead labelling (spec §4.2) is a separate piece of work and is not on the critical path for tonight.
