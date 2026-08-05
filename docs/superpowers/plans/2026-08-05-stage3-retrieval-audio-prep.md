# Stage 3 Retrieval-Audio Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 84.65-minute two-track Danielle+Joshua retrieval recording into per-dialogue training clips with per-turn `⟨ret⟩` markers and matching reference tensors, and make the trainer condition every retrieval turn rather than only the first.

**Architecture:** Pair the two isolated mono tracks into one stereo master (Danielle left, Joshua right), transcribe both channels once with Whisper on the A100, have Gemini segment the merged transcript into dialogue units with a per-turn `grounded`/`decline`/`smalltalk` label and a reference passage per retrieval turn, then cut clips while offset-slicing the existing alignments instead of re-transcribing. Multi-retrieval clips require a new N-tensor reference file format threaded through `precompute_references.py` → `interleaver.py` → `train.py`.

**Tech Stack:** Python 3.13, numpy, soundfile, torch 2.6 (local) / 2.9.1 (A100 fork), safetensors, whisper_timestamped (A100 only), Gemini REST via stdlib `urllib`, pytest.

Spec: `docs/superpowers/specs/2026-08-05-stage3-retrieval-audio-prep-design.md`

## Global Constraints

- **The user runs every A100 / GPU command themselves.** Never ssh to
  `wb-gpu-training`, never run `uv run torchrun`, `annotate.py`, or anything touching the
  `:8001` service. Remote steps are handed to the user as copy-pasteable blocks, and work
  stops until they report the output back. Everything that does not need CUDA — pairing,
  segmentation, clip cutting — runs locally to keep that hand-run queue short.
- **Local tests run under system `python3 -m pytest`**, not `.venv/bin/python` and not `uv run`. The system interpreter has numpy, soundfile, torch 2.6, safetensors and pytest; `.venv` does not.
- **`moshi` is not installed on the Mac.** No test may import it, directly or transitively. `finetune/data/interleaver.py` does `from moshi.conditioners import ConditionAttributes` at line 11, so **no test may import `interleaver`**. Logic that needs a local test goes in a new moshi-free module that `interleaver.py` imports.
- **Audio format is 24 kHz, PCM_24, stereo, channel 0 = Danielle (`SPEAKER_MAIN`), channel 1 = Joshua.** Never resample, never reorder channels.
- **`duration_sec` is 100** in every training config. No clip may exceed 100 seconds.
- **`rag_token_id` is `4`.** The `<RAG>` alignment marker word maps to raw token 4 (`interleaver._tokenize`).
- **Manifest paths must be absolute.** `sphn.dataset_jsonl` resolves manifest paths relative to the jsonl's own directory.
- **Never modify** `replay/QuestionAudio/*.wav` or `finetune/data/datastereo/`. These are source recordings.
- **Gemini calls use the stdlib REST pattern** in `scripts/gen_recording_dialogues.py:96-115` (`call_gemini`), with `GEMINI_API_KEY` from the environment. Never put a key on a command line.
- **Commit after every task.** Branch is `dialogue-data-prep`.

## File Structure

**New scripts (Mac-runnable, no moshi import):**
- `scripts/merge_utterances.py` — word alignments from two channels → speaker-tagged utterances.
- `scripts/segment_retrieval_audio.py` — utterances → Gemini → validated `retrieval_segments.jsonl`.
- `scripts/cut_retrieval_clips.py` — segments + master WAV → per-clip WAV/JSON/manifest.

**New library modules (moshi-free so they are locally testable):**
- `finetune/data/reference_io.py` — read/write the N-tensor `.ref.safetensors` format.
- `finetune/data/reference_injection.py` — build the `reference_with_time` condition tensor from all `⟨ret⟩` frames.

**Modified:**
- `scripts/pair_dialogue_stereo.py` — add `--no-eval`.
- `annotate.py` — add `--channel` and `--out-suffix`.
- `scripts/precompute_references.py` — write N reference tensors per clip.
- `finetune/data/interleaver.py:17-25,28-33` — `_load_reference_tensor` delegates to `reference_io` and returns a list.
- `train.py:254-279` — call `reference_injection.build_reference_condition`.
- `scripts/apply_rag_positioning.py` — emit the new train.py block so a fresh checkout matches.

**New config/docs:**
- `example/moshika_rag_stage3.yaml`
- `docs/stage3_a100_runbook.md`

**Tests:** `tests/test_merge_utterances.py`, `tests/test_segment_retrieval_audio.py`, `tests/test_cut_retrieval_clips.py`, `tests/test_reference_io.py`, `tests/test_reference_injection.py`, plus additions to `tests/test_pair_dialogue_stereo.py`.

---

### Task 1: `--no-eval` flag for pair_dialogue_stereo.py

The retrieval recording is a single conversation we keep whole; eval keeps coming from the natural set's existing `danielle_joshuarhodes_eval.wav`. Today `main()` exits if `--eval-conv` is not found among the discovered conversations (`scripts/pair_dialogue_stereo.py:146-149`).

**Files:**
- Modify: `scripts/pair_dialogue_stereo.py:126-170`
- Test: `tests/test_pair_dialogue_stereo.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `finetune/data/prepared_retrieval/danielle_joshuarhodes_qa.wav`, the stereo master every later task reads.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pair_dialogue_stereo.py`:

```python
def test_main_no_eval_writes_whole_conversations(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _fake_source(src)
    main(["--src", str(src), "--dst", str(dst), "--main-name", "danielle", "--no-eval"])
    names = sorted(p.name for p in dst.glob("*.wav"))
    assert names == ["danielle_clays.wav", "danielle_joshuarhodes.wav"]
    # the 10 s conversation is written whole, not carved
    assert sf.info(str(dst / "danielle_joshuarhodes.wav")).frames == 10 * 24000


def test_main_no_eval_ignores_missing_eval_conv(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _fake_source(src)
    # a bogus --eval-conv must not abort when --no-eval is set
    main(["--src", str(src), "--dst", str(dst), "--main-name", "danielle",
          "--no-eval", "--eval-conv", "9999999999"])
    assert (dst / "danielle_joshuarhodes.wav").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -k no_eval -v`
Expected: FAIL — `SystemExit: --eval-conv 1411304343 not found among ...` (argparse accepts no `--no-eval`, so it errors first with `unrecognized arguments: --no-eval`).

- [ ] **Step 3: Add the flag and skip the carve**

In `main()`, after the `--exclude-conv` argument, add:

```python
    ap.add_argument("--no-eval", action="store_true",
                    help="write every conversation whole; do not carve an eval slice")
```

Replace the `--eval-conv` validation block (currently lines 140-149) with:

```python
    exclude = {x for x in args.exclude_conv.split(",") if x}
    if not args.no_eval and args.eval_conv in exclude:
        raise SystemExit(
            f"--eval-conv {args.eval_conv} is also in --exclude-conv; pick a different eval conv."
        )

    convs = discover_conversations(src, args.main_name)
    if not args.no_eval and args.eval_conv not in convs:
        raise SystemExit(
            f"--eval-conv {args.eval_conv} not found among: {sorted(convs)}"
        )
```

And change the per-conversation branch so the carve is skipped:

```python
        if not args.no_eval and conv_id == args.eval_conv:
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -v`
Expected: PASS, including the pre-existing tests (the real-recording test stays skipped or passes as before).

- [ ] **Step 5: Produce the stereo master from the real recording**

```bash
python3 scripts/pair_dialogue_stereo.py \
  --src "replay/QuestionAudio" \
  --dst finetune/data/prepared_retrieval \
  --main-name danielle --no-eval
```

Expected stdout: one `conv 1685814793  partner=joshuarhodes  len=5079.0s  align_delta=0 frames` line and one `wrote danielle_joshuarhodes.wav`.

Verify:

```bash
python3 -c "
import soundfile as sf
i = sf.info('finetune/data/prepared_retrieval/danielle_joshuarhodes.wav')
print(i.samplerate, i.channels, i.subtype, round(i.duration/60, 2), 'min')
assert (i.samplerate, i.channels, i.subtype) == (24000, 2, 'PCM_24')
assert abs(i.duration - 84.65*60) < 1
print('OK')
"
```

Expected: `24000 2 PCM_24 84.65 min` then `OK`.

- [ ] **Step 6: Commit**

The WAV itself must not be committed. `.gitignore` currently covers
`finetune/data/prepared/*.wav` and `finetune/data/prepared_dialogue/*.wav` (lines 111-112)
but not the new directory, so add a matching line first:

```bash
printf 'finetune/data/prepared_retrieval/*.wav\n' >> .gitignore
git status --short   # confirm danielle_joshuarhodes.wav is no longer listed
```

```bash
git add .gitignore scripts/pair_dialogue_stereo.py tests/test_pair_dialogue_stereo.py
git commit -m "pair_dialogue_stereo: --no-eval for single whole conversations"
```

---

### Task 2: `--channel` and `--out-suffix` for annotate.py

`run()` hardcodes `channel=0` (`annotate.py:188`) and `out_file = path.with_suffix(".json")` (`annotate.py:176`). We need a second pass over channel 1 that does not clobber the channel-0 transcript.

**Files:**
- Modify: `annotate.py:150-190` (`run`), `annotate.py:199-212` (`Params`), `annotate.py:213-250` (`main`)
- Test: `tests/test_annotate_args.py` (create)

**Interfaces:**
- Consumes: the master WAV from Task 1.
- Produces: `out_path_for(path: Path, suffix: str) -> Path`, and the two transcript files
  `danielle_joshuarhodes.json` (channel 0) and `danielle_joshuarhodes.ch1.json` (channel 1)
  that Task 3 reads.

`annotate.py` imports torch, whisper_timestamped and julius at module scope, none of which resolve on the Mac, so the test exercises the naming helper in isolation by importing the file's source, not the module.

- [ ] **Step 1: Write the failing test**

Create `tests/test_annotate_args.py`:

```python
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_out_path_for():
    """annotate.py imports torch/whisper at module scope (A100-only), so exec just
    the helper's source in a bare namespace."""
    src = (REPO / "annotate.py").read_text()
    marker = "def out_path_for("
    start = src.index(marker)
    end = src.index("\ndef ", start + 1)
    ns: dict = {"Path": Path}
    exec(src[start:end], ns)
    return ns["out_path_for"]


def test_default_suffix_is_plain_json():
    out_path_for = _load_out_path_for()
    assert out_path_for(Path("/d/a.wav"), ".json") == Path("/d/a.json")


def test_channel_suffix_does_not_clobber_channel_zero():
    out_path_for = _load_out_path_for()
    assert out_path_for(Path("/d/a.wav"), ".ch1.json") == Path("/d/a.ch1.json")


def test_suffix_replaces_only_the_final_extension():
    out_path_for = _load_out_path_for()
    assert out_path_for(Path("/d/a.b.wav"), ".ch1.json") == Path("/d/a.b.ch1.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_annotate_args.py -v`
Expected: FAIL with `ValueError: substring not found` — `out_path_for` does not exist yet.

- [ ] **Step 3: Add the helper and thread the flags through**

In `annotate.py`, add above `def run(`:

```python
def out_path_for(path: Path, suffix: str) -> Path:
    """'/d/a.wav' + '.ch1.json' -> '/d/a.ch1.json'. Replaces only the final extension."""
    return path.with_suffix("").with_name(path.stem + suffix)
```

In `run()`, replace the two output-path lines:

```python
        out_file = out_path_for(path, params.out_suffix)
        err_file = out_file.with_suffix(out_file.suffix + ".err")
```

and pass the channel through the `process_one` call:

```python
            process_one(
                path,
                out_file,
                channel=params.channel,
                language=params.lang,
                w_model=w_model,
                params=params,
            )
```

In the `Params` dataclass add two fields with defaults so existing call sites keep working:

```python
    channel: int = 0
    out_suffix: str = ".json"
```

In `main()`'s parser add:

```python
    parser.add_argument("--channel", type=int, default=0,
                        help="audio channel to transcribe (0 = Danielle/main, 1 = partner)")
    parser.add_argument("--out-suffix", default=".json",
                        help="output suffix, e.g. .ch1.json so a second pass does not clobber .json")
```

and include `channel=args.channel, out_suffix=args.out_suffix` where `Params(...)` is constructed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_annotate_args.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add annotate.py tests/test_annotate_args.py
git commit -m "annotate: --channel and --out-suffix for two-channel transcription"
```

---

### Task 3: Merge word alignments into speaker-tagged utterances

**Files:**
- Create: `scripts/merge_utterances.py`
- Test: `tests/test_merge_utterances.py`

**Interfaces:**
- Consumes: the two transcript JSONs from Task 2. Each is `{"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]}`.
- Produces:
  - `Utterance = dict` with keys `speaker` (`"DANIELLE"` or `"JOSHUA"`), `start` (float), `end` (float), `text` (str).
  - `merge_utterances(ch0: list, ch1: list, gap: float = 0.6) -> list[Utterance]`
  - `format_transcript(utts: list[Utterance]) -> str`
  Task 4 and Task 7 both call `merge_utterances`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_merge_utterances.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_utterances import format_transcript, merge_utterances


def test_groups_words_within_gap_into_one_utterance():
    ch0 = [["hello", [1.0, 1.2], "SPEAKER_MAIN"], ["there", [1.3, 1.5], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, [], gap=0.6)
    assert len(utts) == 1
    assert utts[0]["speaker"] == "DANIELLE"
    assert utts[0]["text"] == "hello there"
    assert utts[0]["start"] == 1.0
    assert utts[0]["end"] == 1.5


def test_splits_on_gap_larger_than_threshold():
    ch0 = [["hello", [1.0, 1.2], "SPEAKER_MAIN"], ["later", [3.0, 3.2], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, [], gap=0.6)
    assert [u["text"] for u in utts] == ["hello", "later"]


def test_two_channels_interleave_in_time_order():
    ch0 = [["yes", [2.0, 2.2], "SPEAKER_MAIN"]]
    ch1 = [["question", [1.0, 1.4], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, ch1)
    assert [(u["speaker"], u["text"]) for u in utts] == [
        ("JOSHUA", "question"),
        ("DANIELLE", "yes"),
    ]


def test_speaker_change_breaks_utterance_even_within_gap():
    ch0 = [["a", [1.0, 1.1], "SPEAKER_MAIN"], ["c", [1.4, 1.5], "SPEAKER_MAIN"]]
    ch1 = [["b", [1.2, 1.3], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, ch1, gap=0.6)
    assert [(u["speaker"], u["text"]) for u in utts] == [
        ("DANIELLE", "a"), ("JOSHUA", "b"), ("DANIELLE", "c"),
    ]


def test_strips_whisper_leading_spaces():
    ch0 = [[" hello", [1.0, 1.2], "SPEAKER_MAIN"], [" there", [1.3, 1.5], "SPEAKER_MAIN"]]
    utts = merge_utterances(ch0, [])
    assert utts[0]["text"] == "hello there"


def test_format_transcript_is_timestamped_and_labelled():
    utts = [
        {"speaker": "JOSHUA", "start": 12.4, "end": 15.0, "text": "what is diwali"},
        {"speaker": "DANIELLE", "start": 19.1, "end": 24.0, "text": "the festival of lights"},
    ]
    text = format_transcript(utts)
    assert text.splitlines() == [
        "[00:12.4] JOSHUA: what is diwali",
        "[00:19.1] DANIELLE: the festival of lights",
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_merge_utterances.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'merge_utterances'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/merge_utterances.py`:

```python
#!/usr/bin/env python3
"""Merge two channels of Whisper word alignments into speaker-tagged utterances.

annotate.py writes {"alignments": [[word, [start, end], "SPEAKER_MAIN"], ...]} per channel.
Channel 0 is Danielle (the cloned voice), channel 1 is her partner. We interleave both in
time order and group consecutive same-speaker words separated by less than `gap` seconds.
"""
import json

DANIELLE = "DANIELLE"
JOSHUA = "JOSHUA"


def _words(alignments, speaker):
    out = []
    for word, (start, end), _spk in alignments:
        text = word.strip()
        if not text:
            continue
        out.append({"speaker": speaker, "start": float(start), "end": float(end), "text": text})
    return out


def merge_utterances(ch0, ch1, gap: float = 0.6):
    """Interleave both channels in time order, grouping same-speaker words < `gap` apart."""
    words = _words(ch0, DANIELLE) + _words(ch1, JOSHUA)
    words.sort(key=lambda w: (w["start"], w["speaker"]))

    utts = []
    for w in words:
        if utts and utts[-1]["speaker"] == w["speaker"] and w["start"] - utts[-1]["end"] < gap:
            utts[-1]["text"] += " " + w["text"]
            utts[-1]["end"] = w["end"]
        else:
            utts.append(dict(w))
    return utts


def _stamp(t: float) -> str:
    return f"{int(t) // 60:02d}:{t - 60 * (int(t) // 60):04.1f}"


def format_transcript(utts) -> str:
    return "\n".join(f"[{_stamp(u['start'])}] {u['speaker']}: {u['text']}" for u in utts)


def load_alignments(path):
    with open(path) as f:
        return json.load(f)["alignments"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_merge_utterances.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/merge_utterances.py tests/test_merge_utterances.py
git commit -m "merge_utterances: two-channel word alignments -> speaker-tagged utterances"
```

---

### Task 4: Snap and validate segments

Gemini's timestamps are not trusted. This task is the pure validation layer, written and tested before any network call exists.

**Files:**
- Create: `scripts/segment_retrieval_audio.py` (validation half only)
- Test: `tests/test_segment_retrieval_audio.py`

**Interfaces:**
- Consumes: `merge_utterances` output from Task 3.
- Produces, all called by Task 5:
  - `snap_bounds(start, end, utts) -> tuple[float, float]`
  - `validate_segments(segments, utts, max_sec=100.0, min_sec=4.0) -> tuple[list, list[str]]` returning `(kept, problems)`
  - `split_long(segment, utts, max_sec=100.0) -> list[dict]`

  A segment dict is `{"id": str, "start": float, "end": float, "turns": [{"speaker","start","end","kind","reference"}]}` where `kind` is `"grounded" | "decline" | "smalltalk"` and `reference` is a string or `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_segment_retrieval_audio.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from segment_retrieval_audio import snap_bounds, split_long, validate_segments


def _utts(spec):
    """spec: [(speaker, start, end)] -> utterance dicts with filler text."""
    return [{"speaker": s, "start": a, "end": b, "text": "word"} for s, a, b in spec]


def _seg(sid, start, end, turns):
    return {"id": sid, "start": start, "end": end, "turns": turns}


def _turn(speaker, start, end, kind="smalltalk", reference=None):
    return {"speaker": speaker, "start": start, "end": end, "kind": kind, "reference": reference}


def test_snap_moves_bounds_to_nearest_utterance_edges():
    utts = _utts([("JOSHUA", 10.0, 12.0), ("DANIELLE", 13.0, 20.0)])
    assert snap_bounds(10.4, 19.2, utts) == (10.0, 20.0)


def test_validate_drops_segment_shorter_than_min():
    utts = _utts([("JOSHUA", 0.0, 1.0), ("DANIELLE", 1.2, 2.0)])
    segs = [_seg("s0", 0.0, 2.0, [_turn("JOSHUA", 0.0, 1.0), _turn("DANIELLE", 1.2, 2.0)])]
    kept, problems = validate_segments(segs, utts, min_sec=4.0)
    assert kept == []
    assert any("too short" in p for p in problems)


def test_validate_drops_segment_with_no_danielle_turn():
    utts = _utts([("JOSHUA", 0.0, 9.0)])
    segs = [_seg("s0", 0.0, 9.0, [_turn("JOSHUA", 0.0, 9.0)])]
    kept, problems = validate_segments(segs, utts)
    assert kept == []
    assert any("no DANIELLE" in p for p in problems)


def test_validate_drops_overlapping_second_segment():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0),
                  ("JOSHUA", 8.0, 9.0), ("DANIELLE", 9.5, 20.0)])
    segs = [
        _seg("s0", 0.0, 10.0, [_turn("JOSHUA", 0.0, 2.0), _turn("DANIELLE", 2.5, 10.0)]),
        _seg("s1", 8.0, 20.0, [_turn("JOSHUA", 8.0, 9.0), _turn("DANIELLE", 9.5, 20.0)]),
    ]
    kept, problems = validate_segments(segs, utts)
    assert [s["id"] for s in kept] == ["s0"]
    assert any("overlap" in p for p in problems)


def test_validate_keeps_a_good_segment_unchanged():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0)])
    segs = [_seg("s0", 0.0, 10.0, [_turn("JOSHUA", 0.0, 2.0),
                                   _turn("DANIELLE", 2.5, 10.0, "grounded", "a passage")])]
    kept, problems = validate_segments(segs, utts)
    assert len(kept) == 1 and problems == []
    assert kept[0]["turns"][1]["reference"] == "a passage"


def test_split_long_cuts_at_joshua_turn_nearest_midpoint():
    turns = [
        _turn("JOSHUA", 0.0, 5.0), _turn("DANIELLE", 5.0, 60.0, "grounded", "ref A"),
        _turn("JOSHUA", 60.0, 65.0), _turn("DANIELLE", 65.0, 130.0, "grounded", "ref B"),
    ]
    utts = _utts([(t["speaker"], t["start"], t["end"]) for t in turns])
    halves = split_long(_seg("s0", 0.0, 130.0, turns), utts, max_sec=100.0)
    assert len(halves) == 2
    assert halves[0]["start"] == 0.0 and halves[0]["end"] == 60.0
    assert halves[1]["start"] == 60.0 and halves[1]["end"] == 130.0
    # each half keeps only the turns that fall inside it, with their own references
    assert [t["reference"] for t in halves[0]["turns"]] == [None, "ref A"]
    assert [t["reference"] for t in halves[1]["turns"]] == [None, "ref B"]


def test_validate_splits_over_long_segment_rather_than_dropping_it():
    turns = [
        _turn("JOSHUA", 0.0, 5.0), _turn("DANIELLE", 5.0, 60.0),
        _turn("JOSHUA", 60.0, 65.0), _turn("DANIELLE", 65.0, 130.0),
    ]
    utts = _utts([(t["speaker"], t["start"], t["end"]) for t in turns])
    kept, _ = validate_segments([_seg("s0", 0.0, 130.0, turns)], utts, max_sec=100.0)
    assert len(kept) == 2
    assert all(s["end"] - s["start"] <= 100.0 for s in kept)


def test_validate_reports_unclaimed_minutes_are_not_an_error():
    utts = _utts([("JOSHUA", 0.0, 2.0), ("DANIELLE", 2.5, 10.0),
                  ("JOSHUA", 500.0, 502.0), ("DANIELLE", 502.5, 510.0)])
    segs = [_seg("s0", 0.0, 10.0, [_turn("JOSHUA", 0.0, 2.0), _turn("DANIELLE", 2.5, 10.0)])]
    kept, problems = validate_segments(segs, utts)
    assert len(kept) == 1
    assert problems == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_segment_retrieval_audio.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'segment_retrieval_audio'`.

- [ ] **Step 3: Write the validation implementation**

Create `scripts/segment_retrieval_audio.py` with the header and validation half:

```python
#!/usr/bin/env python3
"""Stage 3 — segment the retrieval recording into per-dialogue units with per-turn labels.

Reads the two Whisper transcripts (channel 0 = Danielle, channel 1 = Joshua), merges them
into a speaker-tagged timestamped transcript, and asks Gemini to mark dialogue boundaries
and label each of HER turns grounded / decline / smalltalk. Grounded and decline turns get
a reference passage — the "retrieved document" the trainer encodes via the :8001 ARC encoder.

Per-turn (not per-segment) labelling is what lets one clip teach the mode switch
chit-chat -> factual -> chit-chat, which is the Stage 3 goal.

LLM timestamps are never trusted: every boundary snaps to a real utterance edge and is then
validated (monotonic, non-overlapping, starts on JOSHUA, ends on DANIELLE, 4-100 s).

    export GEMINI_API_KEY=...
    python3 scripts/segment_retrieval_audio.py \
        --ch0 finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
        --ch1 finetune/data/prepared_retrieval/danielle_joshuarhodes.ch1.json \
        --out replay/retrieval_segments.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_utterances import format_transcript, load_alignments, merge_utterances  # noqa: E402

MAX_SEC = 100.0
MIN_SEC = 4.0


def snap_bounds(start, end, utts):
    """Snap a proposed [start, end] to the nearest real utterance edges."""
    starts = [u["start"] for u in utts]
    ends = [u["end"] for u in utts]
    return (
        min(starts, key=lambda s: abs(s - start)),
        min(ends, key=lambda e: abs(e - end)),
    )


def _turns_in(turns, start, end):
    return [t for t in turns if t["start"] >= start - 1e-6 and t["end"] <= end + 1e-6]


def split_long(segment, utts, max_sec=MAX_SEC):
    """Split an over-long segment at the internal JOSHUA turn nearest its midpoint.

    Each half keeps only the turns inside it, with their own kinds and references.
    Recurses so a very long segment yields as many pieces as needed.
    """
    start, end = segment["start"], segment["end"]
    if end - start <= max_sec:
        return [segment]

    mid = (start + end) / 2.0
    candidates = [
        t for t in segment["turns"]
        if t["speaker"] == "JOSHUA" and start < t["start"] < end
    ]
    if not candidates:
        return [segment]          # nothing to split on; caller's validator drops it
    cut = min(candidates, key=lambda t: abs(t["start"] - mid))["start"]
    if cut <= start or cut >= end:
        return [segment]

    left = {"id": segment["id"] + "a", "start": start, "end": cut,
            "turns": _turns_in(segment["turns"], start, cut)}
    right = {"id": segment["id"] + "b", "start": cut, "end": end,
             "turns": _turns_in(segment["turns"], cut, end)}
    return split_long(left, utts, max_sec) + split_long(right, utts, max_sec)


def _check_one(seg, min_sec, max_sec):
    """Return a problem string, or None if the segment is usable."""
    dur = seg["end"] - seg["start"]
    if dur < min_sec:
        return f"{seg['id']}: too short ({dur:.1f}s < {min_sec}s)"
    if dur > max_sec:
        return f"{seg['id']}: too long ({dur:.1f}s > {max_sec}s) and unsplittable"
    if not any(t["speaker"] == "DANIELLE" for t in seg["turns"]):
        return f"{seg['id']}: no DANIELLE turn"
    if not seg["turns"] or seg["turns"][0]["speaker"] != "JOSHUA":
        return f"{seg['id']}: does not start on a JOSHUA turn"
    if seg["turns"][-1]["speaker"] != "DANIELLE":
        return f"{seg['id']}: does not end on a DANIELLE turn"
    return None


def validate_segments(segments, utts, max_sec=MAX_SEC, min_sec=MIN_SEC):
    """Snap, split, order and filter. Returns (kept, problems).

    Audio claimed by no segment is fine and is not reported as a problem — retakes and
    false starts are expected. The caller reports the unclaimed total separately.
    """
    problems = []
    snapped = []
    for seg in segments:
        s, e = snap_bounds(seg["start"], seg["end"], utts)
        if e <= s:
            problems.append(f"{seg['id']}: empty after snapping")
            continue
        snapped.append({**seg, "start": s, "end": e,
                        "turns": _turns_in(seg["turns"], s, e)})

    snapped.sort(key=lambda s: s["start"])

    kept = []
    last_end = float("-inf")
    for seg in snapped:
        if seg["start"] < last_end - 1e-6:
            problems.append(f"{seg['id']}: overlaps the previous segment; dropped")
            continue
        for piece in split_long(seg, utts, max_sec):
            problem = _check_one(piece, min_sec, max_sec)
            if problem:
                problems.append(problem)
                continue
            kept.append(piece)
            last_end = piece["end"]
    return kept, problems
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_segment_retrieval_audio.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/segment_retrieval_audio.py tests/test_segment_retrieval_audio.py
git commit -m "segment_retrieval_audio: snap/split/validate segment boundaries"
```

---

### Task 5: Gemini segmentation with per-turn labels and references

**Files:**
- Modify: `scripts/segment_retrieval_audio.py` (add the Gemini half and `main`)
- Test: `tests/test_segment_retrieval_audio.py` (add cases)

**Interfaces:**
- Consumes: `merge_utterances`, `format_transcript` (Task 3); `validate_segments` (Task 4).
- Produces:
  - `windows(utts, window_sec=720.0, overlap_sec=60.0) -> list[list[Utterance]]`
  - `dedupe(segments, tol=1.0) -> list[dict]`
  - `turn_mix(segments) -> dict` counting her turns by kind — the number that sets `RAG_TOKEN_WEIGHT`.
  - `replay/retrieval_segments.jsonl`, read by Task 7.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_segment_retrieval_audio.py`:

```python
from segment_retrieval_audio import dedupe, turn_mix, windows


def test_windows_overlap_so_boundary_dialogues_are_seen_whole():
    utts = [{"speaker": "JOSHUA", "start": float(t), "end": t + 0.5, "text": "w"}
            for t in range(0, 1500, 10)]
    ws = windows(utts, window_sec=720.0, overlap_sec=60.0)
    assert len(ws) >= 2
    # window 2 starts before window 1 ends
    assert ws[1][0]["start"] < ws[0][-1]["end"]
    # every utterance appears somewhere
    seen = {u["start"] for w in ws for u in w}
    assert seen == {u["start"] for u in utts}


def test_dedupe_drops_segments_repeated_across_window_overlap():
    segs = [
        {"id": "w0-1", "start": 700.0, "end": 750.0, "turns": []},
        {"id": "w1-0", "start": 700.3, "end": 750.2, "turns": []},
        {"id": "w1-1", "start": 800.0, "end": 850.0, "turns": []},
    ]
    out = dedupe(segs, tol=1.0)
    assert [s["start"] for s in out] == [700.0, 800.0]


def test_turn_mix_counts_only_her_turns():
    segs = [{
        "id": "s0", "start": 0.0, "end": 50.0,
        "turns": [
            {"speaker": "JOSHUA", "start": 0.0, "end": 2.0, "kind": "smalltalk", "reference": None},
            {"speaker": "DANIELLE", "start": 2.0, "end": 20.0, "kind": "grounded", "reference": "r"},
            {"speaker": "JOSHUA", "start": 20.0, "end": 22.0, "kind": "smalltalk", "reference": None},
            {"speaker": "DANIELLE", "start": 22.0, "end": 50.0, "kind": "smalltalk", "reference": None},
        ],
    }]
    mix = turn_mix(segs)
    assert mix == {"grounded": 1, "decline": 0, "smalltalk": 1, "total": 2, "retrieval_share": 0.5}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_segment_retrieval_audio.py -k "windows or dedupe or turn_mix" -v`
Expected: FAIL with `ImportError: cannot import name 'dedupe'`.

- [ ] **Step 3: Write the Gemini half**

Append to `scripts/segment_retrieval_audio.py`:

```python
SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "start": {"type": "NUMBER"},
                    "end": {"type": "NUMBER"},
                    "turns": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "speaker": {"type": "STRING"},
                                "start": {"type": "NUMBER"},
                                "end": {"type": "NUMBER"},
                                "kind": {"type": "STRING"},
                                "reference": {"type": "STRING"},
                            },
                            "required": ["speaker", "start", "end", "kind"],
                        },
                    },
                },
                "required": ["start", "end", "turns"],
            },
        },
    },
    "required": ["segments"],
}

PROMPT = """You are given a timestamped transcript of a real recorded conversation between
JOSHUA (asks questions) and DANIELLE (a warm, knowledgeable assistant who answers them).
Timestamps are [MM:SS.s] and are absolute seconds from the start of the recording.

Split the transcript into DIALOGUE UNITS. A unit is a self-contained stretch of conversation:
it begins with a JOSHUA turn and ends with a DANIELLE turn. Prefer units that contain a MODE
CHANGE — casual chat that turns into a factual question, or a factual answer that returns to
casual chat — because those teach the model to switch. Do not split a follow-up away from the
answer it follows up on. Units must not overlap. Aim for 20-90 seconds. Leave retakes, false
starts and dead air out of every unit.

Then label EACH of DANIELLE's turns:
  "grounded"  - she answers a question using specific external facts.
  "decline"   - she says she does not know / cannot answer, instead of guessing.
  "smalltalk" - casual conversation needing no external facts.
Label every JOSHUA turn "smalltalk" with no reference.

For each "grounded" turn write a "reference": a 2-5 sentence encyclopedic passage, in neutral
reference-work prose, that CONTAINS the facts she states. Include some surrounding detail she
did NOT mention, so the passage reads like a retrieved document rather than a restatement of
her answer. Never invent facts that contradict her.

For each "decline" turn write a "reference" about a RELATED BUT DIFFERENT topic — the passage
a retrieval system would have wrongly surfaced. It must not answer the question.

For "smalltalk" turns omit "reference".

Use the exact absolute-second values from the transcript for every start and end.

TRANSCRIPT:
{transcript}
"""


def call_gemini(prompt: str, model: str, key: str, retries: int = 4):
    """Same stdlib REST pattern as scripts/gen_recording_dialogues.py."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json",
                             "responseSchema": SCHEMA},
    }).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                resp = json.load(r)
            return json.loads(resp["candidates"][0]["content"]["parts"][0]["text"])
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            last = e
            print(f"  [retry {attempt+1}/{retries}] {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"gemini failed: {last}")


def windows(utts, window_sec=720.0, overlap_sec=60.0):
    """Slice utterances into overlapping time windows so no dialogue is cut by a window edge."""
    if not utts:
        return []
    end = max(u["end"] for u in utts)
    step = window_sec - overlap_sec
    out = []
    start = 0.0
    while start < end:
        stop = start + window_sec
        chunk = [u for u in utts if u["start"] >= start and u["start"] < stop]
        if chunk:
            out.append(chunk)
        start += step
    return out


def dedupe(segments, tol=1.0):
    """Drop segments repeated across a window overlap (same start within `tol` seconds)."""
    out = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        if out and abs(seg["start"] - out[-1]["start"]) <= tol:
            continue
        out.append(seg)
    return out


def turn_mix(segments):
    """Count HER turns by kind. retrieval_share sets RAG_TOKEN_WEIGHT (see the spec)."""
    counts = {"grounded": 0, "decline": 0, "smalltalk": 0}
    for seg in segments:
        for t in seg["turns"]:
            if t["speaker"] != "DANIELLE":
                continue
            counts[t["kind"]] = counts.get(t["kind"], 0) + 1
    total = sum(counts.values())
    share = (counts["grounded"] + counts["decline"]) / total if total else 0.0
    return {**counts, "total": total, "retrieval_share": round(share, 4)}


def normalize(raw_segments, window_index):
    """Gemini output -> our segment dicts, with ids and reference defaults."""
    out = []
    for i, seg in enumerate(raw_segments):
        turns = []
        for t in seg.get("turns", []):
            kind = t.get("kind", "smalltalk")
            if kind not in ("grounded", "decline", "smalltalk"):
                kind = "smalltalk"
            speaker = "DANIELLE" if t.get("speaker", "").upper().startswith("D") else "JOSHUA"
            if speaker == "JOSHUA":
                kind = "smalltalk"
            reference = t.get("reference") or None
            if kind == "smalltalk":
                reference = None
            turns.append({"speaker": speaker, "start": float(t["start"]),
                          "end": float(t["end"]), "kind": kind, "reference": reference})
        out.append({"id": f"ret-{window_index:02d}-{i:03d}",
                    "start": float(seg["start"]), "end": float(seg["end"]), "turns": turns})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch0", required=True, help="channel-0 (Danielle) alignments json")
    ap.add_argument("--ch1", required=True, help="channel-1 (Joshua) alignments json")
    ap.add_argument("--out", default="replay/retrieval_segments.jsonl")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--window-sec", type=float, default=720.0)
    ap.add_argument("--overlap-sec", type=float, default=60.0)
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (use a rotated key; never pass it on the command line)")

    utts = merge_utterances(load_alignments(args.ch0), load_alignments(args.ch1))
    print(f"{len(utts)} utterances, {max(u['end'] for u in utts) / 60:.1f} min")

    raw = []
    wins = windows(utts, args.window_sec, args.overlap_sec)
    for wi, chunk in enumerate(wins):
        print(f"  window {wi + 1}/{len(wins)}  "
              f"[{chunk[0]['start'] / 60:.1f}-{chunk[-1]['end'] / 60:.1f} min]")
        resp = call_gemini(PROMPT.format(transcript=format_transcript(chunk)), args.model, key)
        raw += normalize(resp.get("segments", []), wi)

    kept, problems = validate_segments(dedupe(raw), utts)

    claimed = sum(s["end"] - s["start"] for s in kept)
    total = max(u["end"] for u in utts)
    mix = turn_mix(kept)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for seg in kept:
            f.write(json.dumps(seg) + "\n")

    for p in problems:
        print(f"  DROP {p}")
    print(f"\nwrote {args.out}: {len(kept)} segments, {len(problems)} dropped")
    print(f"kept {claimed / 60:.1f} min of {total / 60:.1f} min "
          f"({(total - claimed) / 60:.1f} min unclaimed)")
    print(f"her turns: {mix}")
    print(f"-> retrieval_share {mix['retrieval_share']:.2f}: "
          f"use RAG_TOKEN_WEIGHT 15 if smalltalk+decline >= 30% of her turns, else 8-10")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_segment_retrieval_audio.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/segment_retrieval_audio.py tests/test_segment_retrieval_audio.py
git commit -m "segment_retrieval_audio: Gemini per-turn labelling + reference passages"
```

---

### Task 6: Faithfulness filter for generated references

**Files:**
- Create: `scripts/filter_retrieval_segments.py`
- Test: `tests/test_filter_retrieval_segments.py`

**Interfaces:**
- Consumes: `replay/retrieval_segments.jsonl` from Task 5.
- Produces: `replay/retrieval_segments.filtered.jsonl` (read by Task 7) and `replay/retrieval_dropped.jsonl`.
  - `apply_verdicts(segments, verdicts) -> tuple[list, list]` returning `(kept, dropped)`, where `verdicts` maps `"<segment_id>#<turn_index>"` to `True`/`False`.

Only `grounded` turns are judged. A `decline` reference is *supposed* to be unrelated, and `smalltalk` has none — judging either would drop good data.

- [ ] **Step 1: Write the failing test**

Create `tests/test_filter_retrieval_segments.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from filter_retrieval_segments import apply_verdicts, judgeable_turns


def _seg(sid, turns):
    return {"id": sid, "start": 0.0, "end": 50.0, "turns": turns}


def _t(speaker, kind, reference=None, text="said"):
    return {"speaker": speaker, "start": 0.0, "end": 1.0,
            "kind": kind, "reference": reference, "text": text}


def test_only_grounded_turns_are_judged():
    segs = [_seg("s0", [
        _t("JOSHUA", "smalltalk"),
        _t("DANIELLE", "grounded", "a passage"),
        _t("DANIELLE", "decline", "an unrelated passage"),
        _t("DANIELLE", "smalltalk"),
    ])]
    assert judgeable_turns(segs) == [("s0", 1)]


def test_unfaithful_grounded_turn_is_demoted_not_deleted():
    segs = [_seg("s0", [
        _t("JOSHUA", "smalltalk"),
        _t("DANIELLE", "grounded", "wrong-topic passage"),
        _t("DANIELLE", "smalltalk"),
    ])]
    kept, dropped = apply_verdicts(segs, {"s0#1": False})
    assert len(kept) == 1
    # demoted to smalltalk so the clip survives but emits no <RAG>
    assert kept[0]["turns"][1]["kind"] == "smalltalk"
    assert kept[0]["turns"][1]["reference"] is None
    assert dropped == [{"segment": "s0", "turn": 1, "reason": "unfaithful"}]


def test_faithful_turn_is_untouched():
    segs = [_seg("s0", [_t("JOSHUA", "smalltalk"), _t("DANIELLE", "grounded", "good passage")])]
    kept, dropped = apply_verdicts(segs, {"s0#1": True})
    assert kept[0]["turns"][1]["kind"] == "grounded"
    assert kept[0]["turns"][1]["reference"] == "good passage"
    assert dropped == []


def test_segment_with_every_retrieval_turn_demoted_is_still_kept():
    segs = [_seg("s0", [_t("JOSHUA", "smalltalk"), _t("DANIELLE", "grounded", "bad")])]
    kept, _ = apply_verdicts(segs, {"s0#1": False})
    assert len(kept) == 1
    assert all(t["kind"] == "smalltalk" for t in kept[0]["turns"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_filter_retrieval_segments.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'filter_retrieval_segments'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/filter_retrieval_segments.py`:

```python
#!/usr/bin/env python3
"""Stage 3 — faithfulness filter over the generated reference passages.

References here are reverse-engineered from what Danielle actually said, so they should
almost always be faithful; this catches Gemini writing a passage about the wrong topic.

An unfaithful grounded turn is DEMOTED to smalltalk (no <RAG>, no reference) rather than
deleting the clip: the audio is still good natural conversation in her voice, and keeping
it preserves the surrounding turn-taking. Only `grounded` turns are judged — a `decline`
reference is deliberately unrelated and `smalltalk` has none.

    export GEMINI_API_KEY=...
    python3 scripts/filter_retrieval_segments.py \
        --in replay/retrieval_segments.jsonl \
        --out replay/retrieval_segments.filtered.jsonl \
        --dropped replay/retrieval_dropped.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_retrieval_audio import call_gemini  # noqa: E402

VERDICT_SCHEMA = {
    "type": "OBJECT",
    "properties": {"supported": {"type": "BOOLEAN"}},
    "required": ["supported"],
}

JUDGE_PROMPT = """Here is a passage and a spoken answer.

PASSAGE:
{reference}

SPOKEN ANSWER:
{answer}

Is every factual claim in the spoken answer supported by the passage? Answer with
{{"supported": true}} or {{"supported": false}}. Casual phrasing, greetings and
follow-up questions in the answer do not count as factual claims.
"""


def judgeable_turns(segments):
    """[(segment_id, turn_index)] for grounded turns that have a reference."""
    out = []
    for seg in segments:
        for i, t in enumerate(seg["turns"]):
            if t["speaker"] == "DANIELLE" and t["kind"] == "grounded" and t.get("reference"):
                out.append((seg["id"], i))
    return out


def apply_verdicts(segments, verdicts):
    """Demote turns whose verdict is False. Returns (kept_segments, dropped_records)."""
    kept, dropped = [], []
    for seg in segments:
        turns = [dict(t) for t in seg["turns"]]
        for i, t in enumerate(turns):
            key = f"{seg['id']}#{i}"
            if key in verdicts and verdicts[key] is False:
                t["kind"] = "smalltalk"
                t["reference"] = None
                dropped.append({"segment": seg["id"], "turn": i, "reason": "unfaithful"})
        kept.append({**seg, "turns": turns})
    return kept, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="replay/retrieval_segments.jsonl")
    ap.add_argument("--out", default="replay/retrieval_segments.filtered.jsonl")
    ap.add_argument("--dropped", default="replay/retrieval_dropped.jsonl")
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("set GEMINI_API_KEY (use a rotated key)")

    segments = [json.loads(line) for line in open(args.inp) if line.strip()]
    by_id = {s["id"]: s for s in segments}

    verdicts = {}
    targets = judgeable_turns(segments)
    for n, (sid, ti) in enumerate(targets, 1):
        turn = by_id[sid]["turns"][ti]
        prompt = JUDGE_PROMPT.format(reference=turn["reference"], answer=turn.get("text", ""))
        resp = call_gemini(prompt, args.model, key)
        verdicts[f"{sid}#{ti}"] = bool(resp.get("supported", True))
        if n % 20 == 0 or n == len(targets):
            print(f"  judged {n}/{len(targets)}")

    kept, dropped = apply_verdicts(segments, verdicts)

    with open(args.out, "w") as f:
        for seg in kept:
            f.write(json.dumps(seg) + "\n")
    with open(args.dropped, "w") as f:
        for rec in dropped:
            f.write(json.dumps(rec) + "\n")

    print(f"wrote {args.out}: {len(kept)} segments, {len(dropped)} turns demoted -> {args.dropped}")


if __name__ == "__main__":
    main()
```

Note: `call_gemini` in Task 5 passes `SCHEMA` as the response schema. Change its signature there to `call_gemini(prompt, model, key, schema=SCHEMA, retries=4)` and use `schema` in the request body, then call it here as `call_gemini(prompt, args.model, key, schema=VERDICT_SCHEMA)`. Update the Task 5 call site to `call_gemini(..., key)` unchanged (the default keeps working).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_filter_retrieval_segments.py tests/test_segment_retrieval_audio.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/filter_retrieval_segments.py scripts/segment_retrieval_audio.py tests/test_filter_retrieval_segments.py
git commit -m "filter_retrieval_segments: demote unfaithful grounded turns to smalltalk"
```

---

### Task 7: Cut clips with per-turn `<RAG>` markers

**Files:**
- Create: `scripts/cut_retrieval_clips.py`
- Test: `tests/test_cut_retrieval_clips.py`

**Interfaces:**
- Consumes: the master WAV (Task 1), master ch0 alignments (Task 2), `retrieval_segments.filtered.jsonl` (Task 6).
- Produces, all read by Task 8 and the trainer:
  - `slice_alignments(alignments, start, end) -> list` — offset-sliced word alignments.
  - `insert_rag_markers(alignments, retrieval_turn_starts) -> list` — one `<RAG>` per retrieval turn.
  - `replay/retrieval/<id>.wav`, `replay/retrieval/<id>.json`, `replay/retrieval_manifest.jsonl` with rows `{"id", "path", "duration_sec", "references": [...], "kinds": [...]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cut_retrieval_clips.py`:

```python
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from cut_retrieval_clips import cut_all, insert_rag_markers, slice_alignments


def _al(word, start, end):
    return [word, [start, end], "SPEAKER_MAIN"]


def test_slice_keeps_only_words_inside_and_rebases_to_zero():
    alignments = [_al("a", 1.0, 1.5), _al("b", 10.0, 10.5), _al("c", 12.0, 12.5),
                  _al("d", 30.0, 30.5)]
    out = slice_alignments(alignments, 10.0, 13.0)
    assert [w[0] for w in out] == ["b", "c"]
    assert out[0][1] == [0.0, 0.5]
    assert out[1][1] == [2.0, 2.5]


def test_slice_excludes_words_straddling_the_boundary():
    alignments = [_al("straddle", 9.5, 10.5), _al("inside", 11.0, 11.5)]
    out = slice_alignments(alignments, 10.0, 13.0)
    assert [w[0] for w in out] == ["inside"]


def test_inserts_one_rag_marker_before_each_retrieval_turn():
    alignments = [_al("hi", 0.0, 0.5), _al("answer", 5.0, 5.5), _al("more", 20.0, 20.5)]
    out = insert_rag_markers(alignments, [5.0, 20.0])
    assert [w[0] for w in out] == ["hi", "<RAG>", "answer", "<RAG>", "more"]
    assert out[1][1] == [4.9, 5.0]
    assert out[3][1] == [19.9, 20.0]
    assert out[1][2] == "SPEAKER_MAIN"


def test_no_markers_when_no_retrieval_turns():
    alignments = [_al("hi", 0.0, 0.5)]
    assert insert_rag_markers(alignments, []) == alignments


def test_marker_uses_first_word_at_or_after_the_turn_start():
    # Whisper's word start rarely equals the segmenter's turn start exactly
    alignments = [_al("hi", 0.0, 0.5), _al("well", 5.2, 5.6)]
    out = insert_rag_markers(alignments, [5.0])
    assert [w[0] for w in out] == ["hi", "<RAG>", "well"]
    assert out[1][1] == [5.1, 5.2]


def test_marker_clamps_at_zero_for_a_turn_at_clip_start():
    alignments = [_al("answer", 0.05, 0.5)]
    out = insert_rag_markers(alignments, [0.0])
    assert out[0][0] == "<RAG>"
    assert out[0][1][0] == 0.0


def _write_master(path, seconds=60.0, sr=24000):
    n = int(seconds * sr)
    left = np.linspace(-0.5, 0.5, n, dtype="float32")
    right = np.full(n, -0.25, dtype="float32")
    sf.write(str(path), np.column_stack([left, right]), sr, subtype="PCM_24")


def _segment(sid, start, end, turns):
    return {"id": sid, "start": start, "end": end, "turns": turns}


def _turn(speaker, start, end, kind, reference=None):
    return {"speaker": speaker, "start": start, "end": end, "kind": kind, "reference": reference}


def test_cut_all_writes_clip_json_and_manifest(tmp_path):
    master = tmp_path / "master.wav"
    _write_master(master)
    alignments = [_al("hello", 11.0, 11.5), _al("fact", 14.0, 14.5), _al("bye", 18.0, 18.5)]
    segments = [_segment("ret-0", 10.0, 20.0, [
        _turn("JOSHUA", 10.0, 13.0, "smalltalk"),
        _turn("DANIELLE", 14.0, 20.0, "grounded", "a reference passage"),
    ])]
    outdir = tmp_path / "clips"
    manifest = tmp_path / "manifest.jsonl"
    cut_all(master, alignments, segments, outdir, manifest)

    info = sf.info(str(outdir / "ret-0.wav"))
    assert info.channels == 2 and info.samplerate == 24000 and info.subtype == "PCM_24"
    assert info.frames == 10 * 24000

    data = json.load(open(outdir / "ret-0.json"))
    words = [w[0] for w in data["alignments"]]
    assert words == ["hello", "<RAG>", "fact", "bye"]

    row = json.loads(manifest.read_text().strip())
    assert row["id"] == "ret-0"
    assert row["references"] == ["a reference passage"]
    assert row["kinds"] == ["grounded"]
    assert abs(row["duration_sec"] - 10.0) < 1e-6
    assert Path(row["path"]).is_absolute()


def test_cut_all_marker_count_matches_reference_count(tmp_path):
    master = tmp_path / "master.wav"
    _write_master(master)
    alignments = [_al("one", 12.0, 12.5), _al("two", 30.0, 30.5)]
    segments = [_segment("ret-0", 10.0, 40.0, [
        _turn("JOSHUA", 10.0, 11.0, "smalltalk"),
        _turn("DANIELLE", 12.0, 20.0, "grounded", "ref one"),
        _turn("JOSHUA", 20.0, 29.0, "smalltalk"),
        _turn("DANIELLE", 30.0, 40.0, "decline", "ref two"),
    ])]
    outdir = tmp_path / "clips"
    manifest = tmp_path / "manifest.jsonl"
    cut_all(master, alignments, segments, outdir, manifest)

    data = json.load(open(outdir / "ret-0.json"))
    n_markers = sum(1 for w in data["alignments"] if w[0] == "<RAG>")
    row = json.loads(manifest.read_text().strip())
    assert n_markers == len(row["references"]) == 2
    assert row["references"] == ["ref one", "ref two"]


def test_cut_all_skips_a_segment_whose_marker_count_would_mismatch(tmp_path):
    master = tmp_path / "master.wav"
    _write_master(master)
    # no Danielle word at all inside the grounded turn -> no marker can be placed
    alignments = [_al("only", 11.0, 11.5)]
    segments = [_segment("ret-0", 10.0, 20.0, [
        _turn("JOSHUA", 10.0, 11.0, "smalltalk"),
        _turn("DANIELLE", 14.0, 20.0, "grounded", "a reference"),
    ])]
    outdir = tmp_path / "clips"
    manifest = tmp_path / "manifest.jsonl"
    cut_all(master, alignments, segments, outdir, manifest)
    assert not (outdir / "ret-0.wav").exists()
    assert manifest.read_text().strip() == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cut_retrieval_clips.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cut_retrieval_clips'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/cut_retrieval_clips.py`:

```python
#!/usr/bin/env python3
"""Stage 3 — cut per-dialogue clips from the retrieval master, with per-turn <RAG> markers.

For each validated segment: cut the stereo audio, slice the master's channel-0 alignments to
the segment and rebase them to zero, then insert a "<RAG>" alignment before the first word of
EACH grounded/decline turn. interleaver._tokenize maps "<RAG>" to raw token 4, so the model
learns to emit the retrieval trigger at the moment a factual question is answered — not once
at the head of the clip, which is what the synthetic terminal Q->A data taught.

The i-th <RAG> marker corresponds to the i-th entry of the manifest's "references" list.
train.py pairs them positionally, so a segment where they would disagree is skipped.

    python3 scripts/cut_retrieval_clips.py \
        --master finetune/data/prepared_retrieval/danielle_joshuarhodes.wav \
        --alignments finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
        --segments replay/retrieval_segments.filtered.jsonl \
        --outdir replay/retrieval --manifest replay/retrieval_manifest.jsonl
"""
import argparse
import json
import os
from pathlib import Path

import soundfile as sf

TARGET_SR = 24000
RETRIEVAL_KINDS = ("grounded", "decline")


def slice_alignments(alignments, start, end):
    """Words fully inside [start, end), rebased so the clip begins at 0.0."""
    out = []
    for word, (ws, we), speaker in alignments:
        if ws >= start and we <= end:
            out.append([word, [round(ws - start, 3), round(we - start, 3)], speaker])
    return out


def insert_rag_markers(alignments, retrieval_turn_starts):
    """Insert a <RAG> alignment before the first word at or after each retrieval turn start.

    Timestamps are clip-relative. Each marker occupies the 0.1 s before the word it precedes,
    clamped at 0.0.
    """
    if not retrieval_turn_starts:
        return alignments

    anchors = []
    for turn_start in sorted(retrieval_turn_starts):
        for i, (_w, (ws, _we), _s) in enumerate(alignments):
            if ws >= turn_start - 1e-6:
                anchors.append(i)
                break

    out = []
    for i, item in enumerate(alignments):
        for _ in range(anchors.count(i)):
            word_start = item[1][0]
            out.append(["<RAG>", [round(max(0.0, word_start - 0.1), 3), word_start],
                        "SPEAKER_MAIN"])
        out.append(item)
    return out


def _retrieval_turns(segment):
    return [t for t in segment["turns"]
            if t["speaker"] == "DANIELLE" and t["kind"] in RETRIEVAL_KINDS and t.get("reference")]


def cut_all(master_path, alignments, segments, outdir, manifest_path):
    """Cut every segment. Returns the list of manifest rows written."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    info = sf.info(str(master_path))
    if info.samplerate != TARGET_SR or info.channels != 2:
        raise SystemExit(f"{master_path}: expected 2ch/{TARGET_SR} Hz, got "
                         f"{info.channels}ch/{info.samplerate} Hz")

    rows = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        clip_alignments = slice_alignments(alignments, start, end)
        if not clip_alignments:
            print(f"  SKIP {seg['id']}: no Danielle words inside")
            continue

        turns = _retrieval_turns(seg)
        turn_starts = [t["start"] - start for t in turns]
        references = [t["reference"] for t in turns]
        kinds = [t["kind"] for t in turns]

        marked = insert_rag_markers(clip_alignments, turn_starts)
        n_markers = sum(1 for w in marked if w[0] == "<RAG>")
        if n_markers != len(references):
            print(f"  SKIP {seg['id']}: {n_markers} <RAG> markers vs {len(references)} references")
            continue

        frames_start = int(round(start * TARGET_SR))
        frames_stop = int(round(end * TARGET_SR))
        audio, _sr = sf.read(str(master_path), start=frames_start, stop=frames_stop,
                             always_2d=True)
        wav_path = outdir / f"{seg['id']}.wav"
        sf.write(str(wav_path), audio, TARGET_SR, subtype="PCM_24")

        with open(outdir / f"{seg['id']}.json", "w") as f:
            json.dump({"alignments": marked}, f, ensure_ascii=False)

        rows.append({
            "id": seg["id"],
            "path": str(wav_path.resolve()),
            "duration_sec": audio.shape[0] / TARGET_SR,
            "references": references,
            "kinds": kinds,
        })

    with open(manifest_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True)
    ap.add_argument("--alignments", required=True)
    ap.add_argument("--segments", default="replay/retrieval_segments.filtered.jsonl")
    ap.add_argument("--outdir", default="replay/retrieval")
    ap.add_argument("--manifest", default="replay/retrieval_manifest.jsonl")
    args = ap.parse_args()

    with open(args.alignments) as f:
        alignments = json.load(f)["alignments"]
    segments = [json.loads(line) for line in open(args.segments) if line.strip()]

    rows = cut_all(args.master, alignments, segments, args.outdir, args.manifest)

    total = sum(r["duration_sec"] for r in rows)
    n_ret = sum(len(r["references"]) for r in rows)
    multi = sum(1 for r in rows if len(r["references"]) > 1)
    print(f"\nwrote {len(rows)} clips ({total / 60:.1f} min) -> {args.outdir}")
    print(f"{n_ret} retrieval turns; {multi} clips have more than one")
    print(f"manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_cut_retrieval_clips.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

The cut clips are large binaries and `replay/` is not ignored today, so exclude them first:

```bash
printf 'replay/retrieval/\n' >> .gitignore
```

```bash
git add .gitignore scripts/cut_retrieval_clips.py tests/test_cut_retrieval_clips.py
git commit -m "cut_retrieval_clips: per-turn <RAG> markers + ordered reference manifest"
```

---

### Task 8: N-tensor reference file format

`interleaver._load_reference_tensor` reads a single `"reference"` key. Multi-retrieval clips need N, and the loader must not import moshi in a test.

**Files:**
- Create: `finetune/data/reference_io.py`
- Modify: `finetune/data/interleaver.py:17-25`, `:28-33`, `:36-41`
- Modify: `scripts/precompute_references.py:50-56`
- Test: `tests/test_reference_io.py`

**Interfaces:**
- Consumes: `replay/retrieval_manifest.jsonl` (Task 7).
- Produces:
  - `save_references(path: str, tensors: list[torch.Tensor]) -> None` writing keys `reference_0 … reference_{N-1}`.
  - `load_references(wav_path: str) -> list[torch.Tensor] | None` reading `<stem>.ref.safetensors`.
  Task 9 consumes `load_references`' list-of-tensors return.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reference_io.py`:

```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.data.reference_io import load_references, save_references


def test_roundtrip_preserves_order(tmp_path):
    ref_path = tmp_path / "clip.ref.safetensors"
    a = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    b = torch.arange(6, 12, dtype=torch.float32).reshape(3, 2)
    save_references(str(ref_path), [a, b])
    out = load_references(str(tmp_path / "clip.wav"))
    assert len(out) == 2
    assert torch.equal(out[0], a)
    assert torch.equal(out[1], b)


def test_order_holds_past_ten_references(tmp_path):
    """Guards against lexicographic key sorting putting reference_10 before reference_2."""
    ref_path = tmp_path / "clip.ref.safetensors"
    tensors = [torch.full((1, 2), float(i)) for i in range(12)]
    save_references(str(ref_path), tensors)
    out = load_references(str(tmp_path / "clip.wav"))
    assert [float(t[0, 0]) for t in out] == [float(i) for i in range(12)]


def test_missing_file_returns_none(tmp_path):
    assert load_references(str(tmp_path / "absent.wav")) is None


def test_legacy_single_reference_key_still_loads(tmp_path):
    """Synthetic replay clips from Stage 2b use the old single 'reference' key."""
    from safetensors.torch import save_file
    t = torch.ones(2, 3)
    save_file({"reference": t.contiguous()}, str(tmp_path / "clip.ref.safetensors"))
    out = load_references(str(tmp_path / "clip.wav"))
    assert len(out) == 1
    assert torch.equal(out[0], t)


def test_empty_list_writes_nothing(tmp_path):
    save_references(str(tmp_path / "clip.ref.safetensors"), [])
    assert not (tmp_path / "clip.ref.safetensors").exists()
    assert load_references(str(tmp_path / "clip.wav")) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reference_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finetune.data.reference_io'`.

- [ ] **Step 3: Write the module and rewire its two callers**

Create `finetune/data/reference_io.py`:

```python
"""Read/write the per-clip precomputed reference_with_time tensors.

One clip can contain several retrieval turns, so a clip's .ref.safetensors holds N tensors
under keys reference_0 .. reference_{N-1}, in the same order as the clip's <RAG> markers.
train.py pairs the i-th marker with the i-th tensor.

Deliberately free of any `moshi` import so it can be unit-tested off the A100.
"""
import os

from safetensors.torch import load_file, save_file

KEY = "reference"


def ref_path_for(wav_path) -> str:
    return os.path.splitext(str(wav_path))[0] + ".ref.safetensors"


def save_references(path: str, tensors: list) -> None:
    """Write N tensors as reference_0 .. reference_{N-1}. An empty list writes no file."""
    if not tensors:
        return
    save_file({f"{KEY}_{i}": t.contiguous() for i, t in enumerate(tensors)}, path)


def load_references(wav_path):
    """Return the clip's reference tensors in marker order, or None if it has none.

    Accepts the legacy single-`reference`-key files written for the Stage 2b synthetic set.
    """
    path = ref_path_for(wav_path)
    if not os.path.exists(path):
        return None
    tensors = load_file(path)
    if KEY in tensors:                      # legacy single-reference format
        return [tensors[KEY]]
    indexed = []
    for key, value in tensors.items():
        if key.startswith(KEY + "_"):
            indexed.append((int(key[len(KEY) + 1:]), value))
    if not indexed:
        return None
    indexed.sort(key=lambda kv: kv[0])      # numeric, not lexicographic
    return [value for _i, value in indexed]
```

In `finetune/data/interleaver.py`, replace lines 17-25 with:

```python
from .reference_io import load_references as _load_reference_tensors
```

placed with the other imports, and update the two dataclass field comments and the call site:

```python
@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    # list of [T_ref, D] precomputed reference_with_time, one per <RAG> marker; None for
    # plain dialogue with no retrieval
    reference_tensor: list | None = None
```

```python
@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    # per-example list of [T_ref, D] (or None); carries replay reference_with_time
    reference_tensors: list | None = None
```

At line 315 change the call:

```python
            reference_tensor = _load_reference_tensors(path)
```

`Batch.collate` (lines 51-56) needs no change — it already just gathers whatever each sample carries.

In `scripts/precompute_references.py`, replace the single-key write:

```python
        tensor = embed(args.url, rec["reference"])
        out = os.path.join(args.audiodir, f"{rec['id']}.ref.safetensors")
        save_file({"reference": tensor.contiguous()}, out)
```

with a loop over the manifest's `references` list, using the shared writer:

```python
        refs = rec.get("references") or ([rec["reference"]] if rec.get("reference") else [])
        if not refs:
            n_skip += 1
            continue
        tensors = [embed(args.url, text) for text in refs]
        out = os.path.join(args.audiodir, f"{rec['id']}.ref.safetensors")
        save_references(out, tensors)
        n_ref += len(tensors)
```

Replace its `from safetensors.torch import save_file` import with:

```python
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune.data.reference_io import save_references  # noqa: E402
```

(`import sys` is already present.) Also relax the `if not rec.get("retrieval")` guard at line 47 to `if not (rec.get("references") or rec.get("reference")):` so the new manifest shape, which has no `retrieval` boolean, is handled.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_reference_io.py -v`
Expected: 5 passed.

Confirm no test imports moshi:

Run: `python3 -m pytest tests/ -v`
Expected: all pass, no `ModuleNotFoundError: No module named 'moshi'`.

- [ ] **Step 5: Commit**

```bash
git add finetune/data/reference_io.py finetune/data/interleaver.py scripts/precompute_references.py tests/test_reference_io.py
git commit -m "reference_io: N-tensor .ref.safetensors for multi-retrieval clips"
```

---

### Task 9: Condition every `⟨ret⟩` frame, not just the first

`train.py:272-273` takes `start = int(hit[0])` — the first `⟨ret⟩` frame only. Every later retrieval turn in a clip gets no conditioning while the loss still demands a grounded answer, which trains confabulation. Extracting the logic into a moshi-free module also makes it testable off the A100.

**Files:**
- Create: `finetune/data/reference_injection.py`
- Modify: `train.py:254-279`
- Modify: `scripts/apply_rag_positioning.py:58-107`
- Test: `tests/test_reference_injection.py`

**Interfaces:**
- Consumes: `Batch.reference_tensors` as a list-of-lists (Task 8).
- Produces: `build_reference_condition(text_row, refs, dim, dtype, rag_token_id=4) -> tuple[Tensor, Tensor]` returning `(ref_cond [B, S, D], ref_mask [B, S])`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_reference_injection.py`:

```python
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.data.reference_injection import build_reference_condition

RAG = 4
DIM = 3


def _text_row(rows):
    return torch.tensor(rows, dtype=torch.long)


def test_single_reference_lands_at_the_rag_frame():
    text_row = _text_row([[0, 0, RAG, 0, 0, 0]])
    ref = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [[ref]], DIM, torch.float32)
    assert cond.shape == (1, 6, DIM) and mask.shape == (1, 6)
    assert mask.tolist() == [[0, 0, 1, 1, 0, 0]]
    assert torch.equal(cond[0, 2], torch.ones(DIM))
    assert torch.equal(cond[0, 0], torch.zeros(DIM))


def test_second_reference_lands_at_the_second_rag_frame():
    text_row = _text_row([[0, RAG, 0, 0, RAG, 0, 0, 0]])
    a = torch.full((2, DIM), 1.0)
    b = torch.full((2, DIM), 2.0)
    cond, mask = build_reference_condition(text_row, [[a, b]], DIM, torch.float32)
    assert mask.tolist() == [[0, 1, 1, 0, 1, 1, 0, 0]]
    assert torch.equal(cond[0, 1], torch.full((DIM,), 1.0))
    assert torch.equal(cond[0, 4], torch.full((DIM,), 2.0))


def test_long_reference_is_clamped_at_the_next_rag_frame():
    text_row = _text_row([[RAG, 0, RAG, 0, 0, 0]])
    a = torch.full((10, DIM), 1.0)      # would run past the second marker
    b = torch.full((2, DIM), 2.0)
    cond, mask = build_reference_condition(text_row, [[a, b]], DIM, torch.float32)
    assert mask.tolist() == [[1, 1, 1, 1, 0, 0]]
    assert torch.equal(cond[0, 1], torch.full((DIM,), 1.0))
    assert torch.equal(cond[0, 2], torch.full((DIM,), 2.0))   # not overwritten by a


def test_reference_is_truncated_at_sequence_end():
    text_row = _text_row([[0, 0, 0, RAG]])
    ref = torch.ones(5, DIM)
    cond, mask = build_reference_condition(text_row, [[ref]], DIM, torch.float32)
    assert mask.tolist() == [[0, 0, 0, 1]]
    assert torch.equal(cond[0, 3], torch.ones(DIM))


def test_example_with_no_references_stays_zero():
    text_row = _text_row([[0, RAG, 0, 0], [0, 0, 0, 0]])
    ref = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [[ref], None], DIM, torch.float32)
    assert mask.tolist() == [[0, 1, 1, 0], [0, 0, 0, 0]]
    assert torch.equal(cond[1], torch.zeros(4, DIM))


def test_more_markers_than_references_uses_the_pairs_that_match():
    text_row = _text_row([[RAG, 0, RAG, 0]])
    a = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [[a]], DIM, torch.float32)
    # first marker conditioned, second left alone rather than reusing a
    assert mask.tolist() == [[1, 1, 0, 0]]


def test_more_references_than_markers_uses_the_pairs_that_match():
    text_row = _text_row([[0, RAG, 0, 0]])
    a = torch.full((1, DIM), 1.0)
    b = torch.full((1, DIM), 2.0)
    cond, mask = build_reference_condition(text_row, [[a, b]], DIM, torch.float32)
    assert mask.tolist() == [[0, 1, 0, 0]]
    assert torch.equal(cond[0, 1], torch.full((DIM,), 1.0))


def test_example_with_no_marker_falls_back_to_the_prefix():
    """Legacy synthetic clips whose <RAG> was stripped still get conditioned at frame 0."""
    text_row = _text_row([[0, 0, 0, 0]])
    ref = torch.ones(2, DIM)
    cond, mask = build_reference_condition(text_row, [[ref]], DIM, torch.float32)
    assert mask.tolist() == [[1, 1, 0, 0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_reference_injection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'finetune.data.reference_injection'`.

- [ ] **Step 3: Write the module**

Create `finetune/data/reference_injection.py`:

```python
"""Build the reference_with_time condition tensor for a training batch.

At serve time the model emits ⟨ret⟩, the retrieval LLM writes a passage, the :8001 ARC
encoder embeds it, and the result conditions the answer that follows. Training mirrors that:
each precomputed reference is placed AT its own ⟨ret⟩ frame and applied over the following
T_ref frames.

A clip can hold several retrieval turns, so the i-th ⟨ret⟩ frame is paired with the i-th
reference and each span is clamped at the next ⟨ret⟩ frame — otherwise a long passage would
bleed over the following turn, and (in the earlier first-hit-only version) every retrieval
turn after the first would be trained with no conditioning at all, which teaches the model to
answer factual questions from its own head.

Deliberately free of any `moshi` import so it can be unit-tested off the A100.
"""
import torch

RAG_TOKEN_ID = 4


def build_reference_condition(text_row, refs, dim, dtype, rag_token_id=RAG_TOKEN_ID):
    """(ref_cond [B, S, dim], ref_mask [B, S]) from per-example lists of [T_ref, dim] tensors.

    `refs[b]` is a list of tensors, or None. `text_row` is codes[:, 0, :].
    An example whose references outnumber its markers (or vice versa) uses only the
    positions that pair up. An example with references but no marker falls back to the
    sequence prefix, matching the pre-Stage-3 behaviour for legacy clips.
    """
    device = text_row.device
    bsz, seq = text_row.shape
    ref_cond = torch.zeros(bsz, seq, dim, device=device, dtype=dtype)
    ref_mask = torch.zeros(bsz, seq, device=device)

    for bi in range(bsz):
        tensors = refs[bi] if refs is not None else None
        if not tensors:
            continue
        if torch.is_tensor(tensors):        # tolerate a bare tensor
            tensors = [tensors]

        hits = (text_row[bi] == rag_token_id).nonzero(as_tuple=False).flatten().tolist()
        if not hits:
            hits = [0]

        n = min(len(hits), len(tensors))
        for i in range(n):
            start = hits[i]
            limit = hits[i + 1] if i + 1 < len(hits) else seq
            span = min(tensors[i].shape[0], limit - start, seq - start)
            if span <= 0:
                continue
            ref_cond[bi, start:start + span] = tensors[i][:span].to(device=device, dtype=dtype)
            ref_mask[bi, start:start + span] = 1
    return ref_cond, ref_mask
```

Replace `train.py:254-279` with:

```python
            if getattr(batch, "reference_tensors", None) is not None:
                # Inject precomputed reference_with_time (replay) at EVERY rag_token (⟨ret⟩)
                # frame, so multi-retrieval clips condition each answer (matches serve).
                from moshi.conditioners.base import ConditionType

                from finetune.data.reference_injection import build_reference_condition

                refs = batch.reference_tensors
                present = [r for r in refs if r]
                first = present[0]
                dim = (first[0] if isinstance(first, list) else first).shape[-1]
                ref_cond, ref_mask = build_reference_condition(
                    codes[:, 0, :], refs, dim, next(model.parameters()).dtype
                )
                if condition_tensors is None:
                    condition_tensors = {}
                condition_tensors["reference_with_time"] = ConditionType(ref_cond, ref_mask)
```

In `scripts/apply_rag_positioning.py`, update `patch_train()` so a fresh checkout produces the block above: change its already-applied guard from `if "RAG_TOKEN_ID" in t:` to `if "build_reference_condition" in t:`, and set `new_block` to exactly the code above. Its docstring bullet 2 should now read "position the precomputed references at every rag_token frame in the text row, clamped at the next one".

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_reference_injection.py -v`
Expected: 8 passed.

Verify the patcher is a no-op against the edited `train.py`:

Run: `python3 scripts/apply_rag_positioning.py`
Expected: `interleaver.py already has RAG marker.` and `train.py already positions reference at rag frame.` — and `git diff --stat` shows no change.

- [ ] **Step 5: Commit**

```bash
git add finetune/data/reference_injection.py train.py scripts/apply_rag_positioning.py tests/test_reference_injection.py
git commit -m "train: condition every rag_token frame, not only the first"
```

---

### Task 10: Stage 3 training config

**Files:**
- Create: `example/moshika_rag_stage3.yaml`
- Test: `tests/test_stage3_config.py`

**Interfaces:**
- Consumes: `replay/retrieval/train.jsonl` (Task 11 builds it) and `finetune/data/prepared_dialogue/train.jsonl`.
- Produces: the config `torchrun -m train` is given.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage3_config.py`:

```python
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parents[1] / "example/moshika_rag_stage3.yaml"


def _cfg():
    return yaml.safe_load(CONFIG.read_text())


def test_uses_native_sampling_weights_not_a_premixed_manifest():
    sources = _cfg()["data"]["train_data"].split(",")
    assert len(sources) == 2
    weights = [float(s.split(":")[-1]) for s in sources]
    assert weights == [0.5, 0.5]


def test_trains_on_the_rag_base_with_the_spike_config():
    cfg = _cfg()
    assert cfg["moshi_paths"]["hf_repo_id"] == "kyutai/moshika-rag-pytorch-bf16"
    assert cfg["moshi_paths"]["config_path"].endswith("config.spike.json")


def test_duration_and_lora_match_the_stage2b_baseline():
    cfg = _cfg()
    assert cfg["duration_sec"] == 100
    assert cfg["lora"]["rank"] == 64
    assert cfg["lora"]["enable"] is True
    assert cfg["full_finetuning"] is False


def test_is_a_fresh_run_not_a_continuation():
    cfg = _cfg()
    # retrieval must be in the loss from step 0
    assert "initial_model" not in cfg or not cfg["initial_model"]
    assert cfg["run_dir"].endswith("moshika_rag_stage3")


def test_saves_adapters_and_checkpoints_every_hundred_steps():
    cfg = _cfg()
    assert cfg["save_adapters"] is True
    assert cfg["ckpt_freq"] == 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage3_config.py -v`
Expected: FAIL with `FileNotFoundError` for `example/moshika_rag_stage3.yaml`.

- [ ] **Step 3: Write the config**

Create `example/moshika_rag_stage3.yaml`:

```yaml
# Stage 3 — real recorded retrieval dialogue + natural dialogue, one LoRA.
# Goal: casual conversation AND retrieval in one session, switching between them.
# Spec: docs/superpowers/specs/2026-08-05-stage3-retrieval-audio-prep-design.md
#
# Sampling weights (dataset.py parse_data_sources) rather than a premixed manifest: exact,
# independent of each side's duration, and retunable by editing this one line. Underlying
# durations are ~85 min retrieval vs 135.3 min natural.
#
# RAG_TOKEN_WEIGHT is passed in the environment, NOT here. Set it from the turn mix that
# segment_retrieval_audio.py reports: 15 if smalltalk+decline >= 30% of her turns, else 8-10.
data:
  train_data: "/home/leenatantawy/moshi-finetune/replay/retrieval/train.jsonl:0.5,/home/leenatantawy/moshi-finetune/finetune/data/prepared_dialogue/train.jsonl:0.5"
  eval_data: "/home/leenatantawy/moshi-finetune/finetune/data/prepared_dialogue/eval.jsonl"
  shuffle: true
moshi_paths:
  hf_repo_id: "kyutai/moshika-rag-pytorch-bf16"
  # reference_with_time in fuser.streaming_sum (for the injected reference), not a module;
  # first_speaker prepend off. Same config Stage 2b trained under.
  config_path: "/home/leenatantawy/moshi-finetune/checkpoints/moshika-rag/config.spike.json"
full_finetuning: false
lora: { enable: true, rank: 64, scaling: 2., ft_embed: false }
first_codebook_weight_multiplier: 100.
text_padding_weight: .5
duration_sec: 100
batch_size: 8
max_steps: 800
gradient_checkpointing: true
optim: { lr: 4e-6, weight_decay: 0.1, pct_start: 0.05 }
do_eval: false
do_ckpt: true
ckpt_freq: 100
num_ckpt_keep: 10
save_adapters: true
seed: 0
log_freq: 1
run_dir: "/home/leenatantawy/moshi-finetune/runs/moshika_rag_stage3"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_stage3_config.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add example/moshika_rag_stage3.yaml tests/test_stage3_config.py
git commit -m "example: Stage 3 training config with 0.5/0.5 sampling weights"
```

---

### Task 11: A100 runbook and the pre-training verification gate

Everything above is testable on the Mac. This task is the ordered sequence for the box plus the gate that must pass before any training starts.

**Files:**
- Create: `docs/stage3_a100_runbook.md`
- Create: `scripts/verify_retrieval_clips.py`
- Test: `tests/test_verify_retrieval_clips.py`

**Interfaces:**
- Consumes: `replay/retrieval/` clips (Task 7) and their `.ref.safetensors` (Task 8).
- Produces: `check_clips(clip_dir) -> tuple[list[str], dict]` returning `(problems, stats)`; exit code 1 if any problem.

- [ ] **Step 1: Write the failing test**

Create `tests/test_verify_retrieval_clips.py`:

```python
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.data.reference_io import save_references
from verify_retrieval_clips import check_clips


def _clip(d, name, words, n_refs, seconds=10.0, sr=24000):
    n = int(seconds * sr)
    sf.write(str(d / f"{name}.wav"), np.zeros((n, 2), dtype="float32"), sr, subtype="PCM_24")
    alignments = [[w, [float(i), float(i) + 0.5], "SPEAKER_MAIN"] for i, w in enumerate(words)]
    json.dump({"alignments": alignments}, open(d / f"{name}.json", "w"))
    if n_refs:
        save_references(str(d / f"{name}.ref.safetensors"),
                        [torch.ones(2, 3) for _ in range(n_refs)])


def test_clean_directory_reports_no_problems(tmp_path):
    _clip(tmp_path, "a", ["hi", "<RAG>", "fact"], 1)
    _clip(tmp_path, "b", ["just", "chatting"], 0)
    problems, stats = check_clips(tmp_path)
    assert problems == []
    assert stats["clips"] == 2
    assert stats["rag_markers"] == 1
    assert stats["reference_tensors"] == 1


def test_marker_without_a_reference_tensor_is_a_problem(tmp_path):
    _clip(tmp_path, "a", ["hi", "<RAG>", "fact"], 0)
    problems, _ = check_clips(tmp_path)
    assert any("1 <RAG>" in p and "0 reference" in p for p in problems)


def test_reference_tensor_without_a_marker_is_a_problem(tmp_path):
    _clip(tmp_path, "a", ["hi", "fact"], 1)
    problems, _ = check_clips(tmp_path)
    assert any("0 <RAG>" in p and "1 reference" in p for p in problems)


def test_missing_transcript_is_a_problem(tmp_path):
    _clip(tmp_path, "a", ["hi"], 0)
    (tmp_path / "a.json").unlink()
    problems, _ = check_clips(tmp_path)
    assert any("no transcript" in p for p in problems)


def test_empty_transcript_is_a_problem(tmp_path):
    _clip(tmp_path, "a", [], 0)
    problems, _ = check_clips(tmp_path)
    assert any("empty alignments" in p for p in problems)


def test_clip_over_duration_sec_is_a_problem(tmp_path):
    _clip(tmp_path, "a", ["hi"], 0, seconds=120.0)
    problems, _ = check_clips(tmp_path)
    assert any("longer than 100" in p for p in problems)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_verify_retrieval_clips.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'verify_retrieval_clips'`.

- [ ] **Step 3: Write the verifier**

Create `scripts/verify_retrieval_clips.py`:

```python
#!/usr/bin/env python3
"""Stage 3 — gate the retrieval clips before spending GPU time on training.

Checks, per clip: a non-empty sibling .json exists, the clip is not longer than duration_sec,
and the number of <RAG> markers equals the number of reference tensors. A mismatch means
train.py would silently leave a retrieval turn unconditioned while still training its answer.

    python3 scripts/verify_retrieval_clips.py --clips replay/retrieval
"""
import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finetune.data.reference_io import load_references  # noqa: E402

MAX_SEC = 100.0


def check_clips(clip_dir):
    clip_dir = Path(clip_dir)
    problems = []
    stats = {"clips": 0, "rag_markers": 0, "reference_tensors": 0, "minutes": 0.0,
             "multi_retrieval_clips": 0}

    for wav in sorted(clip_dir.glob("*.wav")):
        stats["clips"] += 1
        info = sf.info(str(wav))
        stats["minutes"] += info.duration / 60.0
        if info.duration > MAX_SEC + 1e-6:
            problems.append(f"{wav.name}: longer than {MAX_SEC:.0f}s ({info.duration:.1f}s)")
        if info.channels != 2 or info.samplerate != 24000:
            problems.append(f"{wav.name}: expected 2ch/24000 Hz, got "
                            f"{info.channels}ch/{info.samplerate} Hz")

        transcript = wav.with_suffix(".json")
        if not transcript.exists():
            problems.append(f"{wav.name}: no transcript ({transcript.name})")
            continue
        alignments = json.load(open(transcript)).get("alignments", [])
        if not alignments:
            problems.append(f"{wav.name}: empty alignments")
            continue

        n_markers = sum(1 for w in alignments if w[0] == "<RAG>")
        refs = load_references(str(wav)) or []
        stats["rag_markers"] += n_markers
        stats["reference_tensors"] += len(refs)
        if n_markers > 1:
            stats["multi_retrieval_clips"] += 1
        if n_markers != len(refs):
            problems.append(f"{wav.name}: {n_markers} <RAG> markers but "
                            f"{len(refs)} reference tensors")

    return problems, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="replay/retrieval")
    args = ap.parse_args()

    problems, stats = check_clips(args.clips)
    for p in problems:
        print(f"  FAIL {p}")
    print(f"\n{stats['clips']} clips, {stats['minutes']:.1f} min, "
          f"{stats['rag_markers']} <RAG> markers, "
          f"{stats['reference_tensors']} reference tensors, "
          f"{stats['multi_retrieval_clips']} multi-retrieval clips")
    if problems:
        print(f"{len(problems)} problem(s) — do not train yet.")
        raise SystemExit(1)
    print("gate passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_verify_retrieval_clips.py -v`
Expected: 6 passed.

Then run the whole suite:

Run: `python3 -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Write the runbook**

Create `docs/stage3_a100_runbook.md`:

````markdown
# A100 runbook — Stage 3 retrieval data + train

Prereq (Mac, Task 1): `finetune/data/prepared_retrieval/danielle_joshuarhodes.wav` exists —
84.65 min, 2ch, 24 kHz, PCM_24, Danielle left / Joshua right.

## 1. Upload the master (Mac)
```bash
rsync -avP finetune/data/prepared_retrieval/danielle_joshuarhodes.wav \
  wb-gpu-training:~/moshi-finetune/finetune/data/prepared_retrieval/
```

## 2. Transcribe both channels (A100)
`annotate.py` skips a file whose output already exists, so the two passes are independent.
Build a one-line manifest first — it takes a jsonl, not a wav.
```bash
cd ~/moshi-finetune
uv run python scripts/build_manifest.py \
  --wav-dir finetune/data/prepared_retrieval \
  --out-dir finetune/data/prepared_retrieval \
  --eval-file ""            # no eval file here; all.jsonl is what we use

CUDA_VISIBLE_DEVICES=0 uv run python annotate.py \
  finetune/data/prepared_retrieval/all.jsonl -l --whisper_model medium --lang en \
  --channel 0 --out-suffix .json

CUDA_VISIBLE_DEVICES=0 uv run python annotate.py \
  finetune/data/prepared_retrieval/all.jsonl -l --whisper_model medium --lang en \
  --channel 1 --out-suffix .ch1.json
```
85 min is well inside annotate.py's 4-hour guard. Expect both
`danielle_joshuarhodes.json` and `danielle_joshuarhodes.ch1.json`.

## 3. Pull the transcripts back (Mac)
```bash
rsync -avP 'wb-gpu-training:~/moshi-finetune/finetune/data/prepared_retrieval/*.json' \
  finetune/data/prepared_retrieval/
```

## 4. Segment, filter and cut (Mac)
```bash
source .env                     # GEMINI_API_KEY, rotated
python3 scripts/segment_retrieval_audio.py \
  --ch0 finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
  --ch1 finetune/data/prepared_retrieval/danielle_joshuarhodes.ch1.json \
  --out replay/retrieval_segments.jsonl

python3 scripts/filter_retrieval_segments.py \
  --in replay/retrieval_segments.jsonl \
  --out replay/retrieval_segments.filtered.jsonl \
  --dropped replay/retrieval_dropped.jsonl

python3 scripts/cut_retrieval_clips.py \
  --master finetune/data/prepared_retrieval/danielle_joshuarhodes.wav \
  --alignments finetune/data/prepared_retrieval/danielle_joshuarhodes.json \
  --segments replay/retrieval_segments.filtered.jsonl \
  --outdir replay/retrieval --manifest replay/retrieval_manifest.jsonl
```
**Write down the `her turns:` line** from the segmenter. Its `retrieval_share` sets
`RAG_TOKEN_WEIGHT` in step 7.

Audition two or three clips here, before uploading, and confirm each starts at a question
rather than mid-sentence:
```bash
open replay/retrieval/ret-00-000.wav replay/retrieval/ret-00-001.wav
```

## 5. Upload the clips (Mac)
```bash
rsync -avP replay/retrieval/ wb-gpu-training:~/moshi-finetune/replay/retrieval/
```

## 6. Encode references, build the manifest, run the gate (A100)
The `:8001` reference-encoder service must be up (see the Stage 2 notes in
`NEXT_STEPS.md` Part 3). `build_manifest.py` must run here because manifest paths are
absolute and machine-specific.
```bash
cd ~/moshi-finetune
uv run python scripts/precompute_references.py \
  --manifest replay/retrieval_manifest.jsonl --audiodir replay/retrieval

uv run python scripts/build_manifest.py \
  --wav-dir replay/retrieval --out-dir replay/retrieval --eval-file ""

uv run python scripts/verify_retrieval_clips.py --clips replay/retrieval
```
The gate must print `gate passed.` before step 7. If it reports marker/tensor mismatches,
send the output back rather than training around them — a mismatch means a retrieval turn
would be trained with no conditioning.

## 7. Train (A100)
Set `RAG_TOKEN_WEIGHT` from step 4: 15 if smalltalk+decline are at least 30% of her turns,
otherwise 8-10.

Before the full run, prove the multi-`⟨ret⟩` injection actually works on the real stack —
its unit tests cover the tensor maths but nothing on the Mac can load moshi:
```bash
cp example/moshika_rag_stage3.yaml /tmp/stage3_smoke.yaml
sed -i 's/^max_steps: .*/max_steps: 2/; s/^do_ckpt: .*/do_ckpt: false/' /tmp/stage3_smoke.yaml
RAG_TOKEN_WEIGHT=15 CUDA_VISIBLE_DEVICES=0 \
  uv run torchrun --nproc-per-node 1 -m train /tmp/stage3_smoke.yaml
```
Expected: two steps complete with a finite loss. A crash inside
`build_reference_condition` or a `reference_with_time` shape error means Task 9 needs
fixing before the real run.
```bash
cd ~/moshi-finetune
RAG_TOKEN_WEIGHT=15 CUDA_VISIBLE_DEVICES=0 \
  uv run torchrun --nproc-per-node 1 -m train example/moshika_rag_stage3.yaml
```
tmux does not survive an SSH drop on this VM — use `setsid`/`nohup`.

Checkpoints land in `runs/moshika_rag_stage3/checkpoints/checkpoint_*/consolidated/lora.safetensors`.

## 8. Audition against the success criterion
Not "does retrieval work" but "can it switch", per the spec:
1. Several turns of chit-chat, *then* a factual question — the case trial 4 failed.
2. After a successful retrieval, return to casual talk; she must not stay formal.
3. A wholly casual conversation must emit no `⟨ret⟩`.
4. She must sound like herself in both modes.
````

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_retrieval_clips.py tests/test_verify_retrieval_clips.py docs/stage3_a100_runbook.md
git commit -m "Stage 3: clip verification gate + A100 runbook"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `pair_dialogue_stereo.py --no-eval` | 1 |
| `annotate.py --channel` / `--out-suffix` | 2 |
| Merge to utterances, 0.6 s gap, `[MM:SS.s] SPEAKER: text` | 3 |
| Snap to real utterance edges; monotonic, non-overlapping, starts JOSHUA / ends DANIELLE, 4-100 s, split long | 4 |
| Gemini 12-min windows with 1-min overlap, dedupe, per-turn `grounded`/`decline`/`smalltalk`, reference prose rules | 5 |
| Faithfulness pass, `retrieval_dropped.jsonl` | 6 |
| Cut clips, offset-slice alignments, `<RAG>` before *each* retrieval turn, i-th marker ↔ i-th reference, manifest | 7 |
| N-tensor `.ref.safetensors`; `precompute_references` writes N; `_load_reference_tensor` returns a list | 8 |
| `train.py` loops over all hits, clamps at the next hit, falls back on count mismatch; `apply_rag_positioning.py` updated | 9 |
| Fresh LoRA, 0.5/0.5 sampling weights, `RAG_TOKEN_WEIGHT` deferred, eval file, ckpt every 100 | 10 |
| Gate: transcript present, marker/tensor counts agree, turn mix reported, audition clips; switching-focused eval | 11 |

Two spec items are intentionally *not* code: "prefer segments containing a mode transition" lives in the Task 5 prompt, and the `RAG_TOKEN_WEIGHT` decision is a human judgement made from the Task 5 report and applied in the Task 11 runbook.

`mix_manifests.py` is untouched, as the spec requires — Task 10 uses native sampling weights instead.

**Placeholder scan:** No TBD/TODO. Every code step carries real code; every test step carries real assertions; every run step states the exact command and expected output.

**Type consistency checked:**
- Utterance dict `{speaker,start,end,text}` — produced Task 3, consumed Tasks 4, 5.
- Segment dict `{id,start,end,turns:[{speaker,start,end,kind,reference}]}` — Tasks 4, 5, 6, 7 agree.
- `call_gemini(prompt, model, key, schema=SCHEMA, retries=4)` — defined Task 5, reused Task 6; Task 6 explicitly notes the signature change and that Task 5's call site is unaffected by the default.
- `save_references` / `load_references` — defined Task 8, consumed Tasks 8 (precompute), 9 (via interleaver), 11 (verifier and its test).
- `build_reference_condition(text_row, refs, dim, dtype, rag_token_id=4)` — defined Task 9, called from `train.py` in the same task.
- Manifest row `{id, path, duration_sec, references, kinds}` — written Task 7, read Task 8's `precompute_references` change.
- Clip artifacts `<id>.wav` / `<id>.json` / `<id>.ref.safetensors` — Tasks 7, 8, 11 agree.

One dependency worth flagging to the executor: **Task 9's `train.py` edit cannot be smoke-tested on the Mac** (no moshi, no CUDA). Its unit tests cover the tensor logic, but the first real proof is a 2-step run on the A100 before the full training run. `apply_rag_positioning.py` rewrites `train.py` by string match, so if that patcher is not updated in the same commit, a fresh checkout on the box silently reverts to the first-hit-only behaviour.
