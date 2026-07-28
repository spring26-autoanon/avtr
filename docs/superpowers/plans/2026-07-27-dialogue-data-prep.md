# Dialogue Data Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the three two-speaker recordings into stereo training WAVs (Danielle=left/main, partner=right), with a 10-min eval slice carved from the middle of the Joshua conversation, ready for the fork Path-B retrain.

**Architecture:** One new script `scripts/pair_dialogue_stereo.py` built from three pure/near-pure functions — `parse_track_name` (filename → speaker/idx/conv_id), `combine_to_stereo` (two mono tracks → time-aligned stereo), `carve_eval` (split a stereo array into before/eval/after) — wired by a `main()` CLI. Everything downstream (`build_manifest.py`, `annotate.py`, training) is reused unchanged and runs on the A100.

**Tech Stack:** Python 3, `soundfile` (0.13.1), `numpy` (2.3.3), `pytest` (added). No CUDA locally.

## Global Constraints

- Output audio: **stereo, 24000 Hz, `subtype="PCM_24"`**. Left channel (col 0) = main/Danielle, right (col 1) = partner. Copied verbatim from source — **no resampling, no normalization**.
- Channel assignment is **by speaker identity**, never by the 1/2 participant index (the index is inconsistent across conversations).
- Main speaker matched by **case-insensitive substring** against `--main-name` (default `danielle`); parsed speaker names are lowercased.
- `conv_id` = **the last 10 digits** of the filename before `_24khz`.
- Source files live in `finetune/data/datastereo/clean_moshi_audio_24khz/` and must **never be modified**. Outputs go to a **new** dir `finetune/data/prepared_dialogue/`.
- Eval defaults: `--eval-conv 1411304343`, `--eval-sec 600`, `--eval-center-frac 0.5`.
- Transcript `.json` files and manifests are **A100-only** (Task 5 runbook) — do not attempt to generate them locally.

**Source filenames (ground truth):**
```
audioClayS11556527425_24khz.wav            # Clay,   conv 1556527425
audioDanielleDeLosa21556527425_24khz.wav   # Danielle, conv 1556527425
audioDanielleDeLosa11411304343_24khz.wav   # Danielle, conv 1411304343
audioJoshuaRhodes21411304343_24khz.wav     # Joshua,  conv 1411304343
audioDanielleDeLosa21341305451_24khz.wav   # Danielle, conv 1341305451
audioLeenaTantawy11341305451_24khz.wav     # Leena,   conv 1341305451
```

---

### Task 1: Filename parser

**Files:**
- Create: `scripts/pair_dialogue_stereo.py`
- Test: `tests/test_pair_dialogue_stereo.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `parse_track_name(name: str) -> tuple[str, str, str]` returning `(speaker, idx, conv_id)` where `speaker` is lowercased with trailing digits stripped, `idx` is the participant-index digits (possibly `""`), `conv_id` is the 10-digit trailing group id. Raises `ValueError` on malformed names.

- [ ] **Step 1: Install pytest**

Run: `python3 -m pip install pytest -q`
Expected: installs (or "already satisfied").

- [ ] **Step 2: Write the failing test**

Create `tests/test_pair_dialogue_stereo.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pytest
from pair_dialogue_stereo import parse_track_name


def test_parse_clay():
    assert parse_track_name("audioClayS11556527425_24khz.wav") == ("clays", "1", "1556527425")


def test_parse_danielle_idx2():
    assert parse_track_name("audioDanielleDeLosa21341305451_24khz.wav") == (
        "danielledelosa", "2", "1341305451",
    )


def test_parse_joshua():
    assert parse_track_name("audioJoshuaRhodes21411304343_24khz.wav") == (
        "joshuarhodes", "2", "1411304343",
    )


def test_parse_rejects_short_id():
    with pytest.raises(ValueError):
        parse_track_name("audioBob123_24khz.wav")


def test_parse_rejects_no_audio_prefix():
    with pytest.raises(ValueError):
        parse_track_name("ClayS11556527425_24khz.wav")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pair_dialogue_stereo'` (file doesn't exist yet).

- [ ] **Step 4: Write minimal implementation**

Create `scripts/pair_dialogue_stereo.py`:
```python
#!/usr/bin/env python3
"""Pair two-speaker mono recordings into stereo training WAVs for moshi-finetune.

Each conversation is two isolated mono tracks (one per speaker) that share the
last 10 digits of their filename. We place the MAIN speaker (the cloned voice) on
the LEFT channel (0) and the partner on the RIGHT channel (1), time-aligned, at
24 kHz / PCM_24. Optionally carve an eval slice out of the middle of one
conversation. Originals are never modified.

Usage:
    python scripts/pair_dialogue_stereo.py \
        --src finetune/data/datastereo/clean_moshi_audio_24khz \
        --dst finetune/data/prepared_dialogue \
        --main-name danielle \
        --eval-conv 1411304343 --eval-sec 600 --eval-center-frac 0.5
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 24000


def parse_track_name(name: str) -> tuple[str, str, str]:
    """'audioClayS11556527425_24khz.wav' -> ('clays', '1', '1556527425')."""
    stem = name
    if stem.endswith(".wav"):
        stem = stem[:-4]
    if stem.endswith("_24khz"):
        stem = stem[: -len("_24khz")]
    if not stem.startswith("audio"):
        raise ValueError(f"unexpected name (no 'audio' prefix): {name}")
    stem = stem[len("audio"):]
    if len(stem) < 11 or not stem[-10:].isdigit():
        raise ValueError(f"no 10-digit conversation id in: {name}")
    conv_id = stem[-10:]
    rest = stem[:-10]
    i = len(rest)
    while i > 0 and rest[i - 1].isdigit():
        i -= 1
    idx = rest[i:]
    speaker = rest[:i].lower()
    if not speaker:
        raise ValueError(f"empty speaker name in: {name}")
    return speaker, idx, conv_id
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add scripts/pair_dialogue_stereo.py tests/test_pair_dialogue_stereo.py
git commit -m "feat: filename parser for dialogue track pairing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Combine two mono tracks into time-aligned stereo

**Files:**
- Modify: `scripts/pair_dialogue_stereo.py` (add `combine_to_stereo`)
- Test: `tests/test_pair_dialogue_stereo.py` (add cases + a synthetic-wav helper)

**Interfaces:**
- Consumes: `TARGET_SR`.
- Produces: `combine_to_stereo(main_path: Path, partner_path: Path) -> tuple[np.ndarray, int, int]` returning `(stereo, sr, delta_frames)` where `stereo` has shape `(n, 2)` (col 0 = main, col 1 = partner), `sr == 24000`, and `delta_frames` is the absolute length difference before truncation to the shorter track. Raises `SystemExit` if either input isn't 24 kHz.

- [ ] **Step 1: Add the synthetic-wav helper and failing tests**

Append to `tests/test_pair_dialogue_stereo.py`:
```python
import numpy as np
import soundfile as sf
from pair_dialogue_stereo import combine_to_stereo


def _mono_wav(path, seconds, value, sr=24000):
    n = int(seconds * sr)
    sf.write(str(path), np.full(n, value, dtype="float32"), sr, subtype="PCM_24")


def test_combine_left_is_main_right_is_partner(tmp_path):
    main = tmp_path / "main.wav"
    partner = tmp_path / "partner.wav"
    _mono_wav(main, 1.0, 0.5)
    _mono_wav(partner, 1.0, -0.25)
    stereo, sr, delta = combine_to_stereo(main, partner)
    assert sr == 24000
    assert stereo.shape == (24000, 2)
    assert delta == 0
    assert np.allclose(stereo[:, 0], 0.5, atol=1e-4)
    assert np.allclose(stereo[:, 1], -0.25, atol=1e-4)


def test_combine_truncates_to_shorter(tmp_path):
    main = tmp_path / "main.wav"
    partner = tmp_path / "partner.wav"
    _mono_wav(main, 2.0, 0.5)     # 48000 frames
    _mono_wav(partner, 1.5, -0.25)  # 36000 frames
    stereo, sr, delta = combine_to_stereo(main, partner)
    assert stereo.shape == (36000, 2)
    assert delta == 12000


def test_combine_rejects_wrong_sr(tmp_path):
    main = tmp_path / "main.wav"
    partner = tmp_path / "partner.wav"
    _mono_wav(main, 1.0, 0.5, sr=16000)
    _mono_wav(partner, 1.0, -0.25)
    with pytest.raises(SystemExit):
        combine_to_stereo(main, partner)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -k combine -v`
Expected: FAIL — `ImportError: cannot import name 'combine_to_stereo'`.

- [ ] **Step 3: Add the implementation**

In `scripts/pair_dialogue_stereo.py`, add after `parse_track_name`:
```python
def combine_to_stereo(main_path: Path, partner_path: Path) -> tuple[np.ndarray, int, int]:
    """Read two mono tracks, return (stereo[n,2], sr, delta_frames).

    col 0 = left = main (cloned voice); col 1 = right = partner. Truncated to the
    shorter of the two tracks; delta_frames is the length difference removed.
    """
    main, sr_m = sf.read(str(main_path), always_2d=True)
    partner, sr_p = sf.read(str(partner_path), always_2d=True)
    for path, sr in ((main_path, sr_m), (partner_path, sr_p)):
        if sr != TARGET_SR:
            raise SystemExit(f"{path}: sample rate {sr} != {TARGET_SR}; resample first.")
    m = main[:, 0]
    p = partner[:, 0]
    delta = abs(len(m) - len(p))
    n = min(len(m), len(p))
    stereo = np.column_stack([m[:n], p[:n]])
    return stereo, TARGET_SR, delta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -k combine -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/pair_dialogue_stereo.py tests/test_pair_dialogue_stereo.py
git commit -m "feat: combine two mono tracks into time-aligned stereo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Carve an eval slice from the middle

**Files:**
- Modify: `scripts/pair_dialogue_stereo.py` (add `carve_eval`)
- Test: `tests/test_pair_dialogue_stereo.py` (add cases)

**Interfaces:**
- Consumes: nothing new.
- Produces: `carve_eval(stereo: np.ndarray, sr: int, eval_sec: float, center_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(before, eval_slice, after)`. `eval_slice` is `round(eval_sec*sr)` frames long, centered at `center_frac` of the total length and clamped to fit. Concatenating the three parts reproduces the input exactly. Raises `SystemExit` if the eval slice would not fit.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pair_dialogue_stereo.py`:
```python
from pair_dialogue_stereo import carve_eval


def test_carve_splits_contiguously():
    stereo = np.arange(10 * 24000 * 2, dtype="float32").reshape(10 * 24000, 2)
    before, ev, after = carve_eval(stereo, 24000, eval_sec=2.0, center_frac=0.5)
    assert ev.shape[0] == 2 * 24000
    assert before.shape[0] == 4 * 24000   # centered: start at 5s-1s = 4s
    assert after.shape[0] == 4 * 24000
    assert np.array_equal(np.concatenate([before, ev, after]), stereo)


def test_carve_rejects_too_long():
    stereo = np.zeros((1 * 24000, 2), dtype="float32")
    with pytest.raises(SystemExit):
        carve_eval(stereo, 24000, eval_sec=2.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -k carve -v`
Expected: FAIL — `ImportError: cannot import name 'carve_eval'`.

- [ ] **Step 3: Add the implementation**

In `scripts/pair_dialogue_stereo.py`, add after `combine_to_stereo`:
```python
def carve_eval(
    stereo: np.ndarray, sr: int, eval_sec: float, center_frac: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut an eval slice from the middle. Returns (before, eval, after)."""
    total = stereo.shape[0]
    eval_frames = int(round(eval_sec * sr))
    if eval_frames >= total:
        raise SystemExit(
            f"eval slice ({eval_sec}s) >= conversation length ({total / sr:.1f}s)."
        )
    center = int(round(total * center_frac))
    start = center - eval_frames // 2
    start = max(0, min(start, total - eval_frames))
    end = start + eval_frames
    return stereo[:start], stereo[start:end], stereo[end:]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -k carve -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/pair_dialogue_stereo.py tests/test_pair_dialogue_stereo.py
git commit -m "feat: carve eval slice from middle of a conversation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: CLI wiring, real-data run, and integration verification

**Files:**
- Modify: `scripts/pair_dialogue_stereo.py` (add `discover_conversations`, `main`, `__main__` guard)
- Test: `tests/test_pair_dialogue_stereo.py` (add a full synthetic end-to-end test + a real-data verification test)

**Interfaces:**
- Consumes: `parse_track_name`, `combine_to_stereo`, `carve_eval`.
- Produces:
  - `discover_conversations(src: Path, main_name: str) -> dict[str, tuple[Path, str, Path, str]]` mapping `conv_id -> (main_path, main_speaker, partner_path, partner_speaker)`. Raises `SystemExit` if any conversation lacks exactly one main + one partner track.
  - `main(argv: list[str] | None = None) -> None` — the CLI. Writes 24 kHz / PCM_24 stereo WAVs to `--dst`. For a non-eval conversation writes `{main_name}_{partner_speaker}.wav`; for the eval conversation writes `{main_name}_{partner_speaker}_train_a.wav`, `_eval.wav`, `_train_b.wav`. Empty leading/trailing segments (if `center_frac` is 0 or 1) are skipped.

- [ ] **Step 1: Write the failing end-to-end synthetic test**

Append to `tests/test_pair_dialogue_stereo.py`:
```python
from pair_dialogue_stereo import discover_conversations, main


def _fake_source(dirpath):
    # conv A (no eval): main=danielle val 0.5, partner=clay val -0.25
    _mono_wav(dirpath / "audioDanielleDeLosa20000000001_24khz.wav", 1.0, 0.5)
    _mono_wav(dirpath / "audioClayS10000000001_24khz.wav", 1.0, -0.25)
    # conv B (eval target): main=danielle val 0.5, partner=joshua val -0.5
    _mono_wav(dirpath / "audioDanielleDeLosa10000000002_24khz.wav", 10.0, 0.5)
    _mono_wav(dirpath / "audioJoshuaRhodes20000000002_24khz.wav", 10.0, -0.5)


def test_discover_pairs_by_conv_and_identity(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _fake_source(src)
    convs = discover_conversations(src, "danielle")
    assert set(convs) == {"0000000001", "0000000002"}
    main_path, main_spk, partner_path, partner_spk = convs["0000000001"]
    assert main_spk == "danielledelosa" and partner_spk == "clays"


def test_main_writes_expected_files_and_channels(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    _fake_source(src)
    main([
        "--src", str(src), "--dst", str(dst),
        "--main-name", "danielle",
        "--eval-conv", "0000000002", "--eval-sec", "2.0", "--eval-center-frac", "0.5",
    ])
    names = sorted(p.name for p in dst.glob("*.wav"))
    assert names == [
        "danielle_clays.wav",
        "danielle_joshuarhodes_eval.wav",
        "danielle_joshuarhodes_train_a.wav",
        "danielle_joshuarhodes_train_b.wav",
    ]
    # channel assignment + format on the non-eval conv
    data, sr = sf.read(str(dst / "danielle_clays.wav"), always_2d=True)
    assert sr == 24000 and data.shape[1] == 2
    assert np.allclose(data[:, 0], 0.5, atol=1e-4)   # left = main
    assert np.allclose(data[:, 1], -0.25, atol=1e-4)  # right = partner
    assert sf.info(str(dst / "danielle_clays.wav")).subtype == "PCM_24"
    # eval slice length
    ev = sf.info(str(dst / "danielle_joshuarhodes_eval.wav"))
    assert ev.frames == 2 * 24000
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -k "discover or main_writes" -v`
Expected: FAIL — `ImportError: cannot import name 'discover_conversations'`.

- [ ] **Step 3: Add discovery + CLI implementation**

In `scripts/pair_dialogue_stereo.py`, add after `carve_eval`:
```python
def discover_conversations(
    src: Path, main_name: str
) -> dict[str, tuple[Path, str, Path, str]]:
    """Group *.wav by conv_id and split each into (main, partner) by identity."""
    groups: dict[str, list[tuple[Path, str]]] = {}
    for wav in sorted(src.glob("*.wav")):
        speaker, _idx, conv_id = parse_track_name(wav.name)
        groups.setdefault(conv_id, []).append((wav, speaker))

    convs: dict[str, tuple[Path, str, Path, str]] = {}
    key = main_name.lower()
    for conv_id, tracks in groups.items():
        if len(tracks) != 2:
            raise SystemExit(
                f"conversation {conv_id} has {len(tracks)} tracks, expected 2: "
                + ", ".join(p.name for p, _ in tracks)
            )
        mains = [(p, s) for p, s in tracks if key in s]
        partners = [(p, s) for p, s in tracks if key not in s]
        if len(mains) != 1 or len(partners) != 1:
            raise SystemExit(
                f"conversation {conv_id}: need exactly one '{main_name}' track and "
                f"one partner; got mains={[s for _, s in mains]}, "
                f"partners={[s for _, s in partners]}"
            )
        (main_path, main_spk), (partner_path, partner_spk) = mains[0], partners[0]
        convs[conv_id] = (main_path, main_spk, partner_path, partner_spk)
    return convs


def _write(dst: Path, name: str, stereo: np.ndarray) -> None:
    sf.write(str(dst / name), stereo, TARGET_SR, subtype="PCM_24")
    print(f"  wrote {name:42s} {stereo.shape[0] / TARGET_SR:8.1f}s")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="finetune/data/datastereo/clean_moshi_audio_24khz")
    ap.add_argument("--dst", default="finetune/data/prepared_dialogue")
    ap.add_argument("--main-name", default="danielle",
                    help="case-insensitive substring identifying the cloned voice")
    ap.add_argument("--eval-conv", default="1411304343",
                    help="conv_id to carve an eval slice from")
    ap.add_argument("--eval-sec", type=float, default=600.0)
    ap.add_argument("--eval-center-frac", type=float, default=0.5)
    args = ap.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    convs = discover_conversations(src, args.main_name)
    if args.eval_conv not in convs:
        raise SystemExit(
            f"--eval-conv {args.eval_conv} not found among: {sorted(convs)}"
        )

    prefix = args.main_name.lower()
    for conv_id, (main_path, _ms, partner_path, partner_spk) in sorted(convs.items()):
        stereo, _sr, delta = combine_to_stereo(main_path, partner_path)
        print(f"conv {conv_id}  partner={partner_spk}  "
              f"len={stereo.shape[0] / TARGET_SR:.1f}s  align_delta={delta} frames")
        if conv_id == args.eval_conv:
            before, ev, after = carve_eval(
                stereo, TARGET_SR, args.eval_sec, args.eval_center_frac
            )
            if before.shape[0]:
                _write(dst, f"{prefix}_{partner_spk}_train_a.wav", before)
            _write(dst, f"{prefix}_{partner_spk}_eval.wav", ev)
            if after.shape[0]:
                _write(dst, f"{prefix}_{partner_spk}_train_b.wav", after)
        else:
            _write(dst, f"{prefix}_{partner_spk}.wav", stereo)

    print(f"\nDone -> {dst}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -v`
Expected: PASS (all 12 tests).

- [ ] **Step 5: Run the script on the real recordings**

Run:
```bash
python3 scripts/pair_dialogue_stereo.py \
  --src finetune/data/datastereo/clean_moshi_audio_24khz \
  --dst finetune/data/prepared_dialogue \
  --main-name danielle \
  --eval-conv 1411304343 --eval-sec 600 --eval-center-frac 0.5
```
Expected output (durations ~): conv `1341305451` partner `leenatantawy` ~68 s, `align_delta 0`; conv `1411304343` partner `joshuarhodes` ~5173.7 s split into `train_a` ~2286.9 s / `eval` 600.0 s / `train_b` ~2286.9 s; conv `1556527425` partner `clays` ~3544.5 s, `align_delta 600` frames. Five WAVs written to `finetune/data/prepared_dialogue/`.

- [ ] **Step 6: Write the real-data verification test**

Append to `tests/test_pair_dialogue_stereo.py`:
```python
REAL_SRC = Path(__file__).resolve().parents[1] / "finetune/data/datastereo/clean_moshi_audio_24khz"
REAL_DST = Path(__file__).resolve().parents[1] / "finetune/data/prepared_dialogue"


@pytest.mark.skipif(
    not (REAL_DST / "danielle_clays.wav").exists(),
    reason="run pair_dialogue_stereo.py on the real recordings first",
)
def test_real_outputs_format_and_channels():
    expected = {
        "danielle_clays.wav",
        "danielle_joshuarhodes_train_a.wav",
        "danielle_joshuarhodes_eval.wav",
        "danielle_joshuarhodes_train_b.wav",
        "danielle_leenatantawy.wav",
    }
    assert {p.name for p in REAL_DST.glob("*.wav")} == expected
    for name in expected:
        info = sf.info(str(REAL_DST / name))
        assert info.channels == 2, name
        assert info.samplerate == 24000, name
        assert info.subtype == "PCM_24", name

    # eval slice is exactly 600 s
    assert sf.info(str(REAL_DST / "danielle_joshuarhodes_eval.wav")).frames == 600 * 24000

    # Joshua split lengths sum to the original conversation (within 1 frame)
    orig = sf.info(str(REAL_SRC / "audioDanielleDeLosa11411304343_24khz.wav")).frames
    parts = sum(
        sf.info(str(REAL_DST / n)).frames
        for n in ("danielle_joshuarhodes_train_a.wav",
                  "danielle_joshuarhodes_eval.wav",
                  "danielle_joshuarhodes_train_b.wav")
    )
    assert abs(parts - orig) <= 1

    # left channel == Danielle source, right == Clay source (first 1 s, exact copy)
    out, _ = sf.read(str(REAL_DST / "danielle_clays.wav"), start=0, stop=24000, always_2d=True)
    dan, _ = sf.read(str(REAL_SRC / "audioDanielleDeLosa21556527425_24khz.wav"),
                     start=0, stop=24000, always_2d=True)
    clay, _ = sf.read(str(REAL_SRC / "audioClayS11556527425_24khz.wav"),
                      start=0, stop=24000, always_2d=True)
    assert np.array_equal(out[:, 0], dan[:, 0])
    assert np.array_equal(out[:, 1], clay[:, 0])
```

- [ ] **Step 7: Run the full test suite (incl. real-data verification)**

Run: `python3 -m pytest tests/test_pair_dialogue_stereo.py -v`
Expected: PASS (13 tests; the real-data test now runs instead of skipping).

- [ ] **Step 8: Listen-check one output (manual sync gate)**

Run: `open finetune/data/prepared_dialogue/danielle_leenatantawy.wav`
Expected: ~68 s stereo clip; Danielle on the left, Leena on the right, turns line up as a real back-and-forth (confirms the two source tracks are time-synced, not just equal-length). If they don't line up, STOP — the tracks have an offset and need re-aligning before training.

- [ ] **Step 9: Commit**

```bash
git add scripts/pair_dialogue_stereo.py tests/test_pair_dialogue_stereo.py
git commit -m "feat: CLI + real-data run for dialogue stereo pairing

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

Note: the ~1.4 GB of WAVs in `finetune/data/prepared_dialogue/` are **not** committed (large binaries); they are rsynced to the A100 in Task 5.

---

### Task 5: Point the training config at the new data + write the A100 runbook

**Files:**
- Modify: `example/moshika_rag_voice.yaml:8-11`
- Create: `docs/dialogue_prep_a100_runbook.md`

**Interfaces:**
- Consumes: the five WAVs from Task 4.
- Produces: a config that trains on `finetune/data/prepared_dialogue/`, and a runbook covering rsync → manifest → annotate → verify → train.

- [ ] **Step 1: Update the config data paths**

In `example/moshika_rag_voice.yaml`, replace lines 8-11:
```yaml
# ---- data ----
data:
  train_data: "finetune/data/prepared/train.jsonl"   # 8 files (~51 min); use all.jsonl to train on all 9
  eval_data:  "finetune/data/prepared/eval.jsonl"     # 1 held-out file (movies.wav)
  shuffle: true
```
with:
```yaml
# ---- data ----
data:
  train_data: "finetune/data/prepared_dialogue/train.jsonl"  # 4 files (~136 min dialogue: clay + joshua_train_a/b + leena)
  eval_data:  "finetune/data/prepared_dialogue/eval.jsonl"    # 1 held-out file (danielle_joshuarhodes_eval.wav, 10 min)
  shuffle: true
```

- [ ] **Step 2: Write the A100 runbook**

Create `docs/dialogue_prep_a100_runbook.md`:
```markdown
# A100 runbook — dialogue data prep → fork Path-B retrain

Prereqs: the 5 stereo WAVs exist locally in `finetune/data/prepared_dialogue/`
(produced by `scripts/pair_dialogue_stereo.py`). Do these ON THE A100 box.

## 1. Copy the WAVs to the box
From the Mac:
```bash
rsync -avP finetune/data/prepared_dialogue/*.wav \
  wb-gpu-training:~/moshi-finetune/finetune/data/prepared_dialogue/
```

## 2. Build manifests (absolute paths — must run on the box)
```bash
cd ~/moshi-finetune
uv run python scripts/build_manifest.py \
  --wav-dir finetune/data/prepared_dialogue \
  --out-dir finetune/data/prepared_dialogue \
  --eval-file danielle_joshuarhodes_eval.wav
```
Expect: `train.jsonl` = 4 files, `eval.jsonl` = 1 file, `all.jsonl` = 5 files.

## 3. Transcribe channel 0 (Danielle) — generates the required X.json
`annotate.py` reads channel 0, needs CUDA, and uses `-l` for local (no Slurm).
```bash
CUDA_VISIBLE_DEVICES=0 uv run python annotate.py \
  finetune/data/prepared_dialogue/all.jsonl -l --whisper_model medium --lang en
```
The 86-min file is within annotate.py's 4-hour guard. Each `X.wav` gets a
sibling `X.json` with `{"alignments": [[text,[start,end],"SPEAKER_MAIN"], ...]}`.

## 4. Verify every wav has a non-empty transcript
```bash
for w in finetune/data/prepared_dialogue/*.wav; do
  j="${w%.wav}.json"
  if [ ! -s "$j" ]; then echo "MISSING/EMPTY: $j"; fi
done
echo "check complete"
```
Expect: no "MISSING/EMPTY" lines. (Training crashes without a sibling .json —
interleaver.py:267-270.)

## 5. Train (fork stack, full LoRA on plain moshika)
```bash
CUDA_VISIBLE_DEVICES=0 uv run torchrun --nproc-per-node 1 -m train \
  example/moshika_rag_voice.yaml
```
Watch train vs eval loss. Take the earlier checkpoint if eval loss climbs.
Result: `runs/moshika_rag_voice/checkpoints/checkpoint_*/consolidated/lora.safetensors`.

> Data prep ends here. Serving the adapter on moshika-rag is Part 3 of NEXT_STEPS.
> Hyperparameter tuning for the larger (~2.4 hr) dataset — e.g. raising `max_steps`
> beyond the monologue-era 400 — is a separate follow-up.
```

- [ ] **Step 3: Commit**

```bash
git add example/moshika_rag_voice.yaml docs/dialogue_prep_a100_runbook.md
git commit -m "docs: point config at dialogue data + A100 prep runbook

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Notes for the executor

- The heavy real-data run (Task 4, Step 5) writes ~1.4 GB and reads ~1.3 GB of source audio; it takes a minute or two, not seconds. That's expected.
- Do **not** commit the WAVs in `finetune/data/prepared_dialogue/` — they're rsynced, not versioned. If a `.gitignore` entry is wanted, add `finetune/data/prepared_dialogue/*.wav`.
- Everything after Task 4 (manifests, transcripts, training) happens on the A100; nothing in Task 5 runs locally except the config edit and writing the runbook.
