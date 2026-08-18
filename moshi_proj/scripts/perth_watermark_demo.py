"""PerTh watermark demo — provenance for the cloned voice.

Answers the privacy/security question: every clip our system produces can carry
an inaudible watermark, so synthetic "Danielle" audio is verifiable as synthetic
even after re-encoding, while real recordings of her show no mark.

Setup (once, any Python env — NOT added to pyproject.toml so the box env is untouched):
    pip install resemble-perth librosa soundfile

Usage:
    python scripts/perth_watermark_demo.py <input.wav>            # embed + verify one clip
    python scripts/perth_watermark_demo.py <dir>                  # embed every wav in a directory
    python scripts/perth_watermark_demo.py --check <audio_file>   # detector only

The full demo run:
    python scripts/perth_watermark_demo.py cloned_danielle.wav
prints a table: detection on the original (none), on the watermarked copy
(present), and on an MP3 round-trip of the watermarked copy (still present),
plus the peak difference between original and watermarked audio in dBFS to show
the mark is far below audibility.

Batch mode writes `<name>_watermarked.wav` next to each input and prints one
verdict line per file — run it over the demo-video audio before publishing.
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import perth


def load(path: Path):
    # Perth operates on mono float32; keep the file's native sample rate.
    wav, sr = librosa.load(str(path), sr=None, mono=True)
    return wav.astype(np.float32), sr


def detect(watermarker, path: Path) -> float:
    wav, sr = load(path)
    return float(watermarker.get_watermark(wav, sample_rate=sr))


def embed(watermarker, src: Path, out: Path | None = None) -> Path:
    wav, sr = load(src)
    out = out or src.with_name(src.stem + "_watermarked.wav")
    watermarked = watermarker.apply_watermark(wav, watermark=None, sample_rate=sr)
    sf.write(str(out), watermarked, sr)
    return out


def mp3_roundtrip(src: Path, dst_wav: Path, kbps: int = 64) -> bool:
    """Re-encode through MP3 at a lossy bitrate, back to wav. Needs ffmpeg."""
    if shutil.which("ffmpeg") is None:
        return False
    mp3 = dst_wav.with_suffix(".mp3")
    for cmd in (
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-b:a", f"{kbps}k", str(mp3)],
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3), str(dst_wav)],
    ):
        subprocess.run(cmd, check=True)
    return True


def demo_single(watermarker, src: Path, out: Path | None):
    wav, _ = load(src)
    out = embed(watermarker, src, out)
    marked, _ = load(out)

    # Perceptual impact: peak sample difference relative to full scale.
    n = min(len(wav), len(marked))
    peak_diff = float(np.max(np.abs(marked[:n] - wav[:n])))
    peak_db = 20 * np.log10(peak_diff) if peak_diff > 0 else -np.inf

    rows = [
        ("original (unmarked)", detect(watermarker, src)),
        ("watermarked copy", detect(watermarker, out)),
    ]
    with tempfile.TemporaryDirectory() as td:
        rt = Path(td) / "roundtrip.wav"
        if mp3_roundtrip(out, rt):
            rows.append(("watermarked -> 64kbps MP3 -> wav", detect(watermarker, rt)))
        else:
            print("(ffmpeg not found — skipping MP3 robustness check)")

    print(f"\nwrote: {out}")
    print(f"peak audio difference from watermark: {peak_db:.1f} dBFS (inaudible)\n")
    print(f"{'clip':<38}{'confidence':>10}  verdict")
    for name, score in rows:
        verdict = "SYNTHETIC (ours)" if score > 0.5 else "no watermark"
        print(f"{name:<38}{score:>10.3f}  {verdict}")


def demo_batch(watermarker, src_dir: Path):
    wavs = sorted(p for p in src_dir.glob("*.wav") if not p.stem.endswith("_watermarked"))
    if not wavs:
        print(f"no .wav files in {src_dir}")
        return
    print(f"{'clip':<44}{'confidence':>10}  verdict")
    for src in wavs:
        out = embed(watermarker, src)
        score = detect(watermarker, out)
        verdict = "SYNTHETIC (ours)" if score > 0.5 else "EMBED FAILED"
        print(f"{out.name:<44}{score:>10.3f}  {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path, help="input audio file, or a directory of wavs (batch mode)")
    ap.add_argument("--check", action="store_true", help="only run the detector on the file")
    ap.add_argument("--out", type=Path, default=None, help="where to write the watermarked wav (single-file mode)")
    args = ap.parse_args()

    watermarker = perth.PerthImplicitWatermarker()

    if args.check:
        score = detect(watermarker, args.audio)
        print(f"{args.audio.name}: watermark confidence = {score:.3f} "
              f"({'SYNTHETIC (ours)' if score > 0.5 else 'no watermark found'})")
    elif args.audio.is_dir():
        demo_batch(watermarker, args.audio)
    else:
        demo_single(watermarker, args.audio, args.out)


if __name__ == "__main__":
    main()
