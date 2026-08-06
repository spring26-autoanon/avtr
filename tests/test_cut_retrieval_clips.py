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


def test_markers_stay_in_chronological_order():
    """train.py pairs the i-th marker with the i-th reference, so order is load-bearing."""
    alignments = [_al("one", 1.0, 1.5), _al("two", 9.0, 9.5), _al("three", 20.0, 20.5)]
    out = insert_rag_markers(alignments, [20.0, 1.0, 9.0])   # deliberately unsorted
    starts = [w[1][0] for w in out if w[0] == "<RAG>"]
    assert starts == sorted(starts)
    assert [w[0] for w in out] == ["<RAG>", "one", "<RAG>", "two", "<RAG>", "three"]


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


def test_cut_all_audio_matches_the_master_slice(tmp_path):
    master = tmp_path / "master.wav"
    _write_master(master)
    alignments = [_al("hello", 11.0, 11.5)]
    segments = [_segment("ret-0", 10.0, 20.0, [
        _turn("JOSHUA", 10.0, 11.0, "smalltalk"),
        _turn("DANIELLE", 11.0, 20.0, "smalltalk"),
    ])]
    cut_all(master, alignments, segments, tmp_path / "clips", tmp_path / "m.jsonl")
    clip, _ = sf.read(str(tmp_path / "clips" / "ret-0.wav"), always_2d=True)
    expect, _ = sf.read(str(master), start=10 * 24000, stop=20 * 24000, always_2d=True)
    assert np.array_equal(clip, expect)


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


def test_cut_all_skips_a_segment_with_no_danielle_words(tmp_path):
    master = tmp_path / "master.wav"
    _write_master(master)
    segments = [_segment("ret-0", 10.0, 20.0, [
        _turn("JOSHUA", 10.0, 11.0, "smalltalk"),
        _turn("DANIELLE", 14.0, 20.0, "smalltalk"),
    ])]
    outdir = tmp_path / "clips"
    manifest = tmp_path / "manifest.jsonl"
    cut_all(master, [], segments, outdir, manifest)
    assert not (outdir / "ret-0.wav").exists()


def test_cut_all_smalltalk_only_clip_has_no_markers_and_no_references(tmp_path):
    master = tmp_path / "master.wav"
    _write_master(master)
    alignments = [_al("chatting", 12.0, 12.5)]
    segments = [_segment("ret-0", 10.0, 20.0, [
        _turn("JOSHUA", 10.0, 11.0, "smalltalk"),
        _turn("DANIELLE", 12.0, 20.0, "smalltalk"),
    ])]
    outdir = tmp_path / "clips"
    manifest = tmp_path / "manifest.jsonl"
    cut_all(master, alignments, segments, outdir, manifest)
    data = json.load(open(outdir / "ret-0.json"))
    assert not any(w[0] == "<RAG>" for w in data["alignments"])
    assert json.loads(manifest.read_text().strip())["references"] == []
