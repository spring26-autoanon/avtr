import ast
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
VOICE = REPO / "example/moshika_voice_max.yaml"
S4A = REPO / "example/moshika_rag_stage4a.yaml"
S4B = REPO / "example/moshika_rag_stage4b.yaml"


def cfg(p):
    return yaml.safe_load(p.read_text())


def _trainargs_fields():
    tree = ast.parse((REPO / "finetune/args.py").read_text())
    cls = next(c for c in ast.walk(tree)
               if isinstance(c, ast.ClassDef) and c.name == "TrainArgs")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def test_every_key_is_accepted_by_the_trainer():
    """A typo'd key would only surface on the box, mid-launch."""
    fields = _trainargs_fields()
    for p in (VOICE, S4A, S4B):
        assert not set(cfg(p)) - fields, p.name


def test_voice_track_uses_plain_moshika_and_stripped_config():
    """Under the pinned fork, the DEFAULT architecture (no config.json) includes the RAG
    ARC conditioner, which crashes on the meta device at load — verified live on
    wb-gpu-training 2026-08-07. Plain moshika therefore trains with the conditioner-
    stripped moshika-rag config: identical 7B dims, no conditioners/fuser."""
    c = cfg(VOICE)
    assert c["moshi_paths"]["hf_repo_id"] == "kyutai/moshika-pytorch-bf16"
    assert c["moshi_paths"]["config_path"].endswith("config.stripped.json")


def test_voice_track_matches_the_readme_recommendation():
    c = cfg(VOICE)
    assert c["lora"]["rank"] == 128
    assert c["batch_size"] == 16
    assert c["max_steps"] == 2000
    assert float(c["optim"]["lr"]) == 2e-6


def test_voice_track_data_excludes_marker_bearing_clips():
    """<RAG> maps to raw token 4 — a rag token on moshika-rag, an ordinary token on plain
    moshika. Feeding marked clips to the voice track injects garbage."""
    sources = cfg(VOICE)["data"]["train_data"]
    assert "replay/retrieval/" not in sources, "must use the marker-stripped copy"
    assert "retrieval_nomarkers" in sources


def test_rag_tracks_use_the_rag_base_and_spike_config():
    for p in (S4A, S4B):
        c = cfg(p)
        assert c["moshi_paths"]["hf_repo_id"] == "kyutai/moshika-rag-pytorch-bf16"
        assert c["moshi_paths"]["config_path"].endswith("config.spike.json")


def test_4a_holds_every_stage3_hyperparameter_so_it_isolates_the_delay_fix():
    c = cfg(S4A)
    assert c["lora"]["rank"] == 64
    assert c["batch_size"] == 8
    assert c["max_steps"] == 800
    assert float(c["optim"]["lr"]) == 4e-6


def test_4b_reduces_intervention_relative_to_4a():
    a, b = cfg(S4A), cfg(S4B)
    assert b["lora"]["rank"] < a["lora"]["rank"]
    assert b["max_steps"] < a["max_steps"]
    assert float(b["optim"]["lr"]) < float(a["optim"]["lr"])


def test_4b_rank_is_32_not_16():
    """Rank 64 is the only value with evidence behind it; a two-step drop would make a
    voice regression uninterpretable."""
    assert cfg(S4B)["lora"]["rank"] == 32


def test_all_runs_write_to_distinct_run_dirs():
    dirs = [cfg(p)["run_dir"] for p in (VOICE, S4A, S4B)]
    assert len(set(dirs)) == 3
    # the trainer refuses a pre-existing run_dir rather than clobbering it
    assert all("overwrite_run_dir" not in cfg(p) for p in (VOICE, S4A, S4B))


def test_lr_parses_as_a_float_not_a_string():
    """`lr: 2e-6` is a STRING in YAML — a float needs the decimal point (`2.0e-6`).
    Serializable coerces it, so this has never broken, but the config should not rely on
    that. Every prior config in this repo has the latent form."""
    for p in (VOICE, S4A, S4B):
        assert isinstance(cfg(p)["optim"]["lr"], float), p.name


def test_all_data_paths_are_absolute():
    """sphn.dataset_jsonl resolves manifest paths relative to the jsonl's own directory."""
    for p in (VOICE, S4A, S4B):
        c = cfg(p)
        for src in c["data"]["train_data"].split(","):
            assert src.startswith("/"), (p.name, src)
        assert c["data"]["eval_data"].startswith("/")


def test_checkpoints_are_all_retained():
    for p in (VOICE, S4A, S4B):
        c = cfg(p)
        assert c["max_steps"] // c["ckpt_freq"] <= c["num_ckpt_keep"], p.name


def test_rag_token_weight_stays_out_of_the_configs():
    """Chosen from the measured turn mix and passed in the environment."""
    for p in (S4A, S4B):
        assert "rag_token_weight" not in cfg(p)
