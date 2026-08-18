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


def test_train_sources_are_the_retrieval_clips_and_the_natural_dialogue():
    sources = _cfg()["data"]["train_data"].split(",")
    assert "replay/retrieval" in sources[0]
    assert "prepared_dialogue" in sources[1]


def test_all_data_paths_are_absolute():
    """sphn.dataset_jsonl resolves manifest paths relative to the jsonl's own directory."""
    cfg = _cfg()
    for source in cfg["data"]["train_data"].split(","):
        assert source.startswith("/"), source
    assert cfg["data"]["eval_data"].startswith("/")


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
    assert cfg["max_steps"] // cfg["ckpt_freq"] <= cfg["num_ckpt_keep"]


def test_rag_token_weight_is_not_baked_into_the_config():
    """It is chosen from the turn mix segmentation reports, and passed in the env."""
    assert "RAG_TOKEN_WEIGHT" not in CONFIG.read_text().split("# ")[0]
    assert "rag_token_weight" not in _cfg()
