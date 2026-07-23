import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv

_CHECKPOINTS_YAML = Path(__file__).parent.parent / "configs" / "checkpoints.yaml"
_RETRIEVAL_BACKENDS_YAML = Path(__file__).parent.parent / "configs" / "retrieval_backends.yaml"
_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _interpolate(value):
    """Recursively replace ${VAR} in string values with environment variables."""
    if isinstance(value, str):
        def _replace(m):
            var = m.group(1)
            result = os.environ.get(var)
            if result is None:
                raise ValueError(f"Environment variable ${{{var}}} is not set")
            return result
        return _VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def _load_checkpoint_aliases() -> dict:
    if not _CHECKPOINTS_YAML.exists():
        return {}
    with open(_CHECKPOINTS_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("aliases", {}) if data else {}


def _load_retrieval_backends() -> dict:
    if not _RETRIEVAL_BACKENDS_YAML.exists():
        return {}
    with open(_RETRIEVAL_BACKENDS_YAML) as f:
        data = yaml.safe_load(f)
    return data.get("backends", {}) if data else {}


def load_config(path: str) -> dict:
    load_dotenv()
    with open(path) as f:
        config = yaml.safe_load(f)
    config = _interpolate(config)

    # Resolve checkpoint alias — alias value may itself contain ${VAR}, interpolate after dotenv
    aliases = _load_checkpoint_aliases()
    checkpoint = config.get("model", {}).get("checkpoint")
    if checkpoint and checkpoint in aliases:
        config["model"]["checkpoint"] = _interpolate(aliases[checkpoint])

    # Validate required fields
    retrieval = config.get("model", {}).get("retrieval", {})
    if retrieval.get("enabled"):
        if "latency_gate_ms" not in retrieval:
            raise ValueError("latency_gate_ms is required when retrieval is enabled")

    # Resolve model.retrieval.backend against configs/retrieval_backends.yaml
    # — same alias-file pattern as checkpoints — and attach the resolved
    # definition so callers (evals/runner.py's build_model(),
    # scripts/instrumented_server.py's main()) don't each re-read the file
    # or re-implement the lookup.
    backend_name = retrieval.get("backend")
    if backend_name:
        backends = _load_retrieval_backends()
        if backend_name not in backends:
            raise ValueError(
                f"model.retrieval.backend {backend_name!r} not found in "
                f"configs/retrieval_backends.yaml (known backends: {sorted(backends, key=str)})"
            )
        retrieval["_resolved_backend"] = backends[backend_name]

    return config
