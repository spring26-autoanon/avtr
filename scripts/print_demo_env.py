#!/usr/bin/env python3
"""
Prints a shell-sourceable block of env vars for scripts/run_demo.sh, derived
from the same --config YAML core/config.py resolves for evals — see specs/
moshirag-evals-requirements.md's "Config-derived environment" section under
"Demo" for the full design.

Usage (called from run_demo.sh via `eval "$(...)"`, not typically run directly):
  uv run --all-extras python scripts/print_demo_env.py --config configs/baseline_with_retrieval.yaml

Prints:
  export LLM_BASE_URL=... / LLM_MODEL_NAME=... / LLM_API_KEY=...
    Needed only because ServerState.__init__ unconditionally constructs its
    own LLMReferenceGenerator and ServerState.warmup() makes one real,
    synchronous API call against it before the server accepts a connection
    — these are otherwise vestigial once scripts/instrumented_server.py's
    get_reference_text patch is active (real retrieval never reaches
    LLMReferenceGenerator again). Falls back to core/model_interface.py's
    own defaults when the config's retrieval backend doesn't resolve to a
    Gemini-shaped model/base_url (e.g. retrieval disabled, or a
    null_backend).
  export DEMO_GENERATION_FLAGS=...
    model.generation translated into moshi.server's own CLI flags — a pure
    field-name transform (underscores -> hyphens), since those field names
    were chosen to match moshi.server's argparse flags exactly (see
    core/model_interface.py's "Generation parameters" table). Excludes
    tail_silence_steps, which has no demo equivalent (respond()-only).
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from core.model_interface import _DEFAULT_GENERATION, _DEFAULT_LLM_BASE_URL, _DEFAULT_LLM_MODEL_NAME

# No demo (moshi.server) CLI equivalent — respond()-only, see
# _DEFAULT_GENERATION's own comment in core/model_interface.py.
_DEMO_EXCLUDED_GENERATION_FIELDS = {"tail_silence_steps"}


def _retrieval_env(config: dict) -> dict[str, str]:
    retrieval = config.get("model", {}).get("retrieval", {})
    backend_def = retrieval.get("_resolved_backend") or {}
    api_key_env = backend_def.get("api_key_env", "GEMINI_API_KEY")
    return {
        "LLM_BASE_URL": backend_def.get("base_url", _DEFAULT_LLM_BASE_URL),
        "LLM_MODEL_NAME": backend_def.get("model", _DEFAULT_LLM_MODEL_NAME),
        "LLM_API_KEY": os.environ.get(api_key_env, ""),
    }


def _generation_flags(config: dict) -> str:
    generation = {**_DEFAULT_GENERATION, **config.get("model", {}).get("generation", {})}
    parts: list[str] = []
    for field, value in generation.items():
        if field in _DEMO_EXCLUDED_GENERATION_FIELDS:
            continue
        parts.append("--" + field.replace("_", "-"))
        parts.append(str(value))
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)

    for key, value in _retrieval_env(config).items():
        print(f"export {key}={shlex.quote(value)}")
    print(f"export DEMO_GENERATION_FLAGS={shlex.quote(_generation_flags(config))}")


if __name__ == "__main__":
    main()
