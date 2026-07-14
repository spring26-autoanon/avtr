#!/usr/bin/env python3
"""
MoshiRAG eval runner.

Run evals:
  uv run evals/runner.py --config configs/baseline_with_retrieval.yaml [--mode smoke|sample|full]
  uv run evals/runner.py --config configs/baseline_with_retrieval.yaml --spot-check

Compare runs:
  uv run evals/runner.py --compare evals/results/run-a.json evals/results/run-b.json
"""

import abc
import argparse
import datetime
import importlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config import load_config
from core.model_interface import ModelInterface, StubModelAdapter
from core.retrieval_backend import GeminiAPIBackend, NullBackend

RESULTS_DIR = Path(__file__).parent / "results"
SEP = "─" * 57  # ─────...


# ── EvalResult ────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    eval_name: str
    scores: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    completed: bool = False
    last_completed_index: int = 0


# ── BaseEval ──────────────────────────────────────────────────────────────────


class BaseEval(abc.ABC):
    # Override in each eval: metric_key -> "higher" | "lower"
    METRIC_DIRECTIONS: dict[str, str] = {}

    @abc.abstractmethod
    def run(self, model: ModelInterface, config: dict, mode: str) -> EvalResult: ...

    def format_results(self, result: EvalResult) -> list[str]:
        """Console lines for this eval's result. Override for custom formatting."""
        lines = []
        for key, val in result.scores.items():
            direction = self.METRIC_DIRECTIONS.get(key, "")
            suffix = (
                "↑ higher is better"
                if direction == "higher"
                else "↓ lower is better"
                if direction == "lower"
                else ""
            )
            if isinstance(val, float):
                lines.append(f"  {key:<32} {val:.4g}  {suffix}".rstrip())
            else:
                lines.append(f"  {key:<32} {val}")
        return lines


# ── Model factory ─────────────────────────────────────────────────────────────


def build_model(cfg: dict) -> ModelInterface:
    model_cfg = cfg.get("model", {})
    adapter = model_cfg.get("adapter", "auto")
    retrieval_cfg = model_cfg.get("retrieval", {})

    if retrieval_cfg.get("enabled"):
        backend = GeminiAPIBackend(
            model=retrieval_cfg.get("model", "gemini-2.0-flash"),
            latency_gate_ms=retrieval_cfg["latency_gate_ms"],
        )
    else:
        backend = NullBackend()

    if adapter == "stub":
        return StubModelAdapter(retrieval_backend=backend)

    # Try real adapter; fall back to stub with a warning until Phase 4 (MoshiRAGAdapter)
    try:
        from core.model_interface import MoshiRAGAdapter
        from core.checkpoint import resolve_checkpoint
        local_path = resolve_checkpoint(model_cfg.get("checkpoint", "base"))
        return MoshiRAGAdapter(local_path, backend)
    except (ImportError, Exception) as e:
        print(f"⚠  MoshiRAGAdapter unavailable ({e}) — using StubModelAdapter", file=sys.stderr)
        return StubModelAdapter(retrieval_backend=backend)


# ── Eval discovery ────────────────────────────────────────────────────────────


def load_eval_class(name: str) -> type[BaseEval]:
    """Import evals.registry.<name> and return its BaseEval subclass."""
    module_path = f"evals.registry.{name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import eval '{name}': {e}") from e

    for _, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseEval) and obj is not BaseEval:
            return obj

    raise ValueError(f"No BaseEval subclass found in {module_path}")


# ── Git hash ──────────────────────────────────────────────────────────────────


def _git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ── Partial result detection ──────────────────────────────────────────────────


def find_partial_result(checkpoint: str, config_path: str, mode: str) -> dict | None:
    """Return the most recent incomplete result for this checkpoint+config+mode."""
    if not RESULTS_DIR.exists():
        return None
    candidates = []
    for f in RESULTS_DIR.glob("run-*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            if (
                data.get("checkpoint") == checkpoint
                and data.get("config") == config_path
                and data.get("mode") == mode
                and not data.get("completed", True)
            ):
                candidates.append((f.stat().st_mtime, data, f))
        except Exception:
            continue
    if not candidates:
        return None
    _, data, path = max(candidates, key=lambda x: x[0])
    print(f"Resuming from partial result: {path}")
    return data


# ── JSON I/O ──────────────────────────────────────────────────────────────────


def _result_to_dict(r: EvalResult) -> dict:
    return {
        "scores": r.scores,
        "metadata": r.metadata,
        "errors": r.errors,
        "completed": r.completed,
        "last_completed_index": r.last_completed_index,
    }


def write_results(
    run_doc: dict,
    eval_results: dict[str, EvalResult],
    completed: bool,
    output_path: Path,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    doc = {**run_doc, "completed": completed, "evals": {n: _result_to_dict(r) for n, r in eval_results.items()}}
    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2)


# ── Console formatting ────────────────────────────────────────────────────────


def _print_header(run_doc: dict) -> None:
    print("MoshiRAG Eval Run")
    print(f"checkpoint : {run_doc['checkpoint']}")
    print(f"config     : {run_doc['config']}")
    print(f"mode       : {run_doc['mode']}")
    print(f"git hash   : {run_doc['git_hash']}")
    print(f"timestamp  : {run_doc['timestamp']}")
    print(SEP)
    print()


def _print_eval_result(name: str, result: EvalResult, instance: BaseEval) -> None:
    print(f"[{name}]")
    for line in instance.format_results(result):
        print(line)
    for err in result.errors:
        print(f"  ⚠ {err}")
    if result.metadata.get("spot_check_completed") is False:
        print("  ⚠ spot_check_completed is false — run with --spot-check before trusting this metric")
    if result.metadata.get("gate_breached_p95"):
        gate = result.metadata.get("latency_gate_ms", "?")
        p95 = result.scores.get("total_p95_s", "?")
        print(f"  ⚠ p95 total ({p95}s) exceeds latency gate ({gate}ms) — scores may be quietly degraded")
    print()


# ── Runner ────────────────────────────────────────────────────────────────────


def run_evals(config_path: str, mode: str, spot_check: bool) -> None:
    cfg = load_config(config_path)
    checkpoint_raw = cfg.get("model", {}).get("checkpoint", "unknown")
    # Readable name: alias (e.g. "base") or final path component
    checkpoint_name = Path(checkpoint_raw).name if "/" in checkpoint_raw else checkpoint_raw

    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = now.strftime("%Y-%m-%dT%H-%M-%SZ")
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    run_doc = {
        "run_id": run_id,
        "checkpoint": checkpoint_name,
        "checkpoint_uri": checkpoint_raw,
        "config": config_path,
        "mode": mode,
        "git_hash": _git_hash(),
        "timestamp": timestamp,
    }
    output_path = RESULTS_DIR / f"run-{run_id}.json"

    _print_header(run_doc)

    # Resume from partial run if one exists
    partial = find_partial_result(checkpoint_name, config_path, mode)
    eval_results: dict[str, EvalResult] = {}
    if partial:
        for name, data in partial.get("evals", {}).items():
            eval_results[name] = EvalResult(
                eval_name=name,
                scores=data.get("scores", {}),
                metadata=data.get("metadata", {}),
                errors=data.get("errors", []),
                completed=data.get("completed", False),
                last_completed_index=data.get("last_completed_index", 0),
            )

    model = build_model(cfg)
    eval_names: list[str] = cfg.get("evals", [])
    all_completed = True

    for name in eval_names:
        prior = eval_results.get(name)
        if prior and prior.completed:
            print(f"[{name}] (resumed — already completed)\n")
            continue

        try:
            eval_cls = load_eval_class(name)
        except (ImportError, ValueError) as e:
            print(f"[{name}] ⚠ skipped: {e}\n")
            continue

        instance = eval_cls()
        eval_cfg = {
            **cfg,
            "_spot_check": spot_check,
            "_mode": mode,
            "_resume_from": prior.last_completed_index if prior else 0,
        }

        try:
            result = instance.run(model, eval_cfg, mode)
        except Exception as exc:
            result = EvalResult(eval_name=name, errors=[str(exc)], completed=False)
            eval_results[name] = result
            write_results(run_doc, eval_results, completed=False, output_path=output_path)
            print(f"\n✗ eval '{name}' failed: {exc}")
            print(f"  partial results written to: {output_path}")
            sys.exit(1)

        eval_results[name] = result
        _print_eval_result(name, result, instance)
        if not result.completed:
            all_completed = False

        # Write after each eval so a mid-run crash doesn't lose everything
        write_results(run_doc, eval_results, completed=False, output_path=output_path)

    write_results(run_doc, eval_results, completed=all_completed, output_path=output_path)
    print(SEP)
    print(f"results written to: {output_path}")


# ── Comparison CLI ────────────────────────────────────────────────────────────

# Direction-of-improvement registry — mirrors METRIC_DIRECTIONS on each eval class
_DIRECTIONS: dict[str, str] = {
    "triviaqa_acc": "higher",
    "webq_acc": "higher",
    "llamaq_acc": "higher",
    "ref_acc": "higher",
    "resp_acc": "higher",
    "pause_tor_synthetic": "lower",
    "pause_tor_candor": "lower",
    "backchannel_freq_per_sec": "higher",
    "backchannel_jsd": "lower",
    "turn_taking_tor": "higher",
    "turn_taking_latency_s": "lower",
    "interruption_gpt_score": "higher",
    "interruption_latency_s": "lower",
    "mean_s": "lower",
    "p50_s": "lower",
    "p95_s": "lower",
    "mean_ttfat_s": "lower",
    "mean_keyword_delay_s": "lower",
    "mean_e2ekd_s": "lower",
    "p95_e2ekd_s": "lower",
    "asr_wait_mean_s": "lower",
    "asr_wait_p95_s": "lower",
    "api_call_mean_s": "lower",
    "api_call_p95_s": "lower",
    "context_injection_mean_s": "lower",
    "context_injection_p95_s": "lower",
    "total_mean_s": "lower",
    "total_p95_s": "lower",
    "wer": "lower",
    "speaker_similarity": "higher",
}


def _indicator(key: str, delta: float) -> str:
    d = _DIRECTIONS.get(key)
    if d == "higher":
        return "✓" if delta >= 0 else "↓"
    if d == "lower":
        return "✓" if delta <= 0 else "↓"
    return ""


def compare_runs(path_a: str, path_b: str) -> None:
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)

    date_a = a.get("timestamp", "")[:10]
    date_b = b.get("timestamp", "")[:10]
    print("comparing:")
    print(f"  A  {a['checkpoint']:<10} ({date_a})  {a['config']}  [{a['mode']}]")
    print(f"  B  {b['checkpoint']:<10} ({date_b})  {b['config']}  [{b['mode']}]")
    print(SEP)
    print()

    evals_a = a.get("evals", {})
    evals_b = b.get("evals", {})
    seen: set[str] = set()
    names: list[str] = []
    for n in list(evals_a) + list(evals_b):
        if n not in seen:
            names.append(n)
            seen.add(n)

    gate_msgs: list[str] = []

    for name in names:
        sa = evals_a.get(name, {}).get("scores", {})
        sb = evals_b.get(name, {}).get("scores", {})
        keys = list(dict.fromkeys(list(sa) + list(sb)))
        if not keys:
            continue

        print(name)
        for key in keys:
            va, vb = sa.get(key), sb.get(key)
            if va is None or vb is None:
                continue
            delta = vb - va
            ind = _indicator(key, delta)
            if "acc" in key or "similarity" in key or "wer" in key:
                print(f"  {key:<24} A: {va*100:.1f}%   B: {vb*100:.1f}%   Δ {delta*100:+.1f}pp  {ind}")
            else:
                print(f"  {key:<24} A: {va:<8.4g} B: {vb:<8.4g} Δ {delta:+.4g}  {ind}")

        # rag_lift_pp — resp_acc delta labeled explicitly for halu_eval
        if name == "knowledge.halu_eval_audio" and "resp_acc" in sa and "resp_acc" in sb:
            lift = (sb["resp_acc"] - sa["resp_acc"]) * 100
            print(f"  {'rag_lift_pp':<24} (computed) A→B: {lift:+.1f}pp over baseline")

        # Gate warning accumulation
        ma = evals_a.get(name, {}).get("metadata", {})
        mb = evals_b.get(name, {}).get("metadata", {})
        if ma.get("gate_breached_p95") or mb.get("gate_breached_p95"):
            gate_ms = mb.get("latency_gate_ms") or ma.get("latency_gate_ms", "?")
            both = ma.get("gate_breached_p95") and mb.get("gate_breached_p95")
            pa = sa.get("total_p95_s", "?")
            pb = sb.get("total_p95_s", "?")
            gate_msgs.append(
                f"retrieval p95 exceeds {gate_ms}ms in {'both runs' if both else 'one run'} "
                f"(A={pa}s, B={pb}s) — review infrastructure"
            )

        print()

    print(SEP)
    for msg in gate_msgs:
        print(f"⚠ {msg}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="MoshiRAG eval runner")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument(
        "--mode",
        choices=["smoke", "sample", "full"],
        default="sample",
        help="Eval mode: smoke (~10min) | sample (~1-2h, default) | full (8-16h)",
    )
    parser.add_argument(
        "--spot-check",
        action="store_true",
        help="Output 20 E2EKD examples for human review",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help="Compare two result JSON files",
    )
    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare[0], args.compare[1])
    elif args.config:
        run_evals(args.config, args.mode, args.spot_check)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
