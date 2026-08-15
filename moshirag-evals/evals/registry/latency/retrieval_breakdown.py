import logging

from core.model_interface import ModelInterface
from evals.registry.knowledge.open_audio_bench import _load_audio_bytes
from evals.registry.latency.ttfat import _load_question_pool, _percentile
from evals.runner import BaseEval, EvalResult

logger = logging.getLogger(__name__)

# Same "200 for latency" sample-mode size as latency.ttfat/latency.e2ekd —
# see the Eval Runner mode table.
_MODE_SAMPLE_SIZE = {"tiny": 1, "smoke": 5, "sample": 200}

_STAGES = ("asr_wait", "api_call", "context_injection", "total")


class RetrievalBreakdownEval(BaseEval):
    METRIC_DIRECTIONS = {f"{stage}_{stat}_s": "lower" for stage in _STAGES for stat in ("mean", "p95")}

    def run(self, model: ModelInterface, config: dict, mode: str) -> EvalResult:
        limit = _MODE_SAMPLE_SIZE.get(mode)  # None => full pool

        prior = config.get("_prior_result")
        progress: dict = dict(prior.metadata.get("_progress", {})) if prior is not None else {}
        progress.setdefault("total", 0)
        for stage in _STAGES:
            progress.setdefault(f"{stage}_values", [])
        errors: list[str] = list(prior.errors) if prior is not None else []
        transcript: list[dict] = list(prior.transcript) if prior is not None else []

        pool = _load_question_pool(limit)

        for i, (subset_key, subset_cfg, row) in enumerate(pool):
            if i < progress["total"]:
                continue  # already measured in a prior partial run

            entry: dict = {"index": i, "subset": subset_key}
            try:
                audio_in = _load_audio_bytes(subset_cfg, row)
                _, _, resp_metadata = model.respond(audio_in)
                rag_triggered = bool(resp_metadata.get("rag_triggered"))
                entry["rag_triggered"] = rag_triggered

                if rag_triggered:
                    asr_wait_s = resp_metadata["asr_wait_s"]
                    api_call_s = resp_metadata["retrieval_latency_s"]
                    # Real ARC-encoder /embed round trip (job._async_update_
                    # reference's get_conditioning_remote_async call) — see
                    # core/model_interface.py's conditioning_latency_s
                    # docstring. Previously this stage measured a redundant,
                    # no-op format_context() call instead (always ~0.00s),
                    # missing the actual, and typically dominant, cost of
                    # conditioning the LM on the retrieved reference.
                    context_injection_s = resp_metadata["conditioning_latency_s"]

                    total_s = asr_wait_s + api_call_s + context_injection_s

                    entry.update({
                        "asr_wait_s": asr_wait_s,
                        "api_call_s": api_call_s,
                        "context_injection_s": context_injection_s,
                        "total_s": total_s,
                    })
                    progress["asr_wait_values"].append(asr_wait_s)
                    progress["api_call_values"].append(api_call_s)
                    progress["context_injection_values"].append(context_injection_s)
                    progress["total_values"].append(total_s)
            except Exception as exc:
                logger.warning("retrieval_breakdown[%d] failed: %s", i, exc)
                errors.append(f"[{i}]: {exc}")
                entry["error"] = str(exc)

            transcript.append(entry)
            progress["total"] += 1

        scores: dict = {}
        for stage in _STAGES:
            values = progress[f"{stage}_values"]
            if not values:
                continue
            sorted_values = sorted(values)
            scores[f"{stage}_mean_s"] = sum(sorted_values) / len(sorted_values)
            scores[f"{stage}_p95_s"] = _percentile(sorted_values, 95)

        # n reflects how many questions actually triggered retrieval and
        # produced a real breakdown data point — not progress["total"]
        # (attempted count), since a question with no <ret> contributes no
        # timing sample at all, same convention latency.ttfat uses.
        metadata: dict = {"n": len(progress["total_values"]), "_progress": progress}

        latency_gate_ms = config.get("model", {}).get("retrieval", {}).get("latency_gate_ms")
        if latency_gate_ms is not None:
            metadata["latency_gate_ms"] = latency_gate_ms
            if "total_p95_s" in scores and scores["total_p95_s"] * 1000 > latency_gate_ms:
                metadata["gate_breached_p95"] = True
                errors.append(
                    f"p95 total retrieval latency ({scores['total_p95_s']:.2f}s) exceeds "
                    f"gate ({latency_gate_ms}ms) — scores may be quietly degraded"
                )

        return EvalResult(
            eval_name="latency.retrieval_breakdown",
            scores=scores,
            metadata=metadata,
            errors=errors,
            completed=True,
            last_completed_index=progress["total"],
            transcript=transcript,
        )

    def format_results(self, result: EvalResult) -> list[str]:
        scores = result.scores
        if not scores:
            return []
        labels = [
            ("asr_wait", "asr transcription wait"),
            ("api_call", "gemini api call"),
            ("context_injection", "context injection"),
            ("total", "total retrieval delay"),
        ]
        lines = []
        for key, label in labels:
            mean_key, p95_key = f"{key}_mean_s", f"{key}_p95_s"
            if mean_key in scores:
                lines.append(f"  {label:<23} mean: {scores[mean_key]:.2f}s   p95: {scores[p95_key]:.2f}s")
        return lines
