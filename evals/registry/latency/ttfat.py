import logging

from core.model_interface import ModelInterface
from evals.registry.knowledge.open_audio_bench import _SUBSETS, _load_audio_bytes, _load_rows
from evals.runner import BaseEval, EvalResult

logger = logging.getLogger(__name__)

# "200 for latency" per the Eval Runner mode table (sample mode's latency-eval
# override, distinct from knowledge evals' 100-per-subset). "full" => every
# row across every subset (~1,500 total, exact count is whatever the real
# datasets contain).
_MODE_SAMPLE_SIZE = {"tiny": 1, "smoke": 5, "sample": 200}


def _load_question_pool(limit_total: int | None) -> list[tuple[str, dict, dict]]:
    """
    Combines questions from every knowledge.open_audio_bench subset
    (TriviaQA/WebQ/LlamaQ) into one flat pool — "the knowledge eval
    question set" this eval samples TTFAT from, per
    specs/moshirag-evals-requirements.md. Round-robin across subsets
    (not subset-by-subset) so even a small limit_total gets cross-subset
    diversity rather than being all TriviaQA. Reuses
    knowledge.open_audio_bench's own dataset/subset logic rather than a
    second, independent copy of the same CSV-parsing code.
    """
    per_subset_rows = {key: _load_rows(cfg, None) for key, cfg in _SUBSETS.items()}
    pool: list[tuple[str, dict, dict]] = []
    i = 0
    while any(i < len(rows) for rows in per_subset_rows.values()):
        for subset_key, subset_cfg in _SUBSETS.items():
            rows = per_subset_rows[subset_key]
            if i < len(rows):
                pool.append((subset_key, subset_cfg, rows[i]))
        i += 1
    return pool if limit_total is None else pool[:limit_total]


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy's default method) — no new
    dependency needed for such a small, self-contained computation."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


class TtfatEval(BaseEval):
    METRIC_DIRECTIONS = {"mean_s": "lower", "p50_s": "lower", "p95_s": "lower"}

    def run(self, model: ModelInterface, config: dict, mode: str) -> EvalResult:
        limit = _MODE_SAMPLE_SIZE.get(mode)  # None => full pool

        prior = config.get("_prior_result")
        progress: dict = dict(prior.metadata.get("_progress", {})) if prior is not None else {}
        progress.setdefault("total", 0)
        progress.setdefault("ttfat_values", [])
        progress.setdefault("degenerate_silence", 0)
        errors: list[str] = list(prior.errors) if prior is not None else []
        transcript: list[dict] = list(prior.transcript) if prior is not None else []

        pool = _load_question_pool(limit)

        for i, (subset_key, subset_cfg, row) in enumerate(pool):
            if i < progress["total"]:
                continue  # already measured in a prior partial run

            entry: dict = {"index": i, "subset": subset_key, "question": row.get(subset_cfg["question_col"])}
            try:
                audio_in = _load_audio_bytes(subset_cfg, row)
                _, _, resp_metadata = model.respond(audio_in)
                ttfat_s = resp_metadata["ttfat_s"]
                entry["ttfat_s"] = ttfat_s
                entry["degenerate_silence"] = resp_metadata.get("degenerate_silence", False)
                progress["ttfat_values"].append(ttfat_s)
                if resp_metadata.get("degenerate_silence"):
                    progress["degenerate_silence"] += 1
            except Exception as exc:
                logger.warning("ttfat[%d] failed: %s", i, exc)
                errors.append(f"[{i}]: {exc}")
                entry["error"] = str(exc)

            transcript.append(entry)
            progress["total"] += 1

        values = sorted(progress["ttfat_values"])
        scores = {}
        if values:
            scores["mean_s"] = sum(values) / len(values)
            scores["p50_s"] = _percentile(values, 50)
            scores["p95_s"] = _percentile(values, 95)

        # n reflects the number of valid TTFAT measurements the stats above
        # were actually computed over, not the (possibly larger) attempted
        # count in progress["total"] — a question whose respond() call
        # errored out contributes no data point, so it shouldn't inflate the
        # reported sample size.
        metadata: dict = {"n": len(values), "_progress": progress}
        if progress["degenerate_silence"]:
            metadata["degenerate_silence_count"] = progress["degenerate_silence"]

        return EvalResult(
            eval_name="latency.ttfat",
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
        n = result.metadata.get("n", 0)
        return [
            f"  mean : {scores['mean_s']:.2f}s   p50 : {scores['p50_s']:.2f}s   p95 : {scores['p95_s']:.2f}s",
            f"  (n={n})",
        ]
