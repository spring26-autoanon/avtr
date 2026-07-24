import logging
import re
from pathlib import Path

from datasets import load_dataset

from core.llm_judge import call_gemini
from core.model_interface import ModelInterface
from core.tts import build_tts_backend, describe_tts_backend, synthesize_cached
from evals.runner import BaseEval, DEFAULT_JUDGE_MODEL, EvalResult

logger = logging.getLogger(__name__)

# Not part of the MoshiRAG paper's own eval suite (Table 1 covers
# TriviaQA/WebQ/LlamaQ/HaluEvalAudio only) — added independently to track a
# standard, widely-cited benchmark number. GSM8K has no existing audio
# version, so question audio is synthesized via core/tts.py and disk-cached
# indefinitely (see specs/moshirag-evals-requirements.md's "Knowledge"
# section for the full rationale, including why OpenAudioBench's own
# reasoning_qa subset isn't used as a substitute).
_REPO_ID = "openai/gsm8k"
_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "tts_cache" / "gsm8k"

_MODE_SAMPLE_SIZE = {"tiny": 1, "smoke": 5, "sample": 100}  # "full" => all 1,319 rows

# GSM8K's own convention: the answer field ends in a literal "#### <number>"
# line marking the final answer.
_GROUND_TRUTH_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# A project-owned extraction prompt, not a paper prompt (GSM8K isn't part of
# the MoshiRAG paper's own eval suite) — handles MoshiRAG spelling a number
# out as words rather than digits, which a plain regex over the transcribed
# text can't reliably catch.
_EXTRACTION_PROMPT = """Extract the final numeric answer from the response below. Respond with ONLY the number — digits only, no words, no units, no currency symbols, no punctuation other than a decimal point or minus sign. If the response contains multiple numbers, extract the one that represents the final answer to the question. If no numeric answer can be determined, respond with exactly: NONE

Question:
{question}

Response:
{response}"""

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _load_rows(limit: int | None) -> list[dict]:
    ds = load_dataset(_REPO_ID, "main", split="test")
    rows = ds if limit is None else ds.select(range(min(limit, len(ds))))
    return list(rows)


def _parse_ground_truth(answer: str) -> float:
    m = _GROUND_TRUTH_RE.search(answer)
    if not m:
        raise ValueError(f"GSM8K answer had no parseable '#### <number>' final line: {answer!r}")
    return float(m.group(1).replace(",", ""))


def _extract_model_answer(question: str, response: str, judge_model: str) -> float | None:
    prompt = _EXTRACTION_PROMPT.format(question=question, response=response)
    extracted = call_gemini(judge_model, prompt).strip()
    if extracted.upper() == "NONE":
        return None
    cleaned = re.sub(r"[,$%\s]", "", extracted)
    m = _NUMBER_RE.search(cleaned)
    if not m:
        return None
    return float(m.group(0))


class Gsm8kEval(BaseEval):
    METRIC_DIRECTIONS = {"gsm8k_acc": "higher"}

    def run(self, model: ModelInterface, config: dict, mode: str) -> EvalResult:
        limit = _MODE_SAMPLE_SIZE.get(mode)  # None => full dataset
        judge_model = config.get("judge", {}).get("model", DEFAULT_JUDGE_MODEL)

        tts_cfg = config.get("tts", {})
        backend_name = tts_cfg.get("backend")
        backend_def = tts_cfg.get("_resolved_backend")
        if not backend_name or backend_def is None:
            raise ValueError(
                "knowledge.gsm8k requires a resolved tts.backend in config — "
                "set tts.backend in your config YAML (see configs/tts_backends.yaml) "
                "and load it via core.config.load_config(), not a bare yaml.safe_load()"
            )
        tts_backend = build_tts_backend(backend_name, backend_def)

        prior = config.get("_prior_result")
        progress: dict = dict(prior.metadata.get("_progress", {})) if prior is not None else {}
        progress.setdefault("total", 0)
        progress.setdefault("correct", 0)
        progress.setdefault("degenerate_silence", 0)
        errors: list[str] = list(prior.errors) if prior is not None else []
        transcript: list[dict] = list(prior.transcript) if prior is not None else []

        rows = _load_rows(limit)

        for i, row in enumerate(rows):
            if i < progress["total"]:
                continue  # already scored in a prior partial run

            question = row["question"]
            entry: dict = {"index": i, "question": question}
            correct = False
            try:
                ground_truth = _parse_ground_truth(row["answer"])
                audio_in = synthesize_cached(question, tts_backend, backend_name, backend_def, _CACHE_DIR)
                _, text_out, resp_metadata = model.respond(audio_in)
                model_answer = _extract_model_answer(question, text_out, judge_model)
                correct = model_answer is not None and model_answer == ground_truth

                entry.update({
                    "ground_truth": ground_truth,
                    "model_response": text_out,
                    "extracted_answer": model_answer,
                    "retrieval_context": resp_metadata.get("retrieval_context", ""),
                    "retrieval_text": resp_metadata.get("retrieval_text", ""),
                    "verdict": "correct" if correct else "incorrect",
                    # See MoshiRAGAdapter.respond()'s docstring — an empty
                    # response with no <ret> is a distinct failure mode from
                    # "model got it wrong", worth telling apart in scoring
                    # review rather than silently folding into verdict.
                    "degenerate_silence": resp_metadata.get("degenerate_silence", False),
                })
                if resp_metadata.get("degenerate_silence"):
                    progress["degenerate_silence"] += 1
            except Exception as exc:
                logger.warning("gsm8k[%d] failed: %s", i, exc)
                errors.append(f"[{i}]: {exc}")
                entry["error"] = str(exc)
                correct = False

            transcript.append(entry)
            progress["total"] += 1
            if correct:
                progress["correct"] += 1

        scores = {}
        if progress["total"]:
            scores["gsm8k_acc"] = progress["correct"] / progress["total"]

        metadata: dict = {
            "n": progress["total"],
            "judge_model": judge_model,
            "tts_backend": describe_tts_backend(backend_name, backend_def),
            "_progress": progress,
        }
        if progress["degenerate_silence"]:
            metadata["degenerate_silence_count"] = progress["degenerate_silence"]

        return EvalResult(
            eval_name="knowledge.gsm8k",
            scores=scores,
            metadata=metadata,
            errors=errors,
            completed=True,
            last_completed_index=progress["total"],
            transcript=transcript,
        )

    def format_results(self, result: EvalResult) -> list[str]:
        lines = []
        n = result.metadata.get("n", 0)
        if "gsm8k_acc" in result.scores:
            lines.append(f"  acc              : {result.scores['gsm8k_acc'] * 100:.1f}%   (n={n})")
        judge_model = result.metadata.get("judge_model")
        if judge_model:
            lines.append(f"  extractor model  : {judge_model}")
        tts_backend = result.metadata.get("tts_backend")
        if tts_backend:
            voice_part = f", voice: {tts_backend['voice']}" if "voice" in tts_backend else ""
            lines.append(f"  tts backend      : {tts_backend['name']} ({tts_backend.get('model', '?')}{voice_part})")
        return lines
