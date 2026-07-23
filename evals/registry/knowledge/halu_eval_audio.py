import itertools
import logging
import re

from datasets import load_dataset
from datasets.features import Audio

from core.llm_judge import call_gemini
from core.model_interface import ModelInterface
from evals.runner import BaseEval, DEFAULT_JUDGE_MODEL, EvalResult

logger = logging.getLogger(__name__)

_REPO_ID = "kyutai/HaluEvalAudio_1000"

_MODE_SAMPLE_SIZE = {"tiny": 1, "smoke": 5, "sample": 100}  # "full" => all 1000 rows

# Same judge prompt as knowledge.open_audio_bench — verbatim from the
# MoshiRAG paper appendix, Table 16. Used twice per question here: once
# against the model's final response (resp_acc) and once against whatever
# RetrievalBackend.retrieve() actually returned during that turn (ref_acc) —
# "the ref. columns in Table 1 report the accuracy of the retrieved
# reference information provided by the back ends" (paper, Section 5.1).
_JUDGE_PROMPT = """## Background
You are a professional QA evaluation expert. You need to assess whether the model's answer is correct based on the standard answer.

## Scoring Criteria
Correct: The answer matches or is equivalent to the standard answer, or contains the same core concept.
Incorrect: The answer is wrong or irrelevant to the question

## Evaluation Guidelines
1. The expression of answers can be flexible, not requiring exact matches. For example:
- Numbers can be expressed in either Arabic numerals or words
- Differences in punctuation or simple spelling mistakes can be ignored
2. Focus on whether the core meaning of the answer is correct

## Output Format
Provide the reasoning for your score, then generate the result in "[]" format and make sure it contains "the score is [Correct]" or "the score is [Incorrect]", for example:
The answer is correct and equivalent to the standard answer, the score is [Correct]
or
The answer is incorrect and does not match the standard answer, the score is [Incorrect]

## Question:
{question}

## Standard Answer:
{valid_answers}

## Model's Answer:
{answer}"""

_VERDICT_RE = re.compile(r"the score is \[(correct|incorrect)\]", re.IGNORECASE)


def _judge(question: str, ground_truth_answer: str, candidate: str, judge_model: str) -> bool:
    formatted_answers = f'["{ground_truth_answer}"]'
    prompt = _JUDGE_PROMPT.format(question=question, valid_answers=formatted_answers, answer=candidate)
    verdict_text = call_gemini(judge_model, prompt)
    m = _VERDICT_RE.search(verdict_text)
    if not m:
        raise ValueError(f"Judge response had no parseable verdict: {verdict_text!r}")
    return m.group(1).lower() == "correct"


def _load_rows(limit: int | None) -> list[dict]:
    # Parquet-backed (not per-file + CSV like OpenAudioBench) — stream
    # rather than a full load_dataset() so smoke/sample mode doesn't pull
    # the whole ~670MB dataset. decode=False keeps the audio field as raw
    # embedded WAV bytes (what respond() wants directly) instead of forcing
    # a decode to a float array through the optional `soundfile` dependency
    # this project doesn't otherwise need.
    ds = load_dataset(_REPO_ID, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    rows = ds if limit is None else itertools.islice(ds, limit)
    return list(rows)


class HaluEvalAudioEval(BaseEval):
    METRIC_DIRECTIONS = {"ref_acc": "higher", "resp_acc": "higher"}

    def run(self, model: ModelInterface, config: dict, mode: str) -> EvalResult:
        limit = _MODE_SAMPLE_SIZE.get(mode)  # None => full dataset
        judge_model = config.get("judge", {}).get("model", DEFAULT_JUDGE_MODEL)

        prior = config.get("_prior_result")
        progress = (
            dict(prior.metadata.get("_progress", {}))
            if prior is not None
            else {}
        )
        progress.setdefault("total", 0)
        progress.setdefault("correct_ref", 0)
        progress.setdefault("correct_resp", 0)
        errors: list[str] = list(prior.errors) if prior is not None else []
        transcript: list[dict] = list(prior.transcript) if prior is not None else []

        rows = _load_rows(limit)

        for i, row in enumerate(rows):
            if i < progress["total"]:
                continue  # already scored in a prior partial run

            question = row["text"]
            answer = row["answer"]
            entry: dict = {
                "index": i,
                "question": question,
                "ground_truth_answer": answer,
                "ground_truth_knowledge": row.get("knowledge"),
            }
            try:
                audio_bytes = row["audio"]["bytes"]
                _, text_out, resp_metadata = model.respond(audio_bytes)
                retrieval_text = resp_metadata.get("retrieval_text", "")

                resp_correct = _judge(question, answer, text_out, judge_model)
                # Empty retrieval text can never be judged "Correct" — skip
                # the API call rather than spend one confirming the obvious
                # (always true under NullBackend / retrieval-disabled runs).
                ref_correct = bool(retrieval_text.strip()) and _judge(question, answer, retrieval_text, judge_model)

                entry.update({
                    "model_response": text_out,
                    "retrieval_context": resp_metadata.get("retrieval_context", ""),
                    "retrieval_text": retrieval_text,
                    "resp_verdict": "correct" if resp_correct else "incorrect",
                    "ref_verdict": "correct" if ref_correct else "incorrect",
                })
            except Exception as exc:
                logger.warning("halu_eval_audio[%d] failed: %s", i, exc)
                errors.append(f"[{i}]: {exc}")
                entry["error"] = str(exc)
                resp_correct = False
                ref_correct = False

            transcript.append(entry)
            progress["total"] += 1
            if resp_correct:
                progress["correct_resp"] += 1
            if ref_correct:
                progress["correct_ref"] += 1

        scores = {}
        if progress["total"]:
            scores["resp_acc"] = progress["correct_resp"] / progress["total"]
            scores["ref_acc"] = progress["correct_ref"] / progress["total"]

        metadata = {
            "n": progress["total"],
            "judge_model": judge_model,
            "_progress": progress,
        }

        return EvalResult(
            eval_name="knowledge.halu_eval_audio",
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
        if "ref_acc" in result.scores:
            lines.append(f"  ref acc      : {result.scores['ref_acc'] * 100:.1f}%   (n={n})")
        if "resp_acc" in result.scores:
            lines.append(f"  resp acc     : {result.scores['resp_acc'] * 100:.1f}%   (n={n})")
        judge_model = result.metadata.get("judge_model")
        if judge_model:
            lines.append(f"  judge model  : {judge_model}")
        return lines
