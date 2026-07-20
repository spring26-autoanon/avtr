import csv
import logging
import re

from huggingface_hub import hf_hub_download

from core.llm_judge import call_gemini
from core.model_interface import ModelInterface
from evals.runner import BaseEval, EvalResult

logger = logging.getLogger(__name__)

# See specs/moshirag-evals-requirements-v2.md for why this is baichuan-inc/*
# rather than the originally-assumed AudioLLMs/OpenAudioBench (doesn't exist).
_REPO_ID = "baichuan-inc/OpenAudioBench"
_JUDGE_MODEL = "gemini-3.5-flash"

_MODE_SAMPLE_SIZE = {"tiny": 1, "smoke": 5, "sample": 100}  # "full" => all rows in the CSV

# subset_key -> column layout, verified against the actual CSVs on HF
_SUBSETS = {
    "triviaqa": {
        "csv": "eval_datas/trivia_qa/trivia_qa.csv",
        "audio_dir": "eval_datas/trivia_qa/audios",
        "question_col": "question",
        "answer_col": "answer_normalized_value",
        "aliases_col": "answer_normalized_aliases",
        "aliases_delim": ",",
    },
    "webq": {
        "csv": "eval_datas/web_questions/web_questions.csv",
        "audio_dir": "eval_datas/web_questions/audios",
        "question_col": "question",
        "answer_col": "answers",
        "aliases_col": None,
        "aliases_delim": ";",  # "answers" itself holds ;-separated alternatives
    },
    "llamaq": {
        "csv": "eval_datas/llama_questions/llama_questions.csv",
        "audio_dir": "eval_datas/llama_questions/audios",
        "question_col": "Questions",
        "answer_col": "Answer",
        "aliases_col": None,
        "aliases_delim": None,
    },
}

_LABELS = {"triviaqa": "TriviaQA", "webq": "WebQ", "llamaq": "LlamaQ"}

# Verbatim from the MoshiRAG paper appendix, Table 16 ("The prompt for
# evaluating the correctness ... This prompt is based on the prompts used in
# the OpenAudioBench (Li et al., 2025)") — do not paraphrase.
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


def _valid_answers(subset_cfg: dict, row: dict) -> list[str]:
    answers = [row[subset_cfg["answer_col"]].strip()]
    if subset_cfg["aliases_col"]:
        aliases = row[subset_cfg["aliases_col"]]
        if aliases:
            answers += [a.strip() for a in aliases.split(subset_cfg["aliases_delim"]) if a.strip()]
    elif subset_cfg["aliases_delim"] and subset_cfg["aliases_delim"] in answers[0]:
        answers = [a.strip() for a in answers[0].split(subset_cfg["aliases_delim"]) if a.strip()]
    # de-dupe, preserve order
    seen: set[str] = set()
    out = []
    for a in answers:
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def _load_rows(subset_cfg: dict, limit: int | None) -> list[dict]:
    csv_path = hf_hub_download(repo_id=_REPO_ID, repo_type="dataset", filename=subset_cfg["csv"])
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows if limit is None else rows[:limit]


def _load_audio_bytes(subset_cfg: dict, row: dict) -> bytes:
    audio_path = hf_hub_download(
        repo_id=_REPO_ID,
        repo_type="dataset",
        filename=f"{subset_cfg['audio_dir']}/{row['audio_filename']}",
    )
    with open(audio_path, "rb") as f:
        return f.read()


def _judge(question: str, valid_answers: list[str], model_answer: str) -> bool:
    formatted_answers = "[" + ", ".join(f'"{a}"' for a in valid_answers) + "]"
    prompt = _JUDGE_PROMPT.format(question=question, valid_answers=formatted_answers, answer=model_answer)
    verdict_text = call_gemini(_JUDGE_MODEL, prompt)
    m = _VERDICT_RE.search(verdict_text)
    if not m:
        raise ValueError(f"Judge response had no parseable verdict: {verdict_text!r}")
    return m.group(1).lower() == "correct"


class OpenAudioBenchEval(BaseEval):
    METRIC_DIRECTIONS = {
        "triviaqa_acc": "higher",
        "webq_acc": "higher",
        "llamaq_acc": "higher",
    }

    def run(self, model: ModelInterface, config: dict, mode: str) -> EvalResult:
        limit = _MODE_SAMPLE_SIZE.get(mode)  # None => full dataset

        prior = config.get("_prior_result")
        progress: dict[str, dict] = {}
        if prior is not None:
            progress = {k: dict(v) for k, v in prior.metadata.get("_progress", {}).items()}
        errors: list[str] = list(prior.errors) if prior is not None else []
        transcript: list[dict] = list(prior.transcript) if prior is not None else []

        for subset_key, subset_cfg in _SUBSETS.items():
            rows = _load_rows(subset_cfg, limit)
            state = progress.setdefault(subset_key, {"correct": 0, "total": 0})

            for i, row in enumerate(rows):
                if i < state["total"]:
                    continue  # already scored in a prior partial run

                question = row[subset_cfg["question_col"]]
                entry: dict = {
                    "subset": subset_key,
                    "index": i,
                    "audio_filename": row.get("audio_filename"),
                    "question": question,
                }
                try:
                    audio_in = _load_audio_bytes(subset_cfg, row)
                    _, text_out, resp_metadata = model.respond(audio_in)
                    valid_answers = _valid_answers(subset_cfg, row)
                    correct = _judge(question, valid_answers, text_out)
                    entry.update({
                        "valid_answers": valid_answers,
                        "model_response": text_out,
                        "retrieval_context": resp_metadata.get("retrieval_context", ""),
                        "retrieval_text": resp_metadata.get("retrieval_text", ""),
                        "judge_verdict": "correct" if correct else "incorrect",
                    })
                except Exception as exc:
                    logger.warning("open_audio_bench[%s][%d] failed: %s", subset_key, i, exc)
                    errors.append(f"{subset_key}[{i}]: {exc}")
                    entry["error"] = str(exc)
                    correct = False

                transcript.append(entry)
                state["total"] += 1
                if correct:
                    state["correct"] += 1

        scores = {}
        metadata: dict = {"judge_model": _JUDGE_MODEL, "_progress": progress}
        for subset_key in _SUBSETS:
            state = progress[subset_key]
            if state["total"]:
                scores[f"{subset_key}_acc"] = state["correct"] / state["total"]
            metadata[f"n_{subset_key}"] = state["total"]

        last_completed_index = sum(s["total"] for s in progress.values())

        return EvalResult(
            eval_name="knowledge.open_audio_bench",
            scores=scores,
            metadata=metadata,
            errors=errors,
            completed=True,
            last_completed_index=last_completed_index,
            transcript=transcript,
        )

    def format_results(self, result: EvalResult) -> list[str]:
        lines = []
        for subset_key, label in _LABELS.items():
            acc = result.scores.get(f"{subset_key}_acc")
            if acc is None:
                continue
            n = result.metadata.get(f"n_{subset_key}", 0)
            lines.append(f"  {label:<12} acc: {acc * 100:.1f}%   (n={n})")
        judge_model = result.metadata.get("judge_model")
        if judge_model:
            lines.append(f"  judge model  : {judge_model}")
        return lines
