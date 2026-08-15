import logging
import os
import time

logger = logging.getLogger(__name__)


def generate_gemini_text(client, model: str, prompt: str) -> str:
    """
    One raw Gemini generate_content call, returning response.text (which may
    be empty — callers decide whether that warrants a retry). Raises whatever
    the SDK call itself raises.

    Shared by call_gemini() below (LLM judge / keyword extraction) and
    core/retrieval_backend.py's GeminiAPIBackend.retrieve() — both need the
    same thinking_level="low"/max_output_tokens=1024 workaround for a real
    Gemini 3.x quirk (mandatory "thinking" can silently consume the whole
    output budget and return no visible text), previously duplicated in both
    places independently.
    """
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            # This is a fast fact-lookup/scoring task, not open-ended
            # reasoning — request minimal thinking effort. Gemini 3.x models
            # (unlike the 2.5 family) can't disable thinking entirely, only
            # reduce it; default is "medium".
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            # Generous headroom for thinking + the actual answer. Known,
            # tracked failure mode: Gemini 3.x's mandatory thinking can
            # consume the entire output budget and return zero visible text
            # if this isn't set high enough —
            # https://github.com/googleapis/python-genai/issues/2121
            max_output_tokens=1024,
        ),
    )
    # Read .text inside the caller's try: some SDK versions raise here
    # (rather than just returning empty) when a response has no usable text
    # parts.
    return response.text


def call_gemini(model: str, prompt: str, api_key: str | None = None, _client=None) -> str:
    """
    Call Gemini API for a one-shot text task (LLM judge scoring, keyword
    extraction) and return the response text.

    Retries failed/empty-response calls up to 3 times with exponential
    backoff — Gemini 3.x's mandatory "thinking" can silently consume the
    whole output budget and return no visible text, non-deterministically
    per call.
    """
    from google import genai

    client = _client
    if client is None:
        client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])

    last_exc: Exception | None = None
    for attempt in range(3):
        if attempt:
            wait = 2 ** (attempt - 1)  # 1s then 2s
            logger.warning("Gemini call retry %d/3, waiting %ds", attempt + 1, wait)
            time.sleep(wait)

        try:
            text = generate_gemini_text(client, model, prompt)
        except Exception as exc:
            logger.warning("Gemini API call failed: %s", exc)
            last_exc = exc
            continue

        if not text:
            logger.warning("Gemini response had no visible text content — retrying")
            last_exc = RuntimeError("Gemini returned no text content")
            continue

        return text

    raise last_exc  # type: ignore[misc]
