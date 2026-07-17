import logging
import os
import time

logger = logging.getLogger(__name__)


def call_gemini(model: str, prompt: str, api_key: str | None = None, _client=None) -> str:
    """
    Call Gemini API for a one-shot text task (LLM judge scoring, keyword
    extraction) and return the response text.

    Shares the retry-on-failure / retry-on-empty-response reliability pattern
    of core/retrieval_backend.py's GeminiAPIBackend.retrieve() — Gemini 3.x's
    mandatory "thinking" can silently consume the whole output budget and
    return no visible text, non-deterministically per call.
    """
    from google import genai
    from google.genai import types

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
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="low"),
                    max_output_tokens=1024,
                ),
            )
            text = response.text
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
