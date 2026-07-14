import abc
import logging
import os
import time

logger = logging.getLogger(__name__)

_RETRIEVE_PROMPT = (
    "You are a knowledge retrieval system. Based on the following conversation "
    "context, provide a concise factual reference passage that would help answer "
    "the implied question. Respond with only the reference text, no preamble.\n\n"
    "Context:\n{context}"
)


class LatencyGateError(Exception):
    def __init__(self, latency_s: float, gate_ms: int):
        self.latency_s = latency_s
        self.gate_ms = gate_ms
        super().__init__(
            f"Retrieval latency {latency_s * 1000:.0f}ms exceeds gate {gate_ms}ms"
        )


class RetrievalBackend(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, context: str) -> tuple[str, float]:
        """Return (reference_text, latency_seconds)."""


class NullBackend(RetrievalBackend):
    """No-op backend for Config A (no retrieval)."""

    def retrieve(self, context: str) -> tuple[str, float]:
        return ("", 0.0)


class GeminiAPIBackend(RetrievalBackend):
    def __init__(
        self,
        model: str,
        latency_gate_ms: int,
        api_key: str | None = None,
        _client=None,  # injectable for tests
    ):
        self.model = model
        self.latency_gate_ms = latency_gate_ms
        if _client is not None:
            self._client = _client
        else:
            from google import genai
            self._client = genai.Client(
                api_key=api_key or os.environ["GEMINI_API_KEY"]
            )

    def retrieve(self, context: str) -> tuple[str, float]:
        prompt = _RETRIEVE_PROMPT.format(context=context)
        last_exc: Exception | None = None

        for attempt in range(3):
            if attempt:
                wait = 2 ** (attempt - 1)  # 1s then 2s
                logger.warning("Gemini retrieve retry %d/3, waiting %ds", attempt + 1, wait)
                time.sleep(wait)

            t0 = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
            except Exception as exc:
                latency = time.perf_counter() - t0
                logger.warning("Gemini API call failed after %.3fs: %s", latency, exc)
                last_exc = exc
                continue

            latency = time.perf_counter() - t0
            logger.debug("Gemini API call succeeded in %.3fs", latency)

            if latency * 1000 > self.latency_gate_ms:
                raise LatencyGateError(latency, self.latency_gate_ms)

            return response.text, latency

        raise last_exc  # type: ignore[misc]
