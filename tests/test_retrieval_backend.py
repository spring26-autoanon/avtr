from unittest.mock import MagicMock, patch
import pytest
from core.retrieval_backend import GeminiAPIBackend, LatencyGateError, NullBackend


def test_null_backend():
    text, latency = NullBackend().retrieve("some context")
    assert text == ""
    assert latency == 0.0


def _mock_client(return_text="reference text"):
    client = MagicMock()
    response = MagicMock()
    response.text = return_text
    client.models.generate_content.return_value = response
    return client


def test_gemini_backend_success():
    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=_mock_client())
    text, latency = backend.retrieve("what is the capital of France?")
    assert text == "reference text"
    assert latency >= 0


def test_gemini_backend_latency_gate():
    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=1, _client=_mock_client())
    # Simulate 2s wall time
    with patch("core.retrieval_backend.time.perf_counter", side_effect=[0.0, 2.0]):
        with pytest.raises(LatencyGateError) as exc_info:
            backend.retrieve("context")
    assert exc_info.value.latency_s == 2.0
    assert exc_info.value.gate_ms == 1


def test_gemini_backend_retry_on_transient_failure():
    client = MagicMock()
    response = MagicMock()
    response.text = "eventual success"
    client.models.generate_content.side_effect = [
        RuntimeError("transient"),
        RuntimeError("transient"),
        response,
    ]

    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=client)
    with patch("core.retrieval_backend.time.sleep"):
        text, _ = backend.retrieve("context")

    assert text == "eventual success"
    assert client.models.generate_content.call_count == 3


def test_gemini_backend_raises_after_max_retries():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("persistent error")

    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=client)
    with patch("core.retrieval_backend.time.sleep"):
        with pytest.raises(RuntimeError, match="persistent error"):
            backend.retrieve("context")

    assert client.models.generate_content.call_count == 3


def test_latency_gate_not_retried():
    """LatencyGateError on a successful call should not trigger retry."""
    client = _mock_client()
    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=1, _client=client)

    with patch("core.retrieval_backend.time.perf_counter", side_effect=[0.0, 2.0]):
        with pytest.raises(LatencyGateError):
            backend.retrieve("context")

    # Only one API call made — gate breach doesn't retry
    assert client.models.generate_content.call_count == 1


def test_gemini_backend_retries_on_empty_text():
    """A successful-but-empty response (known Gemini 3.x thinking-model gotcha) retries."""
    client = MagicMock()
    empty_response = MagicMock()
    empty_response.text = ""
    real_response = MagicMock()
    real_response.text = "eventual reference text"
    client.models.generate_content.side_effect = [empty_response, empty_response, real_response]

    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=client)
    with patch("core.retrieval_backend.time.sleep"):
        text, _ = backend.retrieve("context")

    assert text == "eventual reference text"
    assert client.models.generate_content.call_count == 3


def test_gemini_backend_raises_after_max_retries_all_empty():
    client = MagicMock()
    empty_response = MagicMock()
    empty_response.text = ""
    client.models.generate_content.return_value = empty_response

    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=client)
    with patch("core.retrieval_backend.time.sleep"):
        with pytest.raises(RuntimeError, match="Gemini returned no text content"):
            backend.retrieve("context")

    assert client.models.generate_content.call_count == 3
