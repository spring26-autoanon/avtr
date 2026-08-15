from unittest.mock import MagicMock, patch

import pytest

from core.llm_judge import call_gemini, generate_gemini_text


def _mock_client(return_text="the score is [Correct]"):
    client = MagicMock()
    response = MagicMock()
    response.text = return_text
    client.models.generate_content.return_value = response
    return client


def test_generate_gemini_text_returns_response_text():
    client = _mock_client("hello")
    text = generate_gemini_text(client, "gemini-3.5-flash", "prompt")
    assert text == "hello"
    client.models.generate_content.assert_called_once()
    _, kwargs = client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-3.5-flash"
    assert kwargs["contents"] == "prompt"


def test_call_gemini_success():
    client = _mock_client("verdict text")
    text = call_gemini("gemini-3.5-flash", "prompt", _client=client)
    assert text == "verdict text"
    assert client.models.generate_content.call_count == 1


def test_call_gemini_retries_on_transient_failure():
    client = MagicMock()
    response = MagicMock()
    response.text = "eventual success"
    client.models.generate_content.side_effect = [
        RuntimeError("transient"),
        RuntimeError("transient"),
        response,
    ]
    with patch("core.llm_judge.time.sleep"):
        text = call_gemini("gemini-3.5-flash", "prompt", _client=client)
    assert text == "eventual success"
    assert client.models.generate_content.call_count == 3


def test_call_gemini_raises_after_max_retries():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("persistent error")
    with patch("core.llm_judge.time.sleep"):
        with pytest.raises(RuntimeError, match="persistent error"):
            call_gemini("gemini-3.5-flash", "prompt", _client=client)
    assert client.models.generate_content.call_count == 3


def test_call_gemini_retries_on_empty_text():
    client = MagicMock()
    empty_response = MagicMock()
    empty_response.text = ""
    real_response = MagicMock()
    real_response.text = "eventual text"
    client.models.generate_content.side_effect = [empty_response, empty_response, real_response]
    with patch("core.llm_judge.time.sleep"):
        text = call_gemini("gemini-3.5-flash", "prompt", _client=client)
    assert text == "eventual text"
    assert client.models.generate_content.call_count == 3


def test_call_gemini_raises_after_max_retries_all_empty():
    client = MagicMock()
    empty_response = MagicMock()
    empty_response.text = ""
    client.models.generate_content.return_value = empty_response
    with patch("core.llm_judge.time.sleep"):
        with pytest.raises(RuntimeError, match="Gemini returned no text content"):
            call_gemini("gemini-3.5-flash", "prompt", _client=client)
    assert client.models.generate_content.call_count == 3
