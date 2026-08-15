from unittest.mock import MagicMock, patch

import pytest

from core.retrieval_backend import (
    BACKEND_REGISTRY,
    GeminiAPIBackend,
    LatencyGateError,
    NullBackend,
    build_backend_from_retrieval_config,
    build_retrieval_backend,
    describe_backend,
    format_context,
)


def test_null_backend():
    text, latency = NullBackend().retrieve("some context")
    assert text == ""
    assert latency == 0.0


def test_null_backend_ignores_history():
    text, latency = NullBackend().retrieve("some context", history=[(1, "some ref")])
    assert text == ""
    assert latency == 0.0


def _mock_client(return_text="reference text"):
    client = MagicMock()
    response = MagicMock()
    response.text = return_text
    client.models.generate_content.return_value = response
    return client


# Realistic conversation context — matches TurnManager.get_context()'s real
# shape ("user: "/"moshi: " prefixed lines), so format_context() sees at
# least one real turn and doesn't short-circuit on num_turns == 0.
_CTX = "user: what is the capital of France?"


def test_gemini_backend_success():
    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=_mock_client())
    text, latency = backend.retrieve(_CTX)
    assert text == "reference text"
    assert latency >= 0


def test_gemini_backend_latency_gate():
    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=1, _client=_mock_client())
    # Simulate 2s wall time
    with patch("core.retrieval_backend.time.perf_counter", side_effect=[0.0, 2.0]):
        with pytest.raises(LatencyGateError) as exc_info:
            backend.retrieve(_CTX)
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
        text, _ = backend.retrieve(_CTX)

    assert text == "eventual success"
    assert client.models.generate_content.call_count == 3


def test_gemini_backend_raises_after_max_retries():
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("persistent error")

    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=client)
    with patch("core.retrieval_backend.time.sleep"):
        with pytest.raises(RuntimeError, match="persistent error"):
            backend.retrieve(_CTX)

    assert client.models.generate_content.call_count == 3


def test_latency_gate_not_retried():
    """LatencyGateError on a successful call should not trigger retry."""
    client = _mock_client()
    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=1, _client=client)

    with patch("core.retrieval_backend.time.perf_counter", side_effect=[0.0, 2.0]):
        with pytest.raises(LatencyGateError):
            backend.retrieve(_CTX)

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
        text, _ = backend.retrieve(_CTX)

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
            backend.retrieve(_CTX)

    assert client.models.generate_content.call_count == 3


def test_gemini_backend_skips_call_on_empty_context():
    """No recognizable "user:"/"moshi:" turn in the context (e.g. a lone
    greeting was dropped, or a genuinely empty context) -> num_turns == 0 ->
    skip the Gemini call entirely, same short-circuit moshi's own
    generate_reference_text() does."""
    client = _mock_client()
    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=client)

    text, latency = backend.retrieve("no recognizable turn prefix here")

    assert text == ""
    assert latency == 0.0
    client.models.generate_content.assert_not_called()


def test_gemini_backend_uses_prompt_template_file(tmp_path):
    template = tmp_path / "custom.txt"
    template.write_text("CUSTOM PROMPT >>> {context} <<<")
    client = _mock_client()
    backend = GeminiAPIBackend(
        model="gemini-3.5-flash", latency_gate_ms=5000, prompt_template=str(template), _client=client
    )

    backend.retrieve(_CTX)

    _, kwargs = client.models.generate_content.call_args
    assert kwargs["contents"].startswith("CUSTOM PROMPT >>>")
    assert "what is the capital of France?" in kwargs["contents"]


def test_gemini_backend_default_prompt_template_matches_repo_file():
    client = _mock_client()
    backend = GeminiAPIBackend(model="gemini-3.5-flash", latency_gate_ms=5000, _client=client)

    backend.retrieve(_CTX)

    _, kwargs = client.models.generate_content.call_args
    assert "Context:" in kwargs["contents"]


def test_gemini_backend_passes_context_formatting_to_format_context():
    client = _mock_client()
    backend = GeminiAPIBackend(
        model="gemini-3.5-flash",
        latency_gate_ms=5000,
        _client=client,
        context_formatting={"drop_leading_incomplete_turn": True},
    )

    with patch("core.retrieval_backend.format_context", wraps=format_context) as spy:
        backend.retrieve(_CTX)

    spy.assert_called_once()
    assert spy.call_args.kwargs["drop_leading_incomplete_turn"] is True


# ── format_context ────────────────────────────────────────────────────────


def test_format_context_defaults_pass_through_unchanged():
    """With every toggle at its default (off), the context is returned
    completely unchanged — no Human:/moshi: relabeling — matching today's
    plain {context} substitution and pairing with
    configs/prompts/retrieval_reference_simple.txt."""
    ctx = "user: hello\nmoshi: hi there\nuser: how are you"
    formatted, num_turns = format_context(ctx, None)
    assert formatted == ctx
    assert num_turns == 3


def test_format_context_counts_turns_with_no_prefix_as_zero():
    formatted, num_turns = format_context("just some raw text", None)
    assert num_turns == 0
    assert formatted == "just some raw text"


def test_format_context_drop_trailing_incomplete_turn():
    ctx = "user: what did the report say\nmoshi: partial answer in progress"
    formatted, num_turns = format_context(ctx, None, drop_trailing_incomplete_turn=True)
    assert "partial answer" not in formatted
    assert "Human: what did the report say" in formatted
    assert num_turns == 1


def test_format_context_drop_trailing_only_when_last_turn_is_model():
    """The trailing drop only applies when the last turn is the model's —
    a context ending on a user turn is left alone."""
    ctx = "user: hello\nmoshi: hi\nuser: another question"
    formatted, num_turns = format_context(ctx, None, drop_trailing_incomplete_turn=True)
    assert num_turns == 3
    assert "another question" in formatted


def test_format_context_drop_leading_incomplete_turn():
    ctx = "moshi: hello, how can I help?\nuser: what is the capital of France?"
    formatted, num_turns = format_context(ctx, None, drop_leading_incomplete_turn=True)
    assert "hello, how can I help" not in formatted
    assert num_turns == 1
    assert "Human: what is the capital of France?" in formatted


def test_format_context_role_labels_default_to_human_moshi():
    ctx = "user: hi\nmoshi: hello"
    formatted, _ = format_context(ctx, None, drop_leading_incomplete_turn=True)
    # drop_leading has nothing to drop here (first turn is "user"), so both
    # turns remain and get relabeled — confirms the Human:/moshi: default.
    assert "Human: hi" in formatted
    assert "moshi: hello" in formatted


def test_format_context_custom_role_labels():
    ctx = "user: hi\nmoshi: hello"
    formatted, _ = format_context(
        ctx, None, drop_leading_incomplete_turn=True, role_labels={"user": "User", "model": "Assistant"}
    )
    assert "User: hi" in formatted
    assert "Assistant: hello" in formatted


def test_format_context_threads_reference_history_at_the_right_position():
    ctx = "user: question one\nmoshi: answer one\nuser: question two"
    history = [(2, "prior reference text")]
    formatted, num_turns = format_context(ctx, history, thread_reference_history=True)
    lines = formatted.split("\n")
    assert lines[2] == "Reference: prior reference text"
    assert num_turns == 3


def test_format_context_thread_reference_history_false_ignores_history():
    ctx = "user: q1\nmoshi: a1"
    # thread_reference_history is False but drop_leading is True, so we
    # still take the "full rebuild" path — history must be ignored either way.
    formatted, _ = format_context(ctx, [(1, "should not appear")], drop_leading_incomplete_turn=True)
    assert "should not appear" not in formatted


def test_format_context_history_max_entries_drops_oldest_first():
    ctx = "user: q1\nmoshi: a1\nuser: q2\nmoshi: a2\nuser: q3"
    history = [(1, "oldest"), (3, "newest")]
    formatted, _ = format_context(ctx, history, thread_reference_history=True, history_max_entries=1)
    assert "oldest" not in formatted
    assert "newest" in formatted


def test_format_context_empty_history_is_a_no_op():
    ctx = "user: q1\nmoshi: a1"
    formatted, num_turns = format_context(ctx, [], thread_reference_history=True, drop_leading_incomplete_turn=True)
    assert "Reference:" not in formatted
    assert num_turns == 2


# ── describe_backend ─────────────────────────────────────────────────────


def test_describe_backend_includes_model_and_type():
    result = describe_backend("gemini_api", {"type": "gemini_api", "model": "gemini-3.5-flash"})
    assert result == {"name": "gemini_api", "type": "gemini_api", "model": "gemini-3.5-flash"}


def test_describe_backend_omits_missing_fields():
    result = describe_backend("null", {"type": "null_backend"})
    assert result == {"name": "null", "type": "null_backend"}
    assert "model" not in result
    assert "base_url" not in result


def test_describe_backend_never_includes_api_key_env():
    result = describe_backend(
        "gemini_api", {"type": "gemini_api", "model": "gemini-3.5-flash", "api_key_env": "GEMINI_API_KEY"}
    )
    assert "api_key_env" not in result
    assert "GEMINI_API_KEY" not in str(result)


def test_describe_backend_includes_base_url_when_present():
    result = describe_backend("local_llm", {"type": "openai_compatible", "base_url": "http://localhost:8080/v1"})
    assert result["base_url"] == "http://localhost:8080/v1"


# ── BACKEND_REGISTRY / build_retrieval_backend ──────────────────────────────


def test_backend_registry_has_known_types():
    assert BACKEND_REGISTRY["gemini_api"] is GeminiAPIBackend
    assert BACKEND_REGISTRY["null_backend"] is NullBackend


def test_build_retrieval_backend_gemini_api():
    backend = build_retrieval_backend("gemini_api", {"type": "gemini_api", "model": "gemini-3.5-flash"}, 5000)
    assert isinstance(backend, GeminiAPIBackend)
    assert backend.model == "gemini-3.5-flash"
    assert backend.latency_gate_ms == 5000


def test_build_retrieval_backend_null_backend():
    backend = build_retrieval_backend("null", {"type": "null_backend"}, None)
    assert isinstance(backend, NullBackend)


def test_build_retrieval_backend_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown retrieval backend type"):
        build_retrieval_backend("mystery", {"type": "not_a_real_type"}, None)


def test_build_retrieval_backend_gemini_api_passes_prompt_template_and_context_formatting(tmp_path):
    template = tmp_path / "custom.txt"
    template.write_text("{context}")
    backend = build_retrieval_backend(
        "gemini_api",
        {
            "type": "gemini_api",
            "model": "gemini-3.5-flash",
            "prompt_template": str(template),
            "context_formatting": {"thread_reference_history": True},
        },
        5000,
    )
    assert backend._prompt_template == "{context}"
    assert backend.context_formatting == {"thread_reference_history": True}


# ── build_backend_from_retrieval_config ─────────────────────────────────────


def test_build_backend_from_retrieval_config_disabled_returns_null_backend():
    backend = build_backend_from_retrieval_config({"enabled": False})
    assert isinstance(backend, NullBackend)


def test_build_backend_from_retrieval_config_enabled_uses_resolved_backend():
    backend = build_backend_from_retrieval_config(
        {
            "enabled": True,
            "backend": "gemini_api",
            "latency_gate_ms": 5000,
            "_resolved_backend": {"type": "gemini_api", "model": "gemini-3.5-flash"},
        }
    )
    assert isinstance(backend, GeminiAPIBackend)
    assert backend.model == "gemini-3.5-flash"


def test_build_backend_from_retrieval_config_enabled_without_resolution_raises():
    with pytest.raises(ValueError, match="did not resolve"):
        build_backend_from_retrieval_config({"enabled": True, "backend": "gemini_api", "latency_gate_ms": 5000})
