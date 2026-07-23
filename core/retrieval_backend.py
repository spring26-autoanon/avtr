import abc
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_TEMPLATE = Path(__file__).parent.parent / "configs" / "prompts" / "retrieval_reference_simple.txt"

# Matches moshi-rag's own TurnManager.ROLE_PREFIX_MAPPING label choice
# ("user: "/"moshi: ") by default — see format_context()'s docstring.
_DEFAULT_ROLE_LABELS = {"user": "Human", "model": "moshi"}


class LatencyGateError(Exception):
    def __init__(self, latency_s: float, gate_ms: int):
        self.latency_s = latency_s
        self.gate_ms = gate_ms
        super().__init__(
            f"Retrieval latency {latency_s * 1000:.0f}ms exceeds gate {gate_ms}ms"
        )


class RetrievalBackend(abc.ABC):
    @abc.abstractmethod
    def retrieve(self, context: str, history: list[tuple[int, str]] | None = None) -> tuple[str, float]:
        """Return (reference_text, latency_seconds)."""

    @classmethod
    def from_config(cls, backend_def: dict, latency_gate_ms: int | None) -> "RetrievalBackend":
        """
        Construct an instance from a resolved configs/retrieval_backends.yaml
        entry. Default: no-arg construction — override for backends that
        need fields out of backend_def (see GeminiAPIBackend.from_config).
        """
        return cls()


class NullBackend(RetrievalBackend):
    """No-op backend for Config A (no retrieval)."""

    def retrieve(self, context: str, history: list[tuple[int, str]] | None = None) -> tuple[str, float]:
        return ("", 0.0)


def _parse_turns(context: str) -> list[tuple[str, str]]:
    """
    Splits a moshi-rag-shaped conversation context (one turn per line, each
    prefixed "user: "/"moshi: " — the exact shape TurnManager.get_context()
    builds, confirmed against real moshi-rag source: a single newline is
    inserted only at speaker-switch boundaries) into (role, text) pairs.
    role is "user" or "model", matching RetrievalProfile-style naming
    elsewhere in this codebase.

    Lines that don't start with either prefix are silently skipped (not
    conversation turns — e.g. a blank line).
    """
    turns: list[tuple[str, str]] = []
    for line in context.split("\n"):
        if line.startswith("user:"):
            turns.append(("user", line[len("user:"):].strip()))
        elif line.startswith("moshi:"):
            turns.append(("model", line[len("moshi:"):].strip()))
    return turns


def format_context(
    context: str,
    history: list[tuple[int, str]] | None,
    *,
    drop_leading_incomplete_turn: bool = False,
    drop_trailing_incomplete_turn: bool = False,
    thread_reference_history: bool = False,
    history_max_entries: int | None = None,
    role_labels: dict[str, str] | None = None,
) -> tuple[str, int]:
    """
    Reproduces the config-gated subset of moshi-rag's own
    LLMReferenceGenerator.process_reference_text() (turn-trimming,
    reference-history threading) for use by any RetrievalBackend — see
    specs/moshirag-evals-requirements.md's "Context formatting" section for
    the full design rationale.

    Returns (formatted_context, num_turns). num_turns is the post-drop turn
    count (mirrors process_reference_text()'s identical return shape) —
    needed by callers doing reference-history bookkeeping, and by
    GeminiAPIBackend.retrieve() to skip the LLM call entirely on an empty
    context (the same short-circuit moshi's own generate_reference_text()
    does).

    Pure and stateless — no I/O, safe to call more than once with the same
    inputs for the identical result (used deliberately by
    scripts/instrumented_server.py's get_reference_text patch, which calls
    this a second, redundant time purely for its own history bookkeeping —
    see that module for why recomputing is fine here).

    With every one of drop_leading_incomplete_turn/drop_trailing_incomplete_turn/
    thread_reference_history at its default (False), the context is returned
    completely unchanged — no Human:/moshi: relabeling at all — matching
    today's plain-{context} substitution and pairing with
    configs/prompts/retrieval_reference_simple.txt. Turning any one of them
    on switches to a fully rebuilt, labeled transcript (role_labels below),
    pairing with configs/prompts/retrieval_reference_moshi_style.txt.
    """
    labels = role_labels or _DEFAULT_ROLE_LABELS
    turns = _parse_turns(context)

    if drop_trailing_incomplete_turn and turns and turns[-1][0] == "model":
        turns = turns[:-1]
    if drop_leading_incomplete_turn and turns and turns[0][0] == "model":
        turns = turns[1:]

    num_turns = len(turns)

    if not (drop_leading_incomplete_turn or drop_trailing_incomplete_turn or thread_reference_history):
        return context, num_turns

    if thread_reference_history and history:
        entries = list(history)
        if history_max_entries is not None:
            entries = entries[-history_max_entries:]  # oldest dropped first
    else:
        entries = []

    lines: list[str] = []
    j = 0
    for i, (role, text) in enumerate(turns):
        label = labels.get(role, role)
        lines.append(f"{label}: {text}" if text else f"{label}:")
        if j < len(entries) and (i + 1) == entries[j][0]:
            lines.append(f"Reference: {entries[j][1]}")
            j += 1

    return "\n".join(lines), num_turns


class GeminiAPIBackend(RetrievalBackend):
    def __init__(
        self,
        model: str,
        latency_gate_ms: int,
        api_key: str | None = None,
        prompt_template: str | None = None,
        context_formatting: dict | None = None,
        _client=None,  # injectable for tests
    ):
        self.model = model
        self.latency_gate_ms = latency_gate_ms
        self.context_formatting = context_formatting or {}
        template_path = Path(prompt_template) if prompt_template else _DEFAULT_PROMPT_TEMPLATE
        self._prompt_template = template_path.read_text()
        if _client is not None:
            self._client = _client
        else:
            from google import genai
            self._client = genai.Client(
                api_key=api_key or os.environ["GEMINI_API_KEY"]
            )

    @classmethod
    def from_config(cls, backend_def: dict, latency_gate_ms: int | None) -> "GeminiAPIBackend":
        return cls(
            model=backend_def["model"],
            latency_gate_ms=latency_gate_ms,
            prompt_template=backend_def.get("prompt_template"),
            context_formatting=backend_def.get("context_formatting"),
        )

    def retrieve(self, context: str, history: list[tuple[int, str]] | None = None) -> tuple[str, float]:
        formatted_context, num_turns = format_context(context, history, **self.context_formatting)
        if num_turns == 0:
            # Same short-circuit moshi's own generate_reference_text() does
            # on an empty context (e.g. a lone greeting) — no point spending
            # an API call on nothing to retrieve about.
            return "", 0.0

        prompt = self._prompt_template.format(context=formatted_context)
        last_exc: Exception | None = None

        for attempt in range(3):
            if attempt:
                wait = 2 ** (attempt - 1)  # 1s then 2s
                logger.warning("Gemini retrieve retry %d/3, waiting %ds", attempt + 1, wait)
                time.sleep(wait)

            t0 = time.perf_counter()
            try:
                from core.llm_judge import generate_gemini_text
                text = generate_gemini_text(self._client, self.model, prompt)
            except Exception as exc:
                latency = time.perf_counter() - t0
                logger.warning("Gemini API call failed after %.3fs: %s", latency, exc)
                last_exc = exc
                continue

            latency = time.perf_counter() - t0
            logger.debug("Gemini API call succeeded in %.3fs", latency)

            if latency * 1000 > self.latency_gate_ms:
                raise LatencyGateError(latency, self.latency_gate_ms)

            if not text:
                # Same known Gemini 3.x issue as above, manifesting as an
                # empty-but-"successful" response instead of an exception —
                # non-deterministic per call, so retry rather than treat
                # this as a valid empty answer.
                logger.warning("Gemini response had no visible text content — retrying")
                last_exc = RuntimeError("Gemini returned no text content")
                continue

            return text, latency

        raise last_exc  # type: ignore[misc]


BACKEND_REGISTRY: dict[str, type[RetrievalBackend]] = {
    "gemini_api": GeminiAPIBackend,
    "null_backend": NullBackend,
}


def build_retrieval_backend(name: str, backend_def: dict, latency_gate_ms: int | None = None) -> RetrievalBackend:
    """
    Factory: builds a RetrievalBackend instance from a resolved
    configs/retrieval_backends.yaml entry, keyed by its `type` field —
    called from both evals/runner.py's build_model() and
    scripts/instrumented_server.py's main(), the same registry building the
    backend for whichever path is running.
    """
    backend_type = backend_def.get("type")
    cls = BACKEND_REGISTRY.get(backend_type)
    if cls is None:
        raise ValueError(
            f"Unknown retrieval backend type {backend_type!r} for backend {name!r} — "
            f"known types: {sorted(BACKEND_REGISTRY)}"
        )
    return cls.from_config(backend_def, latency_gate_ms)


def build_backend_from_retrieval_config(retrieval_cfg: dict) -> RetrievalBackend:
    """
    Shared "which backend does this config actually want" logic — used by
    both evals/runner.py's build_model() and
    scripts/instrumented_server.py's main(), so the demo and eval paths
    can't drift on how NullBackend vs. a real backend gets selected.
    Requires retrieval_cfg to already have been resolved by
    core/config.py's load_config() (a "backend" name gets "_resolved_backend"
    attached).
    """
    if not retrieval_cfg.get("enabled"):
        return NullBackend()
    backend_def = retrieval_cfg.get("_resolved_backend")
    if backend_def is None:
        raise ValueError(
            "model.retrieval.enabled is true but model.retrieval.backend did "
            "not resolve — load the config via core.config.load_config(), "
            "not a bare yaml.safe_load()"
        )
    return build_retrieval_backend(retrieval_cfg["backend"], backend_def, retrieval_cfg.get("latency_gate_ms"))


def describe_backend(name: str, backend_def: dict) -> dict:
    """
    Display-safe summary of a resolved backend definition — for demo
    session logs (session_start's retrieval_backend field) and the live
    "instrumentation_session" push to the browser. Never includes a secret
    value (api_key_env only names an env var, never its value).
    """
    fields: dict = {"name": name, "type": backend_def.get("type", "")}
    for key in ("model", "base_url"):
        if key in backend_def:
            fields[key] = backend_def[key]
    return fields
