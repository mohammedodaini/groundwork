"""LLM provider abstraction.

Why an abstraction rather than calling the SDK directly (ADR-004):

1. Testability. `FakeLLM` lets the entire agent graph run in CI with no
   network, no keys, and no cost. Every test in `tests/` uses it.
2. Substitutability. Swapping Anthropic for OpenAI is one config value.
3. A single choke point for the things that actually matter in production:
   retries, timeouts, token accounting, and *schema-repair* on malformed output.

The key method is `structured()`: ask for a Pydantic model back, and get one,
or raise. Free-text LLM output has no place in a system meant to be audited.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from groundwork.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Any unrecoverable failure from a provider."""


class StructuredOutputError(LLMError):
    """The model could not be made to emit valid JSON for the target schema."""


class LLMResponse(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


def extract_json_block(text: str) -> str:
    """Pull the most likely JSON object out of a model response.

    Models wrap JSON in prose or fences despite instructions. Rather than fail
    the run, we make one deterministic repair attempt here, then let the caller
    retry with the validation error appended to the prompt.
    """
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


class LLMProvider(ABC):
    """Interface every provider implements."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @abstractmethod
    async def complete(self, *, system: str, user: str) -> LLMResponse:
        """One turn. No tools, no history - the graph owns orchestration."""

    async def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        max_repair_attempts: int = 2,
    ) -> T:
        """Return a validated instance of `schema`, repairing on failure.

        The repair loop is the interesting part for an interview: we feed the
        model its own ValidationError. In practice this fixes the large majority
        of schema misses (wrong enum casing, a stringified number, a missing
        optional) without a second full reasoning pass.
        """
        schema_json = json.dumps(schema.model_json_schema(), indent=2)
        instruction = (
            f"{user}\n\n"
            "Respond with a single JSON object and nothing else. "
            "No prose, no markdown fences.\n"
            f"It must validate against this JSON Schema:\n{schema_json}"
        )

        last_error: Exception | None = None
        for attempt in range(max_repair_attempts + 1):
            response = await self.complete(system=system, user=instruction)
            raw = extract_json_block(response.text)
            try:
                return schema.model_validate_json(raw)
            except (ValidationError, json.JSONDecodeError) as exc:
                last_error = exc
                logger.warning(
                    "structured_output_invalid",
                    extra={"attempt": attempt, "schema": schema.__name__},
                )
                instruction = (
                    f"{user}\n\n"
                    f"Your previous response was invalid:\n{raw[:1500]}\n\n"
                    f"It failed validation with:\n{str(exc)[:1500]}\n\n"
                    "Return ONLY a corrected JSON object matching this schema:\n"
                    f"{schema_json}"
                )

        raise StructuredOutputError(
            f"Could not obtain valid {schema.__name__} after "
            f"{max_repair_attempts + 1} attempts: {last_error}"
        ) from last_error

    # -- accounting --------------------------------------------------------

    def _record(self, response: LLMResponse) -> None:
        self.call_count += 1
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens

    def estimated_cost_usd(self) -> float:
        s = self.settings
        return (
            self.prompt_tokens / 1_000_000 * s.price_per_mtok_input_usd
            + self.completion_tokens / 1_000_000 * s.price_per_mtok_output_usd
        )


class FakeLLM(LLMProvider):
    """Deterministic provider for tests, CI and the offline demo.

    Responses are supplied as a queue or a routing callable. This is not a mock
    of an LLM's *intelligence* - it is a mock of the transport. Tests assert on
    how the graph handles well-formed, malformed and failing responses, which is
    exactly the behaviour that breaks in production.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        responses: list[str] | None = None,
        router: Any = None,
        fail_times: int = 0,
    ) -> None:
        super().__init__(settings)
        self._responses = list(responses or [])
        self._router = router
        self._fail_times = fail_times
        self.seen_prompts: list[tuple[str, str]] = []

    async def complete(self, *, system: str, user: str) -> LLMResponse:
        self.seen_prompts.append((system, user))

        if self._fail_times > 0:
            self._fail_times -= 1
            raise LLMError("Simulated provider failure")

        if self._router is not None:
            text = self._router(system, user)
        elif self._responses:
            text = self._responses.pop(0)
        else:
            text = "{}"

        response = LLMResponse(
            text=text,
            prompt_tokens=max(1, len(system) + len(user)) // 4,
            completion_tokens=max(1, len(text)) // 4,
            model="fake",
        )
        self._record(response)
        return response


class AnthropicLLM(LLMProvider):
    """Anthropic Messages API provider.

    Imported lazily so the package installs and tests run without the SDK.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        if settings.anthropic_api_key is None:
            raise LLMError("ANTHROPIC_API_KEY is required for llm_provider=anthropic")
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise LLMError("pip install anthropic") from exc
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def complete(self, *, system: str, user: str) -> LLMResponse:
        try:
            # Sampling params are omitted unless explicitly configured: current
            # Claude models reject `temperature` outright rather than ignoring it.
            kwargs: dict[str, Any] = {}
            if self.settings.llm_temperature is not None:
                kwargs["temperature"] = self.settings.llm_temperature
            msg = await self._client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        response = LLMResponse(
            text=text,
            prompt_tokens=msg.usage.input_tokens,
            completion_tokens=msg.usage.output_tokens,
            model=msg.model,
        )
        self._record(response)
        return response


class OpenAILLM(LLMProvider):  # pragma: no cover - parity implementation
    """OpenAI provider, and the base for every OpenAI-compatible gateway.

    OpenRouter, Groq, Together, vLLM and Ollama all speak the same Chat
    Completions wire format and differ only in two things: which base URL they
    point at, and which key they read. Subclasses therefore override two
    attributes and nothing else - no duplicated request logic, so a fix to
    token accounting or error mapping applies to all of them at once.
    """

    KEY_NAME = "OPENAI_API_KEY"
    DEFAULT_BASE_URL: str | None = None

    def _credential(self, settings: Settings) -> str | None:
        return (
            settings.openai_api_key.get_secret_value()
            if settings.openai_api_key
            else None
        )

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        api_key = self._credential(settings)
        if api_key is None:
            raise LLMError(
                f"{self.KEY_NAME} is required for llm_provider={settings.llm_provider}"
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LLMError("pip install openai") from exc
        self._client = AsyncOpenAI(
            api_key=api_key,
            # Explicit `or` chain, not a default argument: an operator-supplied
            # LLM_BASE_URL must beat the subclass default, so that pointing at a
            # self-hosted gateway never requires a code change.
            base_url=settings.llm_base_url or self.DEFAULT_BASE_URL,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def complete(self, *, system: str, user: str) -> LLMResponse:
        try:
            kwargs: dict[str, Any] = {}
            if self.settings.llm_temperature is not None:
                kwargs["temperature"] = self.settings.llm_temperature
            res = await self._client.chat.completions.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        except Exception as exc:
            raise LLMError(f"{type(self).__name__} request failed: {exc}") from exc

        # Gateways report upstream failures as HTTP 200 with an `error` envelope
        # and `choices: null`, rather than as an HTTP error status. The SDK
        # therefore does not raise, and indexing choices[0] blows up with an
        # opaque TypeError several frames from the cause. Overloaded free-tier
        # models hit this constantly, so surface the gateway's own message.
        error = getattr(res, "error", None)
        if error is not None or res.choices is None:
            detail = (
                error.get("message")
                if isinstance(error, dict)
                else (str(error) if error else "no choices returned")
            )
            raise LLMError(f"{type(self).__name__} gateway error: {detail}")

        usage = res.usage
        response = LLMResponse(
            text=res.choices[0].message.content or "",
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            model=res.model,
        )
        self._record(response)
        return response


class OpenRouterLLM(OpenAILLM):  # pragma: no cover - network dependent
    """OpenRouter: one endpoint fronting many models, including free ones.

    Free models carry a `:free` suffix, e.g.
    `meta-llama/llama-3.3-70b-instruct:free`. They are rate-limited and rotate
    over time, so the catalogue is deliberately NOT hardcoded here - check
    https://openrouter.ai/models?q=free and set LLM_MODEL. Set the price
    settings to 0.0 so cost estimates do not report spend that never happened.
    """

    KEY_NAME = "OPENROUTER_API_KEY"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def _credential(self, settings: Settings) -> str | None:
        return (
            settings.openrouter_api_key.get_secret_value()
            if settings.openrouter_api_key
            else None
        )


class GroqLLM(OpenAILLM):  # pragma: no cover - network dependent
    """Groq: a free tier over open-weights models, notably fast."""

    KEY_NAME = "GROQ_API_KEY"
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def _credential(self, settings: Settings) -> str | None:
        return (
            settings.groq_api_key.get_secret_value() if settings.groq_api_key else None
        )


class OllamaLLM(OpenAILLM):  # pragma: no cover - needs a local daemon
    """Ollama: models running locally. No key, no network, no cost.

    Ollama ignores the API key but the OpenAI SDK refuses to start without one,
    so we pass a sentinel rather than making the base class's key check
    conditional. The cost of the workaround is one confusing line here; the
    alternative was a branch in every subclass.
    """

    KEY_NAME = "(none required)"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def _credential(self, settings: Settings) -> str:
        return "ollama-local"


def build_llm(settings: Settings) -> LLMProvider:
    """Factory. The only place provider selection happens."""
    match settings.llm_provider:
        case "anthropic":
            return AnthropicLLM(settings)
        case "openai":
            return OpenAILLM(settings)
        case "openrouter":
            return OpenRouterLLM(settings)
        case "groq":
            return GroqLLM(settings)
        case "ollama":
            return OllamaLLM(settings)
        case "fake":
            return FakeLLM(settings)
        case other:  # pragma: no cover - unreachable via validated config
            raise LLMError(f"Unknown llm_provider: {other}")
