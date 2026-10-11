"""Typed application configuration.

All configuration arrives via environment variables and is validated once at
startup. Nothing in the codebase reads `os.environ` directly - that keeps
secrets out of scattered call sites and makes the whole config surface visible
in one file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The localhost origin that ships as the default. Named so the production
# check below can tell "deliberately configured" from "never configured".
DEV_CORS_ORIGINS = ("http://localhost:3000",)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- app ---------------------------------------------------------------
    app_name: str = "groundwork"
    environment: Literal["local", "test", "production"] = "local"
    log_level: str = "INFO"
    log_json: bool = False

    # -- providers ---------------------------------------------------------
    # "fake" needs no credentials and is what the test-suite and the offline
    # demo use. This is why the repo is runnable by a recruiter in 60 seconds.
    llm_provider: Literal["anthropic", "openai", "openrouter", "groq", "ollama", "fake"] = "fake"
    search_provider: Literal["tavily", "brave", "searxng", "fake"] = "fake"

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    tavily_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None

    # Tavily bills 'advanced' at roughly double 'basic'. Exposed because a
    # free tier disappears fast at one research run per few dozen credits.
    tavily_search_depth: Literal["basic", "advanced"] = "advanced"

    # A SearXNG instance you control. Needs `json` in its `search.formats`.
    searxng_base_url: str = "http://localhost:8888"

    # Overrides the provider's default endpoint. Every OpenAI-compatible
    # gateway (OpenRouter, Groq, Together, vLLM, Ollama) differs from
    # OpenAI in this one value, so aiming at one is a config change, not code.
    llm_base_url: str | None = None

    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 4096
    # Current Claude models (Sonnet 5, Opus 5, Opus 4.7+) REMOVED sampling
    # parameters and reject `temperature` with a 400. None means omit it from
    # the request; set a float only for a model that still accepts one.
    llm_temperature: float | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 3

    # Rough per-million-token prices, used only to *estimate* run cost.
    # Kept in config because they change; never hardcoded in logic.
    price_per_mtok_input_usd: float = 3.0
    price_per_mtok_output_usd: float = 15.0

    # -- fetching ----------------------------------------------------------
    fetch_timeout_seconds: float = 15.0
    fetch_max_concurrency: int = 5
    user_agent: str = "GroundworkResearchBot/0.1 (+https://example.org/bot)"

    # -- persistence -------------------------------------------------------
    # SQLite default means `git clone && make run` works with no Docker.
    # docker-compose overrides this to Postgres.
    database_url: str = "sqlite+aiosqlite:///./groundwork.db"

    # -- api ---------------------------------------------------------------
    api_key: SecretStr | None = Field(
        default=None,
        description="If set, every /api endpoint requires X-API-Key. Required in production.",
    )
    cors_origins: list[str] = Field(default_factory=lambda: list(DEV_CORS_ORIGINS))
    rate_limit_per_minute: int = 30

    @field_validator(
        "api_key",
        "anthropic_api_key",
        "openai_api_key",
        "openrouter_api_key",
        "groq_api_key",
        "tavily_api_key",
        "brave_api_key",
        mode="before",
    )
    @classmethod
    def _blank_means_unset(cls, value: object) -> object:
        """Treat `KEY=` as no key at all, not as a key that is the empty string.

        docker compose passes `${API_KEY:-}` through as an empty string, and so
        does a half-edited .env. Read literally, an empty API_KEY would pass the
        production check below as "configured" while every request failed 401,
        and an empty provider key would pass the provider's own check only to
        fail later, on the first paid call, with a less useful error.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _production_must_be_authenticated(self) -> Settings:
        """Refuse to start an unauthenticated production deployment.

        `api_key` defaults to None so local dev stays frictionless. That default
        is fine on a laptop and indefensible in production: every endpoint,
        including the one that spends money on LLM calls and the ones that
        return past research, would be open to anyone who can reach the port.
        A missing environment variable should not be the difference between a
        private service and a public one, so we fail at startup instead.

        CORS is only warned about, not enforced: a deployment that serves the
        bundled UI from this same origin never needs a cross-origin entry, so
        the default there is a smell rather than a hole. The warning is emitted
        in `main.lifespan`, once logging is configured.
        """
        if self.environment == "production" and self.api_key is None:
            raise ValueError(
                "ENVIRONMENT=production requires API_KEY to be set; refusing to "
                "start an unauthenticated deployment. Set API_KEY, or run with "
                "ENVIRONMENT=local if this really is a local instance."
            )
        return self

    @property
    def uses_real_llm(self) -> bool:
        return self.llm_provider != "fake"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. `lru_cache` gives us cheap dependency injection."""
    return Settings()
