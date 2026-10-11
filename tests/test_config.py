"""Configuration that fails fast instead of failing late."""

import pytest

from groundwork.config import Settings


def test_blank_secrets_mean_unset() -> None:
    settings = Settings(_env_file=None, api_key="", anthropic_api_key="   ", tavily_api_key="")
    assert settings.api_key is None
    assert settings.anthropic_api_key is None
    assert settings.tavily_api_key is None


def test_blank_api_key_does_not_satisfy_production() -> None:
    """docker compose passes an unset API_KEY through as an empty string."""
    with pytest.raises(ValueError, match="requires API_KEY"):
        Settings(_env_file=None, environment="production", api_key="")


def test_real_api_key_satisfies_production() -> None:
    settings = Settings(_env_file=None, environment="production", api_key="a-real-key-value")
    assert settings.api_key is not None
