"""Provider-selection tests.

These cover the OpenAI-compatible gateway family (OpenRouter, Groq, Ollama).
The point of the family is that a new gateway is a config value, not code, so
the tests assert on *wiring* - which class, which endpoint, which key - rather
than on network behaviour, which `FakeLLM` already covers everywhere else.
"""

from __future__ import annotations

import pytest

from groundwork.config import Settings
from groundwork.providers.llm import (
    GroqLLM,
    LLMError,
    OllamaLLM,
    OpenAILLM,
    OpenRouterLLM,
    build_llm,
)
from groundwork.providers.search import (
    BraveSearch,
    SearchError,
    SearxngSearch,
    TavilySearch,
    build_search,
)

# The OpenAI SDK is an optional extra; skip rather than fail a clean install.
pytest.importorskip("openai")


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


@pytest.mark.parametrize(
    ("kwargs", "expected_cls", "expected_host"),
    [
        (
            {"llm_provider": "openrouter", "openrouter_api_key": "sk-or-x"},
            OpenRouterLLM,
            "openrouter.ai",
        ),
        (
            {"llm_provider": "groq", "groq_api_key": "gsk-x"},
            GroqLLM,
            "api.groq.com",
        ),
        ({"llm_provider": "ollama"}, OllamaLLM, "localhost"),
        (
            {"llm_provider": "openai", "openai_api_key": "sk-x"},
            OpenAILLM,
            "api.openai.com",
        ),
    ],
)
def test_gateway_selects_class_and_endpoint(kwargs, expected_cls, expected_host) -> None:
    provider = build_llm(_settings(**kwargs))
    assert isinstance(provider, expected_cls)
    assert expected_host in str(provider._client.base_url)


def test_explicit_base_url_overrides_gateway_default() -> None:
    """An operator pointing at a self-hosted gateway must not need a code change."""
    provider = build_llm(
        _settings(
            llm_provider="openrouter",
            openrouter_api_key="sk-or-x",
            llm_base_url="https://vllm.internal/v1",
        )
    )
    assert "vllm.internal" in str(provider._client.base_url)
    assert "openrouter.ai" not in str(provider._client.base_url)


def test_ollama_needs_no_credential() -> None:
    """Local models must run with no account and no key at all."""
    assert isinstance(build_llm(_settings(llm_provider="ollama")), OllamaLLM)


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("openrouter", "OPENROUTER_API_KEY"), ("groq", "GROQ_API_KEY")],
)
def test_missing_key_names_the_variable_to_set(provider: str, key_name: str) -> None:
    """A misconfigured deploy should say which variable is missing, not just fail."""
    with pytest.raises(LLMError, match=key_name):
        build_llm(_settings(llm_provider=provider))


def test_each_gateway_reads_only_its_own_key() -> None:
    """An OPENAI_API_KEY in the environment must not silently satisfy OpenRouter."""
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        build_llm(_settings(llm_provider="openrouter", openai_api_key="sk-openai"))


def test_free_tier_pricing_reports_zero_cost() -> None:
    """Free models must not accrue a phantom spend figure in the metrics."""
    provider = build_llm(
        _settings(
            llm_provider="openrouter",
            openrouter_api_key="sk-or-x",
            price_per_mtok_input_usd=0.0,
            price_per_mtok_output_usd=0.0,
        )
    )
    provider.prompt_tokens = 1_000_000
    provider.completion_tokens = 1_000_000
    assert provider.estimated_cost_usd() == 0.0


# --------------------------------------------------------------------------
# Search providers
#
# Brave and SearXNG exist so the system can do real research without Tavily's
# metered credits. They are a GET returning JSON, so the provider-specific
# surface is exactly two methods - which is what these tests cover.
# --------------------------------------------------------------------------


def test_search_provider_selection() -> None:
    assert isinstance(
        build_search(_settings(search_provider="brave", brave_api_key="b")), BraveSearch
    )
    assert isinstance(build_search(_settings(search_provider="searxng")), SearxngSearch)
    assert isinstance(
        build_search(_settings(search_provider="tavily", tavily_api_key="t")), TavilySearch
    )


def test_brave_requires_its_own_key() -> None:
    with pytest.raises(SearchError, match="BRAVE_API_KEY"):
        build_search(_settings(search_provider="brave"))


def test_brave_sends_key_as_header_not_query_string() -> None:
    """A key in the query string leaks into proxy and server access logs."""
    provider = build_search(_settings(search_provider="brave", brave_api_key="secret-key"))
    _url, params, headers = provider._request("wholesalers gelderland", 5)

    assert headers["X-Subscription-Token"] == "secret-key"
    assert "secret-key" not in str(params)
    assert params["q"] == "wholesalers gelderland"


def test_brave_parses_nested_result_shape() -> None:
    provider = build_search(_settings(search_provider="brave", brave_api_key="k"))
    hits = provider._parse(
        {
            "web": {
                "results": [
                    {"url": "https://a.example/1", "title": "A", "description": "snippet a"},
                    {"title": "no url - must be dropped"},
                ]
            }
        }
    )
    assert len(hits) == 1
    assert hits[0].url == "https://a.example/1"
    assert hits[0].snippet == "snippet a"


def test_searxng_needs_no_credential_and_honours_base_url() -> None:
    """The search-side equivalent of Ollama: local, keyless."""
    provider = build_search(
        _settings(search_provider="searxng", searxng_base_url="http://searx.internal:8888/")
    )
    url, params, _ = provider._request("q", 5)

    assert url == "http://searx.internal:8888/search"  # trailing slash normalised
    assert params["format"] == "json"


def test_tavily_search_depth_is_configurable() -> None:
    """`advanced` bills at roughly double `basic`, which matters on a free tier."""
    provider = build_search(
        _settings(search_provider="tavily", tavily_api_key="t", tavily_search_depth="basic")
    )
    assert provider.settings.tavily_search_depth == "basic"


def test_gateway_error_envelope_becomes_a_readable_llm_error() -> None:
    """OpenRouter reports upstream failures as HTTP 200 + `error`, choices=null.

    Without this guard the SDK does not raise, `choices[0]` throws an opaque
    `TypeError: 'NoneType' object is not subscriptable`, and the real reason -
    an overloaded free model - never reaches the logs.
    """
    import asyncio
    from types import SimpleNamespace

    provider = build_llm(_settings(llm_provider="openrouter", openrouter_api_key="k"))

    async def fake_create(**kwargs):
        return SimpleNamespace(
            choices=None,
            usage=None,
            model=None,
            error={"message": "Upstream error from Nvidia: Service temporarily overloaded"},
        )

    provider._client.chat.completions.create = fake_create

    with pytest.raises(LLMError, match="Service temporarily overloaded"):
        asyncio.run(provider.complete(system="s", user="u"))


def test_missing_choices_without_error_field_still_raises() -> None:
    """Defensive: some gateways omit `error` and simply return no choices."""
    import asyncio
    from types import SimpleNamespace

    provider = build_llm(_settings(llm_provider="groq", groq_api_key="k"))

    async def fake_create(**kwargs):
        return SimpleNamespace(choices=None, usage=None, model=None, error=None)

    provider._client.chat.completions.create = fake_create

    with pytest.raises(LLMError, match="no choices returned"):
        asyncio.run(provider.complete(system="s", user="u"))


def test_temperature_omitted_when_unset() -> None:
    """Sonnet 5 and other current models reject `temperature` with a 400.

    The request must omit the key entirely rather than sending a default, so
    this asserts on the kwargs actually handed to the SDK.
    """
    import asyncio
    from types import SimpleNamespace

    captured: dict = {}
    provider = build_llm(_settings(llm_provider="openrouter", openrouter_api_key="k"))

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="m",
            error=None,
        )

    provider._client.chat.completions.create = fake_create
    asyncio.run(provider.complete(system="s", user="u"))
    assert "temperature" not in captured


def test_temperature_sent_when_explicitly_configured() -> None:
    """Older models still accept it, so an explicit value must pass through."""
    import asyncio
    from types import SimpleNamespace

    captured: dict = {}
    provider = build_llm(
        _settings(llm_provider="openrouter", openrouter_api_key="k", llm_temperature=0.7)
    )

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            model="m",
            error=None,
        )

    provider._client.chat.completions.create = fake_create
    asyncio.run(provider.complete(system="s", user="u"))
    assert captured["temperature"] == 0.7
