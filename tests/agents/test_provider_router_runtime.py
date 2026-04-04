import time

from src.agents.providers.router import ProviderInfo, ProviderRouter


def test_resolve_runtime_config_prefers_configured_provider():
    router = ProviderRouter(cache_ttl=60)
    provider = ProviderInfo(
        id="provider-1",
        provider_type="generic",
        display_name="Generic Provider",
        base_url="https://llm.example.com/v1",
        api_key="secret",
        is_active=True,
        model_list=[{"id": "generic-model-1"}],
        tier_assignment={"opus": "generic-model-opus"},
    )
    router._initialized = True
    router._last_refresh = time.time()
    router._primary_provider = provider
    router._all_providers = {"generic": provider}

    resolved = router.resolve_runtime_config(tier="opus")

    assert resolved is not None
    assert resolved.source == "provider_config"
    assert resolved.provider_type == "generic"
    assert resolved.base_url == "https://llm.example.com/v1"
    assert resolved.model == "generic-model-opus"


def test_resolve_runtime_config_supports_generic_env_contract(monkeypatch):
    router = ProviderRouter(cache_ttl=60)
    router._initialized = True
    router._last_refresh = time.time()
    router._primary_provider = None
    router._all_providers = {}

    monkeypatch.setenv("QMX_LLM_PROVIDER", "custom")
    monkeypatch.setenv("QMX_LLM_API_KEY", "env-secret")
    monkeypatch.setenv("QMX_LLM_BASE_URL", "https://env.example.com")
    monkeypatch.setenv("QMX_LLM_MODEL", "env-model")

    resolved = router.resolve_runtime_config()

    assert resolved is not None
    assert resolved.source == "env"
    assert resolved.provider_type == "custom"
    assert resolved.api_key == "env-secret"
    assert resolved.base_url == "https://env.example.com"
    assert resolved.model == "env-model"


def test_resolve_runtime_config_legacy_env_compatibility(monkeypatch):
    router = ProviderRouter(cache_ttl=60)
    router._initialized = True
    router._last_refresh = time.time()
    router._primary_provider = None
    router._all_providers = {}

    monkeypatch.delenv("QMX_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("QMX_LLM_API_KEY", raising=False)
    monkeypatch.delenv("QMX_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("QMX_LLM_MODEL", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "legacy-secret")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://legacy.example.com")
    monkeypatch.setenv("MINIMAX_MODEL", "legacy-model")

    resolved = router.resolve_runtime_config()

    assert resolved is not None
    assert resolved.source == "env"
    assert resolved.provider_type == "minimax"
    assert resolved.api_key == "legacy-secret"
    assert resolved.base_url == "https://legacy.example.com"
    assert resolved.model == "legacy-model"
