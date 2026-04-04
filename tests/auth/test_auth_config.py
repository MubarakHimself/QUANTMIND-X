from src.auth.config import get_auth0_config


def test_auth0_config_prefers_public_api_base_url(monkeypatch):
    monkeypatch.setenv("AUTH0_DOMAIN", "tenant.example.com")
    monkeypatch.setenv("AUTH0_CLIENT_ID", "client-123")
    monkeypatch.setenv("API_BASE_URL", "https://app.quantmindx.com")
    monkeypatch.setenv("INTERNAL_API_BASE_URL", "https://internal.quantmindx.local")

    config = get_auth0_config()

    assert config.callback_url == "https://app.quantmindx.com/api/auth/callback"


def test_auth0_config_falls_back_to_internal_api_base_url(monkeypatch):
    monkeypatch.setenv("AUTH0_DOMAIN", "tenant.example.com")
    monkeypatch.setenv("AUTH0_CLIENT_ID", "client-123")
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.setenv("INTERNAL_API_BASE_URL", "https://internal.quantmindx.local")

    config = get_auth0_config()

    assert config.callback_url == "https://internal.quantmindx.local/api/auth/callback"
