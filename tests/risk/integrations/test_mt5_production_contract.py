from src.risk.integrations.mt5_client import create_mt5_client, get_mt5_client


def test_create_mt5_client_defaults_disable_simulated_fallback(monkeypatch):
    monkeypatch.delenv("MT5_ALLOW_SIMULATED_FALLBACK", raising=False)

    client = create_mt5_client()

    assert client.fallback_to_simulated is False


def test_create_mt5_client_can_opt_in_to_simulated_fallback(monkeypatch):
    monkeypatch.setenv("MT5_ALLOW_SIMULATED_FALLBACK", "true")

    client = create_mt5_client()

    assert client.fallback_to_simulated is True


def test_get_mt5_client_compat_export_is_available(monkeypatch):
    monkeypatch.delenv("MT5_ALLOW_SIMULATED_FALLBACK", raising=False)

    client = get_mt5_client(reset=True)

    assert client.fallback_to_simulated is False
