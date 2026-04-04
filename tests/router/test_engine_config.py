from src.router.engine import RegimeFetcher


def test_regime_fetcher_defaults_to_container_hmm_service(monkeypatch):
    monkeypatch.delenv("CONTABO_HMM_API_URL", raising=False)

    fetcher = RegimeFetcher()

    assert fetcher._api_url == "http://hmm-inference-api:8001"


def test_regime_fetcher_prefers_env_override(monkeypatch):
    monkeypatch.setenv("CONTABO_HMM_API_URL", "https://hmm.quantmindx.local")

    fetcher = RegimeFetcher()

    assert fetcher._api_url == "https://hmm.quantmindx.local"
