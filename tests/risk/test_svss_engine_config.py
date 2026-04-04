from src.risk.svss_engine import SVSSEngine


def test_svss_engine_uses_redis_url_env_by_default(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://svss-engine.internal:6381/6")

    engine = SVSSEngine(symbols=["EURUSD"])

    assert engine._redis_url == "redis://svss-engine.internal:6381/6"
