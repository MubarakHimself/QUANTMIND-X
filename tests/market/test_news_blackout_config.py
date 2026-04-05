import src.market.news_blackout as news_blackout


def test_news_blackout_not_configured_without_api_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    monkeypatch.setattr(news_blackout, "finnhub", object())
    assert news_blackout.is_news_blackout_configured() is False


def test_news_blackout_not_configured_without_dependency(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "key")
    monkeypatch.setattr(news_blackout, "finnhub", None)
    assert news_blackout.is_news_blackout_configured() is False


def test_news_blackout_configured_with_dependency_and_api_key(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "key")
    monkeypatch.setattr(news_blackout, "finnhub", object())
    assert news_blackout.is_news_blackout_configured() is True
