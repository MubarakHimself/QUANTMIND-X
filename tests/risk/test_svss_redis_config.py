from unittest.mock import MagicMock

from src.risk import svss_rvol_consumer, svss_service


def test_consumer_get_svss_readings_uses_redis_url_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://svss-cache.internal:6390/4")

    mock_client = MagicMock()
    mock_client.get.return_value = None
    mock_from_url = MagicMock(return_value=mock_client)
    monkeypatch.setattr(svss_rvol_consumer.redis, "from_url", mock_from_url)

    result = svss_rvol_consumer.get_svss_readings("EURUSD")

    assert result is None
    mock_from_url.assert_called_once_with("redis://svss-cache.internal:6390/4")


def test_service_get_svss_readings_uses_redis_url_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://svss-cache.internal:6390/5")

    mock_client = MagicMock()
    mock_client.get.return_value = None
    mock_from_url = MagicMock(return_value=mock_client)
    monkeypatch.setattr(svss_service.redis, "from_url", mock_from_url)

    result = svss_service.get_svss_readings("EURUSD")

    assert result is None
    mock_from_url.assert_called_once_with("redis://svss-cache.internal:6390/5")
