from unittest.mock import MagicMock

from src.api import svss_endpoints


def test_get_redis_client_uses_redis_url_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://redis.internal:6380/2")

    mock_client = MagicMock()
    mock_from_url = MagicMock(return_value=mock_client)
    monkeypatch.setattr(svss_endpoints.redis, "from_url", mock_from_url)

    client = svss_endpoints._get_redis_client()

    assert client is mock_client
    mock_from_url.assert_called_once_with("redis://redis.internal:6380/2")
