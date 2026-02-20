import pytest
import json
from unittest.mock import AsyncMock, patch
from src.cache.redis_client import GlobalCache

@pytest.mark.asyncio
async def test_global_cache_set_get():
    """Test basic SET and GET operations with mocking."""
    cache = GlobalCache(host="mock", port=6379)
    
    # Mock redis.Redis
    mock_redis = AsyncMock()
    
    with patch("redis.asyncio.Redis", return_value=mock_redis):
        await cache.connect()
        
        # Test SET
        await cache.set("test_key", {"data": "value"}, ttl=300)
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "quantmind:test_key"
        assert json.loads(args[1]) == {"data": "value"}
        assert kwargs["ex"] == 300
        
        # Test GET
        mock_redis.get.return_value = json.dumps({"data": "value"})
        result = await cache.get("test_key")
        assert result == {"data": "value"}
        mock_redis.get.assert_called_once_with("quantmind:test_key")

@pytest.mark.asyncio
async def test_global_cache_get_or_set():
    """Test cache-aside pattern (get_or_set)."""
    cache = GlobalCache(host="mock", port=6379)
    mock_redis = AsyncMock()
    
    with patch("redis.asyncio.Redis", return_value=mock_redis):
        await cache.connect()
        
        # Scenario 1: Cache miss
        mock_redis.get.return_value = None
        
        async def mock_func():
            return {"new": "data"}
            
        result = await cache.get_or_set("missing_key", mock_func, ttl=100)
        
        assert result == {"new": "data"}
        mock_redis.get.assert_called_with("quantmind:missing_key")
        mock_redis.set.assert_called_once()
        
        # Scenario 2: Cache hit
        mock_redis.set.reset_mock()
        mock_redis.get.return_value = json.dumps({"cached": "data"})
        
        result = await cache.get_or_set("hit_key", mock_func)
        
        assert result == {"cached": "data"}
        # Should not call mock_func again if cached (but our mock_func is just a helper here)
        # We check that .set was NOT called again
        assert mock_redis.set.call_count == 0
