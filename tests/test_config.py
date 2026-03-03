"""Tests for configuration management."""
import os
import pytest


def test_config_from_env_database():
    """Database URL should be configurable via environment variable."""
    os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost/test'
    # Need to reload config to pick up the new env var
    from src.config import get_settings
    settings = get_settings()
    # Force reload by clearing singleton
    import src.config
    src.config._settings = None
    # Re-import with new env
    os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost/test'
    from src.config import get_database_url
    result = get_database_url()
    assert result == 'postgresql://test:test@localhost/test'
    # Cleanup
    del os.environ['DATABASE_URL']
    src.config._settings = None


def test_config_from_env_redis():
    """Redis URL should be configurable via environment variable."""
    os.environ['REDIS_URL'] = 'redis://custom-host:6379'
    from src.config import get_redis_url
    result = get_redis_url()
    assert result == 'redis://custom-host:6379'
    # Cleanup
    del os.environ['REDIS_URL']


def test_config_defaults():
    """Should have sensible defaults when env vars are not set."""
    # Clear any existing env vars
    for var in ['DATABASE_URL', 'REDIS_URL']:
        if var in os.environ:
            del os.environ[var]

    # Reload config
    import src.config
    src.config._settings = None

    from src.config import get_database_url, get_redis_url
    assert get_database_url() == 'sqlite:///data/quantmind.db'
    assert get_redis_url() == 'redis://localhost:6379'
