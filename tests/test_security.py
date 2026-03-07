"""Security tests for configuration."""
import os

from src.config import Settings


def test_no_default_secret():
    """Default secret key should not be in production."""
    settings = Settings()
    assert settings.secret_key != "your-secret-key-change-in-production", \
        "Default secret key is insecure and should not be used in production"


def test_secret_key_is_configurable():
    """Secret key should be configurable via environment variable."""
    # Test that we can set a custom secret key
    os.environ['SECRET_KEY'] = 'test-secret-key'
    settings = Settings()
    assert settings.secret_key == 'test-secret-key'
    # Clean up
    del os.environ['SECRET_KEY']


def test_secret_key_defaults_to_empty():
    """Secret key should default to empty string when not set."""
    # Make sure SECRET_KEY is not in environment
    if 'SECRET_KEY' in os.environ:
        del os.environ['SECRET_KEY']
    settings = Settings()
    assert settings.secret_key is None, \
        "Secret key should default to None when not set"
