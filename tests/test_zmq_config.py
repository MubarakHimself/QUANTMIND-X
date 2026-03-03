"""
Test for ZMQ endpoint configuration.

Verifies that ZMQ_ENDPOINT is properly configured in src.config.
"""

import pytest


def test_zmq_endpoint_configurable():
    """Test that ZMQ_ENDPOINT can be imported and is configurable."""
    from src.config import ZMQ_ENDPOINT

    assert ZMQ_ENDPOINT is not None
    assert isinstance(ZMQ_ENDPOINT, str)
    assert ZMQ_ENDPOINT.startswith('tcp://')


def test_zmq_endpoint_default_value():
    """Test that ZMQ_ENDPOINT has a sensible default."""
    from src.config import ZMQ_ENDPOINT

    # Default should be localhost:5555
    assert 'localhost' in ZMQ_ENDPOINT
    assert '5555' in ZMQ_ENDPOINT


def test_get_zmq_endpoint_function():
    """Test that get_zmq_endpoint helper function exists and works."""
    from src.config import get_zmq_endpoint

    endpoint = get_zmq_endpoint()
    assert endpoint is not None
    assert endpoint.startswith('tcp://')
