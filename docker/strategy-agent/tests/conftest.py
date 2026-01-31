"""
Pytest configuration and fixtures for Docker tests
"""

import pytest
import subprocess
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "docker: mark test as Docker test (requires Docker daemon)"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark all tests in this file as docker tests
        if "test_docker" in str(item.fspath):
            item.add_marker(pytest.mark.docker)


@pytest.fixture(scope="session", autouse=True)
def check_docker_available():
    """Check if Docker is available before running tests"""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            pytest.skip("Docker is not available")
        return True
    except FileNotFoundError:
        pytest.skip("Docker is not installed")
    except Exception as e:
        pytest.skip(f"Docker check failed: {e}")


@pytest.fixture(autouse=True)
def cleanup_containers():
    """Cleanup any test containers after each test"""
    yield
    # Cleanup after test
    try:
        subprocess.run(
            ["docker", "rm", "-f", "test-strategy-agent"],
            capture_output=True,
            timeout=10
        )
    except:
        pass
