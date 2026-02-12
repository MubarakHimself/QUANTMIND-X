# Top-level pytest configuration for custom markers used across test suites.

def pytest_configure(config):
    # Register custom markers to avoid PytestUnknownMarkWarning
    config.addinivalue_line("markers", "router: tests for src/router and extensions")
    config.addinivalue_line("markers", "slow: marks tests as slow (load tests, extended operations)")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests (performance validation)")
