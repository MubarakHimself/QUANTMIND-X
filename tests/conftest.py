# Top-level pytest configuration for custom markers used across test suites.

def pytest_configure(config):
    # Register custom markers to avoid PytestUnknownMarkWarning
    config.addinivalue_line("markers", "router: tests for src/router and extensions")
