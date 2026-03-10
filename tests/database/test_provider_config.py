"""
Test for provider configuration table.

This test verifies that the provider_configs table exists in the database
for storing API keys and base URLs for model providers.
"""

import pytest
from sqlalchemy import text
from src.database.engine import get_session


def test_provider_config_table_exists():
    """Provider config table should exist after migration."""
    session = get_session()
    try:
        result = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='provider_configs'")
        )
        row = result.fetchone()
        assert row is not None, "provider_configs table should exist"
    finally:
        session.close()
