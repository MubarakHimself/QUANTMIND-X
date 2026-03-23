"""
Migration: Add Provider Config Extended Fields
=============================================

Adds new columns to provider_configs table for:
- provider_type (renamed from name)
- display_name
- api_key_encrypted (encrypted API key)
- tier_assignment (JSON)
- model_list (JSON)
- created_at_utc

Version: 015
"""

from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the provider config extended fields migration."""
    return Migration(
        version="015",
        name="add_provider_config_extended_fields",
        description="Add encrypted API key, display_name, tier_assignment, model_list to provider_configs",
        up_sql="""
            -- Rename 'name' to 'provider_type' if it exists
            ALTER TABLE provider_configs RENAME COLUMN name TO provider_type;

            -- Add display_name column
            ALTER TABLE provider_configs ADD COLUMN display_name VARCHAR(100);

            -- Rename api_key to api_key_encrypted
            ALTER TABLE provider_configs RENAME COLUMN api_key TO api_key_encrypted;

            -- Rename enabled to is_active
            ALTER TABLE provider_configs RENAME COLUMN enabled TO is_active;

            -- Rename created_at to created_at_utc
            ALTER TABLE provider_configs RENAME COLUMN created_at TO created_at_utc;

            -- Add new columns
            ALTER TABLE provider_configs ADD COLUMN model_list TEXT;
            ALTER TABLE provider_configs ADD COLUMN tier_assignment TEXT;
        """,
        down_sql="""
            -- Reverse operations (if needed for rollback)
            ALTER TABLE provider_configs DROP COLUMN tier_assignment;
            ALTER TABLE provider_configs DROP COLUMN model_list;
            ALTER TABLE provider_configs RENAME COLUMN created_at_utc TO created_at;
            ALTER TABLE provider_configs RENAME COLUMN is_active TO enabled;
            ALTER TABLE provider_configs RENAME COLUMN api_key_encrypted TO api_key;
            ALTER TABLE provider_configs DROP COLUMN display_name;
            ALTER TABLE provider_configs RENAME COLUMN provider_type TO name;
        """,
    )
