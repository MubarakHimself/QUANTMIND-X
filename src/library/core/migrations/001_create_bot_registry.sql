-- Migration: Create BotRegistry Table
-- ====================================
-- QuantMindLib V1 — BotRegistry persistence layer
-- Version: 001
-- UP: Creates bot_registry table with all RegistryRecord fields
-- DOWN: Drops bot_registry table

-- UP Migration
CREATE TABLE IF NOT EXISTS bot_registry (
    bot_id          TEXT        NOT NULL PRIMARY KEY,
    bot_spec_id     TEXT        NOT NULL,
    status          TEXT        NOT NULL CHECK (status IN ('ACTIVE', 'TRIAL', 'SUSPENDED', 'ARCHIVED')),
    tier            TEXT        NOT NULL CHECK (tier IN ('ELITE', 'PERFORMANCE_TEST', 'STANDARD', 'EVALUATION_CANDIDATE', 'AT_RISK', 'CIRCUIT_BROKEN')),
    registered_at_ms INTEGER    NOT NULL,
    last_updated_ms  INTEGER    NOT NULL,
    owner           TEXT        NOT NULL,
    variant_ids     TEXT        NOT NULL DEFAULT '[]',
    deployed_at     TEXT,
    CONSTRAINT bot_spec_id_not_empty CHECK (length(bot_spec_id) > 0)
);

-- Indexes: bot_spec_id (frequent lookup), status + tier (filtering), owner, registered_at_ms (sorting)
CREATE INDEX IF NOT EXISTS idx_bot_registry_bot_spec_id ON bot_registry(bot_spec_id);
CREATE INDEX IF NOT EXISTS idx_bot_registry_status ON bot_registry(status);
CREATE INDEX IF NOT EXISTS idx_bot_registry_tier ON bot_registry(tier);
CREATE INDEX IF NOT EXISTS idx_bot_registry_owner ON bot_registry(owner);
CREATE INDEX IF NOT EXISTS idx_bot_registry_registered_at_ms ON bot_registry(registered_at_ms);
-- Composite index for common filter + sort pattern
CREATE INDEX IF NOT EXISTS idx_bot_registry_status_tier ON bot_registry(status, tier);

-- DOWN Migration
DROP INDEX IF EXISTS idx_bot_registry_bot_spec_id;
DROP INDEX IF EXISTS idx_bot_registry_status;
DROP INDEX IF EXISTS idx_bot_registry_tier;
DROP INDEX IF EXISTS idx_bot_registry_owner;
DROP INDEX IF EXISTS idx_bot_registry_registered_at_ms;
DROP INDEX IF EXISTS idx_bot_registry_status_tier;
DROP TABLE IF EXISTS bot_registry;