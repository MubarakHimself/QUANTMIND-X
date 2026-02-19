"""
Migration: Add HMM Regime Detection Tables
===========================================

Creates the following tables:
- hmm_models: Trained HMM model metadata
- hmm_shadow_logs: Prediction comparison logs
- hmm_deployments: Deployment state history
- hmm_sync_status: Server sync tracking

Version: 007
"""

from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the HMM tables migration."""
    return Migration(
        version="007",
        name="add_hmm_tables",
        description="Create HMM regime detection tables for models, shadow logs, deployments, and sync status",
        up_sql="""
            -- HMM Models table
            CREATE TABLE IF NOT EXISTS hmm_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version VARCHAR(20) NOT NULL,
                model_type VARCHAR(30) NOT NULL,
                symbol VARCHAR(20),
                timeframe VARCHAR(10),
                n_states INTEGER NOT NULL DEFAULT 4,
                log_likelihood REAL,
                state_distribution JSON,
                transition_matrix JSON,
                training_samples INTEGER NOT NULL DEFAULT 0,
                training_date TIMESTAMP,
                checksum VARCHAR(64),
                file_path VARCHAR(500),
                is_active BOOLEAN NOT NULL DEFAULT 1,
                validation_status VARCHAR(20) NOT NULL DEFAULT 'pending',
                validation_notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_hmm_models_version ON hmm_models(version);
            CREATE INDEX IF NOT EXISTS idx_hmm_models_type_symbol ON hmm_models(model_type, symbol);
            CREATE INDEX IF NOT EXISTS idx_hmm_models_active ON hmm_models(is_active);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_hmm_model_version_symbol_tf ON hmm_models(version, symbol, timeframe);

            -- HMM Shadow Logs table
            CREATE TABLE IF NOT EXISTS hmm_shadow_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                ising_regime VARCHAR(50) NOT NULL,
                ising_confidence REAL NOT NULL DEFAULT 0.0,
                hmm_regime VARCHAR(50) NOT NULL,
                hmm_state INTEGER NOT NULL,
                hmm_confidence REAL NOT NULL DEFAULT 0.0,
                agreement BOOLEAN NOT NULL DEFAULT 0,
                decision_source VARCHAR(20) NOT NULL DEFAULT 'ising',
                market_context JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES hmm_models(id)
            );

            CREATE INDEX IF NOT EXISTS idx_hmm_shadow_timestamp ON hmm_shadow_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_hmm_shadow_symbol_tf ON hmm_shadow_logs(symbol, timeframe);
            CREATE INDEX IF NOT EXISTS idx_hmm_shadow_agreement ON hmm_shadow_logs(agreement);
            CREATE INDEX IF NOT EXISTS idx_hmm_shadow_model ON hmm_shadow_logs(model_id);

            -- HMM Deployments table
            CREATE TABLE IF NOT EXISTS hmm_deployments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                mode VARCHAR(20) NOT NULL,
                previous_mode VARCHAR(20),
                transition_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                approved_by VARCHAR(100),
                approval_token VARCHAR(64),
                performance_metrics JSON,
                rollback_count INTEGER NOT NULL DEFAULT 0,
                is_active BOOLEAN NOT NULL DEFAULT 1,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES hmm_models(id)
            );

            CREATE INDEX IF NOT EXISTS idx_hmm_deployments_mode ON hmm_deployments(mode);
            CREATE INDEX IF NOT EXISTS idx_hmm_deployments_active ON hmm_deployments(is_active);
            CREATE INDEX IF NOT EXISTS idx_hmm_deployments_model ON hmm_deployments(model_id);

            -- HMM Sync Status table
            CREATE TABLE IF NOT EXISTS hmm_sync_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contabo_version VARCHAR(20),
                contabo_last_trained TIMESTAMP,
                cloudzy_version VARCHAR(20),
                cloudzy_last_deployed TIMESTAMP,
                version_mismatch BOOLEAN NOT NULL DEFAULT 0,
                last_sync_attempt TIMESTAMP,
                last_sync_status VARCHAR(20),
                sync_progress REAL NOT NULL DEFAULT 0.0,
                sync_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_hmm_sync_mismatch ON hmm_sync_status(version_mismatch);

            -- Insert initial sync status record
            INSERT INTO hmm_sync_status (contabo_version, cloudzy_version, version_mismatch, sync_message)
            VALUES (NULL, NULL, 0, 'No models synced yet');
        """,
        down_sql="""
            DROP TABLE IF EXISTS hmm_sync_status;
            DROP TABLE IF EXISTS hmm_deployments;
            DROP TABLE IF EXISTS hmm_shadow_logs;
            DROP TABLE IF EXISTS hmm_models;
        """,
        db_type="sqlite"
    )
