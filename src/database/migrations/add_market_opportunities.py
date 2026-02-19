"""
Migration: Add Market Opportunities
====================================

Stores detected market opportunities from the MarketScanner system.

Schema (from spec lines 709-724):
- id: Primary key
- scan_type: Type of scan (SESSION_BREAKOUT, VOLATILITY_SPIKE, NEWS_EVENT, ICT_SETUP)
- symbol: Trading symbol (EURUSD, GBPUSD, etc.)
- session: Market session (asian, london, ny, overlap)
- setup: Specific setup detected
- confidence: Confidence score (0-1)
- recommended_bots: JSON array of bot IDs recommended for this opportunity
- metadata: JSON with additional details (ATR values, price levels, etc.)
- timestamp: When the opportunity was detected
- expires_at: When the opportunity expires
- status: Status (active, expired, triggered)
- triggered_by: Bot ID that was activated for this opportunity

Indexes:
- idx_opportunities_symbol on symbol
- idx_opportunities_timestamp on timestamp
- idx_opportunities_scan_type on scan_type

Version: 010
"""

from src.database.migrations.migration_runner import Migration


def get_migration() -> Migration:
    """Get the market opportunities migration."""
    return Migration(
        version="010",
        name="add_market_opportunities",
        description="Add market_opportunities table for scanner alerts",
        up_sql="""
            -- Create market_opportunities table
            CREATE TABLE IF NOT EXISTS market_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_type VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                session VARCHAR(20),
                setup VARCHAR(100),
                confidence REAL DEFAULT 0.0,
                recommended_bots TEXT,
                metadata TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                status VARCHAR(20) DEFAULT 'active',
                triggered_by VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Create indexes for efficient querying
            CREATE INDEX IF NOT EXISTS idx_opportunities_symbol ON market_opportunities(symbol);
            CREATE INDEX IF NOT EXISTS idx_opportunities_timestamp ON market_opportunities(timestamp);
            CREATE INDEX IF NOT EXISTS idx_opportunities_scan_type ON market_opportunities(scan_type);
            CREATE INDEX IF NOT EXISTS idx_opportunities_status ON market_opportunities(status);
            CREATE INDEX IF NOT EXISTS idx_opportunities_session ON market_opportunities(session);
        """,
        down_sql="""
            -- Drop indexes
            DROP INDEX IF EXISTS idx_opportunities_symbol;
            DROP INDEX IF EXISTS idx_opportunities_timestamp;
            DROP INDEX IF EXISTS idx_opportunities_scan_type;
            DROP INDEX IF EXISTS idx_opportunities_status;
            DROP INDEX IF EXISTS idx_opportunities_session;
            
            -- Drop table
            DROP TABLE IF EXISTS market_opportunities;
        """,
        db_type="sqlite"
    )
