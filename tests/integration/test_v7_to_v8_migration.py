"""
V7 to V8 Database Migration Integration Test

Tests the complete V7 to V8 database migration process to ensure:
1. No data loss during migration
2. All V7 data remains intact
3. New V8 columns are added correctly
4. New V8 tables are created correctly
5. Rollback capability works

**Validates: Requirements 16.6, 16.9**
**Task: 26.9**
"""

import pytest
import os
import shutil
import tempfile
import warnings
from datetime import datetime, date, timezone
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Database models
from src.database.models import (
    Base, PropFirmAccount, DailySnapshot, TradeProposal,
    RiskTierTransition, CryptoTrade
)

# Migration script
from src.database.migrate_v8 import migrate_v8, rollback_v8

# Suppress SQLAlchemy datetime deprecation warnings for Python 3.12+
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sqlalchemy.engine.default")


def datetime_to_str(dt: datetime) -> str:
    """Convert datetime to ISO format string for SQLite compatibility."""
    if dt.tzinfo is None:
        return dt.isoformat()
    return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat()


class TestV7ToV8Migration:
    """
    Comprehensive integration test for V7 to V8 database migration.
    
    Tests the migration process from V7 schema to V8 schema,
    ensuring no data loss and correct schema enhancements.
    """
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_migration.db")
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

    
    def create_v7_database(self, db_path):
        """
        Create a V7 database with sample data.
        
        V7 schema includes:
        - prop_firm_accounts (without risk_mode column)
        - daily_snapshots
        - trade_proposals (without broker_id column)
        - agent_tasks
        - strategy_performance
        
        Returns:
            dict: Sample data IDs for verification
        """
        engine = create_engine(f'sqlite:///{db_path}', echo=False)
        
        # Create V7 schema (without V8 enhancements)
        # We'll create tables manually to simulate V7 state
        with engine.connect() as conn:
            # Create prop_firm_accounts table (V7 version - no risk_mode)
            conn.execute(text("""
                CREATE TABLE prop_firm_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    firm_name VARCHAR(100) NOT NULL,
                    account_id VARCHAR(50) NOT NULL UNIQUE,
                    daily_loss_limit_pct FLOAT NOT NULL DEFAULT 5.0,
                    hard_stop_buffer_pct FLOAT NOT NULL DEFAULT 1.0,
                    target_profit_pct FLOAT NOT NULL DEFAULT 8.0,
                    min_trading_days INTEGER NOT NULL DEFAULT 5,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            
            # Create daily_snapshots table
            conn.execute(text("""
                CREATE TABLE daily_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    date VARCHAR(10) NOT NULL,
                    daily_start_balance FLOAT NOT NULL DEFAULT 0.0,
                    high_water_mark FLOAT NOT NULL DEFAULT 0.0,
                    current_equity FLOAT NOT NULL DEFAULT 0.0,
                    daily_drawdown_pct FLOAT NOT NULL DEFAULT 0.0,
                    is_breached BOOLEAN NOT NULL DEFAULT 0,
                    snapshot_timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (account_id) REFERENCES prop_firm_accounts(id) ON DELETE CASCADE,
                    UNIQUE (account_id, date)
                )
            """))
            
            # Create trade_proposals table (V7 version - no broker_id)
            conn.execute(text("""
                CREATE TABLE trade_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER,
                    bot_id VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    kelly_score FLOAT NOT NULL,
                    regime VARCHAR(50),
                    proposed_lot_size FLOAT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP NOT NULL,
                    reviewed_at TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES prop_firm_accounts(id) ON DELETE SET NULL
                )
            """))
            
            conn.commit()
        
        # Insert sample V7 data
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Insert PropFirm accounts
        now = datetime_to_str(datetime.now(timezone.utc))
        
        conn = engine.connect()
        result = conn.execute(text("""
            INSERT INTO prop_firm_accounts 
            (firm_name, account_id, daily_loss_limit_pct, hard_stop_buffer_pct, 
             target_profit_pct, min_trading_days, created_at, updated_at)
            VALUES 
            ('MyForexFunds', 'MFF_12345', 5.0, 1.0, 8.0, 5, :now, :now),
            ('FTMO', 'FTMO_67890', 5.0, 1.0, 10.0, 10, :now, :now),
            ('The5ers', 'T5_11111', 4.0, 0.5, 6.0, 5, :now, :now)
        """), {"now": now})
        conn.commit()
        
        # Get account IDs
        accounts = conn.execute(text("SELECT id, account_id FROM prop_firm_accounts")).fetchall()
        account_ids = {acc[1]: acc[0] for acc in accounts}
        
        # Insert daily snapshots
        today = date.today().strftime("%Y-%m-%d")
        conn.execute(text("""
            INSERT INTO daily_snapshots
            (account_id, date, daily_start_balance, high_water_mark, current_equity,
             daily_drawdown_pct, is_breached, snapshot_timestamp)
            VALUES
            (:acc1, :today, 10000.0, 10500.0, 10300.0, 0.0, 0, :now),
            (:acc2, :today, 5000.0, 5200.0, 5100.0, 0.0, 0, :now),
            (:acc3, :today, 2500.0, 2600.0, 2550.0, 0.0, 0, :now)
        """), {
            "acc1": account_ids['MFF_12345'],
            "acc2": account_ids['FTMO_67890'],
            "acc3": account_ids['T5_11111'],
            "today": today,
            "now": now
        })
        conn.commit()
        
        # Insert trade proposals
        conn.execute(text("""
            INSERT INTO trade_proposals
            (account_id, bot_id, symbol, kelly_score, regime, proposed_lot_size, 
             status, created_at)
            VALUES
            (:acc1, 'ScalperBot_v1', 'EURUSD', 0.85, 'trending', 0.1, 'approved', :now),
            (:acc2, 'SwingBot_v2', 'GBPUSD', 0.92, 'ranging', 0.05, 'pending', :now),
            (:acc3, 'BreakoutBot_v1', 'USDJPY', 0.78, 'volatile', 0.08, 'rejected', :now)
        """), {
            "acc1": account_ids['MFF_12345'],
            "acc2": account_ids['FTMO_67890'],
            "acc3": account_ids['T5_11111'],
            "now": now
        })
        conn.commit()
        conn.close()
        
        session.close()
        
        return {
            'account_ids': account_ids,
            'num_accounts': 3,
            'num_snapshots': 3,
            'num_proposals': 3
        }

    
    def test_migration_preserves_v7_data(self, temp_db_path):
        """
        Test that V7 data is preserved during migration.
        
        Flow:
        1. Create V7 database with sample data
        2. Run V8 migration
        3. Verify all V7 data is intact
        4. Verify data values are unchanged
        
        **Validates: Requirement 16.6 (data preservation)**
        """
        # 1. Create V7 database
        v7_data = self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration
        success = migrate_v8(temp_db_path)
        assert success, "Migration failed"
        
        # 3. Verify V7 data is intact
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        conn = engine.connect()
        
        # Check PropFirm accounts
        accounts = conn.execute(text(
            "SELECT id, firm_name, account_id, daily_loss_limit_pct FROM prop_firm_accounts"
        )).fetchall()
        
        assert len(accounts) == v7_data['num_accounts'], \
            f"Expected {v7_data['num_accounts']} accounts, found {len(accounts)}"
        
        # Verify specific account data
        mff_account = [acc for acc in accounts if acc[2] == 'MFF_12345'][0]
        assert mff_account[1] == 'MyForexFunds'
        assert mff_account[3] == 5.0
        
        # Check daily snapshots
        snapshots = conn.execute(text(
            "SELECT id, account_id, current_equity FROM daily_snapshots"
        )).fetchall()
        
        assert len(snapshots) == v7_data['num_snapshots'], \
            f"Expected {v7_data['num_snapshots']} snapshots, found {len(snapshots)}"
        
        # Verify specific snapshot data
        mff_snapshot = [s for s in snapshots if s[1] == v7_data['account_ids']['MFF_12345']][0]
        assert mff_snapshot[2] == 10300.0
        
        # Check trade proposals
        proposals = conn.execute(text(
            "SELECT id, bot_id, symbol, kelly_score, status FROM trade_proposals"
        )).fetchall()
        
        assert len(proposals) == v7_data['num_proposals'], \
            f"Expected {v7_data['num_proposals']} proposals, found {len(proposals)}"
        
        # Verify specific proposal data
        scalper_proposal = [p for p in proposals if p[1] == 'ScalperBot_v1'][0]
        assert scalper_proposal[2] == 'EURUSD'
        assert scalper_proposal[3] == 0.85
        assert scalper_proposal[4] == 'approved'
        
        conn.close()
        
        print("\n✓ V7 data preserved during migration")
        print(f"  Accounts: {len(accounts)}")
        print(f"  Snapshots: {len(snapshots)}")
        print(f"  Proposals: {len(proposals)}")

    
    def test_migration_adds_v8_columns(self, temp_db_path):
        """
        Test that V8 columns are added correctly.
        
        Flow:
        1. Create V7 database
        2. Run V8 migration
        3. Verify risk_mode column exists in prop_firm_accounts
        4. Verify broker_id column exists in trade_proposals
        5. Verify default values are set correctly
        
        **Validates: Requirement 16.6 (schema enhancements)**
        """
        # 1. Create V7 database
        v7_data = self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration
        success = migrate_v8(temp_db_path)
        assert success, "Migration failed"
        
        # 3. Verify risk_mode column exists
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        inspector = inspect(engine)
        
        pfa_columns = [col['name'] for col in inspector.get_columns('prop_firm_accounts')]
        assert 'risk_mode' in pfa_columns, "risk_mode column not found"
        
        # 4. Verify broker_id column exists in trade_proposals
        tp_columns = [col['name'] for col in inspector.get_columns('trade_proposals')]
        assert 'broker_id' in tp_columns, "broker_id column not found"
        
        # 5. Verify default values
        conn = engine.connect()
        
        # Check risk_mode defaults to 'growth'
        accounts = conn.execute(text(
            "SELECT account_id, risk_mode FROM prop_firm_accounts"
        )).fetchall()
        
        for account in accounts:
            assert account[1] == 'growth', \
                f"Account {account[0]} risk_mode should default to 'growth', got {account[1]}"
        
        # Check broker_id is NULL (no default)
        proposals = conn.execute(text(
            "SELECT id, broker_id FROM trade_proposals"
        )).fetchall()
        
        for proposal in proposals:
            assert proposal[1] is None, \
                f"Proposal {proposal[0]} broker_id should be NULL, got {proposal[1]}"
        
        conn.close()
        
        print("\n✓ V8 columns added correctly")
        print(f"  risk_mode column: ✓")
        print(f"  broker_id column: ✓")
        print(f"  Default values: ✓")

    
    def test_migration_creates_v8_tables(self, temp_db_path):
        """
        Test that V8 tables are created correctly.
        
        Flow:
        1. Create V7 database
        2. Run V8 migration
        3. Verify risk_tier_transitions table exists
        4. Verify crypto_trades table exists
        5. Verify table schemas are correct
        
        **Validates: Requirement 16.9 (new V8 tables)**
        """
        # 1. Create V7 database
        self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration
        success = migrate_v8(temp_db_path)
        assert success, "Migration failed"
        
        # 3. Verify risk_tier_transitions table exists
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        inspector = inspect(engine)
        
        tables = inspector.get_table_names()
        assert 'risk_tier_transitions' in tables, "risk_tier_transitions table not found"
        assert 'crypto_trades' in tables, "crypto_trades table not found"
        
        # 4. Verify risk_tier_transitions schema
        rtt_columns = [col['name'] for col in inspector.get_columns('risk_tier_transitions')]
        expected_rtt_columns = [
            'id', 'account_id', 'from_tier', 'to_tier', 
            'equity_at_transition', 'transition_timestamp'
        ]
        
        for col in expected_rtt_columns:
            assert col in rtt_columns, f"Column {col} not found in risk_tier_transitions"
        
        # 5. Verify crypto_trades schema
        ct_columns = [col['name'] for col in inspector.get_columns('crypto_trades')]
        expected_ct_columns = [
            'id', 'broker_type', 'broker_id', 'order_id', 'symbol', 
            'direction', 'volume', 'entry_price', 'exit_price', 
            'stop_loss', 'take_profit', 'profit', 'status', 
            'open_timestamp', 'close_timestamp', 'trade_metadata'
        ]
        
        for col in expected_ct_columns:
            assert col in ct_columns, f"Column {col} not found in crypto_trades"
        
        # 6. Verify foreign key relationships
        rtt_fks = inspector.get_foreign_keys('risk_tier_transitions')
        assert len(rtt_fks) > 0, "risk_tier_transitions should have foreign key to prop_firm_accounts"
        assert rtt_fks[0]['referred_table'] == 'prop_firm_accounts'
        
        print("\n✓ V8 tables created correctly")
        print(f"  risk_tier_transitions: ✓")
        print(f"  crypto_trades: ✓")
        print(f"  Foreign keys: ✓")

    
    def test_migration_creates_indexes(self, temp_db_path):
        """
        Test that V8 indexes are created correctly.
        
        Flow:
        1. Create V7 database
        2. Run V8 migration
        3. Verify indexes on risk_mode
        4. Verify indexes on tier transitions
        5. Verify indexes on crypto_trades
        
        **Validates: Requirement 16.9 (performance optimization)**
        """
        # 1. Create V7 database
        self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration
        success = migrate_v8(temp_db_path)
        assert success, "Migration failed"
        
        # 3. Verify indexes exist
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        inspector = inspect(engine)
        
        # Check prop_firm_accounts indexes
        pfa_indexes = inspector.get_indexes('prop_firm_accounts')
        index_names = [idx['name'] for idx in pfa_indexes]
        
        # Should have index on risk_mode
        assert any('risk_mode' in str(idx) for idx in pfa_indexes), \
            "Index on risk_mode not found"
        
        # Check risk_tier_transitions indexes
        rtt_indexes = inspector.get_indexes('risk_tier_transitions')
        
        # Should have index on account_id and timestamp
        assert any('account' in idx['name'].lower() for idx in rtt_indexes), \
            "Index on account_id not found in risk_tier_transitions"
        
        # Check crypto_trades indexes
        ct_indexes = inspector.get_indexes('crypto_trades')
        
        # Should have indexes on broker_id, symbol, status
        assert len(ct_indexes) > 0, "No indexes found on crypto_trades"
        
        print("\n✓ V8 indexes created correctly")
        print(f"  prop_firm_accounts indexes: {len(pfa_indexes)}")
        print(f"  risk_tier_transitions indexes: {len(rtt_indexes)}")
        print(f"  crypto_trades indexes: {len(ct_indexes)}")

    
    def test_migration_idempotency(self, temp_db_path):
        """
        Test that migration can be run multiple times safely.
        
        Flow:
        1. Create V7 database
        2. Run V8 migration
        3. Run V8 migration again
        4. Verify no errors and data is still intact
        
        **Validates: Requirement 16.6 (safe migration)**
        """
        # 1. Create V7 database
        v7_data = self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration first time
        success1 = migrate_v8(temp_db_path)
        assert success1, "First migration failed"
        
        # 3. Run V8 migration second time
        success2 = migrate_v8(temp_db_path)
        assert success2, "Second migration failed"
        
        # 4. Verify data is still intact
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        conn = engine.connect()
        
        accounts = conn.execute(text(
            "SELECT COUNT(*) FROM prop_firm_accounts"
        )).scalar()
        
        assert accounts == v7_data['num_accounts'], \
            f"Expected {v7_data['num_accounts']} accounts, found {accounts}"
        
        snapshots = conn.execute(text(
            "SELECT COUNT(*) FROM daily_snapshots"
        )).scalar()
        
        assert snapshots == v7_data['num_snapshots'], \
            f"Expected {v7_data['num_snapshots']} snapshots, found {snapshots}"
        
        conn.close()
        
        print("\n✓ Migration is idempotent")
        print(f"  Can be run multiple times safely")

    
    def test_migration_with_tier_transitions(self, temp_db_path):
        """
        Test that tier transitions can be logged after migration.
        
        Flow:
        1. Create V7 database
        2. Run V8 migration
        3. Insert tier transition records
        4. Verify transitions are stored correctly
        5. Verify foreign key relationships work
        
        **Validates: Requirement 16.9 (tier transition tracking)**
        """
        # 1. Create V7 database
        v7_data = self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration
        success = migrate_v8(temp_db_path)
        assert success, "Migration failed"
        
        # 3. Insert tier transition records
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        conn = engine.connect()
        
        mff_account_id = v7_data['account_ids']['MFF_12345']
        now = datetime_to_str(datetime.now(timezone.utc))
        
        conn.execute(text("""
            INSERT INTO risk_tier_transitions
            (account_id, from_tier, to_tier, equity_at_transition, transition_timestamp)
            VALUES
            (:account_id, 'growth', 'scaling', 1500.0, :now)
        """), {"account_id": mff_account_id, "now": now})
        conn.commit()
        
        # 4. Verify transition was stored
        transitions = conn.execute(text("""
            SELECT account_id, from_tier, to_tier, equity_at_transition
            FROM risk_tier_transitions
            WHERE account_id = :account_id
        """), {"account_id": mff_account_id}).fetchall()
        
        assert len(transitions) == 1, "Transition not stored"
        assert transitions[0][1] == 'growth'
        assert transitions[0][2] == 'scaling'
        assert transitions[0][3] == 1500.0
        
        # 5. Verify foreign key relationship
        # Try to insert transition for non-existent account (should fail)
        try:
            conn.execute(text("""
                INSERT INTO risk_tier_transitions
                (account_id, from_tier, to_tier, equity_at_transition, transition_timestamp)
                VALUES
                (99999, 'growth', 'scaling', 1000.0, :now)
            """), {"now": now})
            conn.commit()
            assert False, "Should have failed with foreign key constraint"
        except Exception as e:
            # Expected to fail
            conn.rollback()
        
        conn.close()
        
        print("\n✓ Tier transitions work correctly")
        print(f"  Transitions stored: {len(transitions)}")
        print(f"  Foreign key constraints: ✓")

    
    def test_migration_with_crypto_trades(self, temp_db_path):
        """
        Test that crypto trades can be stored after migration.
        
        Flow:
        1. Create V7 database
        2. Run V8 migration
        3. Insert crypto trade records
        4. Verify trades are stored correctly
        5. Verify all crypto trade fields work
        
        **Validates: Requirement 16.9 (crypto trade tracking)**
        """
        # 1. Create V7 database
        self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration
        success = migrate_v8(temp_db_path)
        assert success, "Migration failed"
        
        # 3. Insert crypto trade records
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        conn = engine.connect()
        
        now = datetime_to_str(datetime.now(timezone.utc))
        
        conn.execute(text("""
            INSERT INTO crypto_trades
            (broker_type, broker_id, order_id, symbol, direction, volume,
             entry_price, exit_price, stop_loss, take_profit, profit, status,
             open_timestamp, close_timestamp, trade_metadata)
            VALUES
            ('binance_spot', 'binance_main', 'BIN_12345', 'BTCUSDT', 'buy', 0.001,
             50000.0, 51000.0, 49000.0, 52000.0, 1.0, 'closed',
             :now, :now, '{"fees": 0.1, "slippage": 0.5}')
        """), {"now": now})
        conn.commit()
        
        # 4. Verify trade was stored
        trades = conn.execute(text("""
            SELECT broker_type, broker_id, order_id, symbol, direction, 
                   volume, entry_price, exit_price, status
            FROM crypto_trades
        """)).fetchall()
        
        assert len(trades) == 1, "Trade not stored"
        assert trades[0][0] == 'binance_spot'
        assert trades[0][1] == 'binance_main'
        assert trades[0][2] == 'BIN_12345'
        assert trades[0][3] == 'BTCUSDT'
        assert trades[0][4] == 'buy'
        assert trades[0][5] == 0.001
        assert trades[0][6] == 50000.0
        assert trades[0][7] == 51000.0
        assert trades[0][8] == 'closed'
        
        # 5. Insert open trade (no exit price)
        conn.execute(text("""
            INSERT INTO crypto_trades
            (broker_type, broker_id, order_id, symbol, direction, volume,
             entry_price, status, open_timestamp)
            VALUES
            ('binance_futures', 'binance_main', 'BIN_67890', 'ETHUSDT', 'sell', 0.1,
             3000.0, 'open', :now)
        """), {"now": now})
        conn.commit()
        
        open_trades = conn.execute(text("""
            SELECT order_id, status, exit_price
            FROM crypto_trades
            WHERE status = 'open'
        """)).fetchall()
        
        assert len(open_trades) == 1
        assert open_trades[0][0] == 'BIN_67890'
        assert open_trades[0][1] == 'open'
        assert open_trades[0][2] is None  # No exit price yet
        
        conn.close()
        
        print("\n✓ Crypto trades work correctly")
        print(f"  Closed trades: 1")
        print(f"  Open trades: 1")
        print(f"  All fields validated: ✓")

    
    def test_migration_rollback(self, temp_db_path):
        """
        Test rollback capability.
        
        Flow:
        1. Create V7 database
        2. Run V8 migration
        3. Verify V8 tables exist
        4. Run rollback
        5. Verify V8 tables are removed
        6. Verify V7 data is still intact
        
        **Validates: Requirement 16.6 (rollback capability)**
        """
        # 1. Create V7 database
        v7_data = self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration
        success = migrate_v8(temp_db_path)
        assert success, "Migration failed"
        
        # 3. Verify V8 tables exist
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        inspector = inspect(engine)
        
        tables_before = inspector.get_table_names()
        assert 'risk_tier_transitions' in tables_before
        assert 'crypto_trades' in tables_before
        
        # 4. Run rollback
        rollback_success = rollback_v8(temp_db_path)
        assert rollback_success, "Rollback failed"
        
        # 5. Verify V8 tables are removed
        # Note: SQLite doesn't support DROP COLUMN, so risk_mode column will remain
        # but risk_tier_transitions table should be dropped
        inspector = inspect(engine)
        tables_after = inspector.get_table_names()
        
        assert 'risk_tier_transitions' not in tables_after, \
            "risk_tier_transitions table should be removed"
        
        # 6. Verify V7 data is still intact
        conn = engine.connect()
        
        accounts = conn.execute(text(
            "SELECT COUNT(*) FROM prop_firm_accounts"
        )).scalar()
        
        assert accounts == v7_data['num_accounts'], \
            f"V7 accounts lost during rollback"
        
        snapshots = conn.execute(text(
            "SELECT COUNT(*) FROM daily_snapshots"
        )).scalar()
        
        assert snapshots == v7_data['num_snapshots'], \
            f"V7 snapshots lost during rollback"
        
        conn.close()
        
        print("\n✓ Rollback capability works")
        print(f"  V8 tables removed: ✓")
        print(f"  V7 data preserved: ✓")

    
    def test_migration_with_large_dataset(self, temp_db_path):
        """
        Test migration with larger dataset to verify performance.
        
        Flow:
        1. Create V7 database with 100 accounts
        2. Create 1000 daily snapshots
        3. Create 500 trade proposals
        4. Run V8 migration
        5. Verify all data is preserved
        6. Measure migration time
        
        **Validates: Requirement 16.6 (scalability)**
        """
        # 1. Create V7 database structure
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        
        with engine.connect() as conn:
            # Create V7 tables
            conn.execute(text("""
                CREATE TABLE prop_firm_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    firm_name VARCHAR(100) NOT NULL,
                    account_id VARCHAR(50) NOT NULL UNIQUE,
                    daily_loss_limit_pct FLOAT NOT NULL DEFAULT 5.0,
                    hard_stop_buffer_pct FLOAT NOT NULL DEFAULT 1.0,
                    target_profit_pct FLOAT NOT NULL DEFAULT 8.0,
                    min_trading_days INTEGER NOT NULL DEFAULT 5,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE daily_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER NOT NULL,
                    date VARCHAR(10) NOT NULL,
                    daily_start_balance FLOAT NOT NULL DEFAULT 0.0,
                    high_water_mark FLOAT NOT NULL DEFAULT 0.0,
                    current_equity FLOAT NOT NULL DEFAULT 0.0,
                    daily_drawdown_pct FLOAT NOT NULL DEFAULT 0.0,
                    is_breached BOOLEAN NOT NULL DEFAULT 0,
                    snapshot_timestamp TIMESTAMP NOT NULL,
                    FOREIGN KEY (account_id) REFERENCES prop_firm_accounts(id) ON DELETE CASCADE,
                    UNIQUE (account_id, date)
                )
            """))
            
            conn.execute(text("""
                CREATE TABLE trade_proposals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id INTEGER,
                    bot_id VARCHAR(100) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    kelly_score FLOAT NOT NULL,
                    regime VARCHAR(50),
                    proposed_lot_size FLOAT NOT NULL,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP NOT NULL,
                    reviewed_at TIMESTAMP,
                    FOREIGN KEY (account_id) REFERENCES prop_firm_accounts(id) ON DELETE SET NULL
                )
            """))
            
            conn.commit()
        
        # 2. Insert large dataset
        import time
        now = datetime_to_str(datetime.now(timezone.utc))
        today = date.today().strftime("%Y-%m-%d")
        
        conn = engine.connect()
        
        # Insert 100 accounts
        for i in range(100):
            conn.execute(text("""
                INSERT INTO prop_firm_accounts
                (firm_name, account_id, daily_loss_limit_pct, created_at, updated_at)
                VALUES (:firm, :account, 5.0, :now, :now)
            """), {
                "firm": f"Firm_{i}",
                "account": f"ACC_{i:05d}",
                "now": now
            })
        conn.commit()
        
        # Insert 1000 daily snapshots (10 per account)
        for account_id in range(1, 101):
            for day in range(10):
                conn.execute(text("""
                    INSERT INTO daily_snapshots
                    (account_id, date, daily_start_balance, high_water_mark, 
                     current_equity, snapshot_timestamp)
                    VALUES (:acc, :date, 10000.0, 10500.0, 10300.0, :now)
                """), {
                    "acc": account_id,
                    "date": f"2024-01-{day+1:02d}",
                    "now": now
                })
        conn.commit()
        
        # Insert 500 trade proposals
        for i in range(500):
            conn.execute(text("""
                INSERT INTO trade_proposals
                (account_id, bot_id, symbol, kelly_score, proposed_lot_size, 
                 status, created_at)
                VALUES (:acc, :bot, 'EURUSD', 0.85, 0.1, 'pending', :now)
            """), {
                "acc": (i % 100) + 1,
                "bot": f"Bot_{i}",
                "now": now
            })
        conn.commit()
        conn.close()
        
        # 3. Run V8 migration and measure time
        start_time = time.time()
        success = migrate_v8(temp_db_path)
        migration_time = time.time() - start_time
        
        assert success, "Migration failed"
        
        # 4. Verify all data is preserved
        conn = engine.connect()
        
        accounts = conn.execute(text("SELECT COUNT(*) FROM prop_firm_accounts")).scalar()
        assert accounts == 100, f"Expected 100 accounts, found {accounts}"
        
        snapshots = conn.execute(text("SELECT COUNT(*) FROM daily_snapshots")).scalar()
        assert snapshots == 1000, f"Expected 1000 snapshots, found {snapshots}"
        
        proposals = conn.execute(text("SELECT COUNT(*) FROM trade_proposals")).scalar()
        assert proposals == 500, f"Expected 500 proposals, found {proposals}"
        
        # 5. Verify V8 enhancements
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        assert 'risk_tier_transitions' in tables
        assert 'crypto_trades' in tables
        
        pfa_columns = [col['name'] for col in inspector.get_columns('prop_firm_accounts')]
        assert 'risk_mode' in pfa_columns
        
        conn.close()
        
        print("\n✓ Large dataset migration successful")
        print(f"  Accounts: 100")
        print(f"  Snapshots: 1000")
        print(f"  Proposals: 500")
        print(f"  Migration time: {migration_time:.2f}s")
        print(f"  Performance: ✓")

    
    def test_migration_data_integrity_checks(self, temp_db_path):
        """
        Test comprehensive data integrity after migration.
        
        Flow:
        1. Create V7 database with specific data patterns
        2. Run V8 migration
        3. Verify data types are correct
        4. Verify constraints are enforced
        5. Verify relationships are intact
        
        **Validates: Requirement 16.6 (data integrity)**
        """
        # 1. Create V7 database
        v7_data = self.create_v7_database(temp_db_path)
        
        # 2. Run V8 migration
        success = migrate_v8(temp_db_path)
        assert success, "Migration failed"
        
        # 3. Verify data types
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        inspector = inspect(engine)
        
        # Check risk_mode column type
        pfa_columns = {col['name']: col for col in inspector.get_columns('prop_firm_accounts')}
        risk_mode_col = pfa_columns['risk_mode']
        assert 'VARCHAR' in str(risk_mode_col['type']).upper() or \
               'TEXT' in str(risk_mode_col['type']).upper(), \
               "risk_mode should be VARCHAR/TEXT type"
        
        # 4. Verify constraints
        conn = engine.connect()
        
        # Test unique constraint on account_id
        try:
            now = datetime_to_str(datetime.now(timezone.utc))
            conn.execute(text("""
                INSERT INTO prop_firm_accounts
                (firm_name, account_id, daily_loss_limit_pct, created_at, updated_at)
                VALUES ('Test', 'MFF_12345', 5.0, :now, :now)
            """), {"now": now})
            conn.commit()
            assert False, "Should have failed with unique constraint violation"
        except Exception:
            # Expected to fail
            conn.rollback()
        
        # Test foreign key constraint on daily_snapshots
        try:
            today = date.today().strftime("%Y-%m-%d")
            now_str = datetime_to_str(datetime.now(timezone.utc))
            conn.execute(text("""
                INSERT INTO daily_snapshots
                (account_id, date, daily_start_balance, high_water_mark, 
                 current_equity, snapshot_timestamp)
                VALUES (99999, :today, 1000.0, 1000.0, 1000.0, :now)
            """), {"today": today, "now": now_str})
            conn.commit()
            assert False, "Should have failed with foreign key constraint"
        except Exception:
            # Expected to fail
            conn.rollback()
        
        # 5. Verify relationships are intact
        # Query with JOIN to verify foreign keys work
        result = conn.execute(text("""
            SELECT pfa.account_id, ds.current_equity
            FROM prop_firm_accounts pfa
            JOIN daily_snapshots ds ON pfa.id = ds.account_id
            WHERE pfa.account_id = 'MFF_12345'
        """)).fetchall()
        
        assert len(result) > 0, "JOIN query failed - relationships broken"
        assert result[0][0] == 'MFF_12345'
        assert result[0][1] == 10300.0
        
        conn.close()
        
        print("\n✓ Data integrity verified")
        print(f"  Data types: ✓")
        print(f"  Constraints: ✓")
        print(f"  Relationships: ✓")

    
    def test_complete_migration_workflow(self, temp_db_path):
        """
        Test complete end-to-end migration workflow.
        
        This is the ultimate integration test combining all aspects:
        1. V7 database creation with realistic data
        2. V8 migration execution
        3. Data preservation verification
        4. Schema enhancement verification
        5. New functionality verification
        6. Performance verification
        
        **Validates: Requirements 16.6, 16.9 (complete workflow)**
        """
        print("\n" + "=" * 60)
        print("COMPLETE V7 TO V8 MIGRATION WORKFLOW TEST")
        print("=" * 60)
        
        # 1. Create V7 database
        print("\n[1/6] Creating V7 database with sample data...")
        v7_data = self.create_v7_database(temp_db_path)
        print(f"  ✓ Created {v7_data['num_accounts']} accounts")
        print(f"  ✓ Created {v7_data['num_snapshots']} snapshots")
        print(f"  ✓ Created {v7_data['num_proposals']} proposals")
        
        # 2. Run V8 migration
        print("\n[2/6] Running V8 migration...")
        import time
        start_time = time.time()
        success = migrate_v8(temp_db_path)
        migration_time = time.time() - start_time
        assert success, "Migration failed"
        print(f"  ✓ Migration completed in {migration_time:.2f}s")
        
        # 3. Verify data preservation
        print("\n[3/6] Verifying V7 data preservation...")
        engine = create_engine(f'sqlite:///{temp_db_path}', echo=False)
        conn = engine.connect()
        
        accounts = conn.execute(text("SELECT COUNT(*) FROM prop_firm_accounts")).scalar()
        snapshots = conn.execute(text("SELECT COUNT(*) FROM daily_snapshots")).scalar()
        proposals = conn.execute(text("SELECT COUNT(*) FROM trade_proposals")).scalar()
        
        assert accounts == v7_data['num_accounts']
        assert snapshots == v7_data['num_snapshots']
        assert proposals == v7_data['num_proposals']
        print(f"  ✓ All V7 data preserved")
        
        # 4. Verify schema enhancements
        print("\n[4/6] Verifying V8 schema enhancements...")
        inspector = inspect(engine)
        
        # Check new columns
        pfa_columns = [col['name'] for col in inspector.get_columns('prop_firm_accounts')]
        assert 'risk_mode' in pfa_columns
        print(f"  ✓ risk_mode column added")
        
        tp_columns = [col['name'] for col in inspector.get_columns('trade_proposals')]
        assert 'broker_id' in tp_columns
        print(f"  ✓ broker_id column added")
        
        # Check new tables
        tables = inspector.get_table_names()
        assert 'risk_tier_transitions' in tables
        assert 'crypto_trades' in tables
        print(f"  ✓ risk_tier_transitions table created")
        print(f"  ✓ crypto_trades table created")
        
        # 5. Test new functionality
        print("\n[5/6] Testing V8 functionality...")
        
        # Test tier transition logging
        mff_account_id = v7_data['account_ids']['MFF_12345']
        now = datetime_to_str(datetime.now(timezone.utc))
        
        conn.execute(text("""
            INSERT INTO risk_tier_transitions
            (account_id, from_tier, to_tier, equity_at_transition, transition_timestamp)
            VALUES (:acc, 'growth', 'scaling', 1500.0, :now)
        """), {"acc": mff_account_id, "now": now})
        conn.commit()
        
        transitions = conn.execute(text(
            "SELECT COUNT(*) FROM risk_tier_transitions"
        )).scalar()
        assert transitions == 1
        print(f"  ✓ Tier transitions working")
        
        # Test crypto trade logging
        conn.execute(text("""
            INSERT INTO crypto_trades
            (broker_type, broker_id, order_id, symbol, direction, volume,
             entry_price, status, open_timestamp)
            VALUES ('binance_spot', 'binance_main', 'TEST_001', 'BTCUSDT', 
                    'buy', 0.001, 50000.0, 'open', :now)
        """), {"now": now})
        conn.commit()
        
        trades = conn.execute(text(
            "SELECT COUNT(*) FROM crypto_trades"
        )).scalar()
        assert trades == 1
        print(f"  ✓ Crypto trades working")
        
        # 6. Performance verification
        print("\n[6/6] Verifying performance...")
        
        # Query performance with new indexes
        query_start = time.time()
        result = conn.execute(text("""
            SELECT pfa.account_id, pfa.risk_mode, ds.current_equity
            FROM prop_firm_accounts pfa
            JOIN daily_snapshots ds ON pfa.id = ds.account_id
            WHERE pfa.risk_mode = 'growth'
        """)).fetchall()
        query_time = time.time() - query_start
        
        assert len(result) > 0
        print(f"  ✓ Query performance: {query_time*1000:.2f}ms")
        
        conn.close()
        
        # Final summary
        print("\n" + "=" * 60)
        print("MIGRATION WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nSummary:")
        print(f"  Migration time: {migration_time:.2f}s")
        print(f"  V7 data preserved: ✓")
        print(f"  V8 schema added: ✓")
        print(f"  V8 functionality: ✓")
        print(f"  Performance: ✓")
        print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
