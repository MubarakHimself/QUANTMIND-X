"""
Warm Storage Module

Persists 20-session rolling average volume profile to DuckDB.
Used for RVOL calculation (loading historical volume data at time-of-day).

Storage schema:
    volume_profile_history (
        symbol TEXT,
        session_id TEXT,
        bucket_minute INTEGER,  -- 0-1439 (minutes since midnight GMT)
        avg_volume REAL,
        updated_at TIMESTAMP UTC
    )
"""

import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

import duckdb

logger = logging.getLogger(__name__)


class WarmStorage:
    """
    Warm storage for SVSS volume profile history.

    Persists 20-session rolling average volume profile for RVOL calculation.
    """

    # Minutes in a day
    MINUTES_PER_DAY = 1440

    def __init__(self, db_path: str = "data/svss/warm_storage.db"):
        """
        Initialize warm storage.

        Args:
            db_path: Path to DuckDB database file
        """
        self._db_path = db_path
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
        self._ensure_storage_dir()
        self._connect()
        self._init_schema()

    def _ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        db_dir = os.path.dirname(self._db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

    def _connect(self) -> None:
        """Establish connection to DuckDB."""
        try:
            self._conn = duckdb.connect(self._db_path)
            logger.info(f"Warm storage connected to {self._db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to warm storage: {e}")
            raise

    def _init_schema(self) -> None:
        """Initialize storage schema if not exists."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS volume_profile_history (
            symbol TEXT,
            session_id TEXT,
            bucket_minute INTEGER,
            avg_volume REAL,
            updated_at TIMESTAMP,
            PRIMARY KEY (symbol, session_id, bucket_minute)
        );
        """
        try:
            self._conn.execute(create_table_sql)
        except Exception as e:
            logger.warning(f"Schema creation warning (may already exist): {e}")

    def load_rolling_avg_profile(
        self,
        symbol: str,
        num_sessions: int = 20
    ) -> Dict[int, float]:
        """
        Load 20-session rolling average volume profile for RVOL calculation.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            num_sessions: Number of sessions to average (default: 20)

        Returns:
            Dict mapping minute-of-day (0-1439) to average volume
        """
        # Get the last N sessions for this symbol
        query = """
        WITH session_volumes AS (
            SELECT
                session_id,
                bucket_minute,
                avg_volume,
                ROW_NUMBER() OVER (PARTITION BY bucket_minute ORDER BY updated_at DESC) as rn
            FROM volume_profile_history
            WHERE symbol = ?
        ),
        recent_sessions AS (
            SELECT DISTINCT session_id
            FROM volume_profile_history
            WHERE symbol = ?
            ORDER BY session_id DESC
            LIMIT ?
        )
        SELECT
            sv.bucket_minute,
            AVG(sv.avg_volume) as rolling_avg_volume
        FROM session_volumes sv
        JOIN recent_sessions rs ON sv.session_id = rs.session_id
        WHERE sv.rn = 1
        GROUP BY sv.bucket_minute
        ORDER BY sv.bucket_minute;
        """

        try:
            result = self._conn.execute(query, [symbol.upper(), symbol.upper(), num_sessions])
            rows = result.fetchall()

            profile: Dict[int, float] = {}
            for bucket_minute, avg_volume in rows:
                profile[int(bucket_minute)] = float(avg_volume)

            # Fill missing minutes with 0
            for minute in range(self.MINUTES_PER_DAY):
                if minute not in profile:
                    profile[minute] = 0.0

            logger.debug(
                f"Loaded rolling avg profile for {symbol}: {len(profile)} buckets"
            )
            return profile

        except Exception as e:
            logger.error(f"Failed to load rolling avg profile: {e}")
            return {minute: 0.0 for minute in range(self.MINUTES_PER_DAY)}

    def save_session_profile(
        self,
        symbol: str,
        session_id: str,
        volume_profile: Dict[int, float]
    ) -> bool:
        """
        Save current session's volume profile to warm storage.

        Called on session close to persist data for future RVOL calculations.

        Args:
            symbol: Trading symbol
            session_id: Session identifier
            volume_profile: Dict mapping minute-of-day to volume

        Returns:
            True if saved successfully, False otherwise.
        """
        if not volume_profile:
            logger.warning(f"No volume profile to save for {session_id}")
            return False

        try:
            # Delete existing entry for this session (upsert pattern)
            delete_sql = "DELETE FROM volume_profile_history WHERE symbol = ? AND session_id = ?"
            self._conn.execute(delete_sql, [symbol.upper(), session_id])

            # Insert new profile
            insert_sql = """
            INSERT INTO volume_profile_history (symbol, session_id, bucket_minute, avg_volume, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """
            timestamp = datetime.now(timezone.utc)

            for minute, volume in volume_profile.items():
                self._conn.execute(
                    insert_sql,
                    [symbol.upper(), session_id, minute, volume, timestamp]
                )

            logger.info(f"Saved volume profile for {symbol} session {session_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save session profile: {e}")
            return False

    def get_latest_session_id(self, symbol: str) -> Optional[str]:
        """
        Get the most recent session ID for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Most recent session ID or None if no history.
        """
        query = """
        SELECT session_id
        FROM volume_profile_history
        WHERE symbol = ?
        ORDER BY updated_at DESC
        LIMIT 1;
        """
        try:
            result = self._conn.execute(query, [symbol.upper()])
            row = result.fetchone()
            return row[0] if row else None
        except Exception as e:
            logger.error(f"Failed to get latest session: {e}")
            return None

    def close(self) -> None:
        """Close storage connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Warm storage closed")
