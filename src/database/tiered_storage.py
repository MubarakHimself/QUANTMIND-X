"""
Tiered Storage Router
=====================

Data access layer that routes queries to the appropriate storage tier:
- HOT: PostgreSQL (current tick data, < 1 hour)
- WARM: DuckDB (recent data, < 30 days)
- COLD: Parquet files (historical data, > 30 days)

Usage:
    from src.database.tiered_storage import get_tiered_router, TieredStorageRouter
    
    router = get_tiered_router()
    df = router.query(symbol="EURUSD", timeframe="H1", start_dt="2025-01-01", end_dt="2026-01-01")
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class TieredStorageRouter:
    """
    Routes queries to appropriate storage tier based on time range.
    
    Tiers:
    - HOT: PostgreSQL tick_cache (current data, < 1 hour old)
    - WARM: DuckDB market_data (recent data, < 30 days old)
    - COLD: Parquet files (historical data, > 30 days old)
    """
    
    # Tier thresholds (in days)
    HOT_THRESHOLD_HOURS = 1
    WARM_THRESHOLD_DAYS = 30
    
    def __init__(self):
        """Initialize tiered storage router with configuration."""
        # HOT tier
        self._hot_db_url = os.environ.get("HOT_DB_URL") or os.environ.get("CLOUDZY_HOT_DB_URL")
        
        # WARM tier
        self._warm_db_path = os.environ.get("WARM_DB_PATH", "/data/market_data.duckdb")
        
        # COLD tier
        self._cold_storage_path = os.environ.get("COLD_STORAGE_PATH", "/data/cold_storage")
        
        logger.info(f"TieredStorageRouter initialized:")
        logger.info(f"  HOT: {self._hot_db_url or 'not configured'}")
        logger.info(f"  WARM: {self._warm_db_path}")
        logger.info(f"  COLD: {self._cold_storage_path}")
    
    def _query_hot_tier(
        self,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Query HOT tier (PostgreSQL tick_cache).
        
        Args:
            symbol: Trading symbol
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            DataFrame with tick data
        """
        if not self._hot_db_url:
            logger.warning("HOT tier not configured")
            return pd.DataFrame()
        
        from sqlalchemy import create_engine, text
        
        try:
            engine = create_engine(self._hot_db_url)
            
            query = text("""
                SELECT 
                    symbol,
                    timestamp,
                    bid,
                    ask,
                    volume
                FROM tick_cache
                WHERE symbol = :symbol
                AND timestamp >= :start_dt
                AND timestamp <= :end_dt
                ORDER BY timestamp ASC
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query, {
                    "symbol": symbol,
                    "start_dt": start_dt,
                    "end_dt": end_dt
                })
                
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                
                if not df.empty:
                    # Convert to OHLCV
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                    df = df.resample('1min').agg({
                        'bid': 'ohlc',
                        'ask': 'ohlc',
                        'volume': 'sum'
                    })
                    df.columns = ['open', 'high', 'low', 'close', 'volume']
                    df = df.reset_index()
                
                return df
                
        except Exception as e:
            logger.error(f"Failed to query HOT tier: {e}")
            return pd.DataFrame()
    
    def _query_warm_tier(
        self,
        symbol: str,
        timeframe: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Query WARM tier (DuckDB market_data).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, H1, etc.)
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            DataFrame with market data
        """
        from src.database.duckdb_connection import DuckDBConnection
        
        try:
            with DuckDBConnection(db_path=self._warm_db_path) as conn:
                query = """
                    SELECT * FROM market_data
                    WHERE symbol = :symbol
                    AND timeframe = :timeframe
                    AND timestamp >= :start_dt
                    AND timestamp <= :end_dt
                    ORDER BY timestamp ASC
                """
                
                result = conn.execute_query(query, {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "start_dt": start_dt,
                    "end_dt": end_dt
                })
                
                return result.df()
                
        except Exception as e:
            logger.error(f"Failed to query WARM tier: {e}")
            return pd.DataFrame()
    
    def _query_cold_tier(
        self,
        symbol: str,
        timeframe: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Query COLD tier (Parquet files).
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (M1, M5, H1, etc.)
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            DataFrame with historical data
        """
        cold_path = Path(self._cold_storage_path) / symbol
        
        if not cold_path.exists():
            logger.warning(f"COLD tier path does not exist: {cold_path}")
            return pd.DataFrame()
        
        try:
            # Find all parquet files in date range
            parquet_files = []
            
            start_year = start_dt.year
            end_year = end_dt.year
            
            for year in range(start_year, end_year + 1):
                year_path = cold_path / str(year)
                if not year_path.exists():
                    continue
                
                for month in range(1, 13):
                    if year == start_year and month < start_dt.month:
                        continue
                    if year == end_year and month > end_dt.month:
                        continue
                    
                    month_path = year_path / f"{month:02d}"
                    if not month_path.exists():
                        continue
                    
                    for day_path in month_path.iterdir():
                        if not day_path.is_dir():
                            continue
                        
                        day = int(day_path.name)
                        if year == start_year and month == start_dt.day and day < start_dt.day:
                            continue
                        if year == end_year and month == end_dt.day and day > end_dt.day:
                            continue
                        
                        parquet_file = day_path / "data.parquet"
                        if parquet_file.exists():
                            parquet_files.append(str(parquet_file))
            
            if not parquet_files:
                return pd.DataFrame()
            
            # Read and concatenate all parquet files
            dfs = []
            for pf in parquet_files:
                df = pd.read_parquet(pf)
                dfs.append(df)
            
            result = pd.concat(dfs, ignore_index=True)
            
            # Filter by timeframe
            if timeframe:
                result = result[result['timeframe'] == timeframe]
            
            # Filter by date range
            result['timestamp'] = pd.to_datetime(result['timestamp'])
            result = result[
                (result['timestamp'] >= start_dt) &
                (result['timestamp'] <= end_dt)
            ]
            
            return result.sort_values('timestamp')
            
        except Exception as e:
            logger.error(f"Failed to query COLD tier: {e}")
            return pd.DataFrame()
    
    def query(
        self,
        symbol: str,
        timeframe: str = "H1",
        start_dt: Optional[str] = None,
        end_dt: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Query data from tiered storage.
        
        Automatically routes to appropriate tier based on time range:
        - end_dt > now - 1h: HOT tier (PostgreSQL)
        - end_dt > now - 30d: WARM tier (DuckDB)
        - Otherwise: COLD tier (Parquet)
        
        For queries spanning multiple tiers, fetches from each and concatenates.
        
        Args:
            symbol: Trading symbol (e.g., EURUSD)
            timeframe: Timeframe (M1, M5, M15, H1, H4, D1)
            start_dt: Start date (YYYY-MM-DD or ISO datetime)
            end_dt: End date (YYYY-MM-DD or ISO datetime)
            
        Returns:
            DataFrame with market data
        """
        # Parse dates
        now = datetime.now(timezone.utc)
        
        if end_dt is None:
            end_dt = now
        elif isinstance(end_dt, str):
            end_dt = pd.to_datetime(end_dt)
        
        if start_dt is None:
            start_dt = end_dt - timedelta(days=365)  # Default to 1 year
        elif isinstance(start_dt, str):
            start_dt = pd.to_datetime(start_dt)
        
        # Determine which tiers to query
        use_hot = (now - end_dt).total_seconds() < (self.HOT_THRESHOLD_HOURS * 3600)
        use_warm = (now - end_dt).total_seconds() < (self.WARM_THRESHOLD_DAYS * 3600 * 24)
        
        results = []
        
        # Query HOT tier if needed
        if use_hot:
            logger.info(f"Querying HOT tier for {symbol}")
            hot_df = self._query_hot_tier(symbol, start_dt, end_dt)
            if not hot_df.empty:
                results.append(hot_df)
        
        # Query WARM tier if needed
        if use_warm:
            logger.info(f"Querying WARM tier for {symbol} {timeframe}")
            warm_df = self._query_warm_tier(symbol, timeframe, start_dt, end_dt)
            if not warm_df.empty:
                results.append(warm_df)
        
        # Query COLD tier (always check for historical data)
        logger.info(f"Querying COLD tier for {symbol} {timeframe}")
        cold_df = self._query_cold_tier(symbol, timeframe, start_dt, end_dt)
        if not cold_df.empty:
            results.append(cold_df)
        
        if not results:
            logger.warning(f"No data found for {symbol} {timeframe}")
            return pd.DataFrame()
        
        # Concatenate and deduplicate
        combined = pd.concat(results, ignore_index=True)
        
        # Remove duplicates based on timestamp
        if 'timestamp' in combined.columns:
            combined = combined.drop_duplicates(subset=['timestamp', 'symbol', 'timeframe'], keep='first')
            combined = combined.sort_values('timestamp')
        
        logger.info(f"Retrieved {len(combined)} rows for {symbol} {timeframe}")
        
        return combined


# ============= Singleton Factory =============


_tiered_router: Optional[TieredStorageRouter] = None


def get_tiered_router() -> TieredStorageRouter:
    """
    Get singleton instance of TieredStorageRouter.
    
    Returns:
        TieredStorageRouter instance
    """
    global _tiered_router
    
    if _tiered_router is None:
        _tiered_router = TieredStorageRouter()
    
    return _tiered_router
