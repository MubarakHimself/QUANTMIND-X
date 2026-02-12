"""
DataManager - Hybrid data fetching with Parquet caching.

Task Group 1: Data Manager Implementation

This module implements a comprehensive data management system with:
- Hybrid data fetching: MT5 -> API -> Cache fallback chain
- Parquet caching with symbol/timeframe organization
- Data validation: missing bars, duplicates, price anomalies
- Cache metadata tracking (last_update, data_quality, source)
- Multi-timeframe support (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
- CSV upload and conversion to Parquet

Reuses patterns from: src/backtesting/mt5_engine.py (MQL5Timeframe, MT5 integration)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
    logging.warning("MetaTrader5 package not available. MT5 data fetching disabled.")

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    pa = None
    pq = None
    logging.warning("PyArrow not available. Parquet caching disabled.")

logger = logging.getLogger(__name__)


# =============================================================================
# Data Source Enumeration
# =============================================================================

class DataSource(Enum):
    """Data source enumeration for tracking where data originated."""
    MT5 = "mt5"
    API = "api"
    CACHE = "cache"
    UPLOAD = "upload"


# =============================================================================
# MQL5 Timeframe Constants (reused from mt5_engine.py)
# =============================================================================

class MQL5Timeframe:
    """MQL5 timeframe constants matching MT5 specification.

    Reused from src.backtesting.mt5_engine.MQL5Timeframe
    """

    PERIOD_M1 = 16385   # 1 minute
    PERIOD_M5 = 16386   # 5 minutes
    PERIOD_M15 = 16387  # 15 minutes
    PERIOD_M30 = 16388  # 30 minutes
    PERIOD_H1 = 16393   # 1 hour
    PERIOD_H4 = 16396   # 4 hours
    PERIOD_D1 = 16401   # Daily
    PERIOD_W1 = 16405   # Weekly
    PERIOD_MN1 = 16408  # Monthly

    @classmethod
    def to_minutes(cls, timeframe: int) -> int:
        """Convert timeframe constant to minutes."""
        mapping = {
            cls.PERIOD_M1: 1,
            cls.PERIOD_M5: 5,
            cls.PERIOD_M15: 15,
            cls.PERIOD_M30: 30,
            cls.PERIOD_H1: 60,
            cls.PERIOD_H4: 240,
            cls.PERIOD_D1: 1440,
            cls.PERIOD_W1: 10080,
            cls.PERIOD_MN1: 43200,
        }
        return mapping.get(timeframe, 60)

    @classmethod
    def to_pandas_freq(cls, timeframe: int) -> str:
        """Convert timeframe constant to pandas frequency string."""
        mapping = {
            cls.PERIOD_M1: "1min",
            cls.PERIOD_M5: "5min",
            cls.PERIOD_M15: "15min",
            cls.PERIOD_M30: "30min",
            cls.PERIOD_H1: "1h",
            cls.PERIOD_H4: "4h",
            cls.PERIOD_D1: "1D",
            cls.PERIOD_W1: "1W",
            cls.PERIOD_MN1: "1ME",
        }
        return mapping.get(timeframe, "1h")

    @classmethod
    def to_string(cls, timeframe: int) -> str:
        """Convert timeframe constant to string identifier."""
        mapping = {
            cls.PERIOD_M1: "M1",
            cls.PERIOD_M5: "M5",
            cls.PERIOD_M15: "M15",
            cls.PERIOD_M30: "M30",
            cls.PERIOD_H1: "H1",
            cls.PERIOD_H4: "H4",
            cls.PERIOD_D1: "D1",
            cls.PERIOD_W1: "W1",
            cls.PERIOD_MN1: "MN1",
        }
        return mapping.get(timeframe, "H1")

    @classmethod
    def from_string(cls, tf_str: str) -> int:
        """Convert string identifier to timeframe constant."""
        mapping = {
            "M1": cls.PERIOD_M1,
            "M5": cls.PERIOD_M5,
            "M15": cls.PERIOD_M15,
            "M30": cls.PERIOD_M30,
            "H1": cls.PERIOD_H1,
            "H4": cls.PERIOD_H4,
            "D1": cls.PERIOD_D1,
            "W1": cls.PERIOD_W1,
            "MN1": cls.PERIOD_MN1,
        }
        return mapping.get(tf_str.upper(), cls.PERIOD_H1)


# =============================================================================
# Data Quality Report
# =============================================================================

@dataclass
class DataQualityReport:
    """Data quality validation report."""
    total_bars: int
    missing_bars: int
    duplicates: int
    price_anomalies: int
    quality_score: float  # 0.0 to 1.0
    validation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    issues: List[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        """Check if data passes validation."""
        return (
            self.missing_bars == 0
            and self.duplicates == 0
            and self.price_anomalies == 0
            and self.quality_score >= 0.8
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_bars": self.total_bars,
            "missing_bars": self.missing_bars,
            "duplicates": self.duplicates,
            "price_anomalies": self.price_anomalies,
            "quality_score": self.quality_score,
            "is_valid": self.is_valid(),
            "validation_time": self.validation_time.isoformat(),
            "issues": self.issues
        }


# =============================================================================
# Cache Metadata
# =============================================================================

@dataclass
class CacheMetadata:
    """Metadata for cached data."""
    symbol: str
    timeframe: str
    source: DataSource
    last_update: datetime
    quality_score: float
    total_bars: int
    date_range: Tuple[datetime, datetime]
    file_path: Path

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "source": self.source.value,
            "last_update": self.last_update.isoformat(),
            "quality_score": self.quality_score,
            "total_bars": self.total_bars,
            "date_range": (self.date_range[0].isoformat(), self.date_range[1].isoformat()),
            "file_path": str(self.file_path)
        }


# =============================================================================
# CSV Upload Result
# =============================================================================

@dataclass
class CSVUploadResult:
    """Result of CSV upload operation."""
    success: bool
    symbol: str
    timeframe: str
    rows_imported: int
    message: str
    cached_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows_imported": self.rows_imported,
            "message": self.message,
            "cached_path": str(self.cached_path) if self.cached_path else None
        }


# =============================================================================
# DataManager Class
# =============================================================================

class DataManager:
    """Hybrid data fetching manager with Parquet caching.

    Features:
    - MT5 -> API -> Cache fallback chain for data fetching
    - Parquet caching with symbol/timeframe organization
    - Data validation (missing bars, duplicates, price anomalies)
    - Cache metadata tracking
    - Multi-timeframe support
    - CSV upload and conversion to Parquet

    Example:
        >>> manager = DataManager(cache_dir="./data/historical")
        >>> data = manager.fetch_data("EURUSD", MQL5Timeframe.PERIOD_H1, count=1000)
        >>> report = manager.validate_data(data, MQL5Timeframe.PERIOD_H1)
        >>> manager.upload_csv("GBPUSD", MQL5Timeframe.PERIOD_D1, "data.csv")
        >>> status = manager.get_cache_status()
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_mt5: bool = True,
        enable_api: bool = True,
        api_base_url: str = None,
        mt5_login: Optional[int] = None,
        mt5_password: Optional[str] = None,
        mt5_server: Optional[str] = None
    ):
        """Initialize the DataManager.

        Args:
            cache_dir: Directory for Parquet cache (default: ./data/historical)
            enable_mt5: Enable MT5 data fetching
            enable_api: Enable API data fetching
            api_base_url: Base URL for API data source
            mt5_login: MT5 account login
            mt5_password: MT5 account password
            mt5_server: MT5 server name
        """
        # Cache configuration
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./data/historical")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Data source configuration
        self.enable_mt5 = enable_mt5 and MT5_AVAILABLE
        self.enable_api = enable_api
        self.api_base_url = api_base_url

        # MT5 connection
        self._mt5 = mt5 if MT5_AVAILABLE else None
        self._mt5_connected = False

        # Initialize MT5 if credentials provided
        if self.enable_mt5 and mt5_login and mt5_password and mt5_server:
            self._connect_mt5(mt5_login, mt5_password, mt5_server)

        # Metadata tracking
        self._metadata: Dict[str, Dict[str, CacheMetadata]] = {}
        self._load_metadata()

        logger.info(f"DataManager initialized with cache_dir: {self.cache_dir}")

    # -------------------------------------------------------------------------
    # MT5 Integration
    # -------------------------------------------------------------------------

    def _connect_mt5(self, login: int, password: str, server: str) -> bool:
        """Connect to MetaTrader 5 terminal.

        Args:
            login: MT5 account number
            password: MT5 password
            server: MT5 server name

        Returns:
            True if connection successful
        """
        if not self.enable_mt5:
            return False

        try:
            self._mt5_connected = self._mt5.initialize(login=login, password=password, server=server)
            if self._mt5_connected:
                logger.info(f"Connected to MT5: {server} (login: {login})")
            else:
                logger.error(f"MT5 connection failed: {self._mt5.last_error()}")
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            self._mt5_connected = False

        return self._mt5_connected

    def _fetch_from_mt5(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """Fetch data from MetaTrader 5.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            count: Number of bars to retrieve

        Returns:
            DataFrame with OHLCV data or None
        """
        if not self.enable_mt5 or not self._mt5_connected:
            return None

        try:
            # Map MQL5 timeframe to MT5 constant
            mt5_timeframe = self._get_mt5_timeframe(timeframe)

            # Retrieve data
            rates = self._mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)

            if rates is None or len(rates) == 0:
                logger.debug(f"MT5: No data for {symbol} {MQL5Timeframe.to_string(timeframe)}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

            # Rename columns to standard format
            df = df.rename(columns={
                'tick_volume': 'tick_volume',
                'spread': 'spread',
                'real_volume': 'real_volume'
            })

            # Select OHLCV columns
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

            logger.info(f"MT5: Fetched {len(df)} bars for {symbol} {MQL5Timeframe.to_string(timeframe)}")
            return df

        except Exception as e:
            logger.error(f"MT5 fetch error: {e}")
            return None

    def _get_mt5_timeframe(self, mql5_timeframe: int) -> int:
        """Convert MQL5 timeframe constant to MT5 package constant.

        Args:
            mql5_timeframe: MQL5Timeframe constant

        Returns:
            MT5 package timeframe constant
        """
        mapping = {
            MQL5Timeframe.PERIOD_M1: self._mt5.TIMEFRAME_M1,
            MQL5Timeframe.PERIOD_M5: self._mt5.TIMEFRAME_M5,
            MQL5Timeframe.PERIOD_M15: self._mt5.TIMEFRAME_M15,
            MQL5Timeframe.PERIOD_M30: self._mt5.TIMEFRAME_M30,
            MQL5Timeframe.PERIOD_H1: self._mt5.TIMEFRAME_H1,
            MQL5Timeframe.PERIOD_H4: self._mt5.TIMEFRAME_H4,
            MQL5Timeframe.PERIOD_D1: self._mt5.TIMEFRAME_D1,
            MQL5Timeframe.PERIOD_W1: self._mt5.TIMEFRAME_W1,
            MQL5Timeframe.PERIOD_MN1: self._mt5.TIMEFRAME_MN1,
        }
        return mapping.get(mql5_timeframe, self._mt5.TIMEFRAME_H1)

    # -------------------------------------------------------------------------
    # API Data Fetching (placeholder for future implementation)
    # -------------------------------------------------------------------------

    def _fetch_from_api(self, symbol: str, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        """Fetch data from external API.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            count: Number of bars to retrieve

        Returns:
            DataFrame with OHLCV data or None
        """
        # TODO: Implement API data fetching
        # This would connect to external data providers like:
        # - Alpha Vantage
        # - Yahoo Finance
        # - Binance (for crypto)
        # - Custom data APIs
        return None

    # -------------------------------------------------------------------------
    # Cache Operations
    # -------------------------------------------------------------------------

    def get_cached_data(self, symbol: str, timeframe: int) -> pd.DataFrame:
        """Retrieve data from Parquet cache.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant

        Returns:
            DataFrame with cached data or empty DataFrame if not found
        """
        tf_str = MQL5Timeframe.to_string(timeframe)
        cache_path = self.cache_dir / symbol / tf_str / "data.parquet"

        if not cache_path.exists():
            logger.debug(f"Cache miss: {symbol} {tf_str}")
            return pd.DataFrame()

        try:
            if PARQUET_AVAILABLE:
                df = pd.read_parquet(cache_path)
            else:
                # Fallback to CSV if Parquet not available
                csv_path = cache_path.with_suffix('.csv')
                if csv_path.exists():
                    df = pd.read_csv(csv_path, parse_dates=['time'])
                else:
                    return pd.DataFrame()

            logger.info(f"Cache hit: {symbol} {tf_str} ({len(df)} bars)")
            return df

        except Exception as e:
            logger.error(f"Cache read error: {e}")
            return pd.DataFrame()

    def _save_to_cache(self, symbol: str, timeframe: int, data: pd.DataFrame, source: DataSource) -> bool:
        """Save data to Parquet cache.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            data: OHLCV DataFrame
            source: Data source

        Returns:
            True if successful
        """
        if data is None or len(data) == 0:
            return False

        tf_str = MQL5Timeframe.to_string(timeframe)
        cache_path = self.cache_dir / symbol / tf_str
        cache_path.mkdir(parents=True, exist_ok=True)

        parquet_path = cache_path / "data.parquet"

        try:
            # Ensure timezone-aware datetime
            if 'time' in data.columns:
                if data['time'].dt.tz is None:
                    data['time'] = data['time'].dt.tz_localize('utc')

            # Save to Parquet
            if PARQUET_AVAILABLE:
                data.to_parquet(parquet_path, compression='snappy', index=False)
            else:
                # Fallback to CSV
                data.to_csv(parquet_path.with_suffix('.csv'), index=False)

            # Update metadata
            date_range = (
                data['time'].min().to_pydatetime(),
                data['time'].max().to_pydatetime()
            )

            metadata = CacheMetadata(
                symbol=symbol,
                timeframe=tf_str,
                source=source,
                last_update=datetime.now(timezone.utc),
                quality_score=1.0,  # Will be updated after validation
                total_bars=len(data),
                date_range=date_range,
                file_path=parquet_path
            )

            if symbol not in self._metadata:
                self._metadata[symbol] = {}
            self._metadata[symbol][tf_str] = metadata

            self._save_metadata()

            logger.info(f"Cached {len(data)} bars for {symbol} {tf_str} (source: {source.value})")
            return True

        except Exception as e:
            logger.error(f"Cache write error: {e}")
            return False

    def _load_metadata(self):
        """Load cache metadata from disk."""
        metadata_path = self.cache_dir / ".metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)

                for symbol, tf_dict in data.items():
                    self._metadata[symbol] = {}
                    for tf_str, meta_dict in tf_dict.items():
                        self._metadata[symbol][tf_str] = CacheMetadata(
                            symbol=meta_dict['symbol'],
                            timeframe=meta_dict['timeframe'],
                            source=DataSource(meta_dict['source']),
                            last_update=datetime.fromisoformat(meta_dict['last_update']),
                            quality_score=meta_dict['quality_score'],
                            total_bars=meta_dict['total_bars'],
                            date_range=(
                                datetime.fromisoformat(meta_dict['date_range'][0]),
                                datetime.fromisoformat(meta_dict['date_range'][1])
                            ),
                            file_path=Path(meta_dict['file_path'])
                        )

                logger.debug(f"Loaded metadata for {len(self._metadata)} symbols")

            except Exception as e:
                logger.warning(f"Metadata load error: {e}")

    def _save_metadata(self):
        """Save cache metadata to disk."""
        metadata_path = self.cache_dir / ".metadata.json"

        try:
            data = {}
            for symbol, tf_dict in self._metadata.items():
                data[symbol] = {}
                for tf_str, metadata in tf_dict.items():
                    data[symbol][tf_str] = metadata.to_dict()

            with open(metadata_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Metadata save error: {e}")

    def get_cache_metadata(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get cache metadata as dictionary.

        Returns:
            Dictionary of symbol -> timeframe -> metadata
        """
        result = {}
        for symbol, tf_dict in self._metadata.items():
            result[symbol] = {}
            for tf_str, metadata in tf_dict.items():
                result[symbol][tf_str] = metadata.to_dict()
        return result

    # -------------------------------------------------------------------------
    # Data Fetching (Hybrid: MT5 -> API -> Cache)
    # -------------------------------------------------------------------------

    def fetch_data(
        self,
        symbol: str,
        timeframe: int,
        count: int = 1000,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        refresh: bool = False
    ) -> pd.DataFrame:
        """Fetch OHLCV data using hybrid fallback chain.

        Fetching order: MT5 -> API -> Cache

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: MQL5 timeframe constant
            count: Number of bars to retrieve
            start_date: Start date for data range
            end_date: End date for data range
            refresh: Force refresh from live sources

        Returns:
            DataFrame with OHLCV data
        """
        # Try cache first if not forcing refresh
        if not refresh:
            cached = self.get_cached_data(symbol, timeframe)
            if len(cached) > 0:
                # Filter by date range if specified
                if start_date or end_date:
                    if start_date:
                        cached = cached[cached['time'] >= start_date]
                    if end_date:
                        cached = cached[cached['time'] <= end_date]

                # Return cached if we have enough data
                if len(cached) >= count * 0.9:  # 90% tolerance
                    logger.debug(f"Using cached data for {symbol} {MQL5Timeframe.to_string(timeframe)}")
                    return cached.tail(count)

        # Try MT5
        if self.enable_mt5:
            data = self._fetch_from_mt5(symbol, timeframe, count)
            if data is not None and len(data) > 0:
                # Save to cache
                self._save_to_cache(symbol, timeframe, data, DataSource.MT5)
                return data

        # Try API
        if self.enable_api:
            data = self._fetch_from_api(symbol, timeframe, count)
            if data is not None and len(data) > 0:
                # Save to cache
                self._save_to_cache(symbol, timeframe, data, DataSource.API)
                return data

        # Return cached data as last resort
        cached = self.get_cached_data(symbol, timeframe)
        if len(cached) > 0:
            logger.warning(f"Using stale cached data for {symbol}")
            return cached

        logger.error(f"No data available for {symbol} {MQL5Timeframe.to_string(timeframe)}")
        return pd.DataFrame()

    def refresh_cache(self, symbol: str, timeframe: int) -> bool:
        """Force refresh cache from live sources.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant

        Returns:
            True if refresh successful
        """
        data = self.fetch_data(symbol, timeframe, count=1000, refresh=True)
        return len(data) > 0

    # -------------------------------------------------------------------------
    # Data Validation
    # -------------------------------------------------------------------------

    def validate_data(self, data: pd.DataFrame, timeframe: int) -> DataQualityReport:
        """Validate data quality.

        Checks for:
        - Missing bars (gaps in time series)
        - Duplicate rows
        - Price anomalies (high < low, negative values)

        Args:
            data: OHLCV DataFrame
            timeframe: MQL5 timeframe constant for expected bar frequency

        Returns:
            DataQualityReport with validation results
        """
        if data is None or len(data) == 0:
            return DataQualityReport(
                total_bars=0,
                missing_bars=0,
                duplicates=0,
                price_anomalies=0,
                quality_score=0.0,
                issues=["No data provided"]
            )

        issues = []
        total_bars = len(data)

        # Check for duplicates
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            issues.append(f"Found {duplicates} duplicate rows")

        # Check for missing bars
        missing_bars = 0
        if 'time' in data.columns and len(data) > 1:
            data_sorted = data.sort_values('time')
            expected_freq = MQL5Timeframe.to_pandas_freq(timeframe)
            expected_timedelta = pd.to_timedelta(expected_freq)

            for i in range(1, len(data_sorted)):
                actual_diff = data_sorted.iloc[i]['time'] - data_sorted.iloc[i-1]['time']
                if actual_diff > expected_timedelta * 1.5:  # 50% tolerance
                    missing_bars += int(actual_diff / expected_timedelta) - 1

            if missing_bars > 0:
                issues.append(f"Found {missing_bars} missing bars")

        # Check for price anomalies
        price_anomalies = 0

        # High < Low
        if 'high' in data.columns and 'low' in data.columns:
            invalid_hl = (data['high'] < data['low']).sum()
            price_anomalies += invalid_hl
            if invalid_hl > 0:
                issues.append(f"Found {invalid_hl} rows where high < low")

        # Close outside High-Low range
        if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
            invalid_close = ((data['close'] > data['high']) | (data['close'] < data['low'])).sum()
            price_anomalies += invalid_close
            if invalid_close > 0:
                issues.append(f"Found {invalid_close} rows where close outside high-low range")

        # Negative prices
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                negative = (data[col] <= 0).sum()
                price_anomalies += negative
                if negative > 0:
                    issues.append(f"Found {negative} non-positive values in {col}")

        # Calculate quality score
        max_possible_issues = total_bars * 3  # Heuristic
        total_issues = duplicates + missing_bars + price_anomalies
        quality_score = max(0.0, 1.0 - (total_issues / max(max_possible_issues, 1)))

        return DataQualityReport(
            total_bars=total_bars,
            missing_bars=missing_bars,
            duplicates=duplicates,
            price_anomalies=price_anomalies,
            quality_score=quality_score,
            issues=issues
        )

    # -------------------------------------------------------------------------
    # CSV Upload
    # -------------------------------------------------------------------------

    def upload_csv(self, symbol: str, timeframe: int, csv_path: Path) -> CSVUploadResult:
        """Upload CSV file and convert to Parquet cache.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            csv_path: Path to CSV file

        Returns:
            CSVUploadResult with upload status

        Raises:
            ValueError: If CSV format is invalid
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            return CSVUploadResult(
                success=False,
                symbol=symbol,
                timeframe=MQL5Timeframe.to_string(timeframe),
                rows_imported=0,
                message=f"File not found: {csv_path}"
            )

        try:
            # Read CSV
            df = pd.read_csv(csv_path)

            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                raise ValueError(f"Required columns not found: {missing_cols}")

            # Add time column if missing
            if 'time' not in df.columns:
                df['time'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1h')
            else:
                df['time'] = pd.to_datetime(df['time'], utc=True)

            # Add volume if missing
            if 'tick_volume' not in df.columns and 'volume' in df.columns:
                df['tick_volume'] = df['volume']
            elif 'tick_volume' not in df.columns:
                df['tick_volume'] = 1000

            # Select standard columns
            df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

            # Validate data
            report = self.validate_data(df, timeframe)

            # Save to cache
            success = self._save_to_cache(symbol, timeframe, df, source=DataSource.UPLOAD)

            if success:
                return CSVUploadResult(
                    success=True,
                    symbol=symbol,
                    timeframe=MQL5Timeframe.to_string(timeframe),
                    rows_imported=len(df),
                    message=f"Successfully imported {len(df)} bars. Quality score: {report.quality_score:.2f}",
                    cached_path=self.cache_dir / symbol / MQL5Timeframe.to_string(timeframe) / "data.parquet"
                )
            else:
                return CSVUploadResult(
                    success=False,
                    symbol=symbol,
                    timeframe=MQL5Timeframe.to_string(timeframe),
                    rows_imported=0,
                    message="Failed to save to cache"
                )

        except ValueError as e:
            raise
        except Exception as e:
            return CSVUploadResult(
                success=False,
                symbol=symbol,
                timeframe=MQL5Timeframe.to_string(timeframe),
                rows_imported=0,
                message=f"Error processing CSV: {str(e)}"
            )

    # -------------------------------------------------------------------------
    # Cache Status
    # -------------------------------------------------------------------------

    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache statistics and status.

        Returns:
            Dictionary with cache statistics
        """
        total_symbols = len(self._metadata)
        total_timeframes = sum(len(tf_dict) for tf_dict in self._metadata.values())
        total_bars = sum(
            meta.total_bars
            for tf_dict in self._metadata.values()
            for meta in tf_dict.values()
        )

        sources_count = {}
        for tf_dict in self._metadata.values():
            for meta in tf_dict.values():
                source = meta.source.value
                sources_count[source] = sources_count.get(source, 0) + 1

        # Calculate average quality
        quality_scores = [
            meta.quality_score
            for tf_dict in self._metadata.values()
            for meta in tf_dict.values()
        ]
        avg_quality = np.mean(quality_scores) if quality_scores else 0.0

        # Build symbol list with timeframes
        symbols = {}
        for symbol, tf_dict in self._metadata.items():
            symbols[symbol] = {
                "timeframes": list(tf_dict.keys()),
                "total_bars": sum(meta.total_bars for meta in tf_dict.values())
            }

        return {
            "cache_dir": str(self.cache_dir),
            "total_symbols": total_symbols,
            "total_timeframes": total_timeframes,
            "total_bars": total_bars,
            "average_quality_score": round(avg_quality, 3),
            "data_sources": sources_count,
            "symbols": symbols,
            "mt5_enabled": self.enable_mt5,
            "mt5_connected": self._mt5_connected,
            "api_enabled": self.enable_api,
            "parquet_available": PARQUET_AVAILABLE
        }


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'DataManager',
    'DataSource',
    'DataQualityReport',
    'CacheMetadata',
    'CSVUploadResult',
    'MQL5Timeframe',
]
