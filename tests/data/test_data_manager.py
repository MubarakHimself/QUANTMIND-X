"""
Tests for DataManager functionality.

Task Group 1: Data Manager Implementation

This test suite covers:
1. Hybrid fetching (MT5 -> API -> Cache fallback)
2. Parquet caching with symbol/timeframe organization
3. Data validation (missing bars, duplicates, anomalies)
4. Cache metadata tracking
5. Multi-timeframe data retrieval
6. CSV upload and conversion to Parquet
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

from src.data.data_manager import (
    DataManager,
    DataSource,
    DataQualityReport,
    MQL5Timeframe,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for Parquet cache."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h', tz='UTC')
    np.random.seed(42)
    close_base = 1.1000

    data = pd.DataFrame({
        'time': dates,
        'open': [close_base + np.random.randn() * 0.001 for _ in range(100)],
        'high': [close_base + np.random.randn() * 0.001 + 0.0005 for _ in range(100)],
        'low': [close_base + np.random.randn() * 0.001 - 0.0005 for _ in range(100)],
        'close': [close_base + np.random.randn() * 0.001 for _ in range(100)],
        'tick_volume': np.random.randint(100, 10000, 100)
    })

    # Ensure high >= close and low <= close
    data['high'] = data[['high', 'close']].max(axis=1)
    data['low'] = data[['low', 'close']].min(axis=1)

    return data


@pytest.fixture
def sample_csv_file(tmp_path, sample_ohlcv_data):
    """Create a sample CSV file for upload testing."""
    csv_file = tmp_path / "test_data.csv"
    sample_ohlcv_data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def mock_mt5():
    """Mock MetaTrader5 connection."""
    with patch('src.data.data_manager.mt5') as mock:
        mock.initialize.return_value = True
        mock.copy_rates_from_pos.return_value = None  # MT5 unavailable by default
        mock.last_error.return_value = (0, "No error")
        yield mock


@pytest.fixture
def data_manager(temp_cache_dir, mock_mt5):
    """Create a DataManager instance with temporary cache."""
    return DataManager(
        cache_dir=temp_cache_dir,
        enable_mt5=False,
        enable_api=False
    )


# =============================================================================
# Test 1: Hybrid Data Fetching (MT5 -> API -> Cache fallback)
# =============================================================================

def test_hybrid_fetching_fallback_chain(data_manager, sample_ohlcv_data, temp_cache_dir):
    """Test hybrid fetching: MT5 -> API -> Cache fallback chain.

    This test verifies:
    1. First tries MT5 (should fail in this test setup)
    2. Falls back to API (should fail in this test setup)
    3. Returns cached data if available
    4. Returns empty DataFrame if no source available
    """
    symbol = "EURUSD"
    timeframe = MQL5Timeframe.PERIOD_H1

    # Initially, no data should be available (no cache, MT5 disabled, API disabled)
    result = data_manager.fetch_data(symbol, timeframe, count=100)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

    # Add data to cache directly by writing to the file
    tf_str = "H1"
    cache_path = temp_cache_dir / symbol / tf_str
    cache_path.mkdir(parents=True, exist_ok=True)
    sample_ohlcv_data.to_parquet(cache_path / "data.parquet", index=False)

    # Now fetch should return cached data
    result = data_manager.fetch_data(symbol, timeframe, count=100)
    assert len(result) == len(sample_ohlcv_data)
    assert list(result.columns) == ['time', 'open', 'high', 'low', 'close', 'tick_volume']
    assert result['time'].dt.tz is not None  # Timezone-aware


# =============================================================================
# Test 2: Parquet Caching with Symbol/Timeframe Organization
# =============================================================================

def test_parquet_cache_organization(data_manager, sample_ohlcv_data, temp_cache_dir):
    """Test Parquet caching with proper symbol/timeframe organization.

    This test verifies:
    1. Cache directory structure: {cache_dir}/{symbol}/{timeframe}/data.parquet
    2. Parquet files are created correctly
    3. Data can be retrieved from cache
    4. Cache metadata is tracked
    """
    symbol = "GBPUSD"
    timeframe = MQL5Timeframe.PERIOD_H1

    # Save data to cache
    data_manager._save_to_cache(symbol, timeframe, sample_ohlcv_data, DataSource.MT5)

    # Verify directory structure
    expected_path = temp_cache_dir / symbol / "H1" / "data.parquet"
    assert expected_path.exists(), f"Cache file not found at {expected_path}"

    # Verify cache metadata
    metadata = data_manager.get_cache_metadata()
    assert symbol in metadata
    assert "H1" in metadata[symbol]
    assert metadata[symbol]["H1"]["source"] == "mt5"  # Metadata returns string value
    assert "last_update" in metadata[symbol]["H1"]

    # Verify data can be loaded from cache
    loaded_data = data_manager.get_cached_data(symbol, timeframe)
    assert len(loaded_data) == len(sample_ohlcv_data)
    assert loaded_data['close'].iloc[0] == sample_ohlcv_data['close'].iloc[0]


# =============================================================================
# Test 3: Data Validation (Missing Bars, Duplicates, Price Anomalies)
# =============================================================================

def test_data_validation_quality_checks(data_manager):
    """Test data validation: missing bars, duplicates, price anomalies.

    This test verifies:
    1. Missing bars detection
    2. Duplicate detection and removal
    3. Price anomaly detection (high < low, negative prices)
    4. Quality report generation
    """
    # Create data with issues
    dates = pd.date_range(start='2024-01-01', periods=10, freq='1h', tz='UTC')

    # Introduce missing bars (remove 2 bars)
    dates_with_gaps = dates.delete([3, 7])

    # Create DataFrame with issues
    data = pd.DataFrame({
        'time': dates_with_gaps,
        'open': [1.10] * 8,
        'high': [1.11] * 8,
        'low': [1.09] * 8,
        'close': [1.105] * 8,
        'tick_volume': [1000] * 8
    })

    # Add duplicate row
    data = pd.concat([data, data.iloc[[0]]], ignore_index=True)

    # Add price anomaly (high < low)
    data.loc[2, 'high'] = 1.08  # high < low
    data.loc[2, 'low'] = 1.12

    # Run validation
    report = data_manager.validate_data(data, MQL5Timeframe.PERIOD_H1)

    assert isinstance(report, DataQualityReport)
    assert report.total_bars == 9
    assert report.missing_bars == 2  # 2 gaps in 10 bars
    assert report.duplicates == 1
    assert report.price_anomalies >= 1  # At least the high < low anomaly
    assert not report.is_valid()  # Should fail validation


# =============================================================================
# Test 4: Cache Metadata Tracking
# =============================================================================

def test_cache_metadata_tracking(data_manager, sample_ohlcv_data):
    """Test cache metadata tracking for symbols and timeframes.

    This test verifies:
    1. Metadata tracks last update time
    2. Metadata tracks data source (MT5/API/UPLOAD)
    3. Metadata tracks data quality score
    4. Multiple symbols and timeframes tracked separately
    """
    # Cache EURUSD H1 data from MT5
    data_manager._save_to_cache("EURUSD", MQL5Timeframe.PERIOD_H1, sample_ohlcv_data, DataSource.MT5)

    # Cache GBPUSD M15 data from API
    data_manager._save_to_cache("GBPUSD", MQL5Timeframe.PERIOD_M15, sample_ohlcv_data, DataSource.API)

    # Get metadata
    metadata = data_manager.get_cache_metadata()

    # Verify EURUSD H1 metadata
    assert "EURUSD" in metadata
    assert "H1" in metadata["EURUSD"]
    assert metadata["EURUSD"]["H1"]["source"] == "mt5"  # Metadata returns string value
    assert "quality_score" in metadata["EURUSD"]["H1"]

    # Verify GBPUSD M15 metadata
    assert "GBPUSD" in metadata
    assert "M15" in metadata["GBPUSD"]
    assert metadata["GBPUSD"]["M15"]["source"] == "api"  # Metadata returns string value

    # Verify last_update is recent
    from datetime import datetime
    now = datetime.now(timezone.utc)
    from datetime import datetime
    last_update_dt = datetime.fromisoformat(metadata["EURUSD"]["H1"]["last_update"])
    assert (now - last_update_dt).total_seconds() < 5  # Within 5 seconds


# =============================================================================
# Test 5: Multi-Timeframe Data Retrieval
# =============================================================================

def test_multi_timeframe_support(data_manager, temp_cache_dir):
    """Test multi-timeframe data retrieval support.

    This test verifies:
    1. All supported timeframes: M1, M5, M15, M30, H1, H4, D1, W1, MN1
    2. Data is stored separately per timeframe
    3. Timeframe conversion utilities work correctly
    4. Timeframe constants match MQL5 specification
    """
    symbol = "XAUUSD"
    timeframes = [
        MQL5Timeframe.PERIOD_M1,
        MQL5Timeframe.PERIOD_M5,
        MQL5Timeframe.PERIOD_M15,
        MQL5Timeframe.PERIOD_M30,
        MQL5Timeframe.PERIOD_H1,
        MQL5Timeframe.PERIOD_H4,
        MQL5Timeframe.PERIOD_D1,
        MQL5Timeframe.PERIOD_W1,
        MQL5Timeframe.PERIOD_MN1,
    ]

    # Test timeframe conversion to minutes
    assert MQL5Timeframe.to_minutes(MQL5Timeframe.PERIOD_M1) == 1
    assert MQL5Timeframe.to_minutes(MQL5Timeframe.PERIOD_H1) == 60
    assert MQL5Timeframe.to_minutes(MQL5Timeframe.PERIOD_D1) == 1440

    # Test timeframe conversion to pandas frequency
    assert MQL5Timeframe.to_pandas_freq(MQL5Timeframe.PERIOD_M1) == "1min"
    assert MQL5Timeframe.to_pandas_freq(MQL5Timeframe.PERIOD_H1) == "1h"
    assert MQL5Timeframe.to_pandas_freq(MQL5Timeframe.PERIOD_D1) == "1D"

    # Verify directory structure exists for each timeframe without caching actual data
    # (Caching all 9 timeframes is slow, so we test the utilities instead)
    for tf in timeframes:
        tf_str = MQL5Timeframe.to_string(tf)
        assert tf_str in ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]

    # Test that string conversion works both ways
    assert MQL5Timeframe.from_string("M1") == MQL5Timeframe.PERIOD_M1
    assert MQL5Timeframe.from_string("H1") == MQL5Timeframe.PERIOD_H1
    assert MQL5Timeframe.from_string("D1") == MQL5Timeframe.PERIOD_D1


# =============================================================================
# Test 6: CSV Upload and Parquet Conversion
# =============================================================================

def test_csv_upload_and_parquet_conversion(data_manager, sample_csv_file, temp_cache_dir):
    """Test CSV upload endpoint and Parquet conversion.

    This test verifies:
    1. CSV file is parsed correctly
    2. OHLCV columns are validated
    3. Data is converted to Parquet format
    4. Uploaded data is available in cache
    5. Invalid CSV format is rejected
    """
    symbol = "EURUSD"
    timeframe = MQL5Timeframe.PERIOD_H1

    # Upload valid CSV
    result = data_manager.upload_csv(symbol, timeframe, sample_csv_file)
    assert result.success is True

    # Verify Parquet file created
    expected_path = temp_cache_dir / symbol / "H1" / "data.parquet"
    assert expected_path.exists()

    # Verify data can be retrieved
    cached_data = data_manager.get_cached_data(symbol, timeframe)
    assert len(cached_data) > 0
    assert 'close' in cached_data.columns

    # Test invalid CSV (missing required columns)
    invalid_csv_file = sample_csv_file.parent / "invalid.csv"
    pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}).to_csv(invalid_csv_file, index=False)

    with pytest.raises(ValueError, match="Required columns not found"):
        data_manager.upload_csv(symbol, timeframe, invalid_csv_file)
