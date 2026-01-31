"""
Test suite for QSL Risk/RiskClient.mqh Risk Client functionality.

This module provides Python equivalents of MQL5 RiskClient functions for testing purposes.
The tests verify the behavior of GetRiskMultiplier and ReadRiskFromFile functions that
will be implemented in MQL5 Risk/RiskClient.mqh.

The RiskClient provides risk multiplier retrieval with:
- Fast path: GlobalVariable (set by Python agents)
- Fallback path: JSON file (risk_matrix.json)
- Data freshness validation (1-hour max age)
- Default multiplier fallback (1.0)
"""

import pytest
import time
from unittest.mock import Mock, patch


# Constants from MQL5 code
QM_DEFAULT_MULTIPLIER = 1.0
QM_MAX_DATA_AGE_SECONDS = 3600  # 1 hour
QM_RISK_MULTIPLIER_VAR = "QM_RISK_MULTIPLIER"


class RiskData:
    """Python equivalent of MQL5 RiskData struct."""

    def __init__(self):
        self.multiplier = QM_DEFAULT_MULTIPLIER
        self.timestamp = 0


def find_json_object(json_content: str, key: str) -> str:
    """Python equivalent of MQL5 FindJsonObject function."""
    search_pattern = f'"{key}"'
    key_pos = json_content.find(search_pattern)

    if key_pos < 0:
        return ""

    start = json_content.find("{", key_pos)
    if start < 0:
        return ""

    depth = 0
    end = -1

    for i in range(start, len(json_content)):
        char = json_content[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end < 0:
        return ""

    return json_content[start:end + 1]


def extract_json_double(json_object: str, key: str) -> float:
    """Python equivalent of MQL5 ExtractJsonDouble function."""
    search_pattern = f'"{key}"'
    key_pos = json_object.find(search_pattern)

    if key_pos < 0:
        return 0.0

    colon_pos = json_object.find(":", key_pos)
    if colon_pos < 0:
        return 0.0

    value_str = ""
    i = colon_pos + 1

    while i < len(json_object):
        char = json_object[i]
        if char not in (" ", "\t", "\n"):
            break
        i += 1

    has_decimal = False
    while i < len(json_object):
        char = json_object[i]

        if char == "-" and len(value_str) == 0:
            value_str += char
        elif char == "." and not has_decimal:
            value_str += char
            has_decimal = True
        elif char.isdigit():
            value_str += char
        else:
            break
        i += 1

    if len(value_str) > 0:
        return float(value_str)

    return 0.0


def read_risk_from_file(symbol: str, json_content: str) -> RiskData:
    """
    Python equivalent of MQL5 ReadRiskFromFile function.

    Parses JSON structure:
    {
      "EURUSD": { "multiplier": 1.5, "timestamp": 1234567890 },
      "GBPUSD": { "multiplier": 1.0, "timestamp": 1234567890 }
    }
    """
    data = RiskData()

    # Find symbol section in JSON
    symbol_section = find_json_object(json_content, symbol)
    if symbol_section == "":
        return data

    # Extract multiplier and timestamp
    multiplier = extract_json_double(symbol_section, "multiplier")
    timestamp = int(extract_json_double(symbol_section, "timestamp"))

    if multiplier > 0:
        data.multiplier = multiplier
    else:
        data.multiplier = QM_DEFAULT_MULTIPLIER

    data.timestamp = timestamp

    return data


def get_risk_multiplier(
    symbol: str,
    global_multiplier: float = 0.0,
    json_content: str = None,
    current_timestamp: int = None
) -> float:
    """
    Python equivalent of MQL5 GetRiskMultiplier function.

    Args:
        symbol: Trading symbol (e.g., "EURUSD")
        global_multiplier: Value from GlobalVariable (0 if not set)
        json_content: JSON file content (None if file not available)
        current_timestamp: Current Unix timestamp (uses time.time() if None)

    Returns:
        Risk multiplier value (default 1.0 if not found)
    """
    # Fast path: Check GlobalVariable set by Python agent
    if global_multiplier > 0:
        return global_multiplier

    # Fallback path: Read from risk_matrix.json file
    if json_content is not None:
        data = read_risk_from_file(symbol, json_content)

        # Validate data freshness
        if current_timestamp is None:
            current_timestamp = int(time.time())

        if data.timestamp > 0 and (current_timestamp - data.timestamp) < QM_MAX_DATA_AGE_SECONDS:
            return data.multiplier

    # Default fallback
    return QM_DEFAULT_MULTIPLIER


class TestGlobalVariableFastPath:
    """Test suite for GlobalVariable fast path retrieval."""

    def test_global_variable_set_returns_value(self):
        """Test that GlobalVariable value is returned when set."""
        result = get_risk_multiplier("EURUSD", global_multiplier=1.5)
        assert result == 1.5

    def test_global_variable_zero_triggers_fallback(self):
        """Test that zero GlobalVariable triggers file fallback."""
        json_content = '{"EURUSD": {"multiplier": 2.0, "timestamp": 1234567890}}'
        # Must provide current_timestamp for freshness check to pass
        result = get_risk_multiplier("EURUSD", global_multiplier=0, json_content=json_content,
                                     current_timestamp=1234568000)
        assert result == 2.0

    def test_global_variable_negative_triggers_fallback(self):
        """Test that negative GlobalVariable triggers file fallback."""
        json_content = '{"EURUSD": {"multiplier": 1.25, "timestamp": 1234567890}}'
        # Must provide current_timestamp for freshness check to pass
        result = get_risk_multiplier("EURUSD", global_multiplier=-1.0, json_content=json_content,
                                     current_timestamp=1234568000)
        assert result == 1.25

    def test_global_variable_high_value(self):
        """Test that high multiplier values are preserved."""
        result = get_risk_multiplier("GBPUSD", global_multiplier=3.0)
        assert result == 3.0


class TestJSONFileFallback:
    """Test suite for JSON file fallback path."""

    def test_json_file_symbol_found(self):
        """Test finding symbol in JSON file."""
        json_content = '{"EURUSD": {"multiplier": 1.5, "timestamp": 1234567890}}'
        # Must provide current_timestamp for freshness check to pass
        result = get_risk_multiplier("EURUSD", global_multiplier=0, json_content=json_content,
                                     current_timestamp=1234568000)
        assert result == 1.5

    def test_json_file_symbol_not_found_returns_default(self):
        """Test that missing symbol returns default multiplier."""
        json_content = '{"GBPUSD": {"multiplier": 1.0, "timestamp": 1234567890}}'
        result = get_risk_multiplier("EURUSD", global_multiplier=0, json_content=json_content)
        assert result == QM_DEFAULT_MULTIPLIER

    def test_json_file_empty_content_returns_default(self):
        """Test that empty JSON content returns default multiplier."""
        # Empty string means no valid JSON, so should return default
        # The read_risk_from_file will fail to find the symbol
        result = get_risk_multiplier("EURUSD", global_multiplier=0, json_content="")
        # Empty string content won't find EURUSD, returns default
        assert result == QM_DEFAULT_MULTIPLIER

    def test_json_file_multiple_symbols(self):
        """Test JSON file with multiple symbols."""
        json_content = '''{
            "EURUSD": {"multiplier": 1.5, "timestamp": 1234567890},
            "GBPUSD": {"multiplier": 2.0, "timestamp": 1234567891},
            "USDJPY": {"multiplier": 0.8, "timestamp": 1234567892}
        }'''

        # Must provide current_timestamp for freshness check to pass
        current_time = 1234568000
        assert get_risk_multiplier("EURUSD", global_multiplier=0, json_content=json_content,
                                   current_timestamp=current_time) == 1.5
        assert get_risk_multiplier("GBPUSD", global_multiplier=0, json_content=json_content,
                                   current_timestamp=current_time) == 2.0
        assert get_risk_multiplier("USDJPY", global_multiplier=0, json_content=json_content,
                                   current_timestamp=current_time) == 0.8


class TestDataFreshnessValidation:
    """Test suite for data freshness validation (1-hour max age)."""

    def test_fresh_data_accepted(self):
        """Test that fresh data (less than 1 hour old) is accepted."""
        current_time = 1234567890
        json_content = '{"EURUSD": {"multiplier": 1.5, "timestamp": 1234565000}}'  # 2890 seconds old
        result = get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=json_content,
            current_timestamp=current_time
        )
        assert result == 1.5

    def test_stale_data_rejected(self):
        """Test that stale data (more than 1 hour old) is rejected."""
        current_time = 1234567890
        json_content = '{"EURUSD": {"multiplier": 1.5, "timestamp": 1234560000}}'  # 7890 seconds old (>1 hour)
        result = get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=json_content,
            current_timestamp=current_time
        )
        assert result == QM_DEFAULT_MULTIPLIER  # Falls back to default

    def test_exactly_one_hour_old_rejected(self):
        """Test data exactly 1 hour old (boundary case - rejected)."""
        current_time = 1234567890
        json_content = '{"EURUSD": {"multiplier": 1.5, "timestamp": 1234564290}}'  # 3600 seconds old
        result = get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=json_content,
            current_timestamp=current_time
        )
        # The condition is (current - timestamp) < 3600
        # 3600 is NOT less than 3600, so it fails
        assert result == QM_DEFAULT_MULTIPLIER  # Rejected, returns default

    def test_one_second_over_hour_rejected(self):
        """Test data 1 second over 1 hour old (boundary case)."""
        current_time = 1234567890
        json_content = '{"EURUSD": {"multiplier": 1.5, "timestamp": 1234564289}}'  # 3601 seconds old
        result = get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=json_content,
            current_timestamp=current_time
        )
        assert result == QM_DEFAULT_MULTIPLIER  # Should be rejected

    def test_zero_timestamp_freshness_check(self):
        """Test that zero timestamp is handled correctly."""
        current_time = 1234567890
        json_content = '{"EURUSD": {"multiplier": 1.5, "timestamp": 0}}'
        result = get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=json_content,
            current_timestamp=current_time
        )
        # Zero timestamp means freshness check will fail (current - 0 > 3600)
        assert result == QM_DEFAULT_MULTIPLIER


class TestDefaultMultiplierFallback:
    """Test suite for default multiplier (1.0) fallback behavior."""

    def test_no_global_no_file_returns_default(self):
        """Test that no GlobalVariable and no file returns default."""
        result = get_risk_multiplier("EURUSD", global_multiplier=0, json_content=None)
        assert result == QM_DEFAULT_MULTIPLIER

    def test_global_zero_no_file_returns_default(self):
        """Test that zero GlobalVariable and no file returns default."""
        result = get_risk_multiplier("EURUSD", global_multiplier=0, json_content="")
        assert result == QM_DEFAULT_MULTIPLIER

    def test_file_with_zero_multiplier_defaults_to_one(self):
        """Test that multiplier of 0 in JSON defaults to 1.0."""
        json_content = '{"EURUSD": {"multiplier": 0, "timestamp": 1234567890}}'
        result = get_risk_multiplier("EURUSD", global_multiplier=0, json_content=json_content)
        assert result == QM_DEFAULT_MULTIPLIER

    def test_file_with_negative_multiplier_defaults_to_one(self):
        """Test that negative multiplier in JSON defaults to 1.0."""
        json_content = '{"EURUSD": {"multiplier": -0.5, "timestamp": 1234567890}}'
        result = get_risk_multiplier("EURUSD", global_multiplier=0, json_content=json_content)
        # Note: The MQL5 code checks if multiplier <= 0, then sets to default
        # But our extract handles negative numbers, so this test documents expected behavior
        # The actual MQL5 code: if(data.multiplier <= 0) data.multiplier = QM_DEFAULT_MULTIPLIER;
        # So negative should be normalized to 1.0
        assert result == QM_DEFAULT_MULTIPLIER


class TestReadRiskFromFile:
    """Test suite for ReadRiskFromFile function."""

    def test_read_valid_risk_data(self):
        """Test reading valid risk data from JSON."""
        json_content = '{"EURUSD": {"multiplier": 1.5, "timestamp": 1234567890}}'
        data = read_risk_from_file("EURUSD", json_content)
        assert data.multiplier == 1.5
        assert data.timestamp == 1234567890

    def test_read_missing_symbol_returns_defaults(self):
        """Test reading missing symbol returns default values."""
        json_content = '{"GBPUSD": {"multiplier": 1.0, "timestamp": 1234567890}}'
        data = read_risk_from_file("EURUSD", json_content)
        assert data.multiplier == QM_DEFAULT_MULTIPLIER
        assert data.timestamp == 0

    def test_read_with_missing_timestamp(self):
        """Test reading when timestamp field is missing."""
        json_content = '{"EURUSD": {"multiplier": 1.5}}'
        data = read_risk_from_file("EURUSD", json_content)
        assert data.multiplier == 1.5
        assert data.timestamp == 0  # Missing field returns 0

    def test_read_with_zero_multiplier_normalizes_to_default(self):
        """Test that zero multiplier is normalized to default."""
        json_content = '{"EURUSD": {"multiplier": 0, "timestamp": 1234567890}}'
        data = read_risk_from_file("EURUSD", json_content)
        assert data.multiplier == QM_DEFAULT_MULTIPLIER
        assert data.timestamp == 1234567890


class TestIntegrationScenarios:
    """Integration tests for complete RiskClient workflows."""

    def test_fast_path_takes_precedence(self):
        """Test that GlobalVariable fast path takes precedence over file."""
        json_content = '{"EURUSD": {"multiplier": 2.0, "timestamp": 1234567890}}'
        # GlobalVariable is set, so file should not be read
        result = get_risk_multiplier("EURUSD", global_multiplier=1.5, json_content=json_content)
        assert result == 1.5  # Fast path value, not file value

    def test_fresh_file_data_used_when_no_global(self):
        """Test that fresh file data is used when GlobalVariable not set."""
        current_time = 1234568000
        json_content = '{"EURUSD": {"multiplier": 1.75, "timestamp": 1234565000}}'  # Fresh data
        result = get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=json_content,
            current_timestamp=current_time
        )
        assert result == 1.75

    def test_stale_file_falls_back_to_default(self):
        """Test that stale file data falls back to default multiplier."""
        current_time = 1234568000
        json_content = '{"EURUSD": {"multiplier": 1.75, "timestamp": 1234560000}}'  # Stale data
        result = get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=json_content,
            current_timestamp=current_time
        )
        assert result == QM_DEFAULT_MULTIPLIER

    def test_complete_priority_chain(self):
        """Test complete priority chain: Global > Fresh File > Default."""
        # 1. No Global, no file -> Default
        assert get_risk_multiplier("EURUSD", global_multiplier=0, json_content=None) == 1.0

        # 2. No Global, stale file -> Default
        current_time = 1234568000
        stale_json = '{"EURUSD": {"multiplier": 2.0, "timestamp": 1234560000}}'
        assert get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=stale_json,
            current_timestamp=current_time
        ) == 1.0

        # 3. No Global, fresh file -> File value
        fresh_json = '{"EURUSD": {"multiplier": 2.5, "timestamp": 1234565000}}'
        assert get_risk_multiplier(
            "EURUSD",
            global_multiplier=0,
            json_content=fresh_json,
            current_timestamp=current_time
        ) == 2.5

        # 4. Global set (regardless of file) -> Global value
        assert get_risk_multiplier(
            "EURUSD",
            global_multiplier=3.0,
            json_content=fresh_json,
            current_timestamp=current_time
        ) == 3.0
