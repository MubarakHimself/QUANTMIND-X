"""
Test suite for QSL Utils/JSON.mqh JSON parsing utilities.

This module provides Python equivalents of MQL5 JSON functions for testing purposes.
The tests verify the behavior of FindJsonObject and ExtractJsonDouble functions that
will be implemented in MQL5 Utils/JSON.mqh.

These functions are simple manual JSON parsers used in MQL5 where native JSON support
is limited. They handle:
- Finding JSON objects by key
- Extracting double values from JSON object strings
- Handling nested braces
- Parsing numbers with sign and decimal support
"""

import pytest


def find_json_object(json_content: str, key: str) -> str:
    """
    Python equivalent of MQL5 FindJsonObject function.

    Find JSON object for a given key in JSON string.

    Args:
        json_content: Full JSON content
        key: Key to search for

    Returns:
        JSON object as string, or empty if not found
    """
    # Search for "KEY": {
    search_pattern = f'"{key}"'
    key_pos = json_content.find(search_pattern)

    if key_pos < 0:
        return ""

    # Find opening brace after key
    start = json_content.find("{", key_pos)
    if start < 0:
        return ""

    # Find matching closing brace (handle nested objects)
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
    """
    Python equivalent of MQL5 ExtractJsonDouble function.

    Extract double value from JSON object string.

    Args:
        json_object: JSON object string
        key: Key to extract

    Returns:
        Double value, or 0 if not found
    """
    search_pattern = f'"{key}"'
    key_pos = json_object.find(search_pattern)

    if key_pos < 0:
        return 0.0

    # Find colon after key
    colon_pos = json_object.find(":", key_pos)
    if colon_pos < 0:
        return 0.0

    # Extract value (handle number, including negative and decimals)
    value_str = ""
    i = colon_pos + 1

    # Skip whitespace
    while i < len(json_object):
        char = json_object[i]
        if char not in (" ", "\t", "\n"):
            break
        i += 1

    # Extract number characters
    has_decimal = False
    while i < len(json_object):
        char = json_object[i]

        # Handle negative numbers
        if char == "-" and len(value_str) == 0:
            value_str += char
        # Handle decimal point
        elif char == "." and not has_decimal:
            value_str += char
            has_decimal = True
        # Handle digits
        elif char.isdigit():
            value_str += char
        else:
            # End of number
            break
        i += 1

    if len(value_str) > 0:
        return float(value_str)

    return 0.0


class TestFindJsonObject:
    """Test suite for FindJsonObject function."""

    def test_find_simple_object(self):
        """Test finding a simple JSON object."""
        json_content = '{"EURUSD": {"multiplier": 1.5, "timestamp": 1234567890}, "GBPUSD": {"multiplier": 1.0}}'
        result = find_json_object(json_content, "EURUSD")
        assert result == '{"multiplier": 1.5, "timestamp": 1234567890}'

    def test_find_second_object(self):
        """Test finding the second object in JSON."""
        json_content = '{"EURUSD": {"multiplier": 1.5}, "GBPUSD": {"multiplier": 1.0, "timestamp": 1234567890}}'
        result = find_json_object(json_content, "GBPUSD")
        assert result == '{"multiplier": 1.0, "timestamp": 1234567890}'

    def test_find_with_nested_objects(self):
        """Test finding an object that contains nested objects."""
        json_content = '{"symbol": {"multiplier": 1.5, "config": {"min": 0.1, "max": 2.0}}}'
        result = find_json_object(json_content, "symbol")
        assert result == '{"multiplier": 1.5, "config": {"min": 0.1, "max": 2.0}}'

    def test_key_not_found(self):
        """Test finding a non-existent key returns empty string."""
        json_content = '{"EURUSD": {"multiplier": 1.5}}'
        result = find_json_object(json_content, "JPYUSD")
        assert result == ""

    def test_nested_brace_handling(self):
        """Test correct handling of nested braces."""
        # This tests the depth counter logic
        json_content = '{"outer": {"inner": {"deep": 123}, "other": 456}}'
        result = find_json_object(json_content, "outer")
        assert result == '{"inner": {"deep": 123}, "other": 456}'

    def test_malformed_json_missing_brace(self):
        """Test handling of malformed JSON (missing closing brace)."""
        json_content = '{"EURUSD": {"multiplier": 1.5'
        result = find_json_object(json_content, "EURUSD")
        # Should return empty because matching closing brace not found
        assert result == ""

    def test_malformed_json_no_brace_after_key(self):
        """Test handling when no opening brace found after key."""
        json_content = '{"EURUSD": "value"}'
        result = find_json_object(json_content, "EURUSD")
        assert result == ""


class TestExtractJsonDouble:
    """Test suite for ExtractJsonDouble function."""

    def test_extract_positive_integer(self):
        """Test extracting a positive integer."""
        json_object = '{"multiplier": 123}'
        result = extract_json_double(json_object, "multiplier")
        assert result == 123.0

    def test_extract_positive_decimal(self):
        """Test extracting a positive decimal number."""
        json_object = '{"multiplier": 1.5}'
        result = extract_json_double(json_object, "multiplier")
        assert result == 1.5

    def test_extract_negative_number(self):
        """Test extracting a negative number."""
        json_object = '{"value": -0.25}'
        result = extract_json_double(json_object, "value")
        assert result == -0.25

    def test_extract_negative_integer(self):
        """Test extracting a negative integer."""
        json_object = '{"loss": -100}'
        result = extract_json_double(json_object, "loss")
        assert result == -100.0

    def test_extract_with_whitespace(self):
        """Test extraction handles whitespace after colon."""
        json_object = '{"multiplier" :   1.5}'
        result = extract_json_double(json_object, "multiplier")
        assert result == 1.5

    def test_extract_with_newline(self):
        """Test extraction handles newlines after colon."""
        json_object = '{"multiplier":\n1.5}'
        result = extract_json_double(json_object, "multiplier")
        assert result == 1.5

    def test_key_not_found(self):
        """Test extracting non-existent key returns 0."""
        json_object = '{"multiplier": 1.5}'
        result = extract_json_double(json_object, "nonexistent")
        assert result == 0.0

    def test_extract_zero(self):
        """Test extracting zero value."""
        json_object = '{"value": 0}'
        result = extract_json_double(json_object, "value")
        assert result == 0.0

    def test_extract_scientific_notation_partial(self):
        """Test that scientific notation is NOT fully supported (by design)."""
        # The MQL5 implementation doesn't handle 'e' notation
        # This test documents the expected behavior
        json_object = '{"value": 1.5e10}'
        result = extract_json_double(json_object, "value")
        # Will stop at 'e', returning partial value
        assert result == 1.5

    def test_extract_from_complex_object(self):
        """Test extraction from object with multiple fields."""
        json_object = '{"multiplier": 1.5, "timestamp": 1234567890, "status": 1}'
        result = extract_json_double(json_object, "timestamp")
        assert result == 1234567890.0


class TestIntegrationScenarios:
    """Integration tests combining both functions."""

    def test_full_risk_matrix_workflow(self):
        """Test the complete workflow as used in risk matrix parsing."""
        json_content = '''{
            "EURUSD": {"multiplier": 1.5, "timestamp": 1234567890},
            "GBPUSD": {"multiplier": 1.0, "timestamp": 1234567891}
        }'''

        # Find EURUSD object
        eurusd_obj = find_json_object(json_content, "EURUSD")
        assert eurusd_obj != ""

        # Extract multiplier from EURUSD
        multiplier = extract_json_double(eurusd_obj, "multiplier")
        assert multiplier == 1.5

        # Extract timestamp from EURUSD
        timestamp = extract_json_double(eurusd_obj, "timestamp")
        assert timestamp == 1234567890.0

        # Find GBPUSD object
        gbpusd_obj = find_json_object(json_content, "GBPUSD")
        assert gbpusd_obj != ""

        # Extract multiplier from GBPUSD
        gbp_multiplier = extract_json_double(gbpusd_obj, "multiplier")
        assert gbp_multiplier == 1.0

    def test_negative_multiplier_scenario(self):
        """Test handling negative multipliers (edge case)."""
        json_content = '{"JPYUSD": {"multiplier": -0.5, "adjustment": -1.25}}'

        obj = find_json_object(json_content, "JPYUSD")
        multiplier = extract_json_double(obj, "multiplier")
        assert multiplier == -0.5

        adjustment = extract_json_double(obj, "adjustment")
        assert adjustment == -1.25

    def test_drawdown_calculation_scenario(self):
        """Test typical drawdown calculation values."""
        json_content = '{"account": {"daily_loss_pct": 2.5, "hard_stop_pct": 4.0, "target_pct": 8.0}}'

        obj = find_json_object(json_content, "account")
        daily_loss = extract_json_double(obj, "daily_loss_pct")
        hard_stop = extract_json_double(obj, "hard_stop_pct")
        target = extract_json_double(obj, "target_pct")

        assert daily_loss == 2.5
        assert hard_stop == 4.0
        assert target == 8.0
