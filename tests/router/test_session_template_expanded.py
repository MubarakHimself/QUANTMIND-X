"""
Expanded Tests for SessionTemplate.

Story 16.2: Session Template Class — Configurable 10-Window Canonical Cycle
[P1] Coverage gaps from existing tests

These tests fill gaps identified in the existing test suite:
- validate_24_hour_coverage() NOT TESTED AT ALL
- get_window_at() edge cases (exact boundaries)
- from_dict() malformed data handling
- get_bot_type_mix() invalid window name
"""

import pytest
from datetime import datetime, timezone

from src.router.session_template import (
    SessionTemplate,
    SessionTemplateConfig,
    SessionWindow,
    BotTypeMix,
    EventFilter,
    WindowIntensity,
    CanonicalSessionWindow,
    create_default_template,
    DEAD_ZONE_START,
    DEAD_ZONE_END,
    PREMIUM_KELLY_MULTIPLIER,
)


class TestValidate24HourCoverage:
    """Test validate_24_hour_coverage() method - NOT TESTED IN EXISTING SUITE."""

    def test_default_template_has_24_hour_coverage(self):
        """Test that default template provides 24-hour coverage."""
        template = SessionTemplate()
        assert template.validate_24_hour_coverage() is True

    def test_complete_coverage_all_hours_covered(self):
        """Test that all 24 hours are covered by windows."""
        template = SessionTemplate()

        # Check each hour of the day
        for hour in range(24):
            gmt_time = datetime(2026, 3, 25, hour, 0, tzinfo=timezone.utc)
            window = template.get_window_at(gmt_time)
            assert window is not None, f"Hour {hour} not covered"

    def test_gap_in_coverage_detected(self):
        """Test that gaps in coverage are detected."""
        # Create a config with a gap
        window_with_gap = {
            "Sydney Open": SessionWindow(
                name="Sydney Open",
                start_gmt="22:00",
                end_gmt="07:00",
            ),
            "Tokyo Open": SessionWindow(
                name="Tokyo Open",
                start_gmt="09:00",  # Gap from 07:00-09:00
                end_gmt="12:00",
            ),
        }

        config = SessionTemplateConfig(
            name="gap-test",
            windows=window_with_gap,
        )
        template = SessionTemplate(config)

        # Should detect incomplete coverage
        assert template.validate_24_hour_coverage() is False


class TestGetWindowAtEdgeCases:
    """Test get_window_at() edge cases."""

    def test_exact_window_start_time(self):
        """Test time exactly at window start is included."""
        template = SessionTemplate()

        # London Open starts at 08:00
        gmt_time = datetime(2026, 3, 25, 8, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "London Open"

    def test_exact_window_end_time(self):
        """Test time exactly at window end is excluded."""
        template = SessionTemplate()

        # London Open ends at 12:00 (exclusive)
        gmt_time = datetime(2026, 3, 25, 12, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        # Should NOT be London Open (it's exclusive at 12:00)
        assert window.name != "London Open"

    def test_dead_zone_exact_start(self):
        """Test Dead Zone exact start at 16:00.

        Note: London-NY Overlap (12:00-17:00) is checked before Dead Zone (16:00-22:00)
        in WINDOW_CHECK_ORDER, so at 16:00 London-NY Overlap is returned.
        This is the correct overlap behavior - Dead Zone is the fallback for times
        not covered by any other window.
        """
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 16, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        # London-NY Overlap (12:00-17:00) is checked first and contains 16:00
        assert window.name == "London-NY Overlap"

    def test_dead_zone_exact_end(self):
        """Test Dead Zone exact end at 22:00 is excluded."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 22, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        # 22:00 is NOT in Dead Zone (end is exclusive)
        assert window.name != "Dead Zone"

    def test_overnight_window_exact_start(self):
        """Test overnight window (Sydney Open) exact start at 22:00."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 22, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "Sydney Open"

    def test_overnight_window_exact_end(self):
        """Test overnight window (Sydney Open) exact end at 07:00 is excluded."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 7, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        # 07:00 is NOT in Sydney Open (end is exclusive)
        assert window.name != "Sydney Open"


class TestFromDictMalformedData:
    """Test from_dict() with malformed data."""

    def test_from_dict_with_missing_keys(self):
        """Test from_dict handles missing optional keys."""
        data = {
            "name": "minimal",
            # windows missing
        }

        template = SessionTemplate.from_dict(data)

        assert template.config.name == "minimal"
        assert len(template.config.windows) == 0

    def test_from_dict_with_invalid_window_data_raises(self):
        """Test from_dict raises ValidationError for invalid window data."""
        data = {
            "name": "invalid-windows",
            "windows": {
                "Test Window": {
                    # Missing required 'name' field - raises ValidationError
                    "start_gmt": "08:00",
                    "end_gmt": "12:00",
                }
            },
        }

        # Should raise ValidationError because name is required
        with pytest.raises(Exception):
            SessionTemplate.from_dict(data)

    def test_from_dict_with_extra_keys(self):
        """Test from_dict ignores extra keys."""
        data = {
            "name": "extra-keys",
            "description": "test",
            "extra_field": "should be ignored",
            "windows": {},
        }

        template = SessionTemplate.from_dict(data)
        assert template.config.name == "extra-keys"
        assert template.config.description == "test"

    def test_from_dict_roundtrip_preserves_data(self):
        """Test that from_dict(to_dict()) preserves all data."""
        original = SessionTemplate()
        data = original.to_dict()

        restored = SessionTemplate.from_dict(data)

        # Check key fields
        assert restored.config.name == original.config.name
        assert restored.config.dead_zone_start == original.config.dead_zone_start
        assert restored.config.dead_zone_end == original.config.dead_zone_end
        assert restored.config.premium_kelly_multiplier == original.config.premium_kelly_multiplier


class TestGetBotTypeMixEdgeCases:
    """Test get_bot_type_mix() edge cases."""

    def test_invalid_window_name_raises(self):
        """Test that invalid window name raises ValueError."""
        template = SessionTemplate()

        with pytest.raises(ValueError, match="Window not found"):
            template.get_bot_type_mix("Invalid Window Name")

    def test_valid_windows_all_return_bot_type_mix(self):
        """Test that all valid windows return a bot type mix."""
        template = SessionTemplate()

        for window_name in template.config.windows.keys():
            mix = template.get_bot_type_mix(window_name)
            assert mix is not None
            assert hasattr(mix, 'orb_pct')
            assert hasattr(mix, 'momentum_pct')
            assert hasattr(mix, 'mean_reversion_pct')
            assert hasattr(mix, 'trend_continuation_pct')

    def test_bot_type_mix_total_pct(self):
        """Test bot type mix total percentage calculation."""
        template = SessionTemplate()

        mix = template.get_bot_type_mix("London Open")
        total = mix.total_pct()
        assert total == 100

    def test_dead_zone_bot_type_mix_total_is_zero(self):
        """Test Dead Zone bot type mix totals to 0."""
        template = SessionTemplate()

        mix = template.get_bot_type_mix("Dead Zone")
        assert mix.validate_sum() is True
        assert mix.total_pct() == 0


class TestGetIntensityEdgeCases:
    """Test get_intensity() edge cases."""

    def test_invalid_window_name_raises(self):
        """Test that invalid window name raises ValueError."""
        template = SessionTemplate()

        with pytest.raises(ValueError, match="Window not found"):
            template.get_intensity("Invalid Window")

    def test_all_windows_return_valid_intensity(self):
        """Test all windows return a valid WindowIntensity."""
        template = SessionTemplate()

        for window_name in template.config.windows.keys():
            intensity = template.get_intensity(window_name)
            assert isinstance(intensity, WindowIntensity)


class TestIsFilteredEdgeCases:
    """Test is_filtered() edge cases."""

    def test_invalid_window_name_returns_false(self):
        """Test is_filtered returns False for invalid window name."""
        template = SessionTemplate()

        result = template.is_filtered("Invalid Window", active_events=["SNB"])
        assert result is False

    def test_window_with_no_filters_returns_false(self):
        """Test window with no event filters returns False."""
        template = SessionTemplate()

        # London Open should have no event filters by default
        result = template.is_filtered("London Open", active_events=["SNB", "FOMC"])
        assert result is False

    def test_filter_matches_active_event(self):
        """Test filter applies when active event matches."""
        template = SessionTemplate()

        # Add an SNB filter to Tokyo-London Overlap
        window = template.config.windows["Tokyo-London Overlap"]
        window.event_filters.append(EventFilter(
            filter_type="SNB",
            affected_windows=["Tokyo-London Overlap"],
            description="SNB event filter"
        ))

        result = template.is_filtered("Tokyo-London Overlap", active_events=["SNB"])
        assert result is True

    def test_filter_not_matching_returns_false(self):
        """Test filter does not apply when event doesn't match."""
        template = SessionTemplate()

        # Add SNB filter
        window = template.config.windows["Tokyo-London Overlap"]
        window.event_filters.append(EventFilter(
            filter_type="SNB",
            affected_windows=["Tokyo-London Overlap"],
            description="SNB event filter"
        ))

        # FOMC should not trigger SNB filter
        result = template.is_filtered("Tokyo-London Overlap", active_events=["FOMC"])
        assert result is False


class TestSessionWindowContainsTime:
    """Test SessionWindow.contains_gmt_time() edge cases."""

    def test_time_before_window_returns_false(self):
        """Test time before window returns False."""
        window = SessionWindow(
            name="Test",
            start_gmt="08:00",
            end_gmt="12:00",
        )
        from datetime import time
        assert window.contains_gmt_time(time(7, 59)) is False

    def test_time_after_window_returns_false(self):
        """Test time after window returns False."""
        window = SessionWindow(
            name="Test",
            start_gmt="08:00",
            end_gmt="12:00",
        )
        from datetime import time
        assert window.contains_gmt_time(time(12, 1)) is False

    def test_overnight_window_before_start_returns_false(self):
        """Test for overnight window, time before start returns False."""
        window = SessionWindow(
            name="Test Overnight",
            start_gmt="22:00",
            end_gmt="07:00",
        )
        from datetime import time
        # 21:00 is before 22:00
        assert window.contains_gmt_time(time(21, 0)) is False

    def test_overnight_window_after_end_returns_false(self):
        """Test for overnight window, time after end returns False."""
        window = SessionWindow(
            name="Test Overnight",
            start_gmt="22:00",
            end_gmt="07:00",
        )
        from datetime import time
        # 07:00 is after end (exclusive)
        assert window.contains_gmt_time(time(7, 0)) is False

    def test_overnight_window_in_middle_returns_true(self):
        """Test for overnight window, time in middle returns True."""
        window = SessionWindow(
            name="Test Overnight",
            start_gmt="22:00",
            end_gmt="07:00",
        )
        from datetime import time
        # 23:00 is in the window
        assert window.contains_gmt_time(time(23, 0)) is True
        # 02:00 is in the window
        assert window.contains_gmt_time(time(2, 0)) is True


class TestPremiumKellyMultiplier:
    """Test premium Kelly multiplier application."""

    def test_premium_kelly_multiplier_is_1_4(self):
        """Test premium Kelly multiplier is 1.4."""
        assert PREMIUM_KELLY_MULTIPLIER == 1.4

    def test_tokyo_london_overlap_gets_premium(self):
        """Test Tokyo-London Overlap receives premium Kelly."""
        template = SessionTemplate()
        multiplier = template.get_kelly_multiplier("Tokyo-London Overlap", base_kelly=1.0)
        assert multiplier == 1.4

    def test_london_open_gets_premium(self):
        """Test London Open receives premium Kelly."""
        template = SessionTemplate()
        multiplier = template.get_kelly_multiplier("London Open", base_kelly=1.0)
        assert multiplier == 1.4

    def test_london_ny_overlap_gets_premium(self):
        """Test London-NY Overlap receives premium Kelly."""
        template = SessionTemplate()
        multiplier = template.get_kelly_multiplier("London-NY Overlap", base_kelly=1.0)
        assert multiplier == 1.4

    def test_non_premium_window_no_change(self):
        """Test non-premium windows get base Kelly."""
        template = SessionTemplate()

        for window_name in ["Sydney Open", "Tokyo Open", "London Mid", "NY Wind-Down"]:
            multiplier = template.get_kelly_multiplier(window_name, base_kelly=1.0)
            assert multiplier == 1.0

    def test_dead_zone_zero_kelly(self):
        """Test Dead Zone has zero Kelly multiplier."""
        template = SessionTemplate()
        window = template.config.windows.get("Dead Zone")
        assert window.kelly_multiplier == 0.0


class TestDeadZoneBoundaries:
    """Test Dead Zone boundary conditions."""

    def test_dead_zone_16_00_included(self):
        """Test 16:00 GMT is in Dead Zone."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 16, 0, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is True

    def test_dead_zone_21_59_included(self):
        """Test 21:59 GMT is in Dead Zone."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 21, 59, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is True

    def test_dead_zone_22_00_excluded(self):
        """Test 22:00 GMT is NOT in Dead Zone."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 22, 0, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is False

    def test_dead_zone_15_59_excluded(self):
        """Test 15:59 GMT is NOT in Dead Zone."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 15, 59, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is False


class TestReloadWithCustomConfig:
    """Test reload() with various configurations."""

    def test_reload_restores_default(self):
        """Test reload() with no args restores default."""
        template = SessionTemplate()
        original_name = template.config.name

        # Modify
        template.config.name = "modified"

        # Reload
        template.reload()

        assert template.config.name != "modified"
        assert template.config.name == "default"

    def test_reload_with_custom_config(self):
        """Test reload() with custom config."""
        template = SessionTemplate()

        custom = SessionTemplateConfig(
            name="custom-config",
            windows={},
        )

        template.reload(custom)

        assert template.config.name == "custom-config"


class TestIsTradingAuthorisedEdgeCases:
    """Test is_trading_authorised() edge cases."""

    def test_trading_blocked_in_dead_zone_regardless_of_events(self):
        """Test trading is blocked in Dead Zone even without event filters."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 18, 0, tzinfo=timezone.utc)

        # Even with no active events, Dead Zone blocks trading
        result = template.is_trading_authorised(gmt_time)
        assert result is False

    def test_trading_allowed_outside_dead_zone_no_events(self):
        """Test trading is allowed outside Dead Zone with no events."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc)

        result = template.is_trading_authorised(gmt_time)
        assert result is True

    def test_trading_blocked_by_event_filter(self):
        """Test trading is blocked when event filter matches."""
        template = SessionTemplate()

        # Add SNB filter to Tokyo-London Overlap
        window = template.config.windows["Tokyo-London Overlap"]
        window.event_filters.append(EventFilter(
            filter_type="SNB",
            affected_windows=["Tokyo-London Overlap"],
            description="SNB"
        ))

        # During Tokyo-London Overlap with SNB event active
        gmt_time = datetime(2026, 3, 25, 7, 30, tzinfo=timezone.utc)
        result = template.is_trading_authorised(gmt_time, active_events=["SNB"])
        assert result is False
