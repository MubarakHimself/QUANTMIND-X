"""
Tests for Session Template Module.

Story 16.2: Session Template Class — Configurable 10-Window Canonical Cycle

Tests cover:
- SessionTemplate initialization with all 10 windows
- get_window_at() for each window boundary
- is_premium_window() for premium vs non-premium windows
- is_dead_zone() for times inside and outside Dead Zone
- is_trading_authorised() returns False in Dead Zone
- get_kelly_multiplier() applies premium boost only to premium windows
- BotTypeMix validation (percentages sum to 100%)
- is_filtered() with active events
- to_dict() and from_dict() serialization roundtrip
- reload() updates configuration
- GMT midnight crossing (window spans 00:00 GMT)
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
    PREMIUM_WINDOW_NAMES,
)


class TestSessionTemplateInitialization:
    """Test SessionTemplate initialization with all 10 windows."""

    def test_default_template_creates_10_windows(self):
        """Test that default template creates all 10 canonical windows."""
        template = SessionTemplate()
        assert len(template.config.windows) == 10

    def test_all_canonical_windows_present(self):
        """Test that all 10 canonical windows are defined."""
        template = SessionTemplate()
        for canonical in CanonicalSessionWindow:
            assert canonical.value in template.config.windows
            window = template.config.windows[canonical.value]
            assert window.name == canonical.value

    def test_dead_zone_constants(self):
        """Test Dead Zone constants are correctly defined."""
        assert DEAD_ZONE_START == 16
        assert DEAD_ZONE_END == 22

    def test_premium_kelly_multiplier_constant(self):
        """Test Premium Kelly multiplier is correctly defined."""
        assert PREMIUM_KELLY_MULTIPLIER == 1.4

    def test_premium_windows_list(self):
        """Test premium windows are correctly identified."""
        expected = ["Tokyo-London Overlap", "London Open", "London-NY Overlap"]
        assert PREMIUM_WINDOW_NAMES == expected


class TestGetWindowAt:
    """Test get_window_at() for each window boundary."""

    def test_sydney_open_window(self):
        """Test Sydney Open window detection at 23:00 GMT."""
        template = SessionTemplate()
        # 23:00 GMT should be in Sydney Open (22:00-07:00)
        gmt_time = datetime(2026, 3, 25, 23, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "Sydney Open"
        assert window.intensity == WindowIntensity.LOW

    def test_tokyo_london_overlap_window(self):
        """Test Tokyo-London Overlap window detection at 07:30 GMT."""
        template = SessionTemplate()
        # 07:30 GMT should be in Tokyo-London Overlap (07:00-08:00)
        gmt_time = datetime(2026, 3, 25, 7, 30, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "Tokyo-London Overlap"
        assert window.intensity == WindowIntensity.HIGH

    def test_london_open_window(self):
        """Test London Open window detection at 09:00 GMT."""
        template = SessionTemplate()
        # 09:00 GMT should be in London Open (08:00-12:00)
        gmt_time = datetime(2026, 3, 25, 9, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "London Open"
        assert window.intensity == WindowIntensity.PREMIUM

    def test_london_ny_overlap_window(self):
        """Test London-NY Overlap window detection at 14:00 GMT."""
        template = SessionTemplate()
        # 14:00 GMT should be in London-NY Overlap (12:00-17:00)
        gmt_time = datetime(2026, 3, 25, 14, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "London-NY Overlap"
        assert window.intensity == WindowIntensity.PREMIUM

    def test_dead_zone_window(self):
        """Test Dead Zone window detection at 21:30 GMT (after NY Wind-Down ends)."""
        template = SessionTemplate()
        # 21:30 GMT should be in Dead Zone (16:00-22:00)
        # Note: Dead Zone (16:00-22:00) overlaps with London-NY Overlap (12:00-17:00) and
        # NY Wind-Down (17:00-21:00). We test at 21:30 GMT when only Dead Zone is active
        # (after NY Wind-Down ends at 21:00 GMT).
        gmt_time = datetime(2026, 3, 25, 21, 30, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "Dead Zone"
        assert window.kelly_multiplier == 0.0

    def test_ny_wind_down_window(self):
        """Test NY Wind-Down window detection at 19:00 GMT."""
        template = SessionTemplate()
        # 19:00 GMT should be in NY Wind-Down (17:00-21:00)
        gmt_time = datetime(2026, 3, 25, 19, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "NY Wind-Down"
        assert window.intensity == WindowIntensity.LOW

    def test_midnight_crossing_window(self):
        """Test window spanning midnight - Sydney Open at 02:00 GMT."""
        template = SessionTemplate()
        # 02:00 GMT should be in Sydney Open (22:00-07:00)
        gmt_time = datetime(2026, 3, 25, 2, 0, tzinfo=timezone.utc)
        window = template.get_window_at(gmt_time)
        assert window.name == "Sydney Open"


class TestIsPremiumWindow:
    """Test is_premium_window() for premium vs non-premium windows."""

    def test_tokyo_london_overlap_is_premium(self):
        """Test Tokyo-London Overlap is identified as premium."""
        template = SessionTemplate()
        assert template.is_premium_window("Tokyo-London Overlap") is True

    def test_london_open_is_premium(self):
        """Test London Open is identified as premium."""
        template = SessionTemplate()
        assert template.is_premium_window("London Open") is True

    def test_london_ny_overlap_is_premium(self):
        """Test London-NY Overlap is identified as premium."""
        template = SessionTemplate()
        assert template.is_premium_window("London-NY Overlap") is True

    def test_sydney_open_not_premium(self):
        """Test Sydney Open is not premium."""
        template = SessionTemplate()
        assert template.is_premium_window("Sydney Open") is False

    def test_london_mid_not_premium(self):
        """Test London Mid is not premium."""
        template = SessionTemplate()
        assert template.is_premium_window("London Mid") is False

    def test_dead_zone_not_premium(self):
        """Test Dead Zone is not premium."""
        template = SessionTemplate()
        assert template.is_premium_window("Dead Zone") is False


class TestIsDeadZone:
    """Test is_dead_zone() for times inside and outside Dead Zone."""

    def test_1800_gmt_in_dead_zone(self):
        """Test 18:00 GMT is in Dead Zone."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 18, 0, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is True

    def test_1600_gmt_in_dead_zone(self):
        """Test 16:00 GMT is at start of Dead Zone."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 16, 0, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is True

    def test_2200_gmt_not_in_dead_zone(self):
        """Test 22:00 GMT is at end of Dead Zone (not in Dead Zone)."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 22, 0, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is False

    def test_1500_gmt_not_in_dead_zone(self):
        """Test 15:00 GMT is not in Dead Zone."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 15, 0, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is False

    def test_1000_gmt_not_in_dead_zone(self):
        """Test 10:00 GMT is not in Dead Zone."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc)
        assert template.is_dead_zone(gmt_time) is False


class TestIsTradingAuthorised:
    """Test is_trading_authorised() returns False in Dead Zone."""

    def test_trading_allowed_outside_dead_zone(self):
        """Test trading is allowed outside Dead Zone."""
        template = SessionTemplate()
        # 10:00 GMT is not in Dead Zone
        gmt_time = datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc)
        assert template.is_trading_authorised(gmt_time) is True

    def test_trading_blocked_in_dead_zone(self):
        """Test trading is blocked in Dead Zone."""
        template = SessionTemplate()
        # 18:00 GMT is in Dead Zone
        gmt_time = datetime(2026, 3, 25, 18, 0, tzinfo=timezone.utc)
        assert template.is_trading_authorised(gmt_time) is False

    def test_trading_blocked_at_dead_zone_start(self):
        """Test trading is blocked at 16:00 GMT (start of Dead Zone)."""
        template = SessionTemplate()
        gmt_time = datetime(2026, 3, 25, 16, 0, tzinfo=timezone.utc)
        assert template.is_trading_authorised(gmt_time) is False


class TestGetKellyMultiplier:
    """Test get_kelly_multiplier() applies premium boost only to premium windows."""

    def test_premium_window_gets_boost(self):
        """Test premium windows receive Kelly boost."""
        template = SessionTemplate()
        # London Open is premium
        multiplier = template.get_kelly_multiplier("London Open", base_kelly=1.0)
        assert multiplier == 1.4  # 1.0 * 1.4

    def test_non_premium_window_no_boost(self):
        """Test non-premium windows receive base Kelly."""
        template = SessionTemplate()
        # Sydney Open is not premium
        multiplier = template.get_kelly_multiplier("Sydney Open", base_kelly=1.0)
        assert multiplier == 1.0

    def test_dead_zone_zero_kelly(self):
        """Test Dead Zone has zero Kelly multiplier."""
        template = SessionTemplate()
        # Dead Zone should have 0.0 Kelly
        window = template.config.windows.get("Dead Zone")
        assert window.kelly_multiplier == 0.0


class TestBotTypeMix:
    """Test BotTypeMix validation."""

    def test_orb_dominant_mix(self):
        """Test ORB-dominant bot type mix for Sydney Open."""
        template = SessionTemplate()
        mix = template.get_bot_type_mix("Sydney Open")
        assert mix.orb_pct == 70
        assert mix.momentum_pct == 30
        assert mix.mean_reversion_pct == 0
        assert mix.trend_continuation_pct == 0
        assert mix.validate_sum() is True

    def test_mean_reversion_mix(self):
        """Test mean reversion mix for Tokyo Open."""
        template = SessionTemplate()
        mix = template.get_bot_type_mix("Tokyo Open")
        assert mix.orb_pct == 0
        assert mix.momentum_pct == 30
        assert mix.mean_reversion_pct == 70
        assert mix.trend_continuation_pct == 0
        assert mix.validate_sum() is True

    def test_mixed_mix_london_ny_overlap(self):
        """Test mixed bot type mix for London-NY Overlap."""
        template = SessionTemplate()
        mix = template.get_bot_type_mix("London-NY Overlap")
        assert mix.orb_pct == 55
        assert mix.momentum_pct == 45
        assert mix.mean_reversion_pct == 0
        assert mix.trend_continuation_pct == 0
        assert mix.validate_sum() is True

    def test_dead_zone_no_bots(self):
        """Test Dead Zone has zero allocation to all bot types."""
        template = SessionTemplate()
        mix = template.get_bot_type_mix("Dead Zone")
        assert mix.orb_pct == 0
        assert mix.momentum_pct == 0
        assert mix.mean_reversion_pct == 0
        assert mix.trend_continuation_pct == 0
        assert mix.validate_sum() is True


class TestIsFiltered:
    """Test is_filtered() with active events."""

    def test_no_filter_without_active_events(self):
        """Test no filter when no active events."""
        template = SessionTemplate()
        assert template.is_filtered("Tokyo Open", active_events=[]) is False

    def test_filter_applies_with_matching_event(self):
        """Test filter applies when event matches."""
        template = SessionTemplate()
        # Add SNB event filter to Tokyo-London Overlap
        window = template.config.windows["Tokyo-London Overlap"]
        window.event_filters.append(EventFilter(
            filter_type="SNB",
            affected_windows=["Tokyo-London Overlap"],
            description="Swiss National Bank event filter"
        ))
        assert template.is_filtered("Tokyo-London Overlap", active_events=["SNB"]) is True

    def test_filter_not_applies_with_different_event(self):
        """Test filter does not apply for different event."""
        template = SessionTemplate()
        # Add SNB event filter
        window = template.config.windows["Tokyo-London Overlap"]
        window.event_filters.append(EventFilter(
            filter_type="SNB",
            affected_windows=["Tokyo-London Overlap"],
            description="Swiss National Bank event filter"
        ))
        # FOMC event should not trigger SNB filter
        assert template.is_filtered("Tokyo-London Overlap", active_events=["FOMC"]) is False


class TestSerialization:
    """Test to_dict() and from_dict() serialization roundtrip."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        template = SessionTemplate()
        data = template.to_dict()
        assert data["name"] == "default"
        assert len(data["windows"]) == 10
        assert data["dead_zone_start"] == 16
        assert data["dead_zone_end"] == 22
        assert data["premium_kelly_multiplier"] == 1.4

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        original = SessionTemplate()
        data = original.to_dict()
        restored = SessionTemplate.from_dict(data)

        assert restored.config.name == original.config.name
        assert len(restored.config.windows) == len(original.config.windows)
        assert restored.config.dead_zone_start == original.config.dead_zone_start

    def test_roundtrip_preserves_windows(self):
        """Test roundtrip preserves all window data."""
        original = SessionTemplate()
        data = original.to_dict()
        restored = SessionTemplate.from_dict(data)

        for window_name in original.config.windows:
            original_window = original.config.windows[window_name]
            restored_window = restored.config.windows[window_name]
            assert original_window.start_gmt == restored_window.start_gmt
            assert original_window.end_gmt == restored_window.end_gmt
            assert original_window.intensity == restored_window.intensity
            assert original_window.bot_type_mix.orb_pct == restored_window.bot_type_mix.orb_pct


class TestReload:
    """Test reload() updates configuration."""

    def test_reload_with_default(self):
        """Test reload restores default configuration."""
        template = SessionTemplate()
        # Modify the config
        template.config.name = "modified"

        # Reload
        template.reload()
        assert template.config.name == "default"

    def test_reload_with_custom_config(self):
        """Test reload applies new configuration."""
        template = SessionTemplate()

        # Create custom config
        custom_config = SessionTemplateConfig(
            name="custom",
            description="Custom template",
            windows={},
        )

        # Reload with custom
        template.reload(custom_config)
        assert template.config.name == "custom"


class TestIsActiveTradingWindow:
    """Test is_active_trading_window() method."""

    def test_dead_zone_not_active(self):
        """Test Dead Zone is not an active trading window."""
        template = SessionTemplate()
        assert template.is_active_trading_window("Dead Zone") is False

    def test_london_open_is_active(self):
        """Test London Open is an active trading window."""
        template = SessionTemplate()
        assert template.is_active_trading_window("London Open") is True

    def test_sydney_open_is_active(self):
        """Test Sydney Open is an active trading window."""
        template = SessionTemplate()
        assert template.is_active_trading_window("Sydney Open") is True


class TestGetIntensity:
    """Test get_intensity() method."""

    def test_premium_intensity_london_open(self):
        """Test London Open has PREMIUM intensity."""
        template = SessionTemplate()
        intensity = template.get_intensity("London Open")
        assert intensity == WindowIntensity.PREMIUM

    def test_high_intensity_tokyo_london_overlap(self):
        """Test Tokyo-London Overlap has HIGH intensity."""
        template = SessionTemplate()
        intensity = template.get_intensity("Tokyo-London Overlap")
        assert intensity == WindowIntensity.HIGH

    def test_very_low_intensity_sydney_tokyo_overlap(self):
        """Test Sydney-Tokyo Overlap has VERY_LOW intensity."""
        template = SessionTemplate()
        intensity = template.get_intensity("Sydney-Tokyo Overlap")
        assert intensity == WindowIntensity.VERY_LOW

    def test_invalid_window_raises(self):
        """Test getting intensity for invalid window raises ValueError."""
        template = SessionTemplate()
        with pytest.raises(ValueError, match="Window not found"):
            template.get_intensity("Invalid Window")


class TestSessionWindowContainsTime:
    """Test SessionWindow.contains_gmt_time() for midnight crossing."""

    def test_overnight_window_spanning_midnight(self):
        """Test window spanning midnight correctly contains times."""
        from datetime import time

        window = SessionWindow(
            name="Test Overnight",
            start_gmt="22:00",
            end_gmt="07:00",
        )

        # 23:00 should be in window
        assert window.contains_gmt_time(time(23, 0)) is True
        # 02:00 should be in window
        assert window.contains_gmt_time(time(2, 0)) is True
        # 07:00 should NOT be in window (end is exclusive)
        assert window.contains_gmt_time(time(7, 0)) is False
        # 21:00 should NOT be in window
        assert window.contains_gmt_time(time(21, 0)) is False

    def test_normal_window(self):
        """Test normal window (start < end) correctly contains times."""
        from datetime import time

        window = SessionWindow(
            name="Test Normal",
            start_gmt="08:00",
            end_gmt="12:00",
        )

        # 09:00 should be in window
        assert window.contains_gmt_time(time(9, 0)) is True
        # 08:00 should be in window
        assert window.contains_gmt_time(time(8, 0)) is True
        # 12:00 should NOT be in window (end is exclusive)
        assert window.contains_gmt_time(time(12, 0)) is False
        # 07:59 should NOT be in window
        assert window.contains_gmt_time(time(7, 59)) is False


class TestEventFilter:
    """Test EventFilter model."""

    def test_event_filter_creation(self):
        """Test EventFilter model creation."""
        event_filter = EventFilter(
            filter_type="SNB",
            affected_windows=["Tokyo Open", "Tokyo-London Overlap"],
            description="Swiss National Bank event"
        )
        assert event_filter.filter_type == "SNB"
        assert "Tokyo Open" in event_filter.affected_windows
        assert event_filter.description == "Swiss National Bank event"
