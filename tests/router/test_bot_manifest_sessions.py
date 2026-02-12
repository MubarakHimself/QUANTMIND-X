"""
Tests for BotManifest Session Preferences

Tests TimeWindow and PreferredConditions serialization/deserialization
and backward compatibility.
"""

import pytest
from datetime import datetime
from src.router.bot_manifest import (
    BotManifest,
    TimeWindow,
    PreferredConditions,
    StrategyType,
    TradeFrequency,
    BrokerType
)


class TestTimeWindow:
    """Test TimeWindow dataclass."""

    def test_time_window_creation(self):
        """Test creating TimeWindow."""
        window = TimeWindow(
            start="09:50",
            end="10:10",
            timezone="America/New_York"
        )
        assert window.start == "09:50"
        assert window.end == "10:10"
        assert window.timezone == "America/New_York"

    def test_time_window_to_dict(self):
        """Test TimeWindow serialization."""
        window = TimeWindow(start="08:00", end="09:00", timezone="Europe/London")
        data = window.to_dict()
        assert data == {
            "start": "08:00",
            "end": "09:00",
            "timezone": "Europe/London"
        }

    def test_time_window_from_dict(self):
        """Test TimeWindow deserialization."""
        data = {
            "start": "09:30",
            "end": "10:30",
            "timezone": "America/Chicago"
        }
        window = TimeWindow.from_dict(data)
        assert window.start == "09:30"
        assert window.end == "10:30"
        assert window.timezone == "America/Chicago"

    def test_time_window_from_dict_defaults(self):
        """Test TimeWindow deserialization with defaults."""
        data = {"start": "00:00"}
        window = TimeWindow.from_dict(data)
        assert window.start == "00:00"
        assert window.end == "23:59"  # Default
        assert window.timezone == "UTC"  # Default


class TestPreferredConditions:
    """Test PreferredConditions dataclass."""

    def test_preferred_conditions_sessions_only(self):
        """Test PreferredConditions with sessions only."""
        prefs = PreferredConditions(sessions=["LONDON", "NEW_YORK"])
        assert prefs.sessions == ["LONDON", "NEW_YORK"]
        assert prefs.time_windows == []
        assert prefs.min_volatility is None
        assert prefs.max_volatility is None

    def test_preferred_conditions_with_windows(self):
        """Test PreferredConditions with time windows."""
        window1 = TimeWindow(start="09:50", end="10:10", timezone="America/New_York")
        window2 = TimeWindow(start="08:00", end="09:00", timezone="Europe/London")
        prefs = PreferredConditions(
            sessions=["NEW_YORK"],
            time_windows=[window1, window2],
            min_volatility=0.001,
            max_volatility=0.005
        )
        assert len(prefs.time_windows) == 2
        assert prefs.min_volatility == 0.001
        assert prefs.max_volatility == 0.005

    def test_preferred_conditions_to_dict(self):
        """Test PreferredConditions serialization."""
        window = TimeWindow(start="09:50", end="10:10", timezone="America/New_York")
        prefs = PreferredConditions(
            sessions=["LONDON"],
            time_windows=[window],
            min_volatility=0.0005,
            max_volatility=0.003
        )
        data = prefs.to_dict()
        assert data["sessions"] == ["LONDON"]
        assert len(data["time_windows"]) == 1
        assert data["time_windows"][0]["timezone"] == "America/New_York"
        assert data["min_volatility"] == 0.0005
        assert data["max_volatility"] == 0.003

    def test_preferred_conditions_from_dict(self):
        """Test PreferredConditions deserialization."""
        data = {
            "sessions": ["ASIAN", "OVERLAP"],
            "time_windows": [
                {"start": "00:00", "end": "02:00", "timezone": "Asia/Tokyo"}
            ],
            "min_volatility": 0.001,
            "max_volatility": 0.01
        }
        prefs = PreferredConditions.from_dict(data)
        assert prefs.sessions == ["ASIAN", "OVERLAP"]
        assert len(prefs.time_windows) == 1
        assert prefs.time_windows[0].start == "00:00"
        assert prefs.min_volatility == 0.001

    def test_preferred_conditions_empty_from_dict(self):
        """Test PreferredConditions deserialization with empty data."""
        data = {}
        prefs = PreferredConditions.from_dict(data)
        assert prefs.sessions == []
        assert prefs.time_windows == []
        assert prefs.min_volatility is None
        assert prefs.max_volatility is None


class TestBotManifestWithSessions:
    """Test BotManifest with session preferences."""

    def test_bot_manifest_with_preferred_conditions(self):
        """Test BotManifest with PreferredConditions."""
        window = TimeWindow(start="09:50", end="10:10", timezone="America/New_York")
        prefs = PreferredConditions(sessions=["NEW_YORK"], time_windows=[window])
        manifest = BotManifest(
            bot_id="test_bot",
            name="Test Bot",
            description="Test bot with sessions",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            preferred_conditions=prefs
        )
        assert manifest.preferred_conditions is not None
        assert manifest.preferred_conditions.sessions == ["NEW_YORK"]

    def test_bot_manifest_to_dict_with_sessions(self):
        """Test BotManifest serialization with PreferredConditions."""
        window = TimeWindow(start="08:00", end="09:00", timezone="Europe/London")
        prefs = PreferredConditions(sessions=["LONDON"], time_windows=[window])
        manifest = BotManifest(
            bot_id="test_bot",
            name="Test Bot",
            description="Test bot",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            preferred_conditions=prefs
        )
        data = manifest.to_dict()
        assert "preferred_conditions" in data
        assert data["preferred_conditions"]["sessions"] == ["LONDON"]

    def test_bot_manifest_from_dict_with_sessions(self):
        """Test BotManifest deserialization with PreferredConditions."""
        data = {
            "bot_id": "test_bot",
            "name": "Test Bot",
            "description": "Test",
            "strategy_type": "STRUCTURAL",
            "frequency": "LOW",
            "preferred_conditions": {
                "sessions": ["OVERLAP"],
                "time_windows": [],
                "min_volatility": 0.002,
                "max_volatility": 0.01
            }
        }
        manifest = BotManifest.from_dict(data)
        assert manifest.preferred_conditions is not None
        assert manifest.preferred_conditions.sessions == ["OVERLAP"]
        assert manifest.preferred_conditions.min_volatility == 0.002


class TestBotManifestBackwardCompatibility:
    """Test backward compatibility with manifests without sessions."""

    def test_bot_manifest_without_preferred_conditions(self):
        """Test BotManifest without PreferredConditions (backward compatible)."""
        manifest = BotManifest(
            bot_id="legacy_bot",
            name="Legacy Bot",
            description="Old bot without session preferences",
            strategy_type=StrategyType.SCALPER,
            frequency=TradeFrequency.HIGH
            # No preferred_conditions - should default to None
        )
        assert manifest.preferred_conditions is None

    def test_bot_manifest_to_dict_without_sessions(self):
        """Test serialization without PreferredConditions."""
        manifest = BotManifest(
            bot_id="legacy_bot",
            name="Legacy Bot",
            strategy_type=StrategyType.SCALPER,
            frequency=TradeFrequency.HIGH
        )
        data = manifest.to_dict()
        # Should not have preferred_conditions key if None
        assert "preferred_conditions" not in data or data["preferred_conditions"] is None

    def test_bot_manifest_from_dict_without_sessions(self):
        """Test deserialization without PreferredConditions."""
        data = {
            "bot_id": "legacy_bot",
            "name": "Legacy Bot",
            "description": "Old bot",
            "strategy_type": "SCALPER",
            "frequency": "HIGH",
            "min_capital_req": 100.0,
            "preferred_broker_type": "RAW_ECN",
            "prop_firm_safe": True,
            "symbols": ["EURUSD"],
            "timeframes": ["M1"],
            "max_positions": 1,
            "max_daily_trades": 100,
            "created_at": "2026-02-12T10:00:00",
            "last_trade_at": None,
            "total_trades": 0,
            "win_rate": 0.0,
            "tags": []
        }
        manifest = BotManifest.from_dict(data)
        assert manifest.preferred_conditions is None

    def test_bot_manifest_partial_session_data(self):
        """Test manifest with partial session data (backward compatible)."""
        data = {
            "bot_id": "partial_bot",
            "name": "Partial Bot",
            "description": "Bot with some session data",
            "strategy_type": "STRUCTURAL",
            "frequency": "LOW",
            "min_capital_req": 50.0,
            "preferred_broker_type": "STANDARD",
            "prop_firm_safe": True,
            "symbols": ["GBPUSD"],
            "timeframes": ["M15"],
            "max_positions": 1,
            "max_daily_trades": 5,
            "created_at": "2026-02-12T10:00:00",
            "last_trade_at": None,
            "total_trades": 0,
            "win_rate": 0.6,
            "tags": ["@primal"],
            "preferred_conditions": {
                "sessions": ["LONDON"],
                "time_windows": []
            }
        }
        manifest = BotManifest.from_dict(data)
        assert manifest.preferred_conditions is not None
        assert manifest.preferred_conditions.sessions == ["LONDON"]
        assert manifest.preferred_conditions.time_windows == []
        assert manifest.preferred_conditions.min_volatility is None


class TestExampleManifests:
    """Test example manifests have session preferences."""

    def test_ict_macro_example_has_sessions(self):
        """Test ict_macro_01 example has NY session preference."""
        from src.router.bot_manifest import EXAMPLE_MANIFESTS

        # Find ict_macro_01
        ict_bot = None
        for manifest in EXAMPLE_MANIFESTS:
            if manifest.bot_id == "ict_macro_01":
                ict_bot = manifest
                break

        assert ict_bot is not None
        assert ict_bot.preferred_conditions is not None
        assert "NEW_YORK" in ict_bot.preferred_conditions.sessions
        assert len(ict_bot.preferred_conditions.time_windows) > 0

    def test_london_breakout_example_has_sessions(self):
        """Test london_breakout_01 example has London session preference."""
        from src.router.bot_manifest import EXAMPLE_MANIFESTS

        # Find london_breakout_01
        london_bot = None
        for manifest in EXAMPLE_MANIFESTS:
            if manifest.bot_id == "london_breakout_01":
                london_bot = manifest
                break

        assert london_bot is not None
        assert london_bot.preferred_conditions is not None
        assert "LONDON" in london_bot.preferred_conditions.sessions

    def test_overlap_scalper_example_has_sessions(self):
        """Test overlap_scalper_01 example has Overlap session preference."""
        from src.router.bot_manifest import EXAMPLE_MANIFESTS

        # Find overlap_scalper_01
        overlap_bot = None
        for manifest in EXAMPLE_MANIFESTS:
            if manifest.bot_id == "overlap_scalper_01":
                overlap_bot = manifest
                break

        assert overlap_bot is not None
        assert overlap_bot.preferred_conditions is not None
        assert "OVERLAP" in overlap_bot.preferred_conditions.sessions

    def test_legacy_bots_without_sessions(self):
        """Test other example manifests don't break without sessions."""
        from src.router.bot_manifest import EXAMPLE_MANIFESTS

        # Check hft_scalper_01 (no sessions)
        for manifest in EXAMPLE_MANIFESTS:
            if manifest.bot_id == "hft_scalper_01":
                assert manifest.preferred_conditions is None
                break

        # Check amd_hunter_01 (no sessions)
        for manifest in EXAMPLE_MANIFESTS:
            if manifest.bot_id == "amd_hunter_01":
                assert manifest.preferred_conditions is None
                break
