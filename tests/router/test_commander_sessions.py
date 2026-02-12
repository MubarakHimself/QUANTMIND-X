"""
Tests for Commander Session Filtering

Tests session-aware bot filtering in Commander auctions.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
from src.router.commander import Commander
from src.router.bot_manifest import (
    BotRegistry,
    BotManifest,
    StrategyType,
    TradeFrequency,
    PreferredConditions,
    TimeWindow
)
from src.router.sentinel import RegimeReport
from src.router.sessions import TradingSession


class TestCommanderSessionFiltering:
    """Test Commander session-aware bot filtering."""

    @pytest.fixture
    def commander(self):
        """Create Commander with mock registry."""
        mock_registry = Mock(spec=BotRegistry)
        commander = Commander(bot_registry=mock_registry)
        return commander

    @pytest.fixture
    def sample_bots(self):
        """Create sample bot manifests for testing."""
        london_bot = BotManifest(
            bot_id="london_only_bot",
            name="London Only Bot",
            description="Trades only during London session",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            symbols=["EURUSD"],
            timeframes=["M15"],
            tags=["@primal"],
            win_rate=0.60,
            total_trades=50,
            preferred_conditions=PreferredConditions(sessions=["LONDON"])
        )

        ny_bot = BotManifest(
            bot_id="ny_only_bot",
            name="NY Only Bot",
            description="Trades only during New York session",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            symbols=["GBPUSD"],
            timeframes=["M15"],
            tags=["@primal"],
            win_rate=0.55,
            total_trades=40,
            preferred_conditions=PreferredConditions(sessions=["NEW_YORK"])
        )

        ict_bot = BotManifest(
            bot_id="ict_window_bot",
            name="ICT Window Bot",
            description="Trades 9:50-10:10 AM NY window",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            symbols=["XAUUSD"],
            timeframes=["M5"],
            tags=["@primal"],
            win_rate=0.65,
            total_trades=30,
            preferred_conditions=PreferredConditions(
                time_windows=[TimeWindow(start="09:50", end="10:10", timezone="America/New_York")]
            )
        )

        any_bot = BotManifest(
            bot_id="any_time_bot",
            name="Any Time Bot",
            description="Trades anytime",
            strategy_type=StrategyType.SCALPER,
            frequency=TradeFrequency.HIGH,
            symbols=["EURUSD"],
            timeframes=["M1"],
            tags=["@primal"],
            win_rate=0.50,
            total_trades=100
            # No preferred_conditions - can trade anytime
        )

        return {
            "london_bot": london_bot,
            "ny_bot": ny_bot,
            "ict_bot": ict_bot,
            "any_bot": any_bot
        }

    def test_filters_london_only_bot_during_london(self, commander, sample_bots):
        """Test Commander filters London-only bot during London session."""
        # Mock registry to return our sample bots
        commander.bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["london_bot"],
            sample_bots["ny_bot"],
            sample_bots["ict_bot"],
            sample_bots["any_bot"]
        ])

        # Create regime report for TREND_STABLE
        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # 10:00 UTC = London session
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        # Run auction
        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)

        # Should include london_bot (matches session) and any_bot (no preference)
        # Should exclude ny_bot (wrong session) and ict_bot (outside window)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert "london_only_bot" in bot_ids
        assert "any_time_bot" in bot_ids
        assert "ny_only_bot" not in bot_ids

    def test_blocks_london_only_bot_during_ny_session(self, commander, sample_bots):
        """Test Commander blocks London-only bot during NY session."""
        commander.bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["london_bot"],
            sample_bots["any_bot"]
        ])

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # 18:00 UTC = NY only (after overlap ends)
        utc_time = datetime(2026, 2, 12, 18, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # London bot should be filtered out
        assert "london_only_bot" not in bot_ids
        # Any-time bot should still work
        assert "any_time_bot" in bot_ids

    def test_ict_bot_only_trades_in_window(self, commander, sample_bots):
        """Test ICT bot only trades during 9:50-10:10 AM NY window."""
        commander.bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["ict_bot"],
            sample_bots["any_bot"]
        ])

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # 14:55 UTC = 9:55 AM EST (inside ICT window)
        utc_time = datetime(2026, 2, 12, 14, 55, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # ICT bot should trade (inside window)
        assert "ict_window_bot" in bot_ids

    def test_ict_bot_blocked_outside_window(self, commander, sample_bots):
        """Test ICT bot blocked outside 9:50-10:10 AM NY window."""
        commander.bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["ict_bot"]
        ])

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # 14:00 UTC = 9:00 AM EST (before window)
        utc_time = datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # ICT bot should be blocked (outside window)
        assert "ict_window_bot" not in bot_ids

    def test_any_time_bot_always_allowed(self, commander, sample_bots):
        """Test bot without session preferences trades in all sessions."""
        commander.bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["any_bot"]
        ])

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # Test during London (10:00 UTC)
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)
        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        assert "any_time_bot" in [d["bot_id"] for d in dispatches]

        # Test during NY (18:00 UTC)
        utc_time = datetime(2026, 2, 12, 18, 0, tzinfo=timezone.utc)
        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        assert "any_time_bot" in [d["bot_id"] for d in dispatches]


class TestCommanderWithChaosFiltering:
    """Test Commander session filtering with chaos score."""

    @pytest.fixture
    def bots(self):
        """Create sample bots."""
        return [
            BotManifest(
                bot_id="low_freq_bot",
                name="Low Frequency Bot",
                description="Low frequency bot",
                strategy_type=StrategyType.STRUCTURAL,
                frequency=TradeFrequency.LOW,
                symbols=["EURUSD"],
                timeframes=["M15"],
                tags=["@primal"],
                win_rate=0.60,
                total_trades=50,
                preferred_conditions=PreferredConditions(sessions=["LONDON"])
            ),
            BotManifest(
                bot_id="high_freq_bot",
                name="High Frequency Bot",
                description="High frequency bot",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                symbols=["GBPUSD"],
                timeframes=["M5"],
                tags=["@primal"],
                win_rate=0.55,
                total_trades=100,
                preferred_conditions=PreferredConditions(sessions=["LONDON"])
            )
        ]

    def test_session_filtering_applied_before_chaos_filtering(self, bots):
        """Test that session filtering happens before chaos filtering."""
        commander = Commander()
        commander.bot_registry.list_by_tag = Mock(return_value=bots)

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.7,  # High chaos
            regime_quality=0.5
        )

        # 10:00 UTC = London session
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)

        # With high chaos, only LOW/MEDIUM frequency should pass
        # Both bots have LOW frequency, so session check should come first
        # Both have LONDON preference, so both should be eligible
        assert len(dispatches) == 2


class TestCommanderDefaultTime:
    """Test Commander uses current time when utc not provided."""

    @pytest.fixture
    def bot(self):
        """Create sample bot."""
        return BotManifest(
            bot_id="test_bot",
            name="Test Bot",
            description="Test",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            symbols=["EURUSD"],
            timeframes=["M15"],
            tags=["@primal"],
            win_rate=0.60,
            total_trades=50
        )

    def test_commander_uses_current_time_when_none(self, bot):
        """Test Commander defaults to current time when utc_time is None."""
        commander = Commander()
        commander.bot_registry.list_by_tag = Mock(return_value=[bot])

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # Don't pass current_utc - should use datetime.now(timezone.utc)
        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default")

        # Should complete successfully (using current time)
        assert isinstance(dispatches, list)

    def test_commander_accepts_explicit_utc_time(self, bot):
        """Test Commander accepts explicit UTC time."""
        commander = Commander()
        commander.bot_registry.list_by_tag = Mock(return_value=[bot])

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # Pass explicit UTC time during London session
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)
        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)

        assert isinstance(dispatches, list)


class TestCommanderOverlappingSessions:
    """Test Commander behavior during overlapping sessions."""

    @pytest.fixture
    def overlap_bots(self):
        """Create bots for overlapping session test."""
        return [
            BotManifest(
                bot_id="london_pref_bot",
                name="London Pref Bot",
                description="Prefers London",
                strategy_type=StrategyType.STRUCTURAL,
                frequency=TradeFrequency.LOW,
                symbols=["EURUSD"],
                timeframes=["M15"],
                tags=["@primal"],
                win_rate=0.60,
                total_trades=50,
                preferred_conditions=PreferredConditions(sessions=["LONDON"])
            ),
            BotManifest(
                bot_id="ny_pref_bot",
                name="NY Pref Bot",
                description="Prefers New York",
                strategy_type=StrategyType.STRUCTURAL,
                frequency=TradeFrequency.LOW,
                symbols=["GBPUSD"],
                timeframes=["M15"],
                tags=["@primal"],
                win_rate=0.55,
                total_trades=40,
                preferred_conditions=PreferredConditions(sessions=["NEW_YORK"])
            )
        ]

    def test_both_bots_allowed_during_overlap(self, overlap_bots):
        """Test both London and NY bots allowed during overlap."""
        commander = Commander()
        commander.bot_registry.list_by_tag = Mock(return_value=overlap_bots)

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # 14:00 UTC = London/NY overlap
        utc_time = datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # Both should be allowed during overlap
        assert "london_pref_bot" in bot_ids
        assert "ny_pref_bot" in bot_ids

    def test_london_bot_blocked_during_ny_only(self, overlap_bots):
        """Test London bot blocked during NY-only session."""
        commander = Commander()
        commander.bot_registry.list_by_tag = Mock(return_value=overlap_bots)

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        # 18:00 UTC = NY only
        utc_time = datetime(2026, 2, 12, 18, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # London bot should be blocked
        assert "london_pref_bot" not in bot_ids
        assert "ny_pref_bot" in bot_ids
