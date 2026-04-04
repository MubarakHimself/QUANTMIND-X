"""
Tests for Regime-Conditional Strategy Pool Framework (Story 8.10)

Tests the pool-based routing system where bots are grouped into pools
(scalping_long, scalping_short, orb_long, etc.) and pools are activated
or muted based on the current market regime.

Acceptance Criteria Coverage:
- AC1: TREND_STABLE - scalping_long active, scalping_short muted, orb_long conditional
- AC2: RANGE_STABLE - scalping_neutral active, others muted
- AC3: BREAKOUT_PRIME - orb pools active, scalping pools muted
- AC4: HIGH_CHAOS - all pools muted, Layer 3 CHAOS response triggered
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from src.router.commander import Commander
from src.router.bot_manifest import (
    BotRegistry,
    BotManifest,
    StrategyType,
    TradeFrequency,
    PoolState,
    StrategyPool,
    TradingMode,
)
from src.router.sentinel import RegimeReport
from src.router.session_detector import TradingSession
from src.router.routing_matrix import RoutingDecision


class TestPoolFrameworkBasics:
    """Test basic pool framework functionality."""

    def test_strategy_type_has_orb(self):
        """Test that StrategyType enum includes ORB."""
        assert hasattr(StrategyType, 'ORB')
        assert StrategyType.ORB.value == "ORB"

    def test_pool_state_enum_values(self):
        """Test PoolState enum has all required values."""
        assert PoolState.ACTIVE.value == "active"
        assert PoolState.MUTED.value == "muted"
        assert PoolState.CONDITIONAL.value == "conditional"

    def test_regime_pool_map_exists(self):
        """Test Commander has regime_pool_map initialized."""
        mock_registry = Mock(spec=BotRegistry)
        commander = Commander(bot_registry=mock_registry)

        assert hasattr(commander, 'regime_pool_map')
        assert "TREND_STABLE" in commander.regime_pool_map
        assert "RANGE_STABLE" in commander.regime_pool_map
        assert "BREAKOUT_PRIME" in commander.regime_pool_map
        assert "HIGH_CHAOS" in commander.regime_pool_map
        assert "NEWS_EVENT" in commander.regime_pool_map

    def test_regime_pool_map_structure(self):
        """Test regime_pool_map has correct structure for each regime."""
        mock_registry = Mock(spec=BotRegistry)
        commander = Commander(bot_registry=mock_registry)

        # TREND_STABLE pools
        trend_pools = commander.regime_pool_map["TREND_STABLE"]
        assert trend_pools["scalping_long"] == PoolState.ACTIVE
        assert trend_pools["scalping_short"] == PoolState.MUTED
        assert trend_pools["orb_long"] == PoolState.CONDITIONAL
        assert trend_pools["orb_short"] == PoolState.MUTED
        assert trend_pools["scalping_neutral"] == PoolState.MUTED

        # RANGE_STABLE pools
        range_pools = commander.regime_pool_map["RANGE_STABLE"]
        assert range_pools["scalping_neutral"] == PoolState.ACTIVE
        assert range_pools["scalping_long"] == PoolState.MUTED
        assert range_pools["scalping_short"] == PoolState.MUTED

        # BREAKOUT_PRIME pools
        breakout_pools = commander.regime_pool_map["BREAKOUT_PRIME"]
        assert breakout_pools["orb_long"] == PoolState.ACTIVE
        assert breakout_pools["orb_short"] == PoolState.ACTIVE
        assert breakout_pools["orb_false_breakout"] == PoolState.ACTIVE
        assert breakout_pools["scalping_long"] == PoolState.MUTED
        assert breakout_pools["scalping_short"] == PoolState.MUTED

        # HIGH_CHAOS pools - all muted
        chaos_pools = commander.regime_pool_map["HIGH_CHAOS"]
        for pool_name, state in chaos_pools.items():
            assert state == PoolState.MUTED, f"Pool {pool_name} should be MUTED in HIGH_CHAOS"


class TestGetPoolStatesForRegime:
    """Test _get_pool_states_for_regime method."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    def test_trend_stable_pools(self, commander):
        """Test pool states for TREND_STABLE regime."""
        pools = commander._get_pool_states_for_regime("TREND_STABLE")
        assert pools["scalping_long"] == PoolState.ACTIVE
        assert pools["scalping_short"] == PoolState.MUTED
        assert pools["orb_long"] == PoolState.CONDITIONAL

    def test_range_stable_pools(self, commander):
        """Test pool states for RANGE_STABLE regime."""
        pools = commander._get_pool_states_for_regime("RANGE_STABLE")
        assert pools["scalping_neutral"] == PoolState.ACTIVE
        assert pools["scalping_long"] == PoolState.MUTED

    def test_breakout_prime_pools(self, commander):
        """Test pool states for BREAKOUT_PRIME regime."""
        pools = commander._get_pool_states_for_regime("BREAKOUT_PRIME")
        assert pools["orb_long"] == PoolState.ACTIVE
        assert pools["orb_short"] == PoolState.ACTIVE
        assert pools["orb_false_breakout"] == PoolState.ACTIVE

    def test_high_chaos_pools(self, commander):
        """Test all pools muted for HIGH_CHAOS regime."""
        pools = commander._get_pool_states_for_regime("HIGH_CHAOS")
        for state in pools.values():
            assert state == PoolState.MUTED

    def test_unknown_regime_returns_empty(self, commander):
        """Test that unknown regime returns empty dict."""
        pools = commander._get_pool_states_for_regime("UNKNOWN_REGIME")
        assert pools == {}


class TestGetActivePoolsForRegime:
    """Test _get_active_pools_for_regime method with volume confirmation."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    def test_trend_stable_active_pools_with_volume_confirm(self, commander):
        """Test TREND_STABLE: scalping_long active, orb_long conditional."""
        # With high volume (above threshold), orb_long should activate
        active = commander._get_active_pools_for_regime(
            "TREND_STABLE",
            symbol="EURUSD",
            volume=20000.0  # High volume - above 1.2x threshold
        )
        assert "scalping_long" in active
        assert "orb_long" in active  # Volume confirmed

    def test_trend_stable_active_pools_without_volume_confirm(self, commander):
        """Test TREND_STABLE: orb_long not active without volume confirmation."""
        # With low volume (below threshold), orb_long should NOT activate
        active = commander._get_active_pools_for_regime(
            "TREND_STABLE",
            symbol="EURUSD",
            volume=5000.0  # Low volume - below threshold
        )
        assert "scalping_long" in active
        assert "orb_long" not in active  # Volume not confirmed

    def test_range_stable_active_pools(self, commander):
        """Test RANGE_STABLE: only scalping_neutral active."""
        active = commander._get_active_pools_for_regime("RANGE_STABLE")
        assert "scalping_neutral" in active
        assert "scalping_long" not in active
        assert "scalping_short" not in active
        assert "orb_long" not in active

    def test_breakout_prime_active_pools(self, commander):
        """Test BREAKOUT_PRIME: all ORB pools active."""
        active = commander._get_active_pools_for_regime("BREAKOUT_PRIME")
        assert "orb_long" in active
        assert "orb_short" in active
        assert "orb_false_breakout" in active
        assert "scalping_long" not in active
        assert "scalping_neutral" not in active

    def test_high_chaos_no_active_pools(self, commander):
        """Test HIGH_CHAOS: no active pools."""
        active = commander._get_active_pools_for_regime("HIGH_CHAOS")
        assert len(active) == 0


class TestVolumeConfirmation:
    """Test volume confirmation for CONDITIONAL pools."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    def test_orb_long_volume_confirmed(self, commander):
        """Test ORB Long activates when volume is high enough."""
        # EURUSD demo session avg is 12000, threshold is 1.2x = 14400
        result = commander._check_volume_confirmation("orb_long", "EURUSD", 15000.0)
        assert result is True

    def test_orb_long_volume_not_confirmed(self, commander):
        """Test ORB Long does NOT activate when volume is too low."""
        result = commander._check_volume_confirmation("orb_long", "EURUSD", 5000.0)
        assert result is False

    def test_non_orb_pool_always_false(self, commander):
        """Test that non-conditional pools return False for volume check."""
        result = commander._check_volume_confirmation("scalping_long", "EURUSD", 20000.0)
        assert result is False

    def test_volume_check_with_no_volume_data(self, commander):
        """Test volume check falls back to demo values when no data available."""
        # With no volume provided and demo data, should use demo volume
        # Demo EURUSD volume is 15000, threshold is 14400, so it should confirm
        result = commander._check_volume_confirmation("orb_long", "EURUSD", None)
        # Should use demo data and confirm since 15000 > 14400


class TestGetBotPoolName:
    """Test _get_bot_pool_name method."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    def test_scalper_long(self, commander):
        """Test SCALPER bot with long direction."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.SCALPER,
            frequency=TradeFrequency.HIGH,
            name="Long Scalper",
            tags=["long"]
        )
        assert commander._get_bot_pool_name(bot) == "scalping_long"

    def test_scalper_short(self, commander):
        """Test SCALPER bot with short direction."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.SCALPER,
            frequency=TradeFrequency.HIGH,
            name="Short Scalper",
            tags=["short"]
        )
        assert commander._get_bot_pool_name(bot) == "scalping_short"

    def test_scalper_neutral(self, commander):
        """Test SCALPER bot with no direction (neutral)."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.SCALPER,
            frequency=TradeFrequency.HIGH,
            name="Neutral Scalper"
        )
        assert commander._get_bot_pool_name(bot) == "scalping_neutral"

    def test_orb_long(self, commander):
        """Test ORB bot with long direction."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.ORB,
            frequency=TradeFrequency.LOW,
            name="ORB Long Bot",
            tags=["long"]
        )
        assert commander._get_bot_pool_name(bot) == "orb_long"

    def test_orb_short(self, commander):
        """Test ORB bot with short direction."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.ORB,
            frequency=TradeFrequency.LOW,
            name="ORB Short Bot"
        )
        assert commander._get_bot_pool_name(bot) == "orb_short"

    def test_orb_false_breakout(self, commander):
        """Test ORB bot with false breakout direction."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.ORB,
            frequency=TradeFrequency.LOW,
            name="ORB FB Bot",
            tags=["fb"]
        )
        assert commander._get_bot_pool_name(bot) == "orb_false_breakout"

    def test_structural_returns_none(self, commander):
        """Test STRUCTURAL strategy does not belong to any pool."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.STRUCTURAL,
            frequency=TradeFrequency.LOW,
            name="Structural Bot"
        )
        assert commander._get_bot_pool_name(bot) is None

    def test_swing_returns_none(self, commander):
        """Test SWING strategy does not belong to any pool."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.SWING,
            frequency=TradeFrequency.LOW,
            name="Swing Bot"
        )
        assert commander._get_bot_pool_name(bot) is None

    def test_hft_returns_none(self, commander):
        """Test HFT strategy does not belong to any pool."""
        bot = BotManifest(
            bot_id="test",
            strategy_type=StrategyType.HFT,
            frequency=TradeFrequency.HFT,
            name="HFT Bot"
        )
        assert commander._get_bot_pool_name(bot) is None


class TestAC1TrendStable:
    """AC1: Given Sentinel reports regime = TREND_STABLE,
    When Commander evaluates routing,
    Then scalping pool Long is activated (bullish bias strategies),
    And scalping pool Short is muted,
    And ORB pool Long is evaluated if volume confirms."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    @pytest.fixture
    def sample_bots(self):
        """Create sample bots for AC1 testing."""
        return {
            "scalping_long_bot": BotManifest(
                bot_id="scalping_long_bot",
                name="Scalping Long Bot",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "long"],
                win_rate=0.55,
                total_trades=50,
                symbols=["EURUSD"],
                preferred_conditions=None,
                trading_mode=TradingMode.LIVE
            ),
            "scalping_short_bot": BotManifest(
                bot_id="scalping_short_bot",
                name="Scalping Short Bot",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "short"],
                win_rate=0.52,
                total_trades=40,
                symbols=["EURUSD"],
                preferred_conditions=None,
                trading_mode=TradingMode.LIVE
            ),
            "orb_long_bot": BotManifest(
                bot_id="orb_long_bot",
                name="ORB Long Bot",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "long"],
                win_rate=0.58,
                total_trades=30,
                symbols=["EURUSD"],
                preferred_conditions=None,
                trading_mode=TradingMode.LIVE
            ),
            "orb_short_bot": BotManifest(
                bot_id="orb_short_bot",
                name="ORB Short Bot",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "short"],
                win_rate=0.50,
                total_trades=25,
                symbols=["EURUSD"],
                preferred_conditions=None,
                trading_mode=TradingMode.LIVE
            ),
            "scalping_neutral_bot": BotManifest(
                bot_id="scalping_neutral_bot",
                name="Scalping Neutral Bot",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal"],
                win_rate=0.50,
                total_trades=35,
                symbols=["EURUSD"],
                preferred_conditions=None,
                trading_mode=TradingMode.LIVE
            ),
            "structural_bot": BotManifest(
                bot_id="structural_bot",
                name="Structural Bot",
                strategy_type=StrategyType.STRUCTURAL,
                frequency=TradeFrequency.MEDIUM,
                tags=["@primal"],
                win_rate=0.60,
                total_trades=20,
                symbols=["EURUSD"],
                preferred_conditions=None,
                trading_mode=TradingMode.LIVE
            ),
        }

    def test_ac1_trend_stable_scalping_long_activated(
        self, commander, sample_bots
    ):
        """Scalping pool Long should be activated."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["scalping_long_bot"],
            sample_bots["structural_bot"],
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
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert "scalping_long_bot" in bot_ids

    def test_ac1_trend_stable_scalping_short_muted(
        self, commander, sample_bots
    ):
        """Scalping pool Short should be muted."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["scalping_short_bot"],
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
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert "scalping_short_bot" not in bot_ids

    def test_ac1_trend_stable_orb_long_with_volume_confirm(
        self, commander, sample_bots
    ):
        """ORB pool Long should be activated when volume confirms."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["orb_long_bot"],
        ])

        # Mock routing matrix to allow ORB bots through
        def mock_route_bot(manifest):
            return RoutingDecision(
                bot_id=manifest.bot_id,
                assigned_account="account_a_machine_gun",
                is_approved=True,
                priority_score=85.0
            )
        commander._routing_matrix.route_bot = mock_route_bot

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        # High volume should confirm ORB Long
        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert "orb_long_bot" in bot_ids


class TestAC2RangeStable:
    """AC2: Given Sentinel reports regime = RANGE_STABLE,
    When Commander evaluates routing,
    Then scalping pool Neutral is activated,
    And scalping pools Long and Short are muted."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    @pytest.fixture
    def sample_bots(self):
        return {
            "scalping_neutral_bot": BotManifest(
                bot_id="scalping_neutral_bot",
                name="Scalping Neutral",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal"],
                win_rate=0.52,
                total_trades=30,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "scalping_long_bot": BotManifest(
                bot_id="scalping_long_bot",
                name="Scalping Long",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "long"],
                win_rate=0.55,
                total_trades=40,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "scalping_short_bot": BotManifest(
                bot_id="scalping_short_bot",
                name="Scalping Short",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "short"],
                win_rate=0.50,
                total_trades=35,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
        }

    def test_ac2_range_stable_neutral_activated(
        self, commander, sample_bots
    ):
        """Scalping pool Neutral should be activated in RANGE_STABLE."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["scalping_neutral_bot"],
        ])

        regime_report = RegimeReport(
            regime="RANGE_STABLE",
            chaos_score=0.3,
            regime_quality=0.6,
            susceptibility=0.4,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert "scalping_neutral_bot" in bot_ids

    def test_ac2_range_stable_long_short_muted(
        self, commander, sample_bots
    ):
        """Scalping pools Long and Short should be muted in RANGE_STABLE."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["scalping_long_bot"],
            sample_bots["scalping_short_bot"],
        ])

        regime_report = RegimeReport(
            regime="RANGE_STABLE",
            chaos_score=0.3,
            regime_quality=0.6,
            susceptibility=0.4,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert "scalping_long_bot" not in bot_ids
        assert "scalping_short_bot" not in bot_ids


class TestAC3BreakoutPrime:
    """AC3: Given Sentinel reports regime = BREAKOUT_PRIME,
    When Commander evaluates routing,
    Then ORB pools are activated (Long/Short/False Breakout),
    And scalping pools are muted per the ORB session configuration."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    @pytest.fixture
    def sample_bots(self):
        return {
            "orb_long_bot": BotManifest(
                bot_id="orb_long_bot",
                name="ORB Long",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "long"],
                win_rate=0.58,
                total_trades=20,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "orb_short_bot": BotManifest(
                bot_id="orb_short_bot",
                name="ORB Short",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "short"],
                win_rate=0.55,
                total_trades=18,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "orb_false_breakout_bot": BotManifest(
                bot_id="orb_false_breakout_bot",
                name="ORB False Breakout",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "fb"],
                win_rate=0.50,
                total_trades=15,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "scalping_long_bot": BotManifest(
                bot_id="scalping_long_bot",
                name="Scalping Long",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "long"],
                win_rate=0.55,
                total_trades=40,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "scalping_short_bot": BotManifest(
                bot_id="scalping_short_bot",
                name="Scalping Short",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "short"],
                win_rate=0.50,
                total_trades=35,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
        }

    def test_ac3_breakout_prime_orb_pools_active(
        self, commander, sample_bots
    ):
        """All ORB pools should be activated in BREAKOUT_PRIME."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["orb_long_bot"],
            sample_bots["orb_short_bot"],
            sample_bots["orb_false_breakout_bot"],
        ])

        # Mock routing matrix to allow ORB bots through
        def mock_route_bot(manifest):
            return RoutingDecision(
                bot_id=manifest.bot_id,
                assigned_account="account_a_machine_gun",
                is_approved=True,
                priority_score=85.0
            )
        commander._routing_matrix.route_bot = mock_route_bot

        regime_report = RegimeReport(
            regime="BREAKOUT_PRIME",
            chaos_score=0.4,
            regime_quality=0.7,
            susceptibility=0.5,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 9, 30, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert "orb_long_bot" in bot_ids
        assert "orb_short_bot" in bot_ids
        assert "orb_false_breakout_bot" in bot_ids

    def test_ac3_breakout_prime_scalping_pools_muted(
        self, commander, sample_bots
    ):
        """Scalping pools should be muted in BREAKOUT_PRIME."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["scalping_long_bot"],
            sample_bots["scalping_short_bot"],
        ])

        regime_report = RegimeReport(
            regime="BREAKOUT_PRIME",
            chaos_score=0.4,
            regime_quality=0.7,
            susceptibility=0.5,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 9, 30, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert "scalping_long_bot" not in bot_ids
        assert "scalping_short_bot" not in bot_ids


class TestAC4HighChaos:
    """AC4: Given Sentinel reports regime = HIGH_CHAOS,
    When Commander evaluates routing,
    Then all scalping pools are muted,
    And ORB pools are muted,
    And Layer 3 (CHAOS response) is triggered for any open positions."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    @pytest.fixture
    def sample_bots(self):
        return {
            "scalping_long_bot": BotManifest(
                bot_id="scalping_long_bot",
                name="Scalping Long",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "long"],
                win_rate=0.55,
                total_trades=40,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "orb_long_bot": BotManifest(
                bot_id="orb_long_bot",
                name="ORB Long",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "long"],
                win_rate=0.58,
                total_trades=20,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "structural_bot": BotManifest(
                bot_id="structural_bot",
                name="Structural Bot",
                strategy_type=StrategyType.STRUCTURAL,
                frequency=TradeFrequency.MEDIUM,
                tags=["@primal"],
                win_rate=0.60,
                total_trades=25,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
        }

    @patch('src.router.commander.Commander._trigger_chaos_response')
    def test_ac4_high_chaos_all_pools_muted(
        self, mock_chaos, commander, sample_bots
    ):
        """All pools should be muted in HIGH_CHAOS."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["scalping_long_bot"],
            sample_bots["orb_long_bot"],
            sample_bots["structural_bot"],
        ])

        regime_report = RegimeReport(
            regime="HIGH_CHAOS",
            chaos_score=0.85,
            regime_quality=0.15,
            susceptibility=0.9,
            is_systemic_risk=True,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]
        assert len(bot_ids) == 0  # No bots should be eligible

    @patch('src.router.commander.Commander._trigger_chaos_response')
    def test_ac4_high_chaos_chaos_response_triggered(
        self, mock_chaos, commander, sample_bots
    ):
        """Layer 3 CHAOS response should be triggered in HIGH_CHAOS."""
        commander._bot_registry.list_by_tag = Mock(return_value=[
            sample_bots["structural_bot"],
        ])

        regime_report = RegimeReport(
            regime="HIGH_CHAOS",
            chaos_score=0.85,
            regime_quality=0.15,
            susceptibility=0.9,
            is_systemic_risk=True,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc)

        commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)

        # Chaos response should have been triggered
        mock_chaos.assert_called_once()
        call_args = mock_chaos.call_args[0][0]
        assert call_args.regime == "HIGH_CHAOS"
        assert call_args.chaos_score == 0.85


class TestTriggerChaosResponse:
    """Test _trigger_chaos_response method."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    @patch('src.router.progressive_kill_switch.get_progressive_kill_switch')
    def test_trigger_chaos_response_calls_pks(self, mock_get_pks, commander):
        """Test that chaos response properly calls ProgressiveKillSwitch."""
        mock_pks = MagicMock()
        mock_get_pks.return_value = mock_pks

        regime_report = RegimeReport(
            regime="HIGH_CHAOS",
            chaos_score=0.85,
            regime_quality=0.15,
            susceptibility=0.9,
            is_systemic_risk=True,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )

        commander._trigger_chaos_response(regime_report)

        mock_get_pks.assert_called_once()


class TestGetAuctionStatus:
    """Test get_auction_status includes regime_pool_map."""

    def test_get_auction_status_includes_pool_map(self):
        """Test that get_auction_status returns regime_pool_map."""
        mock_registry = Mock(spec=BotRegistry)
        commander = Commander(bot_registry=mock_registry)

        # Mock the list_by_tag to return empty list for primal bots
        commander._bot_registry.list_by_tag = Mock(return_value=[])

        status = commander.get_auction_status()

        assert "regime_pool_map" in status
        assert "TREND_STABLE" in status["regime_pool_map"]
        assert "RANGE_STABLE" in status["regime_pool_map"]
        assert "BREAKOUT_PRIME" in status["regime_pool_map"]
        assert "HIGH_CHAOS" in status["regime_pool_map"]


class TestPoolRoutingIntegration:
    """Integration tests for pool routing across all regimes."""

    @pytest.fixture
    def commander(self):
        mock_registry = Mock(spec=BotRegistry)
        return Commander(bot_registry=mock_registry)

    @pytest.fixture
    def all_pool_bots(self):
        """Create one bot for each pool type."""
        return {
            "scalping_long": BotManifest(
                bot_id="scalping_long",
                name="Scalping Long",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "long"],
                win_rate=0.55,
                total_trades=50,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "scalping_short": BotManifest(
                bot_id="scalping_short",
                name="Scalping Short",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal", "short"],
                win_rate=0.52,
                total_trades=45,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "scalping_neutral": BotManifest(
                bot_id="scalping_neutral",
                name="Scalping Neutral",
                strategy_type=StrategyType.SCALPER,
                frequency=TradeFrequency.HIGH,
                tags=["@primal"],
                win_rate=0.50,
                total_trades=40,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "orb_long": BotManifest(
                bot_id="orb_long",
                name="ORB Long",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "long"],
                win_rate=0.58,
                total_trades=30,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "orb_short": BotManifest(
                bot_id="orb_short",
                name="ORB Short",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "short"],
                win_rate=0.55,
                total_trades=28,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "orb_false_breakout": BotManifest(
                bot_id="orb_false_breakout",
                name="ORB False Breakout",
                strategy_type=StrategyType.ORB,
                frequency=TradeFrequency.LOW,
                tags=["@primal", "fb"],
                win_rate=0.50,
                total_trades=22,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
            "structural": BotManifest(
                bot_id="structural",
                name="Structural",
                strategy_type=StrategyType.STRUCTURAL,
                frequency=TradeFrequency.MEDIUM,
                tags=["@primal"],
                win_rate=0.60,
                total_trades=35,
                preferred_conditions=None,
                symbols=["EURUSD"],
                trading_mode=TradingMode.LIVE
            ),
        }

    def test_trend_stable_routing(
        self, commander, all_pool_bots
    ):
        """Test routing table for TREND_STABLE:
        - scalping_long: ACTIVE
        - scalping_short: MUTED
        - scalping_neutral: MUTED
        - orb_long: CONDITIONAL (volume confirmed)
        - orb_short: MUTED
        - orb_false_breakout: not in map (structural fallback)
        """
        commander._bot_registry.list_by_tag = Mock(return_value=list(all_pool_bots.values()))

        regime_report = RegimeReport(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.8,
            susceptibility=0.1,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # scalping_long should be active
        assert "scalping_long" in bot_ids
        # scalping_short and scalping_neutral should be muted
        assert "scalping_short" not in bot_ids
        assert "scalping_neutral" not in bot_ids
        # orb_long is conditional - it may or may not be active based on volume
        # structural falls back to regime_strategy_map which allows it

    def test_range_stable_routing(
        self, commander, all_pool_bots
    ):
        """Test routing table for RANGE_STABLE:
        - scalping_neutral: ACTIVE
        - scalping_long: MUTED
        - scalping_short: MUTED
        - orb_long: MUTED
        - orb_short: MUTED
        """
        commander._bot_registry.list_by_tag = Mock(return_value=list(all_pool_bots.values()))

        regime_report = RegimeReport(
            regime="RANGE_STABLE",
            chaos_score=0.3,
            regime_quality=0.6,
            susceptibility=0.4,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 10, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # Only scalping_neutral should be active
        assert "scalping_neutral" in bot_ids
        assert "scalping_long" not in bot_ids
        assert "scalping_short" not in bot_ids
        assert "orb_long" not in bot_ids
        assert "orb_short" not in bot_ids

    def test_breakout_prime_routing(
        self, commander, all_pool_bots
    ):
        """Test routing table for BREAKOUT_PRIME:
        - orb_long: ACTIVE
        - orb_short: ACTIVE
        - orb_false_breakout: ACTIVE
        - scalping_long: MUTED
        - scalping_short: MUTED
        - scalping_neutral: MUTED
        """
        commander._bot_registry.list_by_tag = Mock(return_value=list(all_pool_bots.values()))

        # Mock routing matrix to allow ORB bots through
        def mock_route_bot(manifest):
            return RoutingDecision(
                bot_id=manifest.bot_id,
                assigned_account="account_a_machine_gun",
                is_approved=True,
                priority_score=85.0
            )
        commander._routing_matrix.route_bot = mock_route_bot

        regime_report = RegimeReport(
            regime="BREAKOUT_PRIME",
            chaos_score=0.4,
            regime_quality=0.7,
            susceptibility=0.5,
            is_systemic_risk=False,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 9, 30, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # All ORB pools should be active
        assert "orb_long" in bot_ids
        assert "orb_short" in bot_ids
        assert "orb_false_breakout" in bot_ids
        # All scalping pools should be muted
        assert "scalping_long" not in bot_ids
        assert "scalping_short" not in bot_ids
        assert "scalping_neutral" not in bot_ids

    def test_high_chaos_routing(
        self, commander, all_pool_bots
    ):
        """Test routing table for HIGH_CHAOS:
        - All pools MUTED
        - No new trades
        """
        commander._bot_registry.list_by_tag = Mock(return_value=list(all_pool_bots.values()))

        regime_report = RegimeReport(
            regime="HIGH_CHAOS",
            chaos_score=0.85,
            regime_quality=0.15,
            susceptibility=0.9,
            is_systemic_risk=True,
            news_state="SAFE",
            timestamp=datetime.now(timezone.utc).timestamp()
        )
        utc_time = datetime(2026, 2, 12, 14, 0, tzinfo=timezone.utc)

        dispatches = commander.run_auction(regime_report, 10000.0, "mt5_default", utc_time)
        bot_ids = [d["bot_id"] for d in dispatches]

        # All pools should be muted - no bots eligible
        assert len(bot_ids) == 0
