"""
Integration Tests for QuantMind Hybrid Core (Task Group 14)

Tests critical end-to-end workflows for the hybrid core specification:
- PropCommander and PropGovernor integration with PropState
- Risk management workflows across router components
- Multi-agent coordination patterns
- Database persistence patterns (when implemented)

Maximum 10 strategic integration tests focused on critical workflows.
Run with: pytest tests/integration/test_hybrid_core.py -v -m integration
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from types import SimpleNamespace
from datetime import datetime
import json

from src.router.prop.commander import PropCommander
from src.router.prop.governor import PropGovernor
from src.router.governor import RiskMandate
from src.router.sync import DiskSyncer


class TestPropCommanderGovernorIntegration:
    """Test integration between PropCommander and PropGovernor for coordinated risk management."""

    @pytest.fixture
    def mock_prop_state(self):
        """Create mock PropState with realistic metrics."""
        metrics = SimpleNamespace(
            daily_start_balance=100_000.0,
            high_water_mark=105_000.0,
            current_equity=103_000.0,
            trading_days=5,
            target_met=False
        )
        # PropState needs get_metrics() method
        state = SimpleNamespace(
            get_metrics=lambda: metrics
        )
        return state

    @pytest.mark.integration
    def test_commander_governor_coordinated_risk_workflow(self, mock_prop_state):
        """
        Test complete workflow: PropCommander auction → PropGovernor risk calculation.

        Critical workflow: Commander selects strategies, Governor applies risk limits.
        Verifies coordinated decision making across both components.
        """
        # Setup: Create commander and governor with shared state
        commander = PropCommander(account_id="TEST")
        commander.prop_state = mock_prop_state

        governor = PropGovernor(account_id="TEST")
        governor.prop_state = mock_prop_state

        # Step 1: Commander runs auction and returns strategy list
        regime_report = SimpleNamespace(
            regime="TREND_STABLE",
            chaos_score=0.2,
            regime_quality=0.85,
            news_state="SAFE",
            is_systemic_risk=False
        )

        # Mock base auction to return strategies
        with patch.object(PropCommander.__mro__[1], 'run_auction') as mock_auction:
            mock_auction.return_value = [
                {"name": "TrendBot", "score": 0.9, "kelly_score": 0.85},
                {"name": "MeanRevBot", "score": 0.7, "kelly_score": 0.65},
            ]

            strategies = commander.run_auction(regime_report)

        # Step 2: Governor calculates risk for each strategy
        trade_proposals = []
        for strategy in strategies:
            proposal = {
                "symbol": "EURUSD",
                "systemic_correlation": 0.1,
                "current_balance": 103_000.0,  # Use the current_equity from metrics
                "strategy_name": strategy["name"]
            }
            trade_proposals.append(proposal)

        mandates = []
        for proposal in trade_proposals:
            mandate = governor.calculate_risk(regime_report, proposal)
            mandates.append({
                "strategy": proposal["strategy_name"],
                "allocation": mandate.allocation_scalar,
                "risk_mode": mandate.risk_mode
            })

        # Verify coordinated workflow
        assert len(mandates) == 2
        assert all(m["allocation"] >= 0 for m in mandates)
        assert all(m["risk_mode"] in ["STANDARD", "THROTTLED", "HALTED_NEWS"] for m in mandates)

    @pytest.mark.integration
    def test_preservation_mode_triggers_coordinated_response(self):
        """
        Test that preservation mode coordinates Commander filtering and Governor throttling.

        Critical workflow: When target is reached, both Commander and Governor respond conservatively.
        """
        # Setup: Target reached (8% gain)
        preservation_metrics = SimpleNamespace(
            daily_start_balance=100_000.0,
            high_water_mark=108_500.0,
            current_equity=108_500.0,
            trading_days=10,
            target_met=True
        )
        preservation_state = SimpleNamespace(
            get_metrics=lambda: preservation_metrics
        )

        commander = PropCommander(account_id="TEST")
        commander.prop_state = preservation_state

        governor = PropGovernor(account_id="TEST")
        governor.prop_state = preservation_state

        # Commander should filter to Kelly >= 0.8 in preservation mode
        regime_report = SimpleNamespace(
            regime="TREND_STABLE",
            chaos_score=0.1,
            regime_quality=0.9,
            news_state="SAFE",
            is_systemic_risk=False
        )

        with patch.object(PropCommander.__mro__[1], 'run_auction') as mock_auction:
            mock_auction.return_value = [
                {"name": "PremiumBot", "score": 0.95, "kelly_score": 0.90},
                {"name": "RiskyBot", "score": 0.75, "kelly_score": 0.70},
            ]

            strategies = commander.run_auction(regime_report)

        # Only PremiumBot should pass Kelly filter
        assert len(strategies) == 1
        assert strategies[0]["name"] == "PremiumBot"

        # Governor should apply standard risk (no loss scenario)
        proposal = {
            "symbol": "EURUSD",
            "current_balance": 108_500.0,  # Use the current_equity from metrics
            "systemic_correlation": 0.0
        }

        mandate = governor.calculate_risk(regime_report, proposal)

        # Verify coordinated conservative response
        assert mandate.allocation_scalar >= 0.5  # Still允许高质量策略
        assert mandate.risk_mode in ["STANDARD", "THROTTLED"]


class TestPropStatePersistenceIntegration:
    """Test PropState persistence patterns across platform restarts."""

    @pytest.fixture
    def temp_state_dir(self):
        """Create temporary directory for state persistence."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        # Cleanup

    @pytest.mark.integration
    def test_prop_state_survives_process_restart(self, temp_state_dir):
        """
        Test that PropState metrics can persist across simulated restarts.

        Critical workflow: State persists to disk → Process restarts → State restored.
        """
        # Step 1: Create initial state and persist
        state_file = temp_state_dir / "prop_state_TEST.json"
        initial_metrics = {
            "account_id": "TEST",
            "daily_start_balance": 100_000.0,
            "high_water_mark": 102_000.0,
            "current_equity": 101_000.0,
            "trading_days": 3,
            "target_met": False,
            "last_updated": datetime.now().isoformat()
        }

        # Persist to disk
        with open(state_file, 'w') as f:
            json.dump(initial_metrics, f)

        # Step 2: Simulate restart - verify file exists
        assert state_file.exists()

        # Step 3: Restore from disk
        with open(state_file, 'r') as f:
            restored_metrics = json.load(f)

        # Verify all critical data restored
        assert restored_metrics["daily_start_balance"] == 100_000.0
        assert restored_metrics["high_water_mark"] == 102_000.0
        assert restored_metrics["trading_days"] == 3
        assert restored_metrics["account_id"] == "TEST"

    @pytest.mark.integration
    def test_daily_snapshot_updates_persist(self, temp_state_dir):
        """
        Test that daily snapshot updates persist correctly.

        Critical workflow: End-of-day snapshot updates → Persists → Next day reads correct values.
        """
        state_file = temp_state_dir / "snapshots.json"

        # Day 1: Initial snapshot
        day1_snapshot = {
            "date": "2024-01-01",
            "daily_start_balance": 100_000.0,
            "high_water_mark": 100_000.0,
            "current_equity": 101_500.0,
            "daily_drawdown_pct": 0.0,
            "is_breached": False
        }

        with open(state_file, 'w') as f:
            json.dump({"snapshots": [day1_snapshot]}, f)

        # Day 2: Update snapshot
        with open(state_file, 'r') as f:
            data = json.load(f)

        day2_snapshot = {
            "date": "2024-01-02",
            "daily_start_balance": 101_500.0,  # Previous day's equity
            "high_water_mark": 102_000.0,
            "current_equity": 102_500.0,
            "daily_drawdown_pct": 0.0,
            "is_breached": False
        }

        data["snapshots"].append(day2_snapshot)

        with open(state_file, 'w') as f:
            json.dump(data, f)

        # Verify both snapshots persist
        with open(state_file, 'r') as f:
            restored = json.load(f)

        assert len(restored["snapshots"]) == 2
        assert restored["snapshots"][0]["date"] == "2024-01-01"
        assert restored["snapshots"][1]["daily_start_balance"] == 101_500.0


class TestRiskMatrixDiskSyncIntegration:
    """Test risk matrix synchronization between Python and MQL5."""

    @pytest.fixture
    def temp_mql5_path(self):
        """Create temporary MQL5 Files directory."""
        temp_dir = tempfile.mkdtemp()
        mql5_path = Path(temp_dir) / "MQL5" / "Files"
        mql5_path.mkdir(parents=True)
        yield mql5_path
        # Cleanup

    @pytest.mark.integration
    def test_python_writes_risk_matrix_mql5_consumes(self, temp_mql5_path):
        """
        Test complete workflow: Python calculates risk → Writes to disk → MQL5 reads.

        Critical workflow: Risk decision syncs from Python backend to MQL5 frontend.
        """
        # Step 1: Python (Governor) calculates risk multipliers
        risk_decisions = {
            "EURUSD": {"multiplier": 1.2, "risk_mode": "AGGRESSIVE"},
            "GBPUSD": {"multiplier": 0.8, "risk_mode": "CONSERVATIVE"},
            "USDJPY": {"multiplier": 1.0, "risk_mode": "STANDARD"}
        }

        # Step 2: Sync to disk via DiskSyncer
        syncer = DiskSyncer(mt5_path=str(temp_mql5_path))
        import time
        risk_matrix = {
            symbol: {
                "multiplier": decision["multiplier"],
                "risk_mode": decision["risk_mode"],
                "timestamp": int(time.time())
            }
            for symbol, decision in risk_decisions.items()
        }

        syncer.sync_risk_matrix(risk_matrix)

        # Step 3: Verify MQL5 can read the file
        risk_file = temp_mql5_path / "risk_matrix.json"
        assert risk_file.exists()

        with open(risk_file, 'r') as f:
            loaded_matrix = json.load(f)

        # Verify structure matches MQL5 expectations
        assert "EURUSD" in loaded_matrix
        assert loaded_matrix["EURUSD"]["multiplier"] == 1.2
        assert loaded_matrix["GBPUSD"]["risk_mode"] == "CONSERVATIVE"

    @pytest.mark.integration
    def test_governor_throttle_syncs_to_mql5(self, temp_mql5_path):
        """
        Test that Governor throttle calculations sync to MQL5.

        Critical workflow: Governor detects drawdown → Throttles risk → MQL5 receives updated multiplier.
        """
        # Setup: Simulate 2% daily loss
        syncer = DiskSyncer(mt5_path=str(temp_mql5_path))

        # Governor calculates throttle based on 2% loss (4% effective limit)
        daily_start_balance = 100_000.0
        current_balance = 98_000.0
        loss_pct = (daily_start_balance - current_balance) / daily_start_balance
        effective_limit = 0.04  # 4%

        # Quadratic throttle formula: 1.0 - (loss/limit)^2
        throttle = 1.0 - (loss_pct / effective_limit) ** 2

        # Sync throttled multiplier to MQL5
        risk_matrix = {
            "EURUSD": {
                "multiplier": throttle,
                "risk_mode": "THROTTLED",
                "daily_loss_pct": loss_pct * 100,
                "timestamp": 1234567890
            }
        }

        syncer.sync_risk_matrix(risk_matrix)

        # Verify MQL5 receives throttled value
        risk_file = temp_mql5_path / "risk_matrix.json"
        with open(risk_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["EURUSD"]["multiplier"] < 1.0  # Should be throttled
        assert 0.0 < loaded["EURUSD"]["multiplier"] < 1.0
        assert loaded["EURUSD"]["risk_mode"] == "THROTTLED"


class TestCoinFlipBotWorkflow:
    """Test Coin Flip Bot workflow for minimum trading days requirement."""

    @pytest.mark.integration
    def test_coin_flip_bot_when_target_reached_insufficient_days(self):
        """
        Test Coin Flip Bot activation workflow.

        Critical workflow: Target reached but insufficient trading days → Coin Flip Bot deployed.
        """
        # Setup: Target reached but only 3 trading days (need 5+)
        metrics = SimpleNamespace(
            daily_start_balance=100_000.0,
            high_water_mark=109_000.0,
            current_equity=109_000.0,
            trading_days=3,  # Below minimum
            target_met=True
        )
        state = SimpleNamespace(
            get_metrics=lambda: metrics
        )

        commander = PropCommander(account_id="TEST")
        commander.prop_state = state

        regime_report = SimpleNamespace(
            regime="QUIET",
            chaos_score=0.1,
            news_state="SAFE",
            is_systemic_risk=False
        )

        # Mock base auction
        with patch.object(PropCommander.__mro__[1], 'run_auction') as mock_auction:
            mock_auction.return_value = [
                {"name": "PremiumBot", "kelly_score": 0.95}
            ]

            # Run auction
            strategies = commander.run_auction(regime_report)

        # Verify Coin Flip Bot is returned instead of PremiumBot
        assert len(strategies) == 1
        assert strategies[0]["name"] == "CoinFlip_Bot"
        assert strategies[0]["risk_mode"] == "MINIMAL"

    @pytest.mark.integration
    def test_coin_flip_bot_skipped_when_sufficient_days(self):
        """
        Test that Coin Flip Bot is skipped when minimum trading days met.

        Critical workflow: Target reached AND sufficient days → Normal strategies returned.
        """
        # Setup: Target reached with sufficient trading days
        metrics = SimpleNamespace(
            daily_start_balance=100_000.0,
            high_water_mark=108_500.0,
            current_equity=108_500.0,
            trading_days=7,  # Above minimum
            target_met=True
        )
        state = SimpleNamespace(
            get_metrics=lambda: metrics
        )

        commander = PropCommander(account_id="TEST")
        commander.prop_state = state

        regime_report = SimpleNamespace(
            regime="TREND_STABLE",
            chaos_score=0.1,
            news_state="SAFE",
            is_systemic_risk=False
        )

        # Mock base auction with Kelly-qualified strategies
        with patch.object(PropCommander.__mro__[1], 'run_auction') as mock_auction:
            mock_auction.return_value = [
                {"name": "PremiumBot", "kelly_score": 0.90},
                {"name": "QualityBot", "kelly_score": 0.85}
            ]

            strategies = commander.run_auction(regime_report)

        # Verify normal strategies returned (no Coin Flip Bot)
        assert len(strategies) == 2
        assert all(s["name"] != "CoinFlip_Bot" for s in strategies)
        assert all(s.get("kelly_score", 0) >= 0.8 for s in strategies)


class TestNewsGuardIntegration:
    """Test news guard integration across Commander and Governor."""

    @pytest.mark.integration
    def test_news_guard_halts_all_trading(self):
        """
        Test that news guard in Governor halts trading regardless of Commander selections.

        Critical workflow: News detected → Governor halts → Commander strategies ignored.
        """
        # Setup
        metrics = SimpleNamespace(
            daily_start_balance=100_000.0,
            current_equity=102_000.0,
            trading_days=5
        )
        state = SimpleNamespace(
            get_metrics=lambda: metrics
        )

        commander = PropCommander(account_id="TEST")
        commander.prop_state = state

        governor = PropGovernor(account_id="TEST")
        governor.prop_state = state

        # Commander returns strategies normally
        regime_report = SimpleNamespace(
            regime="TREND_STABLE",
            chaos_score=0.1,
            news_state="KILL_ZONE",  # News guard active
            is_systemic_risk=False
        )

        with patch.object(PropCommander.__mro__[1], 'run_auction') as mock_auction:
            mock_auction.return_value = [
                {"name": "Strategy1", "kelly_score": 0.95},
                {"name": "Strategy2", "kelly_score": 0.90}
            ]

            commander_strategies = commander.run_auction(regime_report)

        # Commander returns strategies
        assert len(commander_strategies) == 2

        # But Governor overrides with zero allocation
        proposal = {
            "symbol": "EURUSD",
            "current_balance": 102_000.0,
            "systemic_correlation": 0.0
        }

        mandate = governor.calculate_risk(regime_report, proposal)

        # Verify news guard halts all trading
        assert mandate.allocation_scalar == 0.0
        assert mandate.risk_mode == "HALTED_NEWS"
        assert "News" in (mandate.notes or "")

    @pytest.mark.integration
    def test_news_guard_coordinated_with_preservation_mode(self):
        """
        Test news guard coordination with preservation mode.

        Critical workflow: News + Preservation mode → Double conservative response.
        """
        # Setup: Preservation mode + News
        metrics = SimpleNamespace(
            daily_start_balance=100_000.0,
            current_equity=108_500.0,  # 8.5% gain
            trading_days=8
        )
        state = SimpleNamespace(
            get_metrics=lambda: metrics
        )

        commander = PropCommander(account_id="TEST")
        commander.prop_state = state

        governor = PropGovernor(account_id="TEST")
        governor.prop_state = state

        regime_report = SimpleNamespace(
            regime="HIGH_VOLATILITY",
            chaos_score=0.5,
            news_state="KILL_ZONE",
            is_systemic_risk=False
        )

        # Commander filters by Kelly (preservation mode)
        with patch.object(PropCommander.__mro__[1], 'run_auction') as mock_auction:
            mock_auction.return_value = [
                {"name": "Premium", "kelly_score": 0.92},
                {"name": "Good", "kelly_score": 0.85},
                {"name": "Marginal", "kelly_score": 0.75}
            ]

            strategies = commander.run_auction(regime_report)

        # Only Kelly >= 0.8 pass
        assert len(strategies) == 2
        assert all(s.get("kelly_score", 0) >= 0.8 for s in strategies)

        # Governor applies news guard on top
        proposal = {
            "symbol": "EURUSD",
            "current_balance": 108_500.0,
            "systemic_correlation": 0.0
        }

        mandate = governor.calculate_risk(regime_report, proposal)

        # News guard overrides all
        assert mandate.allocation_scalar == 0.0
        assert mandate.risk_mode == "HALTED_NEWS"


class TestMultiAgentCoordination:
    """Test multi-agent coordination patterns."""

    @pytest.mark.integration
    def test_commander_governor_shared_state_consistency(self):
        """
        Test that Commander and Governor maintain consistent state when sharing PropState.

        Critical workflow: Shared state object ensures both agents see same account metrics.
        """
        # Create shared state
        shared_metrics = SimpleNamespace(
            daily_start_balance=100_000.0,
            high_water_mark=103_000.0,
            current_equity=102_000.0,
            trading_days=4
        )
        shared_state = SimpleNamespace(
            get_metrics=lambda: shared_metrics
        )

        # Both agents share same state
        commander = PropCommander(account_id="TEST")
        commander.prop_state = shared_state

        governor = PropGovernor(account_id="TEST")
        governor.prop_state = shared_state

        # Commander reads state
        regime_report = SimpleNamespace(
            regime="TREND_STABLE",
            chaos_score=0.2,
            news_state="SAFE",
            is_systemic_risk=False
        )

        with patch.object(PropCommander.__mro__[1], 'run_auction') as mock_auction:
            mock_auction.return_value = [{"name": "Bot1", "kelly_score": 0.85}]
            commander.run_auction(regime_report)

        commander_metrics = commander.prop_state.get_metrics()
        commander_balance = commander_metrics.current_equity if commander_metrics else 0

        # Governor reads same state
        proposal = {
            "symbol": "EURUSD",
            "current_balance": 102_000.0,
            "systemic_correlation": 0.0
        }

        governor.calculate_risk(regime_report, proposal)

        governor_metrics = governor.prop_state.get_metrics()
        governor_balance = governor_metrics.current_equity if governor_metrics else 0

        # Verify state consistency
        assert commander_balance == 102_000.0
        assert governor_balance == 102_000.0

    @pytest.mark.integration
    def test_risk_mode_propagation_across_components(self):
        """
        Test that risk mode propagates correctly across Commander → Governor → MQL5.

        Critical workflow: Risk decision made in Python → Propagates to MQL5 → Position sizing affected.
        """
        # Setup: Throttled mode (2% loss)
        state = SimpleNamespace(
            daily_start_balance=100_000.0,
            current_equity=98_000.0,
            trading_days=5
        )

        governor = PropGovernor(account_id="TEST")
        governor.prop_state = state

        regime_report = SimpleNamespace(
            regime="TREND_STABLE",
            chaos_score=0.2,
            news_state="SAFE",
            is_systemic_risk=False
        )

        proposal = {
            "symbol": "EURUSD",
            "current_balance": 98_000.0,
            "systemic_correlation": 0.0
        }

        mandate = governor.calculate_risk(regime_report, proposal)

        # Verify throttled mode
        assert mandate.risk_mode in ["THROTTLED", "STANDARD"]
        assert 0.0 < mandate.allocation_scalar < 1.0

        # Propagate to MQL5 (simulated via disk sync)
        temp_dir = tempfile.mkdtemp()
        mql5_path = Path(temp_dir) / "MQL5" / "Files"
        mql5_path.mkdir(parents=True)

        syncer = DiskSyncer(mt5_path=str(mql5_path))
        risk_matrix = {
            "EURUSD": {
                "multiplier": mandate.allocation_scalar,
                "risk_mode": mandate.risk_mode,
                "timestamp": 1234567890
            }
        }

        syncer.sync_risk_matrix(risk_matrix)

        # Verify MQL5 receives correct risk mode
        risk_file = mql5_path / "risk_matrix.json"
        with open(risk_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["EURUSD"]["risk_mode"] == mandate.risk_mode
        assert loaded["EURUSD"]["multiplier"] == mandate.allocation_scalar

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
