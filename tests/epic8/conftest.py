"""
Epic 8 Test Fixtures and Factories

Provides shared fixtures for Alpha Forge pipeline testing.
"""

import pytest
import uuid
from datetime import datetime, timezone


@pytest.fixture
def mock_pipeline_run():
    """Factory: Create mock pipeline run data."""
    def _make_run(
        strategy_id=None,
        strategy_name="Test Strategy",
        current_stage="BACKTEST",
        stage_status="running",
        approval_status="none"
    ):
        return {
            "strategy_id": strategy_id or f"strat-{uuid.uuid4().hex[:8]}",
            "strategy_name": strategy_name,
            "current_stage": current_stage,
            "stage_status": stage_status,
            "stages": [
                {"stage": "VIDEO_INGEST", "status": "passed"},
                {"stage": "RESEARCH", "status": "passed"},
                {"stage": "TRD", "status": "passed"},
                {"stage": "DEVELOPMENT", "status": "passed"},
                {"stage": "COMPILE", "status": "passed"},
                {"stage": "BACKTEST", "status": stage_status if current_stage == "BACKTEST" else "waiting"},
                {"stage": "VALIDATION", "status": "waiting"},
                {"stage": "EA_LIFECYCLE", "status": "waiting"},
                {"stage": "APPROVAL", "status": "waiting"},
            ],
            "approval_status": approval_status,
            "started_at": "2026-03-21T10:00:00Z",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    return _make_run


@pytest.fixture
def mock_variant_metrics():
    """Factory: Create mock variant metrics."""
    def _make_metrics(pnl=1500.0, trade_count=150, sharpe=2.5):
        return {
            "pnl": pnl,
            "trade_count": trade_count,
            "drawdown": 12.5,
            "sharpe": sharpe,
            "win_rate": 58.5,
            "avg_profit": 25.0,
            "avg_loss": -15.0,
            "profit_factor": 1.85,
            "max_consecutive_wins": 8,
            "max_consecutive_losses": 3,
        }
    return _make_metrics


@pytest.fixture
def mock_hypothesis():
    """Factory: Create mock research hypothesis."""
    def _make_hypothesis(symbol="EURUSD", timeframe="H4", confidence=0.8):
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "hypothesis": "Test trend following strategy",
            "supporting_evidence": ["Evidence 1", "Evidence 2"],
            "confidence_score": confidence,
            "recommended_next_steps": ["Step 1", "Step 2"]
        }
    return _make_hypothesis


@pytest.fixture
def mock_approval_gate():
    """Factory: Create mock approval gate."""
    def _make_gate(gate_type="alpha_forge_backtest"):
        return {
            "workflow_id": str(uuid.uuid4()),
            "from_stage": "backtest",
            "to_stage": "validation",
            "gate_type": gate_type,
            "strategy_id": str(uuid.uuid4()),
            "metrics_snapshot": {
                "total_trades": 150,
                "win_rate": 0.58,
                "sharpe_ratio": 1.45,
                "max_drawdown": 0.12,
                "net_profit": 12500.00
            }
        }
    return _make_gate


# =============================================================================
# Epic 8.10-8.13 Integration Test Fixtures
# =============================================================================

@pytest.fixture
def sample_regime_report():
    """Sample regime report from Sentinel."""
    from src.router.sentinel import RegimeReport
    return RegimeReport(
        regime="TREND_STABLE",
        chaos_score=0.2,
        timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_trade_outcomes():
    """Sample trade outcomes for HMM lag buffer testing."""
    return [
        {
            "trade_id": "T001",
            "bot_id": "SCALP_L_001",
            "close_date": datetime.now(timezone.utc),
            "outcome": "WIN",
            "pnl": 150.0,
            "holding_time_minutes": 45,
            "regime_at_entry": "TREND_STABLE"
        },
        {
            "trade_id": "T002",
            "bot_id": "SCALP_L_001",
            "close_date": datetime.now(timezone.utc),
            "outcome": "LOSS",
            "pnl": -75.0,
            "holding_time_minutes": 30,
            "regime_at_entry": "TREND_STABLE"
        },
    ]


@pytest.fixture
def sample_dpr_scores():
    """Sample DPR scores for queue remix/rerank testing."""
    from src.router.dpr_scoring_engine import DprScore, DprComponents

    return [
        DprScore(
            bot_id="bot-t1-1",
            composite_score=85.0,
            components=DprComponents(0.75, 500, 0.85, 1.5),
            rank=1,
            tier="T1",
            session_specialist=False,
            session_concern=False,
            consecutive_negative_ev=0
        ),
        DprScore(
            bot_id="bot-t2-1",
            composite_score=65.0,
            components=DprComponents(0.60, 200, 0.65, 0.9),
            rank=2,
            tier="T2",
            session_specialist=False,
            session_concern=False,
            consecutive_negative_ev=0
        ),
        DprScore(
            bot_id="bot-t3-1",
            composite_score=35.0,
            components=DprComponents(0.35, -100, 0.40, 0.3),
            rank=3,
            tier="T3",
            session_specialist=False,
            session_concern=True,
            consecutive_negative_ev=5
        ),
    ]


@pytest.fixture
def mock_datetime_utc():
    """Mock datetime for deterministic UTC testing."""
    return datetime(2026, 3, 24, 12, 0, 0, tzinfo=timezone.utc)
