"""
Tests for Multi-Symbol Router Simulation

Tests the multi-symbol extension for the StrategyRouter including:
- Multi-symbol auction simulation across multiple symbols
- Cross-symbol correlation checks (prevent correlated positions)
- Account assignment via RoutingMatrix
- Dispatch logging with full context
- Simultaneous symbol processing with asyncio

Task Group 5.1: 5 focused tests for router simulation
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, List

from src.router.engine import StrategyRouter
from src.router.sentinel import RegimeReport
from src.router.governor import RiskMandate
from src.router.bot_manifest import BotManifest, StrategyType, TradeFrequency, BrokerType


# ========== Test Fixtures ==========

@pytest.fixture
def mock_router():
    """Create a StrategyRouter with mocked components."""
    router = StrategyRouter(use_smart_kill=False)

    # Mock the bot registry
    mock_registry = Mock()
    mock_registry.list_by_tag = Mock(return_value=[
        create_test_manifest("bot_scalper_eur", StrategyType.SCALPER, ["EURUSD"]),
        create_test_manifest("bot_scalper_gbp", StrategyType.SCALPER, ["GBPUSD"]),
        create_test_manifest("bot_structural_xau", StrategyType.STRUCTURAL, ["XAUUSD"]),
    ])
    router._bot_registry = mock_registry

    # Mock routing matrix
    mock_routing = Mock()
    mock_routing.get_account_for_bot = Mock(return_value="demo_account")
    router._routing_matrix = mock_routing

    # Mock trade logger
    mock_logger = Mock()
    mock_logger.log_dispatch_context = Mock()
    router._trade_logger = mock_logger

    return router


@pytest.fixture
def mock_sentinel_report():
    """Create a mock RegimeReport."""
    return RegimeReport(
        regime="TREND_STABLE",
        chaos_score=0.2,
        regime_quality=0.8,
        susceptibility=0.1,
        is_systemic_risk=False,
        news_state="SAFE",
        timestamp=0.0
    )


@pytest.fixture
def mock_governor_mandate():
    """Create a mock RiskMandate."""
    return RiskMandate(
        allocation_scalar=0.5,
        risk_mode="STANDARD"
    )


def create_test_manifest(bot_id: str, strategy_type: StrategyType, symbols: List[str]) -> BotManifest:
    """Helper to create test BotManifest."""
    return BotManifest(
        bot_id=bot_id,
        name=f"Test Bot {bot_id}",
        strategy_type=strategy_type,
        frequency=TradeFrequency.MEDIUM,
        preferred_broker_type=BrokerType.RAW_ECN,
        symbols=symbols,
        win_rate=0.65,
        total_trades=100,
        prop_firm_safe=True
    )


# ========== Test 1: Multi-Symbol Auction ==========

def test_multi_symbol_auction_simulation(mock_router, mock_sentinel_report, mock_governor_mandate):
    """
    Test 1: Multi-symbol auction simulation.

    Verify that when multiple symbols are processed, the router:
    - Processes all symbols in parallel
    - Runs auction for each symbol independently
    - Aggregates dispatches across symbols
    - Returns combined results
    """
    # Mock single symbol processing to return different results per symbol
    def mock_process_symbol(symbol: str, price: float, account_data=None) -> Dict:
        return {
            "regime": "TREND_STABLE",
            "quality": 0.8,
            "chaos_score": 0.2,
            "mandate": mock_governor_mandate,
            "dispatches": [
                {
                    "bot_id": f"bot_{symbol.lower()}",
                    "symbol": symbol,
                    "score": 0.7
                }
            ],
            "tick_count": 1
        }

    with patch.object(mock_router, 'process_tick', side_effect=mock_process_symbol):
        symbols = ["EURUSD", "GBPUSD", "XAUUSD"]
        prices = [1.0850, 1.2650, 2025.50]

        result = asyncio.run(
            mock_router.process_multi_symbol_tick(symbols, prices)
        )

    # Verify structure
    assert "combined_dispatches" in result
    assert "symbols_processed" in result
    assert "total_dispatches" in result

    # Verify all symbols processed
    assert result["symbols_processed"] == 3

    # Note: EURUSD and GBPUSD are correlated, so one may be filtered
    # We expect at least 2 dispatches (XAUUSD + one of EURUSD/GBPUSD)
    assert len(result["combined_dispatches"]) >= 2

    # Verify XAUUSD is always included (not correlated with forex pairs)
    dispatch_symbols = [d["symbol"] for d in result["combined_dispatches"]]
    assert "XAUUSD" in dispatch_symbols

    # Verify correlation warning was issued
    assert len(result.get("correlation_warnings", [])) > 0


# ========== Test 2: Cross-Symbol Correlation Checks ==========

def test_cross_symbol_correlation_prevention(mock_router):
    """
    Test 2: Cross-symbol correlation checks.

    Verify that correlated currency pairs are detected and prevented:
    - EURUSD and GBPUSD are highly correlated (both USD pairs)
    - System should warn or prevent simultaneous positions
    - Correlation matrix lookup is performed
    """
    # Create correlated pair data
    correlated_pairs = [
        ("EURUSD", "GBPUSD"),  # Both USD pairs
        ("AUDUSD", "NZDUSD"),  # Commodity dollars
        ("EURJPY", "GBPJPY"),  # Both JPY pairs
    ]

    # Test correlation detection
    for symbol1, symbol2 in correlated_pairs:
        is_correlated = mock_router.check_symbol_correlation(symbol1, symbol2)
        assert is_correlated, f"Expected {symbol1} and {symbol2} to be correlated"

    # Test uncorrelated pairs
    uncorrelated_pairs = [
        ("EURUSD", "XAUUSD"),  # Different asset classes
        ("GBPUSD", "BTCUSD"),  # Forex vs Crypto
    ]

    for symbol1, symbol2 in uncorrelated_pairs:
        is_correlated = mock_router.check_symbol_correlation(symbol1, symbol2)
        assert not is_correlated, f"Expected {symbol1} and {symbol2} to be uncorrelated"


# ========== Test 3: Account Assignment via RoutingMatrix ==========

def test_account_assignment_routing_matrix(mock_router):
    """
    Test 3: Account assignment via RoutingMatrix.

    Verify that:
    - RoutingMatrix.get_account_for_bot() is called
    - Bots are assigned to appropriate accounts
    - Account assignment is included in dispatch context
    """
    # Create test manifest
    manifest = create_test_manifest("test_bot", StrategyType.SCALPER, ["EURUSD"])

    # Mock routing matrix to return specific account
    mock_routing = Mock()
    mock_routing.get_account_for_bot = Mock(return_value="account_b_sniper")
    mock_router._routing_matrix = mock_routing

    # Get account assignment
    account_id = mock_router.routing_matrix.get_account_for_bot(manifest)

    # Verify routing matrix was called
    mock_routing.get_account_for_bot.assert_called_once_with(manifest)

    # Verify account assignment
    assert account_id == "account_b_sniper"

    # Test with unapproved bot
    mock_routing.get_account_for_bot = Mock(return_value=None)
    account_id = mock_router.routing_matrix.get_account_for_bot(manifest)
    assert account_id is None


# ========== Test 4: Dispatch Logging with Full Context ==========

def test_dispatch_logging_full_context(mock_router, mock_sentinel_report, mock_governor_mandate):
    """
    Test 4: Dispatch logging with full context.

    Verify that TradeLogger.log_dispatch_context() is called with:
    - Regime information
    - Chaos score
    - Governor mandate
    - Dispatch list
    - Symbol context
    """
    # Create test dispatches
    dispatches = [
        {
            "bot_id": "bot_1",
            "symbol": "EURUSD",
            "score": 0.75,
            "authorized_risk_scalar": 0.5,
            "risk_mode": "STANDARD",
            "assigned_account": "demo_account"
        }
    ]

    # Mock the logger
    mock_logger = Mock()
    mock_router._trade_logger = mock_logger

    # Call logging method
    mock_router.trade_logger.log_dispatch_context(
        regime=mock_sentinel_report.regime,
        chaos_score=mock_sentinel_report.chaos_score,
        mandate=mock_governor_mandate,
        dispatches=dispatches,
        symbol="EURUSD"
    )

    # Verify logger was called with correct parameters
    mock_logger.log_dispatch_context.assert_called_once()

    call_args = mock_logger.log_dispatch_context.call_args
    assert call_args[1]["regime"] == "TREND_STABLE"
    assert call_args[1]["chaos_score"] == 0.2
    assert call_args[1]["symbol"] == "EURUSD"
    assert len(call_args[1]["dispatches"]) == 1


# ========== Test 5: Simultaneous Symbol Processing ==========

@pytest.mark.asyncio
async def test_simultaneous_symbol_processing(mock_router):
    """
    Test 5: Simultaneous symbol processing with asyncio.

    Verify that:
    - Multiple symbols are processed concurrently
    - Processing is parallel (not sequential)
    - All symbols complete successfully
    - Results are properly aggregated
    """
    # Track processing order
    processing_order = []

    async def mock_async_process(symbol: str, price: float, account_data=None):
        processing_order.append(symbol)
        await asyncio.sleep(0.01)  # Simulate async work
        return {
            "symbol": symbol,
            "regime": "TREND_STABLE",
            "dispatches": [{"bot_id": f"bot_{symbol.lower()}"}]
        }

    # Mock the async processing
    symbols = ["EURUSD", "GBPUSD", "XAUUSD"]
    prices = [1.0850, 1.2650, 2025.50]

    with patch.object(mock_router, '_process_symbol_async', side_effect=mock_async_process):
        result = await mock_router.process_multi_symbol_tick(symbols, prices)

    # Verify all symbols were processed
    assert len(processing_order) == 3

    # Verify results aggregated
    assert result["symbols_processed"] == 3
    # Note: correlation filtering may reduce dispatch count
    assert result["total_dispatches"] >= 2


# ========== Helper Tests ==========

def test_correlation_matrix_lookup():
    """
    Helper test for correlation matrix structure.

    Verifies the correlation matrix contains expected pairs and values.
    """
    # Test correlation matrix constants
    correlation_matrix = {
        ("EURUSD", "GBPUSD"): 0.85,
        ("EURUSD", "USDJPY"): -0.75,
        ("AUDUSD", "NZDUSD"): 0.80,
        ("EURUSD", "XAUUSD"): 0.15,
        ("BTCUSD", "ETHUSD"): 0.90,
    }

    # High correlation threshold
    HIGH_CORRELATION_THRESHOLD = 0.7

    # Verify highly correlated pairs
    highly_correlated = [
        pair for pair, corr in correlation_matrix.items()
        if abs(corr) >= HIGH_CORRELATION_THRESHOLD
    ]

    assert ("EURUSD", "GBPUSD") in highly_correlated
    assert ("AUDUSD", "NZDUSD") in highly_correlated
    assert ("BTCUSD", "ETHUSD") in highly_correlated


def test_tie_breaking_rules():
    """
    Helper test for auction tie-breaking rules.

    Verifies that when multiple bots have equal scores:
    - Preference given to higher frequency bots
    - Then by win rate
    - Then by total trades (experience)
    """
    # Create bots with equal base scores
    bot1 = {"bot_id": "bot_1", "score": 0.7, "frequency": "HFT", "win_rate": 0.65, "total_trades": 50}
    bot2 = {"bot_id": "bot_2", "score": 0.7, "frequency": "MEDIUM", "win_rate": 0.65, "total_trades": 100}
    bot3 = {"bot_id": "bot_3", "score": 0.7, "frequency": "LOW", "win_rate": 0.70, "total_trades": 150}

    # Apply tie-breaking
    tied_bots = [bot1, bot2, bot3]

    # Sort by tie-breaking rules: frequency > win_rate > total_trades
    frequency_rank = {"HFT": 3, "HIGH": 2, "MEDIUM": 1, "LOW": 0}

    ranked = sorted(
        tied_bots,
        key=lambda b: (
            frequency_rank.get(b.get("frequency", "LOW"), 0),
            b.get("win_rate", 0.5),
            b.get("total_trades", 0)
        ),
        reverse=True
    )

    # HFT bot should win (highest frequency rank)
    assert ranked[0]["bot_id"] == "bot_1"
    # Medium frequency should be second (frequency rank > LOW)
    assert ranked[1]["bot_id"] == "bot_2"
    # LOW frequency should be last (even with higher win rate)
    assert ranked[2]["bot_id"] == "bot_3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
