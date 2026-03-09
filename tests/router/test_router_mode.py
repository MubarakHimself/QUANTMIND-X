"""
Tests for Router Mode Selection feature.

Tests the Commander's ability to select bots using different modes:
- auction: sort by score (default)
- priority: sort by priority value
- round_robin: rotate through bots equally
"""

import pytest
from src.router.commander import Commander
from src.router.bot_manifest import BotManifest, StrategyType, TradeFrequency


def test_priority_mode_ignores_scores():
    """In priority mode, bots should be selected by priority value, not score."""
    commander = Commander()
    commander.set_router_mode("priority")

    bots = [
        BotManifest(bot_id="bot1", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=1, score=9.0),
        BotManifest(bot_id="bot2", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=2, score=8.0),
        BotManifest(bot_id="bot3", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=3, score=7.0),
    ]

    selected = commander.select_bots_for_auction(bots)

    # Should select by priority (highest first), not score
    assert selected[0].bot_id == "bot3"  # priority 3
    assert selected[1].bot_id == "bot2"  # priority 2
    assert selected[2].bot_id == "bot1"  # priority 1


def test_auction_mode_uses_scores():
    """In auction mode, bots should be selected by score (highest first)."""
    commander = Commander()
    commander.set_router_mode("auction")

    bots = [
        BotManifest(bot_id="bot1", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=1, score=9.0),
        BotManifest(bot_id="bot2", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=2, score=8.0),
        BotManifest(bot_id="bot3", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=3, score=7.0),
    ]

    selected = commander.select_bots_for_auction(bots)

    # Should select by score (highest first), not priority
    assert selected[0].bot_id == "bot1"  # score 9.0
    assert selected[1].bot_id == "bot2"  # score 8.0
    assert selected[2].bot_id == "bot3"  # score 7.0


def test_round_robin_mode_rotates():
    """In round_robin mode, bots should be selected in rotation."""
    commander = Commander()
    commander.set_router_mode("round_robin")

    bots = [
        BotManifest(bot_id="bot1", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=1, score=9.0),
        BotManifest(bot_id="bot2", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=2, score=8.0),
        BotManifest(bot_id="bot3", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=3, score=7.0),
    ]

    # First selection should be bot1 (index 0)
    selected1 = commander.select_bots_for_auction(bots)
    assert selected1[0].bot_id == "bot1"

    # Second selection should be bot2 (rotated)
    selected2 = commander.select_bots_for_auction(bots)
    assert selected2[0].bot_id == "bot2"

    # Third selection should be bot3 (rotated)
    selected3 = commander.select_bots_for_auction(bots)
    assert selected3[0].bot_id == "bot3"

    # Fourth selection should wrap back to bot1
    selected4 = commander.select_bots_for_auction(bots)
    assert selected4[0].bot_id == "bot1"


def test_invalid_router_mode_raises():
    """Setting an invalid router mode should raise ValueError."""
    commander = Commander()

    with pytest.raises(ValueError) as exc_info:
        commander.set_router_mode("invalid_mode")

    assert "Invalid router mode" in str(exc_info.value)


def test_default_router_mode_is_auction():
    """Default router mode should be auction."""
    commander = Commander()

    # Default should be auction
    assert commander.router_mode == "auction"

    bots = [
        BotManifest(bot_id="bot1", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=1, score=9.0),
        BotManifest(bot_id="bot2", strategy_type=StrategyType.SCALPER, frequency=TradeFrequency.HIGH, priority=2, score=8.0),
    ]

    selected = commander.select_bots_for_auction(bots)

    # Default should sort by score
    assert selected[0].bot_id == "bot1"  # score 9.0
    assert selected[1].bot_id == "bot2"  # score 8.0


def test_select_bots_for_auction_empty_list():
    """Should return empty list when no bots provided."""
    commander = Commander()
    commander.router_mode = "auction"

    bots = []
    selected = commander.select_bots_for_auction(bots)

    assert selected == []
