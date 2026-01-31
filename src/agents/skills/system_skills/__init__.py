"""
System Skills Module

Contains skills for system operations including:
- log_trade_event: Log trade events to journal
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional


def log_trade_event(
    event_type: str,
    symbol: str,
    action: str,
    price: float,
    lots: float,
    strategy_name: str,
    pnl: Optional[float] = None
) -> Dict[str, Any]:
    """
    Log a trade event to the trading journal.

    Args:
        event_type: Type of event (entry, exit, stop_hit, take_profit, error)
        symbol: Trading symbol
        action: Trade direction (buy, sell)
        price: Execution price
        lots: Position size in lots
        strategy_name: Name of the strategy
        pnl: Optional profit/loss for exit events

    Returns:
        Dict with logged status, timestamp, and log_id
    """
    # Validate inputs
    valid_event_types = ["entry", "exit", "stop_hit", "take_profit", "error"]
    if event_type not in valid_event_types:
        raise ValueError(f"Invalid event_type: {event_type}. Must be one of {valid_event_types}")

    valid_actions = ["buy", "sell"]
    if action not in valid_actions:
        raise ValueError(f"Invalid action: {action}. Must be one of {valid_actions}")

    if price <= 0:
        raise ValueError("Price must be positive")

    if lots <= 0:
        raise ValueError("Lots must be positive")

    # Generate unique log ID
    log_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()

    # Create log entry
    log_entry = {
        "log_id": log_id,
        "timestamp": timestamp,
        "event_type": event_type,
        "symbol": symbol,
        "action": action,
        "price": price,
        "lots": lots,
        "strategy_name": strategy_name,
        "pnl": pnl
    }

    # In production, this would write to:
    # - SQLite database at /data/journal/trade_journal.db
    # - ChromaDB collection for semantic search
    # - Redis Pub/Sub channel for real-time monitoring

    # For now, we just return success with the log entry data
    # The actual persistence would be handled by the system

    return {
        "logged": True,
        "timestamp": timestamp,
        "log_id": log_id
    }


# Import skill creator meta-skill
from .skill_creator import SkillCreator, SkillGenerationConfig, create_skill

__all__ = ["log_trade_event", "SkillCreator", "SkillGenerationConfig", "create_skill"]
