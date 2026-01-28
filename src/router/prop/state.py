"""
Prop State Management
Responsible for persisting funded account metrics (Daily Loss, High Water Mark).
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

@dataclass
class PropAccountMetrics:
    account_id: str
    daily_start_balance: float
    high_water_mark: float
    current_equity: float
    trading_days: int
    target_met: bool = False
    last_updated: str = ""

class PropState:
    """
    Manages persistent state for Prop Accounts using Redis (or JSON fallback).
    """
    def __init__(self, account_id: str):
        self.account_id = account_id
        # In production, this connects to Redis. For now, in-memory/JSON.
        self_metrics = None 

    def update_snapshot(self, equity: float, balance: float):
        """
        Called at Midnight (00:00) to reset Daily Drawdown markers.
        """
        pass

    def check_daily_loss(self, current_equity: float) -> float:
        """
        Returns drawdown percentage from Daily Start Balance.
        """
        # Logic to be implemented with Persistance
        return 0.0

    def get_quadratic_throttle(self, current_balance: float, limit_pct: float = 0.05) -> float:
        """
        Calculates the 'Distance to Ruin' multiplier.
        """
        # Placeholder for the curve formula
        return 1.0
