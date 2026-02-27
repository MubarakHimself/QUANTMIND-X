"""
Strategy Router Tools for routing and managing strategies.

READ-ONLY for all departments.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy lifecycle status."""
    DRAFT = "draft"
    READY = "ready"
    TESTING = "testing"
    PASSED = "passed"
    FAILED = "failed"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"


@dataclass
class StrategyInfo:
    """Information about a strategy."""
    id: str
    name: str
    description: str
    status: StrategyStatus
    department: str
    created_at: datetime
    updated_at: datetime
    performance_metrics: Dict[str, Any]


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_id: str
    total_trades: int
    win_rate: float
    net_profit: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    last_updated: datetime


class StrategyRouter:
    """
    Strategy routing and management tools.

    READ-ONLY for all departments - no actual routing execution.
    """

    def __init__(self):
        """Initialize strategy router."""
        self._strategies: Dict[str, StrategyInfo] = {}
        self._performance: Dict[str, StrategyPerformance] = {}

    def list_strategies(
        self,
        status: Optional[StrategyStatus] = None,
        department: Optional[str] = None,
    ) -> List[StrategyInfo]:
        """
        List all strategies with optional filters.

        Args:
            status: Filter by status
            department: Filter by department

        Returns:
            List of StrategyInfo objects
        """
        strategies = list(self._strategies.values())

        if status:
            strategies = [s for s in strategies if s.status == status]

        if department:
            strategies = [s for s in strategies if s.department == department]

        return strategies

    def get_strategy(self, strategy_id: str) -> Optional[StrategyInfo]:
        """
        Get strategy by ID.

        Args:
            strategy_id: Strategy identifier

        Returns:
            StrategyInfo or None if not found
        """
        return self._strategies.get(strategy_id)

    def get_strategy_status(self, strategy_id: str) -> Optional[StrategyStatus]:
        """
        Get current status of a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            StrategyStatus or None if not found
        """
        strategy = self.get_strategy(strategy_id)
        return strategy.status if strategy else None

    def get_strategy_performance(
        self,
        strategy_id: str,
    ) -> Optional[StrategyPerformance]:
        """
        Get performance metrics for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            StrategyPerformance or None if not found
        """
        return self._performance.get(strategy_id)

    def get_all_performance_metrics(
        self,
    ) -> Dict[str, StrategyPerformance]:
        """
        Get performance metrics for all strategies.

        Returns:
            Dictionary of strategy_id to StrategyPerformance
        """
        return self._performance.copy()

    def search_strategies(
        self,
        query: str,
        limit: int = 10,
    ) -> List[StrategyInfo]:
        """
        Search strategies by name or description.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching StrategyInfo objects
        """
        query_lower = query.lower()
        results = []

        for strategy in self._strategies.values():
            if (query_lower in strategy.name.lower() or
                query_lower in strategy.description.lower()):
                results.append(strategy)
                if len(results) >= limit:
                    break

        return results

    def get_strategies_by_status(
        self,
        status: StrategyStatus,
    ) -> List[StrategyInfo]:
        """
        Get all strategies with a specific status.

        Args:
            status: Strategy status to filter by

        Returns:
            List of StrategyInfo objects
        """
        return [
            s for s in self._strategies.values()
            if s.status == status
        ]

    def get_active_strategies(self) -> List[StrategyInfo]:
        """
        Get all currently active strategies.

        Returns:
            List of active StrategyInfo objects
        """
        return self.get_strategies_by_status(StrategyStatus.ACTIVE)

    def get_strategy_count_by_status(self) -> Dict[str, int]:
        """
        Get count of strategies by status.

        Returns:
            Dictionary of status to count
        """
        counts = {}
        for status in StrategyStatus:
            counts[status.value] = len(self.get_strategies_by_status(status))
        return counts

    def compare_strategies(
        self,
        strategy_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Compare multiple strategies side by side.

        Args:
            strategy_ids: List of strategy IDs to compare

        Returns:
            Comparison summary
        """
        comparison = {
            "strategies": [],
            "metrics": {
                "net_profit": [],
                "win_rate": [],
                "sharpe_ratio": [],
                "max_drawdown": [],
            },
        }

        for strategy_id in strategy_ids:
            perf = self.get_strategy_performance(strategy_id)
            info = self.get_strategy(strategy_id)

            if perf and info:
                comparison["strategies"].append({
                    "id": strategy_id,
                    "name": info.name,
                    "status": info.status.value,
                })
                comparison["metrics"]["net_profit"].append(perf.net_profit)
                comparison["metrics"]["win_rate"].append(perf.win_rate)
                comparison["metrics"]["sharpe_ratio"].append(perf.sharpe_ratio)
                comparison["metrics"]["max_drawdown"].append(perf.max_drawdown)

        return comparison

    def get_recent_activity(
        self,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get recent strategy activity.

        Args:
            limit: Maximum number of activities

        Returns:
            List of recent activities
        """
        # Simulated recent activity
        activities = [
            {
                "strategy_id": "STRAT001",
                "action": "status_change",
                "from_status": "testing",
                "to_status": "passed",
                "timestamp": datetime.now().isoformat(),
            },
            {
                "strategy_id": "STRAT002",
                "action": "backtest_completed",
                "result": "passed",
                "timestamp": datetime.now().isoformat(),
            },
        ]

        return activities[:limit]

    def get_department_strategy_stats(
        self,
        department: str,
    ) -> Dict[str, Any]:
        """
        Get strategy statistics for a department.

        Args:
            department: Department name

        Returns:
            Department strategy statistics
        """
        dept_strategies = [
            s for s in self._strategies.values()
            if s.department == department
        ]

        return {
            "department": department,
            "total_strategies": len(dept_strategies),
            "active_count": len([s for s in dept_strategies if s.status == StrategyStatus.ACTIVE]),
            "testing_count": len([s for s in dept_strategies if s.status == StrategyStatus.TESTING]),
            "passed_count": len([s for s in dept_strategies if s.status == StrategyStatus.PASSED]),
        }

    def get_performance_summary(
        self,
        strategy_id: str,
    ) -> Dict[str, Any]:
        """
        Get performance summary for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Performance summary
        """
        perf = self.get_strategy_performance(strategy_id)
        if not perf:
            return {"error": "Strategy not found"}

        return {
            "strategy_id": strategy_id,
            "total_trades": perf.total_trades,
            "win_rate": f"{perf.win_rate}%",
            "net_profit": f"${perf.net_profit:,.2f}",
            "profit_factor": f"{perf.profit_factor:.2f}",
            "max_drawdown": f"{perf.max_drawdown}%",
            "sharpe_ratio": f"{perf.sharpe_ratio:.2f}",
            "last_updated": perf.last_updated.isoformat(),
            "overall_rating": self._calculate_rating(perf),
        }

    def _calculate_rating(self, perf: StrategyPerformance) -> str:
        """Calculate overall rating from performance metrics."""
        score = 0

        if perf.net_profit > 0:
            score += 1
        if perf.win_rate >= 50:
            score += 1
        if perf.profit_factor >= 1.5:
            score += 1
        if perf.max_drawdown < 20:
            score += 1
        if perf.sharpe_ratio >= 1.0:
            score += 1

        ratings = {
            5: "Excellent",
            4: "Good",
            3: "Average",
            2: "Poor",
            1: "Very Poor",
            0: "Terrible",
        }

        return ratings.get(score, "Unknown")
