"""
Paper Trading Validator

Validates paper trading agents against performance criteria for promotion to live trading.

Validation Criteria:
- Sharpe Ratio > 1.5
- Win Rate > 55%
- Validation Period >= 30 days
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timezone, timedelta
import math

# Lazy import to avoid module not found errors when mcp_mt5 is not installed
# from mcp_metatrader5_server.src.mcp_mt5.paper_trading.deployer import PaperTradingDeployer

logger = logging.getLogger(__name__)

# Lazy-loaded PaperTradingDeployer
_PaperTradingDeployer = None

def _get_paper_trading_deployer_class():
    """Lazy load PaperTradingDeployer class to avoid import errors."""
    global _PaperTradingDeployer
    if _PaperTradingDeployer is None:
        try:
            from mcp_mt5.paper_trading.deployer import PaperTradingDeployer
            _PaperTradingDeployer = PaperTradingDeployer
        except ImportError as e:
            logger.warning(f"PaperTradingDeployer not available: {e}")
            _PaperTradingDeployer = None
    return _PaperTradingDeployer

# Try to import database components
try:
    from src.database.db_manager import DBManager
    from src.database.models import StrategyPerformance
    DB_AVAILABLE = True
except ImportError:
    logger.warning("Database components not available for performance tracking")
    DB_AVAILABLE = False
    DBManager = None  # type: ignore
    StrategyPerformance = None  # type: ignore


class PaperTradingValidator:
    """
    Validates paper trading agents against performance criteria.
    
    Features:
    - Check validation status (Sharpe, Win Rate, Days)
    - Calculate performance metrics from trade history
    - Update validation status periodically
    - Store metrics for historical tracking
    """
    
    VALIDATION_PERIOD_DAYS = 30
    MIN_SHARPE_RATIO = 1.5
    MIN_WIN_RATE = 0.55

    def __init__(self, db_manager: Optional[Any] = None):
        self.db = db_manager if db_manager and DB_AVAILABLE else None
        
        # Lazy load deployer to avoid import errors
        deployer_class = _get_paper_trading_deployer_class()
        self.deployer = deployer_class() if deployer_class else None
        
        # In-memory cache for performance metrics
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._last_update: Dict[str, datetime] = {}

    def check_validation_status(self, paper_agent_id: str) -> Dict[str, Any]:
        """
        Check validation status for a paper trading agent.
        
        Args:
            paper_agent_id: Paper trading agent identifier
            
        Returns:
            Dict with validation status including:
            - status: "VALIDATED", "VALIDATING", or "PENDING"
            - days_validated: Number of days since deployment
            - meets_criteria: Boolean indicating if performance criteria met
            - metrics: Performance metrics dict
        """
        perf = self.get_performance_metrics(paper_agent_id)
        days_validated = self._get_days_validated(paper_agent_id)
        meets_criteria = self.meets_promotion_criteria(perf)
        
        # Determine status
        if days_validated >= self.VALIDATION_PERIOD_DAYS and meets_criteria:
            status = "VALIDATED"
        elif days_validated > 0:
            status = "VALIDATING"
        else:
            status = "PENDING"
        
        return {
            "paper_agent_id": paper_agent_id,
            "status": status,
            "days_validated": days_validated,
            "meets_criteria": meets_criteria,
            "metrics": perf
        }

    def get_performance_metrics(self, paper_agent_id: str) -> Dict[str, Any]:
        """
        Get performance metrics from PaperTradingDeployer.
        
        Args:
            paper_agent_id: Paper trading agent identifier
            
        Returns:
            Dict with performance metrics or empty dict if unavailable
        """
        # Check cache first (cache valid for 5 minutes)
        if paper_agent_id in self._metrics_cache:
            last_update = self._last_update.get(paper_agent_id)
            if last_update and (datetime.now(timezone.utc) - last_update).total_seconds() < 300:
                return self._metrics_cache[paper_agent_id]
        
        # Return empty if deployer not available
        if self.deployer is None:
            return {}
        
        status = self.deployer.get_agent(paper_agent_id)
        if status is None:
            return {}
        
        # Access metrics safely - status may have metrics as an attribute or part of the object
        metrics = getattr(status, 'metrics', None) or {}
        
        # If no metrics from deployer, try to calculate from trade history
        if not metrics:
            metrics = self.calculate_performance_metrics(paper_agent_id)
        
        # Cache the results
        self._metrics_cache[paper_agent_id] = metrics
        self._last_update[paper_agent_id] = datetime.now(timezone.utc)
        
        return metrics

    def calculate_performance_metrics(self, paper_agent_id: str) -> Dict[str, Any]:
        """
        Calculate performance metrics from trade history.
        
        This method queries the paper trading deployer for trade history
        and calculates comprehensive performance metrics. Handles AgentPerformance
        instances returned by PaperTradingDeployer.get_agent() and maps their fields.
        
        Args:
            paper_agent_id: Paper trading agent identifier
            
        Returns:
            Dict with calculated metrics:
            - total_trades: Number of trades
            - winning_trades: Number of profitable trades
            - losing_trades: Number of losing trades
            - win_rate: Win rate as decimal (0-1)
            - total_pnl: Total profit/loss
            - average_pnl: Average PnL per trade
            - sharpe: Sharpe ratio (if enough data)
            - max_drawdown: Maximum drawdown
            - profit_factor: Gross wins / gross losses
        """
        # Return empty if deployer not available
        if self.deployer is None:
            return {}
        
        # Get agent status for basic info
        agent_status = self.deployer.get_agent(paper_agent_id)
        if agent_status is None:
            return {}
        
        # Initialize default metrics
        metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "average_pnl": 0.0,
            "sharpe": None,
            "max_drawdown": 0.0,
            "profit_factor": 0.0
        }
        
        # Check if agent_status is an AgentPerformance instance
        # AgentPerformance has typed attributes - prefer these first
        if hasattr(agent_status, 'sharpe_ratio'):
            # This is likely an AgentPerformance instance
            metrics["sharpe"] = agent_status.sharpe_ratio
            
            # win_rate in AgentPerformance is 0-100, convert to 0-1
            if hasattr(agent_status, 'win_rate') and agent_status.win_rate is not None:
                metrics["win_rate"] = agent_status.win_rate / 100.0
            
            # Map other AgentPerformance fields
            if hasattr(agent_status, 'total_trades'):
                metrics["total_trades"] = agent_status.total_trades
            if hasattr(agent_status, 'winning_trades'):
                metrics["winning_trades"] = agent_status.winning_trades
            if hasattr(agent_status, 'losing_trades'):
                metrics["losing_trades"] = agent_status.losing_trades
            if hasattr(agent_status, 'total_pnl'):
                metrics["total_pnl"] = agent_status.total_pnl
            if hasattr(agent_status, 'average_pnl'):
                metrics["average_pnl"] = agent_status.average_pnl
            if hasattr(agent_status, 'max_drawdown'):
                metrics["max_drawdown"] = agent_status.max_drawdown
            if hasattr(agent_status, 'profit_factor'):
                metrics["profit_factor"] = agent_status.profit_factor
            
            # Calculate win_rate from winning_trades/total_trades if not already set
            if metrics["win_rate"] == 0.0 and metrics["total_trades"] > 0:
                metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
            
            return metrics
        
        # Fallback: Check for metrics attribute (nested metrics dict)
        nested_metrics = getattr(agent_status, 'metrics', None)
        if nested_metrics and isinstance(nested_metrics, dict):
            # Map from nested metrics dict
            metrics["sharpe"] = nested_metrics.get("sharpe_ratio") or nested_metrics.get("sharpe")
            metrics["total_trades"] = nested_metrics.get("total_trades", 0)
            metrics["winning_trades"] = nested_metrics.get("winning_trades", 0)
            metrics["losing_trades"] = nested_metrics.get("losing_trades", 0)
            metrics["total_pnl"] = nested_metrics.get("total_pnl", 0.0)
            metrics["average_pnl"] = nested_metrics.get("average_pnl", 0.0)
            metrics["max_drawdown"] = nested_metrics.get("max_drawdown", 0.0)
            metrics["profit_factor"] = nested_metrics.get("profit_factor", 0.0)
            
            # Handle win_rate - could be percentage or decimal
            win_rate_val = nested_metrics.get("win_rate", 0)
            if win_rate_val > 1:
                # Convert from percentage to decimal
                metrics["win_rate"] = win_rate_val / 100.0
            else:
                metrics["win_rate"] = win_rate_val
            
            # Calculate win_rate from winning_trades/total_trades if not available
            if metrics["win_rate"] == 0.0 and metrics["total_trades"] > 0:
                metrics["win_rate"] = metrics["winning_trades"] / metrics["total_trades"]
            
            return metrics
        
        # Final fallback: Try to get individual attributes from agent status
        if hasattr(agent_status, 'total_trades'):
            metrics["total_trades"] = agent_status.total_trades
        if hasattr(agent_status, 'win_rate'):
            # Could be percentage or decimal
            wr = agent_status.win_rate
            if wr > 1:
                metrics["win_rate"] = wr / 100.0
            else:
                metrics["win_rate"] = wr
        if hasattr(agent_status, 'pnl'):
            metrics["total_pnl"] = agent_status.pnl
        if hasattr(agent_status, 'winning_trades'):
            metrics["winning_trades"] = agent_status.winning_trades
        if hasattr(agent_status, 'losing_trades'):
            metrics["losing_trades"] = agent_status.losing_trades
        
        return metrics

    def meets_promotion_criteria(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if metrics meet promotion criteria.
        
        Args:
            metrics: Performance metrics dict
            
        Returns:
            True if all criteria are met
        """
        sharpe = metrics.get("sharpe", 0)
        win_rate = metrics.get("win_rate", 0)
        
        # Handle None values
        if sharpe is None:
            sharpe = 0
        if win_rate is None:
            win_rate = 0
            
        return (
            sharpe > self.MIN_SHARPE_RATIO and
            win_rate > self.MIN_WIN_RATE
        )

    def generate_validation_report(self, paper_agent_id: str) -> str:
        """
        Create detailed performance report.
        
        Args:
            paper_agent_id: Paper trading agent identifier
            
        Returns:
            Formatted report string
        """
        status = self.check_validation_status(paper_agent_id)
        metrics = status["metrics"]
        
        return f"""
Paper Trading Validation Report for {paper_agent_id}
==================================================
Status: {status["status"]}
Days Validated: {status["days_validated"]}/{self.VALIDATION_PERIOD_DAYS}
Meets Criteria: {status["meets_criteria"]}

Performance Metrics:
--------------------
Sharpe Ratio: {metrics.get("sharpe", "N/A")} (threshold: > {self.MIN_SHARPE_RATIO})
Win Rate: {metrics.get("win_rate", 0):.2%} (threshold: > {self.MIN_WIN_RATE:.0%})
Total Trades: {metrics.get("total_trades", 0)}
Total PnL: ${metrics.get("total_pnl", 0):.2f}
Max Drawdown: {metrics.get("max_drawdown", 0):.2%}
Profit Factor: {metrics.get("profit_factor", 0):.2f}

Promotion Eligibility: {'✓ ELIGIBLE' if status["status"] == "VALIDATED" else '✗ NOT ELIGIBLE'}
        """

    def update_validation_status(self, paper_agent_id: str) -> Dict[str, Any]:
        """
        Update validation status and trigger notifications if status changed.
        
        This method is called periodically to check and update validation status.
        It broadcasts WebSocket notifications when status changes.
        
        Args:
            paper_agent_id: Paper trading agent identifier
            
        Returns:
            Updated validation status dict
        """
        # Get current status
        validation_result = self.check_validation_status(paper_agent_id)
        
        # Store in database if available
        if self.db and DB_AVAILABLE:
            try:
                self._store_metrics(paper_agent_id, validation_result)
            except Exception as e:
                logger.error(f"Failed to store metrics for {paper_agent_id}: {e}")
        
        # Broadcast update via WebSocket
        try:
            from src.api.websocket_endpoints import broadcast_paper_trading_performance
            import asyncio
            
            # Prepare metrics for broadcast
            metrics = validation_result.get("metrics", {})
            broadcast_metrics = {
                "total_trades": metrics.get("total_trades", 0),
                "winning_trades": metrics.get("winning_trades", 0),
                "losing_trades": metrics.get("losing_trades", 0),
                "win_rate": metrics.get("win_rate", 0.0),
                "total_pnl": metrics.get("total_pnl", 0.0),
                "sharpe_ratio": metrics.get("sharpe", None),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "validation_status": validation_result.get("status", "pending").lower(),
                "days_validated": validation_result.get("days_validated", 0),
                "meets_criteria": validation_result.get("meets_criteria", False)
            }
            
            # Schedule broadcast (non-blocking)
            asyncio.create_task(
                broadcast_paper_trading_performance(paper_agent_id, broadcast_metrics)
            )
        except ImportError:
            logger.debug("WebSocket broadcast not available")
        except Exception as e:
            logger.error(f"Failed to broadcast performance update: {e}")
        
        return validation_result

    def _get_days_validated(self, paper_agent_id: str) -> int:
        """
        Calculate days since validation start.
        
        Args:
            paper_agent_id: Paper trading agent identifier
            
        Returns:
            Number of days since deployment
        """
        # Return 0 if deployer not available
        if self.deployer is None:
            return 0
        
        # Get deploy time from deployer status
        status = self.deployer.get_agent(paper_agent_id)
        if status is None or status.created_at is None:
            return 0
        deploy_time = status.created_at
        delta = datetime.now(timezone.utc) - deploy_time
        return delta.days

    def _store_metrics(self, paper_agent_id: str, validation_result: Dict[str, Any]) -> None:
        """
        Store metrics in database for historical tracking.
        
        Args:
            paper_agent_id: Paper trading agent identifier
            validation_result: Validation result dict to store
        """
        if not self.db or not DB_AVAILABLE:
            return
        
        # This would store metrics in the database
        # Implementation depends on database schema
        logger.debug(f"Storing metrics for {paper_agent_id}")

    def get_all_validated_agents(self) -> List[str]:
        """
        Get list of all agents that have passed validation.
        
        Returns:
            List of agent IDs that are validated
        """
        validated = []
        
        if self.deployer is None:
            return validated
        
        try:
            agents = self.deployer.list_agents()
            for agent in agents:
                status = self.check_validation_status(agent.agent_id)
                if status["status"] == "VALIDATED":
                    validated.append(agent.agent_id)
        except Exception as e:
            logger.error(f"Failed to get validated agents: {e}")
        
        return validated

    def get_agents_eligible_for_promotion(self) -> List[Dict[str, Any]]:
        """
        Get list of agents eligible for promotion to live trading.
        
        Returns:
            List of dicts with agent_id and performance summary
        """
        eligible = []
        
        if self.deployer is None:
            return eligible
        
        try:
            agents = self.deployer.list_agents()
            for agent in agents:
                status = self.check_validation_status(agent.agent_id)
                if status["status"] == "VALIDATED" and status["meets_criteria"]:
                    eligible.append({
                        "agent_id": agent.agent_id,
                        "strategy_name": agent.strategy_name,
                        "symbol": agent.symbol,
                        "days_validated": status["days_validated"],
                        "metrics": status["metrics"]
                    })
        except Exception as e:
            logger.error(f"Failed to get eligible agents: {e}")
        
        return eligible


# Background task for periodic validation checks
async def run_periodic_validation(interval_minutes: int = 5):
    """
    Run periodic validation checks for all paper trading agents.
    
    Args:
        interval_minutes: Interval between checks in minutes
    """
    import asyncio
    
    validator = PaperTradingValidator()
    
    while True:
        # Skip if deployer not available
        if validator.deployer is None:
            logger.debug("PaperTradingDeployer not available, skipping periodic validation")
            await asyncio.sleep(interval_minutes * 60)
            continue
        
        try:
            agents = validator.deployer.list_agents()
            logger.info(f"Running periodic validation for {len(agents)} agents")
            
            for agent in agents:
                try:
                    validator.update_validation_status(agent.agent_id)
                except Exception as e:
                    logger.error(f"Validation update failed for {agent.agent_id}: {e}")
            
        except Exception as e:
            logger.error(f"Periodic validation error: {e}")
        
        await asyncio.sleep(interval_minutes * 60)
