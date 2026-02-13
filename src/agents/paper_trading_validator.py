from typing import Dict, Any, Optional
import logging
from datetime import datetime, timezone, timedelta

# Assume imports
# from mcp_metatrader5_server.src.mcp_mt5.paper_trading.deployer import PaperTradingDeployer
from mcp_mt5.paper_trading.deployer import PaperTradingDeployer
from src.database.db_manager import DBManager
from src.database.models import StrategyPerformance

logger = logging.getLogger(__name__)

class PaperTradingValidator:
    VALIDATION_PERIOD_DAYS = 30

    def __init__(self, db_manager: Optional[DBManager] = None):
        self.db = db_manager or DBManager()
        self.deployer = PaperTradingDeployer()

    def check_validation_status(self, paper_agent_id: str) -> Dict[str, Any]:
        """Queries paper trading agent performance"""
        # TODO: Integrate with PaperTradingDeployer.get_agent_status(paper_agent_id)
        # For now, mock
        perf = self.get_performance_metrics(paper_agent_id)
        days_validated = self._get_days_validated(paper_agent_id)
        meets_criteria = self.meets_promotion_criteria(perf)
        return {
            "paper_agent_id": paper_agent_id,
            "status": "VALIDATED" if days_validated >= self.VALIDATION_PERIOD_DAYS and meets_criteria else "VALIDATING",
            "days_validated": days_validated,
            "meets_criteria": meets_criteria,
            "metrics": perf
        }

    def get_performance_metrics(self, paper_agent_id: str) -> Dict[str, Any]:
        """Get performance metrics from PaperTradingDeployer"""
        status = self.deployer.get_agent_status(paper_agent_id)
        return status.get('metrics', {})

    def meets_promotion_criteria(self, metrics: Dict[str, Any]) -> bool:
        """Validates against thresholds (Sharpe > 1.5, win rate > 55%)"""
        return (
            metrics.get("sharpe", 0) > 1.5 and
            metrics.get("win_rate", 0) > 0.55
        )

    def generate_validation_report(self, paper_agent_id: str) -> str:
        """Creates detailed performance report"""
        status = self.check_validation_status(paper_agent_id)
        metrics = status["metrics"]
        return f"""
Paper Trading Validation Report for {paper_agent_id}
Status: {status["status"]}
Days Validated: {status["days_validated"]}
Sharpe Ratio: {metrics.get("sharpe", 0):.2f}
Win Rate: {metrics.get("win_rate", 0):.2%}
Meets Criteria: {status["meets_criteria"]}
        """

    def _get_days_validated(self, paper_agent_id: str) -> int:
        """Calculate days since validation start"""
        # Get deploy time from deployer status
        status = self.deployer.get_agent_status(paper_agent_id)
        deploy_time_str = status.get('deploy_time')
        if not deploy_time_str:
            return 0
        deploy_time = datetime.fromisoformat(deploy_time_str)
        delta = datetime.now(timezone.utc) - deploy_time
        return delta.days