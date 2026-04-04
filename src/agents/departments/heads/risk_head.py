"""
Risk Department Head

Responsible for:
- Position sizing and exposure management
- Drawdown monitoring and limits
- Value at Risk (VaR) calculations
- Backtest evaluation across all modes
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config
from src.agents.departments.department_mail import MessageType, Priority
from src.api.backtest_endpoints import (
    BacktestMode,
    _completed_backtests,
    _running_backtests,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Backtest Evaluation Models
# =============================================================================

@dataclass
class BacktestEvaluationThresholds:
    """Thresholds for pass/fail evaluation."""
    min_sharpe: float = 1.0
    max_drawdown: float = 15.0
    min_win_rate: float = 50.0


@dataclass
class ModeEvaluationResult:
    """Result of evaluating a single backtest mode."""
    mode: str
    passed: bool
    sharpe: float
    max_drawdown: float
    win_rate: float
    net_pnl: float
    total_trades: int
    reason: str


@dataclass
class BacktestEvaluationResult:
    """Result of full backtest evaluation across all modes."""
    ea_name: str
    pass_verdict: bool
    modes_passed: int
    modes_total: int
    mode_results: List[ModeEvaluationResult]
    thresholds: BacktestEvaluationThresholds
    evaluation_time: datetime


# =============================================================================
# Risk Department Head Implementation
# =============================================================================

class RiskHead(DepartmentHead):
    """Risk Department Head for risk management."""

    # All 6 backtest modes to evaluate
    BACKTEST_MODES = [
        BacktestMode.VANILLA,
        BacktestMode.SPICED,
        BacktestMode.VANILLA_FULL,
        BacktestMode.SPICED_FULL,
        BacktestMode.MODE_B,
        BacktestMode.MODE_C,
    ]

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.RISK)
        super().__init__(config=config, mail_db_path=mail_db_path)
        self.thresholds = BacktestEvaluationThresholds()

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "calculate_position_size",
                "description": "Calculate optimal position size",
                "parameters": {
                    "symbol": "Trading symbol",
                    "account_balance": "Account balance",
                    "risk_percent": "Risk per trade (%)",
                },
            },
            {
                "name": "check_drawdown",
                "description": "Check drawdown limits",
                "parameters": {
                    "account_id": "Account identifier",
                },
            },
            {
                "name": "calculate_var",
                "description": "Calculate Value at Risk",
                "parameters": {
                    "portfolio": "Portfolio holdings",
                    "confidence": "Confidence level (e.g., 0.95)",
                    "timeframe": "Timeframe in days",
                },
            },
            {
                "name": "run_backtest_evaluation",
                "description": "Evaluate a strategy across all 6 backtest modes and return pass/fail verdict",
                "parameters": {
                    "ea_name": "Strategy/EA name to evaluate",
                },
            },
            {
                "name": "get_evaluation_thresholds",
                "description": "Get current evaluation thresholds",
                "parameters": {},
            },
            {
                "name": "set_evaluation_thresholds",
                "description": "Set evaluation thresholds for pass/fail",
                "parameters": {
                    "min_sharpe": "Minimum Sharpe ratio (default 1.0)",
                    "max_drawdown": "Maximum drawdown % (default 15.0)",
                    "min_win_rate": "Minimum win rate % (default 50.0)",
                },
            },
        ]

    # =========================================================================
    # Backtest Evaluation Methods
    # =========================================================================

    def run_backtest_evaluation(self, ea_name: str) -> Dict[str, Any]:
        """
        Evaluate a strategy across all 6 backtest modes.

        Args:
            ea_name: Strategy/EA name to evaluate

        Returns:
            Dictionary containing:
            - pass_verdict: bool - True if >= 4/6 modes pass
            - modes_passed: int - Number of modes that passed
            - modes_total: int - Total modes evaluated (6)
            - mode_results: List of mode evaluation results
            - thresholds: Thresholds used for evaluation
        """
        logger.info(f"Starting backtest evaluation for {ea_name}")

        mode_results: List[ModeEvaluationResult] = []
        modes_passed = 0

        # Evaluate each mode
        for mode in self.BACKTEST_MODES:
            result = self._evaluate_mode(ea_name, mode)
            mode_results.append(result)
            if result.passed:
                modes_passed += 1

        # Determine pass/fail verdict: >= 4/6 modes must pass
        pass_verdict = modes_passed >= 4

        evaluation_result = BacktestEvaluationResult(
            ea_name=ea_name,
            pass_verdict=pass_verdict,
            modes_passed=modes_passed,
            modes_total=len(self.BACKTEST_MODES),
            mode_results=mode_results,
            thresholds=self.thresholds,
            evaluation_time=datetime.now(timezone.utc),
        )

        logger.info(
            f"Evaluation complete for {ea_name}: {modes_passed}/{len(self.BACKTEST_MODES)} "
            f"modes passed, verdict: {'PASS' if pass_verdict else 'FAIL'}"
        )

        return self._serialize_evaluation(evaluation_result)

    def _evaluate_mode(self, ea_name: str, mode: BacktestMode) -> ModeEvaluationResult:
        """
        Evaluate a single backtest mode against thresholds.

        Args:
            ea_name: Strategy name
            mode: Backtest mode to evaluate

        Returns:
            ModeEvaluationResult with pass/fail status and metrics
        """
        # Find completed backtest for this ea_name and mode
        bt = self._find_backtest(ea_name, mode)

        if bt is None:
            # No backtest found - return failed result
            return ModeEvaluationResult(
                mode=mode.value,
                passed=False,
                sharpe=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                net_pnl=0.0,
                total_trades=0,
                reason=f"No backtest found for mode {mode.value}",
            )

        # Evaluate against thresholds
        sharpe_pass = bt.sharpe >= self.thresholds.min_sharpe
        drawdown_pass = bt.max_drawdown <= self.thresholds.max_drawdown
        win_rate_pass = bt.win_rate >= self.thresholds.min_win_rate

        passed = sharpe_pass and drawdown_pass and win_rate_pass

        reasons = []
        if not sharpe_pass:
            reasons.append(f"Sharpe {bt.sharpe:.2f} < {self.thresholds.min_sharpe}")
        if not drawdown_pass:
            reasons.append(f"Drawdown {bt.max_drawdown:.2f}% > {self.thresholds.max_drawdown}%")
        if not win_rate_pass:
            reasons.append(f"Win rate {bt.win_rate:.1f}% < {self.thresholds.min_win_rate}%")

        reason = "; ".join(reasons) if reasons else "All thresholds met"

        return ModeEvaluationResult(
            mode=mode.value,
            passed=passed,
            sharpe=bt.sharpe,
            max_drawdown=bt.max_drawdown,
            win_rate=bt.win_rate,
            net_pnl=bt.net_pnl,
            total_trades=bt.total_trades,
            reason=reason,
        )

    def _find_backtest(self, ea_name: str, mode: BacktestMode):
        """
        Find a completed backtest for the given ea_name and mode.

        Args:
            ea_name: Strategy name
            mode: Backtest mode

        Returns:
            BacktestDetail if found, None otherwise
        """
        # Search through completed backtests
        for bt in _completed_backtests.values():
            if bt.ea_name == ea_name and bt.mode == mode:
                return bt
        return None

    def _serialize_evaluation(self, result: BacktestEvaluationResult) -> Dict[str, Any]:
        """Serialize evaluation result to dictionary."""
        return {
            "ea_name": result.ea_name,
            "pass": result.pass_verdict,
            "modes_passed": result.modes_passed,
            "modes_total": result.modes_total,
            "mode_results": [
                {
                    "mode": mr.mode,
                    "passed": mr.passed,
                    "sharpe": mr.sharpe,
                    "max_drawdown": mr.max_drawdown,
                    "win_rate": mr.win_rate,
                    "net_pnl": mr.net_pnl,
                    "total_trades": mr.total_trades,
                    "reason": mr.reason,
                }
                for mr in result.mode_results
            ],
            "thresholds": {
                "min_sharpe": result.thresholds.min_sharpe,
                "max_drawdown": result.thresholds.max_drawdown,
                "min_win_rate": result.thresholds.min_win_rate,
            },
            "evaluation_time": result.evaluation_time.isoformat(),
        }

    def get_evaluation_thresholds(self) -> Dict[str, float]:
        """Get current evaluation thresholds."""
        return {
            "min_sharpe": self.thresholds.min_sharpe,
            "max_drawdown": self.thresholds.max_drawdown,
            "min_win_rate": self.thresholds.min_win_rate,
        }

    def set_evaluation_thresholds(
        self,
        min_sharpe: Optional[float] = None,
        max_drawdown: Optional[float] = None,
        min_win_rate: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Set evaluation thresholds for pass/fail.

        Args:
            min_sharpe: Minimum Sharpe ratio
            max_drawdown: Maximum drawdown percentage
            min_win_rate: Minimum win rate percentage

        Returns:
            Updated thresholds
        """
        if min_sharpe is not None:
            self.thresholds.min_sharpe = min_sharpe
        if max_drawdown is not None:
            self.thresholds.max_drawdown = max_drawdown
        if min_win_rate is not None:
            self.thresholds.min_win_rate = min_win_rate

        logger.info(f"Updated thresholds: {self.get_evaluation_thresholds()}")
        return self.get_evaluation_thresholds()

    # =========================================================================
    # Risk Management Methods
    # =========================================================================

    def calculate_position_size(
        self,
        symbol: str,
        account_balance: float,
        risk_percent: float,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            symbol: Trading symbol
            account_balance: Total account balance
            risk_percent: Risk per trade as percentage

        Returns:
            Position size calculation result
        """
        risk_amount = account_balance * (risk_percent / 100)

        # Simplified position sizing (in production, would use ATR or other volatility measure)
        # Using a default stop loss of 2% for calculation
        stop_loss_pct = 2.0
        position_size = risk_amount / stop_loss_pct

        return {
            "symbol": symbol,
            "account_balance": account_balance,
            "risk_percent": risk_percent,
            "risk_amount": risk_amount,
            "position_size": round(position_size, 2),
            "stop_loss_pct": stop_loss_pct,
            "methodology": "fixed_risk",
        }

    def check_drawdown(self, account_id: str) -> Dict[str, Any]:
        """
        Check drawdown limits for an account.

        Args:
            account_id: Account identifier

        Returns:
            Drawdown status
        """
        # Demo data — real data requires live broker connection
        return {
            "account_id": account_id,
            "current_drawdown": 0.0,
            "max_allowed_drawdown": 20.0,
            "status": "ok",
            "message": "Account within drawdown limits",
            "demo_mode": True,
            "demo_message": "Connect MT5 broker accounts to see live drawdown data",
        }

    def calculate_var(
        self,
        portfolio: List[Dict[str, float]],
        confidence: float,
        timeframe: int,
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk for a portfolio.

        Args:
            portfolio: List of holdings with weights
            confidence: Confidence level (e.g., 0.95)
            timeframe: Timeframe in days

        Returns:
            VaR calculation result
        """
        # Simplified VaR calculation (in production, would use historical data)
        total_value = sum(h.get("value", 0) for h in portfolio)

        # Using a simplified parametric approach
        var_multiplier = {
            0.90: 1.28,
            0.95: 1.65,
            0.99: 2.33,
        }.get(confidence, 1.65)

        # Assuming 20% daily volatility for simplified calculation
        daily_volatility = 0.20
        var = total_value * daily_volatility * var_multiplier * (timeframe ** 0.5)

        return {
            "portfolio_value": total_value,
            "confidence": confidence,
            "timeframe_days": timeframe,
            "var": round(var, 2),
            "var_percentage": round((var / total_value) * 100, 2) if total_value > 0 else 0,
            "methodology": "simplified_parametric",
        }

    # =========================================================================
    # Claude SDK Integration
    # =========================================================================

    def _format_tools_for_anthropic(self) -> list:
        """Convert the full active tool surface to Anthropic tool definitions."""
        return super()._format_tools_for_anthropic()

    async def process_task(self, task: str, context: dict = None) -> dict:
        """
        Process a risk management task via Claude SDK.

        Args:
            task: Task description string.
            context: Optional canvas context dict.

        Returns:
            Result dict with status, department, content, and tool_calls.
        """
        memory_nodes = None
        try:
            if hasattr(self, "_read_relevant_memory"):
                memory_nodes = await self._read_relevant_memory(task)
        except Exception:
            pass

        full_system = self._build_system_prompt(
            canvas_context=context,
            memory_nodes=memory_nodes,
        )
        tools = self._format_tools_for_anthropic() if hasattr(self, "_format_tools_for_anthropic") else []

        try:
            if hasattr(self, "_invoke_claude"):
                result = await self._invoke_claude(task=task, canvas_context=context, tools=tools if tools else None)
            else:
                import anthropic as _anthropic
                from src.agents.providers.router import get_router

                runtime_config = get_router().resolve_runtime_config()
                if not runtime_config or not runtime_config.api_key:
                    raise RuntimeError(
                        "No LLM runtime configured. Configure a provider in Settings or set QMX_LLM_* environment variables."
                    )
                client = _anthropic.AsyncAnthropic(
                    api_key=runtime_config.api_key,
                    base_url=runtime_config.base_url,
                )
                kwargs = {
                    "model": runtime_config.model,
                    "max_tokens": 4096,
                    "system": full_system,
                    "messages": [{"role": "user", "content": task}],
                }
                if tools:
                    kwargs["tools"] = tools
                resp = await client.messages.create(**kwargs)
                content = "".join(b.text for b in resp.content if b.type == "text")
                result = {"content": content, "tool_calls": []}
        except Exception as e:
            logger.error(f"{self.department.value} Claude call failed: {e}")
            return {"status": "error", "error": str(e), "department": self.department.value}

        # Write opinion node
        try:
            if hasattr(self, "_write_opinion_node") and result.get("content"):
                await self._write_opinion_node(
                    content=f"Task: {task[:200]}\nResult: {result['content'][:500]}",
                    confidence=0.7,
                    tags=[self.department.value],
                )
        except Exception:
            pass

        # Risk-specific: dispatch alert if breach detected
        content_lower = result.get("content", "").lower()
        if "risk breach" in content_lower or "drawdown exceeded" in content_lower:
            try:
                self.mail_service.send(
                    from_dept=self.department.value,
                    to_dept=Department.TRADING.value,
                    type=MessageType.APPROVAL_REQUEST,
                    subject="Risk Alert",
                    body=result["content"][:500],
                    priority=Priority.URGENT,
                )
            except Exception as e:
                logger.warning(f"Risk alert dispatch failed: {e}")

        return {
            "status": "success",
            "department": self.department.value,
            "content": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
        }
