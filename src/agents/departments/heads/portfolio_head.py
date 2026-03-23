"""
Portfolio Management Department Head

Responsible for:
- Portfolio allocation and optimization
- Rebalancing decisions
- Performance tracking and attribution
- Portfolio report generation
"""
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config

logger = logging.getLogger(__name__)


# =============================================================================
# Portfolio Models
# =============================================================================

@dataclass
class StrategyAttribution:
    """P&L attribution for a single strategy."""
    strategy: str
    pnl: float
    percentage: float


@dataclass
class BrokerAttribution:
    """P&L attribution for a single broker."""
    broker: str
    pnl: float
    percentage: float


@dataclass
class AccountDrawdown:
    """Drawdown for a single account."""
    account_id: str
    drawdown_pct: float


@dataclass
class PortfolioReport:
    """Complete portfolio report."""
    total_equity: float
    strategy_attribution: List[StrategyAttribution]
    broker_attribution: List[BrokerAttribution]
    drawdown_by_account: List[AccountDrawdown]
    generated_at: datetime


# =============================================================================
# Portfolio Department Head Implementation
# =============================================================================

class PortfolioHead(DepartmentHead):
    """Portfolio Management Department Head."""

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        config = get_department_config(Department.PORTFOLIO)
        super().__init__(config=config, mail_db_path=mail_db_path)

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "optimize_allocation",
                "description": "Optimize portfolio allocation",
                "parameters": {
                    "assets": "List of assets",
                    "target_return": "Target return",
                    "max_risk": "Maximum risk tolerance",
                },
            },
            {
                "name": "rebalance_portfolio",
                "description": "Rebalance portfolio to target allocation",
                "parameters": {
                    "target_allocation": "Target allocation weights",
                    "threshold": "Rebalance threshold (%)",
                },
            },
            {
                "name": "track_performance",
                "description": "Track portfolio performance",
                "parameters": {
                    "period": "Performance period",
                    "benchmark": "Benchmark for comparison",
                },
            },
            {
                "name": "generate_portfolio_report",
                "description": "Generate portfolio report with P&L attribution and drawdowns",
                "parameters": {},
            },
            {
                "name": "get_total_equity",
                "description": "Get total portfolio equity across all accounts",
                "parameters": {},
            },
            {
                "name": "get_strategy_pnl",
                "description": "Get P&L attribution by strategy",
                "parameters": {
                    "period": "Period for attribution (default: all)",
                },
            },
            {
                "name": "get_broker_pnl",
                "description": "Get P&L attribution by broker",
                "parameters": {
                    "period": "Period for attribution (default: all)",
                },
            },
            {
                "name": "get_account_drawdowns",
                "description": "Get drawdown by account",
                "parameters": {},
            },
        ]

    # =========================================================================
    # Portfolio Report Methods
    # =========================================================================

    def generate_portfolio_report(self) -> Dict[str, Any]:
        """
        Generate complete portfolio report.

        Returns:
            Portfolio report with:
            - total_equity: Total equity across all accounts
            - pnl_attribution: P&L by strategy and by broker
            - drawdown_by_account: Drawdown per account
        """
        logger.info("Generating portfolio report")

        # Get all components
        total_equity = self.get_total_equity()
        strategy_pnl = self.get_strategy_pnl()
        broker_pnl = self.get_broker_pnl()
        account_drawdowns = self.get_account_drawdowns()

        report = PortfolioReport(
            total_equity=total_equity["total_equity"],
            strategy_attribution=[
                StrategyAttribution(
                    strategy=s["strategy"],
                    pnl=s["pnl"],
                    percentage=s["percentage"],
                )
                for s in strategy_pnl["by_strategy"]
            ],
            broker_attribution=[
                BrokerAttribution(
                    broker=b["broker"],
                    pnl=b["pnl"],
                    percentage=b["percentage"],
                )
                for b in broker_pnl["by_broker"]
            ],
            drawdown_by_account=[
                AccountDrawdown(
                    account_id=d["account_id"],
                    drawdown_pct=d["drawdown_pct"],
                )
                for d in account_drawdowns["by_account"]
            ],
            generated_at=datetime.now(timezone.utc),
        )

        return self._serialize_report(report)

    def get_total_equity(self) -> Dict[str, Any]:
        """
        Get total portfolio equity across all accounts.

        Returns:
            Total equity
        """
        # Demo data — real data requires MT5 broker connection
        # demo_mode: true means this data is simulated, not live
        accounts = [
            {"account_id": "acc_main", "balance": 50000.0},
            {"account_id": "acc_backup", "balance": 25000.0},
            {"account_id": "acc_paper", "balance": 10000.0},
        ]

        total_equity = sum(a["balance"] for a in accounts)

        return {
            "total_equity": total_equity,
            "accounts": accounts,
            "account_count": len(accounts),
            "demo_mode": True,
            "demo_message": "Connect MT5 broker accounts to see live equity data",
        }

    def get_strategy_pnl(self, period: str = "all") -> Dict[str, Any]:
        """
        Get P&L attribution by strategy.

        Args:
            period: Period for attribution (default: all)

        Returns:
            P&L by strategy
        """
        # Demo data — real data requires trade history from MT5/broker
        strategies = [
            {"strategy": "TrendFollower_v2.1", "pnl": 2450.0},
            {"strategy": "RangeTrader_v1.5", "pnl": 820.0},
            {"strategy": "BreakoutScaler_v3.0", "pnl": 1530.0},
            {"strategy": "ScalperPro_v1.0", "pnl": -320.0},
        ]

        total_pnl = sum(s["pnl"] for s in strategies)
        for s in strategies:
            s["percentage"] = round((s["pnl"] / total_pnl) * 100, 2) if total_pnl != 0 else 0

        return {
            "period": period,
            "total_pnl": total_pnl,
            "by_strategy": strategies,
            "demo_mode": True,
            "demo_message": "Connect MT5 broker accounts to see live P&L data",
        }

    def get_broker_pnl(self, period: str = "all") -> Dict[str, Any]:
        """
        Get P&L attribution by broker.

        Args:
            period: Period for attribution (default: all)

        Returns:
            P&L by broker
        """
        # Demo data — real data requires broker account connections
        brokers = [
            {"broker": "ICMarkets", "pnl": 3100.0},
            {"broker": "OANDA", "pnl": 1280.0},
            {"broker": "Pepperstone", "pnl": 100.0},
        ]

        total_pnl = sum(b["pnl"] for b in brokers)
        for b in brokers:
            b["percentage"] = round((b["pnl"] / total_pnl) * 100, 2) if total_pnl != 0 else 0

        return {
            "period": period,
            "total_pnl": total_pnl,
            "by_broker": brokers,
            "demo_mode": True,
            "demo_message": "Connect broker accounts to see live attribution",
        }

    def get_account_drawdowns(self) -> Dict[str, Any]:
        """
        Get drawdown by account.

        Returns:
            Drawdown by account
        """
        # Demo data - in production would calculate from equity curves
        accounts = [
            {"account_id": "acc_main", "drawdown_pct": 8.3},
            {"account_id": "acc_backup", "drawdown_pct": 5.1},
            {"account_id": "acc_paper", "drawdown_pct": 2.4},
        ]

        return {
            "by_account": accounts,
        }

    # =========================================================================
    # Story 9.2: Portfolio Metrics & Attribution API Methods
    # =========================================================================

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary with key metrics.

        AC1: Returns total equity, daily P&L, drawdown, active strategies, and per-account details.
        Also checks for drawdown alert threshold (10%).

        Returns:
            Portfolio summary dict with metrics and account details
        """
        logger.info("Generating portfolio summary")

        # Get account data with equity
        equity_data = self.get_total_equity()
        accounts_data = equity_data.get("accounts", [])

        # Get drawdown data
        drawdown_data = self.get_account_drawdowns()
        drawdowns = {d["account_id"]: d["drawdown_pct"] for d in drawdown_data.get("by_account", [])}

        # Demo daily P&L - in production would calculate from positions
        daily_pnl_values = {
            "acc_main": 800.0,
            "acc_backup": 350.50,
            "acc_paper": 100.0,
        }

        # Build account summaries
        accounts = []
        for acc in accounts_data:
            acc_id = acc.get("account_id", "unknown")
            accounts.append({
                "account_id": acc_id,
                "equity": acc.get("balance", 0.0),
                "daily_pnl": daily_pnl_values.get(acc_id, 0.0),
                "drawdown": drawdowns.get(acc_id, 0.0),
            })

        # Calculate totals
        total_equity = equity_data.get("total_equity", 0.0)
        total_daily_pnl = sum(a["daily_pnl"] for a in accounts)
        daily_pnl_pct = round((total_daily_pnl / total_equity) * 100, 2) if total_equity > 0 else 0.0

        # Calculate total drawdown (max of account drawdowns weighted by equity)
        if accounts:
            total_drawdown = round(
                sum(a["drawdown"] * a["equity"] for a in accounts) / total_equity
                if total_equity > 0 else 0.0,
                2
            )
        else:
            total_drawdown = 0.0

        # Active strategies (demo - would query running EAs)
        active_strategies = [
            "TrendFollower_v2.1",
            "RangeTrader_v1.5",
            "BreakoutScaler_v3.0",
        ]

        # AC4: Check drawdown threshold (10%)
        drawdown_alert = total_drawdown > 10.0
        if drawdown_alert:
            logger.warning(f"Portfolio drawdown alert: {total_drawdown}% exceeds 10% threshold")
            # In production, would trigger notification system here

        return {
            "total_equity": total_equity,
            "daily_pnl": total_daily_pnl,
            "daily_pnl_pct": daily_pnl_pct,
            "total_drawdown": total_drawdown,
            "active_strategies": active_strategies,
            "accounts": accounts,
            "drawdown_alert": drawdown_alert,
        }

    def get_attribution(self) -> Dict[str, Any]:
        """
        Get P&L attribution by strategy and broker.

        AC2: Returns P&L attribution per strategy (with equity contribution) and per broker.

        Returns:
            Attribution dict with by_strategy and by_broker
        """
        logger.info("Generating portfolio attribution")

        # Get strategy P&L
        strategy_pnl = self.get_strategy_pnl()
        total_pnl = strategy_pnl.get("total_pnl", 0)

        # Calculate equity contribution for each strategy
        # Demo equity contributions - in production would calculate from positions
        equity_contributions = {
            "TrendFollower_v2.1": 12000.0,
            "RangeTrader_v1.5": 4000.0,
            "BreakoutScaler_v3.0": 7500.0,
            "ScalperPro_v1.0": -1500.0,
        }

        by_strategy = []
        for s in strategy_pnl.get("by_strategy", []):
            strategy_name = s.get("strategy", "")
            by_strategy.append({
                "strategy": strategy_name,
                "pnl": s.get("pnl", 0.0),
                "percentage": s.get("percentage", 0.0),
                "equity_contribution": equity_contributions.get(strategy_name, 0.0),
            })

        # Get broker P&L
        broker_pnl = self.get_broker_pnl()
        by_broker = []
        for b in broker_pnl.get("by_broker", []):
            by_broker.append({
                "broker": b.get("broker", ""),
                "pnl": b.get("pnl", 0.0),
                "percentage": b.get("percentage", 0.0),
            })

        return {
            "by_strategy": by_strategy,
            "by_broker": by_broker,
        }

    def get_correlation_matrix(self, period_days: int = 30) -> Dict[str, Any]:
        """
        Get correlation matrix of strategy returns.

        AC3: Returns NxN correlation matrix of strategy returns.
        Uses Pearson correlation coefficient with configurable period.

        Args:
            period_days: Number of days for correlation calculation (default: 30)

        Returns:
            Correlation matrix dict
        """
        logger.info(f"Generating correlation matrix for {period_days} days")

        # Demo correlation data - in production would calculate from historical returns
        # Using 30-day returns for each strategy
        strategies = [
            "TrendFollower_v2.1",
            "RangeTrader_v1.5",
            "BreakoutScaler_v3.0",
            "ScalperPro_v1.0",
        ]

        # Demo correlation matrix (Pearson correlation coefficients)
        correlation_data = {
            ("TrendFollower_v2.1", "RangeTrader_v1.5"): 0.45,
            ("TrendFollower_v2.1", "BreakoutScaler_v3.0"): 0.78,
            ("TrendFollower_v2.1", "ScalperPro_v1.0"): -0.12,
            ("RangeTrader_v1.5", "BreakoutScaler_v3.0"): 0.32,
            ("RangeTrader_v1.5", "ScalperPro_v1.0"): 0.05,
            ("BreakoutScaler_v3.0", "ScalperPro_v1.0"): -0.08,
        }

        # Build symmetric matrix (all pairs)
        matrix = []
        high_correlation_threshold = 0.7

        for i, strat_a in enumerate(strategies):
            for j, strat_b in enumerate(strategies):
                if i < j:  # Only upper triangle to avoid duplicates
                    correlation = correlation_data.get((strat_a, strat_b), 0.0)
                    matrix.append({
                        "strategy_a": strat_a,
                        "strategy_b": strat_b,
                        "correlation": correlation,
                        "period_days": period_days,
                    })
                    # Log high correlations
                    if abs(correlation) >= high_correlation_threshold:
                        logger.warning(
                            f"High correlation detected: {strat_a} vs {strat_b} = {correlation}"
                        )

        return {
            "matrix": matrix,
            "high_correlation_threshold": high_correlation_threshold,
            "period_days": period_days,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _serialize_report(self, report: PortfolioReport) -> Dict[str, Any]:
        """Serialize portfolio report to dictionary."""
        return {
            "total_equity": report.total_equity,
            "pnl_attribution": {
                "by_strategy": [
                    {
                        "strategy": s.strategy,
                        "pnl": s.pnl,
                        "percentage": s.percentage,
                    }
                    for s in report.strategy_attribution
                ],
                "by_broker": [
                    {
                        "broker": b.broker,
                        "pnl": b.pnl,
                        "percentage": b.percentage,
                    }
                    for b in report.broker_attribution
                ],
            },
            "drawdown_by_account": [
                {
                    "account_id": d.account_id,
                    "drawdown_pct": d.drawdown_pct,
                }
                for d in report.drawdown_by_account
            ],
            "generated_at": report.generated_at.isoformat(),
        }

    # =========================================================================
    # Portfolio Optimization Methods
    # =========================================================================

    def optimize_allocation(
        self,
        assets: List[str],
        target_return: float,
        max_risk: float,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation.

        Args:
            assets: List of assets
            target_return: Target return
            max_risk: Maximum risk tolerance

        Returns:
            Optimized allocation
        """
        # Simplified mean-variance optimization placeholder
        n = len(assets) if assets else 3
        allocation = {asset: round(1.0 / n, 3) for asset in (assets or ["BTC", "ETH", "SPY"])}

        return {
            "assets": assets or ["BTC", "ETH", "SPY"],
            "allocation": allocation,
            "expected_return": target_return,
            "risk_score": max_risk,
            "methodology": "mean_variance",
        }

    def rebalance_portfolio(
        self,
        target_allocation: Dict[str, float],
        threshold: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Rebalance portfolio to target allocation.

        Args:
            target_allocation: Target allocation weights
            threshold: Rebalance threshold percentage

        Returns:
            Rebalance actions needed
        """
        # Demo - would calculate actual vs target
        actions = []
        for asset, target_pct in target_allocation.items():
            actions.append({
                "asset": asset,
                "target_pct": target_pct,
                "current_pct": target_pct + 2.5,
                "action": "sell" if target_pct + 2.5 > target_pct + threshold else "hold",
                "amount": 0.0,  # Would calculate actual amount
            })

        return {
            "rebalance_needed": any(a["action"] != "hold" for a in actions),
            "actions": actions,
            "threshold_pct": threshold,
        }

    def track_performance(
        self,
        period: str = "month",
        benchmark: str = "SPY",
    ) -> Dict[str, Any]:
        """
        Track portfolio performance.

        Args:
            period: Performance period
            benchmark: Benchmark for comparison

        Returns:
            Performance metrics
        """
        return {
            "period": period,
            "benchmark": benchmark,
            "portfolio_return": 5.2,
            "benchmark_return": 3.8,
            "alpha": 1.4,
            "beta": 0.85,
            "sharpe": 1.65,
            "max_drawdown": 8.3,
        }

    # =========================================================================
    # Claude SDK Integration
    # =========================================================================

    def _format_tools_for_anthropic(self) -> list:
        """Format registered tools into Anthropic tool definitions."""
        tools = []
        for tool_name, tool_obj in (self._tools or {}).items():
            try:
                tools.append({
                    "name": tool_name,
                    "description": getattr(tool_obj, "description", f"{tool_name} tool"),
                    "input_schema": getattr(
                        tool_obj,
                        "input_schema",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                })
            except Exception:
                pass
        return tools

    async def process_task(self, task: str, context: dict = None) -> dict:
        """
        Process a portfolio management task via Claude SDK.

        Args:
            task: Task description string.
            context: Optional canvas context dict.

        Returns:
            Result dict with status, department, content, and tool_calls.
        """
        dept_system = self.system_prompt

        # Read relevant memory
        memory_ctx = ""
        try:
            if hasattr(self, "_read_relevant_memory"):
                nodes = await self._read_relevant_memory(task)
                if nodes:
                    memory_ctx = "\n\n## Relevant Memory\n" + "\n".join(
                        f"- {n['content']}" for n in nodes
                    )
        except Exception:
            pass

        full_system = dept_system + memory_ctx
        tools = self._format_tools_for_anthropic() if hasattr(self, "_format_tools_for_anthropic") else []

        try:
            if hasattr(self, "_invoke_claude"):
                result = await self._invoke_claude(task=task, canvas_context=context, tools=tools if tools else None)
            else:
                import os
                import anthropic as _anthropic
                client = _anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                kwargs = {
                    "model": os.getenv("ANTHROPIC_MODEL_SONNET", "claude-sonnet-4-6"),
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

        return {
            "status": "success",
            "department": self.department.value,
            "content": result.get("content", ""),
            "tool_calls": result.get("tool_calls", []),
        }


# =============================================================================
# P&L Calculator - Decimal Precision for Attribution
# =============================================================================

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any


@dataclass
class AttributionResult:
    """Result of P&L attribution calculation."""
    total_pnl: Decimal
    by_trade: List[Dict[str, Any]]
    rounding_detected: bool = False


class PLCalculator:
    """
    P&L Calculator with Decimal-based precision.

    Used for portfolio attribution calculations to avoid floating point
    rounding errors (e.g., 0.1 + 0.2 != 0.3 in float).
    """

    def calculate_attribution(self, data: Dict[str, Any]) -> AttributionResult:
        """
        Calculate P&L attribution from trades.

        Args:
            data: Dict with "trades" key containing list of trade dicts
                  Each trade has "pnl" as Decimal

        Returns:
            AttributionResult with total_pnl (Decimal)
        """
        trades = data.get("trades", [])
        total_pnl = Decimal("0")
        by_trade = []
        rounding_detected = False

        for trade in trades:
            pnl = trade.get("pnl", Decimal("0"))
            if isinstance(pnl, (int, float)):
                pnl = Decimal(str(pnl))
            elif not isinstance(pnl, Decimal):
                pnl = Decimal(str(pnl))

            # Detect potential float rounding issues
            if isinstance(trade.get("pnl"), float):
                rounding_detected = True

            total_pnl += pnl
            by_trade.append({"pnl": pnl})

        return AttributionResult(
            total_pnl=total_pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            by_trade=by_trade,
            rounding_detected=rounding_detected,
        )

    def calculate_multi_broker_attribution(
        self, broker_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate P&L attribution across multiple brokers.

        Args:
            broker_data: Dict mapping broker_id to {"pnl": Decimal, "trades": int}

        Returns:
            Dict mapping broker_id to attribution dict with Decimal pnl
        """
        attribution = {}

        for broker_id, data in broker_data.items():
            pnl = data.get("pnl", Decimal("0"))
            if not isinstance(pnl, Decimal):
                pnl = Decimal(str(pnl))

            trades = data.get("trades", 0)

            attribution[broker_id] = {
                "pnl": pnl.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                "trades": trades,
            }

        return attribution
