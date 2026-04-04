"""
Portfolio Management Department Head

Responsible for:
- Portfolio allocation and optimization
- Rebalancing decisions
- Performance tracking and attribution
- Portfolio report generation
"""
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from math import sqrt

from sqlalchemy import func

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.types import Department, get_department_config
from src.database.models import (
    BrokerAccount,
    HouseMoneyState,
    StrategyPerformance,
    TradeJournal,
    TradingMode,
    db_session_scope,
)

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

    def _load_account_snapshots(self) -> List[Dict[str, Any]]:
        """Load currently registered and/or connected broker accounts."""
        snapshots: Dict[str, Dict[str, Any]] = {}

        try:
            with db_session_scope() as db:
                registered_accounts = (
                    db.query(BrokerAccount)
                    .filter(BrokerAccount.is_active.is_(True))
                    .order_by(BrokerAccount.created_at.desc())
                    .all()
                )

            for account in registered_accounts:
                snapshots[account.account_number] = {
                    "account_id": account.account_number,
                    "broker_name": account.broker_name,
                    "server": account.mt5_server,
                    "account_type": account.account_type.value,
                    "currency": account.currency,
                    "balance": 0.0,
                    "equity": 0.0,
                    "connected": False,
                    "is_active": False,
                    "registered": True,
                }
        except Exception as exc:
            logger.warning(f"Failed to load registered broker accounts: {exc}")

        try:
            from src.api.broker_endpoints import broker_registry, account_switcher

            for broker in [*broker_registry.get_all(), *broker_registry.get_pending()]:
                snapshot = snapshots.setdefault(
                    broker.account_id,
                    {
                        "account_id": broker.account_id,
                        "broker_name": broker.broker_name,
                        "server": broker.server,
                        "account_type": broker.type,
                        "currency": broker.currency,
                        "balance": 0.0,
                        "equity": 0.0,
                        "connected": False,
                        "is_active": False,
                        "registered": False,
                    },
                )
                snapshot.update(
                    {
                        "broker_name": broker.broker_name or snapshot.get("broker_name"),
                        "server": broker.server or snapshot.get("server"),
                        "account_type": broker.type or snapshot.get("account_type"),
                        "currency": broker.currency or snapshot.get("currency"),
                        "balance": float(broker.balance or 0.0),
                        "equity": float(broker.equity or broker.balance or 0.0),
                        "connected": broker.status == "connected",
                        "is_active": broker.account_id == account_switcher.active_account_id,
                    }
                )
        except Exception as exc:
            logger.debug(f"Broker registry unavailable for portfolio snapshots: {exc}")

        return sorted(snapshots.values(), key=lambda item: item["account_id"])

    def _load_latest_house_money_state(self) -> Dict[str, HouseMoneyState]:
        """Load the latest house-money state per account."""
        states: Dict[str, HouseMoneyState] = {}
        try:
            with db_session_scope() as db:
                rows = (
                    db.query(HouseMoneyState)
                    .order_by(HouseMoneyState.updated_at.desc(), HouseMoneyState.created_at.desc())
                    .all()
                )
            for row in rows:
                states.setdefault(row.account_id, row)
        except Exception as exc:
            logger.warning(f"Failed to load house-money state: {exc}")
        return states

    def _period_start(self, period: str) -> Optional[datetime]:
        """Translate a friendly period name into a UTC lower bound."""
        period_map = {
            "today": 1,
            "day": 1,
            "week": 7,
            "month": 30,
            "quarter": 90,
            "year": 365,
            "all": None,
        }
        days = period_map.get((period or "all").lower())
        if days is None:
            return None
        return datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=days)

    def _calculate_drawdown_pct(self, state: HouseMoneyState) -> float:
        """Derive drawdown percentage from the latest house-money state."""
        high_water_mark = float(state.high_water_mark or 0.0)
        if high_water_mark <= 0:
            return 0.0

        current_equity = float(state.daily_start_balance or 0.0) + float(state.current_pnl or 0.0)
        drawdown = max(high_water_mark - current_equity, 0.0)
        return round((drawdown / high_water_mark) * 100, 2)

    def _get_active_strategy_names(self) -> List[str]:
        """Return live strategy names observed in persisted performance or journal data."""
        strategy_names: List[str] = []

        try:
            with db_session_scope() as db:
                live_names = (
                    db.query(StrategyPerformance.strategy_name)
                    .filter(StrategyPerformance.mode == TradingMode.LIVE)
                    .distinct()
                    .all()
                )
                strategy_names = [name for (name,) in live_names if name]

                if not strategy_names:
                    journal_names = (
                        db.query(TradeJournal.bot_id)
                        .filter(TradeJournal.bot_id.isnot(None))
                        .distinct()
                        .all()
                    )
                    strategy_names = [name for (name,) in journal_names if name]
        except Exception as exc:
            logger.warning(f"Failed to load active strategies: {exc}")

        return sorted(strategy_names)

    def _pearson_correlation(self, xs: List[float], ys: List[float]) -> float:
        """Compute Pearson correlation for two aligned sequences."""
        n = min(len(xs), len(ys))
        if n < 2:
            return 0.0

        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        x_var = sum((x - x_mean) ** 2 for x in xs)
        y_var = sum((y - y_mean) ** 2 for y in ys)
        denominator = sqrt(x_var * y_var)
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 4)

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
        accounts = self._load_account_snapshots()
        total_equity = round(sum(float(a.get("equity", 0.0) or 0.0) for a in accounts), 2)

        return {
            "total_equity": total_equity,
            "accounts": accounts,
            "account_count": len(accounts),
            "connected_account_count": sum(1 for a in accounts if a.get("connected")),
        }

    def get_strategy_pnl(self, period: str = "all") -> Dict[str, Any]:
        """
        Get P&L attribution by strategy.

        Args:
            period: Period for attribution (default: all)

        Returns:
            P&L by strategy
        """
        period_start = self._period_start(period)
        strategies: List[Dict[str, Any]] = []
        try:
            with db_session_scope() as db:
                query = (
                    db.query(
                        TradeJournal.bot_id.label("strategy"),
                        func.coalesce(func.sum(TradeJournal.pnl), 0.0).label("pnl"),
                    )
                    .filter(TradeJournal.pnl.isnot(None))
                    .filter(TradeJournal.mode == TradingMode.LIVE)
                )
                if period_start is not None:
                    query = query.filter(TradeJournal.timestamp >= period_start)
                rows = (
                    query.group_by(TradeJournal.bot_id)
                    .order_by(func.coalesce(func.sum(TradeJournal.pnl), 0.0).desc())
                    .all()
                )
            strategies = [
                {"strategy": row.strategy, "pnl": float(row.pnl or 0.0)}
                for row in rows
                if row.strategy
            ]
        except Exception as exc:
            logger.warning(f"Failed to load strategy P&L: {exc}")

        total_pnl = sum(s["pnl"] for s in strategies)
        for s in strategies:
            s["percentage"] = round((s["pnl"] / total_pnl) * 100, 2) if total_pnl != 0 else 0

        return {
            "period": period,
            "total_pnl": total_pnl,
            "by_strategy": strategies,
        }

    def get_broker_pnl(self, period: str = "all") -> Dict[str, Any]:
        """
        Get P&L attribution by broker.

        Args:
            period: Period for attribution (default: all)

        Returns:
            P&L by broker
        """
        period_start = self._period_start(period)
        brokers: List[Dict[str, Any]] = []
        try:
            with db_session_scope() as db:
                query = (
                    db.query(
                        TradeJournal.broker.label("broker"),
                        func.coalesce(func.sum(TradeJournal.pnl), 0.0).label("pnl"),
                    )
                    .filter(TradeJournal.pnl.isnot(None))
                    .filter(TradeJournal.mode == TradingMode.LIVE)
                    .filter(TradeJournal.broker.isnot(None))
                )
                if period_start is not None:
                    query = query.filter(TradeJournal.timestamp >= period_start)
                rows = (
                    query.group_by(TradeJournal.broker)
                    .order_by(func.coalesce(func.sum(TradeJournal.pnl), 0.0).desc())
                    .all()
                )
            brokers = [
                {"broker": row.broker, "pnl": float(row.pnl or 0.0)}
                for row in rows
                if row.broker
            ]
        except Exception as exc:
            logger.warning(f"Failed to load broker P&L: {exc}")

        total_pnl = sum(b["pnl"] for b in brokers)
        for b in brokers:
            b["percentage"] = round((b["pnl"] / total_pnl) * 100, 2) if total_pnl != 0 else 0

        return {
            "period": period,
            "total_pnl": total_pnl,
            "by_broker": brokers,
        }

    def get_account_drawdowns(self) -> Dict[str, Any]:
        """
        Get drawdown by account.

        Returns:
            Drawdown by account
        """
        states = self._load_latest_house_money_state()
        accounts = [
            {
                "account_id": account_id,
                "drawdown_pct": self._calculate_drawdown_pct(state),
            }
            for account_id, state in sorted(states.items())
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

        latest_states = self._load_latest_house_money_state()

        # Build account summaries
        accounts = []
        for acc in accounts_data:
            acc_id = acc.get("account_id", "unknown")
            state = latest_states.get(acc_id)
            daily_pnl = float(state.current_pnl) if state else 0.0
            drawdown = self._calculate_drawdown_pct(state) if state else drawdowns.get(acc_id, 0.0)
            accounts.append({
                "account_id": acc_id,
                "equity": acc.get("equity", acc.get("balance", 0.0)),
                "daily_pnl": daily_pnl,
                "drawdown": drawdown,
                "connected": acc.get("connected", False),
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

        active_strategies = self._get_active_strategy_names()

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

        by_strategy = []
        for s in strategy_pnl.get("by_strategy", []):
            strategy_name = s.get("strategy", "")
            by_strategy.append({
                "strategy": strategy_name,
                "pnl": s.get("pnl", 0.0),
                "percentage": s.get("percentage", 0.0),
                "equity_contribution": 0.0,
                "equity_contribution_available": False,
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

        period_start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=period_days)
        series_by_strategy: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        try:
            with db_session_scope() as db:
                rows = (
                    db.query(TradeJournal.bot_id, TradeJournal.timestamp, TradeJournal.pnl)
                    .filter(TradeJournal.bot_id.isnot(None))
                    .filter(TradeJournal.pnl.isnot(None))
                    .filter(TradeJournal.mode == TradingMode.LIVE)
                    .filter(TradeJournal.timestamp >= period_start)
                    .all()
                )
            for bot_id, timestamp, pnl in rows:
                if not timestamp:
                    continue
                day_key = timestamp.date().isoformat()
                series_by_strategy[bot_id][day_key] += float(pnl or 0.0)
        except Exception as exc:
            logger.warning(f"Failed to load strategy correlation data: {exc}")

        strategies = sorted(series_by_strategy)
        matrix = []
        high_correlation_threshold = 0.7

        for i, strat_a in enumerate(strategies):
            for j, strat_b in enumerate(strategies):
                if i < j:  # Only upper triangle to avoid duplicates
                    days = sorted(set(series_by_strategy[strat_a]) | set(series_by_strategy[strat_b]))
                    xs = [series_by_strategy[strat_a].get(day, 0.0) for day in days]
                    ys = [series_by_strategy[strat_b].get(day, 0.0) for day in days]
                    correlation = self._pearson_correlation(xs, ys)
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
        """Convert the full active tool surface to Anthropic tool definitions."""
        return super()._format_tools_for_anthropic()

    async def process_task(self, task: str, context: dict = None) -> dict:
        """
        Process a portfolio management task via Claude SDK.

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
