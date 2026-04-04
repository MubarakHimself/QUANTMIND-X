"""
Portfolio Subagent

Worker agent for the Portfolio Department.
Responsible for portfolio allocation, rebalancing, and performance tracking.

Model: Haiku (fast, low-cost for worker tasks)
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# Portfolio management system prompt for LLM
PORTFOLIO_SYSTEM_PROMPT = """You are an expert portfolio manager specializing in forex trading.

Your task is to analyze portfolio allocations, suggest rebalancing strategies,
and generate performance reports based on the provided data.

## Portfolio Analysis Guidelines:
1. Consider risk management principles (position sizing, diversification)
2. Analyze correlation between currency pairs
3. Factor in volatility and potential drawdown
4. Consider correlation with other open positions
5. Provide actionable recommendations with specific numbers

## Allocation Analysis:
- Assess current weight vs target weight
- Identify overweight/underweight positions
- Consider risk-adjusted returns
- Evaluate diversification benefits

## Rebalancing Strategy:
- Minimize trading costs
- Consider tax implications
- Maintain target risk profile
- Use systematic rebalancing thresholds (e.g., 5% deviation)

## Performance Metrics to Consider:
- Total return
- Risk-adjusted return (Sharpe ratio)
- Maximum drawdown
- Win rate
- Profit factor
- Risk of ruin

Output clear, concise analysis with specific recommendations.
"""


# Performance report prompt
PERFORMANCE_REPORT_PROMPT = """You are an expert portfolio analyst. Generate a comprehensive performance report.

Include:
1. Summary of portfolio performance
2. Top performers and losers
3. Risk metrics analysis
4. Attribution breakdown
5. Recommendations for improvement

Be specific with numbers and percentages."""


@dataclass
class PortfolioTask:
    """Portfolio task input data."""
    task_type: str  # allocation, rebalancing, performance, optimization
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    volume: float
    entry_price: float
    current_price: float
    side: str = "buy"

    @property
    def value(self) -> float:
        """Position value."""
        return self.volume * self.current_price * 100000

    @property
    def pnl(self) -> float:
        """Profit/Loss."""
        if self.side == "buy":
            return (self.current_price - self.entry_price) * self.volume * 100000
        return (self.entry_price - self.current_price) * self.volume * 100000


class PortfolioSubAgent:
    """
    Portfolio subagent for portfolio management.

    Capabilities:
    - Portfolio allocation management
    - Rebalancing operations
    - Performance tracking and attribution
    - Risk-adjusted returns calculation
    """

    def __init__(
        self,
        agent_id: str,
        task: Optional[PortfolioTask] = None,
        available_tools: Optional[List[str]] = None,
    ):
        """
        Initialize portfolio subagent.

        Args:
            agent_id: Unique identifier for this agent
            task: Task configuration
            available_tools: List of tool names available to this agent
        """
        self.agent_id = agent_id
        self.agent_type = "portfolio"
        self.task = task or PortfolioTask(task_type="allocation")
        self.available_tools = available_tools or []
        self.model_tier = "haiku"
        self._llm_client = None
        self._initialize_tools()
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize LLM client for portfolio analysis."""
        try:
            from src.agents.departments.subagents.llm_utils import get_subagent_client
            self._llm_client, self._llm_model = get_subagent_client()
            logger.info(f"PortfolioSubAgent: LLM client initialized (model={self._llm_model})")
        except Exception as e:
            logger.warning(f"PortfolioSubAgent: LLM init failed: {e}")

    def _call_llm(
        self,
        user_prompt: str,
        system_prompt: str = PORTFOLIO_SYSTEM_PROMPT,
    ) -> str:
        """
        Call LLM to generate portfolio analysis.

        Args:
            user_prompt: User's request
            system_prompt: System instructions

        Returns:
            Generated analysis text
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        try:
            response = self._llm_client.messages.create(
                model=self._llm_model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"PortfolioSubAgent: LLM call failed: {e}")
            raise

    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for this agent."""
        tools = {}

        tool_registry = {
            "get_allocation": self.get_allocation,
            "calculate_rebalance": self.calculate_rebalance,
            "track_performance": self.track_performance,
            "calculate_returns": self.calculate_returns,
            "optimize_weights": self.optimize_weights,
            # LLM-powered methods
            "analyze_allocation": self.analyze_allocation,
            "suggest_rebalance": self.suggest_rebalance,
            "generate_performance_report": self.generate_performance_report,
        }

        for tool_name in self.available_tools:
            if tool_name in tool_registry:
                tools[tool_name] = tool_registry[tool_name]

        return tools

    def analyze_allocation(
        self,
        positions: List[Dict[str, Any]],
        total_value: float,
        analysis_focus: str = "general",
    ) -> Dict[str, Any]:
        """
        Analyze portfolio allocation using LLM from natural language.

        Args:
            positions: List of open positions
            total_value: Total portfolio value
            analysis_focus: Focus area (general, risk, diversification, correlation)

        Returns:
            LLM-generated allocation analysis
        """
        logger.info(f"PortfolioSubAgent: Analyzing allocation with focus: {analysis_focus}")

        if not self._llm_client:
            return {
                "status": "error",
                "message": "LLM client not initialized",
            }

        try:
            # Calculate basic allocation
            allocation_data = self.get_allocation(positions, total_value)

            # Format data for LLM
            allocation_summary = self._format_allocation_for_llm(allocation_data)

            user_prompt = f"""Analyze the following portfolio allocation:

Total Portfolio Value: ${total_value:,.2f}

Current Allocation:
{allocation_summary}

Focus area: {analysis_focus}

Provide a detailed analysis including:
1. Current allocation assessment
2. Risk assessment
3. Diversification analysis
4. Specific recommendations"""

            analysis = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=PORTFOLIO_SYSTEM_PROMPT,
            )

            return {
                "analysis": analysis,
                "allocation": allocation_data,
                "focus": analysis_focus,
                "status": "analyzed",
            }

        except Exception as e:
            logger.error(f"PortfolioSubAgent: Allocation analysis failed: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    def _format_allocation_for_llm(self, allocation_data: Dict[str, Any]) -> str:
        """Format allocation data for LLM consumption."""
        lines = []
        allocation = allocation_data.get("allocation", {})

        for symbol, data in allocation.items():
            weight = data.get("weight", 0) * 100
            value = data.get("value", 0)
            volume = data.get("volume", 0)
            lines.append(f"  - {symbol}: {weight:.1f}% (${value:,.2f}, {volume:.2f} lots)")

        return "\n".join(lines) if lines else "  No positions"

    def suggest_rebalance(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        total_value: float,
        reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Suggest portfolio rebalancing using LLM.

        Args:
            current_allocation: Current allocation weights (as decimals, e.g., 0.25)
            target_allocation: Target allocation weights (as decimals)
            total_value: Total portfolio value
            reasoning: Optional reasoning or context for rebalancing

        Returns:
            LLM-generated rebalancing suggestions
        """
        logger.info("PortfolioSubAgent: Generating rebalancing suggestions")

        if not self._llm_client:
            return {
                "status": "error",
                "message": "LLM client not initialized",
            }

        try:
            # Calculate basic rebalance
            rebalance_data = self.calculate_rebalance(
                current_allocation, target_allocation, total_value
            )

            # Format data for LLM
            current_summary = self._format_weights_for_llm(current_allocation)
            target_summary = self._format_weights_for_llm(target_allocation)

            user_prompt = f"""Suggest portfolio rebalancing strategy:

Total Portfolio Value: ${total_value:,.2f}

Current Allocation:
{current_summary}

Target Allocation:
{target_summary}

Calculated Trades:
{self._format_trades_for_llm(rebalance_data.get('rebalance_trades', []))}

{f"Additional context: {reasoning}" if reasoning else ""}

Provide:
1. Summary of rebalancing needs
2. Risk considerations
3. Priority actions
4. Timing recommendations"""

            suggestions = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=PORTFOLIO_SYSTEM_PROMPT,
            )

            return {
                "suggestions": suggestions,
                "rebalance_trades": rebalance_data,
                "status": "suggested",
            }

        except Exception as e:
            logger.error(f"PortfolioSubAgent: Rebalancing suggestion failed: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    def _format_weights_for_llm(self, weights: Dict[str, float]) -> str:
        """Format weights for LLM consumption."""
        lines = []
        for symbol, weight in weights.items():
            pct = weight * 100
            lines.append(f"  - {symbol}: {pct:.1f}%")
        return "\n".join(lines) if lines else "  None"

    def _format_trades_for_llm(self, trades: List[Dict[str, Any]]) -> str:
        """Format trades for LLM consumption."""
        if not trades:
            return "  No trades needed"
        lines = []
        for trade in trades:
            symbol = trade.get("symbol", "UNKNOWN")
            side = trade.get("side", "buy")
            volume = trade.get("volume", 0)
            value = trade.get("trade_value", 0)
            lines.append(f"  - {side.upper()} {symbol}: {volume:.2f} lots (${value:,.2f})")
        return "\n".join(lines)

    def generate_performance_report(
        self,
        positions: List[Dict[str, Any]],
        period_start_value: float,
        period_end_value: float,
        equity_curve: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Generate performance analysis report using LLM.

        Args:
            positions: Current positions
            period_start_value: Value at start of period
            period_end_value: Value at end of period
            equity_curve: Optional historical equity values

        Returns:
            LLM-generated performance report
        """
        logger.info("PortfolioSubAgent: Generating performance report")

        if not self._llm_client:
            return {
                "status": "error",
                "message": "LLM client not initialized",
            }

        try:
            # Calculate performance metrics
            perf_data = self.track_performance(positions, period_start_value, period_end_value)

            # Calculate returns if equity curve provided
            returns_data = {}
            if equity_curve and len(equity_curve) >= 2:
                returns_data = self.calculate_returns(equity_curve)

            # Format data for LLM
            perf_summary = self._format_performance_for_llm(perf_data)
            returns_summary = self._format_returns_for_llm(returns_data)

            user_prompt = f"""Generate a comprehensive portfolio performance report:

Period Performance:
{perf_summary}

Risk-Adjusted Returns:
{returns_summary}

Current Positions:
{self._format_positions_for_llm(positions)}

Provide:
1. Executive summary
2. Performance attribution
3. Risk analysis
4. Key insights
5. Recommendations for improvement"""

            report = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=PERFORMANCE_REPORT_PROMPT,
            )

            return {
                "report": report,
                "performance": perf_data,
                "returns": returns_data,
                "status": "generated",
            }

        except Exception as e:
            logger.error(f"PortfolioSubAgent: Performance report generation failed: {e}")
            return {
                "status": "error",
                "message": str(e),
            }

    def _format_performance_for_llm(self, perf_data: Dict[str, Any]) -> str:
        """Format performance data for LLM consumption."""
        return f"""  Period Return: {perf_data.get('period_return', 0):.2f}%
  Absolute Return: ${perf_data.get('absolute_return', 0):,.2f}
  Total P&L: ${perf_data.get('total_pnl', 0):,.2f}
  Start Value: ${perf_data.get('period_start_value', 0):,.2f}
  End Value: ${perf_data.get('period_end_value', 0):,.2f}
  Position Count: {perf_data.get('position_count', 0)}"""

    def _format_returns_for_llm(self, returns_data: Dict[str, Any]) -> str:
        """Format returns data for LLM consumption."""
        if not returns_data:
            return "  No returns data available"
        return f"""  Total Return: {returns_data.get('total_return', 0):.2f}%
  Avg Return: {returns_data.get('avg_return', 0):.4f}%
  Volatility: {returns_data.get('volatility', 0):.2f}%
  Sharpe Ratio: {returns_data.get('sharpe_ratio', 0):.2f}
  Max Drawdown: {returns_data.get('max_drawdown', 0):.2f}%"""

    def _format_positions_for_llm(self, positions: List[Dict[str, Any]]) -> str:
        """Format positions for LLM consumption."""
        lines = []
        for pos in positions:
            symbol = pos.get("symbol", "UNKNOWN")
            volume = pos.get("volume", 0)
            entry = pos.get("entry_price", 0)
            current = pos.get("current_price", entry)
            pnl = pos.get("pnl", 0)
            lines.append(f"  - {symbol}: {volume:.2f} lots @ {entry:.5f} (current: {current:.5f}, P&L: ${pnl:,.2f})")
        return "\n".join(lines) if lines else "  No positions"

    def get_allocation(
        self,
        positions: List[Dict[str, Any]],
        total_value: float,
    ) -> Dict[str, Any]:
        """
        Get current portfolio allocation.

        Args:
            positions: List of open positions
            total_value: Total portfolio value

        Returns:
            Allocation breakdown
        """
        if total_value == 0:
            return {
                "status": "error",
                "message": "Total value cannot be zero",
            }

        allocation = {}
        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            volume = position.get("volume", 0)
            current_price = position.get("current_price", position.get("entry_price", 1.0))

            position_value = volume * current_price * 100000
            allocation[symbol] = {
                "value": round(position_value, 2),
                "weight": round(position_value / total_value, 4),
                "volume": volume,
            }

        return {
            "allocation": allocation,
            "total_value": round(total_value, 2),
            "position_count": len(positions),
            "status": "calculated",
        }

    def calculate_rebalance(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        total_value: float,
    ) -> Dict[str, Any]:
        """
        Calculate rebalancing trades needed.

        Args:
            current_allocation: Current allocation weights
            target_allocation: Target allocation weights
            total_value: Total portfolio value

        Returns:
            Rebalancing trades
        """
        rebalance_trades = []

        # Get all symbols
        all_symbols = set(current_allocation.keys()) | set(target_allocation.keys())

        for symbol in all_symbols:
            current_weight = current_allocation.get(symbol, 0)
            target_weight = target_allocation.get(symbol, 0)
            weight_diff = target_weight - current_weight

            # Calculate trade value
            trade_value = weight_diff * total_value
            trade_volume = abs(trade_value) / 100000  # Convert to lots

            if abs(trade_volume) > 0.01:  # Minimum trade size
                rebalance_trades.append({
                    "symbol": symbol,
                    "side": "buy" if trade_value > 0 else "sell",
                    "volume": round(trade_volume, 2),
                    "current_weight": round(current_weight, 4),
                    "target_weight": round(target_weight, 4),
                    "trade_value": round(trade_value, 2),
                })

        return {
            "rebalance_trades": rebalance_trades,
            "total_trades": len(rebalance_trades),
            "total_rebalance_value": round(sum(t["trade_value"] for t in rebalance_trades), 2),
            "status": "calculated",
        }

    def track_performance(
        self,
        positions: List[Dict[str, Any]],
        period_start_value: float,
        period_end_value: float,
    ) -> Dict[str, Any]:
        """
        Track portfolio performance.

        Args:
            positions: Current positions
            period_start_value: Value at start of period
            period_end_value: Value at end of period

        Returns:
            Performance metrics
        """
        # Calculate returns
        period_return = (period_end_value - period_start_value) / period_start_value if period_start_value > 0 else 0

        # Calculate position-level P&L
        total_pnl = 0
        pnl_by_symbol = {}

        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            volume = position.get("volume", 0)
            entry_price = position.get("entry_price", 1.0)
            current_price = position.get("current_price", entry_price)

            position_pnl = (current_price - entry_price) * volume * 100000
            total_pnl += position_pnl
            pnl_by_symbol[symbol] = round(position_pnl, 2)

        return {
            "period_return": round(period_return * 100, 2),
            "absolute_return": round(period_end_value - period_start_value, 2),
            "total_pnl": round(total_pnl, 2),
            "pnl_by_symbol": pnl_by_symbol,
            "period_start_value": round(period_start_value, 2),
            "period_end_value": round(period_end_value, 2),
            "position_count": len(positions),
            "status": "calculated",
        }

    def calculate_returns(
        self,
        equity_curve: List[float],
        risk_free_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate risk-adjusted returns.

        Args:
            equity_curve: List of portfolio values over time
            risk_free_rate: Risk-free rate for Sharpe calculation

        Returns:
            Return metrics
        """
        if len(equity_curve) < 2:
            return {
                "status": "error",
                "message": "Insufficient data for return calculation",
            }

        # Calculate returns
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)

        # Total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

        # Average return
        avg_return = sum(returns) / len(returns) if returns else 0

        # Volatility (annualized)
        import math
        volatility = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / len(returns)) * math.sqrt(252) if returns else 0

        # Sharpe ratio
        sharpe_ratio = (avg_return * 252 - risk_free_rate) / volatility if volatility > 0 else 0

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return {
            "total_return": round(total_return * 100, 2),
            "avg_return": round(avg_return * 100, 4),
            "volatility": round(volatility * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_dd * 100, 2),
            "periods": len(returns),
            "status": "calculated",
        }

    def optimize_weights(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        volatility: Dict[str, float],
        correlation_matrix: Optional[Dict[str, Dict[str, float]]] = None,
        target_return: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Optimize portfolio weights.

        Args:
            symbols: List of symbols
            expected_returns: Expected returns per symbol
            volatility: Volatility per symbol
            correlation_matrix: Correlation matrix
            target_return: Target return (optional)

        Returns:
            Optimized weights
        """
        # Simplified equal-weight optimization as placeholder
        # In production, would use mean-variance optimization

        n = len(symbols)
        if n == 0:
            return {
                "status": "error",
                "message": "No symbols provided",
            }

        # Equal weight as baseline
        weights = {symbol: 1.0 / n for symbol in symbols}

        return {
            "weights": weights,
            "method": "equal_weight",
            "expected_return": sum(expected_returns.get(s, 0) * weights[s] for s in symbols),
            "expected_volatility": sum(volatility.get(s, 0) * weights[s] for s in symbols),
            "status": "optimized",
        }

    def execute(self) -> Dict[str, Any]:
        """
        Execute the assigned task.

        Returns:
            Task execution results
        """
        task_type = self.task.task_type

        if task_type == "allocation":
            return self._execute_allocation()
        elif task_type == "rebalancing":
            return self._execute_rebalancing()
        elif task_type == "performance":
            return self._execute_performance()
        elif task_type == "optimization":
            return self._execute_optimization()
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}",
            }

    def _execute_allocation(self) -> Dict[str, Any]:
        """Execute allocation task."""
        params = self.task.parameters
        positions = params.get("positions", [])
        total_value = params.get("total_value", 10000)
        return self.get_allocation(positions, total_value)

    def _execute_rebalancing(self) -> Dict[str, Any]:
        """Execute rebalancing task."""
        params = self.task.parameters
        current = params.get("current_allocation", {})
        target = params.get("target_allocation", {})
        total_value = params.get("total_value", 10000)
        return self.calculate_rebalance(current, target, total_value)

    def _execute_performance(self) -> Dict[str, Any]:
        """Execute performance tracking task."""
        params = self.task.parameters
        positions = params.get("positions", [])
        start_value = params.get("period_start_value", 10000)
        end_value = params.get("period_end_value", 10500)
        return self.track_performance(positions, start_value, end_value)

    def _execute_optimization(self) -> Dict[str, Any]:
        """Execute optimization task."""
        params = self.task.parameters
        symbols = params.get("symbols", ["EURUSD", "GBPUSD", "USDJPY"])
        expected_returns = params.get("expected_returns", {})
        volatility = params.get("volatility", {})
        correlation = params.get("correlation_matrix", None)
        target = params.get("target_return", None)
        return self.optimize_weights(symbols, expected_returns, volatility, correlation, target)

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "allocation_management",
            "rebalancing",
            "performance_tracking",
            "returns_calculation",
            "portfolio_optimization",
            # LLM-powered capabilities
            "llm_allocation_analysis",
            "llm_rebalance_suggestions",
            "llm_performance_reporting",
        ]
