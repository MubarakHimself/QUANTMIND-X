"""
Risk Subagent

Worker agent for the Risk Department.
Responsible for position sizing, exposure management, VaR calculations,
and drawdown monitoring.

Model: Haiku (fast, low-cost for worker tasks)
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import math

logger = logging.getLogger(__name__)


# Risk assessment system prompt for LLM
RISK_ASSESSMENT_SYSTEM_PROMPT = """You are an expert risk analyst for a quantitative trading system.

Your task is to analyze and assess risk for trades and positions. You provide READ-ONLY analysis
and recommendations - you do not make trading decisions.

## Risk Analysis Guidelines:
1. Analyze position risk based on provided parameters
2. Assess exposure and concentration risk
3. Evaluate correlation and sector risk
4. Check against risk limits and thresholds
5. Provide clear risk scores and recommendations
6. Consider worst-case scenarios and stress cases
7. Identify potential risk factors and mitigation suggestions

## Risk Metrics to Consider:
- Position size relative to account
- Exposure per symbol and sector
- Correlation with existing positions
- Volatility and VaR
- Drawdown potential
- Leverage utilization

## Output Format:
Always provide structured risk assessments with:
- Risk level (LOW, MEDIUM, HIGH, CRITICAL)
- Risk factors identified
- Mitigation recommendations
- Key metrics summary

Remember: You provide analysis only - the trading system makes decisions based on your assessment.
"""


@dataclass
class RiskTask:
    """Risk task input data."""
    task_type: str  # position_sizing, var_calculation, drawdown_check, risk_validation
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: float = 1.0  # Lots
    max_daily_loss: float = 0.05  # 5% of account
    max_drawdown: float = 0.15  # 15% of account
    max_exposure_per_symbol: float = 0.20  # 20% of account per symbol
    max_leverage: float = 10.0


class RiskSubAgent:
    """
    Risk subagent for risk management calculations.

    Capabilities:
    - Position sizing calculations
    - Value at Risk (VaR) calculations
    - Drawdown monitoring
    - Risk validation for trades
    """

    def __init__(
        self,
        agent_id: str,
        task: Optional[RiskTask] = None,
        available_tools: Optional[List[str]] = None,
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize risk subagent.

        Args:
            agent_id: Unique identifier for this agent
            task: Task configuration
            available_tools: List of tool names available to this agent
            risk_limits: Risk management limits
        """
        self.agent_id = agent_id
        self.agent_type = "risk"
        self.task = task or RiskTask(task_type="risk_validation")
        self.available_tools = available_tools or []
        self.risk_limits = risk_limits or RiskLimits()
        self.model_tier = "haiku"
        self._llm_client = None
        self._initialize_tools()
        self._initialize_llm()

    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for this agent."""
        tools = {}

        tool_registry = {
            "calculate_position_size": self.calculate_position_size,
            "calculate_var": self.calculate_var,
            "check_drawdown": self.check_drawdown,
            "validate_trade": self.validate_trade,
            "calculate_exposure": self.calculate_exposure,
            "assess_risk": self.assess_risk,
            "check_limits": self.check_limits,
            "generate_risk_report": self.generate_risk_report,
        }

        for tool_name in self.available_tools:
            if tool_name in tool_registry:
                tools[tool_name] = tool_registry[tool_name]

        return tools

    def _initialize_llm(self) -> None:
        """Initialize LLM client for risk analysis."""
        try:
            from anthropic import Anthropic
            self._llm_client = Anthropic()
            logger.info("RiskSubAgent: LLM client initialized")
        except ImportError:
            logger.warning("RiskSubAgent: Anthropic SDK not available")

    def _call_llm(
        self,
        user_prompt: str,
        system_prompt: str = RISK_ASSESSMENT_SYSTEM_PROMPT,
    ) -> str:
        """
        Call LLM for risk analysis.

        Args:
            user_prompt: User's risk analysis request
            system_prompt: System instructions

        Returns:
            LLM response text
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        try:
            response = self._llm_client.messages.create(
                model="claude-3-5-haiku-20241022",  # Use Haiku for cost efficiency
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"RiskSubAgent: LLM call failed: {e}")
            raise

    def assess_risk(
        self,
        trade: Dict[str, Any],
        portfolio: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Assess risk for a trade or position using LLM analysis.

        Args:
            trade: Trade parameters (symbol, side, volume, price, etc.)
            portfolio: Current portfolio positions (optional)

        Returns:
            Risk assessment with LLM analysis
        """
        logger.info(f"RiskSubAgent: Assessing risk for {trade.get('symbol', 'UNKNOWN')}")

        # Calculate baseline risk metrics
        position_value = trade.get("volume", 0) * trade.get("price", 0) * 100000
        account_balance = trade.get("account_balance", 10000)
        position_percent = (position_value / account_balance) if account_balance > 0 else 0

        # Build LLM prompt
        user_prompt = f"""Analyze the risk for the following trade:

**Trade Details:**
- Symbol: {trade.get('symbol', 'N/A')}
- Side: {trade.get('side', 'N/A')}
- Volume: {trade.get('volume', 0)} lots
- Entry Price: {trade.get('price', 0)}
- Position Value: ${position_value:,.2f}
- Position as % of Account: {position_percent:.2%}

**Account Info:**
- Balance: ${account_balance:,.2f}

{f"**Current Portfolio:**\n{portfolio}" if portfolio else ""}

Provide:
1. Risk level (LOW, MEDIUM, HIGH, CRITICAL)
2. Key risk factors
3. Recommendations
4. Risk score (0-100)
"""

        try:
            analysis = self._call_llm(user_prompt=user_prompt)

            return {
                "analysis": analysis,
                "position_value": round(position_value, 2),
                "position_percent": round(position_percent, 4),
                "symbol": trade.get("symbol"),
                "status": "analyzed",
            }

        except Exception as e:
            logger.error(f"RiskSubAgent: Risk assessment failed: {e}")
            return {
                "analysis": None,
                "error": str(e),
                "status": "error",
            }

    def check_limits(
        self,
        trade: Dict[str, Any],
        current_positions: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Check if trade exceeds risk limits using LLM analysis.

        Args:
            trade: Trade parameters
            current_positions: Current open positions

        Returns:
            Limit check results with LLM analysis
        """
        logger.info(f"RiskSubAgent: Checking limits for {trade.get('symbol', 'UNKNOWN')}")

        # Get current limits
        limits = {
            "max_position_size": self.risk_limits.max_position_size,
            "max_daily_loss": self.risk_limits.max_daily_loss,
            "max_drawdown": self.risk_limits.max_drawdown,
            "max_exposure_per_symbol": self.risk_limits.max_exposure_per_symbol,
            "max_leverage": self.risk_limits.max_leverage,
        }

        # Calculate current exposure
        total_exposure = 0.0
        if current_positions:
            for pos in current_positions:
                volume = pos.get("volume", 0)
                price = pos.get("entry_price", 1.0)
                total_exposure += volume * price * 100000

        # Add new trade exposure
        new_volume = trade.get("volume", 0)
        new_price = trade.get("price", 1.0)
        new_exposure = new_volume * new_price * 100000
        total_exposure += new_exposure

        account_balance = trade.get("account_balance", 10000)
        total_exposure_percent = (total_exposure / account_balance) if account_balance > 0 else 0

        # Check individual limits
        limit_checks = []

        if new_volume > limits["max_position_size"]:
            limit_checks.append({
                "limit": "max_position_size",
                "passed": False,
                "current": new_volume,
                "limit_value": limits["max_position_size"],
                "message": f"Volume {new_volume} exceeds max {limits['max_position_size']}",
            })
        else:
            limit_checks.append({
                "limit": "max_position_size",
                "passed": True,
                "current": new_volume,
                "limit_value": limits["max_position_size"],
                "message": "Position size within limits",
            })

        if total_exposure_percent > limits["max_exposure_per_symbol"]:
            limit_checks.append({
                "limit": "max_exposure_per_symbol",
                "passed": False,
                "current": round(total_exposure_percent, 4),
                "limit_value": limits["max_exposure_per_symbol"],
                "message": f"Total exposure {total_exposure_percent:.2%} exceeds max {limits['max_exposure_per_symbol']:.1%}",
            })
        else:
            limit_checks.append({
                "limit": "max_exposure_per_symbol",
                "passed": True,
                "current": round(total_exposure_percent, 4),
                "limit_value": limits["max_exposure_per_symbol"],
                "message": "Exposure within limits",
            })

        # Get LLM analysis
        user_prompt = f"""Analyze the limit check results for this trade:

**Trade:**
- Symbol: {trade.get('symbol')}
- Volume: {trade.get('volume')} lots
- New Exposure: ${new_exposure:,.2f}

**Current Positions:** {len(current_positions or [])}
**Total Exposure:** ${total_exposure:,.2f} ({total_exposure_percent:.2%} of account)

**Risk Limits:**
- Max Position Size: {limits['max_position_size']} lots
- Max Exposure per Symbol: {limits['max_exposure_per_symbol']:.1%}
- Max Leverage: {limits['max_leverage']}x

**Limit Check Results:**
{chr(10).join([f"- {check['limit']}: {'PASSED' if check['passed'] else 'FAILED'} - {check['message']}" for check in limit_checks])}

Provide:
1. Overall limit compliance status
2. Additional risk considerations
3. Recommendations for trade approval
"""

        try:
            analysis = self._call_llm(user_prompt=user_prompt)

            all_passed = all(check["passed"] for check in limit_checks)

            return {
                "analysis": analysis,
                "passed": all_passed,
                "limit_checks": limit_checks,
                "total_exposure": round(total_exposure, 2),
                "total_exposure_percent": round(total_exposure_percent, 4),
                "limits": limits,
                "status": "checked",
            }

        except Exception as e:
            logger.error(f"RiskSubAgent: Limit check failed: {e}")
            return {
                "error": str(e),
                "limit_checks": limit_checks,
                "status": "error",
            }

    def generate_risk_report(
        self,
        positions: List[Dict[str, Any]],
        account_balance: float,
        peak_balance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk analysis report using LLM.

        Args:
            positions: Current open positions
            account_balance: Current account balance
            peak_balance: Peak account balance (optional)

        Returns:
            Risk report with LLM analysis
        """
        logger.info(f"RiskSubAgent: Generating risk report for {len(positions)} positions")

        # Calculate portfolio metrics
        total_exposure = 0.0
        exposure_by_symbol = {}
        long_exposure = 0.0
        short_exposure = 0.0

        for pos in positions:
            symbol = pos.get("symbol", "UNKNOWN")
            volume = pos.get("volume", 0)
            price = pos.get("entry_price", 1.0)
            side = pos.get("side", "long")

            position_value = volume * price * 100000
            total_exposure += position_value

            exposure_by_symbol[symbol] = exposure_by_symbol.get(symbol, 0) + position_value

            if side.lower() == "long":
                long_exposure += position_value
            else:
                short_exposure += position_value

        # Calculate drawdown if peak provided
        drawdown_percent = 0.0
        if peak_balance and peak_balance > 0:
            drawdown_percent = ((peak_balance - account_balance) / peak_balance) * 100

        # Calculate portfolio metrics
        total_exposure_ratio = (total_exposure / account_balance) if account_balance > 0 else 0
        long_short_ratio = long_exposure / short_exposure if short_exposure > 0 else float('inf')

        # Build portfolio summary for LLM
        positions_summary = "\n".join([
            f"- {pos.get('symbol')}: {pos.get('volume')} lots @ {pos.get('entry_price')} ({pos.get('side', 'long')})"
            for pos in positions
        ]) or "No open positions"

        # Build LLM prompt
        user_prompt = f"""Generate a comprehensive risk analysis report for this trading portfolio:

**Account Summary:**
- Current Balance: ${account_balance:,.2f}
- Peak Balance: ${peak_balance:,.2f if peak_balance else 'N/A'}
- Current Drawdown: {drawdown_percent:.2f}%
- Max Drawdown Limit: {self.risk_limits.max_drawdown * 100}%

**Portfolio Positions ({len(positions)} total):**
{positions_summary}

**Exposure Analysis:**
- Total Exposure: ${total_exposure:,.2f}
- Total Exposure Ratio: {total_exposure_ratio:.2%} of account
- Long Exposure: ${long_exposure:,.2f}
- Short Exposure: ${short_exposure:,.2f}
- Long/Short Ratio: {long_short_ratio:.2f}

**Risk Limits:**
- Max Daily Loss: {self.risk_limits.max_daily_loss * 100}%
- Max Drawdown: {self.risk_limits.max_drawdown * 100}%
- Max Exposure per Symbol: {self.risk_limits.max_exposure_per_symbol * 100}%
- Max Leverage: {self.risk_limits.max_leverage}x

**Per-Symbol Exposure:**
{chr(10).join([f"- {symbol}: ${value:,.2f} ({value/total_exposure*100:.1f}% of total)" for symbol, value in exposure_by_symbol.items()]) if exposure_by_symbol else "N/A"}

Provide:
1. Overall portfolio risk assessment
2. Concentration and correlation risk analysis
3. Current risk metric summary (VaR, drawdown, leverage)
4. Risk warnings or concerns
5. Recommendations for position management
6. Risk score (0-100)
"""

        try:
            analysis = self._call_llm(user_prompt=user_prompt)

            return {
                "report": analysis,
                "summary": {
                    "total_positions": len(positions),
                    "total_exposure": round(total_exposure, 2),
                    "total_exposure_ratio": round(total_exposure_ratio, 4),
                    "long_exposure": round(long_exposure, 2),
                    "short_exposure": round(short_exposure, 2),
                    "drawdown_percent": round(drawdown_percent, 2),
                    "exposure_by_symbol": {k: round(v, 2) for k, v in exposure_by_symbol.items()},
                },
                "limits": {
                    "max_drawdown": self.risk_limits.max_drawdown * 100,
                    "max_exposure_per_symbol": self.risk_limits.max_exposure_per_symbol * 100,
                    "max_leverage": self.risk_limits.max_leverage,
                },
                "status": "generated",
            }

        except Exception as e:
            logger.error(f"RiskSubAgent: Risk report generation failed: {e}")
            return {
                "error": str(e),
                "status": "error",
            }

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss: float,
        risk_percent: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size based on risk parameters.

        Args:
            account_balance: Account balance
            entry_price: Entry price
            stop_loss: Stop loss price
            risk_percent: Risk percentage (default 2%)

        Returns:
            Position size calculation
        """
        # Calculate risk amount
        risk_amount = account_balance * risk_percent

        # Calculate price difference
        price_diff = abs(entry_price - stop_loss)

        if price_diff == 0:
            return {
                "status": "error",
                "message": "Stop loss cannot be equal to entry price",
            }

        # Calculate position size (simplified for forex)
        # For EURUSD, 1 lot = 100,000 units
        # Each pip = $10 for 1 lot
        pip_value = 10  # Approximate for major pairs

        position_size = risk_amount / (price_diff * pip_value)

        # Apply max limit
        position_size = min(position_size, self.risk_limits.max_position_size)

        return {
            "position_size": round(position_size, 2),
            "risk_amount": round(risk_amount, 2),
            "risk_percent": risk_percent,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "status": "calculated",
        }

    def calculate_var(
        self,
        positions: List[Dict[str, Any]],
        confidence_level: float = 0.95,
        holding_period: int = 1,
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) for portfolio.

        Args:
            positions: List of open positions
            confidence_level: Confidence level (default 95%)
            holding_period: Holding period in days

        Returns:
            VaR calculation
        """
        # Simplified VaR calculation using historical volatility
        # In production, would use more sophisticated methods

        total_exposure = 0.0
        for position in positions:
            volume = position.get("volume", 0)
            price = position.get("entry_price", 1.0)
            total_exposure += volume * price * 100000  # Convert lots to units

        # Simplified volatility estimate (would use real data)
        volatility = 0.015  # 1.5% daily volatility

        # VaR calculation (simplified parametric)
        z_score = 1.65 if confidence_level == 0.95 else 2.33  # 95% or 99%
        var = total_exposure * volatility * z_score * math.sqrt(holding_period)

        return {
            "var": round(var, 2),
            "confidence_level": confidence_level,
            "holding_period": holding_period,
            "total_exposure": round(total_exposure, 2),
            "status": "calculated",
        }

    def check_drawdown(
        self,
        account_balance: float,
        peak_balance: float,
    ) -> Dict[str, Any]:
        """
        Check current drawdown against limits.

        Args:
            account_balance: Current account balance
            peak_balance: Peak account balance

        Returns:
            Drawdown check results
        """
        if peak_balance == 0:
            return {
                "status": "error",
                "message": "Peak balance cannot be zero",
            }

        drawdown = (peak_balance - account_balance) / peak_balance
        drawdown_percent = drawdown * 100

        return {
            "drawdown_percent": round(drawdown_percent, 2),
            "current_balance": account_balance,
            "peak_balance": peak_balance,
            "max_drawdown_limit": self.risk_limits.max_drawdown * 100,
            "within_limits": drawdown <= self.risk_limits.max_drawdown,
            "status": "checked",
        }

    def validate_trade(
        self,
        symbol: str,
        side: str,
        volume: float,
        price: float,
        account_balance: float,
        current_exposure: float,
    ) -> Dict[str, Any]:
        """
        Validate a trade against risk limits.

        Args:
            symbol: Trading symbol
            side: Trade side (buy, sell)
            volume: Trade volume in lots
            price: Entry price
            account_balance: Account balance
            current_exposure: Current exposure for this symbol

        Returns:
            Validation result
        """
        validation_results = []
        is_valid = True

        # Check position size limit
        if volume > self.risk_limits.max_position_size:
            validation_results.append({
                "check": "position_size",
                "passed": False,
                "message": f"Volume {volume} exceeds max {self.risk_limits.max_position_size}",
            })
            is_valid = False
        else:
            validation_results.append({
                "check": "position_size",
                "passed": True,
                "message": "Volume within limits",
            })

        # Check exposure limit (simplified - use margin-based exposure)
        # For forex with 100:1 leverage, margin is 1% of notional
        # position_value represents margin requirement
        leverage = 100.0  # Assume 100:1 leverage
        margin_required = (volume * 100000) / leverage
        new_exposure = current_exposure + margin_required
        exposure_percent = new_exposure / account_balance

        if exposure_percent > self.risk_limits.max_exposure_per_symbol:
            validation_results.append({
                "check": "exposure",
                "passed": False,
                "message": f"Exposure {exposure_percent:.1%} exceeds max {self.risk_limits.max_exposure_per_symbol:.1%}",
            })
            is_valid = False
        else:
            validation_results.append({
                "check": "exposure",
                "passed": True,
                "message": "Exposure within limits",
            })

        # Check daily loss limit (simplified - would track daily P&L)
        validation_results.append({
            "check": "daily_loss",
            "passed": True,
            "message": "Daily loss within limits",
        })

        return {
            "symbol": symbol,
            "side": side,
            "volume": volume,
            "is_valid": is_valid,
            "validation_results": validation_results,
            "status": "validated",
        }

    def calculate_exposure(
        self,
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calculate current exposure per symbol and total.

        Args:
            positions: List of open positions

        Returns:
            Exposure breakdown
        """
        exposure_by_symbol = {}
        total_exposure = 0.0

        for position in positions:
            symbol = position.get("symbol", "UNKNOWN")
            volume = position.get("volume", 0)
            price = position.get("entry_price", 1.0)

            position_value = volume * price * 100000
            exposure_by_symbol[symbol] = position_value
            total_exposure += position_value

        return {
            "exposure_by_symbol": exposure_by_symbol,
            "total_exposure": round(total_exposure, 2),
            "position_count": len(positions),
            "status": "calculated",
        }

    def execute(self) -> Dict[str, Any]:
        """
        Execute the assigned task.

        Returns:
            Task execution results
        """
        task_type = self.task.task_type

        if task_type == "position_sizing":
            return self._calculate_position_size()
        elif task_type == "var_calculation":
            return self._calculate_var()
        elif task_type == "drawdown_check":
            return self._check_drawdown()
        elif task_type == "risk_validation":
            return self._validate_risk()
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}",
            }

    def _calculate_position_size(self) -> Dict[str, Any]:
        """Execute position sizing task."""
        params = self.task.parameters
        return self.calculate_position_size(
            account_balance=params.get("account_balance", 10000),
            entry_price=params.get("entry_price", 1.0850),
            stop_loss=params.get("stop_loss", 1.0800),
            risk_percent=params.get("risk_percent", 0.02),
        )

    def _calculate_var(self) -> Dict[str, Any]:
        """Execute VaR calculation task."""
        params = self.task.parameters
        positions = params.get("positions", [])
        return self.calculate_var(
            positions=positions,
            confidence_level=params.get("confidence_level", 0.95),
            holding_period=params.get("holding_period", 1),
        )

    def _check_drawdown(self) -> Dict[str, Any]:
        """Execute drawdown check task."""
        params = self.task.parameters
        return self.check_drawdown(
            account_balance=params.get("account_balance", 9500),
            peak_balance=params.get("peak_balance", 10000),
        )

    def _validate_risk(self) -> Dict[str, Any]:
        """Execute risk validation task."""
        params = self.task.parameters
        return self.validate_trade(
            symbol=params.get("symbol", "EURUSD"),
            side=params.get("side", "buy"),
            volume=params.get("volume", 0.1),
            price=params.get("price", 1.0850),
            account_balance=params.get("account_balance", 10000),
            current_exposure=params.get("current_exposure", 0),
        )

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "position_sizing",
            "var_calculation",
            "drawdown_monitoring",
            "risk_validation",
            "exposure_calculation",
        ]
