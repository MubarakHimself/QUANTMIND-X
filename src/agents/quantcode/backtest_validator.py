"""
Backtest Validation for QuantCode Agent.

This module provides validation of backtest results against minimum requirements
and performance criteria.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation severity levels."""
    CRITICAL = "critical"  # Must pass
    WARNING = "warning"    # Should pass
    INFO = "info"          # Informational


@dataclass
class ValidationRule:
    """A single validation rule."""
    name: str
    description: str
    level: ValidationLevel
    check: callable
    threshold: float
    comparison: str  # 'gte', 'lte', 'eq', 'gt', 'lt'
    
    def validate(self, value: float) -> bool:
        """Validate a value against this rule."""
        if self.comparison == 'gte':
            return value >= self.threshold
        elif self.comparison == 'lte':
            return value <= self.threshold
        elif self.comparison == 'gt':
            return value > self.threshold
        elif self.comparison == 'lt':
            return value < self.threshold
        elif self.comparison == 'eq':
            return value == self.threshold
        return False


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    rule_name: str
    passed: bool
    actual_value: float
    threshold: float
    level: ValidationLevel
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_name": self.rule_name,
            "passed": self.passed,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "level": self.level.value,
            "message": self.message
        }


@dataclass
class BacktestValidationResult:
    """Complete backtest validation result."""
    passed: bool
    critical_passed: bool
    warnings: int
    results: List[ValidationResult]
    summary: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "critical_passed": self.critical_passed,
            "warnings": self.warnings,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "recommendations": self.recommendations
        }


# Default validation rules
DEFAULT_VALIDATION_RULES = [
    # Critical rules - must pass
    ValidationRule(
        name="minimum_trades",
        description="Minimum number of trades for statistical significance",
        level=ValidationLevel.CRITICAL,
        threshold=100,
        comparison="gte",
        check=lambda r: r.get("total_trades", 0)
    ),
    ValidationRule(
        name="minimum_win_rate",
        description="Minimum win rate for profitability",
        level=ValidationLevel.CRITICAL,
        threshold=0.40,
        comparison="gte",
        check=lambda r: r.get("win_rate", 0)
    ),
    ValidationRule(
        name="maximum_drawdown",
        description="Maximum acceptable drawdown",
        level=ValidationLevel.CRITICAL,
        threshold=0.30,
        comparison="lte",
        check=lambda r: r.get("max_drawdown", 1)
    ),
    
    # Warning rules - should pass
    ValidationRule(
        name="minimum_sharpe_ratio",
        description="Minimum Sharpe ratio for risk-adjusted returns",
        level=ValidationLevel.WARNING,
        threshold=1.0,
        comparison="gte",
        check=lambda r: r.get("sharpe_ratio", 0)
    ),
    ValidationRule(
        name="minimum_profit_factor",
        description="Minimum profit factor for profitability",
        level=ValidationLevel.WARNING,
        threshold=1.2,
        comparison="gte",
        check=lambda r: r.get("profit_factor", 0)
    ),
    ValidationRule(
        name="maximum_recovery_factor",
        description="Maximum recovery factor (too high may indicate curve fitting)",
        level=ValidationLevel.WARNING,
        threshold=10.0,
        comparison="lte",
        check=lambda r: r.get("recovery_factor", 0)
    ),
    ValidationRule(
        name="minimum_average_rr",
        description="Minimum average risk-reward ratio",
        level=ValidationLevel.WARNING,
        threshold=0.5,
        comparison="gte",
        check=lambda r: r.get("average_risk_reward", 0)
    ),
    
    # Info rules - informational
    ValidationRule(
        name="minimum_net_profit",
        description="Minimum net profit in account currency",
        level=ValidationLevel.INFO,
        threshold=0,
        comparison="gt",
        check=lambda r: r.get("net_profit", 0)
    ),
    ValidationRule(
        name="maximum_consecutive_losses",
        description="Maximum consecutive losing trades",
        level=ValidationLevel.INFO,
        threshold=10,
        comparison="lte",
        check=lambda r: r.get("consecutive_losses", 0)
    ),
    ValidationRule(
        name="minimum_expectancy",
        description="Minimum expectancy per trade",
        level=ValidationLevel.INFO,
        threshold=0,
        comparison="gt",
        check=lambda r: r.get("expectancy", 0)
    ),
]


class BacktestValidator:
    """
    Validator for backtest results.
    
    Validates backtest results against a set of rules covering:
    - Statistical significance (minimum trades)
    - Profitability (win rate, profit factor)
    - Risk management (max drawdown, Sharpe ratio)
    - Quality indicators (recovery factor, expectancy)
    """
    
    def __init__(
        self,
        rules: Optional[List[ValidationRule]] = None,
        strict_mode: bool = False
    ):
        """
        Initialize backtest validator.
        
        Args:
            rules: Custom validation rules (default: DEFAULT_VALIDATION_RULES)
            strict_mode: If True, warnings also cause validation failure
        """
        self.rules = rules or DEFAULT_VALIDATION_RULES
        self.strict_mode = strict_mode
    
    def validate(self, results: Dict[str, Any]) -> BacktestValidationResult:
        """
        Validate backtest results against all rules.
        
        Args:
            results: Dictionary containing backtest results
            
        Returns:
            BacktestValidationResult with validation status
        """
        validation_results = []
        critical_passed = True
        warning_count = 0
        recommendations = []
        
        for rule in self.rules:
            # Get value from results
            value = rule.check(results)
            
            # Validate
            passed = rule.validate(value)
            
            # Create result
            result = ValidationResult(
                rule_name=rule.name,
                passed=passed,
                actual_value=value,
                threshold=rule.threshold,
                level=rule.level,
                message=self._create_message(rule, value, passed)
            )
            validation_results.append(result)
            
            # Track status
            if rule.level == ValidationLevel.CRITICAL and not passed:
                critical_passed = False
                recommendations.append(self._create_recommendation(rule, value))
            elif rule.level == ValidationLevel.WARNING and not passed:
                warning_count += 1
                recommendations.append(self._create_recommendation(rule, value))
        
        # Determine overall pass status
        if self.strict_mode:
            passed = critical_passed and warning_count == 0
        else:
            passed = critical_passed
        
        # Create summary
        summary = self._create_summary(passed, critical_passed, warning_count)
        
        return BacktestValidationResult(
            passed=passed,
            critical_passed=critical_passed,
            warnings=warning_count,
            results=validation_results,
            summary=summary,
            recommendations=recommendations
        )
    
    def _create_message(
        self,
        rule: ValidationRule,
        value: float,
        passed: bool
    ) -> str:
        """Create validation message."""
        status = "PASSED" if passed else "FAILED"
        comparison_text = {
            'gte': '>=',
            'lte': '<=',
            'gt': '>',
            'lt': '<',
            'eq': '='
        }.get(rule.comparison, '')
        
        return f"{status}: {rule.name} = {value:.4f} (required {comparison_text} {rule.threshold})"
    
    def _create_recommendation(
        self,
        rule: ValidationRule,
        value: float
    ) -> str:
        """Create recommendation for failed rule."""
        recommendations = {
            "minimum_trades": "Increase backtest period or adjust strategy to generate more trades",
            "minimum_win_rate": "Review entry conditions to improve trade selection",
            "maximum_drawdown": "Add stricter risk management or reduce position sizes",
            "minimum_sharpe_ratio": "Improve risk-adjusted returns by filtering low-quality trades",
            "minimum_profit_factor": "Review exit strategy to improve profit taking",
            "maximum_recovery_factor": "Check for potential curve fitting in strategy parameters",
            "minimum_average_rr": "Adjust stop loss and take profit levels for better risk-reward",
            "minimum_net_profit": "Review overall strategy profitability",
            "maximum_consecutive_losses": "Add filters to avoid prolonged losing streaks",
            "minimum_expectancy": "Improve average trade outcome through better entry/exit"
        }
        
        return recommendations.get(
            rule.name,
            f"Review {rule.description}"
        )
    
    def _create_summary(
        self,
        passed: bool,
        critical_passed: bool,
        warning_count: int
    ) -> str:
        """Create validation summary."""
        if passed and warning_count == 0:
            return "All validation checks passed. Strategy meets requirements."
        elif passed:
            return f"Critical checks passed with {warning_count} warnings. Strategy acceptable."
        else:
            return "Critical validation checks failed. Strategy does not meet minimum requirements."


def validate_backtest_results(
    results: Dict[str, Any],
    strict_mode: bool = False
) -> Dict[str, Any]:
    """
    Validate backtest results against minimum requirements.
    
    This is the main entry point for backtest validation.
    
    Args:
        results: Dictionary containing backtest results with keys:
            - total_trades: Total number of trades
            - win_rate: Win rate as decimal (e.g., 0.58)
            - max_drawdown: Maximum drawdown as decimal (e.g., 0.15)
            - sharpe_ratio: Sharpe ratio
            - profit_factor: Profit factor
            - net_profit: Net profit in account currency
            - recovery_factor: Recovery factor
            - average_risk_reward: Average risk-reward ratio
            - consecutive_losses: Maximum consecutive losses
            - expectancy: Expectancy per trade
        strict_mode: If True, warnings also cause validation failure
        
    Returns:
        Dictionary containing validation results
    """
    validator = BacktestValidator(strict_mode=strict_mode)
    result = validator.validate(results)
    return result.to_dict()


def calculate_backtest_metrics(
    trades: List[Dict[str, Any]],
    initial_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    Calculate comprehensive backtest metrics from trade list.
    
    Args:
        trades: List of trade dictionaries with 'profit' key
        initial_balance: Initial account balance
        
    Returns:
        Dictionary containing calculated metrics
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "net_profit": 0,
            "recovery_factor": 0,
            "average_risk_reward": 0,
            "consecutive_losses": 0,
            "expectancy": 0
        }
    
    # Basic metrics
    total_trades = len(trades)
    profits = [t.get("profit", 0) for t in trades]
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    # Profit metrics
    gross_profit = sum(winning_trades)
    gross_loss = abs(sum(losing_trades))
    net_profit = sum(profits)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Drawdown calculation
    balance = initial_balance
    peak_balance = initial_balance
    max_drawdown = 0
    equity_curve = [initial_balance]
    
    for profit in profits:
        balance += profit
        equity_curve.append(balance)
        
        if balance > peak_balance:
            peak_balance = balance
        
        drawdown = (peak_balance - balance) / peak_balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Sharpe ratio (simplified, assuming risk-free rate = 0)
    if len(profits) > 1:
        avg_profit = sum(profits) / len(profits)
        variance = sum((p - avg_profit) ** 2 for p in profits) / len(profits)
        std_dev = variance ** 0.5
        
        # Annualized Sharpe (assuming daily trades)
        sharpe_ratio = (avg_profit * 252) / (std_dev * (252 ** 0.5)) if std_dev > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Recovery factor
    recovery_factor = net_profit / (max_drawdown * initial_balance) if max_drawdown > 0 else 0
    
    # Consecutive losses
    max_consecutive_losses = 0
    current_consecutive = 0
    for profit in profits:
        if profit < 0:
            current_consecutive += 1
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
        else:
            current_consecutive = 0
    
    # Expectancy
    avg_win = gross_profit / win_count if win_count > 0 else 0
    avg_loss = gross_loss / loss_count if loss_count > 0 else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Average risk-reward
    avg_risk_reward = avg_win / avg_loss if avg_loss > 0 else 0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "net_profit": net_profit,
        "recovery_factor": recovery_factor,
        "average_risk_reward": avg_risk_reward,
        "consecutive_losses": max_consecutive_losses,
        "expectancy": expectancy,
        "equity_curve": equity_curve,
        "win_count": win_count,
        "loss_count": loss_count,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }
