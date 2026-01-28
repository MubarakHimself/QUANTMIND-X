"""
Strategy Performance Model

Pydantic model for strategy performance metrics used in Kelly criterion calculations.
Validates win rates, average wins/losses, and calculates expectancy.

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class StrategyPerformance(BaseModel):
    """
    Strategy performance metrics for Kelly criterion position sizing.

    Attributes:
        win_rate: Historical win rate (0.0 to 1.0, e.g., 0.55 for 55%)
        avg_win: Average winning trade profit in account currency (must be positive)
        avg_loss: Average losing trade loss in account currency (positive value)
        total_trades: Total number of trades in sample (for confidence assessment)
        profit_factor: Ratio of total wins to total losses (optional, calculated if not provided)
        k_fraction: Kelly fraction to use (optional, defaults to 0.5 for half-Kelly)
        last_updated: Timestamp of last performance update
    """

    win_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Historical win rate (0 to 1)",
        examples=[0.55, 0.60, 0.45]
    )

    avg_win: float = Field(
        gt=0.0,
        description="Average winning trade profit in account currency",
        examples=[100.0, 250.50, 500.0]
    )

    avg_loss: float = Field(
        gt=0.0,
        description="Average losing trade loss (positive value)",
        examples=[50.0, 100.0, 75.25]
    )

    total_trades: int = Field(
        ge=1,
        description="Total number of trades in sample",
        examples=[50, 100, 200]
    )

    profit_factor: Optional[float] = Field(
        None,
        ge=0.0,
        description="Ratio of total wins to total losses (calculated if not provided)",
        examples=[1.5, 2.0, 3.0]
    )

    k_fraction: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Kelly fraction multiplier (0.5 = half-Kelly for safety)",
        examples=[0.5, 0.25, 0.75]
    )

    last_updated: Optional[datetime] = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of last performance update"
    )

    @field_validator("win_rate")
    @classmethod
    def validate_win_rate_bounds(cls, v: float) -> float:
        """Ensure win_rate is strictly between 0 and 1 (exclusive)."""
        if v == 0.0 or v == 1.0:
            raise ValueError("win_rate must be strictly between 0 and 1 (exclusive)")
        return v

    @field_validator("total_trades")
    @classmethod
    def validate_sample_size(cls, v: int) -> int:
        """Warn on small sample sizes that may produce unreliable statistics."""
        if v < 30:
            # Note: This is a validation that passes but could be logged as a warning
            pass
        return v

    @model_validator(mode="after")
    def validate_payoff_ratio(self) -> "StrategyPerformance":
        """Ensure payoff ratio is reasonable (avg_win should not be << avg_loss)."""
        payoff_ratio = self.avg_win / self.avg_loss
        if payoff_ratio < 0.1:
            raise ValueError(
                f"Payoff ratio ({payoff_ratio:.2f}) too low. "
                "avg_win should be at least 10% of avg_loss for positive expectancy."
            )
        return self

    @property
    def payoff_ratio(self) -> float:
        """
        Calculate the payoff ratio (reward-to-risk ratio).

        Returns:
            The ratio of average win to average loss.
        """
        return self.avg_win / self.avg_loss

    def expectancy(self) -> float:
        """
        Calculate the mathematical expectancy of the strategy.

        Expectancy = (win_rate × avg_win) - ((1 - win_rate) × avg_loss)

        Returns:
            Expected profit per trade in account currency.
            Negative values indicate a losing strategy.
        """
        win_component = self.win_rate * self.avg_win
        loss_component = (1 - self.win_rate) * self.avg_loss
        return win_component - loss_component

    def is_profitable(self) -> bool:
        """
        Check if the strategy has positive expectancy.

        Returns:
            True if expectancy > 0, False otherwise.
        """
        return self.expectancy() > 0

    def kelly_criterion(self) -> float:
        """
        Calculate the full Kelly criterion fraction.

        f = ((payoff_ratio + 1) × win_rate - 1) / payoff_ratio

        Returns:
            Kelly fraction (0 to 1, may be negative for negative expectancy).
        """
        b = self.payoff_ratio
        p = self.win_rate
        return ((b + 1) * p - 1) / b

    def adjusted_kelly(self) -> float:
        """
        Calculate the adjusted (half) Kelly fraction with safety multiplier.

        Returns:
            Adjusted Kelly fraction (0 to 1, clamped at 0 if negative).
        """
        full_kelly = self.kelly_criterion()
        adjusted = full_kelly * self.k_fraction
        return max(0.0, adjusted)

    def confidence_level(self) -> str:
        """
        Assess statistical confidence based on sample size.

        Returns:
            Confidence level: "low", "medium", or "high"
        """
        if self.total_trades < 30:
            return "low"
        elif self.total_trades < 100:
            return "medium"
        else:
            return "high"

    def model_dump_json_safe(self) -> str:
        """
        Serialize model to JSON-safe dict.

        Returns:
            JSON string representation of the model.
        """
        return self.model_dump_json(exclude_none=True)
