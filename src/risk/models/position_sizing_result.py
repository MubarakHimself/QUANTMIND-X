"""
Position Sizing Result Model

Pydantic model for the final position sizing result with margin calculations.
Combines strategy performance, market physics, and sizing recommendation into
a complete result for trading execution.

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class PositionSizingResult(BaseModel):
    """
    Final position sizing result with complete financial breakdown.

    Provides all information needed for trade execution:
    - Position size in lots
    - Risk amount in account currency
    - Margin requirements
    - Step-by-step calculation trace

    Attributes:
        account_balance: Current account balance in base currency
        risk_amount: Dollar amount at risk for this trade
        stop_loss_pips: Stop loss distance in pips/points
        pip_value: Value per pip per lot
        lot_size: Recommended position size in lots
        estimated_margin: Estimated margin required for position
        remaining_margin: Remaining margin after position is opened
        calculation_steps: List of calculation steps for audit trail
    """

    # Account and Risk Parameters
    account_balance: float = Field(
        gt=0.0,
        description="Current account balance in base currency",
        examples=[10000.0, 50000.0, 100000.0]
    )

    risk_amount: float = Field(
        ge=0.0,
        description="Dollar amount at risk for this trade",
        examples=[200.0, 500.0, 100.0]
    )

    # Trade Parameters
    stop_loss_pips: float = Field(
        gt=0.0,
        description="Stop loss distance in pips/points",
        examples=[20.0, 50.0, 15.0]
    )

    pip_value: float = Field(
        gt=0.0,
        description="Value per pip per lot (e.g., $10 for standard lot)",
        examples=[10.0, 1.0, 0.1]
    )

    # Result
    lot_size: float = Field(
        ge=0.0,
        description="Recommended position size in lots",
        examples=[0.15, 1.25, 0.05]
    )

    # Margin Calculations
    estimated_margin: Optional[float] = Field(
        None,
        ge=0.0,
        description="Estimated margin required for position",
        examples=[500.0, 2500.0, 150.0]
    )

    remaining_margin: Optional[float] = Field(
        None,
        ge=0.0,
        description="Remaining margin after position is opened",
        examples=[9500.0, 47500.0, 9850.0]
    )

    # Audit Trail
    calculation_steps: List[str] = Field(
        default_factory=list,
        description="Step-by-step calculation trace for audit"
    )

    @field_validator("account_balance")
    @classmethod
    def validate_account_balance(cls, v: float) -> float:
        """Ensure account balance is reasonable."""
        if v < 100:
            raise ValueError(f"account_balance ({v}) too low, minimum $100")
        if v > 1e9:
            raise ValueError(f"account_balance ({v}) unrealistically high")
        return v

    @field_validator("risk_amount")
    @classmethod
    def validate_risk_amount(cls, v: float) -> float:
        """Ensure risk amount is non-negative and reasonable."""
        if v < 0:
            raise ValueError(f"risk_amount ({v}) cannot be negative")
        return v

    @field_validator("lot_size")
    @classmethod
    def validate_lot_size(cls, v: float) -> float:
        """Ensure lot size is positive or zero."""
        if v < 0:
            raise ValueError(f"lot_size ({v}) cannot be negative")
        return v

    @field_validator("stop_loss_pips", "pip_value")
    @classmethod
    def validate_trade_parameters(cls, v: float, info) -> float:
        """Ensure trade parameters are positive."""
        if v <= 0:
            field_name = info.field_name
            raise ValueError(f"{field_name} ({v}) must be positive")
        return v

    def risk_percentage(self) -> float:
        """
        Calculate risk as percentage of account balance.

        Returns:
            Risk percentage (0 to 1, where 0.02 = 2%)
        """
        if self.account_balance == 0:
            return 0.0
        return self.risk_amount / self.account_balance

    def risk_per_lot(self) -> float:
        """
        Calculate risk amount per lot.

        Returns:
            Risk in account currency per lot
        """
        return self.stop_loss_pips * self.pip_value

    def add_step(self, step_description: str) -> None:
        """
        Add a calculation step to the audit trail.

        Args:
            step_description: Description of the calculation step
        """
        self.calculation_steps.append(step_description)

    def get_calculation_summary(self) -> str:
        """
        Get a formatted summary of the calculation.

        Returns:
            Multi-line string with full calculation breakdown
        """
        lines = [
            f"Position Sizing Calculation",
            f"=" * 40,
            f"Account Balance: ${self.account_balance:,.2f}",
            f"Risk Amount: ${self.risk_amount:,.2f} ({self.risk_percentage()*100:.2f}%)",
            f"Stop Loss: {self.stop_loss_pips:.1f} pips",
            f"Pip Value: ${self.pip_value:.2f} per lot",
            f"Risk per Lot: ${self.risk_per_lot():,.2f}",
            f"-" * 40,
            f"Position Size: {self.lot_size:.2f} lots",
        ]

        if self.estimated_margin is not None:
            lines.append(f"Estimated Margin: ${self.estimated_margin:,.2f}")
        if self.remaining_margin is not None:
            lines.append(f"Remaining Margin: ${self.remaining_margin:,.2f}")

        if self.calculation_steps:
            lines.append(f"-" * 40)
            lines.append("Calculation Steps:")
            for i, step in enumerate(self.calculation_steps, 1):
                lines.append(f"  {i}. {step}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert result to dictionary for serialization.

        Returns:
            Dictionary representation of the result
        """
        return {
            "account_balance": self.account_balance,
            "risk_amount": self.risk_amount,
            "risk_percentage": self.risk_percentage(),
            "stop_loss_pips": self.stop_loss_pips,
            "pip_value": self.pip_value,
            "lot_size": self.lot_size,
            "estimated_margin": self.estimated_margin,
            "remaining_margin": self.remaining_margin,
            "calculation_steps": self.calculation_steps,
        }

    def is_zero_position(self) -> bool:
        """
        Check if result recommends zero position size.

        Returns:
            True if lot_size == 0
        """
        return self.lot_size == 0.0

    def margin_utilization_pct(self) -> Optional[float]:
        """
        Calculate margin utilization as percentage of account.

        Returns:
            Margin utilization percentage (0 to 1), or None if margin not calculated
        """
        if self.estimated_margin is None or self.account_balance == 0:
            return None
        return self.estimated_margin / self.account_balance

    def validate_margin_sufficiency(self) -> bool:
        """
        Check if remaining margin is sufficient after position.

        Returns:
            True if remaining_margin > 0, False if insufficient
        """
        if self.remaining_margin is None:
            # Cannot validate if margin not calculated
            return True
        return self.remaining_margin > 0

    def model_dump_json_safe(self) -> str:
        """
        Serialize model to JSON-safe dict.

        Returns:
            JSON string representation of the model.
        """
        return self.model_dump_json(exclude_none=True)
