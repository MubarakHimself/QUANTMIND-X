"""
Sizing Recommendation Model

Pydantic model for position sizing recommendations with constraint tracking.
Records raw Kelly calculation, physics adjustments, and final position size.

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class SizingRecommendation(BaseModel):
    """
    Position sizing recommendation with full constraint tracking.

    Tracks the journey from raw Kelly calculation through all adjustments
    (physics-based, risk caps, margin limits) to final position size.

    Attributes:
        raw_kelly: Raw Kelly fraction before any adjustments
        physics_multiplier: Multiplier applied based on market physics (0.25 to 1.0)
        final_risk_pct: Final risk percentage after all adjustments (0 to 1)
        position_size_lots: Recommended position size in lots
        constraint_source: Source of the constraint (e.g., "risk_cap", "margin", "physics")
        validation_passed: Whether all validation checks passed
        adjustments_applied: List of adjustment descriptions in order applied
    """

    # Raw Calculation
    raw_kelly: float = Field(
        ge=0.0,
        le=1.0,
        description="Raw Kelly fraction before any adjustments",
        examples=[0.15, 0.25, 0.08]
    )

    # Physics-Based Adjustment
    physics_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Multiplier based on market physics (1.0 = no adjustment)",
        examples=[1.0, 0.75, 0.5, 0.25]
    )

    # Final Values
    final_risk_pct: float = Field(
        ge=0.0,
        le=0.10,  # Max 10% risk (conservative for prop trading)
        description="Final risk percentage after all adjustments",
        examples=[0.02, 0.015, 0.01]
    )

    position_size_lots: float = Field(
        ge=0.0,
        description="Recommended position size in lots",
        examples=[0.15, 1.25, 0.05]
    )

    # Constraint Tracking
    constraint_source: Optional[str] = Field(
        None,
        description="Source of active constraint (e.g., 'risk_cap', 'margin', 'physics')",
        examples=["risk_cap", "margin_limit", "physics_high", "none"]
    )

    validation_passed: bool = Field(
        default=True,
        description="Whether all validation checks passed"
    )

    adjustments_applied: List[str] = Field(
        default_factory=list,
        description="Chronological list of adjustments applied to position size"
    )

    @field_validator("final_risk_pct")
    @classmethod
    def validate_final_risk(cls, v: float) -> float:
        """Ensure final risk is within conservative bounds."""
        if v > 0.10:
            raise ValueError(f"final_risk_pct ({v}) exceeds conservative maximum of 10%")
        if v < 0:
            raise ValueError(f"final_risk_pct ({v}) cannot be negative")
        return v

    @field_validator("position_size_lots")
    @classmethod
    def validate_position_size(cls, v: float) -> float:
        """Ensure position size is positive or zero."""
        if v < 0:
            raise ValueError(f"position_size_lots ({v}) cannot be negative")
        return v

    def apply_penalty(
        self,
        penalty_type: str,
        multiplier: float,
        description: Optional[str] = None
    ) -> None:
        """
        Apply a penalty multiplier to the position size.

        Args:
            penalty_type: Type of penalty (e.g., "physics_high", "margin_limit")
            multiplier: Multiplier to apply (0 to 1)
            description: Optional description of the penalty
        """
        if not 0 <= multiplier <= 1:
            raise ValueError(f"Penalty multiplier must be between 0 and 1, got {multiplier}")

        # Update position size
        self.position_size_lots = self.position_size_lots * multiplier

        # Update risk percentage proportionally
        self.final_risk_pct = self.final_risk_pct * multiplier

        # Update physics multiplier if this is a physics penalty
        if penalty_type.startswith("physics"):
            self.physics_multiplier = self.physics_multiplier * multiplier

        # Track constraint source
        if self.constraint_source is None:
            self.constraint_source = penalty_type

        # Add to adjustments list
        desc = description or f"Applied {penalty_type} penalty (×{multiplier:.2f})"
        self.adjustments_applied.append(desc)

    def is_constrained(self) -> bool:
        """
        Check if position size was constrained from raw Kelly.

        Returns:
            True if final_risk_pct < raw_kelly (i.e., constraints were applied)
        """
        # Adjust raw_kelly by physics_multiplier for comparison
        adjusted_raw = self.raw_kelly * self.physics_multiplier
        return self.final_risk_pct < adjusted_raw

    def constraint_reason(self) -> str:
        """
        Get the reason for position size constraint.

        Returns:
            String describing why position was constrained, or "none" if unconstrained
        """
        if not self.is_constrained():
            return "none"
        return self.constraint_source or "unknown"

    def risk_reduction_pct(self) -> float:
        """
        Calculate percentage reduction from raw Kelly.

        Returns:
            Percentage reduction (0 to 1, where 0.5 = 50% reduction)
        """
        adjusted_raw = self.raw_kelly * self.physics_multiplier
        if adjusted_raw == 0:
            return 0.0
        reduction = (adjusted_raw - self.final_risk_pct) / adjusted_raw
        return max(0.0, min(1.0, reduction))

    def is_zero_position(self) -> bool:
        """
        Check if recommendation is for zero position size.

        Returns:
            True if position_size_lots == 0
        """
        return self.position_size_lots == 0.0

    def add_adjustment(self, description: str) -> None:
        """
        Add an adjustment description to the tracking list.

        Args:
            description: Human-readable description of the adjustment
        """
        self.adjustments_applied.append(description)

    def get_adjustments_summary(self) -> str:
        """
        Get a formatted summary of all adjustments applied.

        Returns:
            Multi-line string describing all adjustments
        """
        if not self.adjustments_applied:
            return "No adjustments applied"

        lines = ["Adjustments Applied:"]
        for i, adj in enumerate(self.adjustments_applied, 1):
            lines.append(f"  {i}. {adj}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Convert recommendation to dictionary for serialization.

        Returns:
            Dictionary representation of the recommendation
        """
        return {
            "raw_kelly": self.raw_kelly,
            "physics_multiplier": self.physics_multiplier,
            "final_risk_pct": self.final_risk_pct,
            "position_size_lots": self.position_size_lots,
            "constraint_source": self.constraint_source,
            "validation_passed": self.validation_passed,
            "is_constrained": self.is_constrained(),
            "constraint_reason": self.constraint_reason(),
            "risk_reduction_pct": self.risk_reduction_pct(),
            "adjustments": self.adjustments_applied,
        }

    def model_dump_json_safe(self) -> str:
        """
        Serialize model to JSON-safe dict.

        Returns:
            JSON string representation of the model.
        """
        return self.model_dump_json(exclude_none=True)
