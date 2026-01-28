"""
Market Physics Model

Pydantic model for market regime indicators derived from econophysics:
- Lyapunov exponents for chaos detection
- Ising model for market magnetization and susceptibility
- Random Matrix Theory for noise threshold detection

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class RiskLevel(str, Enum):
    """Risk classification levels based on physics indicators."""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class MarketPhysics(BaseModel):
    """
    Market regime physics indicators for dynamic risk adjustment.

    Combines three econophysics frameworks:
    1. Lyapunov Exponent: Measures chaos/sensitivity to initial conditions
    2. Ising Model: Market magnetization (trend strength) and susceptibility (volatility)
    3. Random Matrix Theory: Maximum eigenvalue for detecting signal vs noise

    Attributes:
        lyapunov_exponent: Chaos indicator (positive = chaotic, negative = stable)
        ising_susceptibility: Market susceptibility (volatility clustering measure)
        ising_magnetization: Market magnetization (-1 to 1, trend strength/direction)
        rmt_max_eigenvalue: Maximum eigenvalue of correlation matrix
        rmt_noise_threshold: Theoretical maximum for random matrix (typically sqrt(2))
        calculated_at: Timestamp when physics were calculated
        is_stale: Whether the physics data is outdated
    """

    # Lyapunov Exponent (Chaos Theory)
    lyapunov_exponent: float = Field(
        description="Lyapunov exponent (positive = chaotic regime, negative = stable)",
        examples=[0.5, -0.2, 1.2]
    )

    # Ising Model (Statistical Mechanics)
    ising_susceptibility: float = Field(
        ge=0.0,
        description="Market susceptibility (volatility clustering, higher = more fragile)",
        examples=[0.5, 1.0, 2.5]
    )

    ising_magnetization: float = Field(
        ge=-1.0,
        le=1.0,
        description="Market magnetization (-1 bearish to 1 bullish, 0 = random)",
        examples=[0.5, -0.3, 0.0]
    )

    # Random Matrix Theory (RMT)
    rmt_max_eigenvalue: Optional[float] = Field(
        None,
        ge=0.0,
        description="Maximum eigenvalue of correlation matrix",
        examples=[3.5, 5.0, 2.8]
    )

    rmt_noise_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        description="Theoretical maximum for random matrix (typically ~2.0 for N×T)",
        examples=[2.0, 2.1, 1.9]
    )

    # Metadata
    calculated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when physics indicators were calculated"
    )

    is_stale: bool = Field(
        default=False,
        description="Whether the physics data is outdated and should be recalculated"
    )

    @field_validator("lyapunov_exponent")
    @classmethod
    def validate_lyapunov_range(cls, v: float) -> float:
        """Warn if Lyapunov exponent is extremely high (>3 indicates extreme chaos)."""
        if abs(v) > 5.0:
            # Extreme values may indicate calculation errors
            pass
        return v

    @field_validator("ising_susceptibility")
    @classmethod
    def validate_susceptibility(cls, v: float) -> float:
        """Warn if susceptibility is extremely high (>5 indicates critical regime)."""
        if v > 10.0:
            # Very high susceptibility may indicate phase transition
            pass
        return v

    def is_fresh(self, max_age_seconds: int = 3600) -> bool:
        """
        Check if physics data is fresh (not stale).

        Args:
            max_age_seconds: Maximum age in seconds before data is considered stale

        Returns:
            True if data is fresh, False if stale
        """
        age_seconds = (datetime.utcnow() - self.calculated_at).total_seconds()
        return age_seconds <= max_age_seconds

    def mark_stale(self) -> None:
        """Mark the physics data as stale."""
        self.is_stale = True

    def mark_fresh(self) -> None:
        """Mark the physics data as fresh."""
        self.is_stale = False

    def risk_level(self) -> RiskLevel:
        """
        Calculate overall risk level based on physics indicators.

        Risk determination logic:
        - HIGH: Positive Lyapunov (chaotic) OR high susceptibility (>2.0)
        - MODERATE: Negative Lyapunov but susceptibility > 1.0
        - LOW: Negative Lyapunov AND low susceptibility (<1.0)
        - EXTREME: Lyapunov > 2.0 OR susceptibility > 5.0

        Returns:
            RiskLevel enum value
        """
        # Check for extreme conditions first
        if self.lyapunov_exponent > 2.0 or self.ising_susceptibility > 5.0:
            return RiskLevel.EXTREME

        # Check for high risk
        if self.lyapunov_exponent > 0 or self.ising_susceptibility > 2.0:
            return RiskLevel.HIGH

        # Check for moderate risk
        if self.ising_susceptibility > 1.0:
            return RiskLevel.MODERATE

        # Default to low risk
        return RiskLevel.LOW

    def physics_multiplier(self) -> float:
        """
        Calculate position size multiplier based on physics risk level.

        Returns:
            Multiplier to apply to base position size:
            - EXTREME: 0.25 (75% reduction)
            - HIGH: 0.5 (50% reduction)
            - MODERATE: 0.75 (25% reduction)
            - LOW: 1.0 (no reduction)
        """
        risk = self.risk_level()
        multipliers = {
            RiskLevel.EXTREME: 0.25,
            RiskLevel.HIGH: 0.5,
            RiskLevel.MODERATE: 0.75,
            RiskLevel.LOW: 1.0,
        }
        return multipliers.get(risk, 1.0)

    def has_signal(self) -> bool:
        """
        Check if RMT indicates a true signal (not noise).

        Signal exists if max_eigenvalue > noise_threshold.

        Returns:
            True if signal detected, False if purely noise
        """
        if self.rmt_max_eigenvalue is None or self.rmt_noise_threshold is None:
            # Default to assuming signal if RMT data not available
            return True

        return self.rmt_max_eigenvalue > self.rmt_noise_threshold

    def trend_direction(self) -> str:
        """
        Determine trend direction from magnetization.

        Returns:
            "bullish" if magnetization > 0.2
            "bearish" if magnetization < -0.2
            "neutral" if between -0.2 and 0.2
        """
        if self.ising_magnetization > 0.2:
            return "bullish"
        elif self.ising_magnetization < -0.2:
            return "bearish"
        else:
            return "neutral"

    def is_chaotic(self) -> bool:
        """
        Check if market is in chaotic regime.

        Returns:
            True if Lyapunov exponent > 0 (chaotic), False if <= 0 (stable)
        """
        return self.lyapunov_exponent > 0

    def regime_description(self) -> str:
        """
        Get human-readable description of current market regime.

        Returns:
            Descriptive string of market state
        """
        risk = self.risk_level()
        trend = self.trend_direction()
        chaos = "chaotic" if self.is_chaotic() else "stable"

        if risk == RiskLevel.EXTREME:
            return f"EXTREME risk: {chaos} regime, {trend} trend, high fragility"
        elif risk == RiskLevel.HIGH:
            return f"HIGH risk: {chaos} regime, {trend} trend"
        elif risk == RiskLevel.MODERATE:
            return f"MODERATE risk: {chaos} regime, {trend} trend, moderate volatility"
        else:
            return f"LOW risk: {chaos} regime, {trend} trend"

    def model_dump_json_safe(self) -> str:
        """
        Serialize model to JSON-safe dict.

        Returns:
            JSON string representation of the model.
        """
        return self.model_dump_json(exclude_none=True)
