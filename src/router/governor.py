"""
The Governor (Compliance Layer)
Responsible for calculating Risk Scalars based on Global Constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List

@dataclass
class RiskMandate:
    """The Risk Authorization Ticket issued by Governor.

    Extended with Kelly Calculator position sizing fields for fee-aware trading.

    Attributes:
        allocation_scalar: Risk multiplier (0.0 to 1.0)
        risk_mode: Trading mode (STANDARD, CLAMPED, HALTED)
        notes: Optional notes about the risk calculation
        position_size: Calculated position size in lots
        kelly_fraction: Kelly fraction used for position sizing
        risk_amount: Dollar amount at risk
        kelly_adjustments: List of adjustment descriptions for audit trail
    """
    allocation_scalar: float = 1.0  # 0.0 to 1.0
    risk_mode: str = "STANDARD"     # STANDARD, CLAMPED, HALTED
    notes: Optional[str] = None
    # NEW FIELDS for fee-aware Kelly integration
    position_size: float = 0.0
    kelly_fraction: float = 0.0
    risk_amount: float = 0.0
    kelly_adjustments: List[str] = field(default_factory=list)


class Governor:
    """
    The Base Governor enforcing Tier 2 Risk Rules (Portfolio & Swarm).
    """
    def __init__(self):
        self.max_portfolio_risk = 0.20  # 20% Hard Cap
        self.correlation_threshold = 0.80

    def calculate_risk(self, regime_report: 'RegimeReport', trade_proposal: dict, account_balance: Optional[float] = None, broker_id: Optional[str] = None, **kwargs) -> RiskMandate:
        """
        Calculates Tier 2 Risk Scalar based on Market Physics.

        Args:
            regime_report: Current market state from Sentinel
            trade_proposal: Dict containing 'bot_id', 'symbol', etc.
            account_balance: Optional account balance (ignored in base Governor, used by EnhancedGovernor)
            broker_id: Optional broker ID (ignored in base Governor, used by EnhancedGovernor)
            **kwargs: Additional optional parameters for subclass extensions

        Returns:
            RiskMandate object with authorized scalar.
        """
        mandate = RiskMandate()

        # 1. Physics-Based Throttling
        # If Chaos is High, we clamp the entire swarm.
        if regime_report.chaos_score > 0.6:
            mandate.allocation_scalar = 0.2  # Intense clamping
            mandate.risk_mode = "CLAMPED"
            mandate.notes = f"Extreme Chaos ({regime_report.chaos_score:.2f}) - Tier 2 Emergency Clamp"
        elif regime_report.chaos_score > 0.3:
            mandate.allocation_scalar = 0.7  # Moderate clamp
            mandate.risk_mode = "CLAMPED"
            mandate.notes = "Moderate Chaos - Smoothing activated."

        # 2. Systemic Correlation (Stub for RMT Eigenvalue)
        if regime_report.is_systemic_risk:
            mandate.allocation_scalar = min(mandate.allocation_scalar, 0.4)
            mandate.risk_mode = "CLAMPED"
            mandate.notes = (mandate.notes or "") + " Systemic Swarm Coupling detected."

        return mandate

    def check_swarm_cohesion(self, correlation_matrix: Dict) -> float:
        """
        Helper to calculate RMT Max Eigenvalue (if not provided by Sentinel).
        placeholder for complex logic.
        """
        return 0.5
