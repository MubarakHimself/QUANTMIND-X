"""
The Governor (Compliance Layer)
Responsible for calculating Risk Scalars based on Global Constraints.
"""

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RiskMandate:
    """The Risk Authorization Ticket issued by the Governor."""
    allocation_scalar: float = 1.0  # 0.0 to 1.0
    risk_mode: str = "STANDARD"     # STANDARD, CLAMPED, HALTED
    notes: Optional[str] = None

class Governor:
    """
    The Base Governor enforcing Tier 2 Risk Rules (Portfolio & Swarm).
    """
    def __init__(self):
        self.max_portfolio_risk = 0.20  # 20% Hard Cap
        self.correlation_threshold = 0.80

    def calculate_risk(self, regime_report: 'RegimeReport', trade_proposal: dict) -> RiskMandate:
        """
        Calculates the Tier 2 Risk Scalar based on Market Physics.
        
        Args:
            regime_report: Current market state from Sentinel
            trade_proposal: Dict containing 'bot_id', 'symbol', etc.
            
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
