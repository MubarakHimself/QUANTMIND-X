"""
The Prop Governor (Extension)
Inherits from Base Governor and adds Prop Firm "Survival Logic".
"""

from src.router.governor import Governor, RiskMandate
from src.router.prop.state import PropState

class PropGovernor(Governor):
    """
    Extensions:
    1. Quadratic Throttle (Distance to Ruin).
    2. News Guard Override (Hard Stop).
    """
    def __init__(self, account_id: str):
        super().__init__()  # Init Base (VaR, Swarm)
        self.prop_state = PropState(account_id)
        
        # Prop Rules (Configurable)
        self.daily_loss_limit_pct = 0.05   # 5% Max Daily Loss
        self.hard_stop_buffer = 0.01       # 1% Safety Buffer (Stop at 4%)
        self.effective_limit = self.daily_loss_limit_pct - self.hard_stop_buffer # 0.04

    def calculate_risk(self, regime_report: 'RegimeReport', trade_proposal: dict) -> RiskMandate:
        """
        Overrides Base Governor to apply Tier 3 (Prop) Throttle.
        """
        # 1. INHERIT: Get Base Risk (Tier 1 & 2)
        mandate = super().calculate_risk(regime_report, trade_proposal)
        
        # 2. EXTEND: Calculate Prop Throttle
        # We need current_balance from proposal
        current_balance = trade_proposal.get('current_balance', 100000)
        throttle = self._get_quadratic_throttle(current_balance)
        
        # 3. EXTEND: News Guard Check (Physics-Aware Kelly already handles this partly, 
        # but Prop requires Hard Stop)
        if regime_report.news_state == "KILL_ZONE":
             mandate.allocation_scalar = 0.0
             mandate.risk_mode = "HALTED_NEWS"
             mandate.notes = "Hard Stop: News Event (Prop Rules)"
             return mandate

        # 4. COMBINE
        mandate.allocation_scalar *= throttle
        
        if throttle < 1.0:
            mandate.risk_mode = "THROTTLED"
            mandate.notes = f"{(mandate.notes or '')} Prop Throttle Active: {throttle:.2f}x"
            
        return mandate

    def _get_quadratic_throttle(self, current_balance: float) -> float:
        """
        Calculates the Survival Curve.
        Formula: Throttle = 1.0 - (CurrentLoss / EffectiveLimit)^2
        """
        # 1. Get Daily Start Balance (from State)
        # For prototype, assume we have it. In prod, fetch from Redis.
        start_balance = (
            getattr(self.prop_state, 'daily_start_balance', None)
            if hasattr(self.prop_state, 'daily_start_balance') else None
        )
        if start_balance in (None, 0):
            start_balance = current_balance
        
        current_loss = start_balance - current_balance
        
        # If in profit, no throttle
        if current_loss <= 0:
            return 1.0
            
        loss_pct = current_loss / start_balance
        
        # If breach effective limit (4%), Kill Switch
        if loss_pct >= self.effective_limit:
            return 0.0
            
        # Quadratic Curve
        # As loss approaches 4%, risk drops rapidly
        throttle = 1.0 - (loss_pct / self.effective_limit) ** 2
        return max(0.0, throttle)
