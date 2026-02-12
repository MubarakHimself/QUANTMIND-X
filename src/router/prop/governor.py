"""
The Prop Governor (Extension)
Inherits from Base Governor and adds Prop Firm "Survival Logic".
"""

from src.router.governor import Governor, RiskMandate
from src.router.prop.state import PropState
from src.database.engine import get_session
from src.database.models import PropFirmAccount, RiskTierTransition
from datetime import datetime, timezone

class PropGovernor(Governor):
    """
    Extensions:
    1. Quadratic Throttle (Distance to Ruin).
    2. News Guard Override (Hard Stop).
    3. V8: Tiered Risk Engine - Tier Transition Detection and Logging.
    """
    def __init__(self, account_id: str):
        super().__init__()  # Init Base (VaR, Swarm)
        self.prop_state = PropState(account_id)
        self.account_id = account_id
        
        # Prop Rules (Configurable)
        self.daily_loss_limit_pct = 0.05   # 5% Max Daily Loss
        self.hard_stop_buffer = 0.01       # 1% Safety Buffer (Stop at 4%)
        self.effective_limit = self.daily_loss_limit_pct - self.hard_stop_buffer # 0.04
        
        # V8: Tiered Risk Engine - Tier Thresholds
        self.growth_tier_ceiling = 1000.0   # $1,000
        self.scaling_tier_ceiling = 5000.0  # $5,000
        
        # Track current tier for transition detection
        self._current_tier = None
        self._load_current_tier()

    def calculate_risk(self, regime_report: 'RegimeReport', trade_proposal: dict) -> RiskMandate:
        """
        Overrides Base Governor to apply Tier 3 (Prop) Throttle.
        V8: Also checks for tier transitions and applies tier-aware throttling.
        """
        # 1. INHERIT: Get Base Risk (Tier 1 & 2)
        mandate = super().calculate_risk(regime_report, trade_proposal)
        
        # 2. V8: Check for tier transitions
        current_balance = trade_proposal.get('current_balance', 100000)
        self._check_and_log_tier_transition(current_balance)
        
        # 3. V8: Apply tier-aware throttling
        throttle = self._get_tier_aware_throttle(current_balance)
        
        # 4. EXTEND: News Guard Check (Physics-Aware Kelly already handles this partly, 
        # but Prop requires Hard Stop)
        if regime_report.news_state == "KILL_ZONE":
             mandate.allocation_scalar = 0.0
             mandate.risk_mode = "HALTED_NEWS"
             mandate.notes = "Hard Stop: News Event (Prop Rules)"
             return mandate

        # 5. COMBINE
        mandate.allocation_scalar *= throttle
        
        if throttle < 1.0:
            mandate.risk_mode = "THROTTLED"
            mandate.notes = f"{(mandate.notes or '')} Tier {self._current_tier.upper()} Throttle: {throttle:.2f}x"
            
        return mandate

    def _get_tier_aware_throttle(self, current_balance: float) -> float:
        """
        V8: Calculate tier-aware throttle based on current risk tier.
        
        Growth Tier: No throttle (fixed $5 risk handled in KellySizer)
        Scaling Tier: Kelly Criterion only (no quadratic throttle)
        Guardian Tier: Kelly + Quadratic Throttle (full protection)
        
        Args:
            current_balance: Current account balance
            
        Returns:
            Throttle multiplier (0.0 to 1.0)
        """
        if self._current_tier == 'growth':
            # Growth tier: No throttle (fixed risk)
            return 1.0
        elif self._current_tier == 'scaling':
            # Scaling tier: Kelly only (no quadratic throttle)
            return 1.0
        else:  # guardian
            # Guardian tier: Apply quadratic throttle
            return self._get_quadratic_throttle(current_balance)
    
    def _get_quadratic_throttle(self, current_balance: float) -> float:
        """
        Calculates the Survival Curve (Guardian Tier only).
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
    
    def _load_current_tier(self):
        """
        V8: Load the current risk tier from the database.
        """
        try:
            session = get_session()
            account = session.query(PropFirmAccount).filter_by(
                account_id=self.account_id
            ).first()
            
            if account:
                self._current_tier = account.risk_mode
            else:
                # Default to growth tier if account not found
                self._current_tier = 'growth'
                
            session.close()
        except Exception as e:
            print(f"[PropGovernor] Error loading current tier: {e}")
            self._current_tier = 'growth'
    
    def _determine_risk_tier(self, equity: float) -> str:
        """
        V8: Determine risk tier based on account equity.
        
        Args:
            equity: Current account equity
            
        Returns:
            Risk tier string: 'growth', 'scaling', or 'guardian'
        """
        if equity < self.growth_tier_ceiling:
            return 'growth'
        elif equity < self.scaling_tier_ceiling:
            return 'scaling'
        else:
            return 'guardian'
    
    def _check_and_log_tier_transition(self, equity: float):
        """
        V8: Check if risk tier has changed and log the transition.
        
        Args:
            equity: Current account equity
        """
        new_tier = self._determine_risk_tier(equity)
        
        # Check if tier has changed
        if new_tier != self._current_tier:
            self._log_tier_transition(self._current_tier, new_tier, equity)
            self._update_account_risk_mode(new_tier)
            self._current_tier = new_tier
    
    def _log_tier_transition(self, from_tier: str, to_tier: str, equity: float):
        """
        V8: Log a risk tier transition to the database.
        
        Args:
            from_tier: Previous risk tier
            to_tier: New risk tier
            equity: Account equity at transition
        """
        try:
            session = get_session()
            
            # Get account database ID
            account = session.query(PropFirmAccount).filter_by(
                account_id=self.account_id
            ).first()
            
            if not account:
                print(f"[PropGovernor] Warning: Account {self.account_id} not found in database")
                session.close()
                return
            
            # Create transition record
            transition = RiskTierTransition(
                account_id=account.id,
                from_tier=from_tier,
                to_tier=to_tier,
                equity_at_transition=equity,
                transition_timestamp=datetime.now(timezone.utc)
            )
            
            session.add(transition)
            session.commit()
            
            print(f"[PropGovernor] Tier transition logged: {from_tier} -> {to_tier} at equity=${equity:.2f}")
            
            session.close()
        except Exception as e:
            print(f"[PropGovernor] Error logging tier transition: {e}")
            try:
                session.rollback()
                session.close()
            except:
                pass
    
    def _update_account_risk_mode(self, new_tier: str):
        """
        V8: Update the account's risk_mode in the database.
        
        Args:
            new_tier: New risk tier to set
        """
        try:
            session = get_session()
            
            account = session.query(PropFirmAccount).filter_by(
                account_id=self.account_id
            ).first()
            
            if account:
                account.risk_mode = new_tier
                session.commit()
                print(f"[PropGovernor] Account risk_mode updated to: {new_tier}")
            
            session.close()
        except Exception as e:
            print(f"[PropGovernor] Error updating account risk_mode: {e}")
            try:
                session.rollback()
                session.close()
            except:
                pass
