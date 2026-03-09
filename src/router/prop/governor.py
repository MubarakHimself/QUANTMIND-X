"""
The Prop Governor (Extension)
Inherits from Base Governor and adds Prop Firm "Survival Logic".
"""

from typing import Optional, Dict, Any, TYPE_CHECKING
from src.router.governor import Governor, RiskMandate
from src.router.prop.state import PropState
from src.database.engine import get_session
from src.database.models import PropFirmAccount, RiskTierTransition
from datetime import datetime, timezone

if TYPE_CHECKING:
    from src.api.settings_endpoints import RiskSettings

class PropGovernor(Governor):
    """
    Extensions:
    1. Quadratic Throttle (Distance to Ruin).
    2. News Guard Override (Hard Stop).
    3. V8: Tiered Risk Engine - Tier Transition Detection and Logging.
    """
    def __init__(self, account_id: str, settings: Optional['RiskSettings'] = None):
        super().__init__()  # Init Base (VaR, Swarm)
        self.prop_state = PropState(account_id)
        self.account_id = account_id

        # Prop Rules (Configurable via settings)
        # daily_loss_limit_pct: settings.dailyLossLimit is in percent (e.g., 5.0), convert to decimal (0.05)
        self.daily_loss_limit_pct = settings.dailyLossLimit / 100 if settings else 0.05
        self.hard_stop_buffer = settings.hardStopBuffer if settings else 0.01
        self.effective_limit = self.daily_loss_limit_pct - self.hard_stop_buffer

        # V8: Tiered Risk Engine - Tier Thresholds
        # Use settings.balanceZones if available, otherwise fall back to defaults
        if settings and settings.balanceZones:
            self.growth_tier_ceiling = settings.balanceZones.get("growth", 1000.0)
            self.scaling_tier_ceiling = settings.balanceZones.get("scaling", 5000.0)
        else:
            self.growth_tier_ceiling = 1000.0   # $1,000
            self.scaling_tier_ceiling = 5000.0  # $5,000

        # Max risk per trade (from settings if available)
        self.max_risk_per_trade = settings.maxRiskPerTrade if settings else None
        
        # Track current tier for transition detection
        self._current_tier = None
        self._load_current_tier()

    def calculate_risk(
        self,
        regime_report: 'RegimeReport',
        trade_proposal: dict,
        account_balance: Optional[float] = None,
        broker_id: Optional[str] = None,
        account_id: Optional[str] = None,
        mode: str = "live",
        **kwargs: Any
    ) -> RiskMandate:
        """
        Overrides Base Governor to apply Tier 3 (Prop) Throttle.
        V8: Also checks for tier transitions and applies tier-aware throttling.

        Signature matches EnhancedGovernor.calculate_risk() for full compatibility
        with Commander.run_auction().

        Args:
            regime_report: Current market regime from Sentinel
            trade_proposal: Trade proposal dict with symbol, balance, etc.
            account_balance: Current account balance for position sizing
            broker_id: Broker identifier (unused by PropGovernor, forwarded to base)
            account_id: Account identifier override (uses self.account_id if not set)
            mode: Trading mode ('live' or 'demo') — demo uses reduced risk
        """
        # Resolve current balance from args or trade_proposal
        current_balance = (
            account_balance
            or trade_proposal.get('account_balance')
            or trade_proposal.get('current_balance', 100000)
        )
        # Patch trade_proposal so base Governor also sees current_balance
        if current_balance and 'current_balance' not in trade_proposal:
            trade_proposal = dict(trade_proposal)  # copy, don't mutate caller's dict
            trade_proposal['current_balance'] = current_balance

        # 1. INHERIT: Get Base Risk (Tier 1 & 2)
        mandate = super().calculate_risk(regime_report, trade_proposal)

        # 1b. Apply max risk per trade limit if configured
        if self.max_risk_per_trade is not None and mandate.risk_pct > self.max_risk_per_trade:
            mandate.risk_pct = self.max_risk_per_trade
            mandate.notes = f"{(mandate.notes or '')} Capped at max_risk_per_trade: {self.max_risk_per_trade:.2%}"

        # 2. V8: Check for tier transitions
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

        # 5. Demo mode: apply conservative multiplier (50% of normal risk)
        if mode == "demo":
            throttle *= 0.5

        # 6. COMBINE
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
