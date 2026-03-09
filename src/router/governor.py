"""
The Governor (Compliance Layer)
Responsible for calculating Risk Scalars based on Global Constraints.

Updated with mode awareness for demo/live trading distinction.
Updated with EnhancedKellyCalculator integration for fee-aware position sizing.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from src.database.models import TradingMode

logger = logging.getLogger(__name__)


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
        mode: Trading mode (demo or live) for audit trail
    """
    allocation_scalar: float = 1.0  # 0.0 to 1.0
    risk_mode: str = "STANDARD"     # STANDARD, CLAMPED, HALTED
    notes: Optional[str] = None
    # NEW FIELDS for fee-aware Kelly integration
    position_size: float = 0.0
    kelly_fraction: float = 0.0
    risk_amount: float = 0.0
    kelly_adjustments: List[str] = field(default_factory=list)
    # Mode field for demo/live distinction
    mode: str = "live"


class Governor:
    """
    The Base Governor enforcing Tier 2 Risk Rules (Portfolio & Swarm).
    
    Updated to integrate EnhancedKellyCalculator for fee-aware position sizing.
    The Governor now calculates position sizes using Kelly Criterion with:
    - Broker-specific pip values, commissions, and spreads
    - Regime quality from Sentinel
    - Fee kill switch (blocks trades when fees exceed expected profit)
    """
    
    def __init__(self, settings: Optional[object] = None):
        self._settings = settings
        self.max_portfolio_risk = settings.maxPortfolioRisk if settings else 0.20  # 20% Hard Cap
        self.correlation_threshold = settings.correlationThreshold if settings else 0.80
        
        # Initialize EnhancedKellyCalculator for fee-aware position sizing
        try:
            from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator
            self._kelly_calculator = EnhancedKellyCalculator()
            self._kelly_available = True
            logger.info("Governor: EnhancedKellyCalculator initialized for fee-aware position sizing")
        except ImportError as e:
            self._kelly_calculator = None
            self._kelly_available = False
            logger.warning(f"Governor: EnhancedKellyCalculator not available: {e}")

    def calculate_risk(
        self,
        regime_report: 'RegimeReport',
        trade_proposal: dict,
        account_balance: Optional[float] = None,
        broker_id: Optional[str] = None,
        account_id: Optional[str] = None,
        mode: str = "live",
        **kwargs
    ) -> RiskMandate:
        """
        Calculates Tier 2 Risk Scalar based on Market Physics with Kelly position sizing.

        Args:
            regime_report: Current market state from Sentinel
            trade_proposal: Dict containing 'symbol', 'broker_id', 'stop_loss_pips', 
                           'win_rate', 'avg_win', 'avg_loss', 'current_atr', 'average_atr'
            account_balance: Account balance for Kelly calculation
            broker_id: Broker identifier for fee-aware Kelly (e.g., 'icmarkets_raw')
            account_id: Optional account ID for account-specific limits
            mode: Trading mode ('demo' or 'live') - affects risk limits
            **kwargs: Additional optional parameters for subclass extensions

        Returns:
            RiskMandate object with authorized scalar and Kelly-based position sizing.
        """
        mandate = RiskMandate()
        mandate.mode = mode

        # Mode-specific risk adjustments
        # Demo mode: More aggressive risk limits (3% vs 2% for live)
        demo_risk_multiplier = 1.5 if mode == "demo" else 1.0
        # Allow settings to override demo risk multiplier
        if hasattr(self, '_settings') and self._settings:
            demo_risk_multiplier = self._settings.demoRiskMultiplier if hasattr(self._settings, 'demoRiskMultiplier') else demo_risk_multiplier

        # 1. Physics-Based Throttling
        # If Chaos is High, we clamp the entire swarm.
        if regime_report.chaos_score > 0.6:
            mandate.allocation_scalar = 0.2 * demo_risk_multiplier  # Intense clamping
            mandate.risk_mode = "CLAMPED"
            mandate.notes = f"Extreme Chaos ({regime_report.chaos_score:.2f}) - Tier 2 Emergency clamp"
        elif regime_report.chaos_score > 0.3:
            mandate.allocation_scalar = 0.7 * demo_risk_multiplier  # Moderate clamp
            mandate.risk_mode = "CLAMPED"
            mandate.notes = "Moderate Chaos - Smoothing activated."

        # 2. Systemic Correlation (Stub for RMT Eigenvalue)
        if regime_report.is_systemic_risk:
            mandate.allocation_scalar = min(mandate.allocation_scalar, 0.4)
            mandate.risk_mode = "CLAMPED"
            mandate.notes = (mandate.notes or "") + " Systemic Swarm Coupling detected."

        # Add mode tag to notes
        mode_tag = "[DEMO]" if mode == "demo" else "[LIVE]"
        if mandate.notes:
            mandate.notes = f"{mode_tag} {mandate.notes}"
        else:
            mandate.notes = f"{mode_tag} Standard risk calculation"

        # 3. EnhancedKelly Position Sizing (Comment 2 fix)
        # Calculate position size using EnhancedKellyCalculator with broker fees
        if self._kelly_available and account_balance:
            mandate = self._calculate_kelly_position_size(
                regime_report=regime_report,
                trade_proposal=trade_proposal,
                account_balance=account_balance,
                broker_id=broker_id or trade_proposal.get('broker_id', 'mt5_default'),
                mandate=mandate,
                mode=mode
            )

        return mandate
    
    def _calculate_kelly_position_size(
        self,
        regime_report: 'RegimeReport',
        trade_proposal: dict,
        account_balance: float,
        broker_id: str,
        mandate: RiskMandate,
        mode: str = "live"
    ) -> RiskMandate:
        """
        Calculate position size using EnhancedKellyCalculator with broker fees.
        
        Populates mandate fields:
        - position_size: Final position size in lots
        - kelly_fraction: Kelly fraction used
        - risk_amount: Dollar amount at risk
        - kelly_adjustments: List of adjustments for audit trail
        
        Args:
            regime_report: Current market regime from Sentinel
            trade_proposal: Trade proposal dict with symbol, stop_loss_pips, etc.
            account_balance: Account balance for Kelly calculation
            broker_id: Broker identifier for fee lookup
            mandate: Current RiskMandate to update
            mode: Trading mode ('demo' or 'live')
            
        Returns:
            Updated RiskMandate with Kelly-based position sizing
        """
        try:
            # Extract parameters from trade_proposal with defaults
            symbol = trade_proposal.get('symbol', 'EURUSD')
            win_rate = trade_proposal.get('win_rate', 0.55)
            avg_win = trade_proposal.get('avg_win', 400.0)
            avg_loss = trade_proposal.get('avg_loss', 200.0)
            stop_loss_pips = trade_proposal.get('stop_loss_pips', 20.0)
            current_atr = trade_proposal.get('current_atr', 0.0012)
            average_atr = trade_proposal.get('average_atr', 0.0010)
            pip_value = trade_proposal.get('pip_value', 10.0)
            commission_per_lot = trade_proposal.get('commission_per_lot', 0.0)
            spread_pips = trade_proposal.get('spread_pips', 0.0)
            regime_quality = getattr(regime_report, 'regime_quality', 1.0)
            
            # Call EnhancedKellyCalculator
            kelly_result = self._kelly_calculator.calculate(
                account_balance=account_balance,
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                current_atr=current_atr,
                average_atr=average_atr,
                stop_loss_pips=stop_loss_pips,
                pip_value=pip_value,
                regime_quality=regime_quality,
                broker_id=broker_id,
                symbol=symbol,
                commission_per_lot=commission_per_lot,
                spread_pips=spread_pips
            )
            
            # Check for fee kill switch
            if kelly_result.status == 'fee_blocked':
                mandate.risk_mode = "HALTED"
                mandate.position_size = 0.0
                mandate.kelly_fraction = 0.0
                mandate.risk_amount = 0.0
                mandate.kelly_adjustments = kelly_result.adjustments_applied
                mandate.notes = (mandate.notes or "") + " Fee kill switch activated - fees exceed expected profit"
                logger.warning(f"Governor: Fee kill switch for {symbol}@{broker_id} - fees exceed avg_win")
                return mandate
            
            # Apply physics scalar to position size
            physics_scalar = mandate.allocation_scalar
            final_position_size = kelly_result.position_size * physics_scalar
            final_risk_amount = kelly_result.risk_amount * physics_scalar
            final_kelly_fraction = kelly_result.kelly_f * physics_scalar
            
            # Update mandate with Kelly results
            mandate.position_size = final_position_size
            mandate.kelly_fraction = final_kelly_fraction
            mandate.risk_amount = final_risk_amount
            mandate.kelly_adjustments = kelly_result.adjustments_applied
            
            logger.info(
                f"Governor: Kelly sizing for {symbol}@{broker_id}: "
                f"position={final_position_size:.4f} lots, kelly_f={final_kelly_fraction:.4f}, "
                f"risk=${final_risk_amount:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Governor: Kelly calculation failed: {e}")
            # Keep mandate.position_size = 0.0 on error (no fallback)
            mandate.kelly_adjustments = [f"Kelly calculation error: {str(e)}"]
        
        return mandate

    def check_swarm_cohesion(self, correlation_matrix: Dict) -> float:
        """
        Helper to calculate RMT Max Eigenvalue (if not provided by Sentinel).
        placeholder for complex logic.
        """
        return 0.5
