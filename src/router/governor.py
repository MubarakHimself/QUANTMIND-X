"""
The Governor (Compliance Layer)
Responsible for calculating Risk Scalars based on Global Constraints.

Updated with mode awareness for demo/live trading distinction.
Updated with EnhancedKellyCalculator integration for fee-aware position sizing.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, TYPE_CHECKING
import logging
from datetime import datetime

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
        kelly_adjustments: Dict of adjustment details for audit trail (includes session_kelly, risk_mode, etc.)
        mode: Trading mode (demo or live) for audit trail
    """
    allocation_scalar: float = 1.0  # 0.0 to 1.0
    risk_mode: str = "STANDARD"     # STANDARD, CLAMPED, HALTED
    notes: Optional[str] = None
    # NEW FIELDS for fee-aware Kelly integration
    position_size: float = 0.0
    kelly_fraction: float = 0.0
    risk_amount: float = 0.0
    kelly_adjustments: Dict[str, Any] = field(default_factory=dict)
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

        # 0. Account-Level Drawdown Check (ProgressiveKillSwitch Tier 3)
        # Must happen BEFORE position sizing — if account is stopped, no sizing
        if account_id:
            try:
                from src.router.account_monitor import AccountMonitor
                account_monitor = AccountMonitor()
                if not account_monitor.is_account_allowed(account_id):
                    status = account_monitor.get_stop_status(account_id)
                    mandate.risk_mode = "HALTED"
                    mandate.allocation_scalar = 0.0
                    mandate.position_size = 0.0
                    mandate.notes = (
                        f"Account drawdown stop — daily={status.get('daily_stop')}, "
                        f"weekly={status.get('weekly_stop')}"
                    )
                    logger.warning(
                        f"Governor: Account {account_id} drawdown stop — halting all trades "
                        f"(daily_loss={status.get('daily_loss_pct', 0):.2%}, "
                        f"weekly_loss={status.get('weekly_loss_pct', 0):.2%})"
                    )
                    return mandate
            except Exception as e:
                logger.warning(f"Governor: Could not check account drawdown: {e}")

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

        # 5% Concurrent Exposure Cap Check (FIX-005)
        # Check BEFORE position sizing — reject if adding this trade exceeds 5% cap
        if account_id and account_balance:
            exposure_check = self._check_exposure_cap(
                account_id=account_id,
                account_balance=account_balance,
                trade_proposal=trade_proposal,
                mandate=mandate
            )
            if exposure_check is not None:
                # Exposure cap exceeded — return HALTED mandate
                logger.warning(f"Governor: Exposure cap exceeded: {exposure_check}")
                return exposure_check

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

    # === FIX-005: 5% Concurrent Exposure Cap ===

    EXPOSURE_CAP = 0.05  # 5% hard cap

    def _check_exposure_cap(
        self,
        account_id: str,
        account_balance: float,
        trade_proposal: dict,
        mandate: RiskMandate
    ) -> Optional[RiskMandate]:
        """
        Check if approving this trade would exceed the 5% concurrent exposure cap.

        Args:
            account_id: Account identifier
            account_balance: Current account balance/equity
            trade_proposal: Trade proposal dict with stop_loss_pips, symbol, etc.
            mandate: Current RiskMandate being built

        Returns:
            None if within cap (safe to proceed), or HALTED RiskMandate if cap exceeded
        """
        try:
            # Get current exposure from all open positions
            total_exposure = self._get_current_exposure(account_id, account_balance)

            # Estimate proposed trade's risk % (using preliminary Kelly estimate or default)
            proposed_risk_pct = self._estimate_trade_risk_pct(
                trade_proposal, mandate, account_balance
            )

            combined_exposure = total_exposure + proposed_risk_pct

            if combined_exposure > self.EXPOSURE_CAP:
                # Cap exceeded — return HALTED mandate
                halted_mandate = RiskMandate(
                    allocation_scalar=0.0,
                    risk_mode="HALTED",
                    position_size=0.0,
                    kelly_fraction=0.0,
                    risk_amount=0.0,
                    kelly_adjustments=[],
                    notes=(
                        f"5% concurrent exposure cap reached — "
                        f"current={total_exposure:.2%}, proposed={proposed_risk_pct:.2%}, "
                        f"combined={combined_exposure:.2%}, cap=5%"
                    ),
                    mode=mandate.mode
                )

                # Log FAIL-R02 for audit trail
                logger.warning(
                    f"Governor: FAIL-R02: Concurrent Exposure Cap Exceeded | "
                    f"account_id={account_id} | "
                    f"open_positions={self._get_open_positions_detail(account_id)} | "
                    f"total_exposure_pct={total_exposure:.4f} | "
                    f"proposed_risk_pct={proposed_risk_pct:.4f} | "
                    f"combined={combined_exposure:.4f} | cap=0.0500"
                )

                return halted_mandate

            logger.debug(
                f"Governor: Exposure check passed | "
                f"total={total_exposure:.2%} + proposed={proposed_risk_pct:.2%} = "
                f"{combined_exposure:.2%} <= 5%"
            )
            return None  # Within cap — proceed with normal flow

        except Exception as e:
            logger.warning(f"Governor: Exposure cap check failed: {e} — proceeding without check")
            return None  # Don't block trades on error

    def _get_current_exposure(self, account_id: str, account_balance: float) -> float:
        """
        Get sum of all open position risk as % of equity.

        Args:
            account_id: Account identifier
            account_balance: Current account balance/equity for normalization

        Returns:
            Exposure as fraction (e.g., 0.03 = 3%)
        """
        try:
            positions = self._get_open_positions_for_account(account_id)
            if not positions:
                return 0.0

            total_risk_amount = 0.0
            for pos in positions:
                risk_pct = pos.get('risk_pct', 0.0)
                # Convert risk_pct fraction to dollar amount, then normalize back
                risk_amount = risk_pct * account_balance
                total_risk_amount += risk_amount

            exposure = total_risk_amount / account_balance if account_balance > 0 else 0.0
            return exposure

        except Exception as e:
            logger.warning(f"Governor: Could not get current exposure: {e}")
            return 0.0

    def _estimate_trade_risk_pct(
        self,
        trade_proposal: dict,
        mandate: RiskMandate,
        account_balance: float
    ) -> float:
        """
        Estimate risk % of proposed trade.

        Uses preliminary Kelly result if available ( mandate.risk_amount set),
        otherwise falls back to trade_proposal parameters.

        Args:
            trade_proposal: Trade proposal with stop_loss_pips, symbol, etc.
            mandate: Current RiskMandate (may have preliminary risk_amount from early Kelly)
            account_balance: Account balance for normalization

        Returns:
            Estimated risk as fraction (e.g., 0.02 = 2%)
        """
        # If preliminary Kelly has run and set risk_amount, use it
        if mandate.risk_amount > 0 and account_balance > 0:
            return mandate.risk_amount / account_balance

        # Fallback: estimate from trade_proposal using default risk parameters
        try:
            stop_loss_pips = trade_proposal.get('stop_loss_pips', 20.0)
            symbol = trade_proposal.get('symbol', 'EURUSD')
            broker_id = trade_proposal.get('broker_id', 'mt5_default')

            # Get pip value for this broker/symbol
            pip_value = self._get_pip_value(symbol, broker_id)

            # Default to 1 mini lot (0.1 lots) for estimation if no lot specified
            estimated_lot = 0.1

            # Risk amount = lot * stop_loss_pips * pip_value
            risk_amount_est = estimated_lot * stop_loss_pips * pip_value

            # Default max risk per trade at 2% if we can't estimate
            default_risk_pct = 0.02
            risk_pct = min(risk_amount_est / account_balance, default_risk_pct) if account_balance > 0 else default_risk_pct

            return risk_pct

        except Exception as e:
            logger.warning(f"Governor: Could not estimate trade risk: {e}")
            return 0.02  # Conservative default 2%

    def _get_open_positions_for_account(self, account_id: str) -> List[Dict[str, Any]]:
        """
        Get open positions for account from MT5 adapter.

        Args:
            account_id: Account identifier

        Returns:
            List of position dicts with risk_pct calculated
        """
        try:
            from src.api.tick_stream_handler import get_open_positions
            positions = get_open_positions(bot_id=None)  # Get all positions

            # Filter positions for this account (if account_id is tracked in positions)
            # MT5 positions may not have account_id directly — use bot_id routing instead
            account_positions = []
            for pos in positions:
                # Each position has bot_id which maps to an account via routing
                # For now, include all positions and let _get_current_exposure
                # handle any account-specific filtering if needed
                pos_with_risk = self._calculate_position_risk_pct(pos)
                if pos_with_risk.get('risk_pct', 0) > 0:
                    account_positions.append(pos_with_risk)

            return account_positions

        except Exception as e:
            logger.warning(f"Governor: Could not get open positions: {e}")
            return []

    def _calculate_position_risk_pct(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate risk_pct for a single position.

        Args:
            position: Position dict with lot, symbol, sl (stop loss), etc.

        Returns:
            Position dict with risk_pct added
        """
        try:
            symbol = position.get('symbol', 'EURUSD')
            lot = position.get('lot', position.get('volume', 0.0))
            broker_id = position.get('broker_id', 'mt5_default')

            # Get pip value
            pip_value = self._get_pip_value(symbol, broker_id)

            # Try to get stop loss from position
            sl = position.get('sl', position.get('stop_loss', None))

            if sl is None or sl == 0:
                # No stop loss — cannot calculate risk precisely
                # Use a default risk estimate based on lot size
                risk_pct = 0.01 * lot  # Rough estimate: 1% per lot
            else:
                # Risk = lot * sl_pips * pip_value
                # Assuming sl is price-based, convert to pips
                # For forex: sl_pips = |entry - sl| / pip_size
                entry_price = position.get('price', position.get('open_price', 0))
                if entry_price > 0 and sl != entry_price:
                    # Convert price difference to pips (assuming 4th decimal = 1 pip for forex)
                    sl_pips = abs(entry_price - sl) / 0.0001
                    risk_amount = lot * sl_pips * pip_value
                    # Risk pct = risk_amount / equity (will be normalized in caller)
                    risk_pct = risk_amount  # Will be divided by account_balance later
                else:
                    risk_pct = 0.01 * lot  # Default estimate

            position['risk_pct'] = risk_pct
            return position

        except Exception as e:
            logger.debug(f"Governor: Could not calculate position risk: {e}")
            position['risk_pct'] = 0.0
            return position

    def _get_open_positions_detail(self, account_id: str) -> List[Dict[str, Any]]:
        """
        Get detailed open positions for logging (FAIL-R02 context).

        Args:
            account_id: Account identifier

        Returns:
            List of position dicts with bot_id and risk_pct for logging
        """
        try:
            from src.api.tick_stream_handler import get_open_positions
            positions = get_open_positions(bot_id=None)

            return [
                {
                    "bot_id": pos.get('bot_id', 'unknown'),
                    "symbol": pos.get('symbol', 'UNKNOWN'),
                    "risk_pct": self._calculate_position_risk_pct(pos).get('risk_pct', 0.0)
                }
                for pos in positions
                if pos.get('bot_id')
            ]

        except Exception as e:
            logger.warning(f"Governor: Could not get position details: {e}")
            return []

    def _get_pip_value(self, symbol: str, broker_id: str) -> float:
        """
        Get pip value for a symbol/broker combination.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            broker_id: Broker identifier

        Returns:
            Pip value in account currency per lot per pip
        """
        # Default pip values by broker type
        default_pip_values = {
            'mt5_default': 10.0,  # $10 per lot per pip for most forex
            'icmarkets_raw': 10.0,
            'icmarkets': 10.0,
        }

        # Try broker-specific value first
        if broker_id in default_pip_values:
            return default_pip_values[broker_id]

        # Try to extract broker family from broker_id
        broker_family = broker_id.split('_')[0] if '_' in broker_id else broker_id
        if broker_family.lower() in ['icmarkets', 'ic']:
            return 10.0

        # Gold/XAUUSD has different pip value
        if symbol.upper() in ['XAUUSD', 'GOLD']:
            return 100.0  # $100 per lot per pip for gold

        # Default
        return 10.0
