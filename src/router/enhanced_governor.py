"""
Enhanced Governor - Kelly Calculator Integration

Integrates EnhancedKellyCalculator with the Strategy Router for
scientifically optimal position sizing with house money effect,
dynamic pip values, and fee-aware sizing.

Task Group 6: Enhanced Governor Integration

Story 4.10: SessionKellyModifiers integration replaces _update_house_money_effect
with session-scoped Kelly modifiers including House Money Mode (HMM) and
Reverse HMM effects.
"""

import logging
from typing import Dict, Optional, TYPE_CHECKING, Any
from datetime import datetime, timezone

from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator, KellyResult, EnhancedKellyConfig
from src.router.governor import Governor, RiskMandate
from src.risk.sizing.session_kelly_modifiers import SessionKellyModifiers, SessionKellyState
from src.router.sessions import SessionDetector, TradingSession

# Optional imports for database features
if TYPE_CHECKING:
    from src.database.models import PropFirmAccount, DailySnapshot, HouseMoneyState, BrokerRegistry
    from src.router.sentinel import RegimeReport
    from src.api.settings_endpoints import RiskSettings

try:
    from src.database.models import PropFirmAccount, DailySnapshot, HouseMoneyState, BrokerRegistry
    from src.database.engine import get_session
    DB_AVAILABLE = True
except ImportError:
    # Create stubs for type hints when database is not available
    class PropFirmAccount:
        id: int
        account_id: str
    class DailySnapshot:
        id: int
        account_id: int
        date: str
        daily_start_balance: float
        high_water_mark: float
        current_equity: float
        daily_drawdown_pct: float
        is_breached: bool
        def __init__(self, account_id: int, date: str, daily_start_balance: float,
                     high_water_mark: float, current_equity: float,
                     daily_drawdown_pct: float, is_breached: bool):
            self.id = 0
            self.account_id = account_id
            self.date = date
            self.daily_start_balance = daily_start_balance
            self.high_water_mark = high_water_mark
            self.current_equity = current_equity
            self.daily_drawdown_pct = daily_drawdown_pct
            self.is_breached = is_breached
    class HouseMoneyState:
        id: int
        account_id: str
        daily_start_balance: float
        current_pnl: float
        high_water_mark: float
        risk_multiplier: float
        is_preservation_mode: bool
        date: str
        def __init__(self, account_id: str, date: str, daily_start_balance: float,
                     current_pnl: float, high_water_mark: float,
                     risk_multiplier: float, is_preservation_mode: bool):
            self.id = 0
            self.account_id = account_id
            self.date = date
            self.daily_start_balance = daily_start_balance
            self.current_pnl = current_pnl
            self.high_water_mark = high_water_mark
            self.risk_multiplier = risk_multiplier
            self.is_preservation_mode = is_preservation_mode
    class BrokerRegistry:
        id: int
        broker_id: str
        pip_values: Dict[str, float]
    # Stub for get_session when database is not available
    def get_session() -> Any:
        raise RuntimeError("Database not available")
    DB_AVAILABLE = False
    logging.warning("Database models not available - running without database features")

# Import RegimeReport - used at runtime for isinstance checks and type hints
from src.router.sentinel import RegimeReport

logger = logging.getLogger(__name__)


class EnhancedGovernor(Governor):
    """
    Enhanced Governor with Kelly Criterion position sizing.

    Replaces the base Governor with EnhancedKellyCalculator integration.

    Features:
    - Kelly Criterion position sizing with 3-layer protection
    - House Money Effect (risk multiplier based on running P&L)
    - Dynamic pip value calculation (no hardcoding)
    - Fee-aware position sizing
    - Prop firm rule enforcement via PropGovernor
    - Session Kelly Modifiers (Story 4.10): Session-scoped HMM and Reverse HMM

    Attributes:
        kelly_calculator: EnhancedKellyCalculator instance
        house_money_multiplier: Current risk multiplier (1.0x, 1.4x, or 0.5x)
        account_id: Prop firm account ID for tracking
        session_kelly: SessionKellyModifiers instance for session-scoped modifiers
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        config: Optional[EnhancedKellyConfig] = None,
        settings: Optional['RiskSettings'] = None
    ):
        """
        Initialize Enhanced Governor with Kelly Calculator.

        Args:
            account_id: Optional prop firm account ID for house money tracking
            config: Optional Kelly configuration
            settings: Optional RiskSettings for configurable risk parameters
        """
        super().__init__()

        self.kelly_calculator = EnhancedKellyCalculator(config)
        self.account_id = account_id
        self.house_money_multiplier = 1.0
        self._daily_start_balance = 100000.0  # Default, will load from DB
        self._broker_id = 'mt5_default'  # Default broker ID
        self._db_available = DB_AVAILABLE

        # Store settings for risk parameter configuration
        self._settings = settings

        # Extract risk settings with fallback defaults for backwards compatibility
        if settings:
            self._house_money_threshold = settings.houseMoneyThreshold
            self._house_money_multiplier_up = settings.houseMoneyMultiplierUp
            self._house_money_multiplier_down = settings.houseMoneyMultiplierDown
            self._default_risk_per_trade = settings.defaultRiskPerTrade
            # Risk mode: fixed, dynamic, conservative
            self.risk_mode = getattr(settings, 'riskMode', 'dynamic')
        else:
            # Fallback defaults (backwards compatible)
            self._house_money_threshold = 0.05  # 5% threshold
            self._house_money_multiplier_up = 1.5
            self._house_money_multiplier_down = 0.5
            self._default_risk_per_trade = 0.02  # 2% default risk
            self.risk_mode = 'dynamic'

        # Story 4.10: Initialize SessionKellyModifiers
        # Use Story 4.10 thresholds: +8% for HMM, -10% for preservation
        self.session_kelly = SessionKellyModifiers(
            account_id=account_id,
            hmm_profit_threshold=0.08,  # Story 4.10: +8% threshold
            hmm_loss_threshold=-0.10,    # Story 4.10: -10% threshold
        )

        # Story 4.10: Track last session for automatic boundary detection
        self._last_session: Optional[TradingSession] = None
        self._session_initialized: bool = False

        # Load daily state if account_id provided and database is available
        if account_id and self._db_available:
            self._load_daily_state(account_id)

        # Log initialization with settings info
        settings_info = "with custom settings" if settings else "with defaults"
        logger.info(
            f"EnhancedGovernor initialized for account {account_id}, "
            f"house_money_multiplier={self.house_money_multiplier}, "
            f"db_available={self._db_available}, "
            f"{settings_info}"
        )

    def _check_session_boundary(self, utc_now: Optional[datetime] = None) -> None:
        """
        Story 4.10: Check if session has changed and auto-reset modifiers if needed.

        Called at the start of calculate_risk() to detect session boundaries
        and automatically reset session-level modifiers (Reverse HMM, premium boost).

        Args:
            utc_now: Current UTC time (defaults to now)
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)

        current_session = SessionDetector.get_current_session()

        # First call - just set the initial session, don't reset
        if not self._session_initialized:
            self._last_session = current_session
            self._session_initialized = True
            logger.info(f"Session tracking initialized: {current_session.value}")
            return

        # Detect session change
        if current_session != self._last_session:
            logger.info(
                f"Session boundary detected: {self._last_session.value} -> {current_session.value}. "
                f"Resetting session-level modifiers."
            )
            # Reset session-level modifiers
            self.session_kelly.on_session_close(utc_now)
            self._last_session = current_session

    def on_trade_result(self, is_win: bool) -> SessionKellyState:
        """
        Story 4.10: Report trade result for session Kelly modifier tracking.

        Call this when a trade closes to update the session loss counter
        (for Reverse HMM) and win streak counter (for premium re-enable).

        Args:
            is_win: True if trade was a winner, False if loser

        Returns:
            Updated SessionKellyState
        """
        logger.info(f"Trade result reported: is_win={is_win}")
        return self.session_kelly.on_trade_result(is_win)

    def calculate_risk(
        self,
        regime_report: RegimeReport,
        trade_proposal: Dict,
        account_balance: Optional[float] = None,
        broker_id: Optional[str] = None,
        account_id: Optional[str] = None,
        mode: str = "live",
        **kwargs: Any
    ) -> RiskMandate:
        """
        Calculate risk mandate using Kelly Criterion.

        Args:
            regime_report: Current market regime from Sentinel
            trade_proposal: Trade proposal dict with symbol, balance, etc.
            account_balance: Current account balance (optional, extracts from trade_proposal if not provided)
            broker_id: Broker identifier for fee-aware Kelly (optional, extracts from trade_proposal if not provided)
            account_id: Account identifier for account-specific limits (optional, extracts from trade_proposal if not provided)

        Fee Precedence:
            commission_per_lot and spread_pips are resolved as follows:
            1. Explicit values from trade_proposal (highest priority)
            2. Broker registry auto-lookup via broker_id (if values are 0.0)
            3. Default values: commission_per_lot=5.0, spread_pips=1.0

        Returns:
            RiskMandate with Kelly-based allocation scalar, including position_size and kelly_fraction
        """
        # Story 4.10: Check for session boundary and auto-reset modifiers if needed
        self._check_session_boundary()

        # Extract trade parameters
        symbol = trade_proposal.get('symbol', 'EURUSD')
        current_balance = trade_proposal.get('current_balance', self._daily_start_balance)

        # Extract account_balance, broker_id, and account_id from parameters or trade_proposal
        account_balance = account_balance or trade_proposal.get('account_balance', current_balance)
        broker_id = broker_id or trade_proposal.get('broker_id', self._broker_id)
        account_id = account_id or trade_proposal.get('account_id', self.account_id)

        # Log mode for audit trail
        mode_tag = "[DEMO]" if mode == "demo" else "[LIVE]"
        logger.debug(f"EnhancedGovernor: Processing trade for account_id={account_id}, mode={mode}")

        # For demo mode, use virtual balance from VirtualBalanceManager if available
        if mode == "demo":
            try:
                from src.router.virtual_balance import get_virtual_balance_manager
                from src.router.ea_registry import get_ea_registry
                
                ea_registry = get_ea_registry()
                ea_config = ea_registry.get(trade_proposal.get('bot_id', ''))
                
                if ea_config:
                    vb_manager = get_virtual_balance_manager()
                    virtual_account = vb_manager.get_account(ea_config.ea_id)
                    
                    if virtual_account:
                        # Use virtual balance for demo mode calculations
                        account_balance = virtual_account.current_balance
                        logger.info(f"{mode_tag} Using virtual balance: {account_balance:.2f} for EA {ea_config.ea_id}")
            except Exception as e:
                logger.warning(f"Could not get virtual balance for demo mode: {e}")

        # Update house money effect based on current account balance to ensure risk multiplier
        # reflects daily P&L before applying Kelly sizing
        self._update_house_money_effect(account_balance)

        # Get trading statistics for Kelly calculation
        # For now, use conservative defaults. In production, load from strategy performance.
        # F-14 correction: Baseline win rate is 52% (was incorrectly 55%)
        win_rate = trade_proposal.get('win_rate', 0.52)
        avg_win = trade_proposal.get('avg_win', 400.0)
        avg_loss = trade_proposal.get('avg_loss', 200.0)
        stop_loss_pips = trade_proposal.get('stop_loss_pips', 20.0)

        # Get ATR values for volatility adjustment
        current_atr = trade_proposal.get('current_atr', 0.0012)
        average_atr = trade_proposal.get('average_atr', 0.0010)

        # Determine pip_value: prefer trade_proposal value, otherwise use symbol-specific default
        # Kelly Calculator can still override via broker lookup even if we provide a default
        pip_value = trade_proposal.get('pip_value', self._get_pip_value(symbol, broker_id))

        # Extract fee parameters from trade_proposal
        # Precedence: explicit from trade_proposal > broker registry auto-lookup
        # Initialize to 0.0 to allow EnhancedKellyCalculator to auto-fetch from broker registry
        # when broker_id is supplied. Only override with non-zero values if explicitly provided.
        commission_per_lot = trade_proposal.get('commission_per_lot', 0.0)
        spread_pips = trade_proposal.get('spread_pips', 0.0)

        # Calculate Kelly position size with fee-aware parameters
        # Kelly Calculator will auto-fetch pip_value, commission, and spread from broker registry
        # only if the passed values are 0.0 (to maintain backward compatibility)
        kelly_result = self.kelly_calculator.calculate(
            account_balance=account_balance,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            current_atr=current_atr,
            average_atr=average_atr,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,  # Use symbol-specific value, Kelly may override via broker lookup
            regime_quality=regime_report.regime_quality,
            broker_id=broker_id,
            symbol=symbol,
            commission_per_lot=commission_per_lot,
            spread_pips=spread_pips
        )

        # Check for fee kill switch status
        if kelly_result.status == 'fee_blocked':
            # Return halted mandate when fees exceed profit
            return RiskMandate(
                allocation_scalar=0.0,
                risk_mode="HALTED",
                position_size=0.0,
                kelly_fraction=0.0,
                risk_amount=0.0,
                kelly_adjustments=kelly_result.adjustments_applied,
                notes="Fee kill switch activated - fees exceed expected profit"
            )

        # Calculate payoff ratio for risk-mode-aware Kelly fraction
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Use _calculate_kelly_fraction which respects risk_mode setting
        # This applies the risk_mode logic: fixed, conservative, or dynamic
        risk_mode_kelly_f = self._calculate_kelly_fraction(win_rate, payoff_ratio)

        # Apply house money multiplier to the risk-mode-aware Kelly fraction
        kelly_f_with_house_money = risk_mode_kelly_f * self.house_money_multiplier

        # Recalculate position_size and risk_amount using risk-mode-aware Kelly
        # This ensures the risk_mode setting actually affects trading
        scaled_risk_amount = account_balance * kelly_f_with_house_money
        scaled_position_size = scaled_risk_amount / (stop_loss_pips * pip_value) if stop_loss_pips > 0 and pip_value > 0 else 0.0

        # Apply base Governor physics-based throttling
        # Comment 2: Pass mode to base Governor for demo-specific risk scaling
        base_mandate = super().calculate_risk(
            regime_report, 
            trade_proposal, 
            mode=mode  # Comment 2: Pass mode through for demo risk adjustments
        )

        # Keep allocation_scalar as the physics-based clamp (do not re-apply kelly_f here)
        # This avoids double-scaling: kelly_f has already been applied to position_size/risk_amount
        physics_scalar = base_mandate.allocation_scalar

        # Apply physics scalar to all risk metrics for consistency
        # This ensures position sizes, risk amounts, and kelly_fraction are all clamped by governor
        final_position_size = scaled_position_size * physics_scalar
        final_risk_amount = scaled_risk_amount * physics_scalar
        final_kelly_fraction = kelly_f_with_house_money * physics_scalar

        # Build mandate with extended fields
        # Note: allocation_scalar is the physics clamp; kelly_fraction is the final Kelly after physics throttling
        # position_size and risk_amount incorporate both house_money_multiplier and physics_scalar
        # Comment 2: Set mandate.mode to preserve mode tag in risk notes

        # Add risk_mode info to kelly_adjustments
        # Note: adjustments_applied is a List[str], not a Dict - convert for RiskMandate
        if isinstance(kelly_result.adjustments_applied, list):
            risk_mode_adjustments = {
                'adjustments': kelly_result.adjustments_applied.copy() if kelly_result.adjustments_applied else [],
                'risk_mode': self.risk_mode,
                'risk_mode_applied': True
            }
        else:
            risk_mode_adjustments = kelly_result.adjustments_applied.copy() if kelly_result.adjustments_applied else {}
            risk_mode_adjustments['risk_mode'] = self.risk_mode
            risk_mode_adjustments['risk_mode_applied'] = True

        # Story 4.10: Add session Kelly modifier to adjustments
        session_state = self.session_kelly.get_current_state()
        risk_mode_adjustments['session_kelly'] = {
            'session_kelly_multiplier': session_state.session_kelly_multiplier,
            'hmm_multiplier': session_state.hmm_multiplier,
            'reverse_hmm_multiplier': session_state.reverse_hmm_multiplier,
            'session_loss_counter': session_state.session_loss_counter,
            'premium_boost_active': session_state.premium_boost_active,
            'is_premium_session': session_state.is_premium_session,
            'premium_assault': session_state.premium_assault.value if session_state.premium_assault else None,
            'is_house_money_active': session_state.is_house_money_active,
            'is_preservation_mode': session_state.is_preservation_mode,
            'multiplier_chain_display': self.session_kelly.get_multiplier_chain_display(),
        }

        mandate = RiskMandate(
            allocation_scalar=physics_scalar,
            risk_mode=base_mandate.risk_mode,
            position_size=final_position_size,
            kelly_fraction=final_kelly_fraction,
            risk_amount=final_risk_amount,
            kelly_adjustments=risk_mode_adjustments,
            mode=mode,  # Comment 2: Preserve mode in mandate for audit trail
            notes=(
                f"{mode_tag} Kelly: {risk_mode_kelly_f:.4f} [{self.risk_mode}] × "
                f"SessionKelly: {self.house_money_multiplier:.2f} ({self.session_kelly.get_multiplier_chain_display()}) × "
                f"Physics: {physics_scalar:.4f} = {final_kelly_fraction:.4f}. "
                f"Final position: {final_position_size:.4f} lots, risk: ${final_risk_amount:.2f}. "
                f"{base_mandate.notes or ''}"
            )
        )

        logger.info(
            f"EnhancedGovernor risk calculation: {mandate.allocation_scalar:.4f} "
            f"(Kelly: {risk_mode_kelly_f:.4f} [{self.risk_mode}], SessionKelly: {self.house_money_multiplier:.2f}, "
            f"HMM: {session_state.hmm_multiplier:.2f}, ReverseHMM: {session_state.reverse_hmm_multiplier:.2f}, "
            f"Losses: {session_state.session_loss_counter})"
        )

        return mandate

    def _update_house_money_effect(
        self,
        current_balance: float,
        current_session: Optional[TradingSession] = None
    ) -> None:
        """
        Update house money multiplier using SessionKellyModifiers.

        Story 4.10: Uses SessionKellyModifiers for session-scoped HMM with:
        - +8% daily P&L threshold for 1.4x House Money
        - -10% daily P&L threshold for 0.5x Preservation
        - Premium session threshold lowering
        - Reverse HMM (2/4/6 loss penalties)

        Args:
            current_balance: Current account balance
            current_session: Current trading session (optional, auto-detected if not provided)
        """
        if self._daily_start_balance == 0:
            logger.warning("Daily start balance is zero, using current balance")
            self._daily_start_balance = current_balance
            return

        # Calculate daily P&L percentage
        pnl_pct = (current_balance - self._daily_start_balance) / self._daily_start_balance
        current_pnl = current_balance - self._daily_start_balance

        # Detect current session if not provided
        if current_session is None:
            current_session = SessionDetector.get_current_session()

        # Story 4.10: Use SessionKellyModifiers for HMM computation
        utc_now = datetime.now(timezone.utc)
        session_state = self.session_kelly.compute_session_kelly_modifier(
            daily_pnl_pct=pnl_pct,
            current_session=current_session,
            utc_now=utc_now
        )

        # Update house money multiplier from SessionKellyModifiers
        self.house_money_multiplier = session_state.session_kelly_multiplier

        preservation_mode = session_state.is_preservation_mode
        is_house_money = session_state.is_house_money_active

        if is_house_money:
            logger.info(
                f"House Money ACTIVE: +{pnl_pct*100:.1f}% (session={current_session.value}) "
                f"- Risk multiplier: {self.house_money_multiplier:.2f}x"
            )
        elif preservation_mode:
            logger.warning(
                f"Preservation Mode: {pnl_pct*100:.1f}% (session={current_session.value}) "
                f"- Risk multiplier: {self.house_money_multiplier:.2f}x"
            )

        # Update house money state in database if account_id available and DB is enabled
        if self.account_id and self._db_available:
            self._update_house_money_state(current_balance, current_pnl, preservation_mode, session_state)

    def _calculate_kelly_fraction(self, win_rate: float, payoff_ratio: float) -> float:
        """
        Calculate Kelly fraction based on risk mode.

        Risk mode determines how Kelly is applied:
        - fixed: Always use default risk percentage, ignore Kelly
        - dynamic: Use Kelly calculation with fraction (normal behavior)
        - conservative: Cap at 50% of default risk

        Args:
            win_rate: Trade win rate (0.0 to 1.0)
            payoff_ratio: Reward to risk ratio (avg_win / avg_loss)

        Returns:
            Kelly fraction based on risk mode
        """
        # Calculate raw Kelly: f = (p * (b + 1) - 1) / b
        # where p = win_rate, b = payoff_ratio
        raw_kelly = (win_rate * (payoff_ratio + 1) - 1) / payoff_ratio
        raw_kelly = max(0, raw_kelly)  # Can't be negative

        # Get Kelly fraction from calculator config
        kelly_fraction_setting = self.kelly_calculator.config.kelly_fraction if self.kelly_calculator else 0.5

        if self.risk_mode == "fixed":
            # Always use default risk percentage, ignore Kelly
            return self._default_risk_per_trade

        elif self.risk_mode == "conservative":
            # Cap at 50% of default risk
            conservative_limit = self._default_risk_per_trade * 0.5
            kelly_adjusted = raw_kelly * kelly_fraction_setting
            return min(kelly_adjusted, conservative_limit)

        elif self.risk_mode == "dynamic":
            # Normal Kelly calculation with fraction
            kelly_adjusted = raw_kelly * kelly_fraction_setting
            return min(kelly_adjusted, self._default_risk_per_trade)

        # Default fallback (dynamic behavior)
        return min(raw_kelly * kelly_fraction_setting, self._default_risk_per_trade)

    def _get_pip_value(self, symbol: str, broker: str = 'mt5_default') -> float:
        """
        Get dynamic pip value from broker registry.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'XAUUSD')
            broker: Broker identifier

        Returns:
            Pip value per standard lot

        Note:
            Queries broker_registry table for actual pip values if available.
            Falls back to standard values if not found or DB unavailable.
            Uses correct metal pip values (XAUUSD=1.0, XAGUSD=50.0) to avoid 10x underestimation.
        """
        # Correct metal pip values (fallback when DB/registry unavailable)
        # These match BrokerRegistryManager.create_broker() defaults to avoid divergence
        pip_values = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'USDJPY': 9.09,
            'USDCHF': 10.0,
            'AUDUSD': 10.0,
            'NZDUSD': 10.0,
            'USDCAD': 10.0,
            'XAUUSD': 1.0,   # Gold: $1 per pip (not $10 - corrects 10x underestimation)
            'XAGUSD': 50.0,  # Silver: $50 per pip (not $10 - corrects 5x underestimation)
        }

        # If DB not available, use fallback values
        if not self._db_available:
            pip_value = pip_values.get(symbol, 10.0)
            logger.debug(f"Pip value for {symbol}@{broker}: ${pip_value} (fallback, no DB)")
            return pip_value

        try:
            session = get_session()

            # Query broker registry for pip values
            broker_record = session.query(BrokerRegistry).filter_by(
                broker_id=broker
            ).first()

            pip_value = 10.0  # Default

            if broker_record and broker_record.pip_values:
                # Get pip value for symbol from JSON field
                pip_values_dict = broker_record.pip_values
                # Use local pip_values dict as fallback for metals if not in registry
                pip_value = pip_values_dict.get(symbol, pip_values.get(symbol, 10.0))
                logger.debug(f"Pip value for {symbol}@{broker}: ${pip_value} (from registry)")
            else:
                # Fall back to correct defaults (not hardcoded 10.0 for metals)
                pip_value = pip_values.get(symbol, 10.0)
                logger.debug(f"Pip value for {symbol}@{broker}: ${pip_value} (fallback)")

            session.close()
            return pip_value

        except Exception as e:
            logger.error(f"Error getting pip value from broker registry: {e}")
            # Return fallback value on error (use correct metal values)
            return pip_values.get(symbol, 10.0)

    def _load_daily_state(self, account_id: str) -> None:
        """
        Load daily start balance from database.

        Args:
            account_id: Prop firm account ID
        """
        if not self._db_available:
            logger.debug("Database not available - using default daily start balance")
            return

        try:
            session = get_session()

            # Get account
            account = session.query(PropFirmAccount).filter_by(
                account_id=account_id
            ).first()

            if account:
                # Get today's snapshot
                today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                snapshot = session.query(DailySnapshot).filter_by(
                    account_id=account.id,
                    date=today
                ).first()

                if snapshot:
                    self._daily_start_balance = snapshot.daily_start_balance
                    logger.info(
                        f"Loaded daily start balance: ${self._daily_start_balance:.2f} "
                        f"for account {account_id}"
                    )
                else:
                    # Create new snapshot for today
                    self._create_daily_snapshot(session, account, 100000.0)
                    self._daily_start_balance = 100000.0
                    logger.info(f"Created new daily snapshot for account {account_id}")

            session.close()

        except Exception as e:
            logger.error(f"Error loading daily state for account {account_id}: {e}")
            self._daily_start_balance = 100000.0

    def _create_daily_snapshot(
        self,
        session,
        account,
        start_balance: float
    ) -> None:
        """
        Create a new daily snapshot.

        Args:
            session: SQLAlchemy session
            account: PropFirmAccount instance
            start_balance: Starting balance for the day
        """
        if not self._db_available:
            return

        today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

        snapshot = DailySnapshot(
            account_id=account.id,
            date=today,
            daily_start_balance=start_balance,
            high_water_mark=start_balance,
            current_equity=start_balance,
            daily_drawdown_pct=0.0,
            is_breached=False
        )

        session.add(snapshot)
        session.commit()

        logger.info(f"Created daily snapshot for account {account.account_id}")

    def _update_house_money_state(
        self,
        current_balance: float,
        current_pnl: float,
        preservation_mode: bool,
        session_state: Optional['SessionKellyState'] = None
    ) -> None:
        """
        Update house money state in database with session-scoped fields.

        Story 4.10: Stores session-scoped fields including current_session,
        is_premium_session, and premium_assault_type.

        Args:
            current_balance: Current account balance
            current_pnl: Current daily profit/loss
            preservation_mode: Whether preservation mode is active
            session_state: Optional SessionKellyState with session-scoped data
        """
        if not self._db_available:
            return

        try:
            db_session = get_session()

            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

            # Ensure account_id is not None (should be guaranteed by caller)
            if self.account_id is None:
                logger.warning("Cannot update house money state: account_id is None")
                return

            # Check if record exists for today
            state = db_session.query(HouseMoneyState).filter_by(
                account_id=self.account_id,
                date=today
            ).first()

            if state:
                # Update existing record
                state.current_pnl = current_pnl
                state.risk_multiplier = self.house_money_multiplier
                state.is_preservation_mode = preservation_mode

                # Update high water mark
                if current_balance > state.high_water_mark:
                    state.high_water_mark = current_balance

                # Story 4.10: Update session-scoped fields
                if session_state:
                    state.current_session = session_state.current_session
                    state.is_premium_session = session_state.is_premium_session
                    state.premium_assault_type = (
                        session_state.premium_assault.value
                        if session_state.premium_assault else None
                    )
                    state.session_start = session_state.last_updated
            else:
                # Create new record
                state = HouseMoneyState(
                    account_id=self.account_id,
                    date=today,
                    daily_start_balance=self._daily_start_balance,
                    current_pnl=current_pnl,
                    high_water_mark=max(current_balance, self._daily_start_balance),
                    risk_multiplier=self.house_money_multiplier,
                    is_preservation_mode=preservation_mode,
                    # Story 4.10: Session-scoped fields
                    current_session=session_state.current_session if session_state else None,
                    is_premium_session=session_state.is_premium_session if session_state else False,
                    premium_assault_type=(
                        session_state.premium_assault.value
                        if session_state and session_state.premium_assault else None
                    ),
                    session_start=session_state.last_updated if session_state else None,
                )
                db_session.add(state)

            db_session.commit()
            db_session.close()

        except Exception as e:
            logger.error(f"Error updating house money state: {e}")

    def reset_daily_state(self, start_balance: float) -> None:
        """
        Reset daily state (called at start of trading day).

        Args:
            start_balance: Starting balance for the new day
        """
        self._daily_start_balance = start_balance
        self.house_money_multiplier = 1.0

        logger.info(f"Daily state reset: start_balance=${start_balance:.2f}")

        # Update database if account_id available and DB is enabled
        if self.account_id and self._db_available:
            try:
                session = get_session()

                account = session.query(PropFirmAccount).filter_by(
                    account_id=self.account_id
                ).first()

                if account:
                    self._create_daily_snapshot(session, account, start_balance)

                session.close()

            except Exception as e:
                logger.error(f"Error resetting daily state: {e}")
