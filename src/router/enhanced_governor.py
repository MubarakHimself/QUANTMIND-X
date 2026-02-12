"""
Enhanced Governor - Kelly Calculator Integration

Integrates EnhancedKellyCalculator with the Strategy Router for
scientifically optimal position sizing with house money effect,
dynamic pip values, and fee-aware sizing.

Task Group 6: Enhanced Governor Integration
"""

import logging
from typing import Dict, Optional, TYPE_CHECKING, Any
from datetime import datetime, timezone

from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator, KellyResult, EnhancedKellyConfig
from src.router.governor import Governor, RiskMandate

# Optional imports for database features
if TYPE_CHECKING:
    from src.database.models import PropFirmAccount, DailySnapshot, HouseMoneyState, BrokerRegistry
    from src.router.sentinel import RegimeReport

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

    Attributes:
        kelly_calculator: EnhancedKellyCalculator instance
        house_money_multiplier: Current risk multiplier (1.0x, 1.5x, or 0.5x)
        account_id: Prop firm account ID for tracking
    """

    def __init__(self, account_id: Optional[str] = None, config: Optional[EnhancedKellyConfig] = None):
        """
        Initialize Enhanced Governor with Kelly Calculator.

        Args:
            account_id: Optional prop firm account ID for house money tracking
            config: Optional Kelly configuration
        """
        super().__init__()

        self.kelly_calculator = EnhancedKellyCalculator(config)
        self.account_id = account_id
        self.house_money_multiplier = 1.0
        self._daily_start_balance = 100000.0  # Default, will load from DB
        self._broker_id = 'mt5_default'  # Default broker ID
        self._db_available = DB_AVAILABLE

        # Load daily state if account_id provided and database is available
        if account_id and self._db_available:
            self._load_daily_state(account_id)

        logger.info(
            f"EnhancedGovernor initialized for account {account_id}, "
            f"house_money_multiplier={self.house_money_multiplier}, "
            f"db_available={self._db_available}"
        )

    def calculate_risk(
        self,
        regime_report: RegimeReport,
        trade_proposal: Dict,
        account_balance: Optional[float] = None,
        broker_id: Optional[str] = None,
        **kwargs: Any
    ) -> RiskMandate:
        """
        Calculate risk mandate using Kelly Criterion.

        Args:
            regime_report: Current market regime from Sentinel
            trade_proposal: Trade proposal dict with symbol, balance, etc.
            account_balance: Current account balance (optional, extracts from trade_proposal if not provided)
            broker_id: Broker identifier for fee-aware Kelly (optional, extracts from trade_proposal if not provided)

        Returns:
            RiskMandate with Kelly-based allocation scalar, including position_size and kelly_fraction
        """
        # Extract trade parameters
        symbol = trade_proposal.get('symbol', 'EURUSD')
        current_balance = trade_proposal.get('current_balance', self._daily_start_balance)

        # Extract account_balance and broker_id from parameters or trade_proposal
        account_balance = account_balance or trade_proposal.get('account_balance', current_balance)
        broker_id = broker_id or trade_proposal.get('broker_id', self._broker_id)

        # Update house money effect based on current account balance to ensure risk multiplier
        # reflects daily P&L before applying Kelly sizing
        self._update_house_money_effect(account_balance)

        # Get trading statistics for Kelly calculation
        # For now, use conservative defaults. In production, load from strategy performance.
        win_rate = trade_proposal.get('win_rate', 0.55)
        avg_win = trade_proposal.get('avg_win', 400.0)
        avg_loss = trade_proposal.get('avg_loss', 200.0)
        stop_loss_pips = trade_proposal.get('stop_loss_pips', 20.0)

        # Get ATR values for volatility adjustment
        current_atr = trade_proposal.get('current_atr', 0.0012)
        average_atr = trade_proposal.get('average_atr', 0.0010)

        # Determine pip_value: prefer trade_proposal value, otherwise use symbol-specific default
        # Kelly Calculator can still override via broker lookup even if we provide a default
        pip_value = trade_proposal.get('pip_value', self._get_pip_value(symbol, broker_id))

        # Calculate Kelly position size with fee-aware parameters
        # Kelly Calculator will auto-fetch pip_value, commission, and spread from broker registry
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
            commission_per_lot=0.0,  # Kelly will auto-fetch from broker
            spread_pips=0.0  # Kelly will auto-fetch from broker
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

        # Apply house money multiplier consistently to all risk metrics
        kelly_f_with_house_money = kelly_result.kelly_f * self.house_money_multiplier
        scaled_position_size = kelly_result.position_size * self.house_money_multiplier
        scaled_risk_amount = kelly_result.risk_amount * self.house_money_multiplier

        # Apply base Governor physics-based throttling
        base_mandate = super().calculate_risk(regime_report, trade_proposal)

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
        mandate = RiskMandate(
            allocation_scalar=physics_scalar,
            risk_mode=base_mandate.risk_mode,
            position_size=final_position_size,
            kelly_fraction=final_kelly_fraction,
            risk_amount=final_risk_amount,
            kelly_adjustments=kelly_result.adjustments_applied,
            notes=(
                f"Kelly: {kelly_result.kelly_f:.4f} × "
                f"House Money: {self.house_money_multiplier:.2f} × "
                f"Physics: {physics_scalar:.4f} = {final_kelly_fraction:.4f}. "
                f"Final position: {final_position_size:.4f} lots, risk: ${final_risk_amount:.2f}. "
                f"{base_mandate.notes or ''}"
            )
        )

        logger.info(
            f"EnhancedGovernor risk calculation: {mandate.allocation_scalar:.4f} "
            f"(Kelly: {kelly_result.kelly_f:.4f}, House Money: {self.house_money_multiplier:.2f})"
        )

        return mandate

    def _update_house_money_effect(self, current_balance: float) -> None:
        """
        Update house money multiplier based on running P&L.

        Risk multiplier logic:
        - 1.5x when up > 5% (trade with house money)
        - 1.0x when within ±3% (normal trading)
        - 0.5x when down > 3% (preservation mode)

        Args:
            current_balance: Current account balance
        """
        if self._daily_start_balance == 0:
            logger.warning("Daily start balance is zero, using current balance")
            self._daily_start_balance = current_balance
            return

        pnl_pct = (current_balance - self._daily_start_balance) / self._daily_start_balance
        current_pnl = current_balance - self._daily_start_balance

        if pnl_pct > 0.05:
            # Up more than 5% - increase risk (house money)
            self.house_money_multiplier = 1.5
            preservation_mode = False
            logger.info(f"House Money ACTIVE: +{pnl_pct*100:.1f}% - Risk multiplier: 1.5x")
        elif pnl_pct < -0.03:
            # Down more than 3% - decrease risk (preservation mode)
            self.house_money_multiplier = 0.5
            preservation_mode = True
            logger.warning(f"Preservation Mode: {pnl_pct*100:.1f}% - Risk multiplier: 0.5x")
        else:
            # Normal range
            self.house_money_multiplier = 1.0
            preservation_mode = False

        # Update house money state in database if account_id available and DB is enabled
        if self.account_id and self._db_available:
            self._update_house_money_state(current_balance, current_pnl, preservation_mode)

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
        """
        # Standard pip values (fallback when DB unavailable)
        pip_values = {
            'EURUSD': 10.0,
            'GBPUSD': 10.0,
            'USDJPY': 9.09,
            'USDCHF': 10.0,
            'AUDUSD': 10.0,
            'NZDUSD': 10.0,
            'USDCAD': 10.0,
            'XAUUSD': 10.0,
            'XAGUSD': 10.0,
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
                pip_value = pip_values_dict.get(symbol, 10.0)
                logger.debug(f"Pip value for {symbol}@{broker}: ${pip_value} (from registry)")
            else:
                # Fall back to hardcoded values if broker not found
                pip_value = pip_values.get(symbol, 10.0)
                logger.debug(f"Pip value for {symbol}@{broker}: ${pip_value} (fallback)")

            session.close()
            return pip_value

        except Exception as e:
            logger.error(f"Error getting pip value from broker registry: {e}")
            # Return fallback value on error
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
        preservation_mode: bool
    ) -> None:
        """
        Update house money state in database.

        Args:
            current_balance: Current account balance
            current_pnl: Current daily profit/loss
            preservation_mode: Whether preservation mode is active
        """
        if not self._db_available:
            return

        try:
            session = get_session()

            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')

            # Ensure account_id is not None (should be guaranteed by caller)
            if self.account_id is None:
                logger.warning("Cannot update house money state: account_id is None")
                return

            # Check if record exists for today
            state = session.query(HouseMoneyState).filter_by(
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
            else:
                # Create new record
                state = HouseMoneyState(
                    account_id=self.account_id,
                    date=today,
                    daily_start_balance=self._daily_start_balance,
                    current_pnl=current_pnl,
                    high_water_mark=max(current_balance, self._daily_start_balance),
                    risk_multiplier=self.house_money_multiplier,
                    is_preservation_mode=preservation_mode
                )
                session.add(state)

            session.commit()
            session.close()

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
