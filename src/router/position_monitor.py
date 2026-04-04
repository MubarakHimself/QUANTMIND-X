"""
Position Monitor Service (Layer 2) - Cloudzy Local Service

Dynamic SL modification service that runs on Cloudzy (not Contabo) for latency reasons.
Monitors open positions and modifies SL based on:
1. Price moved 1R in favour -> move SL to break-even
2. Sentinel cached regime changed -> move SL to nearest structural level (tighten, not close)

Latency budget: Redis read (1ms) + ZMQ modify (5-20ms broker RT) = 6-21ms total

Multi-timeframe conflict resolution:
- Only M5 regime shifts, H1 intact -> no action
- H1 flips against open position -> tighten SL to nearest structural level
- Both H1 AND H4 flip against open position -> tighten aggressively
- CHAOS signal -> Layer 3 applies

Spec: Addendum Section 3, Layer 2
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter

logger = logging.getLogger(__name__)


# Pip size for common forex pairs (price precision varies)
PIP_SIZE = {
    'EURUSD': 0.0001,
    'GBPUSD': 0.0001,
    'USDJPY': 0.01,
    'USDCHF': 0.0001,
    'AUDUSD': 0.0001,
    'NZDUSD': 0.0001,
    'USDCAD': 0.0001,
    'XAUUSD': 0.01,  # Gold uses 0.01 (cents)
    'XAGUSD': 0.01,  # Silver uses 0.01
}
DEFAULT_PIP_SIZE = 0.0001


@dataclass
class OpenPosition:
    """
    Open position tracked by the Position Monitor Service.

    Attributes:
        ticket: MT5 ticket number
        symbol: Trading symbol (e.g., 'EURUSD')
        entry_price: Price at which position was opened
        current_sl: Current stop loss price
        entry_time: When the position was opened
        side: 'BUY' or 'SELL'
        stop_distance_pips: Original stop loss distance in pips
        has_moved_to_breakeven: Whether SL has already been moved to breakeven
        entry_regime_m5: M5 regime at entry (for conflict resolution)
        entry_regime_h1: H1 regime at entry (for conflict resolution)
    """
    ticket: int
    symbol: str
    entry_price: float
    current_sl: float
    entry_time: datetime
    side: str  # 'BUY' or 'SELL'
    stop_distance_pips: float
    has_moved_to_breakeven: bool = False
    entry_regime_m5: Optional[str] = None
    entry_regime_h1: Optional[str] = None
    entry_regime_h4: Optional[str] = None
    current_price: float = 0.0
    unrealized_profit_pips: float = 0.0

    def update_current_price(self, price: float) -> None:
        """Update current price and calculate unrealized profit in pips."""
        self.current_price = price
        pip_size = PIP_SIZE.get(self.symbol, DEFAULT_PIP_SIZE)

        if self.side.upper() == 'BUY':
            self.unrealized_profit_pips = (price - self.entry_price) / pip_size
        else:  # SELL
            self.unrealized_profit_pips = (self.entry_price - price) / pip_size

    def is_in_profit_by_r(self, r: float = 1.0) -> bool:
        """Check if position is in profit by at least R pips."""
        return self.unrealized_profit_pips >= self.stop_distance_pips * r

    def calculate_break_even_sl(self) -> float:
        """Calculate break-even stop loss price (entry price +/- 0.5 pip buffer)."""
        pip_size = PIP_SIZE.get(self.symbol, DEFAULT_PIP_SIZE)
        buffer = pip_size * 0.5  # 0.5 pip buffer

        if self.side.upper() == 'BUY':
            return self.entry_price + buffer
        else:  # SELL
            return self.entry_price - buffer

    def calculate_tightened_sl(self, tightening_factor: float = 0.5) -> float:
        """
        Calculate tightened SL towards current price.

        Args:
            tightening_factor: Factor to tighten by (0.5 = move 50% of remaining distance)

        Returns:
            New SL price moved closer to current price
        """
        pip_size = PIP_SIZE.get(self.symbol, DEFAULT_PIP_SIZE)

        if self.side.upper() == 'BUY':
            # For BUY: SL moves up towards entry + some profit
            remaining_distance = self.current_price - self.current_sl
            tightened_distance = remaining_distance * tightening_factor
            return self.current_sl + tightened_distance
        else:  # SELL
            # For SELL: SL moves down towards entry + some profit
            remaining_distance = self.current_sl - self.current_price
            tightened_distance = remaining_distance * tightening_factor
            return self.current_sl - tightened_distance


@dataclass
class RegimeState:
    """
    Current regime state from Sentinel (cached in Redis).

    Attributes:
        m5_regime: M5 timeframe regime
        h1_regime: H1 timeframe regime
        h4_regime: H4 timeframe regime
        chaos_signal: Whether CHAOS signal is active
        timestamp: When regime was detected
    """
    m5_regime: Optional[str] = None
    h1_regime: Optional[str] = None
    h4_regime: Optional[str] = None
    chaos_signal: bool = False
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModificationResult:
    """Result of a position modification attempt."""
    success: bool
    ticket: int
    action: str  # 'breakeven', 'regime_tighten', 'aggressive_tighten'
    old_sl: float
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    error: Optional[str] = None
    reason: str = ""


class PositionMonitorService:
    """
    Layer 2 Position Monitor Service for dynamic SL modification.

    Runs on Cloudzy for low-latency Redis reads and ZMQ broker communication.

    Key behaviors:
    1. Move SL to break-even when position is in profit >= 1R
    2. Tighten SL when regime flips against position
    3. Multi-timeframe conflict resolution (M5/H1/H4)
    4. ZMQ order modification with 5-20ms latency budget

    Attributes:
        open_positions: Dict keyed by MT5 ticket
        mt5_adapter: MT5SocketAdapter for ZMQ communication
        redis_client: Redis client for reading Sentinel regime cache
    """

    # Redis key for Sentinel regime cache
    REGIME_CACHE_KEY = "sentinel:regime:{symbol}"

    def __init__(
        self,
        mt5_adapter: 'MT5SocketAdapter',
        redis_client: Any = None,
        symbol: str = "EURUSD"
    ):
        """
        Initialize Position Monitor Service.

        Args:
            mt5_adapter: MT5SocketAdapter for ZMQ order modifications
            redis_client: Redis client for reading Sentinel regime cache
            symbol: Symbol to monitor (default: EURUSD)
        """
        self._mt5 = mt5_adapter
        self._redis = redis_client
        self._symbol = symbol

        # Track open positions by ticket
        self.open_positions: Dict[int, OpenPosition] = {}

        # Current regime state
        self._current_regime = RegimeState()

        logger.info(
            f"PositionMonitorService initialized for {symbol} "
            f"(Cloudzy Layer 2, latency budget: 6-21ms)"
        )

    # =========================================================================
    # Position Management
    # =========================================================================

    def register_position(
        self,
        ticket: int,
        symbol: str,
        entry_price: float,
        current_sl: float,
        side: str,
        stop_distance_pips: float,
        entry_time: Optional[datetime] = None,
        entry_regime_m5: Optional[str] = None,
        entry_regime_h1: Optional[str] = None,
        entry_regime_h4: Optional[str] = None
    ) -> OpenPosition:
        """
        Register a new position for monitoring.

        Called by RiskGovernor after calculate_position_size() returns.

        Args:
            ticket: MT5 ticket number
            symbol: Trading symbol
            entry_price: Position entry price
            current_sl: Current stop loss price
            side: 'BUY' or 'SELL'
            stop_distance_pips: Original SL distance in pips
            entry_time: When position was opened
            entry_regime_m5: M5 regime at entry
            entry_regime_h1: H1 regime at entry
            entry_regime_h4: H4 regime at entry

        Returns:
            OpenPosition that is now being monitored
        """
        position = OpenPosition(
            ticket=ticket,
            symbol=symbol,
            entry_price=entry_price,
            current_sl=current_sl,
            entry_time=entry_time or datetime.now(timezone.utc),
            side=side,
            stop_distance_pips=stop_distance_pips,
            has_moved_to_breakeven=False,
            entry_regime_m5=entry_regime_m5,
            entry_regime_h1=entry_regime_h1,
            entry_regime_h4=entry_regime_h4,
            current_price=entry_price
        )

        self.open_positions[ticket] = position

        logger.info(
            f"Registered position: ticket={ticket}, {symbol} {side} "
            f"@ {entry_price}, SL={current_sl}, dist={stop_distance_pips}pips"
        )

        return position

    def unregister_position(self, ticket: int) -> bool:
        """
        Stop monitoring a position (e.g., when closed).

        Args:
            ticket: MT5 ticket number

        Returns:
            True if position was being monitored
        """
        if ticket in self.open_positions:
            del self.open_positions[ticket]
            logger.info(f"Unregistered position: ticket={ticket}")
            return True
        return False

    # =========================================================================
    # Break-Even Check (AC #1)
    # =========================================================================

    def _check_break_even(self, ticket: int, current_price: float) -> ModificationResult:
        """
        Check if position qualifies for break-even move and execute if so.

        AC #1: Price moved 1R in favour -> move SL to break-even.

        Args:
            ticket: MT5 ticket number
            current_price: Current market price

        Returns:
            ModificationResult with outcome
        """
        position = self.open_positions.get(ticket)
        if not position:
            return ModificationResult(
                success=False,
                ticket=ticket,
                action="breakeven",
                old_sl=0.0,
                error=f"Position {ticket} not monitored"
            )

        # Update current price
        position.update_current_price(current_price)

        # Check if already at breakeven
        if position.has_moved_to_breakeven:
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="breakeven",
                old_sl=position.current_sl,
                new_sl=position.current_sl,
                reason="Already moved to breakeven"
            )

        # Check if position is in profit >= 1R
        if not position.is_in_profit_by_r(1.0):
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="breakeven",
                old_sl=position.current_sl,
                reason=f"Not at 1R yet: {position.unrealized_profit_pips:.1f}pips profit"
            )

        # Calculate break-even SL
        new_sl = position.calculate_break_even_sl()

        # Don't move if current SL is already better
        if position.side.upper() == 'BUY' and position.current_sl >= new_sl:
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="breakeven",
                old_sl=position.current_sl,
                new_sl=position.current_sl,
                reason="SL already better than breakeven"
            )
        if position.side.upper() == 'SELL' and position.current_sl <= new_sl:
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="breakeven",
                old_sl=position.current_sl,
                new_sl=position.current_sl,
                reason="SL already better than breakeven"
            )

        # Execute modification
        return self._send_modify_via_zmq(ticket, new_sl, None, "breakeven")

    # =========================================================================
    # Regime Shift Check (AC #2)
    # =========================================================================

    def _check_regime_shift(
        self,
        ticket: int,
        current_price: float,
        regime_state: RegimeState
    ) -> ModificationResult:
        """
        Check if regime shift requires SL tightening.

        AC #2: Sentinel cached regime changed -> move SL to nearest structural level.

        Multi-timeframe conflict resolution:
        - Only M5 regime shifts, H1 intact -> no action
        - H1 flips against open position -> tighten SL (0.5 factor)
        - Both H1 AND H4 flip against open position -> tighten aggressively (0.75 factor)

        Args:
            ticket: MT5 ticket number
            current_price: Current market price
            regime_state: Current regime state from Sentinel

        Returns:
            ModificationResult with outcome
        """
        position = self.open_positions.get(ticket)
        if not position:
            return ModificationResult(
                success=False,
                ticket=ticket,
                action="regime_tighten",
                old_sl=0.0,
                error=f"Position {ticket} not monitored"
            )

        # Update current price
        position.update_current_price(current_price)

        # Determine regime direction relative to position
        # For BUY positions, bearish regimes are against us
        # For SELL positions, bullish regimes are against us
        is_bearish = regime_state.h1_regime in ['TREND_BEAR', 'RANGE_STABLE', 'RANGE_VOLATILE']
        is_bullish = regime_state.h1_regime in ['TREND_BULL', 'RANGE_STABLE', 'RANGE_VOLATILE']
        is_chaos = regime_state.chaos_signal

        # Check M5 only shift (no action per spec)
        if position.entry_regime_h1 == regime_state.h1_regime:
            # H1 unchanged - no regime shift action
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="regime_tighten",
                old_sl=position.current_sl,
                reason="H1 regime unchanged, no action"
            )

        # Determine tightening action
        action = "regime_tighten"
        tightening_factor = 0.5  # Default: tighten 50%

        if is_chaos:
            # CHAOS -> Layer 3 applies, Layer 2 should not act
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="regime_tighten",
                old_sl=position.current_sl,
                reason="CHAOS signal - Layer 3 handles"
            )

        if position.side.upper() == 'BUY' and is_bearish:
            # BUY position, H1 turned bearish -> tighten
            if regime_state.h4_regime in ['TREND_BEAR', 'RANGE_STABLE', 'RANGE_VOLATILE']:
                # Both H1 and H4 bearish -> aggressive tighten
                action = "aggressive_tighten"
                tightening_factor = 0.75
            else:
                # Only H1 bearish -> moderate tighten
                tightening_factor = 0.5

        elif position.side.upper() == 'SELL' and is_bullish:
            # SELL position, H1 turned bullish -> tighten
            if regime_state.h4_regime in ['TREND_BULL', 'RANGE_STABLE', 'RANGE_VOLATILE']:
                # Both H1 and H4 bullish -> aggressive tighten
                action = "aggressive_tighten"
                tightening_factor = 0.75
            else:
                # Only H1 bullish -> moderate tighten
                tightening_factor = 0.5
        else:
            # Regime shifted in favour or no conflict
            return ModificationResult(
                success=True,
                ticket=ticket,
                action="regime_tighten",
                old_sl=position.current_sl,
                reason="Regime shift not against position"
            )

        # Calculate tightened SL
        new_sl = position.calculate_tightened_sl(tightening_factor)

        # Don't move if already at or better than proposed SL
        if position.side.upper() == 'BUY' and position.current_sl >= new_sl:
            return ModificationResult(
                success=True,
                ticket=ticket,
                action=action,
                old_sl=position.current_sl,
                new_sl=position.current_sl,
                reason="SL already at or better than tightened level"
            )
        if position.side.upper() == 'SELL' and position.current_sl <= new_sl:
            return ModificationResult(
                success=True,
                ticket=ticket,
                action=action,
                old_sl=position.current_sl,
                new_sl=position.current_sl,
                reason="SL already at or better than tightened level"
            )

        # Execute modification
        return self._send_modify_via_zmq(ticket, new_sl, None, action)

    # =========================================================================
    # ZMQ Order Modification
    # =========================================================================

    def _send_modify_via_zmq(
        self,
        ticket: int,
        new_sl: Optional[float],
        new_tp: Optional[float],
        action: str
    ) -> ModificationResult:
        """
        Send order modification via ZMQ to MT5 bridge.

        Uses MT5SocketAdapter.modify_order() which has 5-20ms latency budget.

        Args:
            ticket: MT5 ticket number
            new_sl: New stop loss price
            new_tp: New take profit price
            action: Action description for logging

        Returns:
            ModificationResult with outcome
        """
        if not self._mt5:
            return ModificationResult(
                success=False,
                ticket=ticket,
                action=action,
                old_sl=0.0,
                error="MT5 adapter not available"
            )

        position = self.open_positions.get(ticket)
        old_sl = position.current_sl if position else 0.0

        try:
            # Call MT5SocketAdapter.modify_order
            success = self._mt5.modify_order(
                order_id=str(ticket),
                new_stop_loss=new_sl,
                new_take_profit=new_tp
            )

            if success:
                # Update local state
                if position and new_sl is not None:
                    position.current_sl = new_sl
                    if action == "breakeven":
                        position.has_moved_to_breakeven = True

                logger.info(
                    f"SL modification success: ticket={ticket}, action={action}, "
                    f"old_sl={old_sl:.5f} -> new_sl={new_sl:.5f}"
                )

                return ModificationResult(
                    success=True,
                    ticket=ticket,
                    action=action,
                    old_sl=old_sl,
                    new_sl=new_sl,
                    new_tp=new_tp,
                    reason=f"{action} executed"
                )
            else:
                return ModificationResult(
                    success=False,
                    ticket=ticket,
                    action=action,
                    old_sl=old_sl,
                    error="MT5 modification returned False"
                )

        except Exception as e:
            logger.error(f"ZMQ modification error for ticket={ticket}: {e}")
            return ModificationResult(
                success=False,
                ticket=ticket,
                action=action,
                old_sl=old_sl,
                error=str(e)
            )

    # =========================================================================
    # Main Check Loop
    # =========================================================================

    def check_and_modify_positions(
        self,
        prices: Dict[str, float],
        regime_state: Optional[RegimeState] = None
    ) -> List[ModificationResult]:
        """
        Check all positions and apply modifications as needed.

        Called:
        1. On each Sentinel regime update
        2. Every 10 seconds via ticker for break-even checks

        Args:
            prices: Dict of symbol -> current price
            regime_state: Current regime state (optional, will fetch from Redis if not provided)

        Returns:
            List of ModificationResults for all attempted modifications
        """
        results = []

        # Fetch regime from Redis if not provided
        if regime_state is None:
            regime_state = self._fetch_regime_from_redis()

        # Update current regime
        self._current_regime = regime_state

        for ticket, position in list(self.open_positions.items()):
            symbol = position.symbol
            current_price = prices.get(symbol)

            if current_price is None:
                logger.warning(f"No price available for {symbol}")
                continue

            # 1. Check break-even first
            be_result = self._check_break_even(ticket, current_price)
            if be_result.success and be_result.new_sl is not None:
                results.append(be_result)
                continue  # Don't double-modify

            # 2. Check regime shift
            if regime_state:
                rs_result = self._check_regime_shift(ticket, current_price, regime_state)
                if rs_result.success and rs_result.new_sl is not None:
                    results.append(rs_result)
                elif not rs_result.success:
                    results.append(rs_result)

        return results

    def _fetch_regime_from_redis(self) -> RegimeState:
        """
        Fetch current regime state from Redis cache.

        Redis key format: sentinel:regime:{symbol}

        Returns:
            RegimeState with current regime values
        """
        if not self._redis:
            return RegimeState()

        try:
            # Try M5, H1, H4 keys
            m5_regime = self._redis.get(f"sentinel:regime:{self._symbol}:M5")
            h1_regime = self._redis.get(f"sentinel:regime:{self._symbol}:H1")
            h4_regime = self._redis.get(f"sentinel:regime:{self._symbol}:H4")
            chaos_signal = self._redis.get(f"sentinel:regime:{self._symbol}:CHAOS")

            regime_state = RegimeState(
                m5_regime=m5_regime.decode() if m5_regime else None,
                h1_regime=h1_regime.decode() if h1_regime else None,
                h4_regime=h4_regime.decode() if h4_regime else None,
                chaos_signal=chaos_signal == b'true' if chaos_signal else False
            )

            logger.debug(
                f"Fetched regime from Redis: M5={regime_state.m5_regime}, "
                f"H1={regime_state.h1_regime}, H4={regime_state.h4_regime}, "
                f"CHAOS={regime_state.chaos_signal}"
            )

            return regime_state

        except Exception as e:
            logger.error(f"Error fetching regime from Redis: {e}")
            return RegimeState()

    # =========================================================================
    # Startup: Load Open Positions
    # =========================================================================

    def load_open_positions_from_mt5(self) -> int:
        """
        Load all open positions from MT5 via ZMQ on startup.

        Called once at engine startup to resume monitoring existing positions.

        Returns:
            Number of positions loaded
        """
        if not self._mt5:
            logger.warning("MT5 adapter not available, cannot load positions")
            return 0

        try:
            positions = self._mt5.get_positions()

            for pos in positions:
                ticket = pos.get('ticket')
                if not ticket:
                    continue

                symbol = pos.get('symbol', self._symbol)
                direction = pos.get('type', 'buy').upper()
                entry_price = pos.get('price', 0.0)
                current_sl = pos.get('sl', pos.get('stop_loss', 0.0))
                volume = pos.get('volume', 0.0)

                # Calculate stop distance in pips
                if entry_price > 0 and current_sl > 0:
                    pip_size = PIP_SIZE.get(symbol, DEFAULT_PIP_SIZE)
                    if direction == 'BUY':
                        stop_distance_pips = (entry_price - current_sl) / pip_size
                    else:
                        stop_distance_pips = (current_sl - entry_price) / pip_size
                else:
                    stop_distance_pips = 20.0  # Default

                self.register_position(
                    ticket=ticket,
                    symbol=symbol,
                    entry_price=entry_price,
                    current_sl=current_sl,
                    side=direction,
                    stop_distance_pips=abs(stop_distance_pips),
                    entry_time=datetime.now(timezone.utc)
                )

            logger.info(f"Loaded {len(positions)} open positions from MT5")
            return len(positions)

        except Exception as e:
            logger.error(f"Error loading positions from MT5: {e}")
            return 0

    # =========================================================================
    # Price Updates (for ticker)
    # =========================================================================

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update current price for a symbol.

        Called by price ticker to keep position prices current.

        Args:
            symbol: Trading symbol
            price: Current price
        """
        for position in self.open_positions.values():
            if position.symbol == symbol:
                position.update_current_price(price)
