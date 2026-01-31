"""
MT5 Engine - Python Strategy Tester with MQL5 Built-in Function Overloading

Task Group 6: MT5 Backtesting Engine

This module implements a Python-based strategy tester that:
- Overloads MQL5 built-in functions (iTime, iClose, iHigh, iLow, iVolume)
- Integrates with MetaTrader5 Python package for data retrieval
- Simulates MQL5 environment (bars, ticks, indicators)
- Calculates performance metrics (Sharpe, drawdown, return)
- Supports pandas DataFrame integration with datetime indexing
- Provides configurable parameters (initial cash, commission, slippage)

Coexists with core_engine.py without breaking changes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import logging

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
    logging.warning("MetaTrader5 package not available. Some features will be disabled.")

logger = logging.getLogger(__name__)


# =============================================================================
# MQL5 Timeframe Constants
# =============================================================================

class MQL5Timeframe:
    """MQL5 timeframe constants matching MT5 specification."""

    PERIOD_M1 = 16385   # 1 minute
    PERIOD_M5 = 16386   # 5 minutes
    PERIOD_M15 = 16387  # 15 minutes
    PERIOD_M30 = 16388  # 30 minutes
    PERIOD_H1 = 16393   # 1 hour
    PERIOD_H4 = 16396   # 4 hours
    PERIOD_D1 = 16401   # Daily
    PERIOD_W1 = 16405   # Weekly
    PERIOD_MN1 = 16408  # Monthly

    @classmethod
    def to_minutes(cls, timeframe: int) -> int:
        """Convert timeframe constant to minutes."""
        mapping = {
            cls.PERIOD_M1: 1,
            cls.PERIOD_M5: 5,
            cls.PERIOD_M15: 15,
            cls.PERIOD_M30: 30,
            cls.PERIOD_H1: 60,
            cls.PERIOD_H4: 240,
            cls.PERIOD_D1: 1440,
            cls.PERIOD_W1: 10080,
            cls.PERIOD_MN1: 43200,
        }
        return mapping.get(timeframe, 60)

    @classmethod
    def to_pandas_freq(cls, timeframe: int) -> str:
        """Convert timeframe constant to pandas frequency string."""
        mapping = {
            cls.PERIOD_M1: "1min",
            cls.PERIOD_M5: "5min",
            cls.PERIOD_M15: "15min",
            cls.PERIOD_M30: "30min",
            cls.PERIOD_H1: "1h",
            cls.PERIOD_H4: "4h",
            cls.PERIOD_D1: "1D",
            cls.PERIOD_W1: "1W",
            cls.PERIOD_MN1: "1ME",
        }
        return mapping.get(timeframe, "1h")


# =============================================================================
# Backtest Result Class
# =============================================================================

@dataclass
class MT5BacktestResult:
    """Structured result from MT5 backtest run.

    Follows the pattern from core_engine.py's BacktestResult class.
    """
    sharpe: float
    return_pct: float
    drawdown: float
    trades: int
    log: str
    initial_cash: float = 10000.0
    final_cash: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trade_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "sharpe_ratio": self.sharpe,
            "return_pct": self.return_pct,
            "max_drawdown": self.drawdown,
            "total_trades": self.trades,
            "initial_cash": self.initial_cash,
            "final_cash": self.final_cash,
            "log": self.log,
            "equity_curve": self.equity_curve,
            "trade_history": self.trade_history
        }


# =============================================================================
# Position and Trade Tracking
# =============================================================================

@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    volume: float  # Lots
    entry_price: float
    direction: str  # "buy" or "sell"
    entry_time: datetime
    ticket: int = 0


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    volume: float
    entry_price: float
    exit_price: float
    direction: str
    entry_time: datetime
    exit_time: datetime
    profit: float
    commission: float
    ticket: int = 0


# =============================================================================
# MQL5 Built-in Function Overloading
# =============================================================================

class MQL5Functions:
    """MQL5 built-in function overloading for Python environment.

    Provides Python equivalents for MQL5 functions like iTime, iClose, etc.
    """

    def __init__(self, tester: 'PythonStrategyTester'):
        self._tester = tester

    def iTime(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[datetime]:
        """Get bar opening time.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: MQL5 timeframe constant (e.g., PERIOD_H1)
            shift: Bar index (0 = current, 1 = previous, etc.)

        Returns:
            datetime of bar open time or None if invalid
        """
        data = self._tester._data_cache.get(symbol)
        if data is None or len(data) == 0:
            return None

        try:
            # Negative shift from end of array (MQL5 convention)
            idx = -1 - shift
            if -len(data) <= idx < len(data):
                time_val = data.iloc[idx]['time']
                # Ensure datetime with timezone
                if isinstance(time_val, pd.Timestamp):
                    return time_val.to_pydatetime()
                return time_val
        except (IndexError, KeyError):
            pass

        return None

    def iClose(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
        """Get bar close price.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            shift: Bar index

        Returns:
            Close price or None if invalid
        """
        data = self._tester._data_cache.get(symbol)
        if data is None or len(data) == 0:
            return None

        try:
            idx = -1 - shift
            if -len(data) <= idx < len(data):
                return float(data.iloc[idx]['close'])
        except (IndexError, KeyError):
            pass

        return None

    def iHigh(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
        """Get bar high price."""
        data = self._tester._data_cache.get(symbol)
        if data is None or len(data) == 0:
            return None

        try:
            idx = -1 - shift
            if -len(data) <= idx < len(data):
                return float(data.iloc[idx]['high'])
        except (IndexError, KeyError):
            pass

        return None

    def iLow(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
        """Get bar low price."""
        data = self._tester._data_cache.get(symbol)
        if data is None or len(data) == 0:
            return None

        try:
            idx = -1 - shift
            if -len(data) <= idx < len(data):
                return float(data.iloc[idx]['low'])
        except (IndexError, KeyError):
            pass

        return None

    def iOpen(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
        """Get bar open price."""
        data = self._tester._data_cache.get(symbol)
        if data is None or len(data) == 0:
            return None

        try:
            idx = -1 - shift
            if -len(data) <= idx < len(data):
                return float(data.iloc[idx]['open'])
        except (IndexError, KeyError):
            pass

        return None

    def iVolume(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[int]:
        """Get bar tick volume.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            shift: Bar index

        Returns:
            Tick volume or None if invalid
        """
        data = self._tester._data_cache.get(symbol)
        if data is None or len(data) == 0:
            return None

        try:
            idx = -1 - shift
            if -len(data) <= idx < len(data):
                return int(data.iloc[idx]['tick_volume'])
        except (IndexError, KeyError):
            pass

        return None


# Module-level convenience functions (MQL5 style)
def iTime(symbol: str, timeframe: int, shift: int = 0) -> Optional[datetime]:
    """Module-level iTime function for MQL5 compatibility."""
    return _default_tester.iTime(symbol, timeframe, shift) if _default_tester else None


def iClose(symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
    """Module-level iClose function for MQL5 compatibility."""
    return _default_tester.iClose(symbol, timeframe, shift) if _default_tester else None


def iHigh(symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
    """Module-level iHigh function for MQL5 compatibility."""
    return _default_tester.iHigh(symbol, timeframe, shift) if _default_tester else None


def iLow(symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
    """Module-level iLow function for MQL5 compatibility."""
    return _default_tester.iLow(symbol, timeframe, shift) if _default_tester else None


def iVolume(symbol: str, timeframe: int, shift: int = 0) -> Optional[int]:
    """Module-level iVolume function for MQL5 compatibility."""
    return _default_tester.iVolume(symbol, timeframe, shift) if _default_tester else None


# Global reference for module-level functions
_default_tester = None


# =============================================================================
# Python Strategy Tester
# =============================================================================

class PythonStrategyTester:
    """Python-based strategy tester with MQL5 environment simulation.

    Features:
    - MQL5 built-in function overloading (iTime, iClose, etc.)
    - MetaTrader5 package integration for data retrieval
    - Strategy execution in Python environment
    - Performance metrics calculation (Sharpe, drawdown, return)
    - Configurable parameters (initial cash, commission, slippage)
    - pandas DataFrame integration with datetime indexing

    Example:
        >>> tester = PythonStrategyTester(initial_cash=10000.0, commission=0.001)
        >>> data = pd.DataFrame({
        ...     'time': pd.date_range('2024-01-01', periods=100, freq='1h'),
        ...     'open': [1.10] * 100,
        ...     'high': [1.11] * 100,
        ...     'low': [1.09] * 100,
        ...     'close': [1.105] * 100,
        ...     'tick_volume': [1000] * 100
        ... })
        >>> strategy_code = '''
        ... def on_bar(tester):
        ...     if tester.current_bar == 0:
        ...         tester.buy("EURUSD", 0.01)
        ...     elif tester.current_bar == 99:
        ...         tester.close_all_positions()
        ... '''
        >>> result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)
        >>> print(f"Sharpe: {result.sharpe}, Return: {result.return_pct}%")
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
        mt5_login: Optional[int] = None,
        mt5_password: Optional[str] = None,
        mt5_server: Optional[str] = None
    ):
        """Initialize the strategy tester.

        Args:
            initial_cash: Starting account balance
            commission: Commission per trade (as fraction, e.g., 0.001 = 0.1%)
            slippage: Slippage in price points
            mt5_login: MT5 terminal login (for live data)
            mt5_password: MT5 terminal password
            mt5_server: MT5 server name
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage

        # MT5 connection
        self._mt5 = mt5 if MT5_AVAILABLE else None
        self._mt5_connected = False

        # Data cache
        self._data_cache: Dict[str, pd.DataFrame] = {}

        # MQL5 functions
        self._mql5_funcs = MQL5Functions(self)

        # Trading state
        self.symbol: Optional[str] = None
        self.timeframe: Optional[int] = None
        self.current_bar: int = 0
        self.cash: float = initial_cash
        self.equity: List[float] = []
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.position_size: float = 0.0  # Total lots in open positions
        self._ticket_counter: int = 1

        # Logs
        self._logs: List[str] = []

        # Strategy function
        self._strategy_func: Optional[Callable] = None

        # Set as default tester for module-level functions
        global _default_tester
        _default_tester = self

    # -------------------------------------------------------------------------
    # MT5 Package Integration
    # -------------------------------------------------------------------------

    def connect_mt5(self, login: Optional[int] = None, password: Optional[str] = None, server: Optional[str] = None) -> bool:
        """Connect to MetaTrader 5 terminal.

        Args:
            login: MT5 account number
            password: MT5 password
            server: Broker server name

        Returns:
            True if connection successful, False otherwise
        """
        if not MT5_AVAILABLE:
            logger.warning("MetaTrader5 package not available")
            return False

        try:
            if login and password and server:
                self._mt5_connected = mt5.initialize(login=login, password=password, server=server)
            else:
                self._mt5_connected = mt5.initialize()

            if self._mt5_connected:
                logger.info(f"Connected to MT5 terminal: {mt5.terminal_info()}")
            else:
                logger.error(f"MT5 connection failed: {mt5.last_error()}")

        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            self._mt5_connected = False

        return self._mt5_connected

    def _copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> Optional[pd.DataFrame]:
        """Copy rates from MT5 terminal by position.

        Uses mt5.copy_rates_from_pos() for OHLCV data retrieval.

        Args:
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            start_pos: Starting position (0 = most recent bar)
            count: Number of bars to retrieve

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not MT5_AVAILABLE or not self._mt5_connected:
            logger.warning("MT5 not connected, cannot copy rates")
            return None

        try:
            # Map MQL5 timeframe to MT5 constant
            mt5_timeframe = self._get_mt5_timeframe(timeframe)

            # Retrieve data
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, start_pos, count)

            if rates is None or len(rates) == 0:
                logger.error(f"Failed to copy rates for {symbol}: {mt5.last_error()}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(rates)

            # Convert timestamp to datetime
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

            # Ensure proper column names
            column_mapping = {
                'tick_volume': 'tick_volume',
                'real_volume': 'real_volume'
            }

            return df

        except Exception as e:
            logger.error(f"Error copying rates: {e}")
            return None

    def _get_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current tick data from MT5.

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary with tick data or None
        """
        if not MT5_AVAILABLE or not self._mt5_connected:
            return None

        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None

            return {
                'time': datetime.fromtimestamp(tick.time, tz=timezone.utc),
                'bid': tick.bid,
                'ask': tick.ask,
                'last': tick.last,
                'volume': tick.volume,
                'volume_real': tick.volume_real
            }
        except Exception as e:
            logger.error(f"Error getting tick: {e}")
            return None

    def _get_mt5_timeframe(self, mql5_timeframe: int) -> int:
        """Convert MQL5 timeframe constant to MT5 package constant.

        Args:
            mql5_timeframe: MQL5Timeframe constant

        Returns:
            MT5 package timeframe constant
        """
        if MT5_AVAILABLE:
            # Direct mapping for most common timeframes
            mapping = {
                MQL5Timeframe.PERIOD_M1: mt5.TIMEFRAME_M1 if hasattr(mt5, 'TIMEFRAME_M1') else 16385,
                MQL5Timeframe.PERIOD_H1: mt5.TIMEFRAME_H1 if hasattr(mt5, 'TIMEFRAME_H1') else 16393,
                MQL5Timeframe.PERIOD_D1: mt5.TIMEFRAME_D1 if hasattr(mt5, 'TIMEFRAME_D1') else 16401,
            }
            return mapping.get(mql5_timeframe, mql5_timeframe)
        return mql5_timeframe

    # -------------------------------------------------------------------------
    # MQL5 Built-in Functions (delegated to MQL5Functions)
    # -------------------------------------------------------------------------

    def iTime(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[datetime]:
        """Get bar opening time."""
        return self._mql5_funcs.iTime(symbol, timeframe, shift)

    def iClose(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
        """Get bar close price."""
        return self._mql5_funcs.iClose(symbol, timeframe, shift)

    def iHigh(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
        """Get bar high price."""
        return self._mql5_funcs.iHigh(symbol, timeframe, shift)

    def iLow(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
        """Get bar low price."""
        return self._mql5_funcs.iLow(symbol, timeframe, shift)

    def iOpen(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[float]:
        """Get bar open price."""
        return self._mql5_funcs.iOpen(symbol, timeframe, shift)

    def iVolume(self, symbol: str, timeframe: int, shift: int = 0) -> Optional[int]:
        """Get bar tick volume."""
        return self._mql5_funcs.iVolume(symbol, timeframe, shift)

    # -------------------------------------------------------------------------
    # Trading Operations
    # -------------------------------------------------------------------------

    def buy(self, symbol: str, volume: float, price: Optional[float] = None) -> Optional[int]:
        """Open a buy position.

        Args:
            symbol: Trading symbol
            volume: Volume in lots
            price: Entry price (uses current close if None)

        Returns:
            Position ticket or None
        """
        if price is None:
            price = self.iClose(symbol, self.timeframe, 0)
            if price is None:
                self._log(f"Error: Cannot get price for buy order")
                return None

        # Apply slippage
        entry_price = price + self.slippage

        position = Position(
            symbol=symbol,
            volume=volume,
            entry_price=entry_price,
            direction="buy",
            entry_time=self._get_current_time(),
            ticket=self._ticket_counter
        )

        self.positions.append(position)
        self.position_size += volume

        # Deduct margin/cost from cash
        cost = volume * 100000 * entry_price  # Standard lot = 100,000 units
        self.cash -= cost

        self._log(f"BUY {volume} lots {symbol} @ {entry_price:.5f}, Ticket: {self._ticket_counter}")

        self._ticket_counter += 1
        return position.ticket

    def sell(self, symbol: str, volume: float, price: Optional[float] = None) -> Optional[int]:
        """Open a sell position (for closing long positions).

        Args:
            symbol: Trading symbol
            volume: Volume in lots
            price: Exit price (uses current close if None)

        Returns:
            Trade record or None
        """
        if price is None:
            price = self.iClose(symbol, self.timeframe, 0)
            if price is None:
                self._log(f"Error: Cannot get price for sell order")
                return None

        # Apply slippage
        exit_price = price - self.slippage

        # Find matching buy position
        for pos in self.positions:
            if pos.symbol == symbol and pos.direction == "buy":
                # Close position
                volume_to_close = min(volume, pos.volume)

                # Calculate profit
                profit = (exit_price - pos.entry_price) * volume_to_close * 100000
                commission_cost = self.commission * volume_to_close * 100000 * exit_price

                trade = Trade(
                    symbol=symbol,
                    volume=volume_to_close,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    direction="buy",
                    entry_time=pos.entry_time,
                    exit_time=self._get_current_time(),
                    profit=profit,
                    commission=commission_cost,
                    ticket=self._ticket_counter
                )

                self._ticket_counter += 1
                self.trades.append(trade)

                # Update position
                pos.volume -= volume_to_close
                self.position_size -= volume_to_close

                # Add proceeds to cash
                proceeds = volume_to_close * 100000 * exit_price
                self.cash += proceeds - commission_cost

                self._log(f"SELL {volume_to_close} lots {symbol} @ {exit_price:.5f}, Profit: {profit:.2f}")

                # Remove position if fully closed
                if pos.volume <= 0:
                    self.positions.remove(pos)

                return trade.ticket

        self._log(f"Warning: No buy position to close for {symbol}")
        return None

    def close_all_positions(self) -> int:
        """Close all open positions.

        Returns:
            Number of positions closed
        """
        closed = 0
        for pos in list(self.positions):
            if pos.direction == "buy":
                price = self.iClose(pos.symbol, self.timeframe, 0)
                if price is not None:
                    self.sell(pos.symbol, pos.volume, price)
                    closed += 1

        return closed

    # -------------------------------------------------------------------------
    # Strategy Execution
    # -------------------------------------------------------------------------

    def run(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        symbol: str,
        timeframe: int,
        strategy_name: str = "MyStrategy"
    ) -> MT5BacktestResult:
        """Run a strategy backtest.

        Args:
            strategy_code: Python code with 'on_bar(tester)' function
            data: OHLCV data as DataFrame
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            strategy_name: Name for logging

        Returns:
            MT5BacktestResult with metrics and log

        WARNING: Executes arbitrary code. Ensure sandboxing in production.
        """
        # Reset state
        self._reset_state()

        # Store parameters
        self.symbol = symbol
        self.timeframe = timeframe

        # Prepare data
        prepared_data = self._prepare_data(data)
        if prepared_data is None or len(prepared_data) == 0:
            return MT5BacktestResult(
                sharpe=0.0,
                return_pct=0.0,
                drawdown=0.0,
                trades=0,
                log="Error: Invalid data provided"
            )

        self._data_cache[symbol] = prepared_data

        # Compile strategy function
        try:
            namespace = {}
            exec(strategy_code, globals(), namespace)

            if 'on_bar' not in namespace:
                return MT5BacktestResult(
                    sharpe=0.0,
                    return_pct=0.0,
                    drawdown=0.0,
                    trades=0,
                    log="Error: No 'on_bar(tester)' function found in strategy code"
                )

            self._strategy_func = namespace['on_bar']

        except Exception as e:
            return MT5BacktestResult(
                sharpe=0.0,
                return_pct=0.0,
                drawdown=0.0,
                trades=0,
                log=f"Error compiling strategy: {str(e)}"
            )

        # Run backtest bar by bar
        try:
            for self.current_bar in range(len(prepared_data)):
                # Update equity
                self._update_equity()

                # Call strategy function
                try:
                    self._strategy_func(self)
                except Exception as e:
                    self._log(f"Strategy error at bar {self.current_bar}: {e}")

        except Exception as e:
            self._log(f"Backtest error: {e}")

        # Close remaining positions
        self.close_all_positions()

        # Calculate final metrics
        return self._calculate_result()

    def _prepare_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare data for backtesting.

        Ensures DataFrame has proper structure and datetime handling.

        Args:
            data: Input DataFrame

        Returns:
            Prepared DataFrame or None
        """
        if data is None or len(data) == 0:
            return None

        df = data.copy()

        # Handle datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            df['time'] = df.index
        elif 'time' not in df.columns:
            # Create time column if missing
            df['time'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1h')

        # Ensure datetime is timezone-aware
        if df['time'].dt.tz is None:
            df['time'] = df['time'].dt.tz_localize('utc')

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'tick_volume']
        for col in required_cols:
            if col not in df.columns:
                # Use reasonable defaults
                if col == 'tick_volume':
                    df[col] = 1000
                else:
                    df[col] = df.get('close', 1.0)

        return df

    def _update_equity(self):
        """Update equity curve based on current positions and prices."""
        total_equity = self.cash

        for pos in self.positions:
            if pos.direction == "buy":
                current_price = self.iClose(pos.symbol, self.timeframe, 0)
                if current_price is not None:
                    unrealized_pnl = (current_price - pos.entry_price) * pos.volume * 100000
                    total_equity += unrealized_pnl + pos.volume * 100000 * pos.entry_price

        self.equity.append(total_equity)

    def _get_current_time(self) -> datetime:
        """Get current bar time."""
        if self.symbol in self._data_cache:
            data = self._data_cache[self.symbol]
            if 0 <= self.current_bar < len(data):
                time_val = data.iloc[self.current_bar]['time']
                if isinstance(time_val, pd.Timestamp):
                    return time_val.to_pydatetime()
                return time_val
        return datetime.now(timezone.utc)

    def _reset_state(self):
        """Reset tester state for new backtest."""
        self.cash = self.initial_cash
        self.equity = [self.initial_cash]
        self.positions = []
        self.trades = []
        self.position_size = 0.0
        self._ticket_counter = 1
        self._logs = []
        self._data_cache = {}

    def _log(self, message: str):
        """Add message to backtest log."""
        self._logs.append(message)

    # -------------------------------------------------------------------------
    # Performance Metrics Calculation
    # -------------------------------------------------------------------------

    def _calculate_result(self) -> MT5BacktestResult:
        """Calculate final backtest result with metrics.

        Returns:
            MT5BacktestResult with calculated metrics
        """
        if len(self.equity) < 2:
            return MT5BacktestResult(
                sharpe=0.0,
                return_pct=0.0,
                drawdown=0.0,
                trades=len(self.trades),
                log="\n".join(self._logs),
                initial_cash=self.initial_cash,
                final_cash=self.cash,
                equity_curve=self.equity.copy(),
                trade_history=[t.__dict__ for t in self.trades]
            )

        # Calculate returns
        returns = np.diff(self.equity) / self.equity[:-1]

        # Calculate metrics
        sharpe = self._calculate_sharpe(returns)
        max_dd = self._calculate_max_drawdown(np.array(self.equity))
        total_return = self._calculate_total_return(np.array(self.equity))

        # Build trade history
        trade_history = [t.__dict__ for t in self.trades]

        return MT5BacktestResult(
            sharpe=sharpe,
            return_pct=total_return,
            drawdown=max_dd,
            trades=len(self.trades),
            log="\n".join(self._logs),
            initial_cash=self.initial_cash,
            final_cash=self.equity[-1] if self.equity else self.cash,
            equity_curve=self.equity.copy(),
            trade_history=trade_history
        )

    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio.

        Args:
            returns: Array of periodic returns
            risk_free_rate: Annual risk-free rate (default 0)

        Returns:
            Sharpe ratio (annualized)
        """
        if len(returns) == 0:
            return 0.0

        # Remove NaN values
        returns = returns[~np.isnan(returns)]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Calculate Sharpe (assuming hourly data, annualize with sqrt(252*4))
        excess_returns = returns - (risk_free_rate / 252 / 4)  # Adjust for hourly
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 4)

        return float(sharpe)

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """Calculate maximum drawdown as percentage.

        Args:
            equity: Array of equity values

        Returns:
            Maximum drawdown percentage
        """
        if len(equity) == 0:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Calculate drawdown at each point
        drawdown = (equity - running_max) / running_max * 100

        # Return maximum drawdown
        return float(np.min(drawdown))

    def _calculate_total_return(self, equity: np.ndarray) -> float:
        """Calculate total return as percentage.

        Args:
            equity: Array of equity values

        Returns:
            Total return percentage
        """
        if len(equity) < 2:
            return 0.0

        initial = equity[0]
        final = equity[-1]

        if initial == 0:
            return 0.0

        return float((final - initial) / initial * 100)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'PythonStrategyTester',
    'MT5BacktestResult',
    'MQL5Timeframe',
    'MQL5Functions',
    'iTime',
    'iClose',
    'iHigh',
    'iLow',
    'iVolume',
    'Position',
    'Trade',
]
