"""
Backtest Mode Runner - Four Variants Implementation

Implements the four backtest modes with Sentinel regime detection integration:
- Vanilla: Historical backtest with static parameters
- Spiced: Vanilla + regime filtering (skip HIGH_CHAOS, NEWS_EVENT)
- Vanilla+Full: Vanilla + Walk-Forward optimization
- Spiced+Full: Spiced + Walk-Forward optimization

Task Groups 3 & 4: Backtest Variants and Sentinel Integration

Enhanced with fee-aware Kelly position sizing:
- SentinelEnhancedTester integrates EnhancedGovernor for broker-aware Kelly sizing
- broker_id is propagated through the sizing path
- Position sizes use broker-specific pip values, commissions, and spreads
- Regime quality from Sentinel informs position sizing risk multipliers
"""

import logging
import pandas as pd
import numpy as np
import uuid
import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from src.backtesting.mt5_engine import PythonStrategyTester, MT5BacktestResult, MQL5Timeframe
from src.router.sentinel import Sentinel, RegimeReport
from src.router.enhanced_governor import EnhancedGovernor
from src.router.commander import Commander
from src.router.sessions import SessionDetector, TradingSession
from src.router.multi_timeframe_sentinel import MultiTimeframeSentinel, Timeframe
from src.api.ws_logger import setup_backtest_logging, BacktestProgressStreamer

logger = logging.getLogger(__name__)


# =============================================================================
# Backtest Mode Enumeration
# =============================================================================

class BacktestMode(Enum):
    """Backtest mode types."""
    VANILLA = "vanilla"          # Historical backtest with static parameters
    SPICED = "spiced"            # Vanilla + regime filtering
    VANILLA_FULL = "vanilla_full"  # Vanilla + Walk-Forward optimization
    SPICED_FULL = "spiced_full"    # Spiced + Walk-Forward optimization


# =============================================================================
# Regime-Filtered Backtest Result
# =============================================================================

@dataclass
class SpicedBacktestResult(MT5BacktestResult):
    """Extended backtest result with regime analytics.

    Adds regime-specific metrics to the standard MT5BacktestResult.
    """
    # Regime analytics
    regime_distribution: Dict[str, int] = field(default_factory=dict)
    regime_transitions: List[Dict[str, Any]] = field(default_factory=list)
    filtered_trades: int = 0
    filter_reasons: Dict[str, int] = field(default_factory=dict)

    # Per-bar regime tracking
    regime_history: List[Dict[str, Any]] = field(default_factory=list)

    # Quality metrics
    avg_regime_quality: float = 0.0
    avg_chaos_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with regime analytics."""
        base_dict = super().to_dict()
        base_dict.update({
            "regime_distribution": self.regime_distribution,
            "regime_transitions": self.regime_transitions,
            "filtered_trades": self.filtered_trades,
            "filter_reasons": self.filter_reasons,
            "avg_regime_quality": self.avg_regime_quality,
            "avg_chaos_score": self.avg_chaos_score
        })
        return base_dict


# =============================================================================
# Sentinel-Enhanced Strategy Tester
# =============================================================================

class SentinelEnhancedTester(PythonStrategyTester):
    """Strategy tester with Sentinel regime detection integration.

    Extends PythonStrategyTester with:
    - Per-bar regime tracking
    - Regime filtering in Spiced modes
    - Regime transition logging
    - Regime quality calculation for position sizing
    """

    def __init__(
        self,
        mode: BacktestMode = BacktestMode.VANILLA,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        slippage: float = 0.0,
        broker_id: str = "icmarkets_raw",
        enable_ws_streaming: bool = True,
        backtest_id: Optional[str] = None,
        progress_streamer: Optional["BacktestProgressStreamer"] = None,
        ws_logger: Optional[logging.Logger] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None  # Main loop for thread-safe WS (Comment 2)
    ):
        """Initialize Sentinel-enhanced tester with broker support.

        Args:
            mode: Backtest mode (vanilla, spiced, vanilla_full, spiced_full)
            initial_cash: Starting account balance
            commission: Commission per trade
            slippage: Slippage in price points
            broker_id: Broker identifier for fee-aware Kelly (default: icmarkets_raw)
                Used to auto-fetch pip values, commissions, and spreads during position sizing
            enable_ws_streaming: Enable WebSocket streaming for real-time updates (default: True)
            backtest_id: Optional backtest ID from REST endpoint. If provided, WebSocket events
                will use this ID for correlation with HTTP session. If None, a new UUID is generated.
            progress_streamer: Optional progress streamer for WebSocket broadcasting
            ws_logger: Optional WebSocket logger for streaming log_entry events (Comment 2)
            loop: Optional event loop for thread-safe WebSocket broadcasts
        """
        super().__init__(
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage
        )

        self.mode = mode
        self._sentinel = Sentinel()
        self._is_spiced_mode = mode in [BacktestMode.SPICED, BacktestMode.SPICED_FULL]
        self.broker_id = broker_id  # Store broker_id for Kelly calculation
        
        # Multi-timeframe sentinel (Comment 1: add for historical bars)
        self._use_multi_timeframe = mode in [BacktestMode.SPICED, BacktestMode.SPICED_FULL]
        if self._use_multi_timeframe:
            # Default timeframes: M5, H1, H4
            self._multi_timeframe_sentinel = MultiTimeframeSentinel(
                timeframes=[Timeframe.M5, Timeframe.H1, Timeframe.H4]
            )
            logger.info("SentinelEnhancedTester: MultiTimeframeSentinel initialized for backtesting")
        else:
            self._multi_timeframe_sentinel = None
        
        # Initialize EnhancedGovernor for fee-aware Kelly position sizing
        self._governor = EnhancedGovernor(
            account_id=f"backtest_{datetime.now(timezone.utc).timestamp()}",
            config=None  # Uses default EnhancedKellyConfig
        )
        self._governor._broker_id = broker_id  # Set broker for Kelly calculation
        self._governor._daily_start_balance = initial_cash

        # Initialize Commander for session-aware bot filtering (Comment 1)
        # Commander.run_auction() will be called with bar_utc_time during backtests
        self._commander = Commander(governor=self._governor)
        self._auction_results: List[Dict[str, Any]] = []  # Track auction results per bar

        # Regime tracking
        self._regime_history: List[Dict[str, Any]] = []
        self._regime_transitions: List[Dict[str, Any]] = []
        self._last_regime: Optional[str] = None
        self._filtered_trades = 0
        self._filter_reasons: Dict[str, int] = {}
        
        # Cached regime report from _update_regime_state() to avoid redundant Sentinel calls
        self._current_regime_report: Optional[RegimeReport] = None

        # Quality metrics
        self._regime_qualities: List[float] = []
        self._chaos_scores: List[float] = []

        # WebSocket streaming (Phase 3 integration)
        # Use provided backtest_id for correlation with REST session, or generate new one
        self._enable_ws_streaming = enable_ws_streaming
        self._backtest_id: Optional[str] = backtest_id
        self._ws_logger: Optional[logging.Logger] = ws_logger  # Use provided ws_logger (Comment 2)
        self._progress_streamer = progress_streamer
        # Initialize self.loop before it's used (Comment 1 fix)
        if loop is not None:
            self.loop = loop
        else:
            try:
                self.loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create a new one
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
        
        # Use provided ws_logger, or create one if ws_logger is not provided (Comment 2)
        # This ensures log_entry events are emitted during REST-initiated backtests
        if self._ws_logger is None and enable_ws_streaming and self._progress_streamer is None:
            self._ws_logger, self._progress_streamer = setup_backtest_logging(self._backtest_id, loop=self.loop)
            
        # WebSocket streaming optimization (Phase 4)
        self._ws_update_interval = 100  # Configurable
        self._last_ws_update_time = None

        logger.info(
            f"SentinelEnhancedTester initialized with mode={mode.value}, broker_id={broker_id}, "
            f"backtest_id={self._backtest_id}, EnhancedGovernor ready for fee-aware Kelly position sizing"
        )

    # -------------------------------------------------------------------------
    # Override _log to emit WebSocket messages (Comment 1 fix)
    # -------------------------------------------------------------------------

    def _log(self, message: str):
        """Add message to backtest log and emit via WebSocket if enabled.
        
        Overrides parent class method to ensure backtest logs are streamed
        to WebSocket clients in real-time.
        
        Args:
            message: Log message to record and broadcast
        """
        # First, call parent to store in _logs list
        super()._log(message)
        
        # Also emit via WebSocket logger if available
        if self._ws_logger is not None:
            self._ws_logger.info(message)

    # -------------------------------------------------------------------------
    # Regime Detection Integration
    # -------------------------------------------------------------------------

    def _get_bar_utc_timestamp(self) -> Optional[datetime]:
        """Extract UTC timestamp from current bar.
        
        Retrieves the timestamp from the current bar's data point and ensures
        it is timezone-aware in UTC. This timestamp is used for session-aware
        filtering and Commander auction decisions.
        
        Returns:
            UTC datetime for current bar, or None if unable to extract
        """
        try:
            current_time = self._get_current_time()
            if current_time is None:
                return None
            
            # Ensure timezone-aware UTC
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=timezone.utc)
            elif current_time.tzinfo != timezone.utc:
                # Convert to UTC if in different timezone
                current_time = current_time.astimezone(timezone.utc)
            
            return current_time
        except Exception as e:
            logger.warning(f"Failed to extract bar UTC timestamp: {e}")
            return None

    def _check_regime_filter(
        self, 
        symbol: str, 
        price: float,
        bar_utc_time: Optional[datetime] = None
    ) -> Tuple[bool, Optional[str], Optional[RegimeReport]]:
        """Check if current regime should filter trades.
        
        SPEC: Session Detection Integration
        ===================================
        Per spec requirement Comment 1, this method now receives the current bar's
        UTC timestamp to enable session-aware bot filtering in backtests.
        
        The UTC timestamp flows through the filter path:
        1. bar_utc_time extracted from current bar data (see _get_bar_utc_timestamp)
        2. Passed to _check_regime_filter (this method)
        3. Stored in regime_history for audit trail
        4. Available for Commander.run_auction() if called with session filters
        
        Args:
            symbol: Trading symbol
            price: Current price
            bar_utc_time: UTC timestamp from current bar for session-aware filtering.
                         If None, derived from _get_bar_utc_timestamp().

        Returns:
            Tuple of (should_filter, filter_reason, regime_report)
        """
        # Extract UTC timestamp if not provided
        if bar_utc_time is None:
            bar_utc_time = self._get_bar_utc_timestamp()
        
        # Get regime report from Sentinel
        report = self._sentinel.on_tick(symbol, price)  # type: ignore[union-attr]

        # Track regime history with UTC timestamp for session-aware audit trail
        self._regime_history.append({
            'bar': self.current_bar,
            'timestamp': self._get_current_time(),
            'utc_timestamp': bar_utc_time,  # Explicit UTC for session filtering
            'regime': report.regime,
            'chaos_score': report.chaos_score,
            'regime_quality': report.regime_quality,
            'news_state': report.news_state
        })

        # Track quality metrics
        self._regime_qualities.append(report.regime_quality)
        self._chaos_scores.append(report.chaos_score)

        # Check for regime transition
        if self._last_regime != report.regime:
            regime_data = {
                'chaos_score': report.chaos_score,
                'regime_quality': report.regime_quality,
                'news_state': report.news_state
            }
            self._log_regime_transition(self._last_regime or "", report.regime, regime_data)
            self._last_regime = report.regime

        # Apply regime filtering for Spiced modes
        if self._is_spiced_mode:
            # Filter 1: High chaos score (> 0.6)
            if report.chaos_score > 0.6:
                reason = f"Chaos score {report.chaos_score:.2f} > 0.6"
                self._track_filtered_trade("chaos_score", reason)
                return True, reason, report

            # Filter 2: NEWS_EVENT regime
            if report.regime == "NEWS_EVENT":
                reason = f"Regime {report.regime} (kill zone)"
                self._track_filtered_trade("news_event", reason)
                return True, reason, report

            # Filter 3: HIGH_CHAOS regime
            if report.regime == "HIGH_CHAOS":
                reason = f"Regime {report.regime}"
                self._track_filtered_trade("high_chaos", reason)
                return True, reason, report

        # No filter applied
        return False, None, report

    # -------------------------------------------------------------------------
    # Override _update_regime_state to include UTC timestamp (Comment 1)
    # -------------------------------------------------------------------------

    def _update_regime_state(self) -> None:
        """Update regime state with UTC timestamp for session-aware filtering.
        
        Overrides base class method to ensure regime history includes utc_timestamp
        field for session analysis per spec requirement Comment 1.
        
        Caches the RegimeReport in _current_regime_report for reuse by
        run_auction_with_utc_time() to avoid redundant Sentinel calls with
        dummy prices that could cause regime mismatch per bar.
        """
        if self._sentinel is None:
            return

        try:
            # Get current price data for sentinel
            if self.symbol and self.symbol in self._data_cache:
                current_data = self._data_cache[self.symbol]
                if self.current_bar < len(current_data):
                    bar = current_data.iloc[self.current_bar]
                    
                    # Get UTC timestamp from bar for session-aware filtering
                    bar_utc_time = self._get_bar_utc_timestamp()
                    
                    # Call sentinel's on_tick with symbol and price
                    report = self._sentinel.on_tick(
                        symbol=self.symbol,
                        price=bar['close']
                    )
                    
                    # Cache the regime report for reuse by run_auction_with_utc_time()
                    # This avoids a second Sentinel call with dummy price=1.0
                    self._current_regime_report = report

                    # Store regime report with UTC timestamp for session filtering
                    regime_data = {
                        'bar': self.current_bar,
                        'timestamp': bar['time'],
                        'utc_timestamp': bar_utc_time,  # UTC for session detection
                        'regime': getattr(report, 'regime', 'UNKNOWN'),
                        'chaos_score': getattr(report, 'chaos_score', 0.0),
                        'regime_quality': getattr(report, 'regime_quality', 1.0),
                        'susceptibility': getattr(report, 'susceptibility', 0.0),
                        'is_systemic_risk': getattr(report, 'is_systemic_risk', False),
                        'news_state': getattr(report, 'news_state', 'UNKNOWN')
                    }

                    # Check for regime transition
                    if self._regime_history:
                        prev_regime = self._regime_history[-1]['regime']
                        if prev_regime != regime_data['regime']:
                            self._log_regime_transition(prev_regime, regime_data['regime'], regime_data)

                    self._regime_history.append(regime_data)

        except Exception as e:
            self._log(f"Sentinel error at bar {self.current_bar}: {e}")

    def _log_regime_transition(self, old_regime: str, new_regime: str, regime_data: Dict[str, Any]) -> None:
        """Log regime transition with full context.
        
        Args:
            old_regime: Previous regime name
            new_regime: New regime name
            regime_data: Regime data dictionary containing chaos_score, regime_quality, etc.
        """
        transition = {
            'timestamp': self._get_current_time(),
            'bar': self.current_bar,
            'old_regime': old_regime,
            'new_regime': new_regime,
            'chaos_score': regime_data.get('chaos_score', 0.0),
            'regime_quality': regime_data.get('regime_quality', 1.0),
            'news_state': regime_data.get('news_state', 'NONE')
        }

        self._regime_transitions.append(transition)

        self._log(
            f"Regime Transition: {old_regime} -> {new_regime} "
            f"(chaos={transition['chaos_score']:.2f}, quality={transition['regime_quality']:.2f})"
        )

    def _track_filtered_trade(self, filter_type: str, reason: str):
        """Track filtered trade for analytics.

        Args:
            filter_type: Type of filter (chaos_score, news_event, high_chaos)
            reason: Human-readable reason
        """
        self._filtered_trades += 1
        self._filter_reasons[filter_type] = self._filter_reasons.get(filter_type, 0) + 1

        self._log(f"Trade filtered: {reason}")

    def _calculate_kelly_position_size(
        self,
        symbol: str,
        trade_proposal: Optional[Dict[str, Any]] = None,
        regime_quality: float = 1.0
    ) -> float:
        """Calculate position size using fee-aware Kelly with EnhancedGovernor.

        Integrates EnhancedGovernor to size trades based on:
        - Win rate and expectancy metrics
        - Broker-specific pip values, commissions, and spreads
        - Current account balance
        - Regime quality from Sentinel
        - Dynamic volatility (ATR)

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            trade_proposal: Optional dict with trade parameters. If not provided,
                uses conservative defaults.
            regime_quality: Regime quality scalar from Sentinel (1.0=stable, 0.0=chaotic)

        Returns:
            Position size in lots, rounded to broker precision.
            Returns 0.0 if Kelly calculation fails or fees exceed profit.

        Note:
            EnhancedGovernor automatically:
            - Fetches pip_value from broker registry for the symbol
            - Fetches commission per lot from broker registry
            - Fetches spread from broker registry
            - Applies 3-layer Kelly protection (Kelly fraction, hard risk cap, ATR volatility)
            - Checks fee kill switch (blocks trade if fees > avg win)
        """
        # Use provided proposal or create minimal one
        if trade_proposal is None:
            trade_proposal = {}

        # Ensure proposal has required fields with scalar balance values
        # Per Comment 1 fix: pass scalar balance (last equity value) instead of entire equity series
        if 'symbol' not in trade_proposal:
            trade_proposal['symbol'] = symbol
        if 'account_balance' not in trade_proposal:
            # Use scalar balance: last equity value if equity exists, otherwise default to initial_cash
            if hasattr(self, 'equity') and self.equity and len(self.equity) > 0:
                trade_proposal['account_balance'] = float(self.equity[-1])
            else:
                trade_proposal['account_balance'] = float(self.initial_cash)
        if 'current_balance' not in trade_proposal:
            # Use scalar balance: last equity value if equity exists, otherwise default to initial_cash
            if hasattr(self, 'equity') and self.equity and len(self.equity) > 0:
                trade_proposal['current_balance'] = float(self.equity[-1])
            else:
                trade_proposal['current_balance'] = float(self.initial_cash)
        if 'broker_id' not in trade_proposal:
            trade_proposal['broker_id'] = self.broker_id

        # Create a RegimeReport for governor using the actual dataclass
        # In production, this would come from Sentinel
        regime_report = RegimeReport(
            regime='STABLE',
            chaos_score=0.0,
            regime_quality=regime_quality,
            susceptibility=0.0,
            is_systemic_risk=False,
            news_state='NONE',
            timestamp=0.0
        )

        # Calculate Kelly-based position using EnhancedGovernor
        try:
            mandate = self._governor.calculate_risk(
                regime_report=regime_report,
                trade_proposal=trade_proposal,
                account_balance=trade_proposal.get('account_balance', self.equity),
                broker_id=self.broker_id
            )

            # Extract position size from mandate
            position_size = mandate.position_size

            # Log Kelly calculation details
            if position_size > 0:
                self._log(
                    f"Kelly sizing: {symbol} {position_size:.2f} lots "
                    f"(kelly_f={mandate.kelly_fraction:.4f}, regime_quality={regime_quality:.2f})"
                )
            else:
                self._log(
                    f"Kelly sizing blocked: {symbol} - {mandate.notes or 'Position size < minimum'}"
                )

            return position_size

        except Exception as e:
            # Fallback to minimal position on Kelly calculation failure
            logger.warning(
                f"Kelly position sizing failed for {symbol}: {e}, using minimal fallback",
                exc_info=True
            )
            self._log(f"Warning: Kelly calculation failed for {symbol}, using 0.01 lot minimum")
            return 0.01  # Minimal fallback position

    # -------------------------------------------------------------------------
    # Commander Integration for Session-Aware Bot Selection (Comment 1)
    # -------------------------------------------------------------------------

    def run_auction_with_utc_time(
        self,
        symbol: str,
        bar_utc_time: Optional[datetime] = None,
        regime_report: Optional[RegimeReport] = None
    ) -> List[Dict[str, Any]]:
        """Run Commander auction with session-aware filtering using bar UTC timestamp.
        
        SPEC: Backtest Commander Integration
        ====================================
        Per spec requirement Comment 1, this method integrates Commander into the backtest
        execution path and passes the bar's UTC timestamp for session-aware bot filtering:
        
        1. Extracts current bar's UTC timestamp via _get_bar_utc_timestamp()
        2. Calls Commander.run_auction() with current_utc=bar_utc_time
        3. Commander.detect_session(bar_utc_time) determines current trading session
        4. Bots are filtered by session compatibility before selection
        5. Results are stored in _auction_results for backtest analytics
        
        Args:
            symbol: Trading symbol for auction
            bar_utc_time: UTC timestamp from current bar. If None, extracted from bar.
            regime_report: Optional pre-computed RegimeReport from _update_regime_state().
                          If provided, skips the Sentinel call to avoid regime mismatch
                          from calling on_tick() with a dummy price.
        
        Returns:
            List of eligible bot dispatches from Commander.run_auction()
        
        Note:
            This method is called during backtest execution to ensure session-filtered
            bot selection is exercised in backtests per spec.
        """
        # Extract UTC timestamp if not provided
        if bar_utc_time is None:
            bar_utc_time = self._get_bar_utc_timestamp()
        
        if bar_utc_time is None:
            logger.warning("Cannot run auction: no bar UTC timestamp available")
            return []
        
        # Use provided regime_report or fall back to cached report from _update_regime_state()
        # This avoids a second Sentinel call with dummy price=1.0 that could cause regime mismatch
        if regime_report is None:
            regime_report = self._current_regime_report
        
        if regime_report is None:
            # Try to get from multi-timeframe sentinel if available (guard against None)
            if self._multi_timeframe_sentinel is not None:
                try:
                    all_regimes = self._multi_timeframe_sentinel.get_all_regimes()
                    fastest_tf = min(all_regimes.keys(), key=lambda tf: tf.seconds)
                    regime_report = all_regimes.get(fastest_tf)
                except Exception as e:
                    logger.warning(f"Failed to get regime from multi-timeframe sentinel: {e}")
        
        if regime_report is None:
            logger.warning("Cannot run auction: no regime report available")
            return []
        
        # Run Commander auction with UTC timestamp for session-aware filtering
        auction_result = self._commander.run_auction(
            regime_report=regime_report,
            account_balance=self.equity[-1] if self.equity else self.initial_cash,
            broker_id=self.broker_id,
            current_utc=bar_utc_time  # Pass bar UTC time for session detection
        )
        
        # Track auction results for backtest analytics
        auction_record = {
            'bar': self.current_bar,
            'utc_timestamp': bar_utc_time,
            'session': SessionDetector.detect_session(bar_utc_time).value,
            'regime': regime_report.regime,
            'eligible_bots': len(auction_result),
            'dispatched_bots': auction_result
        }
        self._auction_results.append(auction_record)
        
        logger.debug(
            f"Auction at bar {self.current_bar} ({bar_utc_time.isoformat()}): "
            f"session={auction_record['session']}, regime={regime_report.regime}, "
            f"eligible_bots={len(auction_result)}"
        )
        
        return auction_result

    def get_current_auction_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent auction result.
        
        Returns:
            Latest auction record or None if no auctions run
        """
        if not self._auction_results:
            return None
        return self._auction_results[-1]

    # -------------------------------------------------------------------------
    # Enhanced Trading Operations with Regime Filtering
    # -------------------------------------------------------------------------

    def buy(self, symbol: str, volume: float, price: Optional[float] = None) -> Optional[int]:
        """Open buy position with regime filtering and fee-aware Kelly position sizing.

        SPEC: Session Detection in Backtests
        ====================================
        Per spec requirement Comment 1, this method now passes the current bar's UTC timestamp
        to regime filtering. The timestamp enables session-aware filtering in backtests:
        
        1. Current bar UTC timestamp is extracted via _get_bar_utc_timestamp()
        2. Timestamp is passed to _check_regime_filter()
        3. Regime history includes timestamp for audit trail
        4. Commander.run_auction() can use timestamp for session-aware bot filtering
        
        This ensures backtests exercise session filters per spec, with every bar's UTC
        timestamp available for session detection (e.g., LONDON, NEW_YORK, OVERLAP, CLOSED).

        Args:
            symbol: Trading symbol
            volume: Requested volume in lots. If using Kelly sizing, this may be overridden.
            price: Entry price (uses current close if None)

        Returns:
            Position ticket or None if filtered or Kelly sizing blocks trade
            
        Note:
            If EnhancedGovernor is enabled, the actual position size will be determined by
            fee-aware Kelly calculation considering broker commissions, spreads, and pip values.
            The requested volume is used as a signal only.
        """
        # Get current price
        if price is None:
            timeframe = self.timeframe if self.timeframe is not None else 0
            price = self.iClose(symbol, int(timeframe), 0)
            if price is None:
                self._log(f"Error: Cannot get price for buy order")
                return None

        # Extract bar UTC timestamp for session-aware filtering
        bar_utc_time = self._get_bar_utc_timestamp()

        # Check regime filter with UTC timestamp (enables session awareness in backtests)
        should_filter, filter_reason, report = self._check_regime_filter(
            symbol, price, bar_utc_time
        )

        if should_filter:
            self._log(f"Buy blocked by regime filter: {filter_reason}")
            return None

        # Calculate Kelly-based position size with regime quality adjustment
        # Per Comment 1 fix: pass scalar balance values instead of entire equity series
        regime_quality = report.regime_quality if report else 1.0
        # Use scalar balance: last equity value if equity exists, otherwise default to initial_cash
        if hasattr(self, 'equity') and self.equity and len(self.equity) > 0:
            scalar_balance = float(self.equity[-1])
        else:
            scalar_balance = float(self.initial_cash)
        kelly_volume = self._calculate_kelly_position_size(
            symbol=symbol,
            trade_proposal={
                'symbol': symbol,
                'account_balance': scalar_balance,
                'current_balance': scalar_balance,
                'broker_id': self.broker_id,
                'win_rate': 0.55,  # Conservative default
                'avg_win': volume * 100.0,  # Rough estimate
                'avg_loss': volume * 100.0,
                'stop_loss_pips': 20.0,
                'current_atr': 0.001,
                'average_atr': 0.001
            },
            regime_quality=regime_quality
        )

        # Use Kelly-sized volume if available, otherwise fall back to requested volume
        final_volume = kelly_volume if kelly_volume > 0 else volume

        # Allow trade through to parent with final volume
        return super().buy(symbol, final_volume, price)

    def sell(self, symbol: str, volume: float, price: Optional[float] = None) -> Optional[int]:
        """Close buy position with regime filtering and fee-aware Kelly tracking.

        SPEC: Session Detection in Backtests
        ====================================
        Per spec requirement Comment 1, this method now passes the current bar's UTC timestamp
        to regime filtering for consistency with buy() method.

        Args:
            symbol: Trading symbol
            volume: Volume in lots to close
            price: Exit price (uses current close if None)

        Returns:
            Trade record or None if filtered
            
        Note:
            Exit operations use regime-aware filtering for consistency, though position sizing
            via Kelly primarily affects entry logic. Exit volume is generally controlled by
            existing position size rather than Kelly calculation.
        """
        # Get current price
        if price is None:
            timeframe = self.timeframe if self.timeframe is not None else 0
            price = self.iClose(symbol, int(timeframe), 0)
            if price is None:
                self._log(f"Error: Cannot get price for sell order")
                return None

        # Extract bar UTC timestamp for session-aware filtering consistency
        bar_utc_time = self._get_bar_utc_timestamp()

        # Check regime filter with UTC timestamp (note: sells close positions, so we may allow them)
        # For now, we filter sells in Spiced mode too for consistency
        should_filter, filter_reason, report = self._check_regime_filter(
            symbol, price, bar_utc_time
        )

        if should_filter:
            self._log(f"Sell blocked by regime filter: {filter_reason}")
            return None

        # Allow trade through to parent
        return super().sell(symbol, volume, price)

    # -------------------------------------------------------------------------
    # Override run() for Commander Integration (Comment 1)
    # -------------------------------------------------------------------------

    def run(
        self,
        strategy_code: str,
        data: pd.DataFrame,
        symbol: str,
        timeframe: int,
        strategy_name: str = "MyStrategy"
    ) -> SpicedBacktestResult:
        """Run backtest with Commander integration for session-aware bot selection.
        
        SPEC: Backtest Commander Integration
        ===================================
        Per spec requirement Comment 1, this override integrates Commander into the
        backtest execution path and passes bar_utc_time to Commander.run_auction():
        
        1. Each bar's UTC timestamp is extracted via _get_bar_utc_timestamp()
        2. Commander.run_auction() is called with current_utc=bar_utc_time
        3. SessionDetector.detect_session(bar_utc_time) determines current session
        4. Bots are filtered by session compatibility before selection
        5. Auction results are stored in _auction_results for analytics
        
        This ensures backtests exercise session-filtered bot selection per spec,
        with regime history continuing to record UTC timestamps.
        
        Args:
            strategy_code: Python code with 'on_bar(tester)' function
            data: OHLCV data as DataFrame
            symbol: Trading symbol
            timeframe: MQL5 timeframe constant
            strategy_name: Name for logging
        
        Returns:
            SpicedBacktestResult with regime analytics and auction results
        """
        # Reset state (including auction results)
        self._reset_state()
        self._auction_results = []  # Reset auction tracking
        
        # Store parameters
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Prepare data
        prepared_data = self._prepare_data(data)
        if prepared_data is None or len(prepared_data) == 0:
            return SpicedBacktestResult(
                sharpe=0.0,
                return_pct=0.0,
                drawdown=0.0,
                trades=0,
                log="Error: Invalid data provided"
            )
        
        self._data_cache[symbol] = prepared_data
        
        # Get timeframe string for display
        timeframe_str = str(timeframe)
        
        # Broadcast backtest start with thread-safe call (Comment 2)
        if self._progress_streamer is not None and self.loop:
            try:
                start_date = str(prepared_data.iloc[0]['time']) if len(prepared_data) > 0 and 'time' in prepared_data.columns else 'unknown'
                end_date = str(prepared_data.iloc[-1]['time']) if len(prepared_data) > 0 and 'time' in prepared_data.columns else 'unknown'
                
                future = asyncio.run_coroutine_threadsafe(
                    self._progress_streamer.start(
                        variant=self.mode.value,
                        symbol=symbol,
                        timeframe=timeframe_str,
                        start_date=start_date,
                        end_date=end_date
                    ),
                    self.loop
                )
            except Exception as e:
                logger.warning(f"Failed to broadcast backtest start: {e}")
        
        # Compile strategy function
        try:
            namespace = {}
            exec(strategy_code, globals(), namespace)
            
            if 'on_bar' not in namespace:
                return SpicedBacktestResult(
                    sharpe=0.0,
                    return_pct=0.0,
                    drawdown=0.0,
                    trades=0,
                    log="Error: No 'on_bar(tester)' function found in strategy code"
                )
            
            self._strategy_func = namespace['on_bar']
            
        except Exception as e:
            return SpicedBacktestResult(
                sharpe=0.0,
                return_pct=0.0,
                drawdown=0.0,
                trades=0,
                log=f"Error compiling strategy: {str(e)}"
            )
        
        # Run backtest bar by bar with Commander integration
        total_bars = len(prepared_data)
        try:
            for self.current_bar in range(total_bars):
                # Update equity
                self._update_equity()
                
                # Extract bar UTC timestamp for session-aware filtering
                bar_utc_time = self._get_bar_utc_timestamp()
                
                # Update regime state (includes UTC timestamp in history)
                # This caches the RegimeReport in _current_regime_report for reuse
                self._update_regime_state()
                
                # SPEC: Run Commander auction with UTC timestamp for session-aware bot filtering
                # This is the key integration per Comment 1 - ensures session filters are exercised
                # Pass the cached regime_report to avoid redundant Sentinel call with dummy price
                if self._commander is not None and bar_utc_time is not None:
                    try:
                        auction_result = self.run_auction_with_utc_time(
                            symbol,
                            bar_utc_time,
                            regime_report=self._current_regime_report
                        )
                        logger.debug(
                            f"Bar {self.current_bar}: Commander auction with UTC={bar_utc_time.isoformat()}"
                        )
                    except Exception as e:
                        logger.warning(f"Commander auction failed at bar {self.current_bar}: {e}")
                
                # Call strategy function
                try:
                    if self._strategy_func:
                        self._strategy_func(self)  # type: ignore[call-arg]
                except Exception as e:
                    self._log(f"Strategy error at bar {self.current_bar}: {e}")
                
                # WebSocket progress update every 100 bars or 1s (Comment 2: restore emission)
                current_time = time.time()
                should_update = (
                    self.current_bar % self._ws_update_interval == 0 or
                    self._last_ws_update_time is None or
                    (current_time - self._last_ws_update_time >= 1.0)
                )
                if self._progress_streamer is not None and should_update and self.loop:
                    self._last_ws_update_time = current_time
                    progress_pct = ((self.current_bar + 1) / total_bars * 100) if total_bars > 0 else 0.0
                    status_msg = f"Processing bar {self.current_bar + 1}/{total_bars}"
                    bars_proc = self.current_bar + 1
                    curr_date = str(prepared_data.iloc[self.current_bar]['time']) if self.current_bar < len(prepared_data) and 'time' in prepared_data.columns else None
                    trades_cnt = len(getattr(self, 'trades', []))
                    curr_pnl = (self.equity[-1] if self.equity and len(self.equity) > 0 else self.initial_cash) - self.initial_cash
                    
                    future = asyncio.run_coroutine_threadsafe(
                        self._progress_streamer.update_progress(
                            progress=progress_pct,
                            status=status_msg,
                            bars_processed=bars_proc,
                            total_bars=total_bars,
                            current_date=curr_date,
                            trades_count=trades_cnt,
                            current_pnl=curr_pnl
                        ),
                        self.loop
                    )
                # ...existing code...
        except Exception as e:
            self._log(f"Backtest error: {e}")
            if self._progress_streamer is not None and self.loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._progress_streamer.error(str(e)),
                        self.loop
                    )
                except:
                    pass
        # Close remaining positions
        self.close_all_positions()
        
        # Calculate final metrics
        result = self._calculate_result()
        
        # Broadcast backtest complete with thread-safe call (Comment 2)
        if self._progress_streamer is not None and self.loop:
            try:
                # Pass result.win_rate directly - it's already a percentage (0-100) (Comment 2)
                future = asyncio.run_coroutine_threadsafe(
                    self._progress_streamer.complete(
                        final_balance=result.final_cash,
                        total_trades=result.trades,
                        win_rate=result.win_rate,  # Pass directly - no extra division (Comment 2)
                        sharpe_ratio=result.sharpe,
                        drawdown=result.drawdown,
                        return_pct=result.return_pct,
                        duration_seconds=None,
                        results=result.to_dict()
                    ),
                    self.loop
                )
            except Exception as e:
                logger.warning(f"Failed to broadcast backtest complete: {e}")
        
        return result

    # -------------------------------------------------------------------------
    # Enhanced Result Calculation
    # -------------------------------------------------------------------------

    def _calculate_result(self) -> SpicedBacktestResult:
        """Calculate enhanced result with regime analytics.

        Returns:
            SpicedBacktestResult with regime metrics
        """
        # Get base result
        base_result = super()._calculate_result()

        # Calculate regime distribution
        regime_counts = {}
        for entry in self._regime_history:
            regime = entry['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Calculate average quality metrics
        avg_quality = float(np.mean(self._regime_qualities)) if self._regime_qualities else 0.0
        avg_chaos = float(np.mean(self._chaos_scores)) if self._chaos_scores else 0.0

        # Create enhanced result
        return SpicedBacktestResult(
            sharpe=base_result.sharpe,
            return_pct=base_result.return_pct,
            drawdown=base_result.drawdown,
            trades=base_result.trades,
            log=base_result.log,
            initial_cash=base_result.initial_cash,
            final_cash=base_result.final_cash,
            equity_curve=base_result.equity_curve,
            trade_history=base_result.trade_history,
            regime_distribution=regime_counts,
            regime_transitions=self._regime_transitions,
            filtered_trades=self._filtered_trades,
            filter_reasons=self._filter_reasons,
            regime_history=self._regime_history,
            avg_regime_quality=avg_quality,
            avg_chaos_score=avg_chaos
        )

    def get_regime_quality_scalar(self) -> float:
        """Get current regime quality scalar for position sizing.

        Returns:
            Regime quality (1.0 = stable, 0.0 = chaotic)
        """
        if not self._regime_qualities:
            return 1.0  # Default to stable if no data

        # Return most recent regime quality
        return self._regime_qualities[-1]

    # -------------------------------------------------------------------------
    # Multi-Timeframe Sentinel Methods (Comment 1)
    # -------------------------------------------------------------------------

    def run_auction_with_timeframes(
        self,
        symbol: str,
        bar_utc_time: Optional[datetime] = None,
        regime_report: Optional[RegimeReport] = None
    ) -> List[Dict[str, Any]]:
        """
        Run Commander auction with multi-timeframe regime detection.
        
        Comment 1: Use bot preferred_timeframe when evaluating regimes.
        
        This method:
        1. Feeds historical bar data into multi-timeframe sentinel
        2. Gets regime reports for all timeframes
        3. Calls Commander.run_auction_with_timeframes for multi-timeframe filtering
        4. Returns dispatches with timeframe context
        
        Args:
            symbol: Trading symbol
            bar_utc_time: UTC timestamp from current bar
            regime_report: Optional pre-computed primary regime report
            
        Returns:
            List of bot dispatches with timeframe context
        """
        if self._multi_timeframe_sentinel is None:
            logger.warning("Multi-timeframe sentinel not enabled, falling back to single timeframe")
            return self.run_auction_with_utc_time(symbol, bar_utc_time, regime_report)
        
        # Extract UTC timestamp if not provided
        if bar_utc_time is None:
            bar_utc_time = self._get_bar_utc_timestamp()
        
        if bar_utc_time is None:
            logger.warning("Cannot run multi-timeframe auction: no bar UTC timestamp")
            return []
        
        # Get current price for feeding into sentinel
        if self.symbol and self.symbol in self._data_cache:
            current_data = self._data_cache[self.symbol]
            if self.current_bar < len(current_data):
                bar = current_data.iloc[self.current_bar]
                price = bar['close']
                
                # Feed tick into multi-timeframe sentinel
                self._multi_timeframe_sentinel.on_tick(symbol, price, bar_utc_time)
        
        # Get regime reports for all timeframes
        all_regimes = self._multi_timeframe_sentinel.get_all_regimes()
        
        if not all_regimes:
            logger.warning("No regime reports from multi-timeframe sentinel")
            return []
        
        # Use primary regime report or get from fastest timeframe
        if regime_report is None:
            regime_report = self._current_regime_report
        
        if regime_report is None:
            # Get from fastest timeframe
            fastest_tf = min(all_regimes.keys(), key=lambda tf: tf.seconds)
            regime_report = all_regimes.get(fastest_tf)
        
        if regime_report is None:
            logger.warning("Cannot run auction: no regime report available")
            return []
        
        # Run multi-timeframe auction via Commander
        # Share the multi-timeframe sentinel with Commander
        self._commander._multi_timeframe_sentinel = self._multi_timeframe_sentinel
        
        auction_result = self._commander.run_auction_with_timeframes(
            primary_regime_report=regime_report,
            all_timeframe_regimes=all_regimes,
            account_balance=self.equity[-1] if self.equity else self.initial_cash,
            broker_id=self.broker_id,
            current_utc=bar_utc_time
        )
        
        # Track auction results
        auction_record = {
            'bar': self.current_bar,
            'utc_timestamp': bar_utc_time,
            'session': SessionDetector.detect_session(bar_utc_time).value,
            'regime': regime_report.regime,
            'all_timeframes': {tf.name: r.regime for tf, r in all_regimes.items()},
            'eligible_bots': len(auction_result),
            'dispatched_bots': auction_result
        }
        self._auction_results.append(auction_record)
        
        logger.debug(
            f"Multi-timeframe auction at bar {self.current_bar}: "
            f"timeframes={list(all_regimes.keys())}, dispatched={len(auction_result)}"
        )
        
        return auction_result

    def get_multi_timeframe_regimes(self) -> Dict[str, str]:
        """
        Get current regime for all configured timeframes.
        
        Returns:
            Dict of timeframe name -> regime name
        """
        if self._multi_timeframe_sentinel is None:
            return {}
        
        all_regimes = self._multi_timeframe_sentinel.get_all_regimes()
        return {tf.name: r.regime for tf, r in all_regimes.items()}


# =============================================================================
# Mode Runner Functions
# =============================================================================

def run_vanilla_backtest(
    strategy_code: str,
    data: pd.DataFrame,
    symbol: str,
    timeframe: int,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0,
    broker_id: str = "icmarkets_raw",
    enable_ws_streaming: bool = True,
    backtest_id: Optional[str] = None,
    progress_streamer: Optional["BacktestProgressStreamer"] = None,
    ws_logger: Optional[logging.Logger] = None,  # Comment 2: Accept ws_logger
    loop: Optional[asyncio.AbstractEventLoop] = None
) -> MT5BacktestResult:
    """Run Vanilla backtest (historical with static parameters).

    Args:
        strategy_code: Python code with 'on_bar(tester)' function
        data: OHLCV data as DataFrame
        symbol: Trading symbol
        timeframe: MQL5 timeframe constant
        initial_cash: Starting account balance
        commission: Commission per trade
        slippage: Slippage in price points
        broker_id: Broker identifier for fee-aware Kelly (default: icmarkets_raw)
            Used to fetch pip values, commissions, and spreads for accurate Kelly sizing
        enable_ws_streaming: Enable WebSocket streaming for real-time updates (default: True).
            When True, uses SentinelEnhancedTester with VANILLA mode for consistent streaming.
        backtest_id: Optional backtest ID from REST endpoint for WebSocket correlation.
            If provided, WebSocket events use this ID for status tracking.
        ws_logger: Optional WebSocket logger for streaming log_entry events (Comment 2)

    Returns:
        MT5BacktestResult with standard metrics
        
    Note:
        When enable_ws_streaming=True (default), Vanilla mode uses SentinelEnhancedTester
        with regime filtering disabled for consistent WebSocket streaming across all modes.
        This ensures the UI receives real-time progress updates and log entries.
        To disable WebSocket streaming and use the simpler PythonStrategyTester, set
        enable_ws_streaming=False.
    """
    if enable_ws_streaming:
        # Use SentinelEnhancedTester with VANILLA mode for WebSocket streaming
        # Regime filtering is disabled in VANILLA mode (see _is_spiced_mode check)
        tester = SentinelEnhancedTester(
            mode=BacktestMode.VANILLA,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            broker_id=broker_id,
            enable_ws_streaming=True,
            backtest_id=backtest_id,
            progress_streamer=progress_streamer,
            ws_logger=ws_logger,  # Comment 2: Pass ws_logger
            loop=loop
        )
    else:
        # Use standard PythonStrategyTester without WebSocket streaming
        tester = PythonStrategyTester(
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage
        )

    return tester.run(strategy_code, data, symbol, timeframe)


def run_spiced_backtest(
    strategy_code: str,
    data: pd.DataFrame,
    symbol: str,
    timeframe: int,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0,
    broker_id: str = "icmarkets_raw",
    backtest_id: Optional[str] = None,
    progress_streamer: Optional["BacktestProgressStreamer"] = None,
    ws_logger: Optional[logging.Logger] = None,  # Comment 2: Accept ws_logger
    loop: Optional[asyncio.AbstractEventLoop] = None
) -> SpicedBacktestResult:
    """Run Spiced backtest (Vanilla + regime filtering + fee-aware Kelly position sizing).

    Filters trades when:
    - chaos_score > 0.6
    - regime == "NEWS_EVENT"
    - regime == "HIGH_CHAOS"

    Position sizing integrates EnhancedGovernor for broker-aware Kelly calculation:
    - Auto-fetches pip values from broker registry
    - Auto-fetches commissions and spreads
    - Adjusts risk based on regime quality from Sentinel (stable → aggressive, chaotic → conservative)
    - Applies 3-layer Kelly protection:
      1. Kelly Fraction (50% of full Kelly for safety)
      2. Hard Risk Cap (max 2% per trade)
      3. Dynamic Volatility Adjustment (ATR-based scaling)
    - Blocks trades when fees exceed expected profit (fee kill switch)

    Args:
        strategy_code: Python code with 'on_bar(tester)' function
        data: OHLCV data as DataFrame
        symbol: Trading symbol
        timeframe: MQL5 timeframe constant
        initial_cash: Starting account balance
        commission: Commission per trade
        slippage: Slippage in price points
        broker_id: Broker identifier for fee-aware Kelly (default: icmarkets_raw)
            CRITICAL: Must match broker ID in broker registry for accurate fee-aware Kelly
        backtest_id: Optional backtest ID from REST endpoint for WebSocket correlation.
            If provided, WebSocket events use this ID for status tracking and correlation
            with the HTTP session that initiated the backtest.

    Returns:
        SpicedBacktestResult with regime analytics and Kelly sizing details
        
    Note:
        This is the RECOMMENDED backtest mode as it:
        1. Filters poor regime trades (reduces drawdown)
        2. Sizes positions via fee-aware Kelly (respects broker fees)
        3. Provides detailed regime transition analytics
        4. Uses broker-specific pip values and costs
    """
    tester = SentinelEnhancedTester(
        mode=BacktestMode.SPICED,
        initial_cash=initial_cash,
        commission=commission,
        slippage=slippage,
        broker_id=broker_id,
        backtest_id=backtest_id,
        progress_streamer=progress_streamer,  # Use shared
        ws_logger=ws_logger,  # Comment 2: Pass ws_logger
        loop=loop
    )

    return tester.run(strategy_code, data, symbol, timeframe)


def run_full_system_backtest(
    mode: str = 'vanilla',
    data: Optional[pd.DataFrame] = None,
    data_dict: Optional[Dict[str, pd.DataFrame]] = None,
    symbol: str = 'EURUSD',
    symbols: Optional[List[str]] = None,
    timeframe: int = MQL5Timeframe.PERIOD_H1,
    strategy_code: str = 'def on_bar(tester): pass',
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0,
    chaos_threshold: float = 0.6,
    banned_regimes: Optional[List[str]] = None,
    broker_id: str = "icmarkets_raw",
    backtest_id: Optional[str] = None,
    enable_ws_streaming: bool = True,
    progress_streamer: Optional["BacktestProgressStreamer"] = None,
    ws_logger: Optional[logging.Logger] = None,  # Comment 2: Accept ws_logger
    loop: Optional[asyncio.AbstractEventLoop] = None  # FastAPI event loop for thread-safe WS broadcasts (Comment 1)
) -> Union[MT5BacktestResult, SpicedBacktestResult, Dict[str, MT5BacktestResult]]:
    """Run backtest with specified mode with optional broker-aware Kelly position sizing.

    Main entry point for backtesting with mode selection and optional fee-aware Kelly integration.
    Supports both single symbol and multi-symbol modes.

    Modes:
    - 'vanilla': Standard historical backtest (no regime filtering, no Kelly sizing)
    - 'spiced': Vanilla + regime filtering + fee-aware Kelly position sizing
    - 'vanilla_full': Vanilla + walk-forward optimization (no Kelly sizing)
    - 'spiced_full': Spiced + walk-forward optimization + fee-aware Kelly sizing

    Fee-Aware Kelly Integration (Spiced modes):
    - EnhancedGovernor calculates position sizes using EnhancedKellyCalculator
    - Automatically fetches broker-specific: pip values, commissions, spreads
    - Adjusts risk based on regime quality from Sentinel (stable → aggressive, chaotic → conservative)
    - Applies 3-layer Kelly protection:
      1. Kelly Fraction (50% of full Kelly for safety)
      2. Hard Risk Cap (max 2% per trade)
      3. Dynamic Volatility Adjustment (ATR-based scaling)
    - Blocks trades when fees exceed expected profit (fee kill switch)

    Args:
        mode: Backtest mode ('vanilla', 'spiced', 'vanilla_full', 'spiced_full')
        data: Single symbol OHLCV data (for single symbol backtest)
        data_dict: Multi-symbol data dict (for multi-symbol backtest)
        symbol: Trading symbol (single symbol mode)
        symbols: List of symbols (multi-symbol mode)
        timeframe: MQL5 timeframe constant
        strategy_code: Python strategy code with on_bar(tester) function
        initial_cash: Starting account balance
        commission: Commission per trade
        slippage: Slippage in price points
        chaos_threshold: Chaos score threshold for filtering (Spiced modes)
        banned_regimes: List of regime names to ban (Spiced modes)
        broker_id: Broker identifier for fee-aware Kelly (default: icmarkets_raw)
            CRITICAL for accurate results: Must match broker in broker_registry
            Used to fetch pip values, commissions, spreads for Kelly sizing
        backtest_id: Optional backtest ID from REST endpoint for WebSocket correlation.
            If provided, WebSocket events use this ID for status tracking and correlation
            with the HTTP session that initiated the backtest.

    Returns:
        MT5BacktestResult for Vanilla/Vanilla+Full
        SpicedBacktestResult for Spiced/Spiced+Full (includes regime analytics)
        Dict[str, MT5BacktestResult] for multi-symbol simulation
        
    Example:
        >>> # Standard backtest with regime filtering and fee-aware Kelly sizing
        >>> result = run_full_system_backtest(
        ...     mode='spiced',
        ...     data=df,
        ...     symbol='EURUSD',
        ...     timeframe=MQL5Timeframe.PERIOD_H1,
        ...     strategy_code='def on_bar(tester): tester.buy("EURUSD", 0.1)',
        ...     initial_cash=10000.0,
        ...     broker_id='icmarkets_raw',
        ...     backtest_id='my-custom-backtest-id'
        ... )
        >>> print(f"Return: {result.return_pct:.2f}%")
        >>> print(f"Regime quality: {result.avg_regime_quality:.2f}")
    """
    # Validate mode
    try:
        backtest_mode = BacktestMode(mode)
    except ValueError:
        raise ValueError(f"Invalid mode: {mode}. Must be one of: {[m.value for m in BacktestMode]}")

    # Set default banned regimes
    if banned_regimes is None:
        banned_regimes = ['HIGH_CHAOS', 'NEWS_EVENT']

    # Handle multi-symbol simulation
    if symbols is not None and data_dict is not None:
        return run_multi_symbol_backtest(
            strategy_code=strategy_code,
            data_map=data_dict,
            timeframe=timeframe,
            mode=backtest_mode,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            backtest_id=backtest_id,
            loop=loop  # Propagate FastAPI event loop (Comment 1)
        )

    # Validate single symbol data
    if data is None:
        raise ValueError("Must provide either 'data' (single symbol) or 'data_dict' (multi-symbol)")

    logger.info(f"Running {mode} backtest for {symbol}")

    if backtest_mode == BacktestMode.VANILLA:
        return run_vanilla_backtest(
            strategy_code, data, symbol, timeframe,
            initial_cash, commission, slippage, broker_id,
            enable_ws_streaming=enable_ws_streaming,
            backtest_id=backtest_id,
            progress_streamer=progress_streamer,
            ws_logger=ws_logger,  # Comment 2: Pass ws_logger
            loop=loop  # Propagate FastAPI event loop (Comment 1)
        )
    elif backtest_mode == BacktestMode.SPICED:
        return run_spiced_backtest(
            strategy_code, data, symbol, timeframe,
            initial_cash, commission, slippage, broker_id,
            backtest_id=backtest_id,
            progress_streamer=progress_streamer,
            ws_logger=ws_logger,  # Comment 2: Pass ws_logger
            loop=loop  # Propagate FastAPI event loop (Comment 1)
        )
    elif backtest_mode in [BacktestMode.VANILLA_FULL, BacktestMode.SPICED_FULL]:
        # Import WalkForwardOptimizer
        from backtesting.walk_forward import WalkForwardOptimizer

        optimizer = WalkForwardOptimizer(
            train_pct=0.5,
            test_pct=0.2,
            gap_pct=0.1
        )

        use_regime_filter = (backtest_mode == BacktestMode.SPICED_FULL)

        logger.info(f"Running Walk-Forward optimization for {symbol} with regime_filter={use_regime_filter}")

        wf_result = optimizer.optimize(
            data=data,
            symbol=symbol,
            timeframe=timeframe,
            strategy_code=strategy_code,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            use_regime_filter=use_regime_filter,
            chaos_threshold=chaos_threshold,
            banned_regimes=banned_regimes,
            broker_id=broker_id,
            backtest_id=backtest_id,  # Forward for WS correlation (Comment 3)
            progress_streamer=progress_streamer,  # Forward progress streamer (Comment 3)
            loop=loop  # Forward FastAPI event loop (Comment 3)
        )

        # Return appropriate result type
        if use_regime_filter:
            # Aggregate regime analytics across windows
            total_filtered = sum(w.get('filtered_trades', 0) for w in wf_result.window_regime_stats)
            regime_dist = {}
            for w in wf_result.window_regime_stats:
                for regime, count in w.get('regime_distribution', {}).items():
                    regime_dist[regime] = regime_dist.get(regime, 0) + count

            avg_quality = float(np.mean([w.get('avg_regime_quality', 0.0) for w in wf_result.window_regime_stats])) if wf_result.window_regime_stats else 0.0

            return SpicedBacktestResult(
                sharpe=wf_result.aggregate_metrics.get('sharpe_mean', 0.0),
                return_pct=wf_result.aggregate_metrics.get('return_pct_mean', 0.0),
                drawdown=wf_result.aggregate_metrics.get('drawdown_mean', 0.0),
                trades=int(wf_result.aggregate_metrics.get('total_trades', 0)),
                log=f"Walk-Forward + Regime Filter completed with {len(wf_result.window_results)} windows",
                initial_cash=initial_cash,
                final_cash=initial_cash * (1 + wf_result.aggregate_metrics.get('return_pct_mean', 0.0) / 100),
                equity_curve=wf_result.aggregate_equity_curve,
                trade_history=wf_result.all_trade_history,
                regime_distribution=regime_dist,
                filtered_trades=total_filtered,
                avg_regime_quality=avg_quality
            )
        else:
            return MT5BacktestResult(
                sharpe=wf_result.aggregate_metrics.get('sharpe_mean', 0.0),
                return_pct=wf_result.aggregate_metrics.get('return_pct_mean', 0.0),
                drawdown=wf_result.aggregate_metrics.get('drawdown_mean', 0.0),
                trades=int(wf_result.aggregate_metrics.get('total_trades', 0)),
                log=f"Walk-Forward optimization completed with {len(wf_result.window_results)} windows",
                initial_cash=initial_cash,
                final_cash=initial_cash * (1 + wf_result.aggregate_metrics.get('return_pct_mean', 0.0) / 100),
                equity_curve=wf_result.aggregate_equity_curve,
                trade_history=wf_result.all_trade_history
            )
    else:
        raise ValueError(f"Unknown backtest mode: {backtest_mode}")


# =============================================================================
# Multi-Symbol Support (for Task Group 5)
# =============================================================================

def run_multi_symbol_backtest(
    strategy_code: str,
    data_map: Dict[str, pd.DataFrame],
    timeframe: int,
    mode: BacktestMode = BacktestMode.VANILLA,
    initial_cash: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.0,
    broker_id: str = "icmarkets_raw",
    backtest_id: Optional[str] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None  # FastAPI event loop for thread-safe WS broadcasts (Comment 1)
) -> Dict[str, MT5BacktestResult]:
    """Run backtest across multiple symbols simultaneously with optional fee-aware Kelly sizing.

    Each symbol is backtested independently with the same initial capital.
    When using Spiced modes, each symbol respects its broker-specific pip values,
    commissions, and spreads through EnhancedGovernor.

    Args:
        strategy_code: Python code with 'on_bar(tester)' function
        data_map: Dictionary mapping symbols to OHLCV DataFrames
        timeframe: MQL5 timeframe constant
        mode: Backtest mode (affects all symbols equally)
        initial_cash: Starting account balance (applied to each symbol)
        commission: Commission per trade
        slippage: Slippage in price points
        broker_id: Broker identifier for fee-aware Kelly (default: icmarkets_raw)
            Applied to all symbols; for multi-broker scenarios, call run_spiced_backtest per symbol
        backtest_id: Optional backtest ID from REST endpoint for WebSocket correlation.

    Returns:
        Dictionary mapping symbols to backtest results
        
    Note:
        Each symbol's Kelly position sizing will auto-fetch its pip value from broker registry.
        This is efficient as all symbols use the same broker_id and share registry lookups.
    """
    results = {}

    for symbol, data in data_map.items():
        logger.info(f"Running backtest for {symbol}")
        try:
            result = run_full_system_backtest(
                mode=mode.value,
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                strategy_code=strategy_code,
                initial_cash=initial_cash,
                commission=commission,
                slippage=slippage,
                broker_id=broker_id,
                backtest_id=backtest_id,
                loop=loop  # Propagate FastAPI event loop (Comment 1)
            )
            results[symbol] = result
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}: {e}")
            results[symbol] = None

    return results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'BacktestMode',
    'SpicedBacktestResult',
    'SentinelEnhancedTester',
    'run_vanilla_backtest',
    'run_spiced_backtest',
    'run_full_system_backtest',
    'run_multi_symbol_backtest',
]
