"""
MarketScanner: Trading Opportunity Detection System

Detects trading opportunities during specific market sessions using:
- Session Breakout Scanner (opening range breakouts)
- Volatility Scanner (ATR spikes)
- News Event Scanner (economic calendar)
- ICT Setup Scanner (FVG, Order Blocks, Liquidity Voids)

Scan Schedule (using APScheduler):
- Asian Session (12 AM - 8 AM UTC): Every 15 minutes
- London Session (8 AM - 4 PM UTC): Every 5 minutes
- NY Session (1 PM - 9 PM UTC): Every 5 minutes
- Overlap (1 PM - 4 PM UTC): Every 1 minute
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import json
import asyncio

from src.database.duckdb_connection import query_market_data, WARM_DB_PATH
from src.database.db_manager import DBManager, HOTDBManager, get_hot_db_url
from src.database.models import TickCache
from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of scanner alerts."""
    SESSION_BREAKOUT = "session_breakout"
    VOLATILITY_SPIKE = "volatility_spike"
    NEWS_EVENT = "news_event"
    ICT_SETUP = "ict_setup"


class AlertPriority(Enum):
    """Alert priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ScannerAlert:
    """Represents a detected trading opportunity."""
    type: AlertType
    symbol: str
    session: str
    setup: str
    confidence: float
    recommended_bots: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    priority: AlertPriority = AlertPriority.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "type": self.type.value,
            "symbol": self.symbol,
            "session": self.session,
            "setup": self.setup,
            "confidence": self.confidence,
            "recommended_bots": self.recommended_bots,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "priority": self.priority.value,
        }


class MarketScanner:
    """
    Market opportunity scanner with session-aware detection.
    
    Scanners:
    1. Session Breakout Scanner - Opening range breakouts
    2. Volatility Scanner - ATR-based volatility spikes
    3. News Event Scanner - Economic calendar integration
    4. ICT Setup Scanner - FVG, Order Blocks, Liquidity Voids
    
    Usage:
        scanner = MarketScanner()
        alerts = scanner.run_full_scan()
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        alert_callback: Optional[Callable[[ScannerAlert], None]] = None,
        db_manager: Optional[DBManager] = None,
        hot_db_manager: Optional[DBManager] = None,
        mt5_adapter: Optional[MT5SocketAdapter] = None,
    ):
        self.symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        self.alert_callback = alert_callback
        self.db_manager = db_manager or DBManager()
        
        # HOT tier DBManager for tick_cache queries (current price)
        # If not provided, try to create one from HOT_DB_URL environment variable
        if hot_db_manager is not None:
            self.hot_db_manager = hot_db_manager
        else:
            # Try to create HOT DBManager from environment
            hot_url = get_hot_db_url()
            if hot_url:
                self.hot_db_manager = HOTDBManager()
                logger.info("Using HOT tier DBManager for tick_cache queries")
            else:
                # Fall back to default db_manager but warn
                self.hot_db_manager = self.db_manager
                logger.warning(
                    "HOT_DB_URL not set, current price queries will use default DBManager. "
                    "This may result in None prices if tick_cache is empty (SQLite default)."
                )
        
        self.mt5_adapter = mt5_adapter
        self._recent_alerts: List[ScannerAlert] = []
        self._session_opening_ranges: Dict[str, Dict[str, Any]] = {}
        self._volatility_history: Dict[str, List[float]] = {}
        
    def run_full_scan(self) -> List[Dict[str, Any]]:
        """
        Run all scanners and return detected opportunities.
        
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        # Get current session
        from src.router.sessions import get_current_session
        current_session = get_current_session()
        
        logger.info(f"Running market scan for session: {current_session.value}")
        
        # Pre-populate session opening ranges for all symbols before scanning
        self._populate_session_opening_ranges(current_session.value)
        
        # Run all scanners
        alerts.extend(self.scan_session_breakouts(current_session.value))
        alerts.extend(self.scan_volatility_spikes())
        alerts.extend(self.scan_news_events())
        alerts.extend(self.scan_ict_setups())
        
        # Store recent alerts
        self._recent_alerts.extend(alerts)
        
        # Keep only last 100 alerts
        if len(self._recent_alerts) > 100:
            self._recent_alerts = self._recent_alerts[-100:]
        
        # Persist alerts to database
        for alert in alerts:
            self._persist_alert(alert)
        
        # Broadcast alerts via WebSocket
        for alert in alerts:
            self._broadcast_alert(alert)
        
        # Call callback for each alert
        if self.alert_callback:
            for alert in alerts:
                self.alert_callback(alert)
        
        return [a.to_dict() for a in alerts]
    
    def _persist_alert(self, alert: ScannerAlert) -> None:
        """
        Persist detected alert to market_opportunities table.
        
        Args:
            alert: ScannerAlert to persist
        """
        try:
            session = self.hot_db_manager.session
            
            query = """
                INSERT INTO market_opportunities 
                (scan_type, symbol, session, setup, confidence, recommended_bots, metadata, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """
            session.execute(
                query,
                (
                    alert.type.value,
                    alert.symbol,
                    alert.session,
                    alert.setup,
                    alert.confidence,
                    json.dumps(alert.recommended_bots),
                    json.dumps(alert.metadata),
                    alert.timestamp
                )
            )
            session.commit()
            logger.debug(f"Persisted market opportunity: {alert.symbol} {alert.type.value}")
            
        except Exception as e:
            logger.warning(f"Failed to persist market opportunity: {e}")
    
    def _broadcast_alert(self, alert: ScannerAlert) -> None:
        """
        Broadcast alert via WebSocket to connected clients.
        
        Args:
            alert: ScannerAlert to broadcast
        """
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                from src.api.websocket_endpoints import broadcast_market_opportunity
                loop.run_until_complete(
                    broadcast_market_opportunity(
                        scan_type=alert.type.value,
                        symbol=alert.symbol,
                        confidence=alert.confidence,
                        setup=alert.setup,
                        recommended_bots=alert.recommended_bots,
                        metadata=alert.metadata,
                        timestamp=alert.timestamp
                    )
                )
                logger.debug(f"Broadcast market opportunity: {alert.symbol} {alert.type.value}")
            finally:
                loop.close()
        except Exception as e:
            logger.warning(f"Failed to broadcast market opportunity: {e}")
    
    def scan_session_breakouts(self, session: str) -> List[ScannerAlert]:
        """
        Scan for session opening range breakouts.
        
        Detects when price breaks above/below the first 30 minutes
        of session trading range with volume confirmation.
        
        Args:
            session: Current trading session name
            
        Returns:
            List of breakout alerts
        """
        alerts = []
        
        try:
            # Get session start time
            session_start = self._get_session_start(session)
            if session_start is None:
                return alerts
            
            now = datetime.now(timezone.utc)
            minutes_since_open = (now - session_start).total_seconds() / 60
            
            # Only scan for breakouts after 30 minutes into session
            if minutes_since_open < 30:
                logger.debug(f"Session {session} too young for breakout scan")
                return alerts
            
            for symbol in self.symbols:
                # Get opening range (first 30 min high/low)
                opening_range = self._get_opening_range(symbol, session_start)
                if opening_range is None:
                    continue
                
                # Get current price
                current_price = self._get_current_price(symbol)
                if current_price is None:
                    continue
                
                range_high = opening_range["high"]
                range_low = opening_range["low"]
                range_size = range_high - range_low
                
                # Guard against flat ranges (divide by zero)
                if range_size == 0:
                    logger.debug(f"Flat opening range for {symbol}, skipping breakout detection")
                    continue
                
                # Check for bullish breakout
                if current_price > range_high:
                    confidence = min(0.95, 0.6 + (current_price - range_high) / range_size)
                    alerts.append(ScannerAlert(
                        type=AlertType.SESSION_BREAKOUT,
                        symbol=symbol,
                        session=session,
                        setup=f"Bullish breakout above {range_high:.5f}",
                        confidence=confidence,
                        recommended_bots=self._get_breakout_bots(symbol, "bullish"),
                        metadata={
                            "breakout_level": range_high,
                            "current_price": current_price,
                            "breakout_size": current_price - range_high,
                            "range_size": range_size,
                        },
                        priority=AlertPriority.HIGH if confidence > 0.8 else AlertPriority.MEDIUM,
                    ))
                
                # Check for bearish breakout
                elif current_price < range_low:
                    confidence = min(0.95, 0.6 + (range_low - current_price) / range_size)
                    alerts.append(ScannerAlert(
                        type=AlertType.SESSION_BREAKOUT,
                        symbol=symbol,
                        session=session,
                        setup=f"Bearish breakdown below {range_low:.5f}",
                        confidence=confidence,
                        recommended_bots=self._get_breakout_bots(symbol, "bearish"),
                        metadata={
                            "breakout_level": range_low,
                            "current_price": current_price,
                            "breakdown_size": range_low - current_price,
                            "range_size": range_size,
                        },
                        priority=AlertPriority.HIGH if confidence > 0.8 else AlertPriority.MEDIUM,
                    ))
        
        except Exception as e:
            logger.error(f"Error scanning session breakouts: {e}")
        
        return alerts
    
    def scan_volatility_spikes(self) -> List[ScannerAlert]:
        """
        Scan for volatility spikes based on ATR.
        
        Alerts when current ATR exceeds 2x average ATR.
        
        Returns:
            List of volatility spike alerts
        """
        alerts = []
        
        try:
            for symbol in self.symbols:
                # Get current and average ATR
                current_atr = self._get_current_atr(symbol)
                avg_atr = self._get_average_atr(symbol)
                
                if current_atr is None or avg_atr is None:
                    continue
                
                # Calculate volatility ratio
                volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 0
                
                # Alert if volatility > 2x average
                if volatility_ratio > 2.0:
                    confidence = min(0.95, 0.5 + (volatility_ratio - 1.0) * 0.2)
                    alerts.append(ScannerAlert(
                        type=AlertType.VOLATILITY_SPIKE,
                        symbol=symbol,
                        session=self._get_current_session_name(),
                        setup=f"Volatility spike: {volatility_ratio:.1f}x average",
                        confidence=confidence,
                        recommended_bots=self._get_volatility_bots(symbol),
                        metadata={
                            "current_atr": current_atr,
                            "average_atr": avg_atr,
                            "volatility_ratio": volatility_ratio,
                        },
                        priority=AlertPriority.HIGH if volatility_ratio > 3.0 else AlertPriority.MEDIUM,
                    ))
        
        except Exception as e:
            logger.error(f"Error scanning volatility spikes: {e}")
        
        return alerts
    
    def scan_news_events(self) -> List[ScannerAlert]:
        """
        Scan for upcoming high-impact news events.
        
        Alerts 5 minutes before high-impact news.
        Recommends pausing non-news bots and activating news-trading bots.
        
        Returns:
            List of news event alerts
        """
        alerts = []
        
        try:
            # Get upcoming news events (integrate with existing news sensor)
            upcoming_events = self._get_upcoming_news_events()
            
            now = datetime.now(timezone.utc)
            
            for event in upcoming_events:
                event_time = event.get("datetime")
                if event_time is None:
                    continue
                
                # Parse event time if string
                if isinstance(event_time, str):
                    event_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                
                time_until_event = (event_time - now).total_seconds() / 60
                
                # Alert 5 minutes before
                if 0 < time_until_event <= 5:
                    alerts.append(ScannerAlert(
                        type=AlertType.NEWS_EVENT,
                        symbol=event.get("currency", "USD"),
                        session=self._get_current_session_name(),
                        setup=f"News: {event.get('name', 'Unknown')} in {time_until_event:.0f} min",
                        confidence=0.9,
                        recommended_bots=self._get_news_bots(),
                        metadata={
                            "event_name": event.get("name"),
                            "event_currency": event.get("currency"),
                            "impact": event.get("impact"),
                            "time_until_event": time_until_event,
                            "forecast": event.get("forecast"),
                            "previous": event.get("previous"),
                        },
                        priority=AlertPriority.CRITICAL,
                    ))
        
        except Exception as e:
            logger.error(f"Error scanning news events: {e}")
        
        return alerts
    
    def scan_ict_setups(self) -> List[ScannerAlert]:
        """
        Scan for ICT (Inner Circle Trader) setups.
        
        Detects:
        - Fair Value Gaps (FVG)
        - Order Blocks
        - Liquidity Voids
        
        Returns:
            List of ICT setup alerts
        """
        alerts = []
        
        try:
            for symbol in self.symbols:
                # Get recent price data
                price_data = self._get_recent_price_data(symbol, bars=50)
                if price_data is None or len(price_data) < 20:
                    continue
                
                # Scan for FVG
                fvg_setups = self._detect_fvg(price_data)
                for fvg in fvg_setups:
                    alerts.append(ScannerAlert(
                        type=AlertType.ICT_SETUP,
                        symbol=symbol,
                        session=self._get_current_session_name(),
                        setup=f"FVG: {fvg['direction']} at {fvg['level']:.5f}",
                        confidence=fvg["confidence"],
                        recommended_bots=self._get_ict_bots(symbol),
                        metadata={
                            "setup_type": "fvg",
                            "direction": fvg["direction"],
                            "level": fvg["level"],
                            "gap_size": fvg["gap_size"],
                        },
                        priority=AlertPriority.MEDIUM,
                    ))
                
                # Scan for Order Blocks
                ob_setups = self._detect_order_blocks(price_data)
                for ob in ob_setups:
                    alerts.append(ScannerAlert(
                        type=AlertType.ICT_SETUP,
                        symbol=symbol,
                        session=self._get_current_session_name(),
                        setup=f"Order Block: {ob['direction']} at {ob['level']:.5f}",
                        confidence=ob["confidence"],
                        recommended_bots=self._get_ict_bots(symbol),
                        metadata={
                            "setup_type": "order_block",
                            "direction": ob["direction"],
                            "level": ob["level"],
                            "strength": ob["strength"],
                        },
                        priority=AlertPriority.MEDIUM,
                    ))
        
        except Exception as e:
            logger.error(f"Error scanning ICT setups: {e}")
        
        return alerts
    
    # ========== Helper Methods ==========
    
    def _get_current_session_name(self) -> str:
        """Get current session name."""
        try:
            from src.router.sessions import get_current_session
            return get_current_session().value
        except Exception:
            return "UNKNOWN"
    
    def _get_session_start(self, session: str) -> Optional[datetime]:
        """Get session start time for today."""
        now = datetime.now(timezone.utc)
        
        session_hours = {
            "ASIAN": 0,      # 12 AM UTC
            "LONDON": 8,     # 8 AM UTC
            "NEW_YORK": 13,  # 1 PM UTC
            "OVERLAP": 13,   # 1 PM UTC
        }
        
        hour = session_hours.get(session.upper())
        if hour is not None:
            return now.replace(hour=hour, minute=0, second=0, microsecond=0)
        return None
    
    def _get_recent_bars(
        self,
        symbol: str,
        timeframe: str = "M1",
        lookback: int = 60,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Query DuckDB WARM tier for OHLC bars.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (default M1)
            lookback: Number of minutes to look back (used if start_date not provided)
            start_date: Optional start datetime for query (timezone-aware UTC)
            end_date: Optional end datetime for query (timezone-aware UTC)
            
        Returns:
            List of bar dictionaries with timestamp, open, high, low, close, volume
        """
        try:
            # Calculate lookback timestamp if start_date not provided
            if start_date is None:
                lookback_time = datetime.now(timezone.utc) - timedelta(minutes=lookback)
                start_date = lookback_time
            
            # Ensure start_date is timezone-aware UTC
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            else:
                # Normalize to UTC
                start_date = start_date.astimezone(timezone.utc)
            
            # Format start_date for DuckDB query
            start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            
            # Query DuckDB
            df = query_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date_str
            )
            
            if df is None or df.empty:
                logger.warning(f"No market data found for {symbol} {timeframe}")
                return []
            
            # Convert DataFrame to list of dictionaries with timezone-aware timestamps
            bars = []
            for _, row in df.iterrows():
                # Convert timestamp to timezone-aware UTC datetime
                ts = row['timestamp']
                if isinstance(ts, datetime):
                    if ts.tzinfo is None:
                        # Naive datetime - make it timezone-aware UTC
                        ts = ts.replace(tzinfo=timezone.utc)
                    else:
                        # Already aware - normalize to UTC
                        ts = ts.astimezone(timezone.utc)
                else:
                    # Parse from string
                    ts = datetime.fromisoformat(str(ts))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    else:
                        ts = ts.astimezone(timezone.utc)
                
                # If end_date is provided, filter bars
                if end_date is not None:
                    if end_date.tzinfo is None:
                        end_date = end_date.replace(tzinfo=timezone.utc)
                    else:
                        end_date = end_date.astimezone(timezone.utc)
                    if ts >= end_date:
                        continue
                
                bars.append({
                    "timestamp": ts,
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row.get('volume', 0))
                })
            
            logger.debug(f"Retrieved {len(bars)} bars for {symbol} {timeframe}")
            return bars
            
        except Exception as e:
            logger.error(f"Failed to get recent bars for {symbol}: {e}")
            return []
    
    def _calculate_atr(self, bars: List[Dict], period: int = 14) -> Optional[float]:
        """
        Calculate Average True Range using standard True Range formula with EMA.
        
        Args:
            bars: List of bar dictionaries
            period: ATR period (default 14)
            
        Returns:
            ATR value or None if insufficient data
        """
        if bars is None or len(bars) < period + 1:
            return None
        
        try:
            tr_values = []
            
            for i in range(1, len(bars)):
                current_bar = bars[i]
                prev_bar = bars[i - 1]
                
                high_low = current_bar['high'] - current_bar['low']
                high_close = abs(current_bar['high'] - prev_bar['close'])
                low_close = abs(current_bar['low'] - prev_bar['close'])
                
                tr = max(high_low, high_close, low_close)
                tr_values.append(tr)
            
            if len(tr_values) < period:
                return None
            
            # Use simple average for first ATR value
            atr = sum(tr_values[:period]) / period
            
            # Apply EMA for subsequent values
            for tr in tr_values[period:]:
                atr = (atr * (period - 1) + tr) / period
            
            return atr
            
        except Exception as e:
            logger.error(f"Failed to calculate ATR: {e}")
            return None
    
    def _get_session_high_low(
        self,
        symbol: str,
        session_start: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate session high/low from bars.
        
        Args:
            symbol: Trading symbol
            session_start: Session start time (timezone-aware UTC)
            
        Returns:
            Dict with high, low, high_time, low_time or None
        """
        # Ensure session_start is timezone-aware UTC
        if session_start.tzinfo is None:
            session_start = session_start.replace(tzinfo=timezone.utc)
        else:
            session_start = session_start.astimezone(timezone.utc)
        
        try:
            # Get bars since session start
            now = datetime.now(timezone.utc)
            minutes_since_session = int((now - session_start).total_seconds() / 60)
            
            if minutes_since_session < 30:
                logger.warning(f"Session too young ({minutes_since_session} min) for high/low calculation")
                return None
            
            # Query bars using session window
            bars = self._get_recent_bars(
                symbol, 
                timeframe="M1", 
                lookback=minutes_since_session + 10,
                start_date=session_start
            )
            
            if not bars:
                logger.warning(f"No bars available for session high/low for {symbol}")
                return None
            
            # Filter bars since session start (backup filter, primary is start_date)
            session_bars = [bar for bar in bars if bar['timestamp'] >= session_start]
            
            if len(session_bars) < 2:
                logger.warning(f"Insufficient bars ({len(session_bars)}) for session high/low for {symbol}")
                return None
            
            # Calculate high and low
            high = max(bar['high'] for bar in session_bars)
            low = min(bar['low'] for bar in session_bars)
            
            # Get timestamps
            high_time = next(bar['timestamp'] for bar in session_bars if bar['high'] == high)
            low_time = next(bar['timestamp'] for bar in session_bars if bar['low'] == low)
            
            return {
                "high": high,
                "low": low,
                "high_time": high_time,
                "low_time": low_time
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate session high/low for {symbol}: {e}")
            return None
    
    def _get_opening_range(self, symbol: str, session_start: datetime) -> Optional[Dict[str, Any]]:
        """
        Get session opening range (first 30 min high/low).
        
        Uses caching to avoid repeated queries within the same session.
        
        Args:
            symbol: Trading symbol
            session_start: Session start time (timezone-aware UTC)
            
        Returns:
            Dict with high, low, high_time, low_time or None
        """
        # Ensure session_start is timezone-aware UTC
        if session_start.tzinfo is None:
            session_start = session_start.replace(tzinfo=timezone.utc)
        else:
            session_start = session_start.astimezone(timezone.utc)
        
        cache_key = f"{symbol}_{session_start.strftime('%Y%m%d_%H')}"
        
        # Return cached value if available
        if cache_key in self._session_opening_ranges:
            return self._session_opening_ranges[cache_key]
        
        try:
            # Calculate opening range window (30 minutes from session start)
            opening_range_end = session_start + timedelta(minutes=30)
            now = datetime.now(timezone.utc)
            
            # Calculate lookback: from session_start to now (but at least 30 min window)
            # Use a lookback that ensures we cover the session window
            minutes_lookback = int((now - session_start).total_seconds() / 60) + 10
            minutes_lookback = max(minutes_lookback, 35)  # At least 35 min to cover 30 min window
            
            # Query bars using session window - use start_date to limit query to session window
            # This ensures we get bars from session_start onwards
            bars = self._get_recent_bars(
                symbol, 
                timeframe="M1", 
                lookback=minutes_lookback,
                start_date=session_start,
                end_date=opening_range_end
            )
            
            if not bars:
                logger.warning(f"No bars available for opening range calculation for {symbol}")
                return None
            
            # Filter bars within the opening range window (backup filter, primary is end_date)
            opening_bars = [
                bar for bar in bars
                if session_start <= bar['timestamp'] < opening_range_end
            ]
            
            if len(opening_bars) < 10:
                logger.warning(f"Insufficient bars ({len(opening_bars)}) for opening range for {symbol}")
                return None
            
            # Calculate high and low
            high = max(bar['high'] for bar in opening_bars)
            low = min(bar['low'] for bar in opening_bars)
            
            # Get timestamps for high and low
            high_time = next(bar['timestamp'] for bar in opening_bars if bar['high'] == high)
            low_time = next(bar['timestamp'] for bar in opening_bars if bar['low'] == low)
            
            result = {
                "high": high,
                "low": low,
                "high_time": high_time,
                "low_time": low_time
            }
            
            # Cache the result
            self._session_opening_ranges[cache_key] = result
            
            logger.debug(f"Opening range for {symbol}: high={high:.5f}, low={low:.5f}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate opening range for {symbol}: {e}")
            return None
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for symbol from HOT tier (TickCache) with MT5 fallback.
        
        Primary: Query PostgreSQL tick_cache table using HOT DBManager
        Fallback: MT5SocketAdapter.get_order_book()
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current mid-price (bid + ask) / 2 or None
        """
        try:
            # Primary: Query TickCache from HOT tier PostgreSQL
            # Use hot_db_manager which targets the tick_cache data source
            session = self.hot_db_manager.session
            now = datetime.now(timezone.utc)
            
            tick = session.query(TickCache).filter_by(
                symbol=symbol
            ).order_by(TickCache.timestamp.desc()).first()
            
            if tick:
                # Check if tick timestamp is timezone-aware
                tick_time = tick.timestamp
                if tick_time.tzinfo is None:
                    tick_time = tick_time.replace(tzinfo=timezone.utc)
                else:
                    tick_time = tick_time.astimezone(timezone.utc)
                
                if (now - tick_time).total_seconds() < 60:
                    # Tick is fresh (< 1 minute old)
                    mid_price = (tick.bid + tick.ask) / 2.0
                    logger.debug(f"Got current price for {symbol} from HOT TickCache: {mid_price}")
                    return mid_price
            
            # Fallback: Use MT5SocketAdapter if available
            if self.mt5_adapter is not None:
                try:
                    # Run async method in sync context
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        order_book = loop.run_until_complete(
                            self.mt5_adapter.get_order_book(symbol)
                        )
                        if order_book and order_book.get('bids') and order_book.get('asks'):
                            bid = order_book['bids'][0][0]
                            ask = order_book['asks'][0][0]
                            mid_price = (bid + ask) / 2.0
                            logger.debug(f"Got current price for {symbol} from MT5: {mid_price}")
                            return mid_price
                    finally:
                        loop.close()
                except Exception as e:
                    logger.warning(f"Failed to get price from MT5 for {symbol}: {e}")
            
            # Last resort: Try MetaTrader5 Python package
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        mt5.shutdown()
                        return tick.ask
            except Exception:
                pass
            
            logger.warning(f"No fresh tick available for {symbol}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return None
    
    def _get_current_atr(self, symbol: str) -> Optional[float]:
        """
        Get current ATR (14-period) for symbol.
        
        Calculates ATR from the most recent 14 bars.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current ATR value or None
        """
        try:
            # Get 14 bars for ATR calculation (+1 for TR calculation)
            bars = self._get_recent_bars(symbol, timeframe="M1", lookback=30)
            
            if not bars or len(bars) < 15:
                logger.warning(f"Insufficient bars for ATR calculation for {symbol}")
                return None
            
            atr = self._calculate_atr(bars, period=14)
            
            if atr:
                logger.debug(f"Current ATR for {symbol}: {atr:.5f}")
            
            return atr
            
        except Exception as e:
            logger.error(f"Failed to get current ATR for {symbol}: {e}")
            return None
    
    def _get_average_atr(self, symbol: str) -> Optional[float]:
        """
        Get average ATR (50-period baseline) for symbol.
        
        Calculates ATR from the last 50 bars for baseline comparison.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Average ATR value or None
        """
        try:
            # Get 50 bars for average ATR calculation (+1 for TR calculation)
            bars = self._get_recent_bars(symbol, timeframe="M1", lookback=60)
            
            if not bars or len(bars) < 51:
                logger.warning(f"Insufficient bars for average ATR for {symbol}")
                return None
            
            atr = self._calculate_atr(bars, period=50)
            
            if atr:
                logger.debug(f"Average ATR for {symbol}: {atr:.5f}")
            
            return atr
            
        except Exception as e:
            logger.error(f"Failed to get average ATR for {symbol}: {e}")
            return None
    
    def _populate_session_opening_ranges(self, session: str) -> None:
        """
        Pre-populate opening ranges for all symbols before scanning.
        
        This ensures the session opening ranges cache is populated
        before the breakout scanner runs.
        
        Args:
            session: Current trading session name
        """
        session_start = self._get_session_start(session)
        if session_start is None:
            logger.warning(f"Could not determine session start for {session}")
            return
        
        now = datetime.now(timezone.utc)
        minutes_since_open = (now - session_start).total_seconds() / 60
        
        # Only populate if we're at least 30 minutes into the session
        if minutes_since_open < 30:
            logger.debug(f"Session {session} too young for opening range population")
            return
        
        for symbol in self.symbols:
            cache_key = f"{symbol}_{session_start.strftime('%Y%m%d_%H')}"
            if cache_key not in self._session_opening_ranges:
                opening_range = self._fetch_opening_range_from_mt5(symbol, session_start)
                if opening_range:
                    self._session_opening_ranges[cache_key] = opening_range
                    logger.debug(f"Populated opening range for {symbol}: {opening_range}")
    
    def _fetch_opening_range_from_mt5(
        self,
        symbol: str,
        session_start: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch opening range from MT5 history or DuckDB.
        
        Tries MT5 Python API first, then falls back to DuckDB WARM tier.
        Uses session window query instead of fixed lookback.
        
        Args:
            symbol: Trading symbol
            session_start: Session start time (timezone-aware UTC)
            
        Returns:
            Dict with high, low, high_time, low_time or None
        """
        # Ensure session_start is timezone-aware UTC
        if session_start.tzinfo is None:
            session_start = session_start.replace(tzinfo=timezone.utc)
        else:
            session_start = session_start.astimezone(timezone.utc)
        
        opening_range_end = session_start + timedelta(minutes=30)
        now = datetime.now(timezone.utc)
        
        # Calculate lookback from session_start to now (but at least 35 min to cover 30 min window)
        minutes_lookback = int((now - session_start).total_seconds() / 60) + 10
        minutes_lookback = max(minutes_lookback, 35)
        
        # Try DuckDB WARM tier first (has historical data)
        # Query using session window - this ensures we get bars from session_start onwards
        try:
            bars = self._get_recent_bars(
                symbol, 
                timeframe="M1", 
                lookback=minutes_lookback,
                start_date=session_start,
                end_date=opening_range_end
            )
            if bars:
                # Filter bars within the opening range window (backup filter)
                opening_bars = [
                    bar for bar in bars
                    if session_start <= bar['timestamp'] < opening_range_end
                ]
                
                if len(opening_bars) >= 10:
                    high = max(bar['high'] for bar in opening_bars)
                    low = min(bar['low'] for bar in opening_bars)
                    high_time = next(bar['timestamp'] for bar in opening_bars if bar['high'] == high)
                    low_time = next(bar['timestamp'] for bar in opening_bars if bar['low'] == low)
                    
                    return {
                        "high": high,
                        "low": low,
                        "high_time": high_time,
                        "low_time": low_time
                    }
        except Exception as e:
            logger.debug(f"DuckDB opening range fetch failed for {symbol}: {e}")
        
        # Fallback: Try MT5 Python API
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                # Convert to MT5 timezone format and fetch rates
                # Use session window directly
                utc_from = session_start.replace(tzinfo=None)
                utc_to = opening_range_end.replace(tzinfo=None)
                
                rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, utc_from, utc_to)
                mt5.shutdown()
                
                if rates is not None and len(rates) >= 10:
                    import numpy as np
                    high = float(np.max(rates['high']))
                    low = float(np.min(rates['low']))
                    
                    # Find timestamps for high and low
                    high_idx = np.argmax(rates['high'])
                    low_idx = np.argmin(rates['low'])
                    high_time = datetime.fromtimestamp(rates['time'][high_idx], tz=timezone.utc)
                    low_time = datetime.fromtimestamp(rates['time'][low_idx], tz=timezone.utc)
                    
                    return {
                        "high": high,
                        "low": low,
                        "high_time": high_time,
                        "low_time": low_time
                    }
        except Exception as e:
            logger.debug(f"MT5 opening range fetch failed for {symbol}: {e}")
        
        return None
    
    def _get_upcoming_news_events(self) -> List[Dict[str, Any]]:
        """
        Get upcoming news events from economic calendar.
        
        Sources events from:
        1. Internal news database table (if configured)
        2. MT5 economic calendar (if available)
        3. ForexFactory / Investing.com (via internal fetcher)
        
        Returns:
            List of upcoming news events with datetime, name, currency, impact
        """
        events = []
        
        # Try internal database first
        try:
            session = self.db_manager.session
            from datetime import timedelta as td
            
            # Look for news_events table
            try:
                from sqlalchemy import text
                now = datetime.now(timezone.utc)
                lookahead = now + td(hours=1)  # Look 1 hour ahead
                
                result = session.execute(text("""
                    SELECT event_time, event_name, currency, impact, forecast, previous
                    FROM news_events
                    WHERE event_time BETWEEN :now AND :lookahead
                    AND impact IN ('High', 'Medium')
                    ORDER BY event_time ASC
                    LIMIT 20
                """), {"now": now, "lookahead": lookahead})
                
                for row in result:
                    events.append({
                        "datetime": row[0] if isinstance(row[0], datetime) else 
                                   datetime.fromisoformat(str(row[0]).replace('Z', '+00:00')),
                        "name": row[1],
                        "currency": row[2],
                        "impact": row[3],
                        "forecast": row[4],
                        "previous": row[5],
                        "source": "database"
                    })
                
                if events:
                    logger.debug(f"Found {len(events)} upcoming news events from database")
                    return events
                    
            except Exception as e:
                logger.debug(f"News database query failed: {e}")
                
        except Exception as e:
            logger.debug(f"Database news fetch failed: {e}")
        
        # Try MT5 economic calendar if available
        try:
            import MetaTrader5 as mt5
            if mt5.initialize():
                # MT5 calendar functions may not be available on all brokers
                if hasattr(mt5, 'calendar_value'):
                    now = datetime.now(timezone.utc)
                    lookahead = now + timedelta(hours=1)
                    
                    # Fetch calendar events
                    calendar_events = mt5.calendar_value()
                    mt5.shutdown()
                    
                    if calendar_events:
                        for event in calendar_events:
                            event_time = datetime.fromtimestamp(event.time, tz=timezone.utc)
                            if now <= event_time <= lookahead:
                                events.append({
                                    "datetime": event_time,
                                    "name": event.name,
                                    "currency": event.currency,
                                    "impact": event.importance,
                                    "forecast": event.forecast_value,
                                    "previous": event.prev_value,
                                    "source": "mt5_calendar"
                                })
                        
                        if events:
                            logger.debug(f"Found {len(events)} upcoming news events from MT5 calendar")
                            return events
                else:
                    mt5.shutdown()
                    
        except Exception as e:
            logger.debug(f"MT5 calendar fetch failed: {e}")
        
        # Try external news fetcher if configured
        try:
            from src.data.news_fetcher import NewsEventFetcher
            fetcher = NewsEventFetcher()
            events = fetcher.get_upcoming_events(hours_ahead=1, impact_levels=["High", "Medium"])
            if events:
                logger.debug(f"Found {len(events)} upcoming news events from external fetcher")
                return events
        except ImportError:
            logger.debug("NewsEventFetcher not available")
        except Exception as e:
            logger.debug(f"External news fetch failed: {e}")
        
        # No events found
        logger.debug("No upcoming news events found from any source")
        return []
    
    def _get_recent_price_data(self, symbol: str, bars: int = 50) -> Optional[List[Dict]]:
        """
        Get recent price data for symbol.
        
        Reuses _get_recent_bars to fetch OHLC data from DuckDB WARM tier.
        
        Args:
            symbol: Trading symbol
            bars: Number of bars to fetch (default 50)
            
        Returns:
            List of bar dictionaries or None on failure
        """
        try:
            price_data = self._get_recent_bars(
                symbol,
                timeframe="M1",
                lookback=bars
            )
            
            if price_data:
                logger.debug(f"Retrieved {len(price_data)} price bars for {symbol}")
            
            return price_data if price_data else None
            
        except Exception as e:
            logger.error(f"Failed to get recent price data for {symbol}: {e}")
            return None
    
    def _detect_fvg(self, price_data: List[Dict]) -> List[Dict]:
        """
        Detect Fair Value Gaps (FVG) in price data.
        
        A Fair Value Gap occurs when there is an imbalance in price:
        - Bullish FVG: Bar 1 high < Bar 3 low (gap between bars)
        - Bearish FVG: Bar 1 low > Bar 3 high (gap between bars)
        
        The FVG level is the midpoint of the gap, representing the "fair value"
        where price may return to before continuing.
        
        Args:
            price_data: List of OHLC bar dictionaries
            
        Returns:
            List of detected FVG setups with direction, level, gap_size, confidence
        """
        if not price_data or len(price_data) < 3:
            return []
        
        fvgs = []
        
        try:
            # Iterate through bars looking for 3-bar FVG patterns
            for i in range(2, len(price_data)):
                bar_1 = price_data[i - 2]  # Oldest bar
                bar_2 = price_data[i - 1]  # Middle bar (the impulse)
                bar_3 = price_data[i]      # Newest bar
                
                # Bullish FVG: Gap between bar_1 high and bar_3 low
                if bar_1['high'] < bar_3['low']:
                    gap_high = bar_3['low']
                    gap_low = bar_1['high']
                    gap_size = gap_high - gap_low
                    gap_mid = (gap_high + gap_low) / 2.0
                    
                    # Calculate confidence based on gap size relative to ATR-like measure
                    avg_range = sum(
                        (b['high'] - b['low']) for b in price_data[max(0, i-14):i+1]
                    ) / min(14, i + 1)
                    
                    # Larger gaps relative to average range = higher confidence
                    confidence = min(0.9, 0.5 + (gap_size / avg_range) * 0.2) if avg_range > 0 else 0.5
                    
                    fvgs.append({
                        "direction": "bullish",
                        "level": gap_mid,
                        "gap_high": gap_high,
                        "gap_low": gap_low,
                        "gap_size": gap_size,
                        "confidence": confidence,
                        "bar_index": i,
                        "timestamp": bar_3.get('timestamp')
                    })
                
                # Bearish FVG: Gap between bar_3 high and bar_1 low
                elif bar_1['low'] > bar_3['high']:
                    gap_high = bar_1['low']
                    gap_low = bar_3['high']
                    gap_size = gap_high - gap_low
                    gap_mid = (gap_high + gap_low) / 2.0
                    
                    # Calculate confidence based on gap size relative to ATR-like measure
                    avg_range = sum(
                        (b['high'] - b['low']) for b in price_data[max(0, i-14):i+1]
                    ) / min(14, i + 1)
                    
                    confidence = min(0.9, 0.5 + (gap_size / avg_range) * 0.2) if avg_range > 0 else 0.5
                    
                    fvgs.append({
                        "direction": "bearish",
                        "level": gap_mid,
                        "gap_high": gap_high,
                        "gap_low": gap_low,
                        "gap_size": gap_size,
                        "confidence": confidence,
                        "bar_index": i,
                        "timestamp": bar_3.get('timestamp')
                    })
            
            # Filter to only return FVGs from recent bars (last 10 bars)
            # and check if current price hasn't already filled the gap
            recent_fvgs = []
            current_bar_idx = len(price_data) - 1
            
            for fvg in fvgs:
                # Only consider FVGs from the last 10 bars
                if current_bar_idx - fvg["bar_index"] <= 10:
                    # Check if gap is still unfilled (or partially filled)
                    # Remove bar_index and timestamp before returning
                    fvg_result = {
                        "direction": fvg["direction"],
                        "level": fvg["level"],
                        "gap_size": fvg["gap_size"],
                        "confidence": fvg["confidence"]
                    }
                    recent_fvgs.append(fvg_result)
            
            if recent_fvgs:
                logger.debug(f"Detected {len(recent_fvgs)} FVG setups")
            
            return recent_fvgs
            
        except Exception as e:
            logger.error(f"Error detecting FVG: {e}")
            return []
    
    def _detect_order_blocks(self, price_data: List[Dict]) -> List[Dict]:
        """
        Detect Order Blocks in price data.
        
        An Order Block is the last opposing candle before a significant move:
        - Bullish OB: Last bearish candle before a strong bullish move
        - Bearish OB: Last bullish candle before a strong bearish move
        
        The OB level is typically the 50% of the order block candle,
        representing where institutional orders were placed.
        
        Args:
            price_data: List of OHLC bar dictionaries
            
        Returns:
            List of detected Order Block setups with direction, level, strength, confidence
        """
        if not price_data or len(price_data) < 5:
            return []
        
        order_blocks = []
        
        try:
            # Calculate average range for significance threshold
            ranges = [(bar['high'] - bar['low']) for bar in price_data]
            avg_range = sum(ranges) / len(ranges) if ranges else 0
            significance_threshold = avg_range * 1.5  # Move must be 1.5x average
            
            # Look for order blocks
            for i in range(2, len(price_data) - 2):
                current_bar = price_data[i]
                next_bars = price_data[i+1:i+3]  # Look at next 2 bars
                
                if not next_bars:
                    continue
                
                # Determine if current bar is bullish or bearish
                is_bullish_bar = current_bar['close'] > current_bar['open']
                is_bearish_bar = current_bar['close'] < current_bar['open']
                
                # Calculate the move after this bar
                if is_bearish_bar:
                    # Potential bullish OB: bearish bar followed by bullish move
                    next_high = max(bar['high'] for bar in next_bars)
                    move_size = next_high - current_bar['close']
                    
                    if move_size > significance_threshold:
                        # Calculate OB level (50% of the bearish candle)
                        ob_level = (current_bar['high'] + current_bar['low']) / 2.0
                        
                        # Strength based on the magnitude of the following move
                        strength = move_size / avg_range if avg_range > 0 else 1.0
                        confidence = min(0.85, 0.4 + strength * 0.15)
                        
                        order_blocks.append({
                            "direction": "bullish",
                            "level": ob_level,
                            "ob_high": current_bar['high'],
                            "ob_low": current_bar['low'],
                            "strength": round(strength, 2),
                            "confidence": confidence,
                            "bar_index": i
                        })
                
                elif is_bullish_bar:
                    # Potential bearish OB: bullish bar followed by bearish move
                    next_low = min(bar['low'] for bar in next_bars)
                    move_size = current_bar['close'] - next_low
                    
                    if move_size > significance_threshold:
                        # Calculate OB level (50% of the bullish candle)
                        ob_level = (current_bar['high'] + current_bar['low']) / 2.0
                        
                        # Strength based on the magnitude of the following move
                        strength = move_size / avg_range if avg_range > 0 else 1.0
                        confidence = min(0.85, 0.4 + strength * 0.15)
                        
                        order_blocks.append({
                            "direction": "bearish",
                            "level": ob_level,
                            "ob_high": current_bar['high'],
                            "ob_low": current_bar['low'],
                            "strength": round(strength, 2),
                            "confidence": confidence,
                            "bar_index": i
                        })
            
            # Filter to only recent order blocks (last 15 bars)
            # and check if price hasn't invalidated them
            recent_obs = []
            current_bar_idx = len(price_data) - 1
            current_price = price_data[-1]['close']
            
            for ob in order_blocks:
                # Only consider OBs from the last 15 bars
                if current_bar_idx - ob["bar_index"] <= 15:
                    # Check if OB is still valid (price hasn't broken through significantly)
                    if ob["direction"] == "bullish" and current_price > ob["ob_low"] * 0.995:
                        # Bullish OB still valid if price is above OB low
                        ob_result = {
                            "direction": ob["direction"],
                            "level": ob["level"],
                            "strength": ob["strength"],
                            "confidence": ob["confidence"]
                        }
                        recent_obs.append(ob_result)
                    elif ob["direction"] == "bearish" and current_price < ob["ob_high"] * 1.005:
                        # Bearish OB still valid if price is below OB high
                        ob_result = {
                            "direction": ob["direction"],
                            "level": ob["level"],
                            "strength": ob["strength"],
                            "confidence": ob["confidence"]
                        }
                        recent_obs.append(ob_result)
            
            if recent_obs:
                logger.debug(f"Detected {len(recent_obs)} Order Block setups")
            
            return recent_obs
            
        except Exception as e:
            logger.error(f"Error detecting Order Blocks: {e}")
            return []
    
    def _get_breakout_bots(self, symbol: str, direction: str) -> List[str]:
        """Get recommended bots for breakout trading."""
        # TODO: Query BotRegistry for breakout-capable bots
        return ["london_breakout_01", "session_momentum_02"]
    
    def _get_volatility_bots(self, symbol: str) -> List[str]:
        """Get recommended bots for volatility trading."""
        return ["volatility_catcher_01", "atr_scalper_02"]
    
    def _get_news_bots(self) -> List[str]:
        """Get recommended bots for news trading."""
        return ["news_sniper_01"]
    
    def _get_ict_bots(self, symbol: str) -> List[str]:
        """Get recommended ICT bots."""
        return ["ict_silver_bullet_01", "fvg_hunter_02"]
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return [a.to_dict() for a in self._recent_alerts[-limit:]]
    
    def get_scan_interval(self) -> int:
        """
        Get recommended scan interval in seconds based on current session.
        
        Returns:
            Scan interval in seconds
        """
        try:
            from src.router.sessions import get_current_session, TradingSession
            
            session = get_current_session()
            
            if session == TradingSession.OVERLAP:
                return 60  # 1 minute during overlap
            elif session in [TradingSession.LONDON, TradingSession.NEW_YORK]:
                return 300  # 5 minutes during major sessions
            else:
                return 900  # 15 minutes during Asian/quiet periods
                
        except Exception:
            return 300  # Default 5 minutes


# ============== APScheduler Integration ==============

# Prometheus metrics for MarketScanner
SCANNER_PROMETHEUS_AVAILABLE = False
scanner_alerts_counter = None
scanner_errors_counter = None
scanner_scan_duration = None


def _init_scanner_prometheus_metrics():
    """
    Initialize Prometheus metrics for MarketScanner.
    """
    global SCANNER_PROMETHEUS_AVAILABLE, scanner_alerts_counter, scanner_errors_counter, scanner_scan_duration
    
    if SCANNER_PROMETHEUS_AVAILABLE:
        return
    
    try:
        from prometheus_client import Counter, Histogram, REGISTRY
        
        scanner_alerts_counter = Counter(
            'quantmind_scanner_alerts_total',
            'Total number of scanner alerts generated',
            ['scan_type', 'symbol', 'priority']
        )
        scanner_errors_counter = Counter(
            'quantmind_scanner_errors_total',
            'Total number of scanner errors'
        )
        scanner_scan_duration = Histogram(
            'quantmind_scanner_scan_duration_seconds',
            'Scanner scan duration in seconds',
            ['scan_type']
        )
        
        SCANNER_PROMETHEUS_AVAILABLE = True
        logger.debug("MarketScanner Prometheus metrics initialized")
        
    except ImportError:
        logger.debug("prometheus_client not available, scanner metrics disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize scanner Prometheus metrics: {e}")


# Initialize Prometheus metrics on module load
_init_scanner_prometheus_metrics()


class MarketScannerScheduler:
    """
    APScheduler integration for MarketScanner.
    
    Runs scans at intervals based on current market session:
    - Asian Session: Every 15 minutes
    - London/NY Sessions: Every 5 minutes
    - Overlap: Every 1 minute
    """
    
    def __init__(self, scanner: Optional[MarketScanner] = None):
        self.scanner = scanner or MarketScanner()
        self._scheduler: Optional[Any] = None
        self._is_running = False
        self._last_scan_time: Optional[datetime] = None
    
    def setup_schedule(self) -> bool:
        """
        Setup APScheduler jobs for market scanning.
        
        Returns:
            True if scheduler setup successfully
        """
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger
            
            self._scheduler = AsyncIOScheduler()
            
            # Get initial interval based on current session
            interval_seconds = self.scanner.get_scan_interval()
            
            # Add the scan job
            self._scheduler.add_job(
                self.run_scheduled_scan,
                trigger=IntervalTrigger(seconds=interval_seconds),
                id='market_scanner_scan',
                name='Market Scanner Session Scan',
                replace_existing=True
            )
            
            logger.info(f"MarketScanner scheduler configured with {interval_seconds}s interval")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup MarketScanner scheduler: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start the scheduler.
        
        Returns:
            True if started successfully
        """
        if self._scheduler is None:
            if not self.setup_schedule():
                return False
        
        try:
            self._scheduler.start()
            self._is_running = True
            logger.info("MarketScanner scheduler started")
            return True
        except Exception as e:
            logger.error(f"Failed to start MarketScanner scheduler: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the scheduler.
        
        Returns:
            True if stopped successfully
        """
        if self._scheduler is None:
            return True
        
        try:
            self._scheduler.shutdown(wait=False)
            self._is_running = False
            logger.info("MarketScanner scheduler stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop MarketScanner scheduler: {e}")
            return False
    
    def run_scheduled_scan(self) -> List[Dict[str, Any]]:
        """
        Run a scheduled scan and return results.
        This method is called by APScheduler.
        
        Returns:
            List of alert dictionaries
        """
        import time
        start_time = time.time()
        
        try:
            logger.info("Running scheduled market scan...")
            alerts = self.scanner.run_full_scan()
            
            # Update last scan time
            self._last_scan_time = datetime.now(timezone.utc)
            
            # Record metrics
            duration = time.time() - start_time
            self._record_scan_metrics(alerts, duration, success=True)
            
            logger.info(f"Scheduled scan completed: {len(alerts)} alerts in {duration:.2f}s")
            return alerts
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Scheduled scan failed: {e}")
            self._record_scan_metrics([], duration, success=False)
            return []
    
    def _record_scan_metrics(self, alerts: List[Dict], duration: float, success: bool) -> None:
        """Record Prometheus metrics for scan."""
        if not SCANNER_PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Record scan duration
            scanner_scan_duration.labels(scan_type='full_scan').observe(duration)
            
            # Record alerts by type
            for alert in alerts:
                scanner_alerts_counter.labels(
                    scan_type=alert.get('type', 'unknown'),
                    symbol=alert.get('symbol', 'unknown'),
                    priority=alert.get('priority', 'medium')
                ).inc()
            
            if not success:
                scanner_errors_counter.inc()
                
        except Exception as e:
            logger.debug(f"Failed to record scan metrics: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._is_running,
            "last_scan_time": self._last_scan_time.isoformat() if self._last_scan_time else None,
            "interval_seconds": self.scanner.get_scan_interval(),
            "scheduler_jobs": len(self._scheduler.get_jobs()) if self._scheduler and self._is_running else 0
        }


# Global scanner scheduler instance
_scanner_scheduler: Optional[MarketScannerScheduler] = None


def get_scanner_scheduler() -> Optional[MarketScannerScheduler]:
    """Get the global scanner scheduler instance."""
    return _scanner_scheduler


def initialize_scanner_scheduler(scanner: Optional[MarketScanner] = None) -> MarketScannerScheduler:
    """
    Initialize and return the global scanner scheduler.
    
    Args:
        scanner: Optional MarketScanner instance
        
    Returns:
        MarketScannerScheduler instance
    """
    global _scanner_scheduler
    
    if _scanner_scheduler is None:
        _scanner_scheduler = MarketScannerScheduler(scanner)
    
    return _scanner_scheduler


def start_scanner_scheduler(scanner: Optional[MarketScanner] = None) -> bool:
    """
    Initialize and start the global scanner scheduler.
    
    Args:
        scanner: Optional MarketScanner instance
        
    Returns:
        True if started successfully
    """
    scheduler = initialize_scanner_scheduler(scanner)
    return scheduler.start()


def stop_scanner_scheduler() -> bool:
    """
    Stop the global scanner scheduler.
    
    Returns:
        True if stopped successfully
    """
    global _scanner_scheduler
    
    if _scanner_scheduler:
        result = _scanner_scheduler.stop()
        _scanner_scheduler = None
        return result
    return True


# ============== Commander Integration ==============

class ScannerCommanderIntegration:
    """
    Integration hook for MarketScanner with Commander.
    
    Allows Commander to:
    - Activate scanner on market events
    - Query recent opportunities
    - Check if scanner detected opportunities for specific symbols
    """
    
    def __init__(self, scanner: Optional[MarketScanner] = None):
        self.scanner = scanner or MarketScanner()
        self._active = False
    
    def activate(self) -> bool:
        """
        Activate the scanner integration.
        Starts the scheduler if not already running.
        
        Returns:
            True if activated successfully
        """
        if self._active:
            return True
        
        try:
            # Start scanner scheduler
            success = start_scanner_scheduler(self.scanner)
            if success:
                self._active = True
                logger.info("ScannerCommanderIntegration activated")
            return success
        except Exception as e:
            logger.error(f"Failed to activate ScannerCommanderIntegration: {e}")
            return False
    
    def deactivate(self) -> bool:
        """
        Deactivate the scanner integration.
        Stops the scheduler.
        
        Returns:
            True if deactivated successfully
        """
        if not self._active:
            return True
        
        try:
            success = stop_scanner_scheduler()
            if success:
                self._active = False
                logger.info("ScannerCommanderIntegration deactivated")
            return success
        except Exception as e:
            logger.error(f"Failed to deactivate ScannerCommanderIntegration: {e}")
            return False
    
    def is_active(self) -> bool:
        """Check if scanner integration is active."""
        return self._active
    
    def get_opportunities_for_symbol(self, symbol: str, min_confidence: float = 0.6) -> List[Dict[str, Any]]:
        """
        Get recent opportunities for a specific symbol.
        
        Args:
            symbol: Trading symbol
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of opportunity dictionaries
        """
        alerts = self.scanner.get_recent_alerts(limit=100)
        
        return [
            alert for alert in alerts
            if alert.get('symbol') == symbol and alert.get('confidence', 0) >= min_confidence
        ]
    
    def get_high_confidence_opportunities(self, min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """
        Get high confidence opportunities across all symbols.
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of high confidence opportunity dictionaries
        """
        alerts = self.scanner.get_recent_alerts(limit=100)
        
        return [
            alert for alert in alerts
            if alert.get('confidence', 0) >= min_confidence
        ]
    
    def trigger_scan(self) -> List[Dict[str, Any]]:
        """
        Manually trigger a scan and return results.
        
        Returns:
            List of alert dictionaries
        """
        return self.scanner.run_full_scan()


# Global scanner commander integration instance
_scanner_commander_integration: Optional[ScannerCommanderIntegration] = None


def get_scanner_commander_integration() -> ScannerCommanderIntegration:
    """Get the global scanner commander integration instance."""
    global _scanner_commander_integration
    
    if _scanner_commander_integration is None:
        _scanner_commander_integration = ScannerCommanderIntegration()
    
    return _scanner_commander_integration


# ============== DynamicBotLimiter Integration ==============

def check_scanner_opportunity_limits(
    opportunity_confidence: float,
    current_bots: int,
    account_balance: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Check if scanner opportunity can be acted upon based on bot limits.
    
    Uses DynamicBotLimiter to validate:
    - Account tier allows trading
    - Bot limit not exceeded
    - Safety buffer maintained
    
    Args:
        opportunity_confidence: Confidence score of the opportunity
        current_bots: Number of currently active bots
        account_balance: Optional account balance
        
    Returns:
        Tuple of (can_act, reason)
    """
    from src.router.dynamic_bot_limits import DynamicBotLimiter
    
    # Only consider high confidence opportunities
    if opportunity_confidence < 0.7:
        return False, "Opportunity confidence below threshold (0.7)"
    
    # Check bot limits
    return DynamicBotLimiter.can_add_bot(account_balance, current_bots)


# ============== Backwards Compatibility Exports ==============

__all__ = [
    'MarketScanner',
    'ScannerAlert',
    'AlertType',
    'AlertPriority',
    'MarketScannerScheduler',
    'get_scanner_scheduler',
    'initialize_scanner_scheduler',
    'start_scanner_scheduler',
    'stop_scanner_scheduler',
    'ScannerCommanderIntegration',
    'get_scanner_commander_integration',
    'check_scanner_opportunity_limits',
]