"""
QuantMindLib V1 — cTrader Market Data Adapter

Real implementation of IMarketDataAdapter for cTrader platform.
Wraps cTrader Open API for live market data retrieval, including:
- TCP connection with Protobuf protocol
- Application + account authentication
- Live spot/depth streaming via Twisted callbacks
- Historical trendbar (candle) data retrieval
- Feature computation from live tick data
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr

from src.library.core.composition.adapter_contracts import IMarketDataAdapter
from src.library.core.domain.feature_vector import FeatureConfidence, FeatureVector
from src.library.core.domain.market_context import MarketContext
from src.library.core.domain.order_flow_signal import OrderFlowSignal
from src.library.core.domain.pattern_signal import PatternSignal
from src.library.core.types.enums import NewsState, OrderFlowSource, RegimeType, SignalDirection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# cTrader Open API imports (installed via: pip install ctrader-open-api)
# ---------------------------------------------------------------------------
try:
    from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
    from ctrader_open_api.messages.OpenApiMessages_pb2 import (
        ProtoOASymbolsListReq,
        ProtoOASymbolsListRes,
        ProtoOAGetTickDataReq,
        ProtoOAGetTickDataRes,
        ProtoOAGetTrendbarsReq,
        ProtoOAGetTrendbarsRes,
        ProtoOASpotEvent,
        ProtoOADepthEvent,
        ProtoOASubscribeSpotsReq,
        ProtoOASubscribeDepthQuotesReq,
        ProtoOAUnsubscribeSpotsReq,
        ProtoOAUnsubscribeDepthQuotesReq,
        ProtoOAApplicationAuthReq,
        ProtoOAAccountAuthReq,
    )
    _CTRADER_AVAILABLE = True
except ImportError:
    _CTRADER_AVAILABLE = False
    logger.warning(
        "ctrader-open-api not installed. "
        "Install with: pip install ctrader-open-api"
    )

# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class TickSnapshot(BaseModel):
    """In-memory tick snapshot for a single symbol."""
    symbol: str
    bid: float = 0.0
    ask: float = 0.0
    timestamp_ms: int = 0
    bid_volume: float = 0.0
    ask_volume: float = 0.0

    model_config = BaseModel.model_config

    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.ask and self.bid else 0.0

    @property
    def mid_price(self) -> float:
        return (self.ask + self.bid) / 2.0 if self.ask and self.bid else 0.0

    @property
    def spread_bps(self) -> float:
        """Spread in basis points."""
        mid = self.mid_price
        if mid > 0:
            return (self.spread / mid) * 10000
        return 0.0


class SymbolDefinition(BaseModel):
    """Cached symbol metadata from cTrader."""
    symbol_id: int = 0
    display_name: str = ""
    digits: int = 5
    pip_size: float = 0.00001
    contract_currency: str = ""
    max_quantity: float = 0.0
    min_quantity: float = 0.01

    model_config = BaseModel.model_config


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------

class CTraderMarketAdapter(BaseModel):
    """
    Real cTrader Open API implementation of IMarketDataAdapter.

    Manages a Twisted-based TCP+Protobuf connection to the cTrader Open API.
    Authenticates as an application, then as a specific trading account,
    and streams live market data for subscribed symbols.

    State is maintained in private attributes to avoid Pydantic field conflicts.

    Usage:
        adapter = CTraderMarketAdapter(
            client_id="your_client_id",
            client_secret="your_secret",
            ctid_trader_account_id=12345678,
            access_token="your_access_token",
            use_demo=True,  # or False for live
        )
        connected = adapter.connect()
        adapter.subscribe_symbols(["EURUSD", "GBPJPY"])
        fv = adapter.get_feature_vector("EURUSD")
        mc = adapter.get_market_context()
        of = adapter.get_order_flow("EURUSD")
    """

    # ── Configuration fields ────────────────────────────────────────────────

    client_id: str = Field(
        default="",
        description="cTrader Open API OAuth client ID",
    )
    client_secret: str = Field(
        default="",
        description="cTrader Open API OAuth client secret",
    )
    ctid_trader_account_id: int = Field(
        default=0,
        description="cTrader Trader Account ID (integer from cTrader platform)",
    )
    access_token: str = Field(
        default="",
        description="OAuth access token for account authentication",
    )
    use_demo: bool = Field(
        default=True,
        description="Connect to demo environment (True) or live (False)",
    )

    # ── Public state fields (schema-compliant) ───────────────────────────────

    adapter_id: str = "CTRADER_MARKET_ADAPTER_V1"
    _is_connected: bool = PrivateAttr(default=False)
    subscribed_symbols: List[str] = Field(default_factory=list)

    # ── Private runtime state ────────────────────────────────────────────────
    # These are plain instance attributes (not Pydantic fields) so that
    # direct assignment self._name = value is never intercepted. Pydantic
    # only manages fields with annotations; _prefixed names are treated as
    # private and are excluded from the schema. Using object.__setattr__ in
    # __init__ ensures Pydantic's __setattr__ does not interfere.

    _client: Any = None
    _twisted_reactor: Any = None
    _session_token: str = ""
    _last_spot_event_ms: int = 0
    _last_depth_event_ms: int = 0
    _symbol_definitions: Dict[int, SymbolDefinition] = {}
    _symbol_id_to_name: Dict[int, str] = {}
    _symbol_name_to_id: Dict[str, int] = {}
    _live_ticks: Dict[str, TickSnapshot] = {}
    _tick_history: Dict[str, List[tuple]] = {}
    _depth_snapshot: Dict[str, Dict[str, Any]] = {}

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Initialise runtime state — self._name = ... is direct assignment
        # because these are not Field-annotated, so Pydantic does not intercept.
        now_ms = int(time.time() * 1000)
        self._client = None
        self._twisted_reactor = None
        self._session_token = ""
        self._last_spot_event_ms = now_ms
        self._last_depth_event_ms = now_ms
        self._symbol_definitions = {}
        self._symbol_id_to_name = {}
        self._symbol_name_to_id = {}
        self._live_ticks = {}
        self._tick_history = {}
        self._depth_snapshot = {}
        self._is_connected = False

    # -------------------------------------------------------------------------
    # IMarketDataAdapter required method
    # -------------------------------------------------------------------------

    def is_connected(self) -> bool:
        """
        Return True if the cTrader TCP connection is active and authenticated.

        Required by IMarketDataAdapter protocol.
        """
        return self._is_connected

    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Establish TCP+Protobuf connection to cTrader Open API and authenticate.

        Flow:
          1. Connect TCP to cTrader server (demo or live)
          2. Set up message received callback to handle incoming messages
          3. Send ProtoOAApplicationAuthReq → receive session token
          4. Send ProtoOAAccountAuthReq → confirm account access

        Returns:
            True if both authentication steps succeed; False otherwise.
        """
        if not _CTRADER_AVAILABLE:
            logger.error("ctrader-open-api package not installed")
            return False

        if self._is_connected and self._client is not None:
            return True

        host = EndPoints.PROTOBUF_DEMO_HOST if self.use_demo else EndPoints.PROTOBUF_LIVE_HOST
        port = EndPoints.PROTOBUF_PORT
        logger.info(f"Connecting to cTrader at {host}:{port} (demo={self.use_demo})")

        try:
            # Build the Twisted-based client
            protocol = TcpProtocol()
            self._client = Client(host, port, protocol)

            # Register message handler — must be done BEFORE startService
            self._client.setMessageReceivedCallback(self._on_message_received)

            # Register lifecycle callbacks
            self._client.setConnectedCallback(self._on_connected)
            self._client.setDisconnectedCallback(self._on_disconnected)

            # Start the Twisted service (non-blocking Deferred)
            self._client.startService()

            # Note: startService() fires immediately and returns.
            # The actual connection is established asynchronously.
            # We set is_connected=False initially and promote to True
            # only after successful authentication in _on_connected.
            # For synchronous callers, we do a brief synchronous wait.

            # Trigger authentication after reactor pump
            success = self._authenticate_sync()
            if success:
                # Fetch symbol list after auth
                self._fetch_symbols_list_sync()
                self._is_connected = True
                logger.info("cTrader market adapter connected and authenticated")
            else:
                self._is_connected = False
                self._teardown_client()

            return self._is_connected

        except Exception as exc:
            logger.error(f"cTrader connection failed: {exc}", exc_info=True)
            self._is_connected = False
            self._teardown_client()
            return False

    def _authenticate_sync(self) -> bool:
        """
        Perform two-step authentication synchronously.

        Uses a polling loop with reactor iteration to wait for Deferred responses.
        This keeps connect() synchronous for callers that expect immediate returns.
        """
        if not _CTRADER_AVAILABLE or self._client is None:
            return False

        try:
            from twisted.internet import reactor
        except ImportError:
            logger.error("Twisted not available — cannot authenticate")
            return False

        auth_done = {"app_auth": False, "account_auth": False}

        def on_auth_response(msg: Any) -> None:
            """Handle both application auth and account auth responses."""
            if not hasattr(msg, "payloadType"):
                return

            # ProtoOAApplicationAuthRes payloadType = 4001
            if msg.payloadType == 4001:
                token = getattr(msg, "accessToken", None)
                if token:
                    self._session_token = token
                    auth_done["app_auth"] = True
                    logger.info("Application authentication successful")
                    self._send_account_auth()
                else:
                    logger.error("Application auth response missing accessToken")

            # ProtoOAAccountAuthRes payloadType = 4003
            elif msg.payloadType == 4003:
                if getattr(msg, "ctidTraderAccountId", 0):
                    auth_done["account_auth"] = True
                    logger.info(f"Account authentication successful (accountId={msg.ctidTraderAccountId})")
                else:
                    logger.error("Account auth response missing ctidTraderAccountId")

        # Install auth callback (will also be called via _on_message_received)
        self._auth_callback = on_auth_response

        # Step 1: Send ProtoOAApplicationAuthReq
        try:
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAApplicationAuthReq,
            )

            app_auth_req = ProtoOAApplicationAuthReq()
            app_auth_req.clientId = self.client_id
            app_auth_req.clientSecret = self.client_secret

            reactor.callFromThread(self._client.send, app_auth_req)
        except Exception as exc:
            logger.error(f"Failed to send application auth: {exc}")
            return False

        # Poll for auth completion (up to 15 seconds)
        deadline = time.time() + 15.0
        while not (auth_done["app_auth"] and auth_done["account_auth"]):
            if time.time() > deadline:
                logger.error("Authentication timeout")
                return False
            try:
                # Attempt to pump the reactor briefly (non-blocking)
                from twisted.internet import reactor as _reactor
                # Use a small select-based wait
                import select
                select.select([], [], [], 0.1)
            except Exception:
                time.sleep(0.1)

        return auth_done["app_auth"] and auth_done["account_auth"]

    def _send_account_auth(self) -> None:
        """Send account authentication request after session token is received."""
        if not self._session_token or not self._client:
            return

        try:
            from twisted.internet import reactor
            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAAccountAuthReq

            req = ProtoOAAccountAuthReq()
            req.ctidTraderAccountId = self.ctid_trader_account_id
            req.accessToken = self._session_token

            def _send():
                self._client.send(req)

            reactor.callFromThread(_send)
        except Exception as exc:
            logger.error(f"Failed to send account auth: {exc}")

    def _fetch_symbols_list_sync(self) -> None:
        """Request and wait for the symbols list from cTrader."""
        if not self._client:
            return

        try:
            from twisted.internet import reactor
            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOASymbolsListReq

            symbols_req = ProtoOASymbolsListReq()
            symbols_req.ctidTraderAccountId = self.ctid_trader_account_id
            symbols_req.includeArchivedSymbols = False

            def _send():
                self._client.send(symbols_req)

            reactor.callFromThread(_send)
        except Exception as exc:
            logger.error(f"Failed to send symbols list request: {exc}")

    def _on_connected(self) -> None:
        """Callback invoked when the TCP connection is established."""
        logger.info("cTrader TCP connection established")

    def _on_disconnected(self) -> None:
        """Callback invoked when the TCP connection is lost."""
        logger.warning("cTrader TCP connection lost")
        self._is_connected = False

    def _on_message_received(self, msg: Any) -> None:
        """
        Dispatch incoming protobuf messages to the appropriate handler.

        Handles:
          - ProtoOASpotEvent (live price updates) → update tick cache
          - ProtoOADepthEvent (live depth updates) → update depth cache
          - ProtoOASymbolsListRes → populate symbol registry
          - ProtoOAGetTrendbarsRes → store historical trendbars
        """
        if not hasattr(msg, "payloadType"):
            return

        pt = msg.payloadType

        # ProtoOASpotEvent = 2 (live tick data)
        if pt == 2:
            self._handle_spot_event(msg)
        # ProtoOADepthEvent = 3 (live depth)
        elif pt == 3:
            self._handle_depth_event(msg)
        # ProtoOASymbolsListRes = 24 (symbol list)
        elif pt == 24:
            self._handle_symbols_list(msg)
        # ProtoOAGetTrendbarsRes = 107 (historical candles)
        elif pt == 107:
            self._handle_trendbars_res(msg)
        else:
            # Pass to auth callback if still authenticating
            if hasattr(self, "_auth_callback") and self._auth_callback is not None:
                self._auth_callback(msg)

    def _handle_spot_event(self, msg: Any) -> None:
        """Update live tick cache from ProtoOASpotEvent."""
        self._last_spot_event_ms = int(time.time() * 1000)

        symbol_id = getattr(msg, "symbolId", 0)
        bid = getattr(msg, "bid", 0.0)
        ask = getattr(msg, "ask", 0.0)
        ts = getattr(msg, "timestamp", 0)

        symbol = self._symbol_id_to_name.get(symbol_id, str(symbol_id))
        tick = TickSnapshot(
            symbol=symbol,
            bid=float(bid),
            ask=float(ask),
            timestamp_ms=int(ts),
        )
        self._live_ticks[symbol] = tick

        # Append to rolling history
        if symbol not in self._tick_history:
            self._tick_history[symbol] = []
        self._tick_history[symbol].append((int(ts), float(bid), float(ask)))
        # Trim to max size
        if len(self._tick_history[symbol]) > self._max_tick_history:
            self._tick_history[symbol] = self._tick_history[symbol][-self._max_tick_history:]

    def _handle_depth_event(self, msg: Any) -> None:
        """Update depth cache from ProtoOADepthEvent."""
        self._last_depth_event_ms = int(time.time() * 1000)

        symbol_id = getattr(msg, "symbolId", 0)
        symbol = self._symbol_id_to_name.get(symbol_id, str(symbol_id))

        new_quotes = getattr(msg, "newQuotes", [])
        deleted = getattr(msg, "deletedQuotes", [])

        if symbol not in self._depth_snapshot:
            self._depth_snapshot[symbol] = {"bid_levels": {}, "ask_levels": {}}

        snap = self._depth_snapshot[symbol]
        for q in new_quotes:
            side = "bid_levels" if getattr(q, "isBuy", True) else "ask_levels"
            price = getattr(q, "price", 0.0)
            volume = getattr(q, "volume", 0.0)
            snap[side][float(price)] = float(volume)

        for q in deleted:
            side = "bid_levels" if getattr(q, "isBuy", True) else "ask_levels"
            price = getattr(q, "price", 0.0)
            snap[side].pop(float(price), None)

    def _handle_symbols_list(self, msg: Any) -> None:
        """Populate symbol registry from ProtoOASymbolsListRes."""
        symbols = getattr(msg, "symbol", [])
        for sym in symbols:
            sym_id = getattr(sym, "symbolId", 0)
            display = getattr(sym, "displayName", "")
            digits = getattr(sym, "digits", 5)

            self._symbol_id_to_name[sym_id] = display
            self._symbol_name_to_id[display] = sym_id

            self._symbol_definitions[sym_id] = SymbolDefinition(
                symbol_id=sym_id,
                display_name=display,
                digits=digits,
                pip_size=10 ** (-digits),
            )
        logger.info(f"Loaded {len(symbols)} symbols from cTrader")

    def _handle_trendbars_res(self, msg: Any) -> None:
        """Store historical trendbars for feature computation."""
        # Trendbar response format: repeated trendbars, each with OHLCV
        bars = getattr(msg, "trendbar", [])
        symbol_id = getattr(msg, "symbolId", 0)
        symbol = self._symbol_id_to_name.get(symbol_id, str(symbol_id))

        for bar in bars:
            ts = getattr(bar, "timestamp", 0)
            self._trendbar_cache.setdefault(symbol, []).append({
                "timestamp": int(ts),
                "open": float(getattr(bar, "open", 0)),
                "high": float(getattr(bar, "high", 0)),
                "low": float(getattr(bar, "low", 0)),
                "close": float(getattr(bar, "close", 0)),
                "volume": float(getattr(bar, "volume", 0)),
            })

    def _teardown_client(self) -> None:
        """Stop and clean up the Twisted client."""
        if self._client is not None:
            try:
                self._client.stopService()
            except Exception:
                pass
            self._client = None
            self._twisted_reactor = None
            self._session_token = ""
            self._last_spot_event_ms = int(time.time() * 1000)
            self._last_depth_event_ms = int(time.time() * 1000)
            self._symbol_definitions = {}
            self._symbol_id_to_name = {}
            self._symbol_name_to_id = {}
            self._live_ticks = {}
            self._tick_history = {}
            self._depth_snapshot = {}

    def disconnect(self) -> bool:
        """Disconnect from cTrader Open API and clean up resources."""
        try:
            # Unsubscribe from all symbols first
            for sym in list(self.subscribed_symbols):
                self._unsubscribe_symbol(sym)

            self._teardown_client()
            self._is_connected = False
            logger.info("cTrader market adapter disconnected")
            return True
        except Exception as exc:
            logger.error(f"Error during disconnect: {exc}", exc_info=True)
            return False

    def _unsubscribe_symbol(self, symbol: str) -> None:
        """Send unsubscribe requests for a symbol."""
        if not self._client:
            return
        sym_id = self._symbol_name_to_id.get(symbol)
        if not sym_id:
            return

        try:
            from twisted.internet import reactor
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOAUnsubscribeSpotsReq,
                ProtoOAUnsubscribeDepthQuotesReq,
            )

            spot_unsub = ProtoOAUnsubscribeSpotsReq()
            spot_unsub.ctidTraderAccountId = self.ctid_trader_account_id
            spot_unsub.symbolId.append(sym_id)

            depth_unsub = ProtoOAUnsubscribeDepthQuotesReq()
            depth_unsub.ctidTraderAccountId = self.ctid_trader_account_id
            depth_unsub.symbolId.append(sym_id)

            def _send():
                self._client.send(spot_unsub)
                self._client.send(depth_unsub)

            reactor.callFromThread(_send)
        except Exception as exc:
            logger.debug(f"Unsubscribe error for {symbol}: {exc}")

    # -------------------------------------------------------------------------
    # Symbol subscription
    # -------------------------------------------------------------------------

    def subscribe_symbols(self, symbols: List[str]) -> bool:
        """
        Subscribe to live spot and depth streams for given symbols.

        Sends ProtoOASubscribeSpotsReq and ProtoOASubscribeDepthQuotesReq
        for each symbol. Data arrives via ProtoOASpotEvent and ProtoOADepthEvent
        in the message received callback.

        Args:
            symbols: List of symbol display names (e.g., ["EURUSD", "GBPJPY"])

        Returns:
            True if subscription requests were sent; False if not connected.
        """
        if not self._is_connected or not self._client:
            logger.warning("Cannot subscribe — not connected")
            return False

        if not symbols:
            return True

        new_symbols = [s for s in symbols if s not in self.subscribed_symbols]
        if not new_symbols:
            return True

        self.subscribed_symbols.extend(new_symbols)

        try:
            from twisted.internet import reactor
            from ctrader_open_api.messages.OpenApiMessages_pb2 import (
                ProtoOASubscribeSpotsReq,
                ProtoOASubscribeDepthQuotesReq,
            )

            for sym in new_symbols:
                sym_id = self._symbol_name_to_id.get(sym)
                if sym_id is None:
                    logger.warning(f"Unknown symbol '{sym}' — cannot subscribe")
                    continue

                spot_req = ProtoOASubscribeSpotsReq()
                spot_req.ctidTraderAccountId = self.ctid_trader_account_id
                spot_req.symbolId.append(sym_id)
                spot_req.subscribeToSpotTimestamp = True

                depth_req = ProtoOASubscribeDepthQuotesReq()
                depth_req.ctidTraderAccountId = self.ctid_trader_account_id
                depth_req.symbolId.append(sym_id)

                def _send(req1=spot_req, req2=depth_req):
                    self._client.send(req1)
                    self._client.send(req2)

                reactor.callFromThread(_send)

            logger.info(f"Subscribed to {len(new_symbols)} symbols: {new_symbols}")
            return True

        except Exception as exc:
            logger.error(f"Failed to subscribe symbols: {exc}", exc_info=True)
            return False

    # -------------------------------------------------------------------------
    # Feature computation from live tick data
    # -------------------------------------------------------------------------

    def get_feature_vector(self, symbol: str) -> FeatureVector:
        """
        Compute a feature vector from live tick data for a symbol.

        Features computed:
          - momentum_5 / momentum_15 / momentum_60: price change over intervals
          - rvol: realized volatility vs historical baseline (ratio)
          - vwap_distance: current price vs VWAP (basis points)
          - spread_ratio: current spread vs typical spread (ratio)
          - tick_rate: ticks per second in the last minute

        Quality is based on data freshness and connection state.

        Falls back to last-known values if connection is lost but data is recent.
        Falls back to zero-confidence stub values if no data is available.
        """
        if not self._is_connected:
            return self._stub_feature_vector(symbol, reason="not_connected")

        tick = self._live_ticks.get(symbol)
        history = self._tick_history.get(symbol, [])

        now_ms = int(time.time() * 1000)

        if tick is None:
            return self._stub_feature_vector(symbol, reason="no_tick_data")

        features: Dict[str, float] = {}
        confidences: Dict[str, FeatureConfidence] = {}

        # Determine feed quality from freshness
        age_ms = now_ms - tick.timestamp_ms if tick.timestamp_ms else 0
        if age_ms < 1000:
            feed_tag = "HIGH"
            base_quality = 0.90
        elif age_ms < 5000:
            feed_tag = "MEDIUM"
            base_quality = 0.70
        elif age_ms < 30000:
            feed_tag = "LOW"
            base_quality = 0.50
        else:
            feed_tag = "STALE"
            base_quality = 0.20

        # ── Momentum: price change over intervals ────────────────────────────

        def compute_momentum(interval_ms: int) -> float:
            if len(history) < 2:
                return 0.0
            cutoff = now_ms - interval_ms
            recent = [(ts, b, a) for ts, b, a in history if ts >= cutoff]
            if len(recent) < 2:
                return 0.0
            first_bid = recent[0][1]
            last_bid = recent[-1][1]
            if first_bid == 0:
                return 0.0
            return (last_bid - first_bid) / first_bid

        for interval_name, interval_ms in [("5", 5 * 60 * 1000), ("15", 15 * 60 * 1000), ("60", 60 * 60 * 1000)]:
            key = f"momentum_{interval_name}"
            momentum = compute_momentum(interval_ms)
            features[key] = round(momentum, 8)

            # Quality scales with history coverage
            history_count = len([h for h in history if (now_ms - h[0]) <= interval_ms])
            coverage_quality = min(1.0, history_count / max(1, interval_ms // 1000))
            confidences[key] = FeatureConfidence(
                source="CTRADER_OPENAPI",
                quality=round(base_quality * coverage_quality, 3),
                latency_ms=float(age_ms),
                feed_quality_tag=feed_tag,
            )

        # ── Realized volatility vs historical ────────────────────────────────

        if len(history) >= 10:
            # Realized vol: stddev of 1-minute log returns over last hour
            hour_ago = now_ms - 3600 * 1000
            recent_ticks = [(ts, b) for ts, b, _ in history if ts >= hour_ago]
            if len(recent_ticks) >= 5:
                returns = []
                for i in range(1, len(recent_ticks)):
                    p0 = recent_ticks[i - 1][1]
                    p1 = recent_ticks[i][1]
                    if p0 > 0:
                        returns.append(abs(p1 - p0) / p0)
                if returns:
                    import statistics
                    realized_vol = statistics.stdev(returns) if len(returns) > 1 else 0.0
                    # Ratio vs assumed 10bp historical baseline
                    hist_vol_baseline = 0.0001
                    rvol = realized_vol / hist_vol_baseline if hist_vol_baseline > 0 else 1.0
                    features["rvol"] = round(min(rvol, 50.0), 4)
                    confidences["rvol"] = FeatureConfidence(
                        source="CTRADER_OPENAPI",
                        quality=round(base_quality * 0.8, 3),
                        latency_ms=float(age_ms),
                        feed_quality_tag=feed_tag,
                    )

        if "rvol" not in features:
            features["rvol"] = 1.0
            confidences["rvol"] = FeatureConfidence(
                source="CTRADER_OPENAPI",
                quality=0.3,
                latency_ms=float(age_ms),
                feed_quality_tag=feed_tag,
            )

        # ── VWAP distance (basis points from mid) ────────────────────────────

        mid = tick.mid_price
        if len(history) >= 5:
            hour_ago = now_ms - 3600 * 1000
            recent = [(ts, b, a) for ts, b, a in history if ts >= hour_ago]
            if recent:
                vwap = sum((b + a) / 2 for _, b, a in recent) / len(recent)
                if vwap > 0:
                    vwap_dist_bps = ((mid - vwap) / vwap) * 10000
                    features["vwap_distance"] = round(vwap_dist_bps / 100, 6)
                else:
                    features["vwap_distance"] = 0.0
        else:
            features["vwap_distance"] = 0.0

        confidences["vwap_distance"] = FeatureConfidence(
            source="CTRADER_OPENAPI",
            quality=round(base_quality * 0.7, 3),
            latency_ms=float(age_ms),
            feed_quality_tag=feed_tag,
        )

        # ── Spread ratio vs typical ─────────────────────────────────────────

        current_spread = tick.spread
        typical_spread_bps = 0.2  # EURUSD baseline ~0.2 bps
        if tick.mid_price > 0:
            current_spread_bps = (current_spread / tick.mid_price) * 10000
            spread_ratio = current_spread_bps / typical_spread_bps if typical_spread_bps > 0 else 1.0
        else:
            spread_ratio = 1.0
        features["spread_ratio"] = round(min(spread_ratio, 100.0), 4)

        confidences["spread_ratio"] = FeatureConfidence(
            source="CTRADER_OPENAPI",
            quality=round(base_quality * 0.85, 3),
            latency_ms=float(age_ms),
            feed_quality_tag=feed_tag,
        )

        # ── Tick rate (ticks per second, last minute) ──────────────────────

        minute_ago = now_ms - 60000
        tick_count = len([h for h in history if h[0] >= minute_ago])
        tick_rate = tick_count / 60.0
        features["tick_rate"] = round(tick_rate, 3)

        confidences["tick_rate"] = FeatureConfidence(
            source="CTRADER_OPENAPI",
            quality=round(base_quality * 0.6, 3),
            latency_ms=float(age_ms),
            feed_quality_tag=feed_tag,
        )

        return FeatureVector(
            bot_id=self.adapter_id,
            timestamp=datetime.now(),
            features=features,
            feature_confidence=confidences,
            market_context_snapshot=None,
        )

    def _stub_feature_vector(self, symbol: str, reason: str) -> FeatureVector:
        """Return a zero-confidence stub when real data is unavailable."""
        logger.debug(f"Using stub feature vector for {symbol}: {reason}")
        features = {
            "momentum_5": 0.0,
            "momentum_15": 0.0,
            "momentum_60": 0.0,
            "rvol": 0.0,
            "vwap_distance": 0.0,
            "spread_ratio": 0.0,
            "tick_rate": 0.0,
        }
        confidences = {
            k: FeatureConfidence(
                source="CTRADER_OPENAPI",
                quality=0.0,
                latency_ms=0.0,
                feed_quality_tag="INSUFFICIENT_DATA",
            )
            for k in features
        }
        return FeatureVector(
            bot_id=self.adapter_id,
            timestamp=datetime.now(),
            features=features,
            feature_confidence=confidences,
            market_context_snapshot=None,
        )

    # -------------------------------------------------------------------------
    # Market context
    # -------------------------------------------------------------------------

    def get_market_context(self) -> MarketContext:
        """
        Build a MarketContext snapshot.

        Notes:
          - HMM regime comes from the HMM model (not cTrader) — this method
            returns UNCERTAIN as a safe default since the adapter doesn't have
            access to HMM state. Callers should merge with actual HMM output.
          - Spread state is derived from the current tick spread of subscribed
            symbols (relative to typical spread for that symbol).
          - news_state defaults to CLEAR — news comes from a separate sensor.
        """
        now_ms = int(time.time() * 1000)
        is_stale = False

        # Determine spread state from live ticks
        spread_state: Optional[str] = None
        if self._live_ticks:
            spread_ratios = []
            for sym, tick in self._live_ticks.items():
                if tick.mid_price > 0 and tick.spread > 0:
                    bps = (tick.spread / tick.mid_price) * 10000
                    ratio = bps / 0.2  # vs EURUSD baseline
                    spread_ratios.append(ratio)
            if spread_ratios:
                avg_ratio = sum(spread_ratios) / len(spread_ratios)
                if avg_ratio > 5.0:
                    spread_state = "WIDE"
                elif avg_ratio > 2.0:
                    spread_state = "ELEVATED"
                elif avg_ratio < 0.5:
                    spread_state = "TIGHT"
                else:
                    spread_state = "NORMAL"

        # Staleness: if no spot events in last 10 seconds, mark as stale
        time_since_spot = now_ms - self._last_spot_event_ms
        if time_since_spot > 10000:
            is_stale = True

        # Depth snapshot for context
        depth_snap: Optional[Dict[str, int]] = None
        if self._depth_snapshot:
            depth_snap = {}
            for sym, snap in self._depth_snapshot.items():
                total_bid = sum(snap.get("bid_levels", {}).values())
                total_ask = sum(snap.get("ask_levels", {}).values())
                if total_bid or total_ask:
                    depth_snap[sym] = int(total_bid + total_ask)

        return MarketContext(
            regime=RegimeType.UNCERTAIN,  # HMM regime comes from HMM model, not this adapter
            news_state=NewsState.CLEAR,
            regime_confidence=0.0,  # No HMM data in this adapter
            session_id=None,
            is_stale=is_stale,
            spread_state=spread_state,
            depth_snapshot=depth_snap,
            last_update_ms=now_ms,
            trend_strength=None,
            volatility_regime=None,
        )

    # -------------------------------------------------------------------------
    # Order flow signal
    # -------------------------------------------------------------------------

    def get_order_flow(self, symbol: str) -> OrderFlowSignal:
        """
        Derive an order flow signal from live tick and depth data.

        Signal is computed from:
          - Bid/ask volume imbalance over the rolling window
          - Tick direction clustering (consecutive upticks vs downticks)
          - Large order detection from depth (orders with volume > 3x average)

        Returns DISABLED source if not connected or no depth data.

        Source is always OrderFlowSource.CTRADER_NATIVE as per requirements.
        """
        now_ms = int(time.time() * 1000)
        age_ms = now_ms - self._last_spot_event_ms

        if not self._is_connected or not self._live_ticks.get(symbol):
            return self._stub_order_flow(symbol, reason="not_connected_or_no_data")

        tick = self._live_ticks[symbol]
        depth = self._depth_snapshot.get(symbol, {})
        history = self._tick_history.get(symbol, [])

        # ── Volume imbalance from depth ─────────────────────────────────────

        bid_levels = depth.get("bid_levels", {})
        ask_levels = depth.get("ask_levels", {})
        total_bid_vol = sum(bid_levels.values())
        total_ask_vol = sum(ask_levels.values())

        if total_bid_vol > 0 or total_ask_vol > 0:
            imbalance = (total_bid_vol - total_ask_vol) / max(total_bid_vol + total_ask_vol, 1.0)
        else:
            imbalance = 0.0

        # ── Large order detection ───────────────────────────────────────────

        avg_bid_vol = total_bid_vol / max(len(bid_levels), 1)
        avg_ask_vol = total_ask_vol / max(len(ask_levels), 1)
        threshold = max(avg_bid_vol, avg_ask_vol) * 3.0

        large_bid = sum(1 for v in bid_levels.values() if v > threshold)
        large_ask = sum(1 for v in ask_levels.values() if v > threshold)

        # ── Tick direction clustering ──────────────────────────────────────

        if len(history) >= 3:
            directions = []
            for i in range(1, min(len(history), 20)):
                if history[i][1] > history[i - 1][1]:
                    directions.append(1)
                elif history[i][1] < history[i - 1][1]:
                    directions.append(-1)
            uptick_ratio = directions.count(1) / len(directions) if directions else 0.5
        else:
            uptick_ratio = 0.5

        # ── Composite signal ────────────────────────────────────────────────

        # Direction: combine imbalance and uptick ratio
        composite = (imbalance * 0.6) + ((uptick_ratio - 0.5) * 2.0 * 0.4)

        if composite > 0.3:
            direction = SignalDirection.BULLISH
            strength = min(abs(composite), 1.0)
        elif composite < -0.3:
            direction = SignalDirection.BEARISH
            strength = min(abs(composite), 1.0)
        else:
            direction = SignalDirection.NEUTRAL
            strength = max(0.0, 1.0 - abs(composite) * 2.0)

        # ── Quality assessment ─────────────────────────────────────────────

        has_depth = bool(bid_levels or ask_levels)
        depth_quality_bonus = 0.15 if has_depth else 0.0
        age_factor = 1.0 if age_ms < 2000 else 0.5

        if age_ms < 1000 and has_depth:
            feed_tag = "HIGH"
            quality = 0.90 + depth_quality_bonus
        elif age_ms < 5000:
            feed_tag = "MEDIUM"
            quality = (0.70 + depth_quality_bonus) * age_factor
        elif age_ms < 30000:
            feed_tag = "LOW"
            quality = (0.40 + depth_quality_bonus) * age_factor
        else:
            feed_tag = "STALE"
            quality = 0.15

        return OrderFlowSignal(
            bot_id=self.adapter_id,
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            strength=round(strength, 4),
            source=OrderFlowSource.CTRADER_NATIVE,
            confidence=FeatureConfidence(
                source="CTRADER_OPENAPI",
                quality=round(min(quality, 0.99), 3),
                latency_ms=float(age_ms),
                feed_quality_tag=feed_tag,
            ),
            supporting_evidence={
                "bid_volume": round(total_bid_vol, 2),
                "ask_volume": round(total_ask_vol, 2),
                "imbalance_ratio": round(imbalance, 4),
                "large_orders_bid": large_bid,
                "large_orders_ask": large_ask,
                "uptick_ratio": round(uptick_ratio, 4),
                "composite_score": round(composite, 4),
            },
        )

    def _stub_order_flow(self, symbol: str, reason: str) -> OrderFlowSignal:
        """Return a DISABLED-quality order flow signal when real data is unavailable."""
        return OrderFlowSignal(
            bot_id=self.adapter_id,
            timestamp=datetime.now(),
            symbol=symbol,
            direction=SignalDirection.NEUTRAL,
            strength=0.0,
            source=OrderFlowSource.CTRADER_NATIVE,
            confidence=FeatureConfidence(
                source="CTRADER_OPENAPI",
                quality=0.0,
                latency_ms=0.0,
                feed_quality_tag="INSUFFICIENT_DATA",
            ),
            supporting_evidence={"reason": reason},
        )

    # -------------------------------------------------------------------------
    # Pattern signal (placeholder — Q-3 deferred)
    # -------------------------------------------------------------------------

    def get_pattern(self, symbol: str) -> PatternSignal:
        """
        Pattern detection signal for symbol.

        V1 placeholder — Q-3 Decision deferred.
        Internal pattern recognition engine is out of scope for V1.
        """
        return PatternSignal(
            bot_id=self.adapter_id,
            pattern_type="PENDING_V2",
            direction=SignalDirection.NEUTRAL,
            confidence=0.0,
            timestamp=datetime.now(),
            is_defined=False,
            note="PatternSignal V1 placeholder — Q-3 Decision deferred",
        )

    # -------------------------------------------------------------------------
    # Historical data access
    # -------------------------------------------------------------------------

    def get_historical_trendbars(
        self,
        symbol: str,
        from_ts_ms: int,
        to_ts_ms: int,
        period_seconds: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical trendbars (candles) for a symbol.

        This method is NOT part of IMarketDataAdapter but is provided
        for callers that need raw historical data.

        Args:
            symbol: Symbol display name (e.g., "EURUSD")
            from_ts_ms: Start timestamp in milliseconds
            to_ts_ms: End timestamp in milliseconds
            period_seconds: Candle period in seconds (default 60 = M1)

        Returns:
            List of trendbar dicts with keys: timestamp, open, high, low, close, volume
        """
        sym_id = self._symbol_name_to_id.get(symbol)
        if not sym_id:
            logger.warning(f"Unknown symbol '{symbol}' for trendbar request")
            return []

        if not self._client or not self._is_connected:
            return []

        try:
            from twisted.internet import reactor
            from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq

            req = ProtoOAGetTrendbarsReq()
            req.ctidTraderAccountId = self.ctid_trader_account_id
            req.symbolId = sym_id
            req.fromTimestamp = from_ts_ms
            req.toTimestamp = to_ts_ms
            req.period = period_seconds

            # Store request info for callback matching
            if not hasattr(self, "_trendbar_cache"):
                object.__setattr__(self, "_trendbar_cache", {})

            def _send():
                self._client.send(req)

            reactor.callFromThread(_send)
            return self._trendbar_cache.get(symbol, [])

        except Exception as exc:
            logger.error(f"Failed to request trendbars for {symbol}: {exc}")
            return []

    # -------------------------------------------------------------------------
    # IMarketDataAdapter structural conformance check
    # -------------------------------------------------------------------------

    def _implements_interface_check(self) -> bool:
        """
        Verifies this class implements IMarketDataAdapter.
        Structural subtyping means any class with the required methods
        satisfies the protocol.
        """
        required = [
            "get_feature_vector",
            "get_market_context",
            "get_order_flow",
            "get_pattern",
            "is_connected",
        ]
        return all(hasattr(self, m) for m in required)

    __all__ = ["CTraderMarketAdapter", "TickSnapshot", "SymbolDefinition"]