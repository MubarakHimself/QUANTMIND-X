"""
QuantMindLib V1 — cTrader Execution Adapter

Real implementation of IExecutionAdapter for cTrader platform.
Wraps cTrader Open API for order execution via TCP + Protocol Buffers.

Authentication flow:
  1. ProtoOAApplicationAuthReq(clientId, clientSecret) -> ProtoOAApplicationAuthRes
  2. ProtoOAAccountAuthReq(ctidTraderAccountId, accessToken) -> ProtoOAAccountAuthRes

Rate limits:
  - Non-historical requests: 50 req/s
  - Historical requests: 5 req/s
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOANewOrderReq,
    ProtoOACancelOrderReq,
    ProtoOAClosePositionReq,
    ProtoOAAmendOrderReq,
    ProtoOAAmendPositionSLTPReq,
    ProtoOAAccountAuthReq,
    ProtoOAApplicationAuthReq,
    ProtoOASymbolsListReq,
    ProtoOASymbolsListRes,
    ProtoOAReconcileReq,
    ProtoOAReconcileRes,
    ProtoOAOrderListReq,
    ProtoOAExecutionEvent,
    ProtoOAErrorRes,
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOAPosition,
    ProtoOAOrder,
    ProtoOATradeData,
    ProtoOAOrderType,
    ProtoOAPositionStatus,
)
from pydantic import BaseModel, Field, PrivateAttr

from src.library.core.composition.adapter_contracts import IExecutionAdapter
from src.library.core.domain.execution_directive import ExecutionDirective
from src.library.core.domain.trade_intent import TradeIntent
from src.library.core.types.enums import RiskMode, TradeDirection
from twisted.internet import reactor

logger = logging.getLogger(__name__)

# cTrader order type values
_ORDER_TYPE_MARKET = ProtoOAOrderType.MARKET
_ORDER_TYPE_LIMIT = ProtoOAOrderType.LIMIT
_ORDER_TYPE_STOP = ProtoOAOrderType.STOP
_ORDER_TYPE_STOP_LIMIT = ProtoOAOrderType.STOP_LIMIT
_ORDER_TYPE_MARKET_RANGE = ProtoOAOrderType.MARKET_RANGE

# cTrader trade side values
_TRADE_SIDE_BUY = 1
_TRADE_SIDE_SELL = 2


def _ensure_reactor_running() -> None:
    """Start the Twisted reactor on a daemon thread if not already running."""
    if reactor.running:
        return

    def run_reactor():
        reactor.run(installSignalHandlers=False)

    thread = threading.Thread(target=run_reactor, daemon=True, name="ctrader-reactor")
    thread.start()
    # Brief sleep to let reactor thread initialise
    time.sleep(0.05)


def _sync_call(client: Client, message: Any, timeout: float = 15.0) -> Any:
    """
    Send an async cTrader message and block until the response arrives.

    Uses reactor.callFromThread to safely interact with the Twisted reactor
    from any thread. An Event is used to wait for and retrieve the result.
    """
    event = threading.Event()
    result: List[Any] = [None]

    def _send():
        d = client.send(message, responseTimeoutInSeconds=timeout)

        def _on_response(response):
            result[0] = response
            event.set()

        def _on_error(failure):
            result[0] = failure
            event.set()
            return failure

        d.addCallbacks(_on_response, _on_error)

    reactor.callFromThread(_send)
    if not event.wait(timeout=timeout + 2.0):
        raise TimeoutError(f"cTrader API call timed out after {timeout}s")
    if isinstance(result[0], Exception):
        raise result[0]
    return result[0]


class CTraderExecutionAdapter(BaseModel):
    """
    Concrete implementation of IExecutionAdapter for cTrader platform.
    Wraps cTrader Open API (TCP + Protocol Buffers) for order execution.

    Connection lifecycle:
        adapter.connect(client_id, client_secret, ctid_trader_account_id, access_token)
        # ... use adapter ...
        adapter.disconnect()

    Attributes:
        adapter_id: Unique identifier for this adapter instance.
        is_connected: True when both TCP connection and account auth are complete.
    """

    adapter_id: str = "CTRADER_EXECUTION_ADAPTER_V1"
    # Pydantic field is named with underscore prefix to avoid shadowing the
    # is_connected() method required by IExecutionAdapter. Pydantic field
    # descriptors take precedence over same-named methods.
    _is_connected: bool = PrivateAttr(default=False)
    _client: Optional[Client] = PrivateAttr(default=None)
    _account_id: int = PrivateAttr(default=0)
    _access_token: str = PrivateAttr(default="")
    _client_id: str = PrivateAttr(default="")
    _client_secret: str = PrivateAttr(default="")
    _open_orders: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _positions: Dict[int, float] = PrivateAttr(default_factory=dict)
    # symbol_name -> cTrader symbolId
    _symbol_to_id: Dict[str, int] = PrivateAttr(default_factory=dict)
    # cTrader symbolId -> symbol_name
    _id_to_symbol: Dict[int, str] = PrivateAttr(default_factory=dict)
    _executor: Optional[ThreadPoolExecutor] = PrivateAttr(default=None)
    _auth_error: Optional[str] = PrivateAttr(default=None)

    model_config = BaseModel.model_config

    def connect(
        self,
        client_id: str,
        client_secret: str,
        ctid_trader_account_id: int,
        access_token: str,
        demo: bool = True,
    ) -> bool:
        """
        Connect to cTrader Open API and authenticate.

        Performs two-step auth:
          1. Application-level auth via ProtoOAApplicationAuthReq
          2. Account-level auth via ProtoOAAccountAuthReq

        Also fetches the symbol list to build the symbol-name -> symbolId mapping.

        Args:
            client_id: cTrader Connect application client ID.
            client_secret: cTrader Connect application client secret.
            ctid_trader_account_id: The cTID trader account ID (integer).
            access_token: OAuth access token for the trading account.
            demo: If True connect to demo servers (default); False for live.

        Returns:
            True if connection and both auth steps succeed; False otherwise.
        """
        # Store credentials for potential reconnect
        object.__setattr__(self, "_client_id", client_id)
        object.__setattr__(self, "_client_secret", client_secret)
        object.__setattr__(self, "_account_id", ctid_trader_account_id)
        object.__setattr__(self, "_access_token", access_token)

        host = EndPoints.PROTOBUF_DEMO_HOST if demo else EndPoints.PROTOBUF_LIVE_HOST
        port = EndPoints.PROTOBUF_PORT

        _ensure_reactor_running()

        try:
            client = Client(host, port, TcpProtocol)
            object.__setattr__(self, "_client", client)

            # --- Step 1: Application auth ---
            app_auth = ProtoOAApplicationAuthReq(
                clientId=client_id,
                clientSecret=client_secret,
            )
            try:
                app_res = _sync_call(client, app_auth, timeout=15.0)
            except TimeoutError:
                logger.error("cTrader application auth timed out")
                object.__setattr__(self, "_auth_error", "Application auth timed out")
                return False

            # Application auth response has no success/error payload — connection
            # succeeding means auth succeeded. Any error would close the connection.
            if not client.isConnected:
                logger.error("cTrader TCP connection failed")
                object.__setattr__(self, "_auth_error", "TCP connection failed")
                return False

            # --- Step 2: Account auth ---
            acc_auth = ProtoOAAccountAuthReq(
                ctidTraderAccountId=ctid_trader_account_id,
                accessToken=access_token,
            )
            try:
                acc_res = _sync_call(client, acc_auth, timeout=15.0)
            except TimeoutError:
                logger.error("cTrader account auth timed out")
                object.__setattr__(self, "_auth_error", "Account auth timed out")
                return False

            # --- Step 3: Register execution event handler ---
            client.setMessageReceivedCallback(self._handle_message)

            # --- Step 4: Fetch symbol list ---
            self._fetch_symbols(client)

            # Initial reconcile to populate positions cache
            self._refresh_positions()

            object.__setattr__(self, "_is_connected", True)
            logger.info(
                "cTrader execution adapter connected: account_id=%d host=%s",
                ctid_trader_account_id,
                host,
            )
            return True

        except Exception as exc:
            logger.error("cTrader connection failed: %s", exc, exc_info=True)
            object.__setattr__(self, "_auth_error", str(exc))
            return False

    def _fetch_symbols(self, client: Client) -> None:
        """Request the symbol list and build the symbolId mapping."""
        try:
            req = ProtoOASymbolsListReq()
            # Synchronously fetch with a short timeout
            resp = _sync_call(client, req, timeout=10.0)
            sym_map: Dict[str, int] = {}
            id_map: Dict[int, str] = {}
            for sym in getattr(resp, "symbols", []):
                sym_id = getattr(sym, "symbolId", 0)
                sym_name = getattr(sym, "symbolName", "")
                if sym_id and sym_name:
                    sym_map[sym_name] = int(sym_id)
                    id_map[int(sym_id)] = sym_name
            object.__setattr__(self, "_symbol_to_id", sym_map)
            object.__setattr__(self, "_id_to_symbol", id_map)
            logger.debug("Loaded %d cTrader symbols", len(sym_map))
        except Exception as exc:
            logger.warning("Failed to fetch cTrader symbol list: %s", exc)

    def _refresh_positions(self) -> None:
        """Synchronously fetch current positions from cTrader."""
        client = self._client
        if not client:
            return
        try:
            req = ProtoOAReconcileReq(ctidTraderAccountId=self._account_id)
            resp = _sync_call(client, req, timeout=10.0)
            pos_map: Dict[int, float] = {}
            for pos in getattr(resp, "position", []):
                sym_id = int(getattr(pos, "positionId", 0))
                trade_data = getattr(pos, "tradeData", None)
                if trade_data is None:
                    continue
                volume = getattr(trade_data, "volume", 0)
                trade_side = getattr(trade_data, "tradeSide", 0)
                # volume is in cTrader units; convert to lots (divide by 100,000)
                lots = float(volume) / 100000.0
                if trade_side == _TRADE_SIDE_BUY:
                    pos_map[sym_id] = pos_map.get(sym_id, 0.0) + lots
                elif trade_side == _TRADE_SIDE_SELL:
                    pos_map[sym_id] = pos_map.get(sym_id, 0.0) - lots
            object.__setattr__(self, "_positions", pos_map)
        except Exception as exc:
            logger.warning("Failed to refresh cTrader positions: %s", exc)

    def _handle_message(self, client: Client, message: Any) -> None:
        """
        Process inbound messages from cTrader.

        Handles:
          - ProtoOAExecutionEvent: update _open_orders
          - ProtoOAErrorRes: log errors
          - ProtoOAPosition updates via ReconcileRes
        """
        payload_type = getattr(message, "payloadType", 0)
        payload = Protobuf.extract(message)

        if isinstance(payload, ProtoOAExecutionEvent):
            self._on_execution_event(payload)
        elif isinstance(payload, ProtoOAErrorRes):
            error_code = getattr(payload, "errorCode", "unknown")
            description = getattr(payload, "description", "")
            logger.warning("cTrader error: %s — %s", error_code, description)

    def _on_execution_event(self, event: ProtoOAExecutionEvent) -> None:
        """Process execution events to keep _open_orders in sync."""
        exec_type = getattr(event, "executionType", 0)
        order = getattr(event, "order", None)
        order_id = str(getattr(order, "orderId", 0)) if order else None

        # ACCEPTED = 2, FILLED = 3, PARTIAL_FILL = 11, REPLACED = 4, REJECTED = 7
        if exec_type in (2, 3, 11, 4) and order_id and order_id != "0":
            order_status = getattr(order, "orderStatus", 0)
            order_type = getattr(order, "orderType", 0)
            limit_price = getattr(order, "limitPrice", 0)
            stop_price = getattr(order, "stopPrice", 0)
            self._open_orders[order_id] = {
                "status": order_status,
                "type": order_type,
                "limitPrice": limit_price,
                "stopPrice": stop_price,
                "timestamp_ms": int(time.time() * 1000),
            }
        # CANCELLED = 5, EXPIRED = 6
        elif exec_type in (5, 6) and order_id and order_id != "0":
            self._open_orders.pop(order_id, None)
        # REJECTED = 7, CANCEL_REJECTED = 8
        elif exec_type in (7, 8) and order_id and order_id != "0":
            self._open_orders.pop(order_id, None)
            logger.warning(
                "cTrader order %s rejected/cancelled: error=%s",
                order_id,
                getattr(event, "errorCode", "unknown"),
            )

    def disconnect(self) -> bool:
        """Disconnect from cTrader Open API cleanly."""
        client = self._client
        if client is not None:
            try:
                client.stopService()
            except Exception as exc:
                logger.warning("Error stopping cTrader client: %s", exc)
            object.__setattr__(self, "_client", None)
        object.__setattr__(self, "_is_connected", False)
        logger.info("cTrader execution adapter disconnected")
        return True

    def submit_trade(self, intent: TradeIntent) -> ExecutionDirective:
        """
        Submit a trade intent to cTrader for execution.

        Converts TradeIntent to a ProtoOANewOrderReq and sends it over the
        TCP connection. Returns an ExecutionDirective with the authorization
        status.

        Args:
            intent: The trade intent from the bot/decision layer.

        Returns:
            ExecutionDirective with authorization="SUBMITTED" on success,
            "REJECTED" on failure.
        """
        client = self._client
        if not client or not client.isConnected:
            return self._build_rejected_directive(intent, "Not connected to cTrader")

        # Resolve symbol ID
        symbol_id = self._symbol_to_id.get(intent.symbol)
        if symbol_id is None:
            # Fallback: try numeric parse (in case caller already knows the ID)
            try:
                symbol_id = int(intent.symbol)
            except (ValueError, TypeError):
                return self._build_rejected_directive(
                    intent, f"Unknown symbol: {intent.symbol}"
                )

        # Map TradeDirection -> cTrader BUY/SELL
        trade_side = (
            _TRADE_SIDE_BUY
            if intent.direction == TradeDirection.LONG
            else _TRADE_SIDE_SELL
        )

        # Determine order type from risk mode
        order_type = self._risk_mode_to_order_type(intent)

        # Extract quantity; default to 0.01 lots (minimum)
        quantity = 1000  # 0.01 lots in cTrader units (1000 = 0.01 lots)

        # Build SL/TP (cTrader prices are integers in 1e-5 units — no conversion needed
        # here; we use the ticks values directly as price offsets or zero if absent)
        stop_loss = getattr(intent, "stop_ticks", 0) or 0
        take_profit = getattr(intent, "limit_ticks", 0) or 0

        # clientOrderId is our internal order tracking key
        client_order_id = f"{intent.bot_id}_{int(time.time() * 1000)}"

        msg = ProtoOANewOrderReq(
            ctidTraderAccountId=self._account_id,
            symbolId=symbol_id,
            orderType=order_type,
            tradeSide=trade_side,
            volume=quantity,
            stopLoss=float(stop_loss),
            takeProfit=float(take_profit),
            clientOrderId=client_order_id,
        )

        try:
            resp = _sync_call(client, msg, timeout=10.0)
            # ProtoOANewOrderReq triggers ProtoOAExecutionEvent (ORDER_ACCEPTED)
            # asynchronously; the sync call waits for the ProtoOANewOrderRes if one
            # exists, or for the first ExecutionEvent. A successful response means
            # the order was accepted by cTrader.
            logger.info(
                "cTrader order submitted: symbol=%s side=%s volume=%d clientOrderId=%s",
                intent.symbol,
                trade_side,
                quantity,
                client_order_id,
            )
        except TimeoutError:
            return self._build_rejected_directive(intent, "Order request timed out")
        except Exception as exc:
            return self._build_rejected_directive(intent, f"Order submission failed: {exc}")

        # Retrieve risk_mode with safe fallback
        risk_mode: RiskMode = getattr(intent, "risk_mode", RiskMode.STANDARD)

        return ExecutionDirective(
            bot_id=intent.bot_id,
            direction=intent.direction,
            symbol=intent.symbol,
            quantity=float(quantity) / 100000.0,  # Convert back to lots
            risk_mode=risk_mode,
            max_slippage_ticks=getattr(intent, "max_slippage_ticks", 0) or 0,
            stop_ticks=getattr(intent, "stop_ticks", 0) or 1,
            limit_ticks=getattr(intent, "limit_ticks", None),
            timestamp_ms=int(time.time() * 1000),
            authorization="SUBMITTED",
        )

    def _risk_mode_to_order_type(self, intent: TradeIntent) -> int:
        """
        Map the bot's execution intent to a cTrader order type.

        CLAMPED/HALTED risk modes -> LIMIT (more control)
        STANDARD risk mode         -> MARKET (fastest fill)
        """
        risk_mode: RiskMode = getattr(intent, "risk_mode", RiskMode.STANDARD)
        if risk_mode == RiskMode.STANDARD:
            return _ORDER_TYPE_MARKET
        elif risk_mode == RiskMode.CLAMPED:
            return _ORDER_TYPE_LIMIT
        else:  # HALTED
            return _ORDER_TYPE_LIMIT

    def _build_rejected_directive(
        self, intent: TradeIntent, reason: str
    ) -> ExecutionDirective:
        """Build a REJECTED ExecutionDirective for error paths."""
        logger.warning("cTrader order REJECTED: %s", reason)
        return ExecutionDirective(
            bot_id=intent.bot_id,
            direction=intent.direction,
            symbol=intent.symbol,
            quantity=getattr(intent, "quantity", 0.01) or 0.01,
            risk_mode=getattr(intent, "risk_mode", RiskMode.STANDARD),
            max_slippage_ticks=getattr(intent, "max_slippage_ticks", 0) or 0,
            stop_ticks=getattr(intent, "stop_ticks", 0) or 1,
            limit_ticks=getattr(intent, "limit_ticks", None),
            timestamp_ms=int(time.time() * 1000),
            authorization="REJECTED",
        )

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order by its cTrader order ID.

        Args:
            order_id: The cTrader order ID as a string.

        Returns:
            True if the cancel request was sent successfully; False otherwise.
            Note: The actual cancellation confirmation arrives asynchronously
            via an ExecutionEvent.
        """
        client = self._client
        if not client or not client.isConnected:
            logger.warning("cancel_order: not connected")
            return False

        try:
            order_id_int = int(order_id)
        except (ValueError, TypeError):
            logger.warning("cancel_order: invalid order_id format: %s", order_id)
            return False

        msg = ProtoOACancelOrderReq(
            ctidTraderAccountId=self._account_id,
            orderId=order_id_int,
        )
        try:
            _sync_call(client, msg, timeout=10.0)
            logger.info("cTrader cancel request sent: orderId=%s", order_id)
            return True
        except Exception as exc:
            logger.warning("cTrader cancel_order failed: %s", exc)
            return False

    def get_position(self, symbol: str) -> float:
        """
        Get the current net position for a symbol.

        Positive = long (lots), negative = short (lots).
        Returns 0.0 if the symbol is not found or not connected.

        Args:
            symbol: The trading symbol name (e.g. "EURUSD").

        Returns:
            Net position in lots (positive = long, negative = short).
        """
        client = self._client
        if not client or not client.isConnected:
            return 0.0

        # Resolve symbol name -> cTrader symbolId
        symbol_id = self._symbol_to_id.get(symbol)
        if symbol_id is None:
            try:
                symbol_id = int(symbol)
            except (ValueError, TypeError):
                logger.warning("get_position: unknown symbol: %s", symbol)
                return 0.0

        # Refresh positions from cTrader
        self._refresh_positions()

        positions = self._positions
        return positions.get(symbol_id, 0.0)

    def get_open_orders(self) -> List[str]:
        """
        Return the list of open order IDs tracked locally.

        These are orders that were accepted by cTrader and have not yet been
        filled, cancelled, or expired. The list is updated asynchronously via
        execution events.

        Returns:
            List of order ID strings.
        """
        return list(self._open_orders.keys())

    def is_connected(self) -> bool:
        """Return True if the cTrader TCP connection is active and authenticated."""
        client = self._client
        return client is not None and client.isConnected

    __all__ = ["CTraderExecutionAdapter"]
