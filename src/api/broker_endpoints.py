"""
Broker Management API Endpoints

Provides REST and WebSocket endpoints for broker management,
including auto-detection from MT5 heartbeats and manual configuration.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/brokers", tags=["brokers"])


# ============== Models ==============

class BrokerInfo(BaseModel):
    id: str
    account_id: str
    server: str
    broker_name: str
    balance: float
    equity: float
    margin: float
    leverage: int
    currency: str
    last_seen: datetime
    status: str = "pending"  # connected, disconnected, pending, error
    is_testnet: bool = False
    type: str = "mt5"  # mt5, binance


class BrokerHeartbeat(BaseModel):
    account_id: str
    server: str
    broker_name: str
    balance: float
    equity: float
    margin: float
    leverage: int
    currency: str = "USD"


class ManualBrokerAdd(BaseModel):
    type: str = "mt5"
    account_id: Optional[str] = None
    server: Optional[str] = None
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    is_testnet: bool = False


# ============== Broker Registry ==============

class BrokerRegistry:
    """In-memory broker registry (use database in production)."""

    def __init__(self):
        self.brokers: Dict[str, BrokerInfo] = {}
        self.pending: Dict[str, BrokerInfo] = {}

    def add_or_update(self, heartbeat: BrokerHeartbeat) -> BrokerInfo:
        """Add new broker or update existing from heartbeat."""
        # Check if exists
        for broker_id, broker in self.brokers.items():
            if broker.account_id == heartbeat.account_id and broker.server == heartbeat.server:
                # Update existing
                broker.balance = heartbeat.balance
                broker.equity = heartbeat.equity
                broker.margin = heartbeat.margin
                broker.leverage = heartbeat.leverage
                broker.currency = heartbeat.currency
                broker.last_seen = datetime.now()
                broker.status = "connected"
                return broker

        # Create new broker (pending confirmation)
        import uuid
        broker_id = str(uuid.uuid4())
        broker = BrokerInfo(
            id=broker_id,
            account_id=heartbeat.account_id,
            server=heartbeat.server,
            broker_name=heartbeat.broker_name,
            balance=heartbeat.balance,
            equity=heartbeat.equity,
            margin=heartbeat.margin,
            leverage=heartbeat.leverage,
            currency=heartbeat.currency,
            last_seen=datetime.now(),
            status="pending",
            type="mt5"
        )
        self.pending[broker_id] = broker
        return broker

    def confirm(self, broker_id: str) -> Optional[BrokerInfo]:
        """Confirm a pending broker."""
        if broker_id in self.pending:
            broker = self.pending.pop(broker_id)
            broker.status = "connected"
            self.brokers[broker_id] = broker
            return broker
        return None

    def ignore(self, broker_id: str) -> bool:
        """Ignore and remove a pending broker."""
        if broker_id in self.pending:
            del self.pending[broker_id]
            return True
        return False

    def get(self, broker_id: str) -> Optional[BrokerInfo]:
        """Get broker by ID."""
        return self.brokers.get(broker_id) or self.pending.get(broker_id)

    def get_all(self) -> List[BrokerInfo]:
        """Get all confirmed brokers."""
        return list(self.brokers.values())

    def get_pending(self) -> List[BrokerInfo]:
        """Get all pending brokers."""
        return list(self.pending.values())

    def disconnect(self, broker_id: str) -> bool:
        """Mark broker as disconnected."""
        if broker_id in self.brokers:
            self.brokers[broker_id].status = "disconnected"
            return True
        return False

    def sync(self, broker_id: str) -> Optional[BrokerInfo]:
        """Request sync for a broker."""
        if broker_id in self.brokers:
            self.brokers[broker_id].status = "connected"
            self.brokers[broker_id].last_seen = datetime.now()
            return self.brokers[broker_id]
        return None


# Global registry
broker_registry = BrokerRegistry()

# ============== Account Switching State ==============
class AccountSwitcher:
    """Manages the active account for the trading session."""

    def __init__(self):
        self._active_account_id: Optional[str] = None

    @property
    def active_account_id(self) -> Optional[str]:
        return self._active_account_id

    def set_active_account(self, account_id: str) -> bool:
        """Set the active account. Returns True if successful."""
        # Check if account exists
        for broker_id, broker in broker_registry.brokers.items():
            if broker.account_id == account_id:
                self._active_account_id = account_id
                return True
        # Also check pending accounts
        for broker_id, broker in broker_registry.pending.items():
            if broker.account_id == account_id:
                self._active_account_id = account_id
                return True
        return False

    def get_active_account(self) -> Optional[BrokerInfo]:
        """Get the currently active account details."""
        if not self._active_account_id:
            return None
        for broker in broker_registry.get_all():
            if broker.account_id == self._active_account_id:
                return broker
        return None

    def clear_active_account(self):
        """Clear the active account."""
        self._active_account_id = None


account_switcher = AccountSwitcher()


# ============== WebSocket Manager ==============

class BrokerWebSocketManager:
    """Manages WebSocket connections for broker events."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Broker WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Broker WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        if not self.active_connections:
            return

        message_json = json.dumps(message, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)

    async def notify_broker_detected(self, broker: BrokerInfo):
        """Notify all clients about new broker detection."""
        await self.broadcast({
            "type": "broker_detected",
            "broker": broker.dict()
        })

    async def notify_broker_status(self, broker_id: str, status: str, **kwargs):
        """Notify about broker status change."""
        await self.broadcast({
            "type": "broker_status",
            "broker_id": broker_id,
            "status": status,
            **kwargs
        })


ws_manager = BrokerWebSocketManager()


# ============== HTTP Endpoints ==============

@router.get("", response_model=PaginatedResponse[BrokerInfo])
async def list_brokers(
    status: Optional[str] = Query(None, description="Filter by status"),
    type: Optional[str] = Query(None, description="Filter by type"),
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
):
    """List all registered brokers with pagination."""
    all_brokers = broker_registry.get_all()

    if status:
        all_brokers = [b for b in all_brokers if b.status == status]
    if type:
        all_brokers = [b for b in all_brokers if b.type == type]

    total = len(all_brokers)
    paginated_brokers = all_brokers[offset:offset + limit]

    return PaginatedResponse.create(
        items=paginated_brokers,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/pending", response_model=PaginatedResponse[BrokerInfo])
async def list_pending_brokers(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
):
    """List all pending brokers awaiting confirmation with pagination."""
    all_pending = broker_registry.get_pending()
    total = len(all_pending)
    paginated = all_pending[offset:offset + limit]

    return PaginatedResponse.create(
        items=paginated,
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/{broker_id}", response_model=BrokerInfo)
async def get_broker(broker_id: str):
    """Get broker by ID."""
    broker = broker_registry.get(broker_id)
    if not broker:
        raise HTTPException(status_code=404, detail="Broker not found")
    return broker


@router.post("/heartbeat")
async def broker_heartbeat(heartbeat: BrokerHeartbeat):
    """
    Receive MT5 bridge heartbeat.

    This endpoint is called by the MT5 bridge to report account status.
    If the broker is new, it will be added to pending for confirmation.
    """
    broker = broker_registry.add_or_update(heartbeat)

    # If it's a new broker (pending), notify WebSocket clients
    if broker.status == "pending":
        await ws_manager.notify_broker_detected(broker)
    else:
        # Update status for existing broker
        await ws_manager.notify_broker_status(
            broker.id,
            "connected",
            balance=broker.balance,
            equity=broker.equity,
            margin=broker.margin,
            last_seen=broker.last_seen
        )

    return {"status": "ok", "broker_id": broker.id, "is_new": broker.status == "pending"}


@router.post("/{broker_id}/confirm")
async def confirm_broker(broker_id: str):
    """Confirm a pending broker."""
    broker = broker_registry.confirm(broker_id)
    if not broker:
        raise HTTPException(status_code=404, detail="Pending broker not found")

    await ws_manager.notify_broker_status(broker_id, "connected")
    return {"status": "confirmed", "broker": broker.dict()}


@router.post("/{broker_id}/ignore")
async def ignore_broker(broker_id: str):
    """Ignore and remove a pending broker."""
    if not broker_registry.ignore(broker_id):
        raise HTTPException(status_code=404, detail="Pending broker not found")
    return {"status": "ignored"}


@router.post("/{broker_id}/sync")
async def sync_broker(broker_id: str):
    """Request broker sync."""
    broker = broker_registry.sync(broker_id)
    if not broker:
        raise HTTPException(status_code=404, detail="Broker not found")

    await ws_manager.notify_broker_status(
        broker_id,
        "connected",
        balance=broker.balance,
        equity=broker.equity,
        margin=broker.margin,
        last_seen=broker.last_seen
    )
    return {"status": "synced", "broker": broker.dict()}


@router.post("/{broker_id}/disconnect")
async def disconnect_broker(broker_id: str):
    """Disconnect a broker."""
    if not broker_registry.disconnect(broker_id):
        raise HTTPException(status_code=404, detail="Broker not found")

    await ws_manager.notify_broker_status(broker_id, "disconnected")
    return {"status": "disconnected"}


@router.post("/add")
async def add_broker_manually(broker: ManualBrokerAdd):
    """Manually add a broker."""
    import uuid

    if broker.type == "mt5":
        if not broker.account_id:
            raise HTTPException(status_code=400, detail="account_id required for MT5")

        new_broker = BrokerInfo(
            id=str(uuid.uuid4()),
            account_id=broker.account_id,
            server=broker.server or "",
            broker_name="Manual MT5",
            balance=0,
            equity=0,
            margin=0,
            leverage=100,
            currency="USD",
            last_seen=datetime.now(),
            status="pending",
            type="mt5"
        )
    elif broker.type == "binance":
        if not broker.api_key:
            raise HTTPException(status_code=400, detail="api_key required for Binance")

        new_broker = BrokerInfo(
            id=str(uuid.uuid4()),
            account_id=broker.api_key[:8],  # Use partial key as ID
            server="binance.com",
            broker_name="Binance",
            balance=0,
            equity=0,
            margin=0,
            leverage=1,
            currency="USDT",
            last_seen=datetime.now(),
            status="pending",
            is_testnet=broker.is_testnet,
            type="binance"
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown broker type: {broker.type}")

    # Add to pending
    broker_registry.pending[new_broker.id] = new_broker

    # Notify WebSocket clients
    await ws_manager.notify_broker_detected(new_broker)

    return {"status": "added", "broker": new_broker.dict()}


# ============== WebSocket Endpoint ==============

@router.websocket("/ws")
async def broker_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time broker events."""
    await ws_manager.connect(websocket)

    try:
        # Send current broker list
        await websocket.send_text(json.dumps({
            "type": "broker_list",
            "brokers": [b.dict() for b in broker_registry.get_all()]
        }, default=str))

        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type")

                if msg_type == "confirm_broker":
                    broker_id = message.get("broker_id")
                    broker = broker_registry.confirm(broker_id)
                    if broker:
                        await ws_manager.broadcast({
                            "type": "broker_confirmed",
                            "broker": broker.dict()
                        })

                elif msg_type == "ignore_broker":
                    broker_id = message.get("broker_id")
                    broker_registry.ignore(broker_id)

                elif msg_type == "disconnect_broker":
                    broker_id = message.get("broker_id")
                    broker_registry.disconnect(broker_id)
                    await ws_manager.notify_broker_status(broker_id, "disconnected")

                elif msg_type == "sync_broker":
                    broker_id = message.get("broker_id")
                    broker = broker_registry.sync(broker_id)
                    if broker:
                        await ws_manager.broadcast({
                            "type": "broker_synced",
                            "broker": broker.dict()
                        })

                elif msg_type == "add_broker":
                    # Handle manual add
                    broker_data = message.get("broker", {})
                    new_broker = ManualBrokerAdd(**broker_data)
                    await add_broker_manually(new_broker)

                elif msg_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {data}")

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


# ============== Account Switching Endpoints ==============

class AccountSwitchRequest(BaseModel):
    account_id: str


class AccountSwitchResponse(BaseModel):
    success: bool
    account_id: Optional[str] = None
    account: Optional[Dict[str, Any]] = None
    message: str = ""


@router.get("/accounts", response_model=List[BrokerInfo])
async def list_all_accounts():
    """List all accounts (both connected and pending) with their status."""
    all_accounts = list(broker_registry.brokers.values()) + list(broker_registry.pending.values())
    return all_accounts


@router.get("/accounts/active", response_model=Optional[BrokerInfo])
async def get_active_account():
    """Get the currently active account."""
    return account_switcher.get_active_account()


@router.post("/accounts/active", response_model=AccountSwitchResponse)
async def switch_account(request: AccountSwitchRequest):
    """Switch to a different account."""
    if account_switcher.set_active_account(request.account_id):
        active = account_switcher.get_active_account()
        return AccountSwitchResponse(
            success=True,
            account_id=request.account_id,
            account=active.dict() if active else None,
            message=f"Switched to account {request.account_id}"
        )
    return AccountSwitchResponse(
        success=False,
        account_id=request.account_id,
        message=f"Account {request.account_id} not found"
    )


@router.delete("/accounts/active", response_model=AccountSwitchResponse)
async def clear_active_account():
    """Clear the active account selection."""
    account_switcher.clear_active_account()
    return AccountSwitchResponse(
        success=True,
        message="Active account cleared"
    )
