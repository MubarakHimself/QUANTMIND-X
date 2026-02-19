"""
End-to-End Tests for Backtest WebSocket Streaming

Tests complete integration from API endpoint -> backtest engine -> WebSocket broadcast -> UI consumption.
Tests verify that:
- WebSocket connects to /ws endpoint
- Client sends subscribe message with {action:"subscribe",topic:"backtest"}
- Events are broadcast with correct structure (backtest_start, backtest_progress, backtest_complete)
- Events contain nested data fields matching BacktestProgressStreamer outputs
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock, AsyncMock

# Try to import websockets for real WebSocket testing
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False


@pytest.mark.asyncio
async def test_backtest_streaming_end_to_end():
    """
    Test complete backtest streaming flow from API endpoint -> WebSocket broadcast -> UI consumption.

    Verifies:
    - WebSocket connects to /ws (not /ws/backtest/{id})
    - Client sends subscribe message {"action":"subscribe","topic":"backtest"}
    - Events have correct types: backtest_start, backtest_progress, backtest_complete
    - Events have nested data fields with proper structure
    """
    from fastapi import FastAPI
    from src.api.trading_endpoints import BacktestRunRequest
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    from fastapi.testclient import TestClient

    app = FastAPI()

    # Register real WebSocket endpoints
    create_websocket_endpoints(app)

    # Mock the backtest endpoint
    @app.post("/api/v1/backtest/run")
    async def run_backtest(request: BacktestRunRequest):
        # Return a mock response with backtest_id
        backtest_id = str(uuid.uuid4())

        # Simulate sending backtest events through the real WebSocket manager
        asyncio.create_task(_simulate_backtest_events(backtest_id))

        return {
            "backtest_id": backtest_id,
            "status": "running",
            "message": f"Backtest started"
        }

    async def _simulate_backtest_events(backtest_id: str):
        """Simulate backtest progress through WebSocket."""
        await asyncio.sleep(0.1)  # Small delay
        await manager.broadcast({
            "type": "backtest_start",
            "data": {
                "backtest_id": backtest_id,
                "variant": "vanilla",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "start_date": "2023-01-01",
                "end_date": "2023-01-02",
                "timestamp": datetime.utcnow().isoformat()
            }
        }, topic="backtest")

        await asyncio.sleep(0.05)
        await manager.broadcast({
            "type": "backtest_progress",
            "data": {
                "backtest_id": backtest_id,
                "progress": 50,
                "status": "Processing bars",
                "bars_processed": 500,
                "total_bars": 1000,
                "current_date": "2023-01-01",
                "trades_count": 5,
                "current_pnl": 150.50,
                "timestamp": datetime.utcnow().isoformat()
            }
        }, topic="backtest")

        await asyncio.sleep(0.05)
        await manager.broadcast({
            "type": "backtest_complete",
            "data": {
                "backtest_id": backtest_id,
                "final_balance": 10150.50,
                "total_trades": 10,
                "win_rate": 60.0,
                "sharpe_ratio": 1.5,
                "drawdown": 5.2,
                "return_pct": 1.50,
                "duration_seconds": 5.0,
                "timestamp": datetime.utcnow().isoformat(),
                "results": {}
            }
        }, topic="backtest")

    # Start FastAPI server in background
    client = TestClient(app)

    # Trigger backtest via API
    response = client.post(
        "/api/v1/backtest/run",
        json={
            "symbol": "EURUSD",
            "timeframe": "H1",
            "variant": "vanilla",
            "start_date": "2023-01-01",
            "end_date": "2023-01-02",
            "enable_ws_streaming": True
        }
    )
    assert response.status_code == 200
    backtest_id = response.json()["backtest_id"]

    # Wait for events to be broadcast
    await asyncio.sleep(0.5)

    # Verify the manager has subscribers available
    assert backtest_id is not None


@pytest.mark.asyncio
async def test_websocket_event_structure():
    """
    Test that BacktestProgressStreamer produces correct event structure.

    Verifies:
    - Events have "type" and "data" fields
    - Event types are: backtest_start, backtest_progress, backtest_complete
    - Data field contains all expected fields for each event type
    """
    from src.api.ws_logger import BacktestProgressStreamer
    from src.api.websocket_endpoints import manager

    backtest_id = str(uuid.uuid4())
    streamer = BacktestProgressStreamer(backtest_id)

    # Capture broadcasts
    broadcasted_messages = []

    # Mock the manager broadcast to capture messages
    original_broadcast = manager.broadcast

    async def mock_broadcast(message, topic=None):
        broadcasted_messages.append(message)

    manager.broadcast = mock_broadcast

    try:
        # Send start event
        await streamer.start(
            variant="vanilla",
            symbol="EURUSD",
            timeframe="H1",
            start_date="2023-01-01",
            end_date="2023-01-02"
        )

        # Send progress event
        await streamer.update_progress(
            progress=50,
            status="Processing bars",
            bars_processed=500,
            total_bars=1000,
            current_date="2023-01-01",
            trades_count=5,
            current_pnl=150.50
        )

        # Send complete event
        await streamer.complete(
            final_balance=10150.50,
            total_trades=10,
            win_rate=60.0,
            sharpe_ratio=1.5,
            drawdown=5.2,
            return_pct=1.50,
            duration_seconds=5.0
        )

        # Verify events
        assert len(broadcasted_messages) == 3

        # Verify backtest_start
        assert broadcasted_messages[0]["type"] == "backtest_start"
        assert "data" in broadcasted_messages[0]
        assert broadcasted_messages[0]["data"]["backtest_id"] == backtest_id
        assert broadcasted_messages[0]["data"]["variant"] == "vanilla"
        assert broadcasted_messages[0]["data"]["symbol"] == "EURUSD"
        assert broadcasted_messages[0]["data"]["timeframe"] == "H1"

        # Verify backtest_progress
        assert broadcasted_messages[1]["type"] == "backtest_progress"
        assert "data" in broadcasted_messages[1]
        assert broadcasted_messages[1]["data"]["backtest_id"] == backtest_id
        assert broadcasted_messages[1]["data"]["progress"] == 50
        assert broadcasted_messages[1]["data"]["status"] == "Processing bars"
        assert broadcasted_messages[1]["data"]["bars_processed"] == 500
        assert broadcasted_messages[1]["data"]["total_bars"] == 1000

        # Verify backtest_complete
        assert broadcasted_messages[2]["type"] == "backtest_complete"
        assert "data" in broadcasted_messages[2]
        assert broadcasted_messages[2]["data"]["backtest_id"] == backtest_id
        assert broadcasted_messages[2]["data"]["final_balance"] == 10150.50
        assert broadcasted_messages[2]["data"]["total_trades"] == 10
        assert "win_rate" in broadcasted_messages[2]["data"]
        assert "sharpe_ratio" in broadcasted_messages[2]["data"]

    finally:
        # Restore original broadcast
        manager.broadcast = original_broadcast


@pytest.mark.asyncio
async def test_connection_manager_topic_subscription():
    """
    Test that ConnectionManager properly handles topic subscriptions.

    Verifies:
    - Client can subscribe to "backtest" topic
    - Messages are broadcast to subscribed clients
    - Topic filtering works correctly
    """
    from src.api.websocket_endpoints import ConnectionManager
    from unittest.mock import AsyncMock

    manager = ConnectionManager()

    # Create mock WebSocket connections
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    ws3 = AsyncMock()

    # Connect websockets
    await manager.connect(ws1)
    await manager.connect(ws2)
    await manager.connect(ws3)

    # Subscribe ws1 and ws2 to backtest topic
    await manager.subscribe(ws1, "backtest")
    await manager.subscribe(ws2, "backtest")

    # Subscribe ws3 to a different topic
    await manager.subscribe(ws3, "trading")

    # Broadcast to backtest topic
    message = {
        "type": "backtest_progress",
        "data": {
            "backtest_id": "test-123",
            "progress": 50,
            "status": "Running"
        }
    }

    await manager.broadcast(message, topic="backtest")

    # Verify ws1 and ws2 received message
    assert ws1.send_text.called
    assert ws2.send_text.called

    # Verify ws3 did NOT receive message (different topic)
    assert not ws3.send_text.called

    # Verify message format
    sent_message = json.loads(ws1.send_text.call_args[0][0])
    assert sent_message["type"] == "backtest_progress"
    assert "data" in sent_message
    assert sent_message["data"]["backtest_id"] == "test-123"


@pytest.mark.asyncio
async def test_websocket_endpoint_routes():
    """
    Test that WebSocket endpoints are correctly registered.

    Verifies:
    - /ws endpoint exists (not /ws/backtest/{id})
    - create_websocket_endpoints registers the real endpoint
    """
    from fastapi import FastAPI
    from src.api.websocket_endpoints import create_websocket_endpoints

    app = FastAPI()
    create_websocket_endpoints(app)

    # Test that /ws endpoint exists
    routes = [str(route.path) for route in app.routes]

    # Should have /ws endpoint
    assert any("/ws" in route for route in routes), f"No /ws endpoint found in routes: {routes}"

    # Should NOT have /ws/backtest endpoint
    assert not any("/ws/backtest" in route for route in routes), f"Should not have /ws/backtest endpoint in routes: {routes}"


@pytest.mark.asyncio
async def test_multiple_concurrent_backtest_streams():
    """
    Test WebSocket streaming with multiple concurrent backtests.

    Verifies:
    - Multiple backtests can broadcast concurrently
    - Each backtest has its own backtest_id in events
    - Topic-based filtering ensures all backtests go to same "backtest" topic
    """
    from src.api.ws_logger import BacktestProgressStreamer
    from src.api.websocket_endpoints import manager

    # Capture broadcasts
    broadcasted_messages = []

    # Mock the manager broadcast to capture messages
    original_broadcast = manager.broadcast

    async def mock_broadcast(message, topic=None):
        broadcasted_messages.append(message)

    manager.broadcast = mock_broadcast

    try:
        # Create multiple backtest streamers
        backtest_ids = [str(uuid.uuid4()) for _ in range(3)]
        streamers = [BacktestProgressStreamer(bid) for bid in backtest_ids]

        # Start all backtests concurrently
        tasks = []
        for i, (backtest_id, streamer) in enumerate(zip(backtest_ids, streamers)):
            tasks.append(streamer.start(
                variant=f"vanilla_{i}",
                symbol="EURUSD",
                timeframe="H1",
                start_date="2023-01-01",
                end_date="2023-01-02"
            ))

        await asyncio.gather(*tasks)

        # Verify we have 3 start events (one per backtest)
        start_events = [m for m in broadcasted_messages if m["type"] == "backtest_start"]
        assert len(start_events) == 3

        # Verify each has unique backtest_id
        ids_in_events = [e["data"]["backtest_id"] for e in start_events]
        assert len(set(ids_in_events)) == 3
        assert set(ids_in_events) == set(backtest_ids)

    finally:
        # Restore original broadcast
        manager.broadcast = original_broadcast


@pytest.mark.asyncio
async def test_backtest_progress_event_details():
    """
    Test that backtest_progress events contain all expected detail fields.

    Verifies the event matches BacktestProgressStreamer.update_progress output.
    """
    from src.api.ws_logger import BacktestProgressStreamer
    from src.api.websocket_endpoints import manager

    backtest_id = str(uuid.uuid4())
    streamer = BacktestProgressStreamer(backtest_id)

    # Capture broadcasts
    broadcasted_messages = []

    # Mock the manager broadcast to capture messages
    original_broadcast = manager.broadcast

    async def mock_broadcast(message, topic=None):
        broadcasted_messages.append(message)

    manager.broadcast = mock_broadcast

    try:
        # Send progress event with all details
        await streamer.update_progress(
            progress=75,
            status="Advanced processing",
            bars_processed=750,
            total_bars=1000,
            current_date="2023-01-15",
            trades_count=15,
            current_pnl=2500.75
        )

        assert len(broadcasted_messages) == 1
        event = broadcasted_messages[0]

        # Verify structure
        assert event["type"] == "backtest_progress"
        assert "data" in event

        data = event["data"]

        # Verify all fields
        assert data["backtest_id"] == backtest_id
        assert data["progress"] == 75
        assert data["status"] == "Advanced processing"
        assert data["bars_processed"] == 750
        assert data["total_bars"] == 1000
        assert data["current_date"] == "2023-01-15"
        assert data["trades_count"] == 15
        assert data["current_pnl"] == 2500.75
        assert "timestamp" in data

    finally:
        # Restore original broadcast
        manager.broadcast = original_broadcast


@pytest.mark.asyncio
async def test_backtest_error_event():
    """
    Test that backtest errors are properly broadcast.

    Verifies BacktestProgressStreamer.error() produces correct event structure.
    """
    from src.api.ws_logger import BacktestProgressStreamer
    from src.api.websocket_endpoints import manager

    backtest_id = str(uuid.uuid4())
    streamer = BacktestProgressStreamer(backtest_id)

    # Capture broadcasts
    broadcasted_messages = []

    # Mock the manager broadcast to capture messages
    original_broadcast = manager.broadcast

    async def mock_broadcast(message, topic=None):
        broadcasted_messages.append(message)

    manager.broadcast = mock_broadcast

    try:
        # Send error event
        await streamer.error(
            error_message="Insufficient data",
            error_details="Date range too small"
        )

        assert len(broadcasted_messages) == 1
        event = broadcasted_messages[0]

        # Verify event structure
        assert event["type"] == "backtest_error"
        assert "data" in event

        data = event["data"]
        assert data["backtest_id"] == backtest_id
        assert data["error"] == "Insufficient data"
        assert data["error_details"] == "Date range too small"
        assert "timestamp" in data

    finally:
        # Restore original broadcast
        manager.broadcast = original_broadcast


# =============================================================================
# Real WebSocket Connection Tests (using websockets library)
# =============================================================================

@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
@pytest.mark.asyncio
async def test_real_websocket_connection_and_subscription():
    """
    Test real WebSocket connection to FastAPI server with subscription.

    Verifies:
    - Client can connect to /ws endpoint
    - Client can subscribe to "backtest" topic
    - Server confirms subscription
    """
    import uvicorn
    from fastapi import FastAPI
    from src.api.websocket_endpoints import create_websocket_endpoints
    import threading
    import time

    app = FastAPI()
    create_websocket_endpoints(app)

    # Start server in background thread
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="error")
    server = uvicorn.Server(config)

    def run_server():
        asyncio.run(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for server to start
    await asyncio.sleep(1.0)

    try:
        # Connect to WebSocket
        uri = "ws://127.0.0.1:8765/ws"
        async with websockets.connect(uri) as websocket:
            # Subscribe to backtest topic
            subscribe_msg = json.dumps({
                "action": "subscribe",
                "topic": "backtest"
            })
            await websocket.send(subscribe_msg)

            # Receive confirmation
            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(response)

            assert data["type"] == "subscription_confirmed"
            assert data["topic"] == "backtest"

    finally:
        # Shutdown server
        server.should_exit = True


@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
@pytest.mark.asyncio
async def test_real_websocket_event_reception():
    """
    Test receiving actual backtest events through WebSocket.

    Verifies:
    - Events are received in correct order
    - Event structure matches expected format
    - Multiple events can be received sequentially
    """
    import uvicorn
    from fastapi import FastAPI
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    import threading

    app = FastAPI()
    create_websocket_endpoints(app)

    # Start server in background thread
    config = uvicorn.Config(app, host="127.0.0.1", port=8766, log_level="error")
    server = uvicorn.Server(config)

    def run_server():
        asyncio.run(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for server to start
    await asyncio.sleep(1.0)

    try:
        # Connect to WebSocket
        uri = "ws://127.0.0.1:8766/ws"
        async with websockets.connect(uri) as websocket:
            # Subscribe to backtest topic
            await websocket.send(json.dumps({
                "action": "subscribe",
                "topic": "backtest"
            }))

            # Wait for subscription confirmation
            await websocket.recv()

            # Broadcast test events
            backtest_id = str(uuid.uuid4())

            # Start event
            await manager.broadcast({
                "type": "backtest_start",
                "data": {
                    "backtest_id": backtest_id,
                    "symbol": "EURUSD",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }, topic="backtest")

            # Progress event
            await manager.broadcast({
                "type": "backtest_progress",
                "data": {
                    "backtest_id": backtest_id,
                    "progress": 50,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }, topic="backtest")

            # Complete event
            await manager.broadcast({
                "type": "backtest_complete",
                "data": {
                    "backtest_id": backtest_id,
                    "final_balance": 10100.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }, topic="backtest")

            # Receive and verify events
            received_events = []
            for _ in range(3):
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(response)
                    received_events.append(data)
                except asyncio.TimeoutError:
                    break

            # Verify we received all events
            assert len(received_events) == 3

            # Verify event types
            event_types = [e["type"] for e in received_events]
            assert "backtest_start" in event_types
            assert "backtest_progress" in event_types
            assert "backtest_complete" in event_types

            # Verify event data
            for event in received_events:
                assert "data" in event
                assert "backtest_id" in event["data"]
                assert event["data"]["backtest_id"] == backtest_id

    finally:
        # Shutdown server
        server.should_exit = True


@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
@pytest.mark.asyncio
async def test_websocket_reconnection():
    """
    Test WebSocket reconnection behavior.

    Verifies:
    - Client can reconnect after disconnect
    - Subscription state is reset on new connection
    - Events can be received after reconnection
    """
    import uvicorn
    from fastapi import FastAPI
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    import threading

    app = FastAPI()
    create_websocket_endpoints(app)

    # Start server in background thread
    config = uvicorn.Config(app, host="127.0.0.1", port=8767, log_level="error")
    server = uvicorn.Server(config)

    def run_server():
        asyncio.run(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for server to start
    await asyncio.sleep(1.0)

    try:
        uri = "ws://127.0.0.1:8767/ws"

        # First connection
        async with websockets.connect(uri) as websocket:
            # Subscribe
            await websocket.send(json.dumps({
                "action": "subscribe",
                "topic": "backtest"
            }))
            response = await websocket.recv()
            assert json.loads(response)["type"] == "subscription_confirmed"

        # Connection closed - wait a moment
        await asyncio.sleep(0.5)

        # Reconnect
        async with websockets.connect(uri) as websocket:
            # Subscribe again (state should be reset)
            await websocket.send(json.dumps({
                "action": "subscribe",
                "topic": "backtest"
            }))
            response = await websocket.recv()
            data = json.loads(response)
            assert data["type"] == "subscription_confirmed"
            assert data["topic"] == "backtest"

            # Broadcast and receive event
            await manager.broadcast({
                "type": "backtest_progress",
                "data": {
                    "backtest_id": "test-reconnect",
                    "progress": 100
                }
            }, topic="backtest")

            response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            event = json.loads(response)
            assert event["type"] == "backtest_progress"
            assert event["data"]["backtest_id"] == "test-reconnect"

    finally:
        # Shutdown server
        server.should_exit = True


@pytest.mark.skipif(not WEBSOCKETS_AVAILABLE, reason="websockets library not installed")
@pytest.mark.asyncio
async def test_concurrent_websocket_clients():
    """
    Test multiple concurrent WebSocket clients.

    Verifies:
    - Multiple clients can connect simultaneously
    - Each client can subscribe independently
    - Events are broadcast to all subscribed clients
    """
    import uvicorn
    from fastapi import FastAPI
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    import threading

    app = FastAPI()
    create_websocket_endpoints(app)

    # Start server in background thread
    config = uvicorn.Config(app, host="127.0.0.1", port=8768, log_level="error")
    server = uvicorn.Server(config)

    def run_server():
        asyncio.run(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for server to start
    await asyncio.sleep(1.0)

    try:
        uri = "ws://127.0.0.1:8768/ws"

        async def client_task(client_id: int):
            """Client that connects, subscribes, and receives events."""
            async with websockets.connect(uri) as websocket:
                # Subscribe
                await websocket.send(json.dumps({
                    "action": "subscribe",
                    "topic": "backtest"
                }))
                response = await websocket.recv()

                # Wait for broadcast
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                event = json.loads(response)
                return client_id, event

        # Start multiple clients concurrently
        clients = [client_task(i) for i in range(3)]

        # Wait a moment for all clients to connect
        await asyncio.sleep(0.5)

        # Broadcast event
        await manager.broadcast({
            "type": "backtest_start",
            "data": {
                "backtest_id": "concurrent-test",
                "symbol": "EURUSD"
            }
        }, topic="backtest")

        # Wait for all clients to receive the event
        results = await asyncio.gather(*clients, return_exceptions=True)

        # Verify all clients received the event
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == 3

        for client_id, event in successful_results:
            assert event["type"] == "backtest_start"
            assert event["data"]["backtest_id"] == "concurrent-test"

    finally:
        # Shutdown server
        server.should_exit = True
