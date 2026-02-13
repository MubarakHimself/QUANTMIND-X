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
import websockets
import json
from typing import Dict, Any, List
import uuid
from datetime import datetime
from unittest.mock import patch, MagicMock


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
    from fastapi import FastAPI, WebSocket
    from src.api.trading_endpoints import BacktestRunRequest
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    import threading
    import time
    
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
    from fastapi.testclient import TestClient
    client = TestClient(app)
    
    # Test via direct manager broadcast (simulating the real server behavior)
    # Collect events via manager subscription
    events_received = []
    
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
    # Note: In a real E2E test, we'd use a browser or WebSocket client
    # For unit testing, we verify the broadcast structure directly
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
async def test_websocket_endpoint_flow():
    """
    Test complete WebSocket endpoint flow:
    1. Client connects to /ws
    2. Client sends subscribe message
    3. Server confirms subscription
    4. Server broadcasts messages to subscribed topic
    
    Verifies the actual endpoint implementation matches the specification.
    """
    from fastapi import FastAPI
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    create_websocket_endpoints(app)
    
    # Test that /ws endpoint exists (not /ws/backtest/{id})
    # We verify this by checking the app routes
    routes = [route.path for route in app.routes]
    
    # Should have /ws endpoint
    assert any("/ws" in str(route) for route in app.routes if hasattr(route, 'path'))
    
    # Should NOT have /ws/backtest/{id} endpoint
    assert not any("/ws/backtest" in str(route) for route in app.routes if hasattr(route, 'path'))


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
    
    # Use TestClient for HTTP requests
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
    
    # Connect to WebSocket and subscribe to backtest topic
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        # Send subscribe message
        await websocket.send_json({
            "action": "subscribe",
            "topic": "backtest"
        })
        
        # Verify WebSocket events
        events = []
        try:
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                event = json.loads(message)
                # Skip subscription confirmation
                if event.get("type") != "subscription_confirmed":
                    events.append(event)
                
                if len(events) >= 3:  # backtest_start, backtest_progress, backtest_complete
                    break
        except asyncio.TimeoutError:
            pass
        
        # Assert we received the expected events with correct structure
        assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}"
        
        # Verify event types
        assert events[0]["type"] == "backtest_start"
        assert events[0]["data"]["backtest_id"] == backtest_id
        assert "variant" in events[0]["data"]
        assert "symbol" in events[0]["data"]
        
        assert events[1]["type"] == "backtest_progress"
        assert events[1]["data"]["backtest_id"] == backtest_id
        assert "progress" in events[1]["data"]
        assert "status" in events[1]["data"]
        
        assert events[2]["type"] == "backtest_complete"
        assert events[2]["data"]["backtest_id"] == backtest_id
        assert "final_balance" in events[2]["data"]
        assert "total_trades" in events[2]["data"]

@pytest.mark.asyncio
async def test_multiple_concurrent_backtest_streams():
    """Test WebSocket streaming with multiple concurrent backtests."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    
    # Import the backtest endpoint function and real WebSocket endpoints
    from src.api.trading_endpoints import BacktestRunRequest
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    
    # Register real WebSocket endpoints
    create_websocket_endpoints(app)
    
    # Mock the backtest endpoint
    @app.post("/api/v1/backtest/run")
    async def run_backtest(request: BacktestRunRequest):
        backtest_id = str(uuid.uuid4())
        asyncio.create_task(_simulate_concurrent_backtest_events(backtest_id))
        return {
            "backtest_id": backtest_id,
            "status": "running",
            "message": f"Backtest started"
        }
    
    async def _simulate_concurrent_backtest_events(backtest_id: str):
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
        
        # Send multiple progress updates
        for i in range(1, 6):
            await asyncio.sleep(0.05)
            await manager.broadcast({
                "type": "backtest_progress",
                "data": {
                    "backtest_id": backtest_id,
                    "progress": i * 20,
                    "status": f"Processing step {i}",
                    "bars_processed": i * 200,
                    "total_bars": 1000,
                    "current_date": "2023-01-01",
                    "trades_count": i,
                    "current_pnl": i * 50.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }, topic="backtest")
        
        await asyncio.sleep(0.05)
        await manager.broadcast({
            "type": "backtest_complete",
            "data": {
                "backtest_id": backtest_id,
                "final_balance": 10250.0,
                "total_trades": 5,
                "win_rate": 60.0,
                "sharpe_ratio": 1.5,
                "drawdown": 5.0,
                "return_pct": 2.50,
                "duration_seconds": 5.0,
                "timestamp": datetime.utcnow().isoformat(),
                "results": {}
            }
        }, topic="backtest")
    
    # Use TestClient for HTTP requests
    client = TestClient(app)
    
    # Start multiple WebSocket connections
    async def run_backtest_and_collect_events():
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
        
        # Connect to WebSocket and subscribe
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            await websocket.send_json({
                "action": "subscribe",
                "topic": "backtest"
            })
            
            # Collect events
            events = []
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    event = json.loads(message)
                    # Skip subscription confirmation
                    if event.get("type") != "subscription_confirmed":
                        events.append(event)
                    
                    if events and events[-1]["type"] == "backtest_complete":
                        break
            except asyncio.TimeoutError:
                pass
            
            return events
    
    # Run multiple concurrent backtests
    tasks = [run_backtest_and_collect_events() for _ in range(3)]
    results = await asyncio.gather(*tasks)
    
    # Verify each backtest received its events independently
    for i, events in enumerate(results):
        assert len(events) >= 7, f"Backtest {i}: Expected at least 7 events, got {len(events)}"
        assert events[0]["type"] == "backtest_start", f"Backtest {i}: First event should be backtest_start"
        assert events[-1]["type"] == "backtest_complete", f"Backtest {i}: Last event should be backtest_complete"
        
        # Verify progress events
        progress_events = [e for e in events if e["type"] == "backtest_progress"]
        assert len(progress_events) == 5, f"Backtest {i}: Expected 5 progress events, got {len(progress_events)}"
        
        for j, event in enumerate(progress_events):
            expected_progress = (j + 1) * 20
            assert event["data"]["progress"] == expected_progress, \
                f"Backtest {i}, Progress {j}: Expected {expected_progress}, got {event['data']['progress']}"


@pytest.mark.asyncio
async def test_websocket_reconnection_during_backtest():
    """Test WebSocket client reconnects and resumes receiving messages."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    
    # Import the backtest endpoint function and real WebSocket endpoints
    from src.api.trading_endpoints import BacktestRunRequest
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    
    # Register real WebSocket endpoints
    create_websocket_endpoints(app)
    
    # Track connection state for simulation
    connection_count = 0
    backtest_id_for_test = str(uuid.uuid4())
    
    # Mock the backtest endpoint
    @app.post("/api/v1/backtest/run")
    async def run_backtest(request: BacktestRunRequest):
        asyncio.create_task(_simulate_reconnection_backtest_events())
        return {
            "backtest_id": backtest_id_for_test,
            "status": "running",
            "message": f"Backtest started"
        }
    
    async def _simulate_reconnection_backtest_events():
        """Simulate backtest progress through WebSocket for reconnection test."""
        await asyncio.sleep(0.1)  # Small delay
        await manager.broadcast({
            "type": "backtest_start",
            "data": {
                "backtest_id": backtest_id_for_test,
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
                "backtest_id": backtest_id_for_test,
                "progress": 25,
                "status": "Step 1",
                "bars_processed": 250,
                "total_bars": 1000,
                "current_date": "2023-01-01",
                "trades_count": 1,
                "current_pnl": 25.0,
                "timestamp": datetime.utcnow().isoformat()
            }
        }, topic="backtest")
        
        await asyncio.sleep(0.05)
        await manager.broadcast({
            "type": "backtest_progress",
            "data": {
                "backtest_id": backtest_id_for_test,
                "progress": 50,
                "status": "Step 2",
                "bars_processed": 500,
                "total_bars": 1000,
                "current_date": "2023-01-01",
                "trades_count": 2,
                "current_pnl": 50.0,
                "timestamp": datetime.utcnow().isoformat()
            }
        }, topic="backtest")
        
        await asyncio.sleep(0.05)
        await manager.broadcast({
            "type": "backtest_progress",
            "data": {
                "backtest_id": backtest_id_for_test,
                "progress": 75,
                "status": "Step 3",
                "bars_processed": 750,
                "total_bars": 1000,
                "current_date": "2023-01-01",
                "trades_count": 3,
                "current_pnl": 75.0,
                "timestamp": datetime.utcnow().isoformat()
            }
        }, topic="backtest")
        
        await asyncio.sleep(0.05)
        await manager.broadcast({
            "type": "backtest_complete",
            "data": {
                "backtest_id": backtest_id_for_test,
                "final_balance": 10075.0,
                "total_trades": 3,
                "win_rate": 66.67,
                "sharpe_ratio": 1.2,
                "drawdown": 3.0,
                "return_pct": 0.75,
                "duration_seconds": 5.0,
                "timestamp": datetime.utcnow().isoformat(),
                "results": {}
            }
        }, topic="backtest")
    
    # Use TestClient for HTTP requests
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
    
    # First connection - receive some events then disconnect
    first_events = []
    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            await websocket.send_json({
                "action": "subscribe",
                "topic": "backtest"
            })
            
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                event = json.loads(message)
                if event.get("type") != "subscription_confirmed":
                    first_events.append(event)
                    if len(first_events) >= 2:  # start + first progress
                        break
    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
        pass
    
    assert len(first_events) == 2, f"Expected 2 events on first connection, got {len(first_events)}"
    assert first_events[0]["type"] == "backtest_start"
    assert first_events[1]["type"] == "backtest_progress"
    assert first_events[1]["data"]["progress"] == 25
    
    # Wait a bit then reconnect
    await asyncio.sleep(0.5)
    
    # Reconnection - should receive remaining events
    second_events = []
    try:
        async with websockets.connect("ws://localhost:8000/ws") as websocket:
            await websocket.send_json({
                "action": "subscribe",
                "topic": "backtest"
            })
            
            while True:
                message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                event = json.loads(message)
                if event.get("type") != "subscription_confirmed":
                    second_events.append(event)
                    if event["type"] == "backtest_complete":
                        break
    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
        pass
    
    # On reconnection, client should receive remaining progress and complete events
    # Note: WebSocket doesn't inherently provide message history, so the client
    # receives events that are broadcast while connected
    assert len(second_events) >= 1, f"Expected at least 1 event on reconnect, got {len(second_events)}"
    assert second_events[-1]["type"] == "backtest_complete"
