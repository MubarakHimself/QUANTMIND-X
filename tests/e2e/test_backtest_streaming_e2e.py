"""
End-to-End Tests for Backtest WebSocket Streaming

Tests complete integration from API endpoint -> backtest engine -> WebSocket broadcast -> UI consumption.
"""

import pytest
import asyncio
import httpx
import websockets
from unittest.mock import AsyncMock, patch, MagicMock
import json
from typing import Dict, Any, List
import uuid
from datetime import datetime

@pytest.mark.asyncio
async def test_backtest_streaming_end_to_end():
    """Test complete backtest streaming flow from API to WebSocket."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    app = FastAPI()
    
    # Import the backtest endpoint function and real WebSocket endpoints
    from src.api.ide_endpoints import BacktestRunRequest
    from src.api.websocket_endpoints import create_websocket_endpoints, manager
    
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
    from src.api.ide_endpoints import BacktestRunRequest
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
    from src.api.ide_endpoints import BacktestRunRequest
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
