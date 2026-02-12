"""
Unit Tests for V8 HFT Infrastructure: Socket Server

Tests the ZMQ socket server implementation for sub-5ms latency
trade event communication.
"""

import pytest
import asyncio
import json
import time
from src.router.socket_server import SocketServer, MessageType


class TestSocketServerInitialization:
    """Test socket server initialization."""
    
    def test_server_initialization_default(self):
        """Test server initialization with default parameters."""
        server = SocketServer()
        
        assert server.bind_address == "tcp://*:5555"
        assert server.message_count == 0
        assert server.total_latency == 0.0
        assert server.running == False
    
    def test_server_initialization_custom_address(self):
        """Test server initialization with custom bind address."""
        server = SocketServer(bind_address="tcp://*:6666")
        
        assert server.bind_address == "tcp://*:6666"


class TestMessageTypeEnum:
    """Test message type enumeration."""
    
    def test_message_types_exist(self):
        """Test that all message types are defined."""
        assert MessageType.TRADE_OPEN.value == "trade_open"
        assert MessageType.TRADE_CLOSE.value == "trade_close"
        assert MessageType.TRADE_MODIFY.value == "trade_modify"
        assert MessageType.HEARTBEAT.value == "heartbeat"
        assert MessageType.RISK_UPDATE.value == "risk_update"


class TestMessageProcessing:
    """Test message processing logic."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_process_trade_open_message(self):
        """Test processing trade open message."""
        server = SocketServer()
        
        message = {
            "type": "trade_open",
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "volume": 0.1,
            "magic": 12345
        }
        
        response = await server.process_message(message)
        
        assert response["status"] == "success"
        assert "risk_multiplier" in response
        assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_process_trade_close_message(self):
        """Test processing trade close message."""
        server = SocketServer()
        
        message = {
            "type": "trade_close",
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "ticket": 123456,
            "profit": 10.50
        }
        
        response = await server.process_message(message)
        
        assert response["status"] == "success"
        assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_process_heartbeat_message(self):
        """Test processing heartbeat message."""
        server = SocketServer()
        
        message = {
            "type": "heartbeat",
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "magic": 12345
        }
        
        response = await server.process_message(message)
        
        assert response["status"] == "success"
        assert "risk_multiplier" in response
        assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_process_invalid_message_type(self):
        """Test processing message with invalid type."""
        server = SocketServer()
        
        message = {
            "type": "invalid_type",
            "ea_name": "TestEA"
        }
        
        response = await server.process_message(message)
        
        assert response["status"] == "error"
        assert "Invalid message type" in response["message"]
    
    @pytest.mark.asyncio
    async def test_process_missing_message_type(self):
        """Test processing message without type field."""
        server = SocketServer()
        
        message = {
            "ea_name": "TestEA",
            "symbol": "EURUSD"
        }
        
        response = await server.process_message(message)
        
        assert response["status"] == "error"
        assert "Missing message type" in response["message"]


class TestConnectionRegistry:
    """Test connection registry functionality."""
    
    @pytest.mark.asyncio
    async def test_heartbeat_updates_connection_registry(self):
        """Test that heartbeat updates connection registry."""
        server = SocketServer()
        
        message = {
            "type": "heartbeat",
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "magic": 12345
        }
        
        # Send heartbeat
        await server.handle_heartbeat(message)
        
        # Check connection registry
        connection_key = "TestEA_12345"
        assert connection_key in server.connections
        assert server.connections[connection_key]["ea_name"] == "TestEA"
        assert server.connections[connection_key]["symbol"] == "EURUSD"
        assert server.connections[connection_key]["magic"] == 12345
    
    @pytest.mark.asyncio
    async def test_multiple_heartbeats_increment_count(self):
        """Test that multiple heartbeats increment message count."""
        server = SocketServer()
        
        message = {
            "type": "heartbeat",
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "magic": 12345
        }
        
        # Send multiple heartbeats
        await server.handle_heartbeat(message)
        await server.handle_heartbeat(message)
        await server.handle_heartbeat(message)
        
        # Check message count
        connection_key = "TestEA_12345"
        assert server.connections[connection_key]["message_count"] == 3


class TestRiskMultiplier:
    """Test risk multiplier retrieval."""
    
    @pytest.mark.asyncio
    async def test_get_risk_multiplier_default(self):
        """Test getting default risk multiplier."""
        server = SocketServer()
        
        multiplier = await server.get_risk_multiplier("TestEA")
        
        assert multiplier == 1.0  # Default multiplier


class TestServerStatistics:
    """Test server statistics."""
    
    def test_get_statistics_initial(self):
        """Test getting statistics with no messages."""
        server = SocketServer()
        
        stats = server.get_statistics()
        
        assert stats["message_count"] == 0
        assert stats["average_latency_ms"] == 0.0
        assert stats["active_connections"] == 0
    
    @pytest.mark.asyncio
    async def test_get_statistics_after_messages(self):
        """Test getting statistics after processing messages."""
        server = SocketServer()
        
        # Process some messages
        message1 = {"type": "heartbeat", "ea_name": "EA1", "symbol": "EURUSD", "magic": 1}
        message2 = {"type": "heartbeat", "ea_name": "EA2", "symbol": "GBPUSD", "magic": 2}
        
        await server.handle_heartbeat(message1)
        await server.handle_heartbeat(message2)
        
        stats = server.get_statistics()
        
        assert stats["active_connections"] == 2
        assert "EA1_1" in stats["connections"]
        assert "EA2_2" in stats["connections"]


class TestLatencyTracking:
    """Test latency tracking functionality."""
    
    def test_latency_calculation(self):
        """Test that latency is calculated correctly."""
        server = SocketServer()
        
        # Simulate message processing
        server.message_count = 10
        server.total_latency = 35.5  # Total 35.5ms over 10 messages
        
        stats = server.get_statistics()
        
        assert stats["average_latency_ms"] == 3.55


class TestTradeEventHandlers:
    """Test trade event handler methods."""
    
    @pytest.mark.asyncio
    async def test_handle_trade_open(self):
        """Test trade open handler."""
        server = SocketServer()
        
        message = {
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "volume": 0.1,
            "magic": 12345
        }
        
        response = await server.handle_trade_open(message)
        
        assert response["status"] == "success"
        assert response["risk_multiplier"] == 1.0
        assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_handle_trade_close(self):
        """Test trade close handler."""
        server = SocketServer()
        
        message = {
            "ea_name": "TestEA",
            "symbol": "EURUSD",
            "ticket": 123456
        }
        
        response = await server.handle_trade_close(message)
        
        assert response["status"] == "success"
        assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_handle_trade_modify(self):
        """Test trade modify handler."""
        server = SocketServer()
        
        message = {
            "ea_name": "TestEA",
            "ticket": 123456
        }
        
        response = await server.handle_trade_modify(message)
        
        assert response["status"] == "success"
        assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_handle_risk_update(self):
        """Test risk update handler."""
        server = SocketServer()
        
        message = {
            "ea_name": "TestEA",
            "risk_multiplier": 0.5
        }
        
        response = await server.handle_risk_update(message)
        
        assert response["status"] == "success"
        assert "timestamp" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
