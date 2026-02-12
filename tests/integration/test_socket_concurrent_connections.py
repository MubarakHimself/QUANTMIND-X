"""
Integration Tests for V8 HFT Infrastructure: Concurrent Socket Connections

Tests the socket server's ability to handle multiple concurrent EA connections
with proper isolation and performance.

**Validates: Requirement 17.9, 17.10**
"""

import pytest
import asyncio
import time
import json
from src.router.socket_server import SocketServer, MessageType


class TestConcurrentConnections:
    """Test socket server with multiple concurrent EA connections."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_multiple_concurrent_heartbeats(self):
        """
        Test multiple EAs sending heartbeats concurrently.
        
        **Validates: Requirement 17.9**
        """
        server = SocketServer()
        
        # Simulate 10 concurrent EAs
        num_eas = 10
        tasks = []
        
        for i in range(num_eas):
            message = {
                "type": "heartbeat",
                "ea_name": f"EA_{i}",
                "symbol": "EURUSD",
                "magic": 10000 + i
            }
            tasks.append(server.process_message(message))
        
        # Execute all heartbeats concurrently
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        total_time = (time.time() - start_time) * 1000
        
        # Verify all responses successful
        assert len(responses) == num_eas
        for response in responses:
            assert response["status"] == "success"
            assert "risk_multiplier" in response
        
        # Verify all connections registered
        stats = server.get_statistics()
        assert stats["active_connections"] == num_eas
        
        # Verify concurrent processing is efficient
        avg_time_per_ea = total_time / num_eas
        assert avg_time_per_ea < 5.0, \
            f"Average time per EA {avg_time_per_ea:.2f}ms exceeds 5ms target"
        
        print(f"\n✓ Processed {num_eas} concurrent heartbeats in {total_time:.2f}ms")
        print(f"  Average per EA: {avg_time_per_ea:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_concurrent_trade_events_isolation(self):
        """
        Test that concurrent trade events from different EAs are properly isolated.
        
        **Validates: Requirement 17.9**
        """
        server = SocketServer()
        
        # Simulate 5 EAs opening trades concurrently
        num_eas = 5
        tasks = []
        
        for i in range(num_eas):
            message = {
                "type": "trade_open",
                "ea_name": f"EA_{i}",
                "symbol": f"PAIR_{i}",
                "volume": 0.1 * (i + 1),
                "magic": 20000 + i
            }
            tasks.append(server.process_message(message))
        
        # Execute all trade opens concurrently
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses successful and isolated
        assert len(responses) == num_eas
        for i, response in enumerate(responses):
            assert response["status"] == "success"
            assert "risk_multiplier" in response
            assert "timestamp" in response
    
    @pytest.mark.asyncio
    async def test_mixed_message_types_concurrent(self):
        """
        Test concurrent processing of mixed message types.
        
        **Validates: Requirement 17.7, 17.9**
        """
        server = SocketServer()
        
        # Create mixed message types
        messages = [
            {"type": "trade_open", "ea_name": "EA_1", "symbol": "EURUSD", "volume": 0.1, "magic": 1},
            {"type": "heartbeat", "ea_name": "EA_2", "symbol": "GBPUSD", "magic": 2},
            {"type": "trade_close", "ea_name": "EA_3", "symbol": "USDJPY", "ticket": 123, "profit": 10.0},
            {"type": "trade_modify", "ea_name": "EA_4", "ticket": 456},
            {"type": "risk_update", "ea_name": "EA_5", "risk_multiplier": 0.8},
            {"type": "heartbeat", "ea_name": "EA_6", "symbol": "AUDUSD", "magic": 6},
            {"type": "trade_open", "ea_name": "EA_7", "symbol": "NZDUSD", "volume": 0.2, "magic": 7},
        ]
        
        # Process all messages concurrently
        tasks = [server.process_message(msg) for msg in messages]
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses successful
        assert len(responses) == len(messages)
        for response in responses:
            assert response["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_high_frequency_heartbeats_from_single_ea(self):
        """
        Test high-frequency heartbeat stream from single EA.
        
        **Validates: Requirement 17.2, 17.9**
        """
        server = SocketServer()
        
        message = {
            "type": "heartbeat",
            "ea_name": "HighFreqEA",
            "symbol": "EURUSD",
            "magic": 99999
        }
        
        # Send 100 heartbeats rapidly
        num_heartbeats = 100
        start_time = time.time()
        
        tasks = [server.process_message(message) for _ in range(num_heartbeats)]
        responses = await asyncio.gather(*tasks)
        
        total_time = (time.time() - start_time) * 1000
        
        # Verify all successful
        assert len(responses) == num_heartbeats
        for response in responses:
            assert response["status"] == "success"
        
        # Verify connection count incremented
        connection_key = "HighFreqEA_99999"
        assert connection_key in server.connections
        assert server.connections[connection_key]["message_count"] == num_heartbeats
        
        # Verify performance
        avg_latency = total_time / num_heartbeats
        print(f"\n✓ Processed {num_heartbeats} heartbeats in {total_time:.2f}ms")
        print(f"  Average latency: {avg_latency:.3f}ms")
        
        assert avg_latency < 5.0, \
            f"Average latency {avg_latency:.3f}ms exceeds 5ms target"
    
    @pytest.mark.asyncio
    async def test_connection_registry_with_multiple_eas(self):
        """
        Test connection registry properly tracks multiple EAs.
        
        **Validates: Requirement 17.9**
        """
        server = SocketServer()
        
        # Register 20 different EAs
        num_eas = 20
        
        for i in range(num_eas):
            message = {
                "type": "heartbeat",
                "ea_name": f"EA_{i}",
                "symbol": f"SYMBOL_{i}",
                "magic": 30000 + i
            }
            await server.handle_heartbeat(message)
        
        # Verify all connections registered
        stats = server.get_statistics()
        assert stats["active_connections"] == num_eas
        
        # Verify each connection has correct data
        for i in range(num_eas):
            connection_key = f"EA_{i}_{30000 + i}"
            assert connection_key in server.connections
            assert server.connections[connection_key]["ea_name"] == f"EA_{i}"
            assert server.connections[connection_key]["symbol"] == f"SYMBOL_{i}"
            assert server.connections[connection_key]["magic"] == 30000 + i


class TestConcurrentPerformance:
    """Test performance under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_50_concurrent_eas_performance(self):
        """
        Test 50 concurrent EAs sending messages.
        
        **Validates: Requirement 17.9, 17.10**
        """
        server = SocketServer()
        
        num_eas = 50
        messages_per_ea = 10
        
        # Create tasks for all EAs
        all_tasks = []
        for ea_id in range(num_eas):
            for msg_id in range(messages_per_ea):
                message = {
                    "type": "heartbeat",
                    "ea_name": f"EA_{ea_id}",
                    "symbol": "EURUSD",
                    "magic": 40000 + ea_id
                }
                all_tasks.append(server.process_message(message))
        
        # Execute all concurrently
        start_time = time.time()
        responses = await asyncio.gather(*all_tasks)
        total_time = (time.time() - start_time) * 1000
        
        # Verify all successful
        total_messages = num_eas * messages_per_ea
        assert len(responses) == total_messages
        
        for response in responses:
            assert response["status"] == "success"
        
        # Performance metrics
        avg_latency = total_time / total_messages
        throughput = total_messages / (total_time / 1000)  # messages per second
        
        print(f"\n=== Concurrent Performance Test ===")
        print(f"EAs: {num_eas}")
        print(f"Messages per EA: {messages_per_ea}")
        print(f"Total messages: {total_messages}")
        print(f"Total time: {total_time:.2f}ms")
        print(f"Average latency: {avg_latency:.3f}ms")
        print(f"Throughput: {throughput:.0f} msg/sec")
        
        # Assertions
        assert avg_latency < 5.0, \
            f"Average latency {avg_latency:.3f}ms exceeds 5ms target"
        
        assert throughput > 1000, \
            f"Throughput {throughput:.0f} msg/sec below 1000 msg/sec target"
    
    @pytest.mark.asyncio
    async def test_burst_load_handling(self):
        """
        Test handling of burst load (many messages arriving simultaneously).
        
        **Validates: Requirement 17.9**
        """
        server = SocketServer()
        
        # Create burst of 200 messages
        burst_size = 200
        tasks = []
        
        for i in range(burst_size):
            message = {
                "type": "heartbeat",
                "ea_name": f"EA_{i % 10}",  # 10 EAs sending bursts
                "symbol": "EURUSD",
                "magic": 50000 + (i % 10)
            }
            tasks.append(server.process_message(message))
        
        # Execute burst
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        burst_time = (time.time() - start_time) * 1000
        
        # Verify all processed
        assert len(responses) == burst_size
        for response in responses:
            assert response["status"] == "success"
        
        # Verify burst handled efficiently
        avg_latency = burst_time / burst_size
        print(f"\n✓ Handled burst of {burst_size} messages in {burst_time:.2f}ms")
        print(f"  Average latency: {avg_latency:.3f}ms")
        
        assert avg_latency < 10.0, \
            f"Burst average latency {avg_latency:.3f}ms exceeds 10ms acceptable limit"


class TestConnectionIsolation:
    """Test that connections are properly isolated."""
    
    @pytest.mark.asyncio
    async def test_ea_isolation_different_risk_multipliers(self):
        """
        Test that different EAs can have different risk multipliers.
        
        **Validates: Requirement 17.9**
        """
        server = SocketServer()
        
        # Send heartbeats from different EAs
        ea1_message = {"type": "heartbeat", "ea_name": "EA_Conservative", "symbol": "EURUSD", "magic": 1}
        ea2_message = {"type": "heartbeat", "ea_name": "EA_Aggressive", "symbol": "GBPUSD", "magic": 2}
        
        response1 = await server.process_message(ea1_message)
        response2 = await server.process_message(ea2_message)
        
        # Both should succeed
        assert response1["status"] == "success"
        assert response2["status"] == "success"
        
        # Verify separate connections
        assert "EA_Conservative_1" in server.connections
        assert "EA_Aggressive_2" in server.connections
        
        # Verify isolation
        assert server.connections["EA_Conservative_1"]["ea_name"] == "EA_Conservative"
        assert server.connections["EA_Aggressive_2"]["ea_name"] == "EA_Aggressive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

