"""
Load Tests for V8 HFT Infrastructure: Socket Server

Tests the socket server under extreme load conditions to validate
scalability and performance targets.

**Target: 1000 concurrent EA connections with 100 msg/sec per EA**
**Validates: Requirement 17.9, 17.10**
"""

import pytest
import asyncio
import time
import statistics
from src.router.socket_server import SocketServer, MessageType


class TestSocketLoadScenarios:
    """Load test scenarios for socket server."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.timeout(30)
    async def test_load_1000_concurrent_connections(self):
        """
        Load Test: 1000 concurrent EA connections.
        
        **Target**: Handle 1000 concurrent connections
        **Validates**: Requirement 17.9
        """
        server = SocketServer()
        
        num_eas = 1000
        
        # Register 1000 EAs via heartbeat
        tasks = []
        for i in range(num_eas):
            message = {
                "type": "heartbeat",
                "ea_name": f"LoadEA_{i}",
                "symbol": "EURUSD",
                "magic": 100000 + i
            }
            tasks.append(server.process_message(message))
        
        # Execute all registrations
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        registration_time = (time.time() - start_time) * 1000
        
        # Verify all successful
        assert len(responses) == num_eas
        success_count = sum(1 for r in responses if r["status"] == "success")
        assert success_count == num_eas, \
            f"Only {success_count}/{num_eas} connections successful"
        
        # Verify all connections registered
        stats = server.get_statistics()
        assert stats["active_connections"] == num_eas
        
        # Performance metrics
        avg_registration_time = registration_time / num_eas
        
        print(f"\n=== Load Test: 1000 Concurrent Connections ===")
        print(f"Total EAs: {num_eas}")
        print(f"Registration time: {registration_time:.2f}ms")
        print(f"Average per EA: {avg_registration_time:.3f}ms")
        print(f"Active connections: {stats['active_connections']}")
        
        # Assertions
        assert avg_registration_time < 10.0, \
            f"Average registration time {avg_registration_time:.3f}ms exceeds 10ms limit"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_100_messages_per_second_per_ea(self):
        """
        Load Test: 100 messages/second per EA.
        
        **Target**: Each EA sends 100 msg/sec
        **Validates**: Requirement 17.9, 17.10
        """
        server = SocketServer()
        
        num_eas = 10  # Use 10 EAs for this test (1000 would be 100k msg/sec)
        messages_per_ea = 100
        duration_seconds = 1.0
        
        # Calculate message interval
        interval = duration_seconds / messages_per_ea
        
        async def ea_message_stream(ea_id):
            """Simulate EA sending messages at 100 msg/sec."""
            latencies = []
            
            for msg_id in range(messages_per_ea):
                message = {
                    "type": "heartbeat",
                    "ea_name": f"EA_{ea_id}",
                    "symbol": "EURUSD",
                    "magic": 200000 + ea_id
                }
                
                start_time = time.time()
                response = await server.process_message(message)
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
                
                assert response["status"] == "success"
                
                # Wait for next message interval
                await asyncio.sleep(interval)
            
            return latencies
        
        # Run all EA streams concurrently
        start_time = time.time()
        all_latencies_per_ea = await asyncio.gather(*[
            ea_message_stream(ea_id) for ea_id in range(num_eas)
        ])
        total_time = time.time() - start_time
        
        # Flatten latencies
        all_latencies = [lat for ea_latencies in all_latencies_per_ea for lat in ea_latencies]
        
        # Calculate statistics
        total_messages = num_eas * messages_per_ea
        avg_latency = statistics.mean(all_latencies)
        median_latency = statistics.median(all_latencies)
        max_latency = max(all_latencies)
        p95_latency = sorted(all_latencies)[int(len(all_latencies) * 0.95)]
        p99_latency = sorted(all_latencies)[int(len(all_latencies) * 0.99)]
        throughput = total_messages / total_time
        
        print(f"\n=== Load Test: 100 msg/sec per EA ===")
        print(f"EAs: {num_eas}")
        print(f"Messages per EA: {messages_per_ea}")
        print(f"Total messages: {total_messages}")
        print(f"Duration: {total_time:.2f}s")
        print(f"Throughput: {throughput:.0f} msg/sec")
        print(f"\nLatency Statistics:")
        print(f"  Average: {avg_latency:.3f}ms")
        print(f"  Median: {median_latency:.3f}ms")
        print(f"  P95: {p95_latency:.3f}ms")
        print(f"  P99: {p99_latency:.3f}ms")
        print(f"  Max: {max_latency:.3f}ms")
        
        # Assertions
        assert avg_latency < 5.0, \
            f"Average latency {avg_latency:.3f}ms exceeds 5ms target"
        
        assert p95_latency < 10.0, \
            f"P95 latency {p95_latency:.3f}ms exceeds 10ms acceptable limit"
        
        assert throughput >= (num_eas * 100 * 0.9), \
            f"Throughput {throughput:.0f} msg/sec below 90% of target"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_sustained_high_throughput(self):
        """
        Load Test: Sustained high throughput over 10 seconds.
        
        **Target**: Maintain performance under sustained load
        **Validates**: Requirement 17.9, 17.10
        """
        server = SocketServer()
        
        num_eas = 50
        duration_seconds = 10
        target_msg_per_sec = 50  # 50 msg/sec per EA = 2500 total msg/sec
        
        async def sustained_ea_stream(ea_id):
            """Simulate EA sending sustained message stream."""
            interval = 1.0 / target_msg_per_sec
            message_count = 0
            start_time = time.time()
            
            while (time.time() - start_time) < duration_seconds:
                message = {
                    "type": "heartbeat",
                    "ea_name": f"SustainedEA_{ea_id}",
                    "symbol": "EURUSD",
                    "magic": 300000 + ea_id
                }
                
                response = await server.process_message(message)
                assert response["status"] == "success"
                
                message_count += 1
                await asyncio.sleep(interval)
            
            return message_count
        
        # Run sustained load
        start_time = time.time()
        message_counts = await asyncio.gather(*[
            sustained_ea_stream(ea_id) for ea_id in range(num_eas)
        ])
        actual_duration = time.time() - start_time
        
        # Calculate metrics
        total_messages = sum(message_counts)
        actual_throughput = total_messages / actual_duration
        expected_throughput = num_eas * target_msg_per_sec
        
        print(f"\n=== Load Test: Sustained High Throughput ===")
        print(f"EAs: {num_eas}")
        print(f"Duration: {actual_duration:.2f}s")
        print(f"Total messages: {total_messages}")
        print(f"Actual throughput: {actual_throughput:.0f} msg/sec")
        print(f"Expected throughput: {expected_throughput:.0f} msg/sec")
        print(f"Efficiency: {(actual_throughput/expected_throughput)*100:.1f}%")
        
        # Verify server statistics
        stats = server.get_statistics()
        print(f"\nServer Statistics:")
        print(f"  Active connections: {stats['active_connections']}")
        print(f"  Total messages processed: {stats['message_count']}")
        print(f"  Average latency: {stats['average_latency_ms']:.3f}ms")
        
        # Assertions
        assert actual_throughput >= (expected_throughput * 0.85), \
            f"Throughput {actual_throughput:.0f} below 85% of expected {expected_throughput:.0f}"
        
        assert stats['average_latency_ms'] < 5.0, \
            f"Average latency {stats['average_latency_ms']:.3f}ms exceeds 5ms target"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_load_mixed_message_types_high_volume(self):
        """
        Load Test: High volume of mixed message types.
        
        **Target**: Handle mixed workload efficiently
        **Validates**: Requirement 17.7, 17.9
        """
        server = SocketServer()
        
        num_eas = 100
        messages_per_ea = 50
        
        async def mixed_message_stream(ea_id):
            """Send mixed message types."""
            latencies = []
            
            for msg_id in range(messages_per_ea):
                # Rotate through message types
                msg_type_idx = msg_id % 5
                
                if msg_type_idx == 0:
                    message = {
                        "type": "trade_open",
                        "ea_name": f"MixedEA_{ea_id}",
                        "symbol": "EURUSD",
                        "volume": 0.1,
                        "magic": 400000 + ea_id
                    }
                elif msg_type_idx == 1:
                    message = {
                        "type": "heartbeat",
                        "ea_name": f"MixedEA_{ea_id}",
                        "symbol": "EURUSD",
                        "magic": 400000 + ea_id
                    }
                elif msg_type_idx == 2:
                    message = {
                        "type": "trade_close",
                        "ea_name": f"MixedEA_{ea_id}",
                        "symbol": "EURUSD",
                        "ticket": msg_id,
                        "profit": 10.0
                    }
                elif msg_type_idx == 3:
                    message = {
                        "type": "trade_modify",
                        "ea_name": f"MixedEA_{ea_id}",
                        "ticket": msg_id
                    }
                else:
                    message = {
                        "type": "risk_update",
                        "ea_name": f"MixedEA_{ea_id}",
                        "risk_multiplier": 0.8
                    }
                
                start_time = time.time()
                response = await server.process_message(message)
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
                
                assert response["status"] == "success"
            
            return latencies
        
        # Execute mixed workload
        start_time = time.time()
        all_latencies_per_ea = await asyncio.gather(*[
            mixed_message_stream(ea_id) for ea_id in range(num_eas)
        ])
        total_time = (time.time() - start_time) * 1000
        
        # Flatten latencies
        all_latencies = [lat for ea_latencies in all_latencies_per_ea for lat in ea_latencies]
        
        # Calculate statistics
        total_messages = num_eas * messages_per_ea
        avg_latency = statistics.mean(all_latencies)
        max_latency = max(all_latencies)
        throughput = total_messages / (total_time / 1000)
        
        print(f"\n=== Load Test: Mixed Message Types ===")
        print(f"EAs: {num_eas}")
        print(f"Messages per EA: {messages_per_ea}")
        print(f"Total messages: {total_messages}")
        print(f"Total time: {total_time:.2f}ms")
        print(f"Average latency: {avg_latency:.3f}ms")
        print(f"Max latency: {max_latency:.3f}ms")
        print(f"Throughput: {throughput:.0f} msg/sec")
        
        # Assertions
        assert avg_latency < 5.0, \
            f"Average latency {avg_latency:.3f}ms exceeds 5ms target"
        
        assert throughput > 1000, \
            f"Throughput {throughput:.0f} msg/sec below 1000 msg/sec minimum"


class TestSocketStressScenarios:
    """Stress test scenarios to find breaking points."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_stress_rapid_connection_churn(self):
        """
        Stress Test: Rapid connection/disconnection cycles.
        
        **Validates**: Connection handling robustness
        """
        server = SocketServer()
        
        num_cycles = 100
        eas_per_cycle = 10
        
        for cycle in range(num_cycles):
            # Register EAs
            tasks = []
            for i in range(eas_per_cycle):
                message = {
                    "type": "heartbeat",
                    "ea_name": f"ChurnEA_{cycle}_{i}",
                    "symbol": "EURUSD",
                    "magic": 500000 + (cycle * 100) + i
                }
                tasks.append(server.process_message(message))
            
            responses = await asyncio.gather(*tasks)
            
            # Verify all successful
            for response in responses:
                assert response["status"] == "success"
        
        # Verify final state
        stats = server.get_statistics()
        expected_connections = num_cycles * eas_per_cycle
        
        print(f"\n=== Stress Test: Connection Churn ===")
        print(f"Cycles: {num_cycles}")
        print(f"EAs per cycle: {eas_per_cycle}")
        print(f"Total connections: {expected_connections}")
        print(f"Active connections: {stats['active_connections']}")
        
        assert stats['active_connections'] == expected_connections


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "slow"])

