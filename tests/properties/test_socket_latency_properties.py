"""
Property-Based Tests for V8 HFT Infrastructure: Socket Latency

**Feature: quantmindx-unified-backend, Property 33: Socket Message Latency**

*For any* socket message transmission, round-trip latency SHALL be less than 5ms 
under normal conditions.

**Validates: Requirement 17.2, 17.3**
"""

import pytest
import asyncio
import time
from hypothesis import given, strategies as st, settings, HealthCheck
from src.router.socket_server import SocketServer, MessageType


# Strategy for generating valid message types
message_types = st.sampled_from([
    MessageType.TRADE_OPEN,
    MessageType.TRADE_CLOSE,
    MessageType.TRADE_MODIFY,
    MessageType.HEARTBEAT,
    MessageType.RISK_UPDATE
])

# Strategy for generating EA names
ea_names = st.text(min_size=1, max_size=20, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    min_codepoint=65, max_codepoint=122
))

# Strategy for generating symbols
symbols = st.sampled_from(['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'BTCUSDT', 'ETHUSDT'])

# Strategy for generating volumes
volumes = st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False)

# Strategy for generating magic numbers
magic_numbers = st.integers(min_value=1, max_value=999999)


class TestSocketLatencyProperties:
    """
    Property-based tests for socket message latency.
    
    **Property 33: Socket Message Latency**
    *For any* socket message transmission, round-trip latency SHALL be less than 5ms
    under normal conditions.
    """
    
    @pytest.mark.asyncio
    @given(
        msg_type=message_types,
        ea_name=ea_names,
        symbol=symbols,
        volume=volumes,
        magic=magic_numbers
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    @pytest.mark.timeout(30)
    async def test_property_33_socket_message_latency(
        self, msg_type, ea_name, symbol, volume, magic
    ):
        """
        **Property 33: Socket Message Latency**
        
        *For any* socket message transmission, round-trip latency SHALL be 
        less than 5ms under normal conditions.
        
        **Validates: Requirement 17.2, 17.3**
        """
        server = SocketServer()
        
        # Build message based on type
        if msg_type == MessageType.TRADE_OPEN:
            message = {
                "type": msg_type.value,
                "ea_name": ea_name,
                "symbol": symbol,
                "volume": volume,
                "magic": magic
            }
        elif msg_type == MessageType.TRADE_CLOSE:
            message = {
                "type": msg_type.value,
                "ea_name": ea_name,
                "symbol": symbol,
                "ticket": magic,  # Reuse magic as ticket
                "profit": volume * 10  # Arbitrary profit
            }
        elif msg_type == MessageType.TRADE_MODIFY:
            message = {
                "type": msg_type.value,
                "ea_name": ea_name,
                "ticket": magic
            }
        elif msg_type == MessageType.HEARTBEAT:
            message = {
                "type": msg_type.value,
                "ea_name": ea_name,
                "symbol": symbol,
                "magic": magic
            }
        else:  # RISK_UPDATE
            message = {
                "type": msg_type.value,
                "ea_name": ea_name,
                "risk_multiplier": volume  # Reuse volume as multiplier
            }
        
        # Measure round-trip latency
        start_time = time.time()
        response = await server.process_message(message)
        latency_ms = (time.time() - start_time) * 1000
        
        # Property: Latency must be less than 5ms
        assert latency_ms < 5.0, \
            f"Socket latency {latency_ms:.2f}ms exceeds 5ms target for {msg_type.value}"
        
        # Verify response is valid
        assert response["status"] == "success", \
            f"Message processing failed for {msg_type.value}"
    
    @pytest.mark.asyncio
    @given(
        ea_name=ea_names,
        symbol=symbols,
        magic=magic_numbers
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_33_heartbeat_latency_consistency(
        self, ea_name, symbol, magic
    ):
        """
        **Property 33 Extension: Heartbeat Latency Consistency**
        
        *For any* sequence of heartbeat messages, latency SHALL remain 
        consistently under 5ms.
        
        **Validates: Requirement 17.2**
        """
        server = SocketServer()
        
        message = {
            "type": "heartbeat",
            "ea_name": ea_name,
            "symbol": symbol,
            "magic": magic
        }
        
        # Send multiple heartbeats and measure latency
        latencies = []
        for _ in range(10):
            start_time = time.time()
            response = await server.process_message(message)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
            
            assert response["status"] == "success"
        
        # Property: All latencies must be under 5ms
        max_latency = max(latencies)
        avg_latency = sum(latencies) / len(latencies)
        
        assert max_latency < 5.0, \
            f"Max heartbeat latency {max_latency:.2f}ms exceeds 5ms target"
        
        assert avg_latency < 3.0, \
            f"Average heartbeat latency {avg_latency:.2f}ms exceeds 3ms optimal target"
    
    @pytest.mark.asyncio
    @given(
        ea_name=ea_names,
        symbol=symbols,
        volume=volumes,
        magic=magic_numbers
    )
    @settings(
        max_examples=50,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    async def test_property_33_trade_open_latency_critical_path(
        self, ea_name, symbol, volume, magic
    ):
        """
        **Property 33 Critical Path: Trade Open Latency**
        
        *For any* trade open event (most critical path), latency SHALL be 
        less than 5ms to enable HFT scalping.
        
        **Validates: Requirement 17.2, 17.9**
        """
        server = SocketServer()
        
        message = {
            "type": "trade_open",
            "ea_name": ea_name,
            "symbol": symbol,
            "volume": volume,
            "magic": magic
        }
        
        # Measure trade open latency (critical path)
        start_time = time.time()
        response = await server.handle_trade_open(message)
        latency_ms = (time.time() - start_time) * 1000
        
        # Property: Trade open latency must be under 5ms (critical for HFT)
        assert latency_ms < 5.0, \
            f"Trade open latency {latency_ms:.2f}ms exceeds 5ms HFT target"
        
        # Verify response contains required fields
        assert response["status"] == "success"
        assert "risk_multiplier" in response
        assert "timestamp" in response
        assert isinstance(response["risk_multiplier"], float)
        assert response["risk_multiplier"] >= 0.0
        assert response["risk_multiplier"] <= 1.0


class TestSocketLatencyBenchmarks:
    """
    Benchmark tests for socket latency to validate V8 performance targets.
    """
    
    @pytest.mark.asyncio
    async def test_benchmark_average_latency_under_2ms(self):
        """
        Benchmark: Average latency should be well under 5ms target (aim for <2ms).
        """
        server = SocketServer()
        
        message = {
            "type": "heartbeat",
            "ea_name": "BenchmarkEA",
            "symbol": "EURUSD",
            "magic": 12345
        }
        
        # Run 1000 iterations
        latencies = []
        for _ in range(1000):
            start_time = time.time()
            await server.process_message(message)
            latency_ms = (time.time() - start_time) * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        
        print(f"\n=== Socket Latency Benchmark ===")
        print(f"Average: {avg_latency:.3f}ms")
        print(f"Max: {max_latency:.3f}ms")
        print(f"P95: {p95_latency:.3f}ms")
        print(f"P99: {p99_latency:.3f}ms")
        print(f"Target: <5ms (V8 requirement)")
        print(f"Optimal: <2ms (HFT grade)")
        
        # Assertions
        assert avg_latency < 2.0, f"Average latency {avg_latency:.3f}ms exceeds 2ms optimal target"
        assert p95_latency < 5.0, f"P95 latency {p95_latency:.3f}ms exceeds 5ms target"
        assert p99_latency < 5.0, f"P99 latency {p99_latency:.3f}ms exceeds 5ms target"
        assert max_latency < 10.0, f"Max latency {max_latency:.3f}ms exceeds 10ms absolute limit"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

