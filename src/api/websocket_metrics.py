from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import time

@dataclass
class WebSocketMetrics:
    active_connections: int = 0
    total_messages_sent: int = 0
    messages_by_topic: Optional[Dict[str, int]] = None
    avg_send_latency_ms: float = 0.0
    connection_errors: int = 0
    
    def __post_init__(self):
        if self.messages_by_topic is None:
            self.messages_by_topic = {}

# Global metrics instance (Phase 5.1)
_metrics = WebSocketMetrics()

def get_metrics() -> WebSocketMetrics:
    """Get current WebSocket metrics."""
    return _metrics

def record_message_sent(topic: str, latency_ms: float):
    """Record a message sent event."""
    global _metrics
    _metrics.total_messages_sent += 1
    if _metrics.messages_by_topic is None:
        _metrics.messages_by_topic = {}
    _metrics.messages_by_topic[topic] = _metrics.messages_by_topic.get(topic, 0) + 1
    
    # Update rolling average latency
    alpha = 0.1  # Smoothing factor
    _metrics.avg_send_latency_ms = (
        alpha * latency_ms + (1 - alpha) * _metrics.avg_send_latency_ms
    )

def record_connection_error():
    """Record a connection error."""
    global _metrics
    _metrics.connection_errors += 1

def update_active_connections(count: int):
    """Update active connection count."""
    global _metrics
    _metrics.active_connections = count
