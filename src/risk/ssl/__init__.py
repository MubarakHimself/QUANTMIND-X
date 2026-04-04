"""
SSL Package — Survivorship Selection Loop.

Story 18.1: Per-Bot Consecutive Loss Counter & Paper Rotation

Exports:
- SSLCircuitBreaker: Main circuit breaker class
- SSLCircuitBreakerState: State persistence layer
- SSLState: SSL state enum
- BotTier: Bot tier enum
- SSLEventType: SSL event type enum
- SSLCircuitBreakerEvent: SSL event model
"""

from .circuit_breaker import SSLCircuitBreaker
from .state import SSLCircuitBreakerState, SSLState, BotTier, is_valid_transition
from src.events.ssl import SSLCircuitBreakerEvent, SSLEventType, TradeOutcome

__all__ = [
    "SSLCircuitBreaker",
    "SSLCircuitBreakerState",
    "SSLState",
    "BotTier",
    "SSLEventType",
    "SSLCircuitBreakerEvent",
    "TradeOutcome",
]
