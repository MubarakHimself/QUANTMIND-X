"""
QuantMindLib V1 — Shared Enum Types
Sourced from: docs/planning/quantmindlib/07_shared_object_model.md §Schema Type Enums
"""

from enum import StrEnum


class RegimeType(StrEnum):
    """Market regime classification. Used in MarketContext."""

    TREND_STABLE = "TREND_STABLE"
    RANGE_STABLE = "RANGE_STABLE"
    HIGH_CHAOS = "HIGH_CHAOS"
    BREAKOUT_PRIME = "BREAKOUT_PRIME"
    NEWS_EVENT = "NEWS_EVENT"
    UNCERTAIN = "UNCERTAIN"


class NewsState(StrEnum):
    """News event state for kill-zone management. Used in MarketContext."""

    ACTIVE = "ACTIVE"
    KILL_ZONE = "KILL_ZONE"
    CLEAR = "CLEAR"


class TradeDirection(StrEnum):
    """Trade direction (long/short). Used in TradeIntent, ExecutionDirective."""

    LONG = "LONG"
    SHORT = "SHORT"


class SignalDirection(StrEnum):
    """Signal direction for order flow / microstructure signals."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class OrderFlowSource(StrEnum):
    """Source of order flow signal. Used in OrderFlowSignal."""

    CTRADER_NATIVE = "CTRADER_NATIVE"
    PROXY_INFERRED = "PROXY_INFERRED"
    EXTERNAL = "EXTERNAL"
    APPROXIMATED = "APPROXIMATED"
    DISABLED = "DISABLED"


class RiskMode(StrEnum):
    """Risk mode authorization level. Used in RiskEnvelope."""

    STANDARD = "STANDARD"
    CLAMPED = "CLAMPED"
    HALTED = "HALTED"


class ActivationState(StrEnum):
    """Bot activation state lifecycle. Used in BotRuntimeProfile."""

    ACTIVE = "ACTIVE"
    CAUTIOUS = "CAUTIOUS"
    INACTIVE = "INACTIVE"
    DEGRADED = "DEGRADED"
    UNKNOWN = "UNKNOWN"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"


class BotHealth(StrEnum):
    """Bot health status. Used in BotRuntimeProfile."""

    HEALTHY = "HEALTHY"
    CAUTIOUS = "CAUTIOUS"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"
    FAILING = "FAILING"
    OFFLINE = "OFFLINE"


class EvaluationMode(StrEnum):
    """Backtest / evaluation mode. Used in EvaluationResult."""

    VANILLA = "VANILLA"
    SPICED = "SPICED"
    VANILLA_FULL = "VANILLA_FULL"
    SPICED_FULL = "SPICED_FULL"
    MODE_B = "MODE_B"
    MODE_C = "MODE_C"


class RegistryStatus(StrEnum):
    """Bot registry record status. Used in RegistryRecord."""

    ACTIVE = "ACTIVE"
    TRIAL = "TRIAL"
    SUSPENDED = "SUSPENDED"
    ARCHIVED = "ARCHIVED"


class FeatureConfidenceLevel(StrEnum):
    """Feature feed quality level. Used in FeatureConfidence."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    DISABLED = "DISABLED"


class BotTier(StrEnum):
    """
    Bot tier classification — label only, no threshold logic embedded.
    Used in BotRuntimeProfile and SSLCircuitBreaker.
    """

    ELITE = "ELITE"
    PERFORMANCE_TEST = "PERFORMANCE_TEST"
    STANDARD = "STANDARD"
    EVALUATION_CANDIDATE = "EVALUATION_CANDIDATE"
    AT_RISK = "AT_RISK"
    CIRCUIT_BROKEN = "CIRCUIT_BROKEN"


class ErrorSeverity(StrEnum):
    """
    Severity for library-internal engineering errors only. NOT for trading operations.
    Out of scope for V1: NOT wired into exception classes or trading contracts.
    """

    LOW = "LOW"       # Config warnings, optional dependency missing
    MEDIUM = "MEDIUM"  # Validation failure, non-critical bridge timeout
    HIGH = "HIGH"     # Critical config invalid, bridge target unreachable
    CRITICAL = "CRITICAL"  # Validation error preventing bot from running
