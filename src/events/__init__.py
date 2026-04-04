"""
Events Package.

Contains event models for the trading system.
"""
from .regime import RegimeShiftEvent, RegimeSuitability, RegimeType
from .chaos import (
    ChaosEvent,
    ChaosLevel,
    ForcedExitOutcome,
    KillSwitchResult,
    RVOLWarningEvent,
)
from .tilt import (
    TiltState,
    TiltPhase,
    TiltPhaseEvent,
    TiltSessionBoundaryEvent,
    TiltChaosSuspendEvent,
    TiltChaosResumeEvent,
    TiltTransitionAuditLog,
    REGIME_PERSISTENCE_SECONDS,
)
from .session_template import SessionTemplateEvent, SessionTemplateEventType
from .cooldown import (
    CooldownState,
    CooldownPhase,
    CooldownPhaseEvent,
    InterSessionCooldownStateEvent,
    InterSessionCooldownCompletionEvent,
    NYQueueCandidate,
    CooldownAuditLog,
    COOLDOWN_STATE_TO_PHASE,
    STEP_WINDOWS,
)
from .dpr import (
    DPRComponentScores,
    DPRScoreEvent,
    DPRConcernEvent,
    DPR_WEIGHTS,
)
from .ssl import (
    SSLState,
    SSLEventType,
    BotTier,
    SSLCircuitBreakerEvent,
)

__all__ = [
    "RegimeShiftEvent",
    "RegimeSuitability",
    "RegimeType",
    "ChaosEvent",
    "ChaosLevel",
    "ForcedExitOutcome",
    "KillSwitchResult",
    "RVOLWarningEvent",
    "TiltState",
    "TiltPhase",
    "TiltPhaseEvent",
    "TiltSessionBoundaryEvent",
    "TiltChaosSuspendEvent",
    "TiltChaosResumeEvent",
    "TiltTransitionAuditLog",
    "REGIME_PERSISTENCE_SECONDS",
    "SessionTemplateEvent",
    "SessionTemplateEventType",
    "CooldownState",
    "CooldownPhase",
    "CooldownPhaseEvent",
    "InterSessionCooldownStateEvent",
    "InterSessionCooldownCompletionEvent",
    "NYQueueCandidate",
    "CooldownAuditLog",
    "COOLDOWN_STATE_TO_PHASE",
    "STEP_WINDOWS",
    # DPR Events (Story 17.1)
    "DPRComponentScores",
    "DPRScoreEvent",
    "DPRConcernEvent",
    "DPR_WEIGHTS",
    # SSL Events (Story 18.1)
    "SSLState",
    "SSLEventType",
    "BotTier",
    "SSLCircuitBreakerEvent",
]
