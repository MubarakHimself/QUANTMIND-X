"""
DPR Package — Daily Performance Ranking Scoring Engine.

Story 17.1: DPR Composite Score Calculation
Story 17.2: DPR Queue Tier Remix

Provides:
- DPRScoringEngine: Core scoring engine for bot composite score calculation
- DPRScoreHistory: Persistence for score history and week-over-week delta
- DPRQueueManager: Queue manager for T1/T3/T2 tier remix
"""

from .scoring_engine import DPRScoringEngine, DPRScore, DPRComponentScores
from .history import DPRScoreHistory, DPRScoreAuditLog
from .queue_manager import DPRQueueManager, DPRQueueAuditLog, DPRSSLAuditLog
from .queue_models import (
    Tier,
    QueueEntry,
    DPRQueueOutput,
    DPRQueueAuditRecord,
)
from .ssl_consumer import DPRSSLConsumer, DPRSSLEventEmitter
from .dpr_emitter import DPRSSLEmitter
from src.events.dpr import SSLEvent, SSLEventType

__all__ = [
    "DPRScoringEngine",
    "DPRScore",
    "DPRComponentScores",
    "DPRScoreHistory",
    "DPRScoreAuditLog",
    "DPRQueueManager",
    "DPRQueueAuditLog",
    "DPRSSLAuditLog",
    "Tier",
    "QueueEntry",
    "DPRQueueOutput",
    "DPRQueueAuditRecord",
    "SSLEvent",
    "SSLEventType",
    "DPRSSLConsumer",
    "DPRSSLEventEmitter",
    "DPRSSLEmitter",
]
