"""
DPR Queue Manager — Daily Performance Ranking Queue Tier Remix.

Story 17.2: DPR Queue Tier Remix

Provides DPRQueueManager class for assembling ranked session queues
with TIER_1, TIER_2, TIER_3 interleaving based on DPR composite scores.

Queue Ordering (per AC #1):
    TIER_1 recovery bots: positions 1-N (if 2 consecutive paper wins)
    TIER_3 bots: next 40% of positions
    TIER_2 bots: remaining positions (always after TIER_3)

Specialist Boost (per AC #3):
    +5 DPR boost for SESSION_SPECIALIST bots in their specialist session
    Specialist at position 1 takes priority over TIER_1 recovery candidate

Concern Sub-Queue (per AC #4):
    Bots with SESSION_CONCERN flag stay in DPR-ranked position
    but are flagged for Workflow 3 Queue Re-rank at 17:30 GMT

NY Hybrid Queue (per AC #5, Story 16.3):
    Position 1: Best London performer (SESSION_SPECIALIST + TIER_1 recovery)
    Position 2: TIER_1 recovery (if not already position 1)
    Positions 3-N: TIER_3 DPR-ranked bots
    Remaining: TIER_2 fresh candidates (always after TIER_3)

Queue Lock (per AC #6):
    Queue is locked at session start
    Mid-session SSL events are QUEUED, not applied
    Events applied at next Dead Zone queue lock

Per NFR-M2: DPR is a synchronous queue manager — NO LLM calls in remix path.
Per NFR-D1: All queue outputs logged with timestamps before session lock.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, Any

from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Index, Text

from src.database.models import SessionLocal
from src.database.models.base import Base
from src.risk.dpr.scoring_engine import DPRScoringEngine
from src.risk.dpr.queue_models import (
    Tier,
    QueueEntry,
    DPRQueueOutput,
    DPRQueueAuditRecord,
)
from src.events.dpr import SSLEvent, SSLEventType


logger = logging.getLogger(__name__)


class DPRQueueAuditLog(Base):
    """
    DPR Queue Audit Log for immutable queue decision records.

    Persisted to SQLite for full audit trail per AC #7 and NFR-D1.

    Attributes:
        id: Primary key
        session_id: Session identifier
        bot_id: Bot identifier
        queue_position: Assigned queue position
        dpr_composite_score: DPR score at queue time
        tier: Bot tier
        specialist_flag: Whether SESSION_SPECIALIST tag was present
        concern_flag: Whether SESSION_CONCERN flag was set
        timestamp_utc: When decision was made
    """
    __tablename__ = 'dpr_queue_audit'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), nullable=False, index=True)
    bot_id = Column(String(100), nullable=False, index=True)
    queue_position = Column(Integer, nullable=False)
    dpr_composite_score = Column(Integer, nullable=False)
    tier = Column(String(20), nullable=False)
    specialist_flag = Column(Boolean, nullable=False, default=False)
    concern_flag = Column(Boolean, nullable=False, default=False)
    timestamp_utc = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )

    __table_args__ = (
        Index('idx_queue_audit_session_bot', 'session_id', 'bot_id'),
        Index('idx_queue_audit_timestamp', 'timestamp_utc'),
    )

    def __repr__(self):
        return (
            f"<DPRQueueAuditLog(id={self.id}, bot={self.bot_id}, "
            f"session={self.session_id}, pos={self.queue_position})>"
        )


class DPRSSLAuditLog(Base):
    """
    DPR SSL Event Audit Log for immutable SSL event records.

    Story 18.2: SSL → DPR Integration

    Persisted to SQLite for full audit trail per AC #6 and NFR-D1.

    Attributes:
        id: Primary key
        session_id: Session identifier
        bot_id: Bot identifier
        magic_number: MT5 magic number
        ssl_event_type: Type of SSL event (move_to_paper, recovery_step_1, etc.)
        ssl_state: SSL state at time of event (paper, recovery, retired)
        tier: Paper tier if applicable (TIER_1 or TIER_2)
        dpr_composite_score: DPR composite score at time of event
        consecutive_losses: Consecutive loss count at event time
        recovery_win_count: Recovery win count at event time
        timestamp_utc: When event occurred
        metadata_json: JSON-encoded additional context
    """
    __tablename__ = 'dpr_ssl_audit'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), nullable=False, index=True)
    bot_id = Column(String(100), nullable=False, index=True)
    magic_number = Column(String(50), nullable=True)
    ssl_event_type = Column(String(50), nullable=False)
    ssl_state = Column(String(20), nullable=False)
    tier = Column(String(20), nullable=True)
    dpr_composite_score = Column(Integer, nullable=False)
    consecutive_losses = Column(Integer, nullable=False, default=0)
    recovery_win_count = Column(Integer, nullable=False, default=0)
    timestamp_utc = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True
    )
    metadata_json = Column(Text, nullable=True)

    __table_args__ = (
        Index('idx_ssl_audit_session_bot', 'session_id', 'bot_id'),
        Index('idx_ssl_audit_timestamp', 'timestamp_utc'),
        Index('idx_ssl_audit_event_type', 'ssl_event_type'),
    )

    def __repr__(self):
        return (
            f"<DPRSSLAuditLog(id={self.id}, bot={self.bot_id}, "
            f"event={self.ssl_event_type}, state={self.ssl_state})>"
        )


class DPRQueueManager:
    """
    DPR Queue Manager for assembling ranked session queues.

    Interleaves TIER_1, TIER_2, TIER_3 bots based on DPR composite scores
    with specialist boosts and concern flag handling.

    Attributes:
        scoring_engine: DPRScoringEngine for score calculations
        db_session: SQLAlchemy session for database access
    """

    def __init__(
        self,
        scoring_engine: Optional[DPRScoringEngine] = None,
        db_session: Optional[Session] = None,
    ):
        """
        Initialize DPR Queue Manager.

        Args:
            scoring_engine: DPRScoringEngine instance (creates if None)
            db_session: SQLAlchemy session (creates if None)
        """
        self._scoring_engine = scoring_engine
        self._db_session = db_session
        self._locked_queues: Dict[str, bool] = {}  # session_id -> locked
        self._queued_events: List[SSLEvent] = []

    @property
    def scoring_engine(self) -> DPRScoringEngine:
        """Get or create DPR scoring engine."""
        if self._scoring_engine is None:
            self._scoring_engine = DPRScoringEngine(db_session=self.db_session)
        return self._scoring_engine

    @property
    def db_session(self) -> Session:
        """Get or create database session."""
        if self._db_session is None:
            self._db_session = SessionLocal()
        return self._db_session

    def tier_assignment(self, bot_id: str) -> Tier:
        """
        Get tier assignment for a bot from circuit breaker data.

        AC #1: Bots are assigned to tiers based on performance:
        - TIER_1: Recovery-eligible quarantined bots with 2 consecutive paper wins
        - TIER_2: Fresh candidates from AlphaForge (minimum 2-week paper)
        - TIER_3: Active bots not in TIER_1 or TIER_2

        TIER_2 assignment per story spec:
        - Bots are assigned TIER_2 when they are fresh candidates from AlphaForge
        - Fresh AlphaForge candidates have NEVER been live (mode = DEMO)
        - They must have demonstrated at least 2 weeks of consistent paper performance
        - TIER_2 is NOT based on quarantine duration - quarantined bots were LIVE
          and failed, so they cannot be fresh AlphaForge candidates

        Args:
            bot_id: Bot identifier

        Returns:
            Tier classification (TIER_1, TIER_2, TIER_3)
        """
        from src.database.repositories.circuit_breaker_repository import CircuitBreakerRepository
        from src.database.models.base import TradingMode

        repo = CircuitBreakerRepository()
        cb = repo.get_by_bot_id(bot_id)

        if cb is None:
            return Tier.TIER_3  # Default to TIER_3 if no circuit breaker record

        # TIER_1: Quarantined bots in recovery (2+ consecutive wins)
        # These WERE live and failed - now recovering back to live
        if cb.is_quarantined:
            recovery_step = self.get_recovery_step(bot_id)
            if recovery_step == 2:
                return Tier.TIER_1  # Eligible for recovery - goes to queue position 1
            # recovery_step 0 or 1: still quarantined, not TIER_2 (quarantine != fresh AlphaForge)
            return Tier.TIER_3

        # TIER_2: Fresh candidates from AlphaForge (NEVER live, minimum 2-week paper)
        # These are bots that came through AlphaForge paper trading and have never been live
        if cb.mode == TradingMode.DEMO:
            # Check paper trading duration - fresh AlphaForge candidates need 2+ weeks
            days_in_paper = (datetime.now(timezone.utc) - cb.created_at).days
            if days_in_paper >= 14:
                return Tier.TIER_2  # Fresh AlphaForge candidate with 2+ week paper history
            # Less than 2 weeks in paper - still maturing, not yet TIER_2 eligible
            return Tier.TIER_3

        # Not quarantined and not DEMO mode = active live bot = TIER_3
        return Tier.TIER_3

    def get_recovery_step(self, bot_id: str) -> int:
        """
        Get recovery step for a TIER_1 bot.

        Recovery progression:
        - 0: Not in recovery
        - 1: First consecutive paper win achieved
        - 2: Two consecutive paper wins — eligible for queue position 1

        Args:
            bot_id: Bot identifier

        Returns:
            Recovery step: 0, 1, or 2
        """
        from src.database.repositories.circuit_breaker_repository import CircuitBreakerRepository
        from src.database.models import BotCircuitBreaker

        repo = CircuitBreakerRepository()
        cb = repo.get_by_bot_id(bot_id)

        if cb is None:
            return 0

        if not cb.is_quarantined:
            return 0

        # Use consecutive_session_wins as proxy for recovery wins
        # This tracks consecutive winning sessions
        wins = cb.consecutive_session_wins

        if wins >= 2:
            return 2  # Eligible for recovery
        elif wins == 1:
            return 1  # First win achieved
        else:
            return 0  # Not in recovery

    def get_recovery_eligible_bots(self) -> List[str]:
        """
        Get list of TIER_1 bots eligible for recovery (2 consecutive paper wins).

        AC #2: TIER_1 recovery eligible bots with 2 consecutive paper wins
        are placed at queue position 1.

        Returns:
            List of bot IDs eligible for recovery
        """
        from src.database.repositories.circuit_breaker_repository import CircuitBreakerRepository

        repo = CircuitBreakerRepository()
        quarantined = repo.get_quarantined()

        eligible = []
        for cb in quarantined:
            if self.get_recovery_step(cb.bot_id) == 2:
                eligible.append(cb.bot_id)

        return eligible

    def tier_sort(
        self,
        tier_bots: List[str],
        scores: Dict[str, int],
    ) -> List[str]:
        """
        Sort bots within a tier by DPR composite score descending.

        Uses 4-level tie-break cascade from DPRScoringEngine:
        1. Higher session-specific win rate wins
        2. Lower max drawdown wins
        3. Higher trade count wins
        4. Lower Magic Number wins

        Args:
            tier_bots: List of bot IDs in the tier
            scores: Dict mapping bot_id -> DPR composite score

        Returns:
            Sorted list of bot IDs (highest score first)
        """
        def sort_key(bot_id: str) -> Tuple:
            score = scores.get(bot_id, 0)
            return (-score, bot_id)  # Negative for descending

        return sorted(tier_bots, key=sort_key)

    def _build_queue_entry(
        self,
        bot_id: str,
        queue_position: int,
        dpr_composite_score: int,
        tier: Tier,
        specialist_session: Optional[str],
        specialist_boost_applied: bool,
        concern_flag: bool,
        recovery_step: int,
        in_concern_subqueue: bool,
    ) -> QueueEntry:
        """
        Build a QueueEntry with SSL state integration (AC#7).

        Gets current SSL state for the bot and includes it in the entry.

        Args:
            bot_id: Bot identifier
            queue_position: 1-indexed position in queue
            dpr_composite_score: DPR composite score (0-100)
            tier: Bot tier classification
            specialist_session: Session if SESSION_SPECIALIST, else None
            specialist_boost_applied: Whether +5 boost was applied
            concern_flag: Whether SESSION_CONCERN flag is set
            recovery_step: Recovery step (0=not in recovery, 1=first win, 2=eligible)
            in_concern_subqueue: Whether in concern sub-queue

        Returns:
            QueueEntry with SSL state fields populated
        """
        ssl_state = self.get_ssl_state_for_bot(bot_id)

        return QueueEntry(
            bot_id=bot_id,
            queue_position=queue_position,
            dpr_composite_score=dpr_composite_score,
            tier=tier,
            specialist_session=specialist_session,
            specialist_boost_applied=specialist_boost_applied,
            concern_flag=concern_flag,
            recovery_step=recovery_step,
            in_concern_subqueue=in_concern_subqueue,
            ssl_state=ssl_state.get("state", "live"),
            ssl_tier=ssl_state.get("tier"),
            is_paper_only=ssl_state.get("is_paper_only", False),
            paper_entry_timestamp=ssl_state.get("paper_entry_timestamp"),
        )

    def queue_remix(self, session_id: str) -> DPRQueueOutput:
        """
        Assemble the ranked queue for a session with T1/T3/T2 interleaving.

        AC #1: When queue tier remix runs, bots are sorted by DPR composite
        score descending within each tier, then interleaved:
        - TIER_1 recovery bots: positions 1-N
        - TIER_3 bots: next 40% of positions
        - TIER_2 bots: remaining positions

        Args:
            session_id: Session identifier (e.g., "LONDON", "NY")

        Returns:
            DPRQueueOutput with full queue and positions
        """
        from src.database.repositories.bot_repository import BotRepository

        # Apply any queued SSL events first
        self._apply_queued_events()

        # Get all active bots
        bot_repo = BotRepository()
        active_bots = bot_repo.get_active_bots()
        active_bot_ids = [b.bot_name for b in active_bots]

        if not active_bot_ids:
            return DPRQueueOutput(
                session_id=session_id,
                locked=self.queue_locked(session_id),
                bots=[],
            )

        # Get DPR scores for all bots
        scores: Dict[str, int] = {}
        specialist_sessions: Dict[str, Optional[str]] = {}
        concern_flags: Dict[str, bool] = {}

        for bot_id in active_bot_ids:
            dpr_score = self.scoring_engine.get_dpr_score(bot_id, session_id)
            if dpr_score is not None:
                # Apply specialist boost for positioning
                boosted = self.scoring_engine.apply_specialist_boost(
                    dpr_score.composite_score, bot_id, session_id
                )
                scores[bot_id] = boosted
                specialist_sessions[bot_id] = (
                    session_id if self.scoring_engine._is_specialist(bot_id, session_id) else None
                )
                concern_flags[bot_id] = self.scoring_engine.check_concern_flag(bot_id)
            else:
                scores[bot_id] = 0
                specialist_sessions[bot_id] = None
                concern_flags[bot_id] = False

        # Assign tiers
        tier_bots = {Tier.TIER_1: [], Tier.TIER_2: [], Tier.TIER_3: []}
        for bot_id in active_bot_ids:
            tier = self.tier_assignment(bot_id)
            tier_bots[tier].append(bot_id)

        # Sort within tiers by DPR score descending
        for tier in tier_bots:
            tier_bots[tier] = self.tier_sort(tier_bots[tier], scores)

        # Build queue with T1/T3/T2 interleaving
        queue_entries: List[QueueEntry] = []
        position = 1

        # Position 1-N: TIER_1 recovery eligible bots
        tier1_recovery = [
            b for b in tier_bots[Tier.TIER_1]
            if self.get_recovery_step(b) == 2
        ]
        # Check for specialist at position 1 (takes priority over TIER_1 recovery)
        specialist_at_1 = None
        for bot_id in active_bot_ids:
            if specialist_sessions.get(bot_id) == session_id:
                # This is a specialist for this session
                specialist_at_1 = bot_id
                break

        # Position 1: Specialist OR first TIER_1 recovery
        if specialist_at_1:
            queue_entries.append(self._build_queue_entry(
                bot_id=specialist_at_1,
                queue_position=position,
                dpr_composite_score=scores.get(specialist_at_1, 0),
                tier=self.tier_assignment(specialist_at_1),
                specialist_session=specialist_sessions.get(specialist_at_1),
                specialist_boost_applied=True,
                concern_flag=concern_flags.get(specialist_at_1, False),
                recovery_step=self.get_recovery_step(specialist_at_1),
                in_concern_subqueue=concern_flags.get(specialist_at_1, False),
            ))
            position += 1
            # Remove from tier1_recovery to avoid double-placement
            if specialist_at_1 in tier1_recovery:
                tier1_recovery.remove(specialist_at_1)

        # Remaining TIER_1 recovery bots
        for bot_id in tier1_recovery:
            queue_entries.append(self._build_queue_entry(
                bot_id=bot_id,
                queue_position=position,
                dpr_composite_score=scores.get(bot_id, 0),
                tier=Tier.TIER_1,
                specialist_session=specialist_sessions.get(bot_id),
                specialist_boost_applied=specialist_sessions.get(bot_id) == session_id,
                concern_flag=concern_flags.get(bot_id, False),
                recovery_step=self.get_recovery_step(bot_id),
                in_concern_subqueue=concern_flags.get(bot_id, False),
            ))
            position += 1

        # Non-recovery TIER_1 bots (TIER_1 but not in recovery) - placed after recovery
        tier1_non_recovery = [
            b for b in tier_bots[Tier.TIER_1]
            if self.get_recovery_step(b) == 0 and b != specialist_at_1
        ]
        for bot_id in tier1_non_recovery:
            queue_entries.append(self._build_queue_entry(
                bot_id=bot_id,
                queue_position=position,
                dpr_composite_score=scores.get(bot_id, 0),
                tier=Tier.TIER_1,
                specialist_session=specialist_sessions.get(bot_id),
                specialist_boost_applied=False,
                concern_flag=concern_flags.get(bot_id, False),
                recovery_step=0,
                in_concern_subqueue=concern_flags.get(bot_id, False),
            ))
            position += 1

        # TIER_3 next (40% of remaining queue positions)
        remaining_slots = max(1, int(len(active_bot_ids) * 0.4))
        for bot_id in tier_bots[Tier.TIER_3][:remaining_slots]:
            queue_entries.append(self._build_queue_entry(
                bot_id=bot_id,
                queue_position=position,
                dpr_composite_score=scores.get(bot_id, 0),
                tier=Tier.TIER_3,
                specialist_session=specialist_sessions.get(bot_id),
                specialist_boost_applied=specialist_sessions.get(bot_id) == session_id,
                concern_flag=concern_flags.get(bot_id, False),
                recovery_step=self.get_recovery_step(bot_id),
                in_concern_subqueue=concern_flags.get(bot_id, False),
            ))
            position += 1

        # TIER_2 remaining (always after TIER_3 regardless of score)
        for bot_id in tier_bots[Tier.TIER_2]:
            queue_entries.append(self._build_queue_entry(
                bot_id=bot_id,
                queue_position=position,
                dpr_composite_score=scores.get(bot_id, 0),
                tier=Tier.TIER_2,
                specialist_session=specialist_sessions.get(bot_id),
                specialist_boost_applied=specialist_sessions.get(bot_id) == session_id,
                concern_flag=concern_flags.get(bot_id, False),
                recovery_step=self.get_recovery_step(bot_id),
                in_concern_subqueue=concern_flags.get(bot_id, False),
            ))
            position += 1

        # Persist audit log
        self._persist_audit_log(session_id, queue_entries)

        return DPRQueueOutput(
            session_id=session_id,
            locked=self.queue_locked(session_id),
            bots=queue_entries,
        )

    def assemble_ny_hybrid_queue(self, session_id: str = "NY") -> DPRQueueOutput:
        """
        Assemble the NY hybrid queue per Story 16.3 Inter-Session Cooldown.

        AC #5: NY hybrid queue built as:
        - Position 1: Best London performer (SESSION_SPECIALIST + TIER_1 recovery)
        - Position 2: TIER_1 recovery candidate (if not already position 1)
        - Positions 3-N: TIER_3 DPR-ranked bots
        - Remaining: TIER_2 fresh candidates (always after TIER_3)

        Args:
            session_id: Session identifier (default "NY")

        Returns:
            DPRQueueOutput with NY hybrid queue
        """
        from src.database.repositories.bot_repository import BotRepository

        bot_repo = BotRepository()
        active_bots = bot_repo.get_active_bots()
        active_bot_ids = [b.bot_name for b in active_bots]

        if not active_bot_ids:
            return DPRQueueOutput(
                session_id=session_id,
                ny_hybrid_override=True,
                locked=self.queue_locked(session_id),
                bots=[],
            )

        # Get DPR scores
        scores: Dict[str, int] = {}
        for bot_id in active_bot_ids:
            dpr_score = self.scoring_engine.get_dpr_score(bot_id, session_id)
            if dpr_score is not None:
                scores[bot_id] = dpr_score.composite_score
            else:
                scores[bot_id] = 0

        # Assign tiers
        tier_bots = {Tier.TIER_1: [], Tier.TIER_2: [], Tier.TIER_3: []}
        for bot_id in active_bot_ids:
            tier = self.tier_assignment(bot_id)
            tier_bots[tier].append(bot_id)

        # Sort within tiers
        for tier in tier_bots:
            tier_bots[tier] = self.tier_sort(tier_bots[tier], scores)

        # Get London specialist (position 1)
        london_specialist = None
        london_specialist_score = -1
        for bot_id in tier_bots[Tier.TIER_1]:
            if self.scoring_engine._is_specialist(bot_id, "LONDON"):
                score = scores.get(bot_id, 0)
                if score > london_specialist_score:
                    london_specialist = bot_id
                    london_specialist_score = score

        # If no London specialist in TIER_1, look in TIER_3
        if london_specialist is None:
            for bot_id in tier_bots[Tier.TIER_3]:
                if self.scoring_engine._is_specialist(bot_id, "LONDON"):
                    score = scores.get(bot_id, 0)
                    if score > london_specialist_score:
                        london_specialist = bot_id
                        london_specialist_score = score

        # Get TIER_1 recovery eligible
        tier1_recovery = [
            b for b in tier_bots[Tier.TIER_1]
            if self.get_recovery_step(b) == 2
        ]

        # Build NY hybrid queue
        queue_entries: List[QueueEntry] = []
        position = 1

        # Position 1: London specialist
        if london_specialist:
            queue_entries.append(self._build_queue_entry(
                bot_id=london_specialist,
                queue_position=position,
                dpr_composite_score=scores.get(london_specialist, 0),
                tier=self.tier_assignment(london_specialist),
                specialist_session="LONDON",
                specialist_boost_applied=True,
                concern_flag=False,
                recovery_step=self.get_recovery_step(london_specialist),
                in_concern_subqueue=False,
            ))
            position += 1
            # Remove from tier1_recovery if present
            if london_specialist in tier1_recovery:
                tier1_recovery.remove(london_specialist)

        # Position 2: TIER_1 recovery (if not already position 1)
        if tier1_recovery:
            bot_id = tier1_recovery[0]
            queue_entries.append(self._build_queue_entry(
                bot_id=bot_id,
                queue_position=position,
                dpr_composite_score=scores.get(bot_id, 0),
                tier=Tier.TIER_1,
                specialist_session=None,
                specialist_boost_applied=False,
                concern_flag=False,
                recovery_step=self.get_recovery_step(bot_id),
                in_concern_subqueue=False,
            ))
            position += 1

        # Positions 3-N: TIER_3 DPR-ranked bots
        for bot_id in tier_bots[Tier.TIER_3]:
            if len(queue_entries) >= position:
                break
            queue_entries.append(self._build_queue_entry(
                bot_id=bot_id,
                queue_position=position,
                dpr_composite_score=scores.get(bot_id, 0),
                tier=Tier.TIER_3,
                specialist_session=None,
                specialist_boost_applied=False,
                concern_flag=False,
                recovery_step=self.get_recovery_step(bot_id),
                in_concern_subqueue=False,
            ))
            position += 1

        # Remaining: TIER_2 fresh candidates (always after TIER_3)
        for bot_id in tier_bots[Tier.TIER_2]:
            queue_entries.append(self._build_queue_entry(
                bot_id=bot_id,
                queue_position=position,
                dpr_composite_score=scores.get(bot_id, 0),
                tier=Tier.TIER_2,
                specialist_session=None,
                specialist_boost_applied=False,
                concern_flag=False,
                recovery_step=self.get_recovery_step(bot_id),
                in_concern_subqueue=False,
            ))
            position += 1

        # Persist audit log
        self._persist_audit_log(session_id, queue_entries)

        return DPRQueueOutput(
            session_id=session_id,
            ny_hybrid_override=True,
            locked=self.queue_locked(session_id),
            bots=queue_entries,
        )

    def lock_queue(self, session_id: str) -> None:
        """
        Lock the queue for a session at session start.

        AC #6: Queue is locked for entire session duration.
        Mid-session SSL events are QUEUED, not applied immediately.

        Args:
            session_id: Session identifier
        """
        self._locked_queues[session_id] = True
        logger.info(f"Queue locked for session {session_id}")

    def queue_locked(self, session_id: str) -> bool:
        """
        Check if queue is locked for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if queue is locked
        """
        return self._locked_queues.get(session_id, False)

    def queue_event(self, event: SSLEvent) -> None:
        """
        Queue an SSL event for application at next Dead Zone.

        AC #6: SSL mid-session events are QUEUED, not applied immediately.
        Events are applied at the next Dead Zone queue lock.

        Args:
            event: SSLEvent to queue
        """
        if self.queue_locked(event.session_id):
            self._queued_events.append(event)
            logger.info(f"Queued SSL event for session {event.session_id}: {event}")
        else:
            logger.warning(
                f"Queue event received for unlocked session {event.session_id}"
            )

    def _apply_queued_events(self) -> None:
        """
        Apply queued SSL events at Dead Zone.

        Events are applied in order received.
        """
        if not self._queued_events:
            return

        logger.info(f"Applying {len(self._queued_events)} queued SSL events")
        for event in self._queued_events:
            self._apply_single_event(event)
        self._queued_events.clear()

    def _apply_single_event(self, event: SSLEvent) -> None:
        """
        Apply a single SSL event to queue state.

        Args:
            event: SSLEvent to apply
        """
        from src.database.repositories.circuit_breaker_repository import CircuitBreakerRepository

        repo = CircuitBreakerRepository()

        if event.event_type == SSLEventType.MOVE_TO_PAPER:
            repo.quarantine(event.bot_id, reason="SSL mid-session move to paper")
        elif event.event_type == SSLEventType.RECOVERY_STEP_1:
            cb = repo.get_by_bot_id(event.bot_id)
            if cb:
                cb.consecutive_session_wins = 1
        elif event.event_type == SSLEventType.RECOVERY_CONFIRMED:
            cb = repo.get_by_bot_id(event.bot_id)
            if cb:
                cb.consecutive_session_wins = 2
        elif event.event_type == SSLEventType.RETIRED:
            repo.quarantine(event.bot_id, reason="SSL retirement")

    def get_session_queue(self, session_id: str) -> DPRQueueOutput:
        """
        Get the current queue for a session.

        Args:
            session_id: Session identifier

        Returns:
            DPRQueueOutput with current queue state
        """
        return self.queue_remix(session_id)

    def get_queue_audit(self, session_id: str) -> List[DPRQueueAuditRecord]:
        """
        Get audit trail for a session's queue.

        Args:
            session_id: Session identifier

        Returns:
            List of audit records sorted by timestamp descending
        """
        records = self.db_session.query(DPRQueueAuditLog).filter(
            DPRQueueAuditLog.session_id == session_id
        ).order_by(DPRQueueAuditLog.timestamp_utc.desc()).all()

        return [
            DPRQueueAuditRecord(
                session_id=r.session_id,
                bot_id=r.bot_id,
                queue_position=r.queue_position,
                dpr_composite_score=r.dpr_composite_score,
                tier=r.tier,
                specialist_flag=r.specialist_flag,
                concern_flag=r.concern_flag,
                timestamp_utc=r.timestamp_utc,
            )
            for r in records
        ]

    def _persist_audit_log(
        self,
        session_id: str,
        queue_entries: List[QueueEntry],
    ) -> None:
        """
        Persist queue entries to audit log.

        Args:
            session_id: Session identifier
            queue_entries: List of queue entries
        """
        from src.database.repositories.bot_repository import BotRepository

        bot_repo = BotRepository()

        for entry in queue_entries:
            # Get bot's specialist/Concern flags from repository
            specialist_flag = False
            concern_flag = entry.concern_flag

            try:
                bot = bot_repo.get_by_name(entry.bot_id)
                if bot and bot.tags:
                    specialist_flag = 'SESSION_SPECIALIST' in bot.tags
                    concern_flag = concern_flag or 'SESSION_CONCERN' in bot.tags
            except Exception:
                pass

            audit_record = DPRQueueAuditLog(
                session_id=session_id,
                bot_id=entry.bot_id,
                queue_position=entry.queue_position,
                dpr_composite_score=entry.dpr_composite_score,
                tier=entry.tier.value,
                specialist_flag=specialist_flag or entry.specialist_boost_applied,
                concern_flag=concern_flag,
            )
            self.db_session.add(audit_record)

        self.db_session.commit()

    def append_ssl_event_to_audit(
        self,
        bot_id: str,
        ssl_event,
        dpr_composite_score: int,
        session_id: str,
    ) -> None:
        """
        Append SSL event to DPR audit log.

        AC #6: DPR receives SSL state event, appends to audit log with:
        - event type, timestamp, bot ID, Magic Number, session ID
        - current DPR composite score at time of event

        Args:
            bot_id: Bot identifier
            ssl_event: SSLCircuitBreakerEvent from SSL system
            dpr_composite_score: DPR composite score at time of event
            session_id: Session identifier
        """
        try:
            # Get or create SSL audit log record
            ssl_audit = DPRSSLAuditLog(
                session_id=session_id,
                bot_id=bot_id,
                magic_number=ssl_event.magic_number,
                ssl_event_type=ssl_event.event_type.value,
                ssl_state=ssl_event.new_state.value,
                tier=ssl_event.tier,
                dpr_composite_score=dpr_composite_score,
                consecutive_losses=ssl_event.consecutive_losses,
                recovery_win_count=ssl_event.recovery_win_count,
                timestamp_utc=ssl_event.timestamp_utc,
                metadata_json=json.dumps(ssl_event.metadata) if ssl_event.metadata else None,
            )
            self.db_session.add(ssl_audit)
            self.db_session.commit()
            logger.info(f"Appended SSL event to DPR audit log: {ssl_event}")
        except Exception as e:
            logger.error(f"Error appending SSL event to DPR audit log: {e}")
            self.db_session.rollback()

    def get_ssl_state_for_bot(self, bot_id: str) -> Dict[str, Any]:
        """
        Get current SSL state for a bot.

        AC #7: DPR reads current SSL state for all bots at 11:30 GMT
        as part of Inter-Session Cooldown queue remix.

        Args:
            bot_id: Bot identifier

        Returns:
            Dict with SSL state info: state, tier, recovery_step, etc.
        """
        from src.risk.ssl.state import SSLCircuitBreakerState

        state_manager = SSLCircuitBreakerState(db_session=self.db_session)

        state = state_manager.get_state(bot_id)
        tier = state_manager.get_tier(bot_id)
        consecutive_losses = state_manager.get_consecutive_losses(bot_id)
        recovery_win_count = state_manager.get_recovery_win_count(bot_id)
        paper_entry_timestamp = state_manager.get_paper_entry_timestamp(bot_id)

        # Determine recovery step
        recovery_step = 0
        if state.value == "recovery":
            recovery_step = 2  # recovery_confirmed
        elif state.value == "paper" and tier and tier.value == "TIER_1":
            if recovery_win_count >= 2:
                recovery_step = 2
            elif recovery_win_count == 1:
                recovery_step = 1

        return {
            "state": state.value,
            "tier": tier.value if tier else None,
            "consecutive_losses": consecutive_losses,
            "recovery_win_count": recovery_win_count,
            "recovery_step": recovery_step,
            "paper_entry_timestamp": (
                paper_entry_timestamp.isoformat()
                if isinstance(paper_entry_timestamp, datetime)
                else None
            ),
            "is_retired": state.value == "retired",
            "is_paper_only": state.value == "paper",
        }

    def close(self):
        """Close database session if we created it."""
        if self._db_session is not None:
            self._db_session.close()
            self._db_session = None
