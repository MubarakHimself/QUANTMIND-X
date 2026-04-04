"""
DPR API Endpoints — Daily Performance Ranking REST API.

Story 17.1: DPR Composite Score Calculation
Story 17.2: DPR Queue Tier Remix

Provides REST endpoints for:
- GET  /api/dpr/scores              — All active bot DPR scores
- GET  /api/dpr/scores/{bot_id}     — Single bot score detail
- POST /api/dpr/calculate           — Trigger DPR recalculation
- GET  /api/dpr/history/{bot_id}    — Bot's DPR score history
- GET  /api/dpr/queue/{session_id}  — Get current session queue
- POST /api/dpr/queue/remix         — Trigger queue remix
- GET  /api/dpr/queue/audit/{session_id} — Get queue audit trail

Per NFR-M2: DPR is a synchronous scoring engine — NO LLM calls in hot path.
Per NFR-D1: All DPR score calculations logged before API response.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.database.models import SessionLocal
from src.risk.dpr import DPRScoringEngine, DPRScoreHistory
from src.risk.dpr.queue_manager import DPRQueueManager
from src.risk.dpr.queue_models import QueueEntry as QueueEntryModel, Tier


router = APIRouter(prefix="/api/dpr", tags=["dpr"])


# ==================== Pydantic Models ====================


class ComponentScoresResponse(BaseModel):
    """Component scores in API response."""
    win_rate: float = Field(..., description="Win rate normalized score (0-100)")
    pnl: float = Field(..., description="Net PnL normalized score (0-100)")
    consistency: float = Field(..., description="Consistency normalized score (0-100)")
    ev_per_trade: float = Field(..., description="EV per trade normalized score (0-100)")
    weights: tuple = Field(
        default=(0.25, 0.30, 0.20, 0.25),
        description="Component weights"
    )


class DPRScoreResponse(BaseModel):
    """DPR score in API response."""
    bot_id: str = Field(..., description="Bot identifier")
    session_id: str = Field(..., description="Session identifier")
    composite_score: int = Field(..., description="Final composite score (0-100)")
    component_scores: ComponentScoresResponse
    specialist_boost_applied: bool = Field(
        default=False,
        description="Whether SESSION_SPECIALIST boost was applied"
    )
    trade_count: int = Field(..., description="Number of trades in scoring window")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When score was calculated"
    )


class DPRScoresListResponse(BaseModel):
    """List of DPR scores response."""
    scores: List[DPRScoreResponse] = Field(..., description="List of bot scores")
    count: int = Field(..., description="Number of scores returned")
    scoring_window: str = Field(default="session", description="Scoring window used")


class DPRHistoryRecordResponse(BaseModel):
    """DPR history record in API response."""
    bot_id: str = Field(..., description="Bot identifier")
    session_id: str = Field(..., description="Session identifier")
    composite_score: int = Field(..., description="Score at this point")
    timestamp_utc: datetime = Field(..., description="When score was recorded")


class DPRHistoryResponse(BaseModel):
    """DPR score history response."""
    bot_id: str = Field(..., description="Bot identifier")
    history: List[DPRHistoryRecordResponse] = Field(
        ..., description="Score history sorted by timestamp descending"
    )
    count: int = Field(..., description="Number of history records")


class DPRCalculateRequest(BaseModel):
    """Request to trigger DPR recalculation."""
    session_id: str = Field(
        default="LONDON",
        description="Session to recalculate scores for"
    )
    scoring_window: str = Field(
        default="session",
        description="Scoring window (session or fortnight)"
    )
    bot_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific bot IDs to recalculate (all if None)"
    )


class DPRCalculateResponse(BaseModel):
    """Response from DPR recalculation."""
    status: str = Field(..., description="Status of recalculation")
    scores_calculated: int = Field(..., description="Number of scores calculated")
    session_id: str = Field(..., description="Session that was recalculated")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When recalculation occurred"
    )


# ==================== Dependency Injection ====================


def get_dpr_engine():
    """Get DPR scoring engine with database session."""
    engine = DPRScoringEngine(db_session=SessionLocal())
    try:
        yield engine
    finally:
        engine.close()


def get_dpr_history():
    """Get DPR history with database session."""
    history = DPRScoreHistory(db_session=SessionLocal())
    try:
        yield history
    finally:
        history.close()


# ==================== API Endpoints ====================


@router.get(
    "/scores",
    response_model=DPRScoresListResponse,
    summary="Get all active bot DPR scores",
    description="Returns DPR composite scores for all active bots in the scoring window.",
)
async def get_all_scores(
    session_id: str = "LONDON",
    scoring_window: str = "session",
    engine: DPRScoringEngine = Depends(get_dpr_engine),
):
    """
    Get DPR scores for all active bots.

    Returns composite scores for all bots that have completed at least
    one trade in the scoring window.
    """
    # Get active bot IDs from trading pipeline (BotRepository)
    from src.database.repositories.bot_repository import BotRepository

    bot_repo = BotRepository()
    active_bots = bot_repo.get_active_bots()

    scores = []
    for bot in active_bots:
        dpr_score = engine.get_dpr_score(bot.bot_name, session_id)
        if dpr_score is not None:
            scores.append(DPRScoreResponse(
                bot_id=dpr_score.bot_id,
                session_id=dpr_score.session_id,
                composite_score=dpr_score.composite_score,
                component_scores=ComponentScoresResponse(
                    win_rate=dpr_score.component_scores.win_rate,
                    pnl=dpr_score.component_scores.pnl,
                    consistency=dpr_score.component_scores.consistency,
                    ev_per_trade=dpr_score.component_scores.ev_per_trade,
                    weights=dpr_score.component_scores.weights,
                ),
                specialist_boost_applied=dpr_score.specialist_boost_applied,
                trade_count=dpr_score.trade_count,
                timestamp_utc=datetime.now(timezone.utc),
            ))

    return DPRScoresListResponse(
        scores=scores,
        count=len(scores),
        scoring_window=scoring_window,
    )


@router.get(
    "/scores/{bot_id}",
    response_model=DPRScoreResponse,
    summary="Get single bot DPR score detail",
    description="Returns detailed DPR composite score for a specific bot.",
)
async def get_bot_score(
    bot_id: str,
    session_id: str = "LONDON",
    scoring_window: str = "session",
    engine: DPRScoringEngine = Depends(get_dpr_engine),
):
    """
    Get DPR score detail for a single bot.

    Args:
        bot_id: Bot identifier
        session_id: Session identifier
        scoring_window: Scoring window (session or fortnight)

    Returns:
        Detailed DPR score for the bot

    Raises:
        HTTPException: 404 if bot not found or not eligible
    """
    dpr_score = engine.get_dpr_score(bot_id, session_id)

    if dpr_score is None:
        raise HTTPException(
            status_code=404,
            detail=f"Bot {bot_id} not found or not eligible (< 1 trade in window)",
        )

    return DPRScoreResponse(
        bot_id=dpr_score.bot_id,
        session_id=dpr_score.session_id,
        composite_score=dpr_score.composite_score,
        component_scores=ComponentScoresResponse(
            win_rate=dpr_score.component_scores.win_rate,
            pnl=dpr_score.component_scores.pnl,
            consistency=dpr_score.component_scores.consistency,
            ev_per_trade=dpr_score.component_scores.ev_per_trade,
            weights=dpr_score.component_scores.weights,
        ),
        specialist_boost_applied=dpr_score.specialist_boost_applied,
        trade_count=dpr_score.trade_count,
        timestamp_utc=datetime.now(timezone.utc),
    )


@router.post(
    "/calculate",
    response_model=DPRCalculateResponse,
    summary="Trigger DPR recalculation",
    description="Triggers DPR score recalculation for specified bots or all active bots.",
)
async def calculate_scores(
    request: DPRCalculateRequest,
    engine: DPRScoringEngine = Depends(get_dpr_engine),
    history: DPRScoreHistory = Depends(get_dpr_history),
):
    """
    Trigger DPR score recalculation.

    Args:
        request: Calculation parameters

    Returns:
        Status of recalculation with count of scores calculated
    """
    from src.database.repositories.bot_repository import BotRepository

    bot_repo = BotRepository()

    # Determine which bots to recalculate
    if request.bot_ids:
        bot_ids = request.bot_ids
    else:
        # Get all active bots
        active_bots = bot_repo.get_active_bots()
        bot_ids = [bot.bot_name for bot in active_bots]

    scores_calculated = 0

    for bot_id in bot_ids:
        # Calculate composite score
        composite_score = engine.calculate_composite_score(
            bot_id, request.session_id, request.scoring_window
        )

        if composite_score is not None:
            # Get full DPR score for audit logging
            dpr_score = engine.get_dpr_score(bot_id, request.session_id)

            if dpr_score is not None:
                # Check for SESSION_CONCERN flag
                session_concern = engine.check_concern_flag(bot_id)

                # Persist to audit log
                history.persist_score(
                    bot_id=bot_id,
                    session_id=request.session_id,
                    composite_score=composite_score,
                    component_scores={
                        "win_rate": dpr_score.component_scores.win_rate,
                        "pnl": dpr_score.component_scores.pnl,
                        "consistency": dpr_score.component_scores.consistency,
                        "ev_per_trade": dpr_score.component_scores.ev_per_trade,
                    },
                    scoring_window=request.scoring_window,
                    specialist_boost_applied=dpr_score.specialist_boost_applied,
                    session_concern_flag=session_concern,
                )

                scores_calculated += 1

    return DPRCalculateResponse(
        status="completed",
        scores_calculated=scores_calculated,
        session_id=request.session_id,
        timestamp_utc=datetime.now(timezone.utc),
    )


@router.get(
    "/history/{bot_id}",
    response_model=DPRHistoryResponse,
    summary="Get bot DPR score history",
    description="Returns DPR score history for week-over-week delta calculation.",
)
async def get_bot_history(
    bot_id: str,
    session_id: Optional[str] = None,
    limit: int = 10,
    history: DPRScoreHistory = Depends(get_dpr_history),
):
    """
    Get DPR score history for a bot.

    Args:
        bot_id: Bot identifier
        session_id: Optional session filter
        limit: Maximum number of records

    Returns:
        Score history sorted by timestamp descending
    """
    records = history.get_bot_scores(bot_id, session_id=session_id, limit=limit)

    return DPRHistoryResponse(
        bot_id=bot_id,
        history=[
            DPRHistoryRecordResponse(
                bot_id=r.bot_id,
                session_id=r.session_id,
                composite_score=r.composite_score,
                timestamp_utc=r.timestamp_utc,
            )
            for r in records
        ],
        count=len(records),
    )


# ==================== Queue Models ====================


class QueueEntryResponse(BaseModel):
    """Queue entry in API response."""
    bot_id: str = Field(..., description="Bot identifier")
    queue_position: int = Field(..., description="1-indexed queue position")
    dpr_composite_score: int = Field(..., description="DPR composite score (0-100)")
    tier: str = Field(..., description="Bot tier (TIER_1, TIER_2, TIER_3)")
    specialist_session: Optional[str] = Field(
        None, description="Session if SESSION_SPECIALIST"
    )
    specialist_boost_applied: bool = Field(
        default=False, description="Whether +5 specialist boost was applied"
    )
    concern_flag: bool = Field(
        default=False, description="Whether SESSION_CONCERN flag is set"
    )
    recovery_step: int = Field(
        default=0, description="Recovery step: 0=not in recovery, 1=first win, 2=eligible"
    )
    in_concern_subqueue: bool = Field(
        default=False, description="Whether in concern sub-queue"
    )
    # AC#7: SSL state integration
    ssl_state: str = Field(
        default="live",
        description="SSL state: live, paper, recovery, retired"
    )
    ssl_tier: Optional[str] = Field(
        default=None,
        description="Paper tier if in paper: TIER_1 or TIER_2"
    )
    is_paper_only: bool = Field(
        default=False,
        description="True if bot is in paper-only mode"
    )
    paper_entry_timestamp: Optional[str] = Field(
        default=None,
        description="ISO8601 timestamp when bot entered paper tier"
    )


class DPRQueueResponse(BaseModel):
    """DPR queue output in API response."""
    session_id: str = Field(..., description="Session identifier")
    queue_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When queue was assembled"
    )
    locked: bool = Field(default=False, description="Whether queue is locked")
    bots: List[QueueEntryResponse] = Field(
        default_factory=list, description="Ordered list of queue entries"
    )
    ny_hybrid_override: bool = Field(
        default=False, description="True when NY hybrid queue is active"
    )


class DPRQueueAuditRecordResponse(BaseModel):
    """DPR queue audit record in API response."""
    session_id: str = Field(..., description="Session identifier")
    bot_id: str = Field(..., description="Bot identifier")
    queue_position: int = Field(..., description="Assigned queue position")
    dpr_composite_score: int = Field(..., description="DPR composite score (0-100)")
    tier: str = Field(..., description="Bot tier")
    specialist_flag: bool = Field(
        default=False, description="Whether SESSION_SPECIALIST tag was present"
    )
    concern_flag: bool = Field(
        default=False, description="Whether SESSION_CONCERN flag was set"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When decision was made"
    )


class DPRQueueAuditResponse(BaseModel):
    """DPR queue audit trail response."""
    session_id: str = Field(..., description="Session identifier")
    records: List[DPRQueueAuditRecordResponse] = Field(
        ..., description="Audit records sorted by timestamp descending"
    )
    count: int = Field(..., description="Number of audit records")


class DPRQueueRemixRequest(BaseModel):
    """Request to trigger DPR queue remix."""
    session_id: str = Field(
        default="LONDON",
        description="Session to remix queue for"
    )
    ny_hybrid: bool = Field(
        default=False,
        description="Whether to assemble NY hybrid queue"
    )


class DPRQueueRemixResponse(BaseModel):
    """Response from DPR queue remix."""
    status: str = Field(..., description="Status of remix")
    session_id: str = Field(..., description="Session that was remixed")
    queue_position_count: int = Field(
        ..., description="Number of bots in queue"
    )
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When remix occurred"
    )


# ==================== Queue Dependency Injection ====================


def get_dpr_queue_manager():
    """Get DPR queue manager with database session."""
    manager = DPRQueueManager(db_session=SessionLocal())
    try:
        yield manager
    finally:
        manager.close()


# ==================== Queue API Endpoints ====================


@router.get(
    "/queue/{session_id}",
    response_model=DPRQueueResponse,
    summary="Get current session queue",
    description="Returns the current ranked queue for a session with all positions and flags.",
)
async def get_session_queue(
    session_id: str,
    manager: DPRQueueManager = Depends(get_dpr_queue_manager),
):
    """
    Get the current queue for a session.

    Args:
        session_id: Session identifier (e.g., "LONDON", "NY")

    Returns:
        Full queue output with positions and flags
    """
    queue = manager.get_session_queue(session_id)

    return DPRQueueResponse(
        session_id=queue.session_id,
        queue_timestamp=queue.queue_timestamp,
        locked=queue.locked,
        bots=[
            QueueEntryResponse(
                bot_id=entry.bot_id,
                queue_position=entry.queue_position,
                dpr_composite_score=entry.dpr_composite_score,
                tier=entry.tier.value,
                specialist_session=entry.specialist_session,
                specialist_boost_applied=entry.specialist_boost_applied,
                concern_flag=entry.concern_flag,
                recovery_step=entry.recovery_step,
                in_concern_subqueue=entry.in_concern_subqueue,
                ssl_state=entry.ssl_state,
                ssl_tier=entry.ssl_tier,
                is_paper_only=entry.is_paper_only,
                paper_entry_timestamp=entry.paper_entry_timestamp,
            )
            for entry in queue.bots
        ],
        ny_hybrid_override=queue.ny_hybrid_override,
    )


@router.post(
    "/queue/remix",
    response_model=DPRQueueRemixResponse,
    summary="Trigger DPR queue remix",
    description="Triggers queue remix to reassemble ranked positions with T1/T3/T2 interleaving.",
)
async def remix_queue(
    request: DPRQueueRemixRequest,
    manager: DPRQueueManager = Depends(get_dpr_queue_manager),
):
    """
    Trigger DPR queue remix.

    Args:
        request: Remix parameters

    Returns:
        Status of remix with queue position count
    """
    if request.ny_hybrid:
        queue = manager.assemble_ny_hybrid_queue(request.session_id)
    else:
        queue = manager.queue_remix(request.session_id)

    return DPRQueueRemixResponse(
        status="completed",
        session_id=request.session_id,
        queue_position_count=len(queue.bots),
        timestamp_utc=datetime.now(timezone.utc),
    )


@router.get(
    "/queue/audit/{session_id}",
    response_model=DPRQueueAuditResponse,
    summary="Get queue audit trail",
    description="Returns the audit trail for a session's queue decisions.",
)
async def get_queue_audit(
    session_id: str,
    manager: DPRQueueManager = Depends(get_dpr_queue_manager),
):
    """
    Get audit trail for a session's queue.

    Args:
        session_id: Session identifier

    Returns:
        Audit trail sorted by timestamp descending
    """
    records = manager.get_queue_audit(session_id)

    return DPRQueueAuditResponse(
        session_id=session_id,
        records=[
            DPRQueueAuditRecordResponse(
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
        ],
        count=len(records),
    )


# ==================== Leaderboard Pydantic Models ====================


class LeaderboardEntry(BaseModel):
    """Single bot entry in the leaderboard."""
    bot_id: str = Field(..., description="Bot identifier")
    bot_name: str = Field(..., description="Bot display name")
    composite_score: int = Field(..., description="DPR composite score (0-100)")
    tier: str = Field(..., description="Bot tier (TIER_1, TIER_2, TIER_3)")
    session_specialist: bool = Field(default=False, description="Whether bot is session specialist")
    session_win_rate: float = Field(..., description="Session win rate (0-1)")
    net_pnl: float = Field(..., description="Net PnL in session window")
    consistency: float = Field(..., description="Consistency score (0-1)")
    ev_per_trade: float = Field(..., description="EV per trade normalized score")
    queue_position: int = Field(..., description="Current queue position")
    rank: int = Field(..., description="Leaderboard rank (1-indexed)")


class DPRLeaderboardResponse(BaseModel):
    """Full DPR leaderboard response."""
    session_id: str = Field(..., description="Session identifier")
    entries: List[LeaderboardEntry] = Field(..., description="Leaderboard entries ranked by composite score")
    count: int = Field(..., description="Number of bots in leaderboard")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When leaderboard was generated"
    )


class BotDetailResponse(BaseModel):
    """Detailed DPR breakdown for a single bot."""
    bot_id: str = Field(..., description="Bot identifier")
    bot_name: str = Field(..., description="Bot display name")
    composite_score: int = Field(..., description="DPR composite score (0-100)")
    tier: str = Field(..., description="Bot tier (TIER_1, TIER_2, TIER_3)")
    session_specialist: bool = Field(default=False, description="Whether bot is session specialist")
    session_win_rate: float = Field(..., description="Session win rate (0-1)")
    net_pnl: float = Field(..., description="Net PnL in session window")
    consistency: float = Field(..., description="Consistency score (0-1)")
    ev_per_trade: float = Field(..., description="EV per trade")
    queue_position: int = Field(..., description="Current queue position")
    specialist_boost_applied: bool = Field(default=False, description="Whether +5 specialist boost was applied")
    concern_flag: bool = Field(default=False, description="Whether SESSION_CONCERN flag is set")
    consecutive_negative_ev: int = Field(default=0, description="Consecutive negative EV sessions")
    ssl_state: str = Field(default="live", description="SSL state: live, paper, recovery, retired")
    component_scores: ComponentScoresResponse
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When breakdown was generated"
    )


class TierDistribution(BaseModel):
    """Tier distribution response."""
    tier1: List[str] = Field(default_factory=list, description="TIER_1 bot IDs")
    tier2: List[str] = Field(default_factory=list, description="TIER_2 bot IDs")
    tier3: List[str] = Field(default_factory=list, description="TIER_3 bot IDs")
    counts: dict = Field(..., description="Tier counts")
    session_id: str = Field(..., description="Session identifier")
    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When distribution was generated"
    )


# ==================== Leaderboard API Endpoints ====================


@router.get(
    "/leaderboard",
    response_model=DPRLeaderboardResponse,
    summary="Get DPR leaderboard",
    description="Returns full DPR ranking table with all bots ranked by composite score.",
)
async def get_leaderboard(
    session_id: str = "LONDON",
    scoring_window: str = "session",
    engine: DPRScoringEngine = Depends(get_dpr_engine),
    manager: DPRQueueManager = Depends(get_dpr_queue_manager),
):
    """
    Get DPR leaderboard with all bots ranked by composite score.

    Returns composite score + component breakdown for each bot,
    along with tier, queue position, and specialist status.
    """
    from src.database.repositories.bot_repository import BotRepository

    bot_repo = BotRepository()
    active_bots = bot_repo.get_active_bots()

    if not active_bots:
        return DPRLeaderboardResponse(
            session_id=session_id,
            entries=[],
            count=0,
            timestamp_utc=datetime.now(timezone.utc),
        )

    # Get scores and queue data for all bots
    bot_scores: List[Tuple[str, str, int, DPRScore]] = []  # (bot_id, bot_name, queue_pos, dpr_score)

    for bot in active_bots:
        bot_id = bot.bot_name
        dpr_score = engine.get_dpr_score(bot_id, session_id)
        if dpr_score is None:
            continue

        # Get queue position
        queue = manager.get_session_queue(session_id)
        queue_pos = 0
        for entry in queue.bots:
            if entry.bot_id == bot_id:
                queue_pos = entry.queue_position
                break

        bot_scores.append((bot_id, bot.bot_name, queue_pos, dpr_score))

    # Sort by composite score descending
    bot_scores.sort(key=lambda x: x[3].composite_score if x[3] else 0, reverse=True)

    # Build leaderboard entries
    entries = []
    for rank, (bot_id, bot_name, queue_pos, dpr_score) in enumerate(bot_scores, start=1):
        if dpr_score is None:
            continue

        # Determine tier
        tier = manager.tier_assignment(bot_id)

        # Check specialist status
        specialist = engine.specialist_session_check(bot_id, session_id)

        # Get component raw values for display
        component = dpr_score.component_scores

        entry = LeaderboardEntry(
            bot_id=bot_id,
            bot_name=bot_name,
            composite_score=dpr_score.composite_score,
            tier=tier.value,
            session_specialist=specialist,
            session_win_rate=dpr_score.session_win_rate,
            net_pnl=component.pnl,  # Normalized PnL score
            consistency=component.consistency / 100.0,  # Denormalize for display
            ev_per_trade=component.ev_per_trade / 100.0,  # Denormalize for display
            queue_position=queue_pos,
            rank=rank,
        )
        entries.append(entry)

    return DPRLeaderboardResponse(
        session_id=session_id,
        entries=entries,
        count=len(entries),
        timestamp_utc=datetime.now(timezone.utc),
    )


@router.get(
    "/bot/{bot_id}",
    response_model=BotDetailResponse,
    summary="Get detailed DPR breakdown for bot",
    description="Returns detailed DPR breakdown including all component scores for a single bot.",
)
async def get_bot_detail(
    bot_id: str,
    session_id: str = "LONDON",
    scoring_window: str = "session",
    engine: DPRScoringEngine = Depends(get_dpr_engine),
    manager: DPRQueueManager = Depends(get_dpr_queue_manager),
):
    """
    Get detailed DPR breakdown for a single bot.

    Returns composite score, all component scores, tier, queue position,
    specialist status, concern flag, and SSL state.
    """
    from src.database.repositories.bot_repository import BotRepository

    dpr_score = engine.get_dpr_score(bot_id, session_id)

    if dpr_score is None:
        raise HTTPException(
            status_code=404,
            detail=f"Bot {bot_id} not found or not eligible (< 1 trade in window)",
        )

    # Get bot name
    bot_repo = BotRepository()
    bot = bot_repo.get_by_name(bot_id)
    bot_name = bot.bot_name if bot else bot_id

    # Get tier
    tier = manager.tier_assignment(bot_id)

    # Get queue position
    queue = manager.get_session_queue(session_id)
    queue_pos = 0
    for entry in queue.bots:
        if entry.bot_id == bot_id:
            queue_pos = entry.queue_position
            break

    # Check specialist and concern flags
    specialist = engine.specialist_session_check(bot_id, session_id)
    concern = engine.check_concern_flag(bot_id)

    # Get SSL state from queue entry if available
    ssl_state = "live"
    for entry in queue.bots:
        if entry.bot_id == bot_id:
            ssl_state = entry.ssl_state
            break

    component = dpr_score.component_scores

    return BotDetailResponse(
        bot_id=bot_id,
        bot_name=bot_name,
        composite_score=dpr_score.composite_score,
        tier=tier.value,
        session_specialist=specialist,
        session_win_rate=dpr_score.session_win_rate,
        net_pnl=component.pnl,
        consistency=component.consistency / 100.0,
        ev_per_trade=component.ev_per_trade / 100.0,
        queue_position=queue_pos,
        specialist_boost_applied=dpr_score.specialist_boost_applied,
        concern_flag=concern,
        consecutive_negative_ev=dpr_score.consecutive_negative_ev,
        ssl_state=ssl_state,
        component_scores=ComponentScoresResponse(
            win_rate=component.win_rate,
            pnl=component.pnl,
            consistency=component.consistency,
            ev_per_trade=component.ev_per_trade,
            weights=component.weights,
        ),
        timestamp_utc=datetime.now(timezone.utc),
    )


@router.get(
    "/tiers",
    response_model=TierDistribution,
    summary="Get tier distribution",
    description="Returns tier distribution with bot IDs grouped by tier.",
)
async def get_tier_distribution(
    session_id: str = "LONDON",
    scoring_window: str = "session",
    engine: DPRScoringEngine = Depends(get_dpr_engine),
    manager: DPRQueueManager = Depends(get_dpr_queue_manager),
):
    """
    Get tier distribution of active bots.

    Returns lists of bot IDs grouped by tier (TIER_1, TIER_2, TIER_3)
    sorted by DPR composite score within each tier.
    """
    from src.database.repositories.bot_repository import BotRepository

    bot_repo = BotRepository()
    active_bots = bot_repo.get_active_bots()

    tier_bots: Dict[str, List[Tuple[str, int]]] = {
        "TIER_1": [],  # (bot_id, composite_score)
        "TIER_2": [],
        "TIER_3": [],
    }

    for bot in active_bots:
        bot_id = bot.bot_name
        dpr_score = engine.get_dpr_score(bot_id, session_id)

        if dpr_score is None:
            continue

        tier = manager.tier_assignment(bot_id)
        tier_bots[tier.value].append((bot_id, dpr_score.composite_score))

    # Sort within tiers by composite score descending
    for tier_key in tier_bots:
        tier_bots[tier_key].sort(key=lambda x: x[1], reverse=True)

    return TierDistribution(
        tier1=[bot_id for bot_id, _ in tier_bots["TIER_1"]],
        tier2=[bot_id for bot_id, _ in tier_bots["TIER_2"]],
        tier3=[bot_id for bot_id, _ in tier_bots["TIER_3"]],
        counts={
            "tier1": len(tier_bots["TIER_1"]),
            "tier2": len(tier_bots["TIER_2"]),
            "tier3": len(tier_bots["TIER_3"]),
        },
        session_id=session_id,
        timestamp_utc=datetime.now(timezone.utc),
    )
