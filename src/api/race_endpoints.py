"""
Strategy Race Board API

Manages strategy race competitions with live tracking and leaderboard.
"""
from fastapi import APIRouter, HTTPException
from dataclasses import dataclass, asdict
from typing import List, Optional
from datetime import datetime
import uuid
import logging

from src.cache.redis_client import get_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/races")

# Redis key constants
RACE_KEY_PREFIX = "race:"
RACE_IDS_KEY = "race:ids"


@dataclass
class StrategyRace:
    """Represents a strategy race competition."""
    race_id: str
    participants: List[str]
    start_date: str
    end_date: str
    conditions: dict
    status: str  # RUNNING, COMPLETED, CANCELLED
    results: dict


def _get_race_key(race_id: str) -> str:
    """Get Redis key for a race."""
    return f"{RACE_KEY_PREFIX}{race_id}"


async def _save_race(race: StrategyRace) -> None:
    """Save race to Redis."""
    cache = get_cache()
    try:
        # Serialize to JSON compatible format
        race_data = asdict(race)
        # Use 30 day TTL for running races, 7 day TTL for completed/cancelled
        ttl = 2592000 if race.status == "RUNNING" else 604800
        await cache.set(_get_race_key(race.race_id), race_data, ttl=ttl)
        # Add to set of all race IDs
        if cache._redis:
            await cache._redis.sadd(RACE_IDS_KEY, race.race_id)
    except Exception as e:
        logger.warning(f"Failed to save race {race.race_id} to Redis: {e}")


async def _get_race(race_id: str) -> Optional[StrategyRace]:
    """Get race from Redis."""
    cache = get_cache()
    try:
        race_data = await cache.get(_get_race_key(race_id))
        if race_data:
            return StrategyRace(**race_data)
        return None
    except Exception as e:
        logger.warning(f"Failed to get race {race_id} from Redis: {e}")
        return None


async def _list_all_race_ids() -> List[str]:
    """Get all race IDs from Redis."""
    cache = get_cache()
    try:
        if cache._redis:
            return list(await cache._redis.smembers(RACE_IDS_KEY))
        return []
    except Exception as e:
        logger.warning(f"Failed to list race IDs from Redis: {e}")
        return []


@router.post("/start")
async def start_race(participant_ids: List[str], duration_days: int, conditions: dict):
    """
    Start a new strategy race.

    Args:
        participant_ids: List of strategy/EA IDs to race
        duration_days: Number of days for the race
        conditions: Race conditions and parameters

    Returns:
        Race ID and status

    Raises:
        HTTPException(422): If conditions missing required fields
    """
    # Validate required fields in conditions dict
    if not isinstance(conditions, dict) or "name" not in conditions or "strategy_ids" not in conditions:
        raise HTTPException(status_code=422, detail="Invalid race conditions")
    race_id = f"race_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
    race = StrategyRace(
        race_id=race_id,
        participants=participant_ids,
        start_date=datetime.now().isoformat(),
        end_date="",
        conditions=conditions,
        status="RUNNING",
        results={}
    )
    await _save_race(race)
    return {"race_id": race_id, "status": "RUNNING"}


@router.get("/{race_id}/live")
async def get_race_live_data(race_id: str):
    """
    Get live race data including current standings.

    Args:
        race_id: Race identifier

    Returns:
        Race data with live metrics
    """
    race = await _get_race(race_id)
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")
    return race


@router.post("/{race_id}/complete")
async def complete_race(race_id: str):
    """
    Mark a race as completed.

    Args:
        race_id: Race identifier

    Returns:
        Completed race status
    """
    race = await _get_race(race_id)
    if not race:
        raise HTTPException(status_code=404, detail="Race not found")
    race.status = "COMPLETED"
    await _save_race(race)
    return {"race_id": race_id, "status": "COMPLETED"}


@router.get("")
async def list_races(status: Optional[str] = None):
    """
    List all races, optionally filtered by status.

    Args:
        status: Optional status filter (RUNNING, COMPLETED, CANCELLED)

    Returns:
        List of races
    """
    race_ids = await _list_all_race_ids()
    races = []
    for race_id in race_ids:
        race = await _get_race(race_id)
        if race:
            races.append(race)
    if status:
        races = [r for r in races if r.status == status]
    return races
