"""
Economic Calendar REST API Endpoints

Provides REST API for economic calendar data:
- Today's economic events with impact levels
- Events for specific date
- Active blackout windows

These endpoints complement the existing CalendarGate blackout indicators
and expose full economic calendar visibility.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk/economic-calendar", tags=["economic-calendar"])


class ImpactLevel(str, Enum):
    """Economic event impact levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EconomicEvent(BaseModel):
    """A single economic calendar event."""
    time: str = Field(description="ISO timestamp of the event time (UTC)")
    currency: str = Field(description="Currency code (e.g., USD, EUR, GBP)")
    event_name: str = Field(description="Name of the economic event")
    impact: ImpactLevel = Field(description="Impact level: high, medium, or low")
    previous: Optional[str] = Field(default=None, description="Previous value")
    forecast: Optional[str] = Field(default=None, description="Forecasted value")
    actual: Optional[str] = Field(default=None, description="Actual released value")
    is_blackout: bool = Field(default=False, description="Whether trading is blacked out for this event")


class BlackoutWindow(BaseModel):
    """An active trading blackout window."""
    start: str = Field(description="ISO timestamp of blackout start (UTC)")
    end: str = Field(description="ISO timestamp of blackout end (UTC)")
    currency: str = Field(description="Primary currency affected")
    reason: Optional[str] = Field(default=None, description="Reason for the blackout")


class EconomicCalendarResponse(BaseModel):
    """Response containing economic events for a day."""
    date: str = Field(description="Date in YYYY-MM-DD format")
    events: List[EconomicEvent] = Field(default_factory=list, description="List of economic events")
    blackouts: List[BlackoutWindow] = Field(default_factory=list, description="Active blackout windows")


class BlackoutListResponse(BaseModel):
    """Response containing all active blackout windows."""
    blackouts: List[BlackoutWindow] = Field(default_factory=list, description="List of active blackouts")


# =============================================================================
# Mock Economic Calendar Data Store
# =============================================================================

# Currency flag emoji mapping (for UI display)
CURRENCY_FLAGS: Dict[str, str] = {
    "USD": "$",
    "EUR": "\u20ac",
    "GBP": "\u00a3",
    "JPY": "\u00a5",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "CHF",
    "NZD": "NZ$",
    "CNY": "\u00a5",
    "HKD": "HK$",
}

# Currency country mapping for badges
CURRENCY_COUNTRIES: Dict[str, str] = {
    "USD": "US",
    "EUR": "EU",
    "GBP": "UK",
    "JPY": "JP",
    "AUD": "AU",
    "CAD": "CA",
    "CHF": "CH",
    "NZD": "NZ",
    "CNY": "CN",
    "HKD": "HK",
}


def _get_currency_flag(currency: str) -> str:
    """Get flag character for a currency."""
    return CURRENCY_FLAGS.get(currency, currency)


def _generate_mock_events_for_date(date_str: str) -> List[EconomicEvent]:
    """DEPRECATED — mock replaced by live Finnhub via NewsBlackoutService.
    Kept only so existing callers (tests, hot reload) that pass a date_str work
    without modification. Production code should use get_live_events().
    """
    # Try to get from the NewsBlackoutService if running in the backend
    try:
        from src.api.server import app as backend_app
        if hasattr(backend_app.state, "news_blackout"):
            service = backend_app.state.news_blackout
            events = service.get_upcoming_high_impact_events(hours_ahead=168)  # 7 days
            filtered = [
                e for e in events
                if datetime.fromisoformat(e["time"].replace("Z", "+00:00")).strftime("%Y-%m-%d") == date_str
            ]
            return [
                EconomicEvent(
                    time=e["time"].isoformat() if isinstance(e["time"], datetime) else e["time"],
                    currency=e.get("currency", ""),
                    event_name=e.get("title", ""),
                    impact=ImpactLevel.HIGH,  # only HIGH events come from service
                    previous=str(e.get("previous")) if e.get("previous") is not None else None,
                    forecast=str(e.get("forecast")) if e.get("forecast") is not None else None,
                    actual=str(e.get("actual")) if e.get("actual") is not None else None,
                    is_blackout=e.get("minutes_to_event", 999) <= 15 and e.get("minutes_to_event", 999) >= -15,
                )
                for e in filtered
            ]
    except Exception:
        pass
    return []


def get_live_events(hours_ahead: int = 168) -> List[EconomicEvent]:
    """
    Fetch live HIGH/MEDIUM/LOW events from the NewsBlackoutService.
    Falls back to empty list if the service is not running.
    """
    try:
        from src.api.server import app as backend_app
        if hasattr(backend_app.state, "news_blackout"):
            service = backend_app.state.news_blackout
            raw_events = service.get_upcoming_high_impact_events(hours_ahead=hours_ahead)
            return [
                EconomicEvent(
                    time=e["time"].isoformat() if isinstance(e["time"], datetime) else e["time"],
                    currency=e.get("currency", ""),
                    event_name=e.get("title", ""),
                    impact=ImpactLevel(e.get("impact", "high").lower()),
                    previous=str(e.get("previous")) if e.get("previous") is not None else None,
                    forecast=str(e.get("forecast")) if e.get("forecast") is not None else None,
                    actual=str(e.get("actual")) if e.get("actual") is not None else None,
                    is_blackout=abs(e.get("minutes_to_event", 9999)) <= 15,
                )
                for e in raw_events
            ]
    except Exception as e:
        logger.warning(f"Could not reach NewsBlackoutService: {e}")
    return []


def _generate_blackouts_for_date(date_str: str) -> List[BlackoutWindow]:
    """
    Generate blackout windows for a given date, derived from live HIGH-impact events
    via the NewsBlackoutService (±15 min around each event).
    Falls back to empty list if service is not running.
    """
    try:
        from src.api.server import app as backend_app
        if hasattr(backend_app.state, "news_blackout"):
            service = backend_app.state.news_blackout
            events = service.get_upcoming_high_impact_events(hours_ahead=168)
            blackouts = []
            base_date = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)

            for ev in events:
                ev_time = ev.get("time")
                if isinstance(ev_time, str):
                    ev_time = datetime.fromisoformat(ev_time.replace("Z", "+00:00"))
                ev_date_str = ev_time.strftime("%Y-%m-%d")
                if ev_date_str != date_str:
                    continue

                kill_start = ev_time - timedelta(minutes=15)
                kill_end = ev_time + timedelta(minutes=15)
                blackouts.append(BlackoutWindow(
                    start=kill_start.isoformat(),
                    end=kill_end.isoformat(),
                    currency=ev.get("currency", ""),
                    reason=f"HIGH impact: {ev.get('title')}",
                ))
            return blackouts
    except Exception:
        pass
    return []


# =============================================================================
# API Endpoints
# =============================================================================

@router.get("", response_model=EconomicCalendarResponse)
async def get_todays_economic_calendar() -> EconomicCalendarResponse:
    """
    Get today's economic calendar events.

    Returns all scheduled economic events for today including:
    - Event time (UTC)
    - Currency affected
    - Event name
    - Impact level (high/medium/low)
    - Previous, forecast, and actual values
    - Whether trading is blacked out

    Data sourced from NewsBlackoutService (Finnhub /calendar/economic).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    events = get_live_events(hours_ahead=24)
    # Filter to today
    events = [
        e for e in events
        if e.time.startswith(today)
    ]
    blackouts = _generate_blackouts_for_date(today)

    return EconomicCalendarResponse(
        date=today,
        events=events,
        blackouts=blackouts
    )


@router.get("/{date}", response_model=EconomicCalendarResponse)
async def get_economic_calendar_for_date(date: str) -> EconomicCalendarResponse:
    """
    Get economic calendar for a specific date.

    Args:
        date: Date in YYYY-MM-DD format

    Returns:
        Economic calendar events for the specified date

    Raises:
        400: Invalid date format
        404: Date not found or no data available
    """
    # Validate date format
    try:
        datetime.fromisoformat(date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {date}. Use YYYY-MM-DD format."
        )

    # Validate date is not too far in the past or future
    try:
        query_date = datetime.fromisoformat(date).replace(tzinfo=timezone.utc)
        today = datetime.now(timezone.utc)
        days_diff = abs((query_date - today).days)

        if days_diff > 365:
            raise HTTPException(
                status_code=400,
                detail=f"Date {date} is too far from today. Please query within the last year."
            )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date: {date}"
        )

    events = get_live_events(hours_ahead=int(days_diff * 24 + 24))
    # Filter to the requested date
    events = [
        e for e in events
        if e.time.startswith(date)
    ]
    blackouts = _generate_blackouts_for_date(date)

    return EconomicCalendarResponse(
        date=date,
        events=events,
        blackouts=blackouts
    )


@router.get("/blackouts/active", response_model=BlackoutListResponse)
async def get_active_blackouts() -> BlackoutListResponse:
    """
    Get currently active blackout windows.

    Returns all active trading blackout windows that are currently in effect.
    Blackouts are typically active around high-impact economic events.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    blackouts = _generate_blackouts_for_date(today)

    # Filter to only currently active blackouts
    now = datetime.now(timezone.utc)
    active_blackouts = []

    for blackout in blackouts:
        start = datetime.fromisoformat(blackout.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(blackout.end.replace("Z", "+00:00"))

        if start <= now <= end:
            active_blackouts.append(blackout)

    return BlackoutListResponse(blackouts=active_blackouts)


@router.get("/blackouts", response_model=BlackoutListResponse)
async def get_all_blackouts(
    start_date: Optional[str] = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(default=None, description="End date (YYYY-MM-DD)"),
) -> BlackoutListResponse:
    """
    Get blackout windows, optionally filtered by date range.

    Args:
        start_date: Optional start date for filtering (YYYY-MM-DD)
        end_date: Optional end date for filtering (YYYY-MM-DD)

    Returns:
        List of blackout windows in the specified range
    """
    if start_date is None:
        start_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if end_date is None:
        end_date = (datetime.now(timezone.utc) + timedelta(days=7)).strftime("%Y-%m-%d")

    all_blackouts = []

    # Generate blackouts for each day in range
    try:
        start = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            daily_blackouts = _generate_blackouts_for_date(date_str)
            all_blackouts.extend(daily_blackouts)
            current += timedelta(days=1)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {e}"
        )

    return BlackoutListResponse(blackouts=all_blackouts)
