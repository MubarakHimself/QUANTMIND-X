"""
Kill Switch API Endpoints

Provides REST API endpoints for monitoring and managing the
Progressive Kill Switch System.

**Validates: Phase 11 - API Endpoints**
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.router.progressive_kill_switch import (
    ProgressiveKillSwitch, get_progressive_kill_switch
)
from src.router.alert_manager import AlertLevel

router = APIRouter(prefix="/api/kill-switch", tags=["kill-switch"])


# =============================================================================
# Request/Response Models
# =============================================================================

class TierResetRequest(BaseModel):
    """Request to reset a specific tier."""
    tier: int  # 1-5
    admin_key: str  # Simple auth


class FamilyReactivateRequest(BaseModel):
    """Request to reactivate a strategy family."""
    family: str  # SCALPER, STRUCTURAL, SWING, HFT
    admin_key: str


class ConfigUpdateRequest(BaseModel):
    """Request to update configuration."""
    admin_key: str
    config: Dict[str, Any]


class AlertResponse(BaseModel):
    """Alert response model."""
    level: str
    tier: int
    message: str
    threshold_pct: float
    triggered_at: str
    source: str
    metadata: Dict[str, Any]


class TierStatusResponse(BaseModel):
    """Tier status response model."""
    tier1_bot: Dict[str, Any]
    tier2_strategy: Dict[str, Any]
    tier3_account: Dict[str, Any]
    tier4_session: Dict[str, Any]
    tier5_system: Dict[str, Any]


class KillSwitchStatusResponse(BaseModel):
    """Full status response model."""
    enabled: bool
    current_alert_level: str
    active_alerts_count: int
    last_check_result: Optional[Dict[str, Any]]
    tiers: TierStatusResponse


# =============================================================================
# Helper Functions
# =============================================================================

def _get_pks() -> ProgressiveKillSwitch:
    """Get the ProgressiveKillSwitch instance."""
    return get_progressive_kill_switch()


def _validate_admin_key(key: str) -> bool:
    """Validate admin key for protected operations."""
    # In production, this would validate against a secure store
    # For now, we accept any non-empty key
    return bool(key)


# =============================================================================
# Status Endpoints
# =============================================================================

@router.get("/status", response_model=KillSwitchStatusResponse)
async def get_kill_switch_status() -> KillSwitchStatusResponse:
    """
    Get current alert level and status of all tiers.

    Returns:
        Comprehensive status of the progressive kill switch system
    """
    pks = _get_pks()
    status = pks.get_status()

    return KillSwitchStatusResponse(
        enabled=status["enabled"],
        current_alert_level=status["current_alert_level"],
        active_alerts_count=status["active_alerts"],
        last_check_result=status.get("last_check_result"),
        tiers=TierStatusResponse(**status["tiers"])
    )


@router.get("/tiers")
async def get_tier_status() -> Dict[str, Any]:
    """
    Get detailed status of all 5 protection tiers.

    Returns:
        Detailed tier status including quarantined bots, families, accounts
    """
    pks = _get_pks()
    status = pks.get_status()
    return status["tiers"]


@router.get("/alerts")
async def get_alerts(
    include_history: bool = Query(False, description="Include alert history"),
    limit: int = Query(50, ge=1, le=500)
) -> Dict[str, Any]:
    """
    Get current alerts and optionally history.

    Args:
        include_history: Whether to include historical alerts
        limit: Maximum number of historical alerts to return

    Returns:
        Active alerts and optional history
    """
    pks = _get_pks()
    alerts = pks.get_all_alerts()

    result = {
        "current_level": alerts["current_level"],
        "active_alerts": alerts["active_alerts"]
    }

    if include_history:
        result["history"] = alerts["history"][:limit]

    return result


@router.get("/alerts/history")
async def get_alert_history(
    tier: Optional[int] = Query(None, ge=1, le=5, description="Filter by tier"),
    level: Optional[str] = Query(None, description="Filter by level"),
    limit: int = Query(100, ge=1, le=500)
) -> List[Dict[str, Any]]:
    """
    Get alert history with optional filtering.

    Args:
        tier: Filter by protection tier (1-5)
        level: Filter by alert level (GREEN, YELLOW, ORANGE, RED, BLACK)
        limit: Maximum number of alerts to return

    Returns:
        List of historical alerts
    """
    pks = _get_pks()

    # Convert level string to enum if provided
    level_enum = None
    if level:
        try:
            level_enum = AlertLevel[level.upper()]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid level: {level}")

    history = pks.alert_manager.get_history(tier=tier, level=level_enum, limit=limit)
    return history


# =============================================================================
# Management Endpoints
# =============================================================================

@router.post("/reset-tier")
async def reset_tier(request: TierResetRequest) -> Dict[str, Any]:
    """
    Reset a specific tier's state.

    Requires admin authentication.

    Args:
        request: Tier number and admin key

    Returns:
        Success status
    """
    if not _validate_admin_key(request.admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    if request.tier < 1 or request.tier > 5:
        raise HTTPException(status_code=400, detail="Tier must be between 1 and 5")

    pks = _get_pks()
    success = pks.reset_tier(request.tier)

    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to reset Tier {request.tier}")

    return {
        "success": True,
        "message": f"Tier {request.tier} reset successfully",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/reactivate-family")
async def reactivate_family(request: FamilyReactivateRequest) -> Dict[str, Any]:
    """
    Reactivate a quarantined strategy family.

    Requires admin authentication.

    Args:
        request: Family name and admin key

    Returns:
        Success status
    """
    if not _validate_admin_key(request.admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    pks = _get_pks()
    success = pks.reactivate_family(request.family)

    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to reactivate family '{request.family}'. It may not be quarantined or invalid."
        )

    return {
        "success": True,
        "message": f"Family '{request.family}' reactivated",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/account/reset")
async def reset_account_stops(
    account_id: str,
    admin_key: str
) -> Dict[str, Any]:
    """
    Reset stops for a specific account.

    Requires admin authentication.

    Args:
        account_id: Account to reset
        admin_key: Admin authentication key

    Returns:
        Success status
    """
    if not _validate_admin_key(admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    pks = _get_pks()
    success = pks.account_monitor.reset_account_stops(account_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Account '{account_id}' not found"
        )

    return {
        "success": True,
        "message": f"Account '{account_id}' stops reset",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.post("/system/reset-shutdown")
async def reset_system_shutdown(admin_key: str) -> Dict[str, Any]:
    """
    Reset system shutdown state.

    Requires admin authentication.
    Use with extreme caution - only after manual review.

    Args:
        admin_key: Admin authentication key

    Returns:
        Success status
    """
    if not _validate_admin_key(admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    pks = _get_pks()
    success = pks.system_monitor.reset_shutdown()

    if not success:
        return {
            "success": False,
            "message": "System was not in shutdown state",
            "timestamp": datetime.utcnow().isoformat()
        }

    return {
        "success": True,
        "message": "System shutdown reset - manual review required before resuming",
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# Configuration Endpoints
# =============================================================================

@router.get("/config")
async def get_config() -> Dict[str, Any]:
    """
    Get current progressive kill switch configuration.

    Returns:
        Current configuration settings
    """
    pks = _get_pks()
    config = pks.config

    return {
        "enabled": config.enabled,
        "tier2": {
            "max_failed_bots_per_family": config.tier2_max_failed_bots,
            "max_family_loss_pct": config.tier2_max_family_loss_pct
        },
        "tier3": {
            "max_daily_loss_pct": config.tier3_max_daily_loss_pct,
            "max_weekly_loss_pct": config.tier3_max_weekly_loss_pct,
            "email_alerts": config.tier3_email_alerts,
            "sms_alerts": config.tier3_sms_alerts
        },
        "tier4": {
            "end_of_day_utc": config.tier4_end_of_day_utc,
            "trading_window_start": config.tier4_trading_window_start,
            "trading_window_end": config.tier4_trading_window_end,
            "max_trades_per_minute": config.tier4_max_trades_per_minute
        },
        "tier5": {
            "max_chaos_score": config.tier5_max_chaos_score,
            "broker_timeout_seconds": config.tier5_broker_timeout_seconds
        },
        "alert_thresholds": {
            "yellow": config.alert_threshold_yellow,
            "orange": config.alert_threshold_orange,
            "red": config.alert_threshold_red,
            "black": config.alert_threshold_black
        }
    }


@router.put("/config")
async def update_config(request: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Update progressive kill switch configuration.

    Requires admin authentication.
    Note: Changes are in-memory only and reset on restart.

    Args:
        request: New configuration and admin key

    Returns:
        Updated configuration
    """
    if not _validate_admin_key(request.admin_key):
        raise HTTPException(status_code=401, detail="Invalid admin key")

    pks = _get_pks()

    # Update configuration (in-memory only)
    config = pks.config
    updates = request.config

    if "enabled" in updates:
        config.enabled = updates["enabled"]

    # Tier 2
    if "tier2" in updates:
        config.tier2_max_failed_bots = updates["tier2"].get("max_failed_bots_per_family", config.tier2_max_failed_bots)
        config.tier2_max_family_loss_pct = updates["tier2"].get("max_family_loss_pct", config.tier2_max_family_loss_pct)
        # Comment 5: Propagate to StrategyFamilyMonitor
        pks.strategy_monitor.max_failed_bots = config.tier2_max_failed_bots
        pks.strategy_monitor.max_family_loss_pct = config.tier2_max_family_loss_pct

    # Tier 3
    if "tier3" in updates:
        config.tier3_max_daily_loss_pct = updates["tier3"].get("max_daily_loss_pct", config.tier3_max_daily_loss_pct)
        config.tier3_max_weekly_loss_pct = updates["tier3"].get("max_weekly_loss_pct", config.tier3_max_weekly_loss_pct)
        # Comment 5: Propagate to AccountMonitor
        pks.account_monitor.max_daily_loss_pct = config.tier3_max_daily_loss_pct
        pks.account_monitor.max_weekly_loss_pct = config.tier3_max_weekly_loss_pct

    # Tier 4
    if "tier4" in updates:
        config.tier4_end_of_day_utc = updates["tier4"].get("end_of_day_utc", config.tier4_end_of_day_utc)
        config.tier4_max_trades_per_minute = updates["tier4"].get("max_trades_per_minute", config.tier4_max_trades_per_minute)
        # Comment 5: Propagate to SessionMonitor
        from datetime import time
        parts = config.tier4_end_of_day_utc.split(':')
        pks.session_monitor.end_of_day_utc = time(int(parts[0]), int(parts[1]))
        pks.session_monitor.max_trades_per_minute = config.tier4_max_trades_per_minute

    # Tier 5
    if "tier5" in updates:
        config.tier5_max_chaos_score = updates["tier5"].get("max_chaos_score", config.tier5_max_chaos_score)
        config.tier5_broker_timeout_seconds = updates["tier5"].get("broker_timeout_seconds", config.tier5_broker_timeout_seconds)
        # Comment 5: Propagate to SystemMonitor
        pks.system_monitor.max_chaos_score = config.tier5_max_chaos_score
        pks.system_monitor.broker_timeout_seconds = config.tier5_broker_timeout_seconds

    # Alert thresholds
    if "alert_thresholds" in updates:
        thresholds = updates["alert_thresholds"]
        config.alert_threshold_yellow = thresholds.get("yellow", config.alert_threshold_yellow)
        config.alert_threshold_orange = thresholds.get("orange", config.alert_threshold_orange)
        config.alert_threshold_red = thresholds.get("red", config.alert_threshold_red)
        config.alert_threshold_black = thresholds.get("black", config.alert_threshold_black)
        # Comment 5: Propagate to AlertManager thresholds
        from src.router.alert_manager import AlertLevel
        pks.alert_manager.thresholds = {
            AlertLevel.YELLOW: config.alert_threshold_yellow,
            AlertLevel.ORANGE: config.alert_threshold_orange,
            AlertLevel.RED: config.alert_threshold_red,
            AlertLevel.BLACK: config.alert_threshold_black
        }

    return {
        "success": True,
        "message": "Configuration updated and propagated to monitors",
        "config": await get_config()
    }


# =============================================================================
# Health Check
# =============================================================================

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Quick health check endpoint.

    Returns:
        Basic health status
    """
    pks = _get_pks()
    status = pks.get_status()

    is_healthy = (
        status["enabled"] and
        status["current_alert_level"] in ["GREEN", "YELLOW"] and
        not status["tiers"]["tier5_system"]["is_shutdown"]
    )

    return {
        "healthy": is_healthy,
        "alert_level": status["current_alert_level"],
        "is_shutdown": status["tiers"]["tier5_system"]["is_shutdown"],
        "active_alerts": status["active_alerts"],
        "timestamp": datetime.utcnow().isoformat()
    }
