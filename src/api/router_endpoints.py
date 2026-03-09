"""
Router State API Endpoints

Provides real-time information about the Strategy Router including
auction queue, bot rankings, market regime, and correlations.

Also includes endpoints for:
- Bot lifecycle management (tag progression)
- Bot limit status checking
- Market scanner alerts
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import random

from src.api.pagination import PaginatedResponse, DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT

router = APIRouter(prefix="/api/router", tags=["router"])

# Global router instance (set by server.py on startup)
_strategy_router: Optional[Any] = None

def set_strategy_router(router_instance):
    """Set the global StrategyRouter instance."""
    global _strategy_router
    _strategy_router = router_instance

# Data models
class MarketRegime(BaseModel):
    quality: float
    trend: str  # 'bullish', 'bearish', 'ranging'
    chaos: float
    volatility: str  # 'low', 'medium', 'high'

class SymbolData(BaseModel):
    symbol: str
    price: float
    change: float
    spread: float

class BotSignal(BaseModel):
    id: str
    name: str
    symbol: str
    status: str  # 'idle', 'ready', 'paused', 'quarantined'
    signalStrength: float
    conditions: List[str]
    score: float
    lastSignal: Optional[str]

class Auction(BaseModel):
    id: str
    timestamp: str
    participants: List[str]
    winner: str
    winningScore: float
    status: str  # 'pending', 'completed', 'failed'

class RouterState(BaseModel):
    active: bool
    mode: str  # 'auction', 'priority', 'round-robin'
    auctionInterval: int
    lastAuction: Optional[str]
    queuedSignals: int
    activeAuctions: int

class HouseMoneyState(BaseModel):
    dailyProfit: float
    threshold: float
    houseMoneyAmount: float
    mode: str  # 'conservative', 'normal', 'aggressive'

# Mock data for development
def get_mock_router_state() -> Dict[str, Any]:
    """Get mock router state for development."""
    return {
        "active": True,
        "mode": "auction",
        "auctionInterval": 5000,
        "lastAuction": datetime.utcnow().isoformat(),
        "queuedSignals": 0,
        "activeAuctions": 1
    }

def get_mock_market_state() -> Dict[str, Any]:
    """Get mock market state."""
    return {
        "regime": {
            "quality": 0.82,
            "trend": "bullish",
            "chaos": 18.5,
            "volatility": "medium"
        },
        "symbols": [
            {"symbol": "EURUSD", "price": 1.0876, "change": 0.12, "spread": 1.2},
            {"symbol": "GBPUSD", "price": 1.2654, "change": -0.08, "spread": 1.5},
            {"symbol": "USDJPY", "price": 149.85, "change": 0.25, "spread": 0.8}
        ]
    }

def get_mock_bots() -> List[Dict[str, Any]]:
    """Get mock bot signals."""
    return [
        {
            "id": "ict-eur",
            "name": "ICT Scalper",
            "symbol": "EURUSD",
            "status": "ready",
            "signalStrength": 0.85,
            "conditions": ["fvg", "order_block", "london_session"],
            "score": 8.5,
            "lastSignal": datetime.utcnow().isoformat()
        },
        {
            "id": "smc-gbp",
            "name": "SMC Reversal",
            "symbol": "GBPUSD",
            "status": "ready",
            "signalStrength": 0.72,
            "conditions": ["choch", "bos", "session_filter"],
            "score": 7.2,
            "lastSignal": datetime.utcnow().isoformat()
        }
    ]

def get_mock_auctions() -> List[Dict[str, Any]]:
    """Get mock auction queue."""
    return [
        {
            "id": "auction-1",
            "timestamp": datetime.utcnow().isoformat(),
            "participants": ["ict-eur", "smc-gbp"],
            "winner": "ict-eur",
            "winningScore": 8.5,
            "status": "completed"
        }
    ]

def get_mock_rankings() -> Dict[str, List[Dict[str, Any]]]:
    """Get mock bot rankings."""
    return {
        "daily": [
            {"botId": "ict-eur", "name": "ICT Scalper", "profit": 245.80, "trades": 12, "winRate": 75.0},
            {"botId": "smc-gbp", "name": "SMC Reversal", "profit": 128.50, "trades": 8, "winRate": 62.5},
            {"botId": "breakthrough-eur", "name": "Breakthrough", "profit": 0.0, "trades": 0, "winRate": 0.0}
        ],
        "weekly": [
            {"botId": "ict-eur", "name": "ICT Scalper", "profit": 1245.60, "trades": 58, "winRate": 70.7},
            {"botId": "smc-gbp", "name": "SMC Reversal", "profit": 678.30, "trades": 42, "winRate": 64.3}
        ]
    }

def get_mock_correlations() -> List[Dict[str, Any]]:
    """Get mock correlation data."""
    return [
        {"pair": "EURUSD/GBPUSD", "value": 0.72, "status": "warning"},
        {"pair": "EURUSD/USDJPY", "value": -0.45, "status": "ok"},
        {"pair": "GBPUSD/USDJPY", "value": -0.38, "status": "ok"}
    ]

def get_mock_house_money() -> Dict[str, Any]:
    """Get mock house money state."""
    return {
        "dailyProfit": 374.30,
        "threshold": 0.5,
        "houseMoneyAmount": 187.15,
        "mode": "aggressive"
    }

# Endpoints

@router.get("/state")
async def get_router_state() -> RouterState:
    """Get current router state."""
    # Get real router mode from settings
    from src.api.settings_endpoints import load_risk_settings
    risk_settings = load_risk_settings()
    router_mode = getattr(risk_settings, 'routerMode', 'auction')

    # Get other state from mock (or real implementation if available)
    mock_state = get_mock_router_state()
    mock_state["mode"] = router_mode

    return RouterState(**mock_state)

@router.post("/toggle")
async def toggle_router(active: bool = None) -> Dict[str, Any]:
    """Toggle router on/off."""
    state = get_mock_router_state()
    if active is not None:
        state["active"] = active
    return {"success": True, "active": state["active"]}

@router.post("/settings")
async def save_router_settings(settings: RouterState) -> Dict[str, Any]:
    """Save router settings."""
    # For now, just return success - in production this would persist
    return {
        "success": True,
        "mode": settings.mode,
        "auctionInterval": settings.auctionInterval
    }

@router.get("/market")
async def get_market_state() -> Dict[str, Any]:
    """Get current market state including regime and symbols."""
    return get_mock_market_state()

@router.get("/bots", response_model=PaginatedResponse[BotSignal])
async def get_bot_signals(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
) -> PaginatedResponse[BotSignal]:
    """Get current bot signals with pagination."""
    all_bots = [BotSignal(**bot) for bot in get_mock_bots()]
    total = len(all_bots)
    paginated = all_bots[offset:offset + limit]

    return PaginatedResponse.create(
        items=paginated,
        total=total,
        limit=limit,
        offset=offset
    )

@router.get("/auctions", response_model=PaginatedResponse[Auction])
async def get_auction_queue(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
) -> PaginatedResponse[Auction]:
    """Get current auction queue with pagination."""
    all_auctions = [Auction(**auction) for auction in get_mock_auctions()]
    total = len(all_auctions)
    paginated = all_auctions[offset:offset + limit]

    return PaginatedResponse.create(
        items=paginated,
        total=total,
        limit=limit,
        offset=offset
    )

@router.post("/auction")
async def run_auction() -> Dict[str, Any]:
    """Run a new auction round."""
    # Mock auction result
    bots = get_mock_bots()
    ready_bots = [b for b in bots if b["status"] == "ready"]

    if not ready_bots:
        raise HTTPException(status_code=400, detail="No bots ready for auction")

    # Select winner based on score
    winner = max(ready_bots, key=lambda x: x["score"])

    auction = {
        "id": f"auction-{datetime.utcnow().timestamp()}",
        "timestamp": datetime.utcnow().isoformat(),
        "participants": [b["id"] for b in ready_bots],
        "winner": winner["id"],
        "winningScore": winner["score"],
        "status": "completed"
    }

    return {"success": True, "auction": auction}

@router.get("/rankings")
async def get_rankings(period: str = "daily") -> List[Dict[str, Any]]:
    """Get bot rankings for specified period."""
    rankings = get_mock_rankings()
    if period not in rankings:
        raise HTTPException(status_code=400, detail="Invalid period. Use 'daily' or 'weekly'")
    return rankings[period]

@router.get("/correlations")
async def get_correlations() -> List[Dict[str, Any]]:
    """Get current symbol correlations."""
    return get_mock_correlations()

@router.get("/house-money")
async def get_house_money_state() -> HouseMoneyState:
    """Get current house money state."""
    return HouseMoneyState(**get_mock_house_money())

@router.post("/settings")
async def update_router_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Update router settings."""
    # In production, this would save to database
    return {"success": True, "settings": settings}


@router.get("/system-status")
async def get_system_status() -> Dict[str, Any]:
    """
    Get current system status from StrategyRouter.

    Returns real metrics from:
    - Sentinel for market regime
    - Governor for Kelly fraction
    - AccountMonitor for daily P&L
    - Commander for active bot count
    """
    if _strategy_router is None:
        # Fallback to mock data if router not initialized
        return {
            "active_bots": 0,
            "pnl_today": 0.0,
            "regime": "UNKNOWN",
            "kelly": 0.0
        }

    status = _strategy_router.get_status()

    # Get current regime from sentinel (real value)
    current_regime = status.get("sentinel", {}).get("current_regime", "UNKNOWN")

    # Get Kelly fraction from governor (real value)
    kelly_fraction = 0.0
    if hasattr(_strategy_router.governor, 'current_kelly_fraction'):
        kelly_fraction = _strategy_router.governor.current_kelly_fraction

    # Get active bot count (real value)
    active_bots = status.get("commander", {}).get("active_bots", 0)

    # Get real P&L from AccountMonitor via ProgressiveKillSwitch
    pnl_today = 0.0
    pks = _strategy_router.progressive_kill_switch
    if pks and hasattr(pks, 'account_monitor') and pks.account_monitor:
        # Aggregate daily P&L across all tracked accounts
        account_states = pks.account_monitor.get_all_states()
        pnl_today = sum(
            state.get("daily_pnl", 0.0)
            for state in account_states.values()
        )

    return {
        "active_bots": active_bots,
        "pnl_today": pnl_today,
        "regime": current_regime,
        "kelly": kelly_fraction
    }


@router.get("/active-bots")
async def get_active_bots() -> List[Dict[str, Any]]:
    """Get list of active bots from Commander."""
    if _strategy_router is None:
        return []

    bots = []
    commander = _strategy_router.commander

    # Get bots from BotRegistry if available
    if commander.bot_registry:
        primal_bots = commander.bot_registry.list_by_tag("@primal")
        for manifest in primal_bots:
            bots.append({
                "id": manifest.bot_id,
                "name": manifest.name,
                "state": "primal",  # Tag as state
                "symbol": manifest.symbols[0] if manifest.symbols else "UNKNOWN"
            })
    else:
        # Fallback to active_bots dict
        for bot_id, bot_data in commander.active_bots.items():
            tags = bot_data.get('tags', [])
            state = "primal" if "@primal" in tags else "pending"
            bots.append({
                "id": bot_id,
                "name": bot_data.get('name', bot_id),
                "state": state,
                "symbol": bot_data.get('symbols', ['UNKNOWN'])[0]
            })

    return bots


@router.get("/fee-monitor")
async def get_fee_monitor_data() -> Dict[str, Any]:
    """
    Get current fee monitoring data including:
    - daily_fees: Total fees paid today
    - daily_fee_burn_pct: Fee burn percentage
    - kill_switch_active: Whether fee kill switch is active
    - fee_breakdown: Per-bot fee breakdown
    """
    from datetime import date
    from src.router.fee_monitor import FeeMonitor

    # Default values if router not initialized
    account_id = "default"
    account_balance = 10000.0

    if _strategy_router is not None:
        # Try to get account info from AccountMonitor
        pks = _strategy_router.progressive_kill_switch
        if pks and hasattr(pks, 'account_monitor') and pks.account_monitor:
            account_states = pks.account_monitor.get_all_states()
            if account_states:
                # Get first account's info
                first_account_id = list(account_states.keys())[0]
                account_id = first_account_id
                # Get balance from state if available
                state = account_states[first_account_id]
                if 'balance' in state:
                    account_balance = state['balance']

    # Get fee monitoring data
    fee_monitor = FeeMonitor(account_id, account_balance=account_balance)
    today_str = date.today().strftime('%Y-%m-%d')

    # Get daily report
    report = fee_monitor.get_daily_report(today_str)

    if report:
        return {
            "daily_fees": report.total_fees,
            "daily_fee_burn_pct": report.fee_burn_pct,
            "kill_switch_active": report.status == "KILL_SWITCH_ACTIVE",
            "fee_breakdown": fee_monitor.get_fee_breakdown_by_bot(today_str)
        }
    else:
        # No data for today, return defaults
        return {
            "daily_fees": 0.0,
            "daily_fee_burn_pct": 0.0,
            "kill_switch_active": False,
            "fee_breakdown": fee_monitor.get_fee_breakdown_by_bot(today_str)
        }


# ========== Lifecycle Management Endpoints ==========

@router.get("/lifecycle/status")
async def get_lifecycle_status() -> List[Dict[str, Any]]:
    """
    Get lifecycle status for all bots.
    
    Returns tag progression status, promotion criteria, and stats.
    """
    try:
        from src.router.lifecycle_manager import LifecycleManager
        manager = LifecycleManager()
        return manager.get_all_lifecycle_statuses()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lifecycle status: {e}")


@router.get("/lifecycle/status/{bot_id}")
async def get_bot_lifecycle_status(bot_id: str) -> Dict[str, Any]:
    """
    Get lifecycle status for a specific bot.
    
    Args:
        bot_id: Bot identifier
        
    Returns:
        Lifecycle status with promotion criteria and stats
    """
    try:
        from src.router.lifecycle_manager import LifecycleManager
        manager = LifecycleManager()
        status = manager.get_bot_lifecycle_status(bot_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return status
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get lifecycle status: {e}")


@router.post("/lifecycle/check")
async def trigger_lifecycle_check() -> Dict[str, Any]:
    """
    Manually trigger a lifecycle check.
    
    Runs the daily evaluation for all bots and returns the report.
    """
    try:
        from src.router.lifecycle_manager import LifecycleManager
        manager = LifecycleManager()
        report = manager.run_daily_check()
        return report.to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lifecycle check failed: {e}")


@router.post("/lifecycle/promote/{bot_id}")
async def manually_promote_bot(bot_id: str, force: bool = Query(False, description="Bypass criteria check")) -> Dict[str, Any]:
    """
    Manually promote a bot to the next tag level.
    
    Args:
        bot_id: Bot identifier
        force: If True, bypass promotion criteria check
        
    Returns:
        Promotion result
    """
    try:
        from src.router.lifecycle_manager import LifecycleManager
        manager = LifecycleManager()
        result = manager.manually_promote_bot(bot_id, force=force)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Promotion failed"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Promotion failed: {e}")


@router.post("/lifecycle/quarantine/{bot_id}")
async def manually_quarantine_bot(bot_id: str, reason: str = Query("", description="Reason for quarantine")) -> Dict[str, Any]:
    """
    Manually quarantine a bot.
    
    Args:
        bot_id: Bot identifier
        reason: Reason for quarantine
        
    Returns:
        Quarantine result
    """
    try:
        from src.router.lifecycle_manager import LifecycleManager
        manager = LifecycleManager()
        result = manager.manually_quarantine_bot(bot_id, reason=reason)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Quarantine failed"))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quarantine failed: {e}")


# ========== Bot Limits Endpoints ==========

@router.get("/bot-limits/status")
async def get_bot_limits_status(account_balance: Optional[float] = Query(None, description="Account balance")) -> Dict[str, Any]:
    """
    Get current bot limit status for UI display.
    
    Args:
        account_balance: Account balance (optional, uses default if not provided)
        
    Returns:
        Current tier, max bots, active bots, and warnings
    """
    try:
        from src.router.dynamic_bot_limits import DynamicBotLimiter
        from src.router.bot_manifest import BotRegistry
        
        # Get account balance from router if not provided
        if account_balance is None and _strategy_router is not None:
            pks = _strategy_router.progressive_kill_switch
            if pks and hasattr(pks, 'account_monitor') and pks.account_monitor:
                account_states = pks.account_monitor.get_all_states()
                if account_states:
                    first_state = list(account_states.values())[0]
                    account_balance = first_state.get('balance', 10000.0)
        
        if account_balance is None:
            account_balance = 10000.0  # Default
        
        limiter = DynamicBotLimiter()
        max_bots = limiter.get_max_bots(account_balance)
        tier_info = limiter.get_tier_info(account_balance)
        
        # Get active bot count
        active_bots = 0
        try:
            registry = BotRegistry()
            # Count all non-dead bots
            for bot in registry.list_all():
                if "@dead" not in bot.tags:
                    active_bots += 1
        except Exception:
            # Fallback
            if _strategy_router and hasattr(_strategy_router, 'commander'):
                active_bots = len(_strategy_router.commander.active_bots)
        
        # Calculate warnings
        warnings = []
        if active_bots >= max_bots:
            warnings.append(f"Bot limit reached ({active_bots}/{max_bots})")
        elif active_bots >= max_bots * 0.8:
            warnings.append(f"Approaching bot limit ({active_bots}/{max_bots})")
        
        # Check safety buffer
        safety_buffer = limiter.calculate_safety_buffer(account_balance, active_bots)
        
        return {
            "account_balance": account_balance,
            "tier": tier_info.get("tier", "unknown"),
            "tier_range": tier_info.get("range", "unknown"),
            "max_bots": max_bots,
            "active_bots": active_bots,
            "available_slots": max(0, max_bots - active_bots),
            "safety_buffer_required": safety_buffer,
            "can_add_bot": active_bots < max_bots,
            "warnings": warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get bot limits: {e}")


# ========== Market Scanner Endpoints ==========

# In-memory storage for recent alerts (in production, use database)
_recent_scanner_alerts: List[Dict[str, Any]] = []

@router.get("/scanner/alerts")
async def get_scanner_alerts(limit: int = Query(50, description="Maximum alerts to return")) -> List[Dict[str, Any]]:
    """
    Get recent market scanner alerts.
    
    Args:
        limit: Maximum number of alerts to return
        
    Returns:
        List of recent alerts
    """
    # Return recent alerts (limited)
    return _recent_scanner_alerts[-limit:]


@router.get("/scanner/status")
async def get_scanner_status() -> Dict[str, Any]:
    """
    Get market scanner status.
    
    Returns current session, active scanners, and statistics.
    """
    try:
        from src.router.sessions import SessionDetector, get_current_session
        
        current_session = get_current_session()
        session_info = SessionDetector.get_session_info(datetime.utcnow())
        
        return {
            "active": True,
            "current_session": current_session.value,
            "session_info": {
                "is_active": session_info.is_active,
                "time_until_close": session_info.time_until_close_str,
                "next_session": session_info.next_session.value if session_info.next_session else None,
            },
            "scanners": {
                "session_breakout": True,
                "volatility": True,
                "news_events": True,
                "ict_setups": True,
            },
            "alerts_count": len(_recent_scanner_alerts),
            "last_scan": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        return {
            "active": False,
            "error": str(e),
        }


@router.post("/scanner/scan")
async def trigger_market_scan() -> Dict[str, Any]:
    """
    Manually trigger a market scan.
    
    Runs all scanners and returns any new alerts.
    """
    try:
        from src.router.market_scanner import MarketScanner
        
        scanner = MarketScanner()
        alerts = scanner.run_full_scan()
        
        # Store alerts
        _recent_scanner_alerts.extend(alerts)
        
        # Keep only last 1000 alerts
        if len(_recent_scanner_alerts) > 1000:
            _recent_scanner_alerts[:] = _recent_scanner_alerts[-1000:]
        
        return {
            "success": True,
            "alerts_found": len(alerts),
            "alerts": alerts,
        }
    except ImportError:
        # MarketScanner not yet implemented
        return {
            "success": False,
            "error": "MarketScanner not available",
            "alerts_found": 0,
            "alerts": [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market scan failed: {e}")
