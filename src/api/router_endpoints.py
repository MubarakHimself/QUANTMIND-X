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


# Endpoints

@router.get("/state")
async def get_router_state() -> RouterState:
    """Get current router state."""
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured. Set global StrategyRouter via set_strategy_router()."
        )

    status = _strategy_router.get_status()
    return RouterState(
        active=status.get("router", {}).get("active", False),
        mode=status.get("router", {}).get("mode", "auction"),
        auctionInterval=status.get("router", {}).get("auction_interval_ms", 5000),
        lastAuction=status.get("router", {}).get("last_auction"),
        queuedSignals=status.get("router", {}).get("queued_signals", 0),
        activeAuctions=status.get("router", {}).get("active_auctions", 0),
    )

@router.post("/toggle")
async def toggle_router(active: bool = None) -> Dict[str, Any]:
    """Toggle router on/off."""
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    if active is not None:
        _strategy_router.set_active(active)
    return {"success": True, "active": _strategy_router.is_active()}

@router.post("/settings")
async def save_router_settings(settings: RouterState) -> Dict[str, Any]:
    """Save router settings."""
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    _strategy_router.set_mode(settings.mode)
    _strategy_router.set_auction_interval(settings.auctionInterval)
    return {
        "success": True,
        "mode": settings.mode,
        "auctionInterval": settings.auctionInterval
    }

@router.get("/market")
async def get_market_state() -> Dict[str, Any]:
    """Get current market state including regime and symbols."""
    if _strategy_router is None:
        return {
            "regime": None,
            "symbols": [],
            "unavailable_reason": "Router not configured",
        }

    status = _strategy_router.get_status()
    sentinel = status.get("sentinel", {})

    # Get symbol data from MT5 bridge if available
    symbols = []
    try:
        from src.data.brokers.mt5_socket_adapter import get_mt5_adapter
        adapter = get_mt5_adapter()
        if adapter:
            for symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
                try:
                    tick = adapter.get_tick(symbol)
                    if tick:
                        symbols.append({
                            "symbol": symbol,
                            "price": tick.bid,
                            "change": 0.0,
                            "spread": tick.spread
                        })
                except Exception:
                    pass
    except ImportError:
        pass

    if not symbols:
        return {
            "regime": {
                "quality": sentinel.get("regime_quality", 0.0),
                "trend": sentinel.get("current_regime", "UNKNOWN"),
                "chaos": sentinel.get("chaos", 0.0),
                "volatility": sentinel.get("volatility", "UNKNOWN")
            },
            "symbols": [],
            "unavailable_reason": "Market data not available. MT5 adapter not connected.",
        }

    return {
        "regime": {
            "quality": sentinel.get("regime_quality", 0.0),
            "trend": sentinel.get("current_regime", "UNKNOWN"),
            "chaos": sentinel.get("chaos", 0.0),
            "volatility": sentinel.get("volatility", "UNKNOWN")
        },
        "symbols": symbols,
        "unavailable_reason": None,
    }

@router.get("/bots", response_model=PaginatedResponse[BotSignal])
async def get_bot_signals(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip")
) -> PaginatedResponse[BotSignal]:
    """Get current bot signals with pagination."""
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    commander = _strategy_router.commander
    all_bots = []

    # Get bots from BotRegistry if available
    if commander.bot_registry:
        for manifest in commander.bot_registry.list_all():
            all_bots.append(BotSignal(
                id=manifest.bot_id,
                name=manifest.name,
                symbol=manifest.symbols[0] if manifest.symbols else "UNKNOWN",
                status="ready",
                signalStrength=0.0,
                conditions=[],
                score=0.0,
                lastSignal=None
            ))

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
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    # Get real auctions from router
    status = _strategy_router.get_status()
    auctions_data = status.get("auctions", [])

    all_auctions = [Auction(**a) for a in auctions_data]
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
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    try:
        winner_id, winning_score = _strategy_router.run_auction()
        return {
            "success": True,
            "auction": {
                "id": f"auction-{datetime.utcnow().timestamp()}",
                "timestamp": datetime.utcnow().isoformat(),
                "winner": winner_id,
                "winningScore": winning_score,
                "status": "completed"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/rankings")
async def get_rankings(period: str = "daily") -> List[Dict[str, Any]]:
    """Get bot rankings for specified period."""
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    if period not in ("daily", "weekly"):
        raise HTTPException(status_code=400, detail="Invalid period. Use 'daily' or 'weekly'")

    # Get rankings from router's commander
    commander = _strategy_router.commander
    if hasattr(commander, 'get_bot_rankings'):
        return commander.get_bot_rankings(period)

    raise HTTPException(
        status_code=503,
        detail="Rankings not available from router"
    )

@router.get("/correlations")
async def get_correlations() -> List[Dict[str, Any]]:
    """Get current symbol correlations."""
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    # Get correlation data from correlation sensor
    try:
        from src.risk.physics.correlation_sensor import get_correlation_sensor
        sensor = get_correlation_sensor()
        return sensor.get_current_correlations()
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Correlation sensor not available"
        )

@router.get("/house-money")
async def get_house_money_state() -> HouseMoneyState:
    """Get current house money state."""
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    # Get house money state from progressive kill switch
    pks = _strategy_router.progressive_kill_switch
    if pks and hasattr(pks, 'get_house_money_state'):
        state = pks.get_house_money_state()
        return HouseMoneyState(**state)

    raise HTTPException(
        status_code=503,
        detail="House money state not available"
    )

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
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

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
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

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
    if _strategy_router is None:
        raise HTTPException(
            status_code=503,
            detail="Router not configured"
        )

    from datetime import date
    from src.router.fee_monitor import FeeMonitor

    # Default values if router not initialized
    account_id = "default"
    account_balance = 10000.0

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
    except ImportError as e:
        raise HTTPException(
            status_code=503,
            detail=f"MarketScanner not available: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market scan failed: {e}")
