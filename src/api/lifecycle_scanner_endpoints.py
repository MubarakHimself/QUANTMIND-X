"""
Lifecycle and Market Scanner API Endpoints

Provides RESTful API endpoints for:
- Manual trigger of lifecycle checks
- Market scanner trigger and alert retrieval
- Scheduler status

Usage:
    These endpoints are automatically registered when included in the app.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["lifecycle", "scanner"])


# =============================================================================
# Lifecycle Manager Endpoints
# =============================================================================

class LifecycleCheckResponse(BaseModel):
    """Response from lifecycle check trigger."""
    success: bool
    message: str
    timestamp: str
    report: Optional[Dict[str, Any]] = None


class LifecycleStatusResponse(BaseModel):
    """Response with lifecycle scheduler status."""
    scheduler_running: bool
    last_check_time: Optional[str]
    last_check_status: str
    next_check_time: Optional[str]


@router.post("/lifecycle/check", response_model=LifecycleCheckResponse)
async def trigger_lifecycle_check():
    """
    Manually trigger a lifecycle check.
    
    This endpoint allows manual invocation of LifecycleManager.run_daily_check()
    for verification purposes.
    
    Returns:
        LifecycleCheckResponse with check results
    """
    try:
        from src.router.lifecycle_manager import LifecycleManager
        
        logger.info("Manual lifecycle check triggered via API")
        
        manager = LifecycleManager()
        report = manager.run_daily_check()
        
        return LifecycleCheckResponse(
            success=True,
            message=f"Lifecycle check completed: {report.to_dict()['summary']}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            report=report.to_dict()
        )
        
    except Exception as e:
        logger.error(f"Lifecycle check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/lifecycle/status", response_model=LifecycleStatusResponse)
async def get_lifecycle_status(request: Request):
    """
    Get lifecycle scheduler status.
    
    Returns:
        LifecycleStatusResponse with scheduler information
    """
    # Query actual scheduler state from app state
    lifecycle_scheduler = getattr(request.app.state, 'lifecycle_scheduler', None)
    
    if lifecycle_scheduler is not None:
        try:
            status = lifecycle_scheduler.get_status()
            return LifecycleStatusResponse(
                scheduler_running=status.get('scheduler_running', False),
                last_check_time=status.get('last_check_time'),
                last_check_status=status.get('last_check_status', 'unknown'),
                next_check_time=status.get('next_check_time')
            )
        except Exception as e:
            logger.error(f"Failed to get lifecycle scheduler status: {e}")
    
    # Fallback to defaults if scheduler not available
    return LifecycleStatusResponse(
        scheduler_running=False,
        last_check_time=None,
        last_check_status="not_configured",
        next_check_time=None
    )


# =============================================================================
# Market Scanner Endpoints
# =============================================================================

class ScannerScanRequest(BaseModel):
    """Request to trigger a market scan."""
    symbols: Optional[List[str]] = Field(
        default=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
        description="List of symbols to scan"
    )


class ScannerAlertResponse(BaseModel):
    """Response containing scanner alerts."""
    success: bool
    message: str
    timestamp: str
    alerts: List[Dict[str, Any]]
    alert_count: int


class ScannerStatusResponse(BaseModel):
    """Response with scanner scheduler status."""
    scheduler_running: bool
    last_scan_time: Optional[str]
    last_scan_status: str
    total_alerts: int
    next_scan_time: Optional[str]


@router.post("/scanner/scan", response_model=ScannerAlertResponse)
async def trigger_market_scan(request: ScannerScanRequest):
    """
    Manually trigger a market scan.
    
    This endpoint allows manual invocation of MarketScanner.run_full_scan()
    for verification or on-demand scanning.
    
    Args:
        request: ScannerScanRequest with optional symbols
        
    Returns:
        ScannerAlertResponse with detected alerts
    """
    try:
        from src.router.market_scanner import MarketScanner
        
        logger.info(f"Manual market scan triggered via API for symbols: {request.symbols}")
        
        scanner = MarketScanner(symbols=request.symbols)
        alerts = scanner.run_full_scan()
        
        return ScannerAlertResponse(
            success=True,
            message=f"Market scan completed: {len(alerts)} opportunities detected",
            timestamp=datetime.now(timezone.utc).isoformat(),
            alerts=alerts,
            alert_count=len(alerts)
        )
        
    except Exception as e:
        logger.error(f"Market scan failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scanner/alerts", response_model=ScannerAlertResponse)
async def get_recent_alerts(limit: int = 100):
    """
    Get recent scanner alerts.
    
    Args:
        limit: Maximum number of alerts to return (default 100)
        
    Returns:
        ScannerAlertResponse with recent alerts
    """
    try:
        from src.router.market_scanner import MarketScanner
        
        scanner = MarketScanner()
        
        # Get recent alerts from the scanner's internal storage
        recent_alerts = scanner._recent_alerts[-limit:] if hasattr(scanner, '_recent_alerts') else []
        
        return ScannerAlertResponse(
            success=True,
            message=f"Retrieved {len(recent_alerts)} recent alerts",
            timestamp=datetime.now(timezone.utc).isoformat(),
            alerts=[a.to_dict() for a in recent_alerts],
            alert_count=len(recent_alerts)
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scanner/status", response_model=ScannerStatusResponse)
async def get_scanner_status():
    """
    Get market scanner scheduler status.
    
    Returns:
        ScannerStatusResponse with scheduler information
    """
    # Query actual scheduler state
    try:
        from src.router.market_scanner import get_scanner_scheduler
        scheduler = get_scanner_scheduler()
        
        if scheduler is not None:
            status = scheduler.get_status()
            
            # Get total alerts count
            total_alerts = 0
            if hasattr(scheduler, 'scanner') and hasattr(scheduler.scanner, '_recent_alerts'):
                total_alerts = len(scheduler.scanner._recent_alerts)
            
            return ScannerStatusResponse(
                scheduler_running=status.get('running', False),
                last_scan_time=status.get('last_scan_time'),
                last_scan_status=status.get('last_scan_status', 'unknown'),
                total_alerts=total_alerts,
                next_scan_time=status.get('next_scan_time')
            )
    except Exception as e:
        logger.error(f"Failed to get scanner scheduler status: {e}")
    
    # Fallback to defaults if scheduler not available
    return ScannerStatusResponse(
        scheduler_running=False,
        last_scan_time=None,
        last_scan_status="not_configured",
        total_alerts=0,
        next_scan_time=None
    )


# =============================================================================
# Router Registration Helper
# =============================================================================

def include_router(app):
    """Include these endpoints in the FastAPI app."""
    app.include_router(router)
    logger.info("Lifecycle and Scanner endpoints registered")
