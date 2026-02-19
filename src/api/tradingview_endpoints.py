"""
TradingView Webhook Endpoints for QuantMind.

Handles incoming webhook alerts from TradingView with security validation
and bot execution triggers.
"""

import hashlib
import hmac
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from ipaddress import ip_address, ip_network

from fastapi import APIRouter, HTTPException, Request, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis

from src.config import get_settings
from src.database.models import WebhookLog, SessionLocal

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/tradingview", tags=["tradingview"])

# Configuration
settings = get_settings()

# TradingView IP ranges (update as needed)
TRADINGVIEW_IP_WHITELIST = [
    "52.89.214.238",
    "34.212.75.30",
    "54.218.53.128",
    "52.32.178.7",
    "127.0.0.1",  # For local testing
]

# Rate limiting configuration
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds

# In-memory rate limiting storage
_rate_limit_storage: Dict[str, list] = {}
_redis_client: Optional[redis.Redis] = None


def _get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client for rate limiting."""
    global _redis_client
    if _redis_client is None:
        try:
            redis_url = getattr(settings, 'REDIS_URL', None)
            if redis_url:
                _redis_client = redis.from_url(redis_url)
        except Exception as e:
            logger.warning(f"Could not connect to Redis: {e}")
    return _redis_client


# Pydantic Models
class WebhookAlert(BaseModel):
    """TradingView webhook alert payload."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    action: str = Field(..., description="Action: buy, sell, close, close_all")
    price: Optional[float] = Field(None, description="Entry/exit price")
    strategy: Optional[str] = Field(None, description="Strategy name")
    timeframe: Optional[str] = Field(None, description="Chart timeframe")
    timestamp: Optional[int] = Field(None, description="Unix timestamp")
    signature: Optional[str] = Field(None, description="HMAC-SHA256 signature")
    volume: Optional[float] = Field(None, description="Position size in lots")
    stop_loss: Optional[float] = Field(None, description="Stop loss price")
    take_profit: Optional[float] = Field(None, description="Take profit price")
    comment: Optional[str] = Field(None, description="Additional notes")

    @validator('action')
    def validate_action(cls, v):
        allowed = ['buy', 'sell', 'close', 'close_all']
        if v.lower() not in allowed:
            raise ValueError(f"Action must be one of: {allowed}")
        return v.lower()

    @validator('symbol')
    def validate_symbol(cls, v):
        return v.upper().replace("/", "").replace("-", "")


class WebhookResponse(BaseModel):
    """Response for webhook alert processing."""
    status: str = Field(..., description="Processing status")
    bot_id: Optional[str] = Field(None, description="Triggered bot ID")
    order_id: Optional[str] = Field(None, description="MT5 order ID")
    execution_time_ms: float = Field(..., description="Processing time in milliseconds")
    message: Optional[str] = Field(None, description="Additional message")


class WebhookStatus(BaseModel):
    """Webhook service status."""
    status: str = "operational"
    total_alerts_today: int = 0
    successful_alerts_today: int = 0
    failed_alerts_today: int = 0


# Security Functions
def validate_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Validate HMAC-SHA256 signature from TradingView.
    
    Args:
        payload: Raw request body bytes
        signature: Signature from X-Signature header
        secret: Webhook secret from environment
    
    Returns:
        True if signature is valid
    """
    if not secret or not signature:
        return False
    
    expected_signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)


def validate_timestamp(timestamp: Optional[int], max_age_seconds: int = 60) -> bool:
    """
    Validate timestamp is within acceptable range.
    
    Args:
        timestamp: Unix timestamp from webhook
        max_age_seconds: Maximum age in seconds
    
    Returns:
        True if timestamp is valid and fresh
    """
    if timestamp is None:
        return True  # Skip validation if no timestamp provided
    
    current_time = int(time.time())
    age = abs(current_time - timestamp)
    
    return age <= max_age_seconds


def is_ip_allowed(client_ip: str) -> bool:
    """
    Check if client IP is in TradingView whitelist.
    
    Args:
        client_ip: Client IP address
    
    Returns:
        True if IP is whitelisted
    """
    try:
        client = ip_address(client_ip)
        for allowed in TRADINGVIEW_IP_WHITELIST:
            if '/' in allowed:
                if client in ip_network(allowed, strict=False):
                    return True
            elif client == ip_address(allowed):
                return True
        return False
    except ValueError:
        return False


# Rate Limiting
def _check_memory_rate_limit(client_ip: str) -> bool:
    """
    In-memory rate limiting fallback.
    
    Args:
        client_ip: Client IP address
    
    Returns:
        True if within rate limit, False if exceeded
    """
    import time
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW
    
    # Clean up old entries and get current requests
    if client_ip not in _rate_limit_storage:
        _rate_limit_storage[client_ip] = []
    
    # Filter to only requests within the window
    _rate_limit_storage[client_ip] = [
        ts for ts in _rate_limit_storage[client_ip] if ts > window_start
    ]
    
    # Check if limit exceeded
    if len(_rate_limit_storage[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    _rate_limit_storage[client_ip].append(current_time)
    return True


async def check_rate_limit(client_ip: str, redis_client: Optional[redis.Redis] = None) -> bool:
    """
    Check if client has exceeded rate limit.
    
    Args:
        client_ip: Client IP address
        redis_client: Optional Redis client for distributed rate limiting
    
    Returns:
        True if within rate limit, False if exceeded
    """
    # Try Redis first if not provided
    if redis_client is None:
        redis_client = _get_redis_client()
    
    if redis_client:
        try:
            key = f"rate_limit:tradingview:{client_ip}"
            current = await redis_client.incr(key)
            if current == 1:
                await redis_client.expire(key, RATE_LIMIT_WINDOW)
            return current <= RATE_LIMIT_REQUESTS
        except Exception as e:
            logger.warning(f"Redis rate limiting failed, falling back to in-memory: {e}")
    
    # Fallback to in-memory rate limiting
    return _check_memory_rate_limit(client_ip)


async def trigger_bot_execution(alert: WebhookAlert) -> Dict[str, Any]:
    """
    Trigger bot execution based on webhook alert.
    
    Args:
        alert: Validated webhook alert
    
    Returns:
        Execution result with bot_id and order_id
    """
    from src.router.commander import Commander
    
    try:
        commander = Commander()
        
        # Map action to execution type
        action_map = {
            'buy': 'long',
            'sell': 'short',
            'close': 'close',
            'close_all': 'close_all'
        }
        
        execution_params = {
            'symbol': alert.symbol,
            'action': action_map.get(alert.action, alert.action),
            'volume': alert.volume or 0.01,  # Default to minimum lot size
            'price': alert.price,
            'stop_loss': alert.stop_loss,
            'take_profit': alert.take_profit,
            'comment': alert.comment or f"TradingView: {alert.strategy or 'webhook'}",
            'strategy': alert.strategy,
            'timeframe': alert.timeframe,
        }
        
        result = await commander.execute_signal(execution_params)
        
        return {
            'bot_id': result.get('bot_id'),
            'order_id': result.get('order_id'),
            'status': 'success',
            'message': result.get('message')
        }
    except Exception as e:
        logger.error(f"Bot execution failed: {e}")
        return {
            'bot_id': None,
            'order_id': None,
            'status': 'error',
            'message': str(e)
        }


async def log_webhook(
    source_ip: str,
    alert_payload: Dict[str, Any],
    signature_valid: bool,
    bot_triggered: bool,
    order_id: Optional[str],
    execution_time_ms: float,
    error_message: Optional[str]
):
    """
    Log webhook attempt to database.
    
    Args:
        source_ip: Client IP address
        alert_payload: Received alert data
        signature_valid: Whether signature was valid
        bot_triggered: Whether bot was triggered
        order_id: MT5 order ID if placed
        execution_time_ms: Processing time
        error_message: Error message if any
    """
    try:
        with SessionLocal() as session:
            log_entry = WebhookLog(
                source_ip=source_ip,
                alert_payload=alert_payload,
                signature_valid=signature_valid,
                bot_triggered=bot_triggered,
                order_id=order_id,
                execution_time_ms=execution_time_ms,
                error_message=error_message
            )
            session.add(log_entry)
            session.commit()
    except Exception as e:
        logger.error(f"Failed to log webhook: {e}")


# API Endpoints
@router.post("/webhook", response_model=WebhookResponse)
async def receive_webhook(
    request: Request,
    alert: WebhookAlert,
    background_tasks: BackgroundTasks
):
    """
    Receive and process TradingView webhook alerts.
    
    This endpoint:
    1. Validates IP whitelist
    2. Validates HMAC signature
    3. Validates timestamp freshness
    4. Checks rate limits
    5. Triggers bot execution
    6. Logs the attempt
    
    Security:
    - HMAC-SHA256 signature validation
    - IP whitelist for TradingView IPs
    - Rate limiting (100 requests/minute per IP)
    - Timestamp validation (60 second window)
    """
    start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    # Get raw body for signature validation
    body = await request.body()
    # Accept both X-Webhook-Signature (TradingView) and X-Signature (legacy)
    signature = request.headers.get("X-Webhook-Signature") or request.headers.get("X-Signature", "")
    header_used = "X-Webhook-Signature" if request.headers.get("X-Webhook-Signature") else "X-Signature"
    logger.debug(f"Using signature from {header_used} header")
    
    # Step 1: IP whitelist validation
    if not is_ip_allowed(client_ip):
        logger.warning(f"Rejected webhook from non-whitelisted IP: {client_ip}")
        raise HTTPException(status_code=403, detail="IP not whitelisted")
    
    # Step 2: Signature validation
    webhook_secret = getattr(settings, 'TRADINGVIEW_WEBHOOK_SECRET', '')
    signature_valid = validate_signature(body, signature, webhook_secret)
    
    if webhook_secret and not signature_valid:
        execution_time = (time.time() - start_time) * 1000
        background_tasks.add_task(
            log_webhook,
            client_ip,
            alert.dict(),
            False,
            False,
            None,
            execution_time,
            "Invalid signature"
        )
        raise HTTPException(status_code=401, detail="Invalid signature")
    
    # Step 3: Timestamp validation
    if not validate_timestamp(alert.timestamp):
        execution_time = (time.time() - start_time) * 1000
        background_tasks.add_task(
            log_webhook,
            client_ip,
            alert.dict(),
            signature_valid,
            False,
            None,
            execution_time,
            "Stale timestamp"
        )
        raise HTTPException(status_code=400, detail="Alert timestamp too old")
    
    # Step 4: Rate limiting
    redis_client = _get_redis_client()
    if not await check_rate_limit(client_ip, redis_client):
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Step 5: Trigger bot execution
    try:
        result = await trigger_bot_execution(alert)
        execution_time = (time.time() - start_time) * 1000
        
        # Step 6: Log successful webhook
        background_tasks.add_task(
            log_webhook,
            client_ip,
            alert.dict(),
            signature_valid,
            True,
            result.get('order_id'),
            execution_time,
            None
        )
        
        logger.info(
            f"Webhook processed: {alert.symbol} {alert.action} "
            f"(bot: {result.get('bot_id')}, order: {result.get('order_id')}, "
            f"time: {execution_time:.2f}ms)"
        )
        
        return WebhookResponse(
            status=result.get('status', 'success'),
            bot_id=result.get('bot_id'),
            order_id=result.get('order_id'),
            execution_time_ms=execution_time,
            message=result.get('message')
        )
        
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        error_msg = str(e)
        
        # Log failed webhook
        background_tasks.add_task(
            log_webhook,
            client_ip,
            alert.dict(),
            signature_valid,
            False,
            None,
            execution_time,
            error_msg
        )
        
        logger.error(f"Webhook execution failed: {error_msg}")
        
        return WebhookResponse(
            status="error",
            execution_time_ms=execution_time,
            message=error_msg
        )


@router.get("/status", response_model=WebhookStatus)
async def get_webhook_status():
    """
    Get webhook service status and statistics.
    
    Returns:
        Status with alert counts for today
    """
    try:
        with SessionLocal() as session:
            today = datetime.utcnow().date()
            today_start = datetime.combine(today, datetime.min.time())
            
            total = session.query(WebhookLog).filter(
                WebhookLog.timestamp >= today_start
            ).count()
            
            successful = session.query(WebhookLog).filter(
                WebhookLog.timestamp >= today_start,
                WebhookLog.bot_triggered == True
            ).count()
            
            failed = session.query(WebhookLog).filter(
                WebhookLog.timestamp >= today_start,
                WebhookLog.bot_triggered == False
            ).count()
            
            return WebhookStatus(
                status="operational",
                total_alerts_today=total,
                successful_alerts_today=successful,
                failed_alerts_today=failed
            )
    except Exception as e:
        logger.error(f"Failed to get webhook status: {e}")
        return WebhookStatus(status="degraded")


@router.get("/logs")
async def get_webhook_logs(limit: int = 100, offset: int = 0):
    """
    Get recent webhook logs.
    
    Args:
        limit: Maximum number of logs to return
        offset: Number of logs to skip
    
    Returns:
        List of recent webhook logs
    """
    try:
        with SessionLocal() as session:
            logs = session.query(WebhookLog).order_by(
                WebhookLog.timestamp.desc()
            ).offset(offset).limit(limit).all()
            
            return [
                {
                    "id": log.id,
                    "timestamp": log.timestamp.isoformat(),
                    "source_ip": log.source_ip,
                    "symbol": log.alert_payload.get("symbol"),
                    "action": log.alert_payload.get("action"),
                    "signature_valid": log.signature_valid,
                    "bot_triggered": log.bot_triggered,
                    "order_id": log.order_id,
                    "execution_time_ms": log.execution_time_ms,
                    "error_message": log.error_message
                }
                for log in logs
            ]
    except Exception as e:
        logger.error(f"Failed to get webhook logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")


@router.post("/test")
async def test_webhook(alert: WebhookAlert):
    """
    Test endpoint for webhook integration (development only).
    
    Bypasses signature validation for testing purposes.
    Should be disabled in production.
    """
    if not getattr(settings, 'DEBUG', False):
        raise HTTPException(status_code=403, detail="Test endpoint disabled in production")
    
    logger.info(f"Test webhook received: {alert.symbol} {alert.action}")
    
    result = await trigger_bot_execution(alert)
    
    return {
        "status": "test_success",
        "alert": alert.dict(),
        "execution_result": result
    }