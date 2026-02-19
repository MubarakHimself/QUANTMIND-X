"""
Alert Service
=============
Production-ready alert system for QuantMindX trading operations.

Features:
- Email notifications via SMTP
- Rate limiting to prevent alert spam
- Alert severity levels
- Async email sending
- Alert history logging
- Template-based email formatting

Future Extensions:
- Telegram notifications
- Discord webhooks  
- Dashboard push notifications
"""

import asyncio
import logging
import smtplib
import ssl
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Callable
from collections import deque
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    """Alert categories for filtering."""
    TRADE = "trade"
    RISK = "risk"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    CONNECTION = "connection"


@dataclass
class EmailConfig:
    """Email notification configuration."""
    
    enabled: bool = False
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    username: str = ""
    password: str = ""  # App password for Gmail
    from_address: str = ""
    to_addresses: list[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if config has required fields."""
        return (
            self.enabled and
            self.smtp_server and
            self.username and
            self.password and
            self.from_address and
            len(self.to_addresses) > 0
        )


@dataclass
class AlertConfig:
    """Main alert configuration."""
    
    email: EmailConfig = field(default_factory=EmailConfig)
    
    # Rate limiting
    rate_limit_per_minute: int = 10
    rate_limit_per_hour: int = 50
    
    # Filtering
    min_severity: AlertSeverity = AlertSeverity.INFO
    enabled_categories: list[AlertCategory] = field(
        default_factory=lambda: list(AlertCategory)
    )
    
    # Quiet hours (no non-critical alerts)
    quiet_hours_start: int = 23  # 11 PM
    quiet_hours_end: int = 7    # 7 AM
    respect_quiet_hours: bool = True
    
    @classmethod
    def from_file(cls, path: str) -> "AlertConfig":
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        email_data = data.get('email', {})
        return cls(
            email=EmailConfig(**email_data),
            rate_limit_per_minute=data.get('rate_limit_per_minute', 10),
            rate_limit_per_hour=data.get('rate_limit_per_hour', 50),
            min_severity=AlertSeverity(data.get('min_severity', 'info')),
            quiet_hours_start=data.get('quiet_hours_start', 23),
            quiet_hours_end=data.get('quiet_hours_end', 7),
            respect_quiet_hours=data.get('respect_quiet_hours', True)
        )
    
    def to_file(self, path: str) -> None:
        """Save config to JSON file."""
        data = {
            'email': asdict(self.email),
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'rate_limit_per_hour': self.rate_limit_per_hour,
            'min_severity': self.min_severity.value,
            'quiet_hours_start': self.quiet_hours_start,
            'quiet_hours_end': self.quiet_hours_end,
            'respect_quiet_hours': self.respect_quiet_hours
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


@dataclass
class Alert:
    """Individual alert instance."""
    
    title: str
    message: str
    severity: AlertSeverity
    category: AlertCategory
    timestamp: str = ""
    data: dict = field(default_factory=dict)
    sent_email: bool = False
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ============================================================================
# Alert Service
# ============================================================================

class AlertService:
    """
    Production-ready alert service with email notifications.
    
    Features:
    - Rate limiting to prevent spam
    - Quiet hours support
    - Async email sending
    - Alert history
    
    Usage:
        service = AlertService()
        service.configure_email(
            smtp_server="smtp.gmail.com",
            username="your@gmail.com",
            password="your_app_password",
            to_addresses=["alerts@example.com"]
        )
        
        await service.send_alert(
            title="Large Drawdown Detected",
            message="Account drawdown exceeded 5%",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK
        )
    """
    
    def __init__(self, config: AlertConfig = None, config_path: str = None):
        """
        Initialize Alert Service.
        
        Args:
            config: AlertConfig instance.
            config_path: Path to config file (alternative to config object).
        """
        if config:
            self.config = config
        elif config_path and Path(config_path).exists():
            self.config = AlertConfig.from_file(config_path)
        else:
            self.config = AlertConfig()
        
        self._config_path = config_path
        
        # Alert history (last 1000 alerts)
        self._history: deque[Alert] = deque(maxlen=1000)
        
        # Rate limiting
        self._minute_window: deque[datetime] = deque()
        self._hour_window: deque[datetime] = deque()
        self._rate_limit_lock = threading.Lock()
        
        # Async event loop for background email sending
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        logger.info("AlertService initialized")
    
    def configure_email(
        self,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_address: str = None,
        to_addresses: list[str] = None,
        use_tls: bool = True
    ) -> None:
        """
        Configure email notifications.
        
        Args:
            smtp_server: SMTP server address.
            smtp_port: SMTP port (587 for TLS, 465 for SSL).
            username: SMTP username.
            password: SMTP password (use app password for Gmail).
            from_address: Sender email (defaults to username).
            to_addresses: List of recipient emails.
            use_tls: Whether to use TLS encryption.
        """
        self.config.email = EmailConfig(
            enabled=True,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            use_tls=use_tls,
            username=username,
            password=password,
            from_address=from_address or username,
            to_addresses=to_addresses or []
        )
        
        if self._config_path:
            self.config.to_file(self._config_path)
        
        logger.info(f"Email configured: {smtp_server}:{smtp_port}")
    
    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        with self._rate_limit_lock:
            now = datetime.now()
            
            # Clean old entries
            minute_ago = now - timedelta(minutes=1)
            hour_ago = now - timedelta(hours=1)
            
            while self._minute_window and self._minute_window[0] < minute_ago:
                self._minute_window.popleft()
            
            while self._hour_window and self._hour_window[0] < hour_ago:
                self._hour_window.popleft()
            
            # Check limits
            if len(self._minute_window) >= self.config.rate_limit_per_minute:
                logger.warning("Rate limit exceeded (per minute)")
                return False
            
            if len(self._hour_window) >= self.config.rate_limit_per_hour:
                logger.warning("Rate limit exceeded (per hour)")
                return False
            
            return True
    
    def _record_alert_sent(self) -> None:
        """Record an alert was sent for rate limiting."""
        with self._rate_limit_lock:
            now = datetime.now()
            self._minute_window.append(now)
            self._hour_window.append(now)
    
    def _is_quiet_hours(self) -> bool:
        """Check if we're in quiet hours."""
        if not self.config.respect_quiet_hours:
            return False
        
        hour = datetime.now().hour
        start = self.config.quiet_hours_start
        end = self.config.quiet_hours_end
        
        if start < end:
            return start <= hour < end
        else:  # Wraps around midnight
            return hour >= start or hour < end
    
    def _should_send_alert(self, alert: Alert) -> bool:
        """Determine if an alert should be sent based on config."""
        # Check severity
        severity_order = [AlertSeverity.INFO, AlertSeverity.WARNING, 
                         AlertSeverity.ERROR, AlertSeverity.CRITICAL]
        
        min_idx = severity_order.index(self.config.min_severity)
        alert_idx = severity_order.index(alert.severity)
        
        if alert_idx < min_idx:
            return False
        
        # Check category
        if alert.category not in self.config.enabled_categories:
            return False
        
        # Check quiet hours (allow CRITICAL alerts)
        if self._is_quiet_hours() and alert.severity != AlertSeverity.CRITICAL:
            logger.debug(f"Alert suppressed (quiet hours): {alert.title}")
            return False
        
        # Check rate limit
        if not self._check_rate_limit():
            return False
        
        return True
    
    def _send_email_sync(self, alert: Alert) -> bool:
        """Send email synchronously."""
        if not self.config.email.is_valid():
            logger.warning("Email not configured properly")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[QuantMindX {alert.severity.value.upper()}] {alert.title}"
            msg['From'] = self.config.email.from_address
            msg['To'] = ", ".join(self.config.email.to_addresses)
            
            # Plain text version
            text_content = f"""
QuantMindX Trading Alert
========================

Severity: {alert.severity.value.upper()}
Category: {alert.category.value}
Time: {alert.timestamp}

{alert.message}

---
This is an automated message from QuantMindX Trading System.
            """.strip()
            
            # HTML version
            severity_colors = {
                AlertSeverity.INFO: "#17a2b8",
                AlertSeverity.WARNING: "#ffc107",
                AlertSeverity.ERROR: "#dc3545",
                AlertSeverity.CRITICAL: "#721c24"
            }
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f4f4f4; }}
        .container {{ max-width: 600px; margin: 0 auto; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .header {{ background: {severity_colors[alert.severity]}; color: white; padding: 20px; }}
        .content {{ padding: 20px; }}
        .meta {{ color: #666; font-size: 14px; margin-bottom: 15px; }}
        .footer {{ background: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0;">{alert.title}</h1>
        </div>
        <div class="content">
            <div class="meta">
                <strong>Severity:</strong> {alert.severity.value.upper()} | 
                <strong>Category:</strong> {alert.category.value} | 
                <strong>Time:</strong> {alert.timestamp}
            </div>
            <p>{alert.message}</p>
        </div>
        <div class="footer">
            QuantMindX Trading System
        </div>
    </div>
</body>
</html>
            """.strip()
            
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.config.email.smtp_server, self.config.email.smtp_port) as server:
                if self.config.email.use_tls:
                    server.starttls(context=context)
                server.login(self.config.email.username, self.config.email.password)
                server.send_message(msg)
            
            logger.info(f"Email sent: {alert.title}")
            return True
            
        except Exception as e:
            logger.exception(f"Failed to send email: {e}")
            return False
    
    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        category: AlertCategory = AlertCategory.SYSTEM,
        data: dict = None
    ) -> Alert:
        """
        Send an alert notification.
        
        Args:
            title: Alert title/subject.
            message: Alert message body.
            severity: Alert severity level.
            category: Alert category.
            data: Additional data to include.
            
        Returns:
            Alert object with sent status.
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            category=category,
            data=data or {}
        )
        
        # Add to history
        self._history.append(alert)
        
        # Check if we should send
        if not self._should_send_alert(alert):
            return alert
        
        # Send email in background
        if self.config.email.enabled:
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, self._send_email_sync, alert)
            alert.sent_email = success
            
            if success:
                self._record_alert_sent()
        
        return alert
    
    def send_alert_sync(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        category: AlertCategory = AlertCategory.SYSTEM,
        data: dict = None
    ) -> Alert:
        """
        Send an alert synchronously (blocking).
        
        Use this from non-async contexts.
        """
        alert = Alert(
            title=title,
            message=message,
            severity=severity,
            category=category,
            data=data or {}
        )
        
        self._history.append(alert)
        
        if not self._should_send_alert(alert):
            return alert
        
        if self.config.email.enabled:
            alert.sent_email = self._send_email_sync(alert)
            if alert.sent_email:
                self._record_alert_sent()
        
        return alert
    
    def get_history(
        self, 
        limit: int = 50,
        severity: AlertSeverity = None,
        category: AlertCategory = None
    ) -> list[dict]:
        """
        Get recent alert history.
        
        Args:
            limit: Maximum number of alerts to return.
            severity: Filter by severity.
            category: Filter by category.
            
        Returns:
            List of alert dictionaries.
        """
        alerts = list(self._history)
        alerts.reverse()  # Most recent first
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        return [asdict(a) for a in alerts[:limit]]
    
    def test_email(self) -> dict[str, Any]:
        """
        Send a test email to verify configuration.
        
        Returns:
            Dictionary with success status and any errors.
        """
        if not self.config.email.is_valid():
            return {
                "success": False,
                "error": "Email not configured. Call configure_email() first."
            }
        
        alert = Alert(
            title="Test Alert - Email Configuration",
            message="This is a test email from QuantMindX. If you received this, email alerts are working correctly!",
            severity=AlertSeverity.INFO,
            category=AlertCategory.SYSTEM
        )
        
        success = self._send_email_sync(alert)
        
        return {
            "success": success,
            "recipients": self.config.email.to_addresses,
            "error": None if success else "Failed to send email. Check logs for details."
        }


# ============================================================================
# Global Instance
# ============================================================================

_alert_service: Optional[AlertService] = None


def get_alert_service(config_path: str = None) -> AlertService:
    """Get or create the global Alert Service instance."""
    global _alert_service
    if _alert_service is None:
        if config_path is None:
            config_path = str(
                Path.home() / ".quantmindx" / "alerts.json"
            )
        _alert_service = AlertService(config_path=config_path)
    return _alert_service
