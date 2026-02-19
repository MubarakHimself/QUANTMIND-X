"""
Alert Service MCP Tools
=======================
MCP tool wrappers for the alert notification system.
"""

from typing import Any

from .alert_service import (
    AlertService,
    AlertSeverity,
    AlertCategory,
    get_alert_service,
)


def register_alert_tools(mcp):
    """
    Register alert tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance.
    """
    
    @mcp.tool()
    def configure_email_alerts(
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        to_addresses: list[str] = None
    ) -> dict[str, Any]:
        """
        Configure email notifications for trading alerts.
        
        For Gmail, use an App Password (not your regular password):
        1. Go to Google Account > Security > 2-Step Verification
        2. At the bottom, select "App passwords"
        3. Generate a new app password for "Mail"
        
        Args:
            smtp_server: SMTP server address (default: Gmail).
            smtp_port: SMTP port (587 for TLS, 465 for SSL).
            username: Your email address.
            password: SMTP password or app password.
            to_addresses: List of email addresses to receive alerts.
            
        Returns:
            Dictionary with configuration status.
            
        Example:
            configure_email_alerts(
                username="your@gmail.com",
                password="your_app_password",
                to_addresses=["alerts@example.com", "backup@example.com"]
            )
        """
        try:
            service = get_alert_service()
            service.configure_email(
                smtp_server=smtp_server,
                smtp_port=smtp_port,
                username=username,
                password=password,
                to_addresses=to_addresses or []
            )
            return {
                "success": True,
                "smtp_server": smtp_server,
                "recipients": to_addresses
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    def test_email_alert() -> dict[str, Any]:
        """
        Send a test email to verify alert configuration.
        
        Returns:
            Dictionary with:
            - success: True if email was sent
            - recipients: List of email addresses
            - error: Error message if failed
        """
        try:
            service = get_alert_service()
            return service.test_email()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    def send_trading_alert(
        title: str,
        message: str,
        severity: str = "info",
        category: str = "trade",
        data: dict = None
    ) -> dict[str, Any]:
        """
        Send a trading alert notification.
        
        Alerts can be sent for various trading events:
        - Trade execution (entry/exit)
        - Risk warnings (drawdown, position size)
        - System events (connection loss, EA status)
        - Performance updates (daily P&L summary)
        
        Args:
            title: Alert title/subject line.
            message: Alert message body.
            severity: Alert severity level:
                - "info": Informational (trade filled, etc.)
                - "warning": Warning (approaching risk limits)
                - "error": Error (trade failed)
                - "critical": Critical (margin call, connection loss)
            category: Alert category:
                - "trade": Trade-related alerts
                - "risk": Risk management alerts
                - "system": System/technical alerts
                - "performance": Performance updates
                - "connection": Connection status
            data: Additional data to include (optional).
            
        Returns:
            Dictionary with:
            - success: True if alert was processed
            - sent_email: True if email was sent
            - alert_id: Unique alert identifier
            
        Example:
            # Trade alert
            send_trading_alert(
                title="Position Opened - EURUSD",
                message="Buy 0.1 lot EURUSD at 1.0850, SL: 1.0820, TP: 1.0900",
                severity="info",
                category="trade"
            )
            
            # Risk warning
            send_trading_alert(
                title="Drawdown Warning",
                message="Daily drawdown reached 3.5% (limit: 5%)",
                severity="warning",
                category="risk"
            )
        """
        try:
            service = get_alert_service()
            
            alert = service.send_alert_sync(
                title=title,
                message=message,
                severity=AlertSeverity(severity),
                category=AlertCategory(category),
                data=data
            )
            
            return {
                "success": True,
                "sent_email": alert.sent_email,
                "timestamp": alert.timestamp,
                "severity": alert.severity.value,
                "category": alert.category.value
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    def get_alert_history(
        limit: int = 50,
        severity: str = None,
        category: str = None
    ) -> list[dict[str, Any]]:
        """
        Get recent alert history.
        
        Args:
            limit: Maximum number of alerts to return (default: 50).
            severity: Filter by severity (info, warning, error, critical).
            category: Filter by category (trade, risk, system, performance, connection).
            
        Returns:
            List of alert dictionaries with title, message, severity,
            category, timestamp, and sent status.
        """
        try:
            service = get_alert_service()
            
            sev = AlertSeverity(severity) if severity else None
            cat = AlertCategory(category) if category else None
            
            return service.get_history(limit=limit, severity=sev, category=cat)
        except Exception as e:
            return [{"error": str(e)}]
    
    @mcp.tool()
    def get_alert_config() -> dict[str, Any]:
        """
        Get current alert configuration.
        
        Returns:
            Dictionary with email settings, rate limits, and filtering options.
            Note: Password is not included for security.
        """
        try:
            service = get_alert_service()
            config = service.config
            
            return {
                "email_enabled": config.email.enabled,
                "smtp_server": config.email.smtp_server,
                "from_address": config.email.from_address,
                "to_addresses": config.email.to_addresses,
                "rate_limit_per_minute": config.rate_limit_per_minute,
                "rate_limit_per_hour": config.rate_limit_per_hour,
                "min_severity": config.min_severity.value,
                "quiet_hours": {
                    "enabled": config.respect_quiet_hours,
                    "start": config.quiet_hours_start,
                    "end": config.quiet_hours_end
                }
            }
        except Exception as e:
            return {"error": str(e)}
