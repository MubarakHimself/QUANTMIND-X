"""
Monitoring models.

Contains models for alerts, webhooks, and system monitoring.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Index, Text
from ..models.base import Base


class AlertHistory(Base):
    """
    Alert History for Progressive Kill Switch System.

    Stores all raised alerts for audit and analysis.

    Attributes:
        id: Primary key
        level: Alert level (GREEN, YELLOW, ORANGE, RED, BLACK)
        tier: Protection tier (1-5)
        message: Alert description
        threshold_pct: Threshold percentage when raised
        triggered_at: When alert was raised
        source: Component that raised the alert
        alert_metadata: JSON with additional context
        is_active: Whether alert is still active
        cleared_at: When alert was cleared
        created_at: Record creation timestamp
    """
    __tablename__ = 'alert_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(10), nullable=False, index=True)
    tier = Column(Integer, nullable=False, index=True)
    message = Column(Text, nullable=False)
    threshold_pct = Column(Float, nullable=False)
    triggered_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    source = Column(String(50), nullable=False, index=True)
    alert_metadata = Column(String, nullable=True)  # JSON string
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    cleared_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_alert_history_level_tier', 'level', 'tier'),
        Index('idx_alert_history_triggered', 'triggered_at'),
    )

    def __repr__(self):
        return f"<AlertHistory(id={self.id}, level={self.level}, tier={self.tier}, source={self.source})>"


class WebhookLog(Base):
    """
    Webhook Log for TradingView alert tracking.

    Records all incoming webhook alerts from TradingView for audit,
    debugging, and performance analysis.

    Attributes:
        id: Primary key
        timestamp: When the webhook was received
        source_ip: IP address of the sender
        alert_payload: JSON with the full alert data
        signature_valid: Whether HMAC signature was valid
        bot_triggered: Whether a bot was successfully triggered
        order_id: MT5 order ID if trade was placed
        execution_time_ms: Processing time in milliseconds
        error_message: Error message if processing failed
        created_at: Record creation timestamp
    """
    __tablename__ = 'webhook_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    source_ip = Column(String(50), nullable=False, index=True)
    alert_payload = Column(String, nullable=False)  # JSON string
    signature_valid = Column(Boolean, nullable=False, default=False)
    bot_triggered = Column(Boolean, nullable=False, default=False, index=True)
    order_id = Column(String(100), nullable=True)
    execution_time_ms = Column(Float, nullable=False, default=0.0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_webhook_logs_timestamp', 'timestamp'),
        Index('idx_webhook_logs_bot_triggered', 'bot_triggered'),
        Index('idx_webhook_logs_source_ip', 'source_ip'),
    )

    def __repr__(self):
        return f"<WebhookLog(id={self.id}, ip={self.source_ip}, triggered={self.bot_triggered}, time={self.execution_time_ms}ms)>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source_ip": self.source_ip,
            "alert_payload": self.alert_payload,
            "signature_valid": self.signature_valid,
            "bot_triggered": self.bot_triggered,
            "order_id": self.order_id,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
