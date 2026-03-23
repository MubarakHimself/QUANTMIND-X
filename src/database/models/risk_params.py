"""
Risk Parameters Database Model.

Stores risk parameters per account tag for the Risk canvas UI.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from ..models.base import Base


class RiskParams(Base):
    """
    Risk parameters for a specific account tag.

    Attributes:
        id: Primary key
        account_tag: Unique identifier for the account (e.g., "main", "prop-firm-1")
        daily_loss_cap_pct: Maximum daily loss as percentage (e.g., 5.0 = 5%)
        max_trades_per_day: Maximum number of trades allowed per day
        kelly_fraction: Kelly criterion fraction (0 < value <= 1.0)
        position_multiplier: Position size multiplier
        lyapunov_threshold: Lyapunov exponent threshold for chaos detection
        hmm_retrain_trigger: HMM model retrain trigger threshold
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'risk_params'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_tag = Column(String(50), nullable=False, unique=True, index=True)
    daily_loss_cap_pct = Column(Float, nullable=False, default=5.0)
    max_trades_per_day = Column(Integer, nullable=False, default=10)
    kelly_fraction = Column(Float, nullable=False, default=0.5)
    position_multiplier = Column(Float, nullable=False, default=1.0)
    lyapunov_threshold = Column(Float, nullable=False, default=0.3)
    hmm_retrain_trigger = Column(Float, nullable=False, default=0.7)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    __table_args__ = (
        Index('idx_risk_params_account_tag', 'account_tag'),
    )

    def __repr__(self):
        return f"<RiskParams(account_tag={self.account_tag}, kelly={self.kelly_fraction})>"


class RiskParamsAudit(Base):
    """
    Audit log for risk parameter changes.

    Attributes:
        id: Primary key
        account_tag: Account tag that was modified
        field_changed: Name of the field that was changed
        old_value: Previous value (stored as string for flexibility)
        new_value: New value (stored as string for flexibility)
        changed_at: Timestamp of the change
        changed_by: Source of the change (e.g., "api", "system")
    """
    __tablename__ = 'risk_params_audit'

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_tag = Column(String(50), nullable=False, index=True)
    field_changed = Column(String(50), nullable=False)
    old_value = Column(String(100), nullable=True)
    new_value = Column(String(100), nullable=True)
    changed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    changed_by = Column(String(50), nullable=False, default='api')

    __table_args__ = (
        Index('idx_risk_audit_account_tag', 'account_tag'),
        Index('idx_risk_audit_changed_at', 'changed_at'),
    )

    def __repr__(self):
        return f"<RiskParamsAudit(account_tag={self.account_tag}, field={self.field_changed})>"