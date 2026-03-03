"""
HMM (Hidden Markov Model) regime detection models.

Contains models for HMM model metadata, shadow logs, deployments, and sync status.
"""

from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Index, UniqueConstraint, Text
from sqlalchemy.orm import relationship
from ..models.base import Base


class HMMModel(Base):
    """
    HMM Model for Regime Detection.

    Stores trained Hidden Markov Model metadata for regime detection.
    Models are trained on Contabo (training server) and synced to Cloudzy.

    Attributes:
        id: Primary key
        version: Model version string (e.g., "1.0.0", "1.1.0")
        model_type: Training hierarchy level ('universal', 'per_symbol', 'per_symbol_timeframe')
        symbol: Trading symbol (e.g., "EURUSD", "GBPUSD") - null for universal
        timeframe: Timeframe (e.g., "M5", "H1", "H4") - null for universal/per_symbol
        n_states: Number of hidden states (typically 4)
        log_likelihood: Training log-likelihood score
        state_distribution: JSON with state distribution percentages
        transition_matrix: JSON with transition probability matrix
        training_samples: Number of samples used for training
        training_date: When model was trained
        checksum: SHA256 checksum of model file for integrity
        file_path: Path to .pkl model file
        is_active: Whether model is currently active
        validation_status: Validation status (pending, validated, rejected)
        validation_notes: Notes from validation process
        created_at: Record creation timestamp
    """
    __tablename__ = 'hmm_models'

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(20), nullable=False, index=True)
    model_type = Column(String(30), nullable=False, index=True)
    symbol = Column(String(20), nullable=True, index=True)
    timeframe = Column(String(10), nullable=True, index=True)
    n_states = Column(Integer, nullable=False, default=4)
    log_likelihood = Column(Float, nullable=True)
    state_distribution = Column(String, nullable=True)  # JSON string
    transition_matrix = Column(String, nullable=True)  # JSON string
    training_samples = Column(Integer, nullable=False, default=0)
    training_date = Column(DateTime, nullable=True)
    checksum = Column(String(64), nullable=True)
    file_path = Column(String(500), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    validation_status = Column(String(20), nullable=False, default='pending', index=True)
    validation_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    deployments = relationship("HMMDeployment", back_populates="model", cascade="all, delete-orphan")
    shadow_logs = relationship("HMMShadowLog", back_populates="model")

    __table_args__ = (
        Index('idx_hmm_models_version', 'version'),
        Index('idx_hmm_models_type_symbol', 'model_type', 'symbol'),
        Index('idx_hmm_models_active', 'is_active'),
        UniqueConstraint('version', 'symbol', 'timeframe', name='uq_hmm_model_version_symbol_tf'),
    )

    def __repr__(self):
        return f"<HMMModel(id={self.id}, version={self.version}, type={self.model_type}, symbol={self.symbol}, tf={self.timeframe})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "version": self.version,
            "model_type": self.model_type,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_states": self.n_states,
            "log_likelihood": self.log_likelihood,
            "state_distribution": self.state_distribution,
            "transition_matrix": self.transition_matrix,
            "training_samples": self.training_samples,
            "training_date": self.training_date.isoformat() if self.training_date else None,
            "checksum": self.checksum,
            "file_path": self.file_path,
            "is_active": self.is_active,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class HMMShadowLog(Base):
    """
    HMM Shadow Mode Log for prediction comparison.

    Records HMM vs Ising predictions during shadow mode for validation
    and performance comparison.

    Attributes:
        id: Primary key
        model_id: Foreign key to HMMModel
        timestamp: When the prediction was made
        symbol: Trading symbol
        timeframe: Timeframe
        ising_regime: Ising model regime prediction
        ising_confidence: Ising model confidence (0-1)
        hmm_regime: HMM regime prediction
        hmm_state: HMM state ID (0-3)
        hmm_confidence: HMM confidence (0-1)
        agreement: Whether Ising and HMM agree
        decision_source: Which model was used for final decision
        market_context: JSON with market data at prediction time
        created_at: Record creation timestamp
    """
    __tablename__ = 'hmm_shadow_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('hmm_models.id'), nullable=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    ising_regime = Column(String(50), nullable=False)
    ising_confidence = Column(Float, nullable=False, default=0.0)
    hmm_regime = Column(String(50), nullable=False)
    hmm_state = Column(Integer, nullable=False)
    hmm_confidence = Column(Float, nullable=False, default=0.0)
    agreement = Column(Boolean, nullable=False, default=False)
    decision_source = Column(String(20), nullable=False, default='ising')  # 'ising', 'hmm', 'weighted'
    market_context = Column(String, nullable=True)  # JSON string
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    model = relationship("HMMModel", back_populates="shadow_logs")

    __table_args__ = (
        Index('idx_hmm_shadow_timestamp', 'timestamp'),
        Index('idx_hmm_shadow_symbol_tf', 'symbol', 'timeframe'),
        Index('idx_hmm_shadow_agreement', 'agreement'),
    )

    def __repr__(self):
        return f"<HMMShadowLog(id={self.id}, symbol={self.symbol}, ising={self.ising_regime}, hmm={self.hmm_regime}, agree={self.agreement})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "ising_regime": self.ising_regime,
            "ising_confidence": self.ising_confidence,
            "hmm_regime": self.hmm_regime,
            "hmm_state": self.hmm_state,
            "hmm_confidence": self.hmm_confidence,
            "agreement": self.agreement,
            "decision_source": self.decision_source,
            "market_context": self.market_context,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class HMMDeployment(Base):
    """
    HMM Deployment State for tracking deployment history.

    Tracks the state transitions and deployment history for HMM models
    from shadow mode through hybrid to production.

    Attributes:
        id: Primary key
        model_id: Foreign key to HMMModel
        mode: Current deployment mode ('ising_only', 'hmm_shadow', 'hmm_hybrid_20', 'hmm_hybrid_50', 'hmm_hybrid_80', 'hmm_only')
        previous_mode: Previous deployment mode
        transition_date: When the mode transition occurred
        approved_by: Who approved the transition (user ID or 'auto')
        approval_token: Token used for approval
        performance_metrics: JSON with performance metrics at transition
        rollback_count: Number of rollbacks for this deployment
        is_active: Whether this is the current active deployment
        notes: Deployment notes
        created_at: Record creation timestamp
    """
    __tablename__ = 'hmm_deployments'

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('hmm_models.id'), nullable=False, index=True)
    mode = Column(String(20), nullable=False, index=True)
    previous_mode = Column(String(20), nullable=True)
    transition_date = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    approved_by = Column(String(100), nullable=True)
    approval_token = Column(String(64), nullable=True)
    performance_metrics = Column(String, nullable=True)  # JSON string
    rollback_count = Column(Integer, nullable=False, default=0)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationship
    model = relationship("HMMModel", back_populates="deployments")

    __table_args__ = (
        Index('idx_hmm_deployments_mode', 'mode'),
        Index('idx_hmm_deployments_active', 'is_active'),
    )

    def __repr__(self):
        return f"<HMMDeployment(id={self.id}, model_id={self.model_id}, mode={self.mode}, active={self.is_active})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "model_id": self.model_id,
            "mode": self.mode,
            "previous_mode": self.previous_mode,
            "transition_date": self.transition_date.isoformat() if self.transition_date else None,
            "approved_by": self.approved_by,
            "performance_metrics": self.performance_metrics,
            "rollback_count": self.rollback_count,
            "is_active": self.is_active,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class HMMSyncStatus(Base):
    """
    HMM Sync Status for tracking model synchronization between servers.

    Tracks the synchronization state between Contabo (training) and
    Cloudzy (trading) servers for HMM models.

    Attributes:
        id: Primary key
        contabo_version: Current model version on Contabo
        contabo_last_trained: Last training date on Contabo
        cloudzy_version: Current model version on Cloudzy
        cloudzy_last_deployed: Last deployment date on Cloudzy
        version_mismatch: Whether versions are out of sync
        last_sync_attempt: When last sync was attempted
        last_sync_status: Status of last sync ('success', 'failed', 'in_progress')
        sync_progress: Sync progress percentage (0-100)
        sync_message: Current sync status message
        created_at: Record creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'hmm_sync_status'

    id = Column(Integer, primary_key=True, autoincrement=True)
    contabo_version = Column(String(20), nullable=True)
    contabo_last_trained = Column(DateTime, nullable=True)
    cloudzy_version = Column(String(20), nullable=True)
    cloudzy_last_deployed = Column(DateTime, nullable=True)
    version_mismatch = Column(Boolean, nullable=False, default=False, index=True)
    last_sync_attempt = Column(DateTime, nullable=True)
    last_sync_status = Column(String(20), nullable=True)
    sync_progress = Column(Float, nullable=False, default=0.0)
    sync_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<HMMSyncStatus(id={self.id}, contabo={self.contabo_version}, cloudzy={self.cloudzy_version}, mismatch={self.version_mismatch})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "contabo_version": self.contabo_version,
            "contabo_last_trained": self.contabo_last_trained.isoformat() if self.contabo_last_trained else None,
            "cloudzy_version": self.cloudzy_version,
            "cloudzy_last_deployed": self.cloudzy_last_deployed.isoformat() if self.cloudzy_last_deployed else None,
            "version_mismatch": self.version_mismatch,
            "last_sync_attempt": self.last_sync_attempt.isoformat() if self.last_sync_attempt else None,
            "last_sync_status": self.last_sync_status,
            "sync_progress": self.sync_progress,
            "sync_message": self.sync_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
