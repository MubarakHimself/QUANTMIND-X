"""
HMM Repository.

Provides database operations for HMM models (HMMModel, HMMShadowLog, HMMDeployment, HMMSyncStatus).
"""

from typing import Optional, List
from datetime import datetime
from src.database.repositories.base_repository import BaseRepository
from src.database.models import HMMModel, HMMShadowLog, HMMDeployment, HMMSyncStatus


class HMMModelRepository(BaseRepository[HMMModel]):
    """Repository for HMMModel database operations."""

    model = HMMModel

    def get_by_version(self, version: str, limit: int = 100) -> List[HMMModel]:
        """Get HMM models by version."""
        return self.filter_by(limit=limit, version=version)

    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[HMMModel]:
        """Get HMM models by symbol."""
        with self.get_session() as session:
            models = session.query(HMMModel).filter(
                HMMModel.symbol == symbol
            ).order_by(HMMModel.created_at.desc()).limit(limit).all()
            for model in models:
                session.expunge(model)
            return models

    def get_by_type(self, model_type: str, limit: int = 100) -> List[HMMModel]:
        """Get HMM models by type."""
        return self.filter_by(limit=limit, model_type=model_type)

    def get_active_models(self, limit: int = 100) -> List[HMMModel]:
        """Get all active HMM models."""
        with self.get_session() as session:
            models = session.query(HMMModel).filter(
                HMMModel.is_active == True
            ).order_by(HMMModel.created_at.desc()).limit(limit).all()
            for model in models:
                session.expunge(model)
            return models

    def get_validated_models(self, limit: int = 100) -> List[HMMModel]:
        """Get validated HMM models."""
        with self.get_session() as session:
            models = session.query(HMMModel).filter(
                HMMModel.validation_status == 'validated'
            ).order_by(HMMModel.created_at.desc()).limit(limit).all()
            for model in models:
                session.expunge(model)
            return models


class HMMShadowLogRepository(BaseRepository[HMMShadowLog]):
    """Repository for HMMShadowLog database operations."""

    model = HMMShadowLog

    def get_by_model_id(self, model_id: int, limit: int = 100) -> List[HMMShadowLog]:
        """Get shadow logs by model ID."""
        with self.get_session() as session:
            logs = session.query(HMMShadowLog).filter(
                HMMShadowLog.model_id == model_id
            ).order_by(HMMShadowLog.timestamp.desc()).limit(limit).all()
            for log in logs:
                session.expunge(log)
            return logs

    def get_by_symbol(self, symbol: str, limit: int = 100) -> List[HMMShadowLog]:
        """Get shadow logs by symbol."""
        with self.get_session() as session:
            logs = session.query(HMMShadowLog).filter(
                HMMShadowLog.symbol == symbol
            ).order_by(HMMShadowLog.timestamp.desc()).limit(limit).all()
            for log in logs:
                session.expunge(log)
            return logs

    def get_by_timeframe(self, timeframe: str, limit: int = 100) -> List[HMMShadowLog]:
        """Get shadow logs by timeframe."""
        return self.filter_by(limit=limit, timeframe=timeframe)

    def get_agreements(self, limit: int = 100) -> List[HMMShadowLog]:
        """Get logs where Ising and HMM agreed."""
        with self.get_session() as session:
            logs = session.query(HMMShadowLog).filter(
                HMMShadowLog.agreement == True
            ).order_by(HMMShadowLog.timestamp.desc()).limit(limit).all()
            for log in logs:
                session.expunge(log)
            return logs

    def get_disagreements(self, limit: int = 100) -> List[HMMShadowLog]:
        """Get logs where Ising and HMM disagreed."""
        with self.get_session() as session:
            logs = session.query(HMMShadowLog).filter(
                HMMShadowLog.agreement == False
            ).order_by(HMMShadowLog.timestamp.desc()).limit(limit).all()
            for log in logs:
                session.expunge(log)
            return logs


class HMMDeploymentRepository(BaseRepository[HMMDeployment]):
    """Repository for HMMDeployment database operations."""

    model = HMMDeployment

    def get_by_model_id(self, model_id: int, limit: int = 100) -> List[HMMDeployment]:
        """Get deployments by model ID."""
        with self.get_session() as session:
            deployments = session.query(HMMDeployment).filter(
                HMMDeployment.model_id == model_id
            ).order_by(HMMDeployment.transition_date.desc()).limit(limit).all()
            for deployment in deployments:
                session.expunge(deployment)
            return deployments

    def get_active_deployment(self) -> Optional[HMMDeployment]:
        """Get the currently active deployment."""
        with self.get_session() as session:
            deployment = session.query(HMMDeployment).filter(
                HMMDeployment.is_active == True
            ).first()
            if deployment is not None:
                session.expunge(deployment)
            return deployment

    def get_by_mode(self, mode: str, limit: int = 100) -> List[HMMDeployment]:
        """Get deployments by mode."""
        return self.filter_by(limit=limit, mode=mode)


class HMMSyncStatusRepository(BaseRepository[HMMSyncStatus]):
    """Repository for HMMSyncStatus database operations."""

    model = HMMSyncStatus

    def get_mismatch_status(self) -> List[HMMSyncStatus]:
        """Get sync status records with version mismatch."""
        with self.get_session() as session:
            records = session.query(HMMSyncStatus).filter(
                HMMSyncStatus.version_mismatch == True
            ).order_by(HMMSyncStatus.updated_at.desc()).all()
            for record in records:
                session.expunge(record)
            return records

    def get_latest(self) -> Optional[HMMSyncStatus]:
        """Get the most recent sync status."""
        with self.get_session() as session:
            record = session.query(HMMSyncStatus).order_by(
                HMMSyncStatus.updated_at.desc()
            ).first()
            if record is not None:
                session.expunge(record)
            return record
