"""
Broker Repository.

Provides database operations for broker registry.
"""

from typing import Optional, List
from src.database.repositories.base_repository import BaseRepository
from src.database.models import BrokerRegistry


class BrokerRepository(BaseRepository[BrokerRegistry]):
    """Repository for BrokerRegistry database operations."""

    model = BrokerRegistry

    def get_by_broker_id(self, broker_id: str) -> Optional[BrokerRegistry]:
        """Get a broker by broker ID."""
        with self.get_session() as session:
            broker = session.query(BrokerRegistry).filter(
                BrokerRegistry.broker_id == broker_id
            ).first()
            if broker is not None:
                session.expunge(broker)
            return broker

    def create(
        self,
        broker_id: str,
        broker_name: str,
        spread_avg: float = 0.0,
        commission_per_lot: float = 0.0,
        lot_step: float = 0.01,
        min_lot: float = 0.01,
        max_lot: float = 100.0,
        pip_values: dict = None,
        preference_tags: list = None
    ) -> BrokerRegistry:
        """Create a new broker registry entry."""
        return super().create(
            broker_id=broker_id,
            broker_name=broker_name,
            spread_avg=spread_avg,
            commission_per_lot=commission_per_lot,
            lot_step=lot_step,
            min_lot=min_lot,
            max_lot=max_lot,
            pip_values=pip_values or {},
            preference_tags=preference_tags or []
        )

    def get_all_brokers(self) -> List[BrokerRegistry]:
        """Get all registered brokers."""
        return self.get_all()

    def get_by_tag(self, tag: str) -> List[BrokerRegistry]:
        """Get brokers by preference tag."""
        with self.get_session() as session:
            brokers = session.query(BrokerRegistry).filter(
                BrokerRegistry.preference_tags.contains(tag)
            ).all()
            for broker in brokers:
                session.expunge(broker)
            return brokers
