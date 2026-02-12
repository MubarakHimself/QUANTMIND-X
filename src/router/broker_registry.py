"""
Broker Registry Manager

Manages broker profiles with fee structures and trading parameters.
Used for fee-aware position sizing and dynamic pip value calculation.

**Validates: Task Group 7.2 - BrokerRegistry table and manager**
"""

import logging
from typing import Optional, Dict, List, Any
from datetime import datetime

from src.database.manager import DatabaseManager
from src.database.models import BrokerRegistry

logger = logging.getLogger(__name__)


class BrokerRegistryManager:
    """
    Manager for Broker Registry operations.

    Provides methods for:
    - Creating and updating broker profiles
    - Retrieving pip values per symbol/broker
    - Getting commission structures
    - Finding brokers by preference tags

    Usage:
        manager = BrokerRegistryManager()
        pip_value = manager.get_pip_value("XAUUSD", "icmarkets_raw")
        commission = manager.get_commission("icmarkets_raw")
    """

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize BrokerRegistry manager.

        Args:
            db_manager: Optional DatabaseManager instance (creates singleton if not provided)
        """
        self.db = db_manager or DatabaseManager()

    def get_broker(self, broker_id: str) -> Optional[BrokerRegistry]:
        """
        Retrieve a broker profile by broker ID.

        Args:
            broker_id: Unique broker identifier (e.g., "icmarkets_raw")

        Returns:
            BrokerRegistry object or None if not found
        """
        with self.db.get_session() as session:
            broker = session.query(BrokerRegistry).filter(
                BrokerRegistry.broker_id == broker_id
            ).first()

            if broker is not None:
                session.expunge(broker)
            return broker

    def create_broker(
        self,
        broker_id: str,
        broker_name: str,
        spread_avg: float = 0.0,
        commission_per_lot: float = 0.0,
        lot_step: float = 0.01,
        min_lot: float = 0.01,
        max_lot: float = 100.0,
        pip_values: Optional[Dict[str, float]] = None,
        preference_tags: Optional[List[str]] = None
    ) -> BrokerRegistry:
        """
        Create a new broker profile.

        Args:
            broker_id: Unique broker identifier
            broker_name: Human-readable broker name
            spread_avg: Average spread in points
            commission_per_lot: Commission per standard lot
            lot_step: Minimum lot step increment
            min_lot: Minimum lot size
            max_lot: Maximum lot size
            pip_values: Dict mapping symbols to pip values
            preference_tags: List of broker tags

        Returns:
            Created BrokerRegistry object
        """
        if pip_values is None:
            # Default pip values for common symbols
            pip_values = {
                "EURUSD": 10.0,
                "GBPUSD": 10.0,
                "USDJPY": 9.09,
                "XAUUSD": 1.0,
                "XAGUSD": 50.0,
            }

        if preference_tags is None:
            preference_tags = ["STANDARD"]

        with self.db.get_session() as session:
            broker = BrokerRegistry(
                broker_id=broker_id,
                broker_name=broker_name,
                spread_avg=spread_avg,
                commission_per_lot=commission_per_lot,
                lot_step=lot_step,
                min_lot=min_lot,
                max_lot=max_lot,
                pip_values=pip_values,
                preference_tags=preference_tags
            )
            session.add(broker)
            session.flush()
            session.refresh(broker)
            session.expunge(broker)
            return broker

    def update_broker(
        self,
        broker_id: str,
        **kwargs
    ) -> Optional[BrokerRegistry]:
        """
        Update an existing broker profile.

        Args:
            broker_id: Broker to update
            **kwargs: Fields to update (spread_avg, commission_per_lot, etc.)

        Returns:
            Updated BrokerRegistry object or None if not found
        """
        with self.db.get_session() as session:
            broker = session.query(BrokerRegistry).filter(
                BrokerRegistry.broker_id == broker_id
            ).first()

            if broker is None:
                return None

            # Update provided fields
            for key, value in kwargs.items():
                if hasattr(broker, key):
                    setattr(broker, key, value)

            broker.updated_at = datetime.utcnow()
            session.flush()
            session.refresh(broker)
            session.expunge(broker)
            return broker

    def get_pip_value(self, symbol: str, broker_id: str) -> float:
        """
        Get pip value for a specific symbol from broker registry.

        Used for dynamic position sizing calculation.

        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            broker_id: Broker identifier

        Returns:
            Pip value for the symbol (e.g., 1.0 for XAUUSD, 10.0 for EURUSD)
            Returns 10.0 as default if symbol not found
        """
        broker = self.get_broker(broker_id)

        if broker is None:
            logger.warning(f"Broker {broker_id} not found, using default pip value")
            return 10.0

        # Look up symbol in pip_values dict
        pip_value = broker.pip_values.get(symbol.upper())

        if pip_value is None:
            logger.warning(f"Pip value for {symbol} not found in {broker_id}, using default")
            return 10.0

        return pip_value

    def get_commission(self, broker_id: str) -> float:
        """
        Get commission per lot for a broker.

        Used for fee-aware position sizing.

        Args:
            broker_id: Broker identifier

        Returns:
            Commission per standard lot
        """
        broker = self.get_broker(broker_id)

        if broker is None:
            logger.warning(f"Broker {broker_id} not found, assuming no commission")
            return 0.0

        return broker.commission_per_lot

    def get_spread(self, broker_id: str) -> float:
        """
        Get average spread for a broker.

        Used for fee-aware position sizing.

        Args:
            broker_id: Broker identifier

        Returns:
            Average spread in points
        """
        broker = self.get_broker(broker_id)

        if broker is None:
            logger.warning(f"Broker {broker_id} not found, assuming zero spread")
            return 0.0

        return broker.spread_avg

    def get_lot_step(self, broker_id: str) -> float:
        """
        Get lot step increment for a broker.

        Used for validating lot sizes.

        Args:
            broker_id: Broker identifier

        Returns:
            Lot step increment (e.g., 0.01)
        """
        broker = self.get_broker(broker_id)

        if broker is None:
            return 0.01  # Default lot step

        return broker.lot_step

    def get_lot_limits(self, broker_id: str) -> tuple[float, float]:
        """
        Get minimum and maximum lot sizes for a broker.

        Args:
            broker_id: Broker identifier

        Returns:
            Tuple of (min_lot, max_lot)
        """
        broker = self.get_broker(broker_id)

        if broker is None:
            return (0.01, 100.0)  # Default limits

        return (broker.min_lot, broker.max_lot)

    def find_brokers_by_tag(self, tag: str) -> List[BrokerRegistry]:
        """
        Find brokers with a specific preference tag.

        Args:
            tag: Preference tag to search for (e.g., "RAW_ECN")

        Returns:
            List of BrokerRegistry objects with the tag
        """
        with self.db.get_session() as session:
            brokers = session.query(BrokerRegistry).filter(
                BrokerRegistry.preference_tags.contains(tag)
            ).all()

            # Detach from session
            for broker in brokers:
                session.expunge(broker)

            return brokers

    def list_all_brokers(self) -> List[Dict[str, Any]]:
        """
        List all registered brokers.

        Returns:
            List of dictionaries with broker info
        """
        with self.db.get_session() as session:
            brokers = session.query(BrokerRegistry).all()

            return [
                {
                    "id": b.id,
                    "broker_id": b.broker_id,
                    "broker_name": b.broker_name,
                    "spread_avg": b.spread_avg,
                    "commission_per_lot": b.commission_per_lot,
                    "lot_step": b.lot_step,
                    "min_lot": b.min_lot,
                    "max_lot": b.max_lot,
                    "preference_tags": b.preference_tags,
                }
                for b in brokers
            ]

    def delete_broker(self, broker_id: str) -> bool:
        """
        Delete a broker profile.

        Args:
            broker_id: Broker to delete

        Returns:
            True if deleted, False if not found
        """
        with self.db.get_session() as session:
            broker = session.query(BrokerRegistry).filter(
                BrokerRegistry.broker_id == broker_id
            ).first()

            if broker is None:
                return False

            session.delete(broker)
            return True
