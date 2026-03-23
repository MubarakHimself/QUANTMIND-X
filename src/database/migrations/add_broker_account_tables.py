"""
Migration for adding broker_accounts and routing_rules tables.

Story 9.1: Broker Account Registry & Routing Matrix API
"""

import logging
from sqlalchemy import text

logger = logging.getLogger(__name__)


def upgrade(engine):
    """Create broker_accounts and routing_rules tables."""
    from src.database.models.broker_account import BrokerAccount, RoutingRule
    from src.database.models.base import Base

    # Create tables
    Base.metadata.create_all(engine)
    logger.info("Created broker_accounts and routing_rules tables")


def downgrade(engine):
    """Drop broker_accounts and routing_rules tables."""
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS routing_rules"))
        conn.execute(text("DROP TABLE IF EXISTS broker_accounts"))
        conn.commit()
    logger.info("Dropped broker_accounts and routing_rules tables")