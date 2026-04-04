"""
QuantMind Broker Accounts Handler

Business logic for broker account operations.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BrokerAccountsAPIHandler:
    """Handler for broker account operations."""

    def __init__(self):
        pass

    def list_broker_accounts(self) -> List[Dict[str, Any]]:
        """List available broker accounts from the broker registry."""
        # Import here to avoid circular imports
        from src.api.broker_endpoints import broker_connections, account_switcher

        all_accounts = list(broker_connections.brokers.values()) + list(broker_connections.pending.values())

        # No mock rows in production paths. Surface an honest empty state instead.
        if not all_accounts:
            return []

        # Convert BrokerInfo objects to dict format expected by frontend
        result = []
        active_id = account_switcher.active_account_id

        for broker in all_accounts:
            result.append({
                "broker_id": broker.id,
                "broker_name": broker.broker_name,
                "account_id": broker.account_id,
                "server": broker.server,
                "account_type": broker.type,
                "balance": broker.balance,
                "equity": broker.equity,
                "margin": broker.margin,
                "leverage": broker.leverage,
                "currency": broker.currency,
                "connected": broker.status == "connected",
                "is_active": broker.account_id == active_id,
                "status": broker.status,
            })

        return result

    def get_broker_account(self, broker_id: str) -> Optional[Dict[str, Any]]:
        """Get broker account details."""
        accounts = self.list_broker_accounts()
        for account in accounts:
            if account.get("broker_id") == broker_id or account.get("account_id") == broker_id:
                return account
        return None
