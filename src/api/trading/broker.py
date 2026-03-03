"""
Broker API Handlers

Handles broker registry and connection API endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger(__name__)


class BrokerRegistryAPIHandler:
    """
    Handles broker registry API endpoints.

    **Validates: Task Group 7.5 - Broker Registry API endpoints**

    Integrates with:
    - src/router/broker_registry.py for broker management
    - src/database/models.py for BrokerRegistry table
    """

    def __init__(self):
        """Initialize broker registry handler."""
        from src.router.broker_registry import BrokerRegistryManager
        self._broker_manager = BrokerRegistryManager()

    def list_brokers(self) -> 'BrokersListResponse':
        """
        List all registered broker profiles.

        Returns:
            BrokersListResponse with all broker profiles
        """
        from .models import BrokersListResponse

        try:
            brokers = self._broker_manager.list_all_brokers()
            return BrokersListResponse(
                brokers=brokers,
                count=len(brokers)
            )
        except Exception as e:
            logger.error(f"Error listing brokers: {e}")
            return BrokersListResponse(
                brokers=[],
                count=0
            )

    def create_broker(self, request) -> 'BrokerResponse':
        """
        Create a new broker profile.

        Args:
            request: Broker creation request

        Returns:
            BrokerResponse with created profile
        """
        from .models import BrokerResponse

        try:
            broker = self._broker_manager.create_broker(
                broker_id=request.broker_id,
                broker_name=request.broker_name,
                spread_avg=request.spread_avg,
                commission_per_lot=request.commission_per_lot,
                lot_step=request.lot_step,
                min_lot=request.min_lot,
                max_lot=request.max_lot,
                pip_values=request.pip_values,
                preference_tags=request.preference_tags
            )

            return BrokerResponse(
                id=broker.id,
                broker_id=broker.broker_id,
                broker_name=broker.broker_name,
                spread_avg=broker.spread_avg,
                commission_per_lot=broker.commission_per_lot,
                lot_step=broker.lot_step,
                min_lot=broker.min_lot,
                max_lot=broker.max_lot,
                pip_values=broker.pip_values,
                preference_tags=broker.preference_tags
            )

        except Exception as e:
            logger.error(f"Error creating broker: {e}")
            raise


class BrokerConnectionHandler:
    """
    Handles broker connection API endpoints with actual MT5 integration.

    **Phase 5: Broker Connection with MT5 Integration**

    Integrates with:
    - MetaTrader5 Python package for terminal connection
    """

    def __init__(self):
        """Initialize broker connection handler."""
        self._connected = False
        self._mt5 = None

    def connect_broker(self, request) -> 'BrokerConnectResponse':
        """
        Connect to MT5 broker with actual MT5 integration.

        Args:
            request: Broker connection request

        Returns:
            BrokerConnectResponse with connection result
        """
        from .models import BrokerConnectResponse

        try:
            # Try to import MetaTrader5 package
            try:
                import MetaTrader5 as mt5
                self._mt5 = mt5
            except ImportError:
                return BrokerConnectResponse(
                    success=False,
                    broker_id=request.broker_id,
                    login=request.login,
                    server=request.server,
                    connected=False,
                    account_info=None,
                    error="MetaTrader5 package not installed. Install with: pip install MetaTrader5"
                )

            # Initialize MT5
            if not mt5.initialize():
                error_code = mt5.last_error()
                return BrokerConnectResponse(
                    success=False,
                    broker_id=request.broker_id,
                    login=request.login,
                    server=request.server,
                    connected=False,
                    account_info=None,
                    error=f"MT5 initialize failed: {error_code}"
                )

            # Login to account
            if not mt5.login(request.login, request.password, request.server):
                error_code = mt5.last_error()
                mt5.shutdown()
                return BrokerConnectResponse(
                    success=False,
                    broker_id=request.broker_id,
                    login=request.login,
                    server=request.server,
                    connected=False,
                    account_info=None,
                    error=f"MT5 login failed: {error_code}"
                )

            # Get account information
            account_info_dict = mt5.account_info()._asdict()

            # Get terminal info
            terminal_info = mt5.terminal_info()._asdict() if mt5.terminal_info() else {}

            # Shutdown MT5
            mt5.shutdown()

            self._connected = True

            logger.info(f"Connected to broker {request.broker_id}: {request.login}@{request.server}")

            return BrokerConnectResponse(
                success=True,
                broker_id=request.broker_id,
                login=request.login,
                server=request.server,
                connected=True,
                account_info={
                    "login": account_info_dict.get("login"),
                    "server": account_info_dict.get("server"),
                    "balance": account_info_dict.get("balance"),
                    "equity": account_info_dict.get("equity"),
                    "margin": account_info_dict.get("margin"),
                    "free_margin": account_info_dict.get("margin_free"),
                    "margin_level": account_info_dict.get("margin_level"),
                    "currency": account_info_dict.get("currency"),
                    "company": account_info_dict.get("company"),
                    "name": account_info_dict.get("name"),
                }
            )

        except Exception as e:
            logger.error(f"Broker connection failed: {e}")
            return BrokerConnectResponse(
                success=False,
                broker_id=request.broker_id,
                login=request.login,
                server=request.server,
                connected=False,
                account_info=None,
                error=str(e)
            )

    def disconnect_broker(self) -> Dict[str, Any]:
        """Disconnect from MT5 broker."""
        try:
            if self._mt5 and self._connected:
                self._mt5.shutdown()
                self._connected = False

            return {
                "success": True,
                "message": "Disconnected from broker",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Disconnect failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
