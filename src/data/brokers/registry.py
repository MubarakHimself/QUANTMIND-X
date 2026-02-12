"""
V8 Unified Broker Registry

Factory pattern for managing multiple broker connections with
automatic adapter selection and connection pooling.

**Validates: Tasks 25.1-25.7, 25.18-25.20**
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Type
from pathlib import Path

from src.integrations.crypto.broker_client import BrokerClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BrokerRegistry:
    """
    Factory for managing multiple broker connections.
    
    Features:
    - YAML configuration loading
    - Automatic adapter selection
    - Connection pooling (single instance per broker_id)
    - Connection status caching
    - Multi-broker audit trail
    
    **Validates: Tasks 25.1-25.7**
    """
    
    # Adapter mapping (populated dynamically)
    ADAPTER_MAP: Dict[str, Type[BrokerClient]] = {}
    
    def __init__(self, config_path: str = "config/brokers.yaml"):
        """
        Initialize broker registry.
        
        Args:
            config_path: Path to YAML configuration file
            
        **Validates: Task 25.3**
        """
        self.config_path = config_path
        self.brokers: Dict[str, BrokerClient] = {}
        self.connection_status: Dict[str, bool] = {}
        self.config: Dict[str, Any] = {}
        
        # Register adapters
        self._register_adapters()
        
        # Load configuration
        if os.path.exists(config_path):
            self.load_config()
            logger.info(f"BrokerRegistry initialized with config: {config_path}")
        else:
            logger.warning(f"Config file not found: {config_path}. Registry initialized empty.")
    
    def _register_adapters(self):
        """
        Register available broker adapters.
        
        Dynamically imports adapters to avoid circular dependencies.
        """
        try:
            from .mock_mt5_adapter import MockMT5Adapter
            self.ADAPTER_MAP['mt5_mock'] = MockMT5Adapter
            logger.debug("Registered MockMT5Adapter")
        except ImportError as e:
            logger.warning(f"Could not register MockMT5Adapter: {e}")
        
        try:
            from .mt5_socket_adapter import MT5SocketAdapter
            self.ADAPTER_MAP['mt5_socket'] = MT5SocketAdapter
            self.ADAPTER_MAP['mt5'] = MT5SocketAdapter  # Alias
            logger.debug("Registered MT5SocketAdapter")
        except ImportError as e:
            logger.warning(f"Could not register MT5SocketAdapter: {e}")
        
        try:
            from .binance_adapter import BinanceSpotAdapter, BinanceFuturesAdapter
            self.ADAPTER_MAP['binance_spot'] = BinanceSpotAdapter
            self.ADAPTER_MAP['binance_futures'] = BinanceFuturesAdapter
            logger.debug("Registered Binance adapters")
        except ImportError as e:
            logger.warning(f"Could not register Binance adapters: {e}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load broker configuration from YAML file.
        
        Returns:
            Parsed configuration dictionary
            
        **Validates: Task 25.4**
        """
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Validate configuration
            if not self.config or 'brokers' not in self.config:
                raise ValueError("Invalid config: missing 'brokers' section")
            
            # Substitute environment variables
            self._substitute_env_vars(self.config)
            
            # Validate each broker config
            self._validate_config()
            
            logger.info(f"Loaded {len(self.config['brokers'])} broker configurations")
            return self.config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _substitute_env_vars(self, config: Dict[str, Any]):
        """
        Substitute environment variables in configuration.
        
        Replaces ${VAR_NAME} with environment variable values.
        
        Args:
            config: Configuration dictionary (modified in-place)
        """
        def substitute(value):
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                var_name = value[2:-1]
                return os.getenv(var_name, value)
            elif isinstance(value, dict):
                return {k: substitute(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute(item) for item in value]
            return value
        
        for broker_id, broker_config in config.get('brokers', {}).items():
            config['brokers'][broker_id] = substitute(broker_config)
    
    def _validate_config(self):
        """
        Validate broker configuration.
        
        Ensures all required fields are present and valid.
        
        **Validates: Task 25.18**
        """
        for broker_id, broker_config in self.config.get('brokers', {}).items():
            # Check required fields
            if 'type' not in broker_config:
                raise ValueError(f"Broker {broker_id}: missing 'type' field")
            
            broker_type = broker_config['type']
            
            # Check if adapter exists
            if broker_type not in self.ADAPTER_MAP:
                raise ValueError(
                    f"Broker {broker_id}: unknown type '{broker_type}'. "
                    f"Available: {list(self.ADAPTER_MAP.keys())}"
                )
            
            # Check enabled flag
            if 'enabled' not in broker_config:
                broker_config['enabled'] = True  # Default to enabled
            
            logger.debug(f"Validated broker config: {broker_id} ({broker_type})")
    
    def register_broker(self, broker_id: str, broker_config: Optional[Dict[str, Any]] = None):
        """
        Register and instantiate a broker adapter.
        
        Args:
            broker_id: Unique broker identifier
            broker_config: Broker configuration (if None, loads from config)
            
        **Validates: Task 25.5**
        """
        # Get config
        if broker_config is None:
            if broker_id not in self.config.get('brokers', {}):
                raise ValueError(f"Broker {broker_id} not found in configuration")
            broker_config = self.config['brokers'][broker_id]
        
        # Check if enabled
        if not broker_config.get('enabled', True):
            logger.info(f"Broker {broker_id} is disabled, skipping registration")
            return
        
        # Check if already registered
        if broker_id in self.brokers:
            logger.warning(f"Broker {broker_id} already registered, skipping")
            return
        
        # Get adapter class
        broker_type = broker_config['type']
        adapter_class = self.ADAPTER_MAP.get(broker_type)
        
        if adapter_class is None:
            raise ValueError(f"No adapter found for type: {broker_type}")
        
        # Instantiate adapter
        try:
            adapter = adapter_class(broker_config)
            self.brokers[broker_id] = adapter
            
            # Cache connection status
            self.connection_status[broker_id] = adapter.validate_connection()
            
            logger.info(
                f"Registered broker: {broker_id} ({broker_type}) - "
                f"Connected: {self.connection_status[broker_id]}"
            )
            
        except Exception as e:
            logger.error(f"Failed to register broker {broker_id}: {e}")
            raise
    
    def get_broker(self, broker_id: str) -> BrokerClient:
        """
        Get registered broker instance.
        
        Lazy-loads broker if not already registered.
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            Broker adapter instance
            
        Raises:
            ValueError: If broker not found or registration fails
            
        **Validates: Task 25.6**
        """
        # Check if already registered
        if broker_id in self.brokers:
            return self.brokers[broker_id]
        
        # Try to register from config
        if broker_id in self.config.get('brokers', {}):
            self.register_broker(broker_id)
            return self.brokers[broker_id]
        
        # Broker not found
        raise ValueError(
            f"Broker {broker_id} not found. "
            f"Available: {list(self.config.get('brokers', {}).keys())}"
        )
    
    def list_brokers(self) -> List[str]:
        """
        List all configured broker IDs.
        
        Returns:
            List of broker identifiers
            
        **Validates: Task 25.7**
        """
        return list(self.config.get('brokers', {}).keys())
    
    def list_registered_brokers(self) -> List[str]:
        """
        List currently registered (instantiated) broker IDs.
        
        Returns:
            List of registered broker identifiers
        """
        return list(self.brokers.keys())
    
    def validate_all_connections(self) -> Dict[str, bool]:
        """
        Validate all registered broker connections.
        
        Returns:
            Dictionary mapping broker_id to connection status
            
        **Validates: Task 25.19**
        """
        results = {}
        
        for broker_id, broker in self.brokers.items():
            try:
                is_connected = broker.validate_connection()
                results[broker_id] = is_connected
                self.connection_status[broker_id] = is_connected
                
                logger.info(f"Broker {broker_id}: {'Connected' if is_connected else 'Disconnected'}")
                
            except Exception as e:
                logger.error(f"Broker {broker_id} validation failed: {e}")
                results[broker_id] = False
                self.connection_status[broker_id] = False
        
        return results
    
    def get_connection_status(self, broker_id: str) -> bool:
        """
        Get cached connection status for broker.
        
        Args:
            broker_id: Broker identifier
            
        Returns:
            True if connected, False otherwise
        """
        return self.connection_status.get(broker_id, False)
    
    def register_all(self):
        """
        Register all enabled brokers from configuration.
        
        Useful for initializing all brokers at startup.
        """
        for broker_id in self.list_brokers():
            try:
                self.register_broker(broker_id)
            except Exception as e:
                logger.error(f"Failed to register broker {broker_id}: {e}")
    
    def unregister_broker(self, broker_id: str):
        """
        Unregister a broker and clean up resources.
        
        Args:
            broker_id: Broker identifier
        """
        if broker_id in self.brokers:
            del self.brokers[broker_id]
            if broker_id in self.connection_status:
                del self.connection_status[broker_id]
            logger.info(f"Unregistered broker: {broker_id}")
    
    def reload_config(self):
        """
        Reload configuration from file.
        
        Useful for updating configuration without restarting.
        """
        logger.info("Reloading broker configuration...")
        self.load_config()
        logger.info("Configuration reloaded successfully")
    
    def __repr__(self):
        return (
            f"<BrokerRegistry("
            f"configured={len(self.list_brokers())}, "
            f"registered={len(self.list_registered_brokers())}"
            f")>"
        )


