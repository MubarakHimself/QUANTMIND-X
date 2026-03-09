"""
Configuration Watcher for Agent Factory

Provides hot-reload functionality for agent configuration files.

**Validates: Phase 3.3 - Configuration Hot Reload**
"""

import logging
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

# Try to import watchdog, but make it optional
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = object
    FileModifiedEvent = object


from src.agents.config import AgentConfig

# Factory removed - using department system instead
# from src.agents.factory import get_factory
# from src.agents.registry import get_registry

# Default config directory
DEFAULT_CONFIG_DIR = "config/agents"


class ConfigFileHandler(FileSystemEventHandler):
    """Handler for configuration file changes."""
    
    def __init__(
        self,
        watcher: "ConfigWatcher",
        config_dir: str
    ):
        """
        Initialize the handler.
        
        Args:
            watcher: ConfigWatcher instance
            config_dir: Configuration directory
        """
        self.watcher = watcher
        self.config_dir = Path(config_dir)
        self._last_reload: Dict[str, float] = {}
        self._reload_debounce = 1.0  # seconds
    
    def on_modified(self, event):
        """
        Handle file modification events.
        
        Args:
            event: File system event
        """
        if event.is_directory:
            return
        
        # Check if it's a YAML file
        path = Path(event.src_path)
        if path.suffix not in ('.yaml', '.yml'):
            return
        
        # Debounce reloads
        current_time = time.time()
        last_reload = self._last_reload.get(str(path), 0)
        
        if current_time - last_reload < self._reload_debounce:
            return
        
        self._last_reload[str(path)] = current_time
        
        # Trigger reload
        logger.info(f"Configuration file modified: {path}")
        try:
            self.watcher.reload_config(str(path))
        except Exception as e:
            logger.error(f"Error reloading config {path}: {e}")


class ConfigWatcher:
    """
    Watches configuration files for changes and reloads agents.
    
    Requires watchdog library for file system events.
    """
    
    def __init__(
        self,
        config_dir: str = DEFAULT_CONFIG_DIR,
        registry: Optional[Any] = None,
        factory: Optional[Any] = None,
    ):
        """
        Initialize the config watcher.

        Args:
            config_dir: Directory to watch for config files
            registry: Agent registry
            factory: Deprecated - use department system instead

        Note: Factory-based config watching is deprecated.
        Use floor_manager with department configurations instead.
        """
        self.config_dir = config_dir
        self.registry = None  # Deprecated
        self.factory = None  # Deprecated - use floor_manager

        self._observer: Optional[Any] = None
        self._is_running = False

        logger.info(
            "ConfigWatcher is deprecated. "
            "Use floor_manager /api/floor-manager endpoints instead."
        )
        
        logger.info(f"ConfigWatcher initialized for: {config_dir}")
    
    def start(self) -> bool:
        """
        Start watching configuration files.
        
        Returns:
            True if started successfully, False if watchdog unavailable
            
        Raises:
            RuntimeError: If watcher is already running
        """
        if self._is_running:
            raise RuntimeError("ConfigWatcher is already running")
        
        if not WATCHDOG_AVAILABLE:
            logger.error("Cannot start ConfigWatcher: watchdog not available")
            return False
        
        # Create observer
        self._observer = Observer()
        
        # Create event handler
        handler = ConfigFileHandler(self, self.config_dir)
        
        # Schedule watching
        config_path = Path(self.config_dir)
        if not config_path.exists():
            logger.warning(f"Config directory does not exist: {self.config_dir}")
            # Create it if it doesn't exist
            config_path.mkdir(parents=True, exist_ok=True)
        
        self._observer.schedule(handler, str(config_path), recursive=False)
        self._observer.start()
        
        self._is_running = True
        logger.info(f"ConfigWatcher started watching: {self.config_dir}")
        
        return True
    
    def stop(self) -> None:
        """
        Stop watching configuration files.
        """
        if not self._is_running:
            return
        
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
        
        self._is_running = False
        logger.info("ConfigWatcher stopped")
    
    def reload_config(self, config_path: str) -> bool:
        """
        Reload a configuration file and recreate the agent.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            True if reload successful, False otherwise
        """
        path = Path(config_path)
        
        if not path.exists():
            logger.error(f"Config file not found: {config_path}")
            return False
        
        try:
            # Load new configuration
            config = AgentConfig.from_yaml(str(path))
            
            agent_id = config.agent_id
            
            # Check if agent already exists
            existing_agent = self.registry.get(agent_id)
            
            if existing_agent:
                # Unregister existing agent
                logger.info(f"Unregistering existing agent: {agent_id}")
                self.registry.unregister(agent_id)
            
            # Create new agent
            logger.info(f"Creating new agent from config: {agent_id}")
            logger.warning("Factory-based agent creation is deprecated")
            # agent = self.factory.create(config)  # Deprecated
            
            # Register new agent
            self.registry.register(agent)
            
            logger.info(f"Agent reloaded successfully: {agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload config {config_path}: {e}")
            return False
    
    def reload_all(self) -> Dict[str, bool]:
        """
        Reload all configuration files in the config directory.
        
        Returns:
            Dictionary mapping config paths to reload success status
        """
        results = {}
        
        config_path = Path(self.config_dir)
        if not config_path.exists():
            logger.error(f"Config directory does not exist: {self.config_dir}")
            return results
        
        # Find all YAML files
        for yaml_file in config_path.glob("*.yaml"):
            results[str(yaml_file)] = self.reload_config(str(yaml_file))
        
        for yaml_file in config_path.glob("*.yml"):
            results[str(yaml_file)] = self.reload_config(str(yaml_file))
        
        return results
    
    @property
    def is_running(self) -> bool:
        """Check if the watcher is running."""
        return self._is_running
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Convenience function
def create_config_watcher(
    config_dir: str = DEFAULT_CONFIG_DIR,
) -> ConfigWatcher:
    """
    Create a config watcher.
    
    Args:
        config_dir: Configuration directory to watch
        
    Returns:
        ConfigWatcher instance
    """
    return ConfigWatcher(config_dir=config_dir)
