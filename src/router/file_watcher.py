"""
File Watcher for Risk Matrix Changes

Monitors risk_matrix.json for changes and triggers callbacks.
Uses watchdog library for cross-platform file system monitoring.
"""

import os
import logging
import time
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from threading import Thread, Event
import json

logger = logging.getLogger(__name__)


class RiskMatrixWatcher:
    """
    File system watcher for risk_matrix.json changes.
    
    Monitors the risk matrix file and triggers callbacks when changes are detected.
    Uses polling-based approach for maximum compatibility.
    """
    
    def __init__(
        self,
        file_path: str,
        callback: Callable[[Dict[str, Any]], None],
        poll_interval: float = 1.0
    ):
        """
        Initialize the file watcher.
        
        Args:
            file_path: Path to risk_matrix.json file
            callback: Function to call when file changes (receives parsed JSON)
            poll_interval: Polling interval in seconds (default: 1.0)
        """
        self.file_path = Path(file_path)
        self.callback = callback
        self.poll_interval = poll_interval
        
        self._stop_event = Event()
        self._watch_thread: Optional[Thread] = None
        self._last_mtime: Optional[float] = None
        self._last_content: Optional[str] = None
        
        logger.info(f"RiskMatrixWatcher initialized for {self.file_path}")
    
    def start(self):
        """Start watching the file for changes."""
        if self._watch_thread and self._watch_thread.is_alive():
            logger.warning("Watcher already running")
            return
        
        self._stop_event.clear()
        self._watch_thread = Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        logger.info("File watcher started")
    
    def stop(self):
        """Stop watching the file."""
        if not self._watch_thread or not self._watch_thread.is_alive():
            logger.warning("Watcher not running")
            return
        
        self._stop_event.set()
        self._watch_thread.join(timeout=5.0)
        logger.info("File watcher stopped")
    
    def _watch_loop(self):
        """Main watch loop running in separate thread."""
        while not self._stop_event.is_set():
            try:
                self._check_file()
            except Exception as e:
                logger.error(f"Error in watch loop: {e}")
            
            # Sleep with interruptible wait
            self._stop_event.wait(self.poll_interval)
    
    def _check_file(self):
        """Check if file has changed and trigger callback if needed."""
        if not self.file_path.exists():
            # File doesn't exist yet, skip
            return
        
        try:
            # Get file modification time
            current_mtime = self.file_path.stat().st_mtime
            
            # Check if file has been modified
            if self._last_mtime is not None and current_mtime == self._last_mtime:
                # No change
                return
            
            # Read file content
            with open(self.file_path, 'r') as f:
                content = f.read()
            
            # Check if content actually changed (avoid spurious mtime updates)
            if content == self._last_content:
                # Content unchanged, just update mtime
                self._last_mtime = current_mtime
                return
            
            # Parse JSON
            try:
                risk_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse risk matrix JSON: {e}")
                return
            
            # Update tracking
            self._last_mtime = current_mtime
            self._last_content = content
            
            # Trigger callback
            logger.debug(f"Risk matrix changed, triggering callback")
            self.callback(risk_data)
            
        except Exception as e:
            logger.error(f"Error checking file: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class WatchdogRiskMatrixWatcher:
    """
    Alternative implementation using watchdog library for event-driven monitoring.
    
    More efficient than polling but requires watchdog package.
    """
    
    def __init__(
        self,
        file_path: str,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """
        Initialize the watchdog-based file watcher.
        
        Args:
            file_path: Path to risk_matrix.json file
            callback: Function to call when file changes (receives parsed JSON)
        """
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            
            self.file_path = Path(file_path)
            self.callback = callback
            self._observer = Observer()
            self._handler = self._create_handler()
            
            logger.info(f"WatchdogRiskMatrixWatcher initialized for {self.file_path}")
            
        except ImportError:
            raise ImportError(
                "watchdog package not installed. "
                "Install with: pip install watchdog"
            )
    
    def _create_handler(self):
        """Create file system event handler."""
        from watchdog.events import FileSystemEventHandler
        
        class RiskMatrixHandler(FileSystemEventHandler):
            def __init__(self, watcher):
                self.watcher = watcher
                self._last_content = None
            
            def on_modified(self, event):
                if event.src_path == str(self.watcher.file_path):
                    self.watcher._on_file_changed()
        
        return RiskMatrixHandler(self)
    
    def _on_file_changed(self):
        """Handle file change event."""
        try:
            # Read and parse file
            with open(self.file_path, 'r') as f:
                content = f.read()
            
            risk_data = json.loads(content)
            
            # Trigger callback
            logger.debug(f"Risk matrix changed (watchdog), triggering callback")
            self.callback(risk_data)
            
        except Exception as e:
            logger.error(f"Error handling file change: {e}")
    
    def start(self):
        """Start watching the file."""
        watch_dir = self.file_path.parent
        self._observer.schedule(self._handler, str(watch_dir), recursive=False)
        self._observer.start()
        logger.info("Watchdog file watcher started")
    
    def stop(self):
        """Stop watching the file."""
        self._observer.stop()
        self._observer.join(timeout=5.0)
        logger.info("Watchdog file watcher stopped")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
