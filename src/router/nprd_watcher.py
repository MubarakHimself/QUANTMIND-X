"""
NPRD File Watcher for Automatic Workflow Triggering.

This module implements a file watcher that monitors the NPRD output directory
and automatically triggers the workflow orchestrator when new NPRD JSON files are created.

Wired to submit real tasks to the WorkflowOrchestrator for processing.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

logger = logging.getLogger(__name__)


@dataclass
class NPRDWatchEvent:
    """Event data for NPRD file watch."""
    event_type: str  # "created", "modified"
    file_path: Path
    timestamp: str
    nprd_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    workflow_id: Optional[str] = None


class NPRDFileHandler(FileSystemEventHandler):
    """
    File system event handler for NPRD files.
    
    Handles creation and modification of NPRD JSON files in the watched directory.
    """
    
    def __init__(
        self,
        callback: Callable[[NPRDWatchEvent], None],
        file_pattern: str = "*.json"
    ):
        """
        Initialize NPRD file handler.
        
        Args:
            callback: Callback function to invoke on file events
            file_pattern: Glob pattern for files to watch (default: "*.json")
        """
        super().__init__()
        self.callback = callback
        self.file_pattern = file_pattern
        self._processed_files: set = set()
    
    def on_created(self, event: FileCreatedEvent) -> None:
        """Handle file creation event."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if file matches pattern
        if not file_path.match(self.file_pattern):
            return
        
        logger.info(f"New NPRD file detected: {file_path}")
        
        # Create watch event
        watch_event = NPRDWatchEvent(
            event_type="created",
            file_path=file_path,
            timestamp=datetime.now().isoformat()
        )
        
        # Try to load NPRD data
        try:
            # Wait a moment for file to be fully written
            import time
            time.sleep(0.5)
            
            with open(file_path, 'r') as f:
                watch_event.nprd_data = json.load(f)
            
            # Mark as processed
            self._processed_files.add(str(file_path))
            
        except json.JSONDecodeError as e:
            watch_event.error = f"Invalid JSON: {e}"
            logger.error(f"Failed to parse NPRD file {file_path}: {e}")
            
        except Exception as e:
            watch_event.error = str(e)
            logger.error(f"Error reading NPRD file {file_path}: {e}")
        
        # Invoke callback
        try:
            self.callback(watch_event)
        except Exception as e:
            logger.error(f"Callback error for {file_path}: {e}")
    
    def on_modified(self, event: FileModifiedEvent) -> None:
        """Handle file modification event."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Check if file matches pattern
        if not file_path.match(self.file_pattern):
            return
        
        # Skip if already processed (avoid duplicate triggers)
        if str(file_path) in self._processed_files:
            return
        
        logger.info(f"NPRD file modified: {file_path}")
        
        # Create watch event
        watch_event = NPRDWatchEvent(
            event_type="modified",
            file_path=file_path,
            timestamp=datetime.now().isoformat()
        )
        
        # Try to load NPRD data
        try:
            with open(file_path, 'r') as f:
                watch_event.nprd_data = json.load(f)
            
            # Mark as processed
            self._processed_files.add(str(file_path))
            
        except json.JSONDecodeError as e:
            watch_event.error = f"Invalid JSON: {e}"
            logger.error(f"Failed to parse NPRD file {file_path}: {e}")
            
        except Exception as e:
            watch_event.error = str(e)
            logger.error(f"Error reading NPRD file {file_path}: {e}")
        
        # Invoke callback
        try:
            self.callback(watch_event)
        except Exception as e:
            logger.error(f"Callback error for {file_path}: {e}")


class NPRDFileWatcher:
    """
    File watcher for NPRD output directory.
    
    Monitors the NPRD output directory and triggers the workflow orchestrator
    when new NPRD JSON files are created.
    
    Wired to submit real tasks to the WorkflowOrchestrator.
    
    Usage:
        watcher = NPRDFileWatcher(
            watch_dir=Path("workspaces/nprd/outputs"),
            orchestrator=get_orchestrator()
        )
        watcher.start()
        
        # Later...
        watcher.stop()
    """
    
    def __init__(
        self,
        watch_dir: Path,
        orchestrator: Optional[Any] = None,
        file_pattern: str = "*.json",
        recursive: bool = False
    ):
        """
        Initialize NPRD file watcher.
        
        Args:
            watch_dir: Directory to watch for NPRD files
            orchestrator: WorkflowOrchestrator instance for submitting tasks
            file_pattern: Glob pattern for files to watch (default: "*.json")
            recursive: Watch subdirectories recursively (default: False)
        """
        self.watch_dir = Path(watch_dir)
        self.orchestrator = orchestrator
        self.file_pattern = file_pattern
        self.recursive = recursive
        
        # Create file handler
        self.handler = NPRDFileHandler(
            callback=self._handle_event,
            file_pattern=file_pattern
        )
        
        # Observer
        self.observer = Observer()
        
        # State
        self._running = False
        self._event_queue: List[NPRDWatchEvent] = []
        self._workflow_map: Dict[str, str] = {}  # file_path -> workflow_id
        
        logger.info(f"NPRDFileWatcher initialized for {watch_dir}")
    
    def _handle_event(self, event: NPRDWatchEvent) -> None:
        """Handle file system event by submitting to orchestrator."""
        # Add to event queue
        self._event_queue.append(event)
        
        # Submit to orchestrator if available
        if self.orchestrator:
            try:
                # Create async task for submission
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    workflow_id = loop.run_until_complete(
                        self._submit_to_orchestrator(event)
                    )
                    event.workflow_id = workflow_id
                    self._workflow_map[str(event.file_path)] = workflow_id
                    logger.info(f"Submitted NPRD {event.file_path} as workflow {workflow_id}")
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Failed to submit NPRD to orchestrator: {e}")
    
    async def _submit_to_orchestrator(self, event: NPRDWatchEvent) -> str:
        """Submit NPRD event to the workflow orchestrator."""
        if not self.orchestrator:
            raise RuntimeError("No orchestrator configured")
        
        # Submit the NPRD task
        workflow_id = await self.orchestrator.submit_nprd_task(
            nprd_file=event.file_path,
            metadata={
                "source": "nprd_watcher",
                "event_type": event.event_type,
                "timestamp": event.timestamp
            }
        )
        
        return workflow_id
    
    def set_orchestrator(self, orchestrator: Any) -> None:
        """Set or update the orchestrator instance."""
        self.orchestrator = orchestrator
        logger.info("Orchestrator updated for NPRD watcher")
    
    def start(self) -> None:
        """Start watching the directory."""
        if self._running:
            logger.warning("NPRDFileWatcher already running")
            return
        
        # Ensure directory exists
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        
        # Schedule observer
        self.observer.schedule(
            self.handler,
            str(self.watch_dir),
            recursive=self.recursive
        )
        
        # Start observer
        self.observer.start()
        self._running = True
        
        logger.info(f"NPRDFileWatcher started watching {self.watch_dir}")
    
    def stop(self) -> None:
        """Stop watching the directory."""
        if not self._running:
            return
        
        self.observer.stop()
        self.observer.join()
        self._running = False
        
        logger.info("NPRDFileWatcher stopped")
    
    def get_pending_events(self) -> List[NPRDWatchEvent]:
        """Get list of pending events."""
        return self._event_queue.copy()
    
    def clear_events(self) -> None:
        """Clear event queue."""
        self._event_queue.clear()
    
    def get_workflow_for_file(self, file_path: str) -> Optional[str]:
        """Get the workflow ID for a file path."""
        return self._workflow_map.get(file_path)
    
    def scan_existing_files(self) -> List[Path]:
        """
        Scan for existing NPRD files in the watch directory.
        
        Returns:
            List of existing NPRD file paths
        """
        if not self.watch_dir.exists():
            return []
        
        files = list(self.watch_dir.glob(self.file_pattern))
        
        if self.recursive:
            files.extend([
                f for f in self.watch_dir.rglob(self.file_pattern)
                if f not in files
            ])
        
        return files
    
    def process_existing_files(self) -> None:
        """
        Process existing NPRD files in the watch directory.
        
        This is useful for processing files that existed before
        the watcher was started.
        """
        existing_files = self.scan_existing_files()
        
        for file_path in existing_files:
            if str(file_path) in self.handler._processed_files:
                continue
            
            logger.info(f"Processing existing NPRD file: {file_path}")
            
            # Create watch event
            watch_event = NPRDWatchEvent(
                event_type="existing",
                file_path=file_path,
                timestamp=datetime.now().isoformat()
            )
            
            # Try to load NPRD data
            try:
                with open(file_path, 'r') as f:
                    watch_event.nprd_data = json.load(f)
                
                # Mark as processed
                self.handler._processed_files.add(str(file_path))
                
            except json.JSONDecodeError as e:
                watch_event.error = f"Invalid JSON: {e}"
                logger.error(f"Failed to parse NPRD file {file_path}: {e}")
                
            except Exception as e:
                watch_event.error = str(e)
                logger.error(f"Error reading NPRD file {file_path}: {e}")
            
            # Invoke callback
            try:
                self._handle_event(watch_event)
            except Exception as e:
                logger.error(f"Callback error for {file_path}: {e}")
    
    @property
    def is_running(self) -> bool:
        """Check if watcher is running."""
        return self._running
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# =============================================================================
# Analyst Agent Trigger Function (Legacy Support)
# =============================================================================

async def trigger_analyst_agent(nprd_event: NPRDWatchEvent) -> Dict[str, Any]:
    """
    Trigger the Analyst agent with NPRD content.
    
    This function is called when a new NPRD file is detected.
    It submits the NPRD to the workflow orchestrator for processing.
    
    Args:
        nprd_event: NPRD watch event containing the file data
        
    Returns:
        Dictionary containing trigger result
    """
    logger.info(f"Triggering workflow for NPRD: {nprd_event.file_path}")
    
    if nprd_event.error:
        logger.error(f"NPRD event has error: {nprd_event.error}")
        return {
            "success": False,
            "error": nprd_event.error,
            "file_path": str(nprd_event.file_path)
        }
    
    if not nprd_event.nprd_data:
        logger.error("NPRD event has no data")
        return {
            "success": False,
            "error": "No NPRD data in event",
            "file_path": str(nprd_event.file_path)
        }
    
    # Get orchestrator and submit
    try:
        from src.router.workflow_orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        
        workflow_id = await orchestrator.submit_nprd_task(
            nprd_file=nprd_event.file_path,
            metadata={
                "source": "nprd_watcher_trigger",
                "event_type": nprd_event.event_type
            }
        )
        
        result = {
            "success": True,
            "nprd_file": str(nprd_event.file_path),
            "workflow_id": workflow_id,
            "timeline_clips": len(nprd_event.nprd_data.get("timeline", [])),
            "triggered_at": datetime.now().isoformat(),
            "agent": "orchestrator"
        }
        
        logger.info(f"Workflow {workflow_id} started for NPRD {nprd_event.file_path}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to submit NPRD to orchestrator: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": str(nprd_event.file_path)
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_nprd_watcher(
    watch_dir: Optional[Path] = None,
    orchestrator: Optional[Any] = None,
    on_nprd_created: Optional[Callable[[NPRDWatchEvent], None]] = None
) -> NPRDFileWatcher:
    """
    Create an NPRD file watcher with default configuration.
    
    Args:
        watch_dir: Directory to watch (default: workspaces/nprd/outputs)
        orchestrator: WorkflowOrchestrator instance for submitting tasks
        on_nprd_created: Optional callback for new NPRD files (overrides default)
        
    Returns:
        Configured NPRDFileWatcher instance
    """
    # Default watch directory
    if watch_dir is None:
        watch_dir = Path("workspaces/nprd/outputs")
    
    # Get default orchestrator if not provided
    if orchestrator is None:
        try:
            from src.router.workflow_orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
        except Exception as e:
            logger.warning(f"Could not get default orchestrator: {e}")
    
    watcher = NPRDFileWatcher(
        watch_dir=watch_dir,
        orchestrator=orchestrator
    )
    
    # Override callback if provided
    if on_nprd_created:
        watcher.handler.callback = lambda event: (
            on_nprd_created(event),
            watcher._handle_event(event)
        )
    
    return watcher
