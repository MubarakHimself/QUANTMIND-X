"""Video Ingest File Watcher - Triggers workflow when video analysis completes."""

import json
import logging
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent

logger = logging.getLogger(__name__)


@dataclass
class VideoIngestWatchEvent:
    """Event data for video ingest watch."""
    event_type: str
    video_id: str
    file_path: Path
    timestamp: str
    analysis_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class VideoIngestFileHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[VideoIngestWatchEvent], None]):
        super().__init__()
        self.callback = callback
        self._processed_files: set = set()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if not file_path.name == "analysis.json":
            return

        # Extract video_id from path: video_in/downloads/{video_id}/analysis.json
        try:
            video_id = file_path.parent.name
        except Exception:
            return

        watch_event = VideoIngestWatchEvent(
            event_type="created",
            video_id=video_id,
            file_path=file_path,
            timestamp=datetime.now().isoformat()
        )

        try:
            with open(file_path, 'r') as f:
                watch_event.analysis_data = json.load(f)
            self._processed_files.add(str(file_path))
        except Exception as e:
            watch_event.error = str(e)

        self.callback(watch_event)


class VideoIngestWatcher:
    def __init__(self, watch_dir: Path):
        self.watch_dir = Path(watch_dir)
        self.handler = VideoIngestFileHandler(callback=self._handle_event)
        self.observer = Observer()
        self._running = False
        self._events: List[VideoIngestWatchEvent] = []

    def _handle_event(self, event: VideoIngestWatchEvent) -> None:
        self._events.append(event)
        logger.info(f"Video analysis complete: {event.video_id}")

    def start(self) -> None:
        if self._running:
            return
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.observer.schedule(self.handler, str(self.watch_dir), recursive=True)
        self.observer.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self.observer.stop()
        self.observer.join()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
