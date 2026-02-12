"""
NPRD Logging System.

This module provides structured logging for the NPRD system with support for:
- Job submission logging (URL, timestamp)
- State transition logging
- Processing time logging
- Error logging with video URL context
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
"""

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
import threading


class LogLevel(str, Enum):
    """Log levels for NPRD logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class LogEntry:
    """
    Structured log entry for NPRD events.
    
    Provides consistent structure for all NPRD log events, enabling
    easy parsing and analysis.
    """
    timestamp: str
    level: str
    event_type: str
    message: str
    job_id: Optional[str] = None
    video_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    state_from: Optional[str] = None
    state_to: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in asdict(self).items():
            if value is not None and (key != 'extra' or value):
                result[key] = value
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class NPRDLogger:
    """
    Structured logger for NPRD system.
    
    Provides consistent logging for all NPRD events with support for:
    - Multiple output destinations (console, file, callback)
    - Structured log entries (JSON format)
    - Event-specific logging methods
    - Processing time measurement
    - Log aggregation and querying
    
    Log Events:
    - job_submitted: New job submitted for processing
    - state_changed: Job state transition
    - processing_started: Processing of a video started
    - processing_completed: Processing completed successfully
    - processing_failed: Processing failed with error
    - download_started: Video download started
    - download_completed: Video download completed
    - extraction_started: Frame/audio extraction started
    - extraction_completed: Frame/audio extraction completed
    - analysis_started: AI analysis started
    - analysis_completed: AI analysis completed
    """
    
    def __init__(
        self,
        name: str = "nprd",
        level: LogLevel = LogLevel.INFO,
        log_file: Optional[Path] = None,
        json_format: bool = True,
        callback: Optional[callable] = None,
    ):
        """
        Initialize NPRD Logger.
        
        Args:
            name: Logger name
            level: Minimum log level
            log_file: Optional file path for log output
            json_format: If True, write JSON-formatted logs
            callback: Optional callback function for log entries
        """
        self.name = name
        self.level = level
        self.log_file = log_file
        self.json_format = json_format
        self.callback = callback
        
        # Internal state
        self._lock = threading.Lock()
        self._entries: List[LogEntry] = []
        self._processing_times: Dict[str, Dict[str, float]] = {}
        
        # Setup Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.value))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level.value))
        
        if json_format:
            console_handler.setFormatter(logging.Formatter('%(message)s'))
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        self._logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file))
            file_handler.setLevel(getattr(logging, level.value))
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(file_handler)
    
    def _log(self, entry: LogEntry) -> None:
        """
        Internal method to log an entry.
        
        Args:
            entry: Log entry to record
        """
        with self._lock:
            # Store entry
            self._entries.append(entry)
            
            # Get Python log level
            log_level = getattr(logging, entry.level, logging.INFO)
            
            # Format message
            if self.json_format:
                message = entry.to_json()
            else:
                message = f"[{entry.event_type}] {entry.message}"
                if entry.job_id:
                    message = f"[{entry.job_id}] {message}"
            
            # Log to Python logger
            self._logger.log(log_level, message)
            
            # Call callback if registered
            if self.callback:
                try:
                    self.callback(entry)
                except Exception as e:
                    self._logger.error(f"Log callback failed: {e}")
    
    def _create_entry(
        self,
        level: LogLevel,
        event_type: str,
        message: str,
        **kwargs
    ) -> LogEntry:
        """
        Create a log entry with current timestamp.
        
        Args:
            level: Log level
            event_type: Type of event
            message: Log message
            **kwargs: Additional entry fields
            
        Returns:
            LogEntry instance
        """
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            event_type=event_type,
            message=message,
            **kwargs
        )
    
    # =========================================================================
    # Job Submission Logging (Property 19)
    # =========================================================================
    
    def log_job_submitted(
        self,
        job_id: str,
        video_url: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log job submission event.
        
        Records the URL and timestamp when a new job is submitted.
        
        Args:
            job_id: Unique job identifier
            video_url: URL of video to process
            extra: Optional additional data
        """
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="job_submitted",
            message=f"Job submitted for URL: {video_url}",
            job_id=job_id,
            video_url=video_url,
            extra=extra or {}
        )
        self._log(entry)
    
    # =========================================================================
    # State Transition Logging (Property 20)
    # =========================================================================
    
    def log_state_changed(
        self,
        job_id: str,
        state_from: str,
        state_to: str,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log job state transition event.
        
        Records state transitions during job processing.
        
        Args:
            job_id: Job identifier
            state_from: Previous state
            state_to: New state
            video_url: Optional video URL
            extra: Optional additional data
        """
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="state_changed",
            message=f"State changed: {state_from} -> {state_to}",
            job_id=job_id,
            video_url=video_url,
            state_from=state_from,
            state_to=state_to,
            extra=extra or {}
        )
        self._log(entry)
    
    # =========================================================================
    # Processing Time Logging (Property 21)
    # =========================================================================
    
    def start_timing(self, job_id: str, phase: str = "total") -> None:
        """
        Start timing a processing phase.
        
        Args:
            job_id: Job identifier
            phase: Phase name (e.g., "download", "extraction", "analysis")
        """
        with self._lock:
            if job_id not in self._processing_times:
                self._processing_times[job_id] = {}
            self._processing_times[job_id][f"{phase}_start"] = time.time()
    
    def end_timing(self, job_id: str, phase: str = "total") -> float:
        """
        End timing a processing phase and return duration.
        
        Args:
            job_id: Job identifier
            phase: Phase name
            
        Returns:
            Duration in seconds (0 if timing not started)
        """
        with self._lock:
            if job_id not in self._processing_times:
                return 0.0
            
            start_key = f"{phase}_start"
            if start_key not in self._processing_times[job_id]:
                return 0.0
            
            start_time = self._processing_times[job_id][start_key]
            duration = time.time() - start_time
            self._processing_times[job_id][f"{phase}_duration"] = duration
            
            return duration
    
    def log_processing_time(
        self,
        job_id: str,
        duration_seconds: float,
        video_url: Optional[str] = None,
        phase: str = "total",
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log processing time for a job.
        
        Records the time taken to process a video or phase.
        
        Args:
            job_id: Job identifier
            duration_seconds: Processing duration in seconds
            video_url: Optional video URL
            phase: Processing phase (e.g., "download", "total")
            extra: Optional additional data
        """
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="processing_time",
            message=f"Processing {phase} completed in {duration_seconds:.2f}s",
            job_id=job_id,
            video_url=video_url,
            duration_seconds=duration_seconds,
            extra={**(extra or {}), "phase": phase}
        )
        self._log(entry)
    
    @contextmanager
    def timed_phase(
        self,
        job_id: str,
        phase: str,
        video_url: Optional[str] = None
    ):
        """
        Context manager for timing a processing phase.
        
        Automatically logs the duration when the context exits.
        
        Args:
            job_id: Job identifier
            phase: Phase name
            video_url: Optional video URL
            
        Yields:
            None
        """
        self.start_timing(job_id, phase)
        try:
            yield
        finally:
            duration = self.end_timing(job_id, phase)
            self.log_processing_time(job_id, duration, video_url, phase)
    
    # =========================================================================
    # Error Logging (Property 22)
    # =========================================================================
    
    def log_error(
        self,
        job_id: str,
        error: Exception,
        video_url: Optional[str] = None,
        phase: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error event.
        
        Records error details including the video URL for context.
        
        Args:
            job_id: Job identifier
            error: Exception that occurred
            video_url: Video URL for context
            phase: Optional processing phase where error occurred
            extra: Optional additional data
        """
        error_type = type(error).__name__
        error_details = str(error)
        
        entry = self._create_entry(
            level=LogLevel.ERROR,
            event_type="error",
            message=f"Error in {phase or 'processing'}: {error_type}: {error_details}",
            job_id=job_id,
            video_url=video_url,
            error_type=error_type,
            error_details=error_details,
            extra={**(extra or {}), "phase": phase} if phase else (extra or {})
        )
        self._log(entry)
    
    # =========================================================================
    # Phase-Specific Logging
    # =========================================================================
    
    def log_download_started(
        self,
        job_id: str,
        video_url: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log download start event."""
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="download_started",
            message=f"Starting download: {video_url}",
            job_id=job_id,
            video_url=video_url,
            extra=extra or {}
        )
        self._log(entry)
    
    def log_download_completed(
        self,
        job_id: str,
        video_url: str,
        duration_seconds: float,
        file_size_bytes: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log download completion event."""
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="download_completed",
            message=f"Download completed in {duration_seconds:.2f}s",
            job_id=job_id,
            video_url=video_url,
            duration_seconds=duration_seconds,
            extra={**(extra or {}), "file_size_bytes": file_size_bytes}
        )
        self._log(entry)
    
    def log_extraction_started(
        self,
        job_id: str,
        extraction_type: str,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log extraction start event (frames or audio)."""
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="extraction_started",
            message=f"Starting {extraction_type} extraction",
            job_id=job_id,
            video_url=video_url,
            extra={**(extra or {}), "extraction_type": extraction_type}
        )
        self._log(entry)
    
    def log_extraction_completed(
        self,
        job_id: str,
        extraction_type: str,
        duration_seconds: float,
        count: Optional[int] = None,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log extraction completion event."""
        message = f"{extraction_type.capitalize()} extraction completed in {duration_seconds:.2f}s"
        if count:
            message += f" ({count} items)"
        
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="extraction_completed",
            message=message,
            job_id=job_id,
            video_url=video_url,
            duration_seconds=duration_seconds,
            extra={
                **(extra or {}),
                "extraction_type": extraction_type,
                "count": count
            }
        )
        self._log(entry)
    
    def log_analysis_started(
        self,
        job_id: str,
        provider: str,
        frame_count: int,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log AI analysis start event."""
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="analysis_started",
            message=f"Starting analysis with {provider} ({frame_count} frames)",
            job_id=job_id,
            video_url=video_url,
            extra={
                **(extra or {}),
                "provider": provider,
                "frame_count": frame_count
            }
        )
        self._log(entry)
    
    def log_analysis_completed(
        self,
        job_id: str,
        provider: str,
        duration_seconds: float,
        clip_count: int,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log AI analysis completion event."""
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="analysis_completed",
            message=f"Analysis completed with {provider} in {duration_seconds:.2f}s ({clip_count} clips)",
            job_id=job_id,
            video_url=video_url,
            duration_seconds=duration_seconds,
            extra={
                **(extra or {}),
                "provider": provider,
                "clip_count": clip_count
            }
        )
        self._log(entry)
    
    def log_processing_completed(
        self,
        job_id: str,
        video_url: str,
        duration_seconds: float,
        clip_count: int,
        provider: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log complete processing finished event."""
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="processing_completed",
            message=f"Processing completed in {duration_seconds:.2f}s ({clip_count} clips)",
            job_id=job_id,
            video_url=video_url,
            duration_seconds=duration_seconds,
            extra={
                **(extra or {}),
                "provider": provider,
                "clip_count": clip_count
            }
        )
        self._log(entry)
    
    def log_processing_failed(
        self,
        job_id: str,
        video_url: str,
        error: Exception,
        duration_seconds: float,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log processing failure event."""
        entry = self._create_entry(
            level=LogLevel.ERROR,
            event_type="processing_failed",
            message=f"Processing failed after {duration_seconds:.2f}s: {error}",
            job_id=job_id,
            video_url=video_url,
            duration_seconds=duration_seconds,
            error_type=type(error).__name__,
            error_details=str(error),
            extra=extra or {}
        )
        self._log(entry)
    
    # =========================================================================
    # Debug and Warning Logging
    # =========================================================================
    
    def debug(
        self,
        message: str,
        job_id: Optional[str] = None,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a debug message."""
        entry = self._create_entry(
            level=LogLevel.DEBUG,
            event_type="debug",
            message=message,
            job_id=job_id,
            video_url=video_url,
            extra=extra or {}
        )
        self._log(entry)
    
    def info(
        self,
        message: str,
        job_id: Optional[str] = None,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an info message."""
        entry = self._create_entry(
            level=LogLevel.INFO,
            event_type="info",
            message=message,
            job_id=job_id,
            video_url=video_url,
            extra=extra or {}
        )
        self._log(entry)
    
    def warning(
        self,
        message: str,
        job_id: Optional[str] = None,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log a warning message."""
        entry = self._create_entry(
            level=LogLevel.WARNING,
            event_type="warning",
            message=message,
            job_id=job_id,
            video_url=video_url,
            extra=extra or {}
        )
        self._log(entry)
    
    def error(
        self,
        message: str,
        job_id: Optional[str] = None,
        video_url: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log an error message."""
        entry = self._create_entry(
            level=LogLevel.ERROR,
            event_type="error",
            message=message,
            job_id=job_id,
            video_url=video_url,
            extra=extra or {}
        )
        self._log(entry)
    
    # =========================================================================
    # Log Querying and Aggregation
    # =========================================================================
    
    def get_entries(
        self,
        job_id: Optional[str] = None,
        event_type: Optional[str] = None,
        level: Optional[LogLevel] = None,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[LogEntry]:
        """
        Get log entries with optional filtering.
        
        Args:
            job_id: Filter by job ID
            event_type: Filter by event type
            level: Filter by minimum log level
            since: Filter entries after this time
            limit: Maximum entries to return
            
        Returns:
            List of matching LogEntry objects
        """
        with self._lock:
            results = []
            level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]
            
            for entry in reversed(self._entries):
                # Apply filters
                if job_id and entry.job_id != job_id:
                    continue
                if event_type and entry.event_type != event_type:
                    continue
                if level:
                    entry_level = LogLevel(entry.level)
                    if level_order.index(entry_level) < level_order.index(level):
                        continue
                if since:
                    entry_time = datetime.fromisoformat(entry.timestamp)
                    if entry_time < since:
                        continue
                
                results.append(entry)
                if len(results) >= limit:
                    break
            
            return list(reversed(results))
    
    def get_job_entries(self, job_id: str) -> List[LogEntry]:
        """
        Get all entries for a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of LogEntry objects for the job
        """
        return self.get_entries(job_id=job_id)
    
    def get_error_entries(
        self,
        job_id: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[LogEntry]:
        """
        Get error entries.
        
        Args:
            job_id: Optional job ID filter
            since: Optional time filter
            
        Returns:
            List of error LogEntry objects
        """
        return self.get_entries(job_id=job_id, level=LogLevel.ERROR, since=since)
    
    def get_processing_times(self, job_id: str) -> Dict[str, float]:
        """
        Get all recorded processing times for a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Dictionary of phase names to durations
        """
        with self._lock:
            if job_id not in self._processing_times:
                return {}
            
            return {
                k.replace("_duration", ""): v
                for k, v in self._processing_times[job_id].items()
                if k.endswith("_duration")
            }
    
    def clear_entries(self) -> None:
        """Clear all stored log entries."""
        with self._lock:
            self._entries.clear()
            self._processing_times.clear()


# Global logger instance
_default_logger: Optional[NPRDLogger] = None


def get_logger(
    name: str = "nprd",
    **kwargs
) -> NPRDLogger:
    """
    Get or create the default NPRD logger.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for NPRDLogger
        
    Returns:
        NPRDLogger instance
    """
    global _default_logger
    
    if _default_logger is None or name != _default_logger.name:
        _default_logger = NPRDLogger(name=name, **kwargs)
    
    return _default_logger


def set_log_level(level: LogLevel) -> None:
    """
    Set the log level for the default logger.
    
    Args:
        level: New log level
    """
    global _default_logger
    
    if _default_logger:
        _default_logger.level = level
        _default_logger._logger.setLevel(getattr(logging, level.value))
