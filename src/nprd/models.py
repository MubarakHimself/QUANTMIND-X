"""
Data models for NPRD system.

These models define the core data structures used throughout the NPRD video processing pipeline.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from enum import Enum


class JobState(str, Enum):
    """Job processing states."""
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    PROCESSING = "PROCESSING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class TimelineClip:
    """
    A single clip in the timeline with transcript and visual description.
    
    Represents a 30-second segment of the video with extracted content.
    """
    clip_id: int
    timestamp_start: str  # Format: "HH:MM:SS"
    timestamp_end: str    # Format: "HH:MM:SS"
    transcript: str       # Verbatim transcript of speech
    visual_description: str  # Objective description of visual content
    frame_path: str       # Path to extracted frame image
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TimelineOutput:
    """
    Complete timeline output from NPRD processing.
    
    Contains metadata about the video and an array of timeline clips.
    """
    video_url: str
    title: str
    duration_seconds: int
    processed_at: str  # ISO 8601 format
    model_provider: str  # "gemini" or "qwen"
    version: str = "1.0.0"
    timeline: List[TimelineClip] = field(default_factory=list)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            "meta": {
                "video_url": self.video_url,
                "title": self.title,
                "duration_seconds": self.duration_seconds,
                "processed_at": self.processed_at,
                "model_provider": self.model_provider,
                "version": self.version,
            },
            "timeline": [clip.to_dict() for clip in self.timeline]
        }
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TimelineOutput":
        """Create TimelineOutput from JSON string."""
        data = json.loads(json_str)
        meta = data["meta"]
        timeline_data = data["timeline"]
        
        timeline = [TimelineClip(**clip) for clip in timeline_data]
        
        return cls(
            video_url=meta["video_url"],
            title=meta["title"],
            duration_seconds=meta["duration_seconds"],
            processed_at=meta["processed_at"],
            model_provider=meta["model_provider"],
            version=meta.get("version", "1.0.0"),
            timeline=timeline
        )
    
    def save_to_file(self, file_path: Path) -> None:
        """Save timeline output to JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> "TimelineOutput":
        """Load timeline output from JSON file."""
        with open(file_path, 'r') as f:
            return cls.from_json(f.read())


@dataclass
class VideoMetadata:
    """
    Metadata about a downloaded video.
    
    Contains information extracted during the download process.
    """
    file_path: Path
    duration: float  # Duration in seconds
    resolution: str  # e.g., "1920x1080"
    format: str      # e.g., "mp4", "webm"
    file_size: int   # Size in bytes
    url: str         # Original URL
    title: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "duration": self.duration,
            "resolution": self.resolution,
            "format": self.format,
            "file_size": self.file_size,
            "url": self.url,
            "title": self.title,
        }


@dataclass
class JobStatus:
    """
    Status of a video processing job.
    
    Tracks the current state and progress of a job through the pipeline.
    """
    job_id: str
    status: JobState
    progress: int  # 0-100
    video_url: str
    created_at: datetime
    updated_at: datetime
    logs: List[str] = field(default_factory=list)
    result_path: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "video_url": self.video_url,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "logs": self.logs,
            "result_path": self.result_path,
            "error": self.error,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobStatus":
        """Create JobStatus from dictionary."""
        return cls(
            job_id=data["job_id"],
            status=JobState(data["status"]),
            progress=data["progress"],
            video_url=data["video_url"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            logs=data.get("logs", []),
            result_path=data.get("result_path"),
            error=data.get("error"),
        )


@dataclass
class JobOptions:
    """
    Configuration options for a video processing job.
    
    Allows customization of processing behavior per job.
    """
    output_dir: Optional[Path] = None
    model_provider: Optional[str] = None  # "gemini" or "qwen", None = auto-select
    frame_interval: int = 30  # Seconds between frame extractions
    audio_bitrate: str = "128k"  # Audio bitrate for extraction
    audio_channels: int = 1  # 1 = mono, 2 = stereo
    cache_enabled: bool = True
    retry_attempts: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "model_provider": self.model_provider,
            "frame_interval": self.frame_interval,
            "audio_bitrate": self.audio_bitrate,
            "audio_channels": self.audio_channels,
            "cache_enabled": self.cache_enabled,
            "retry_attempts": self.retry_attempts,
        }


@dataclass
class RateLimit:
    """
    Rate limit configuration for a model provider.
    
    Tracks request limits and current usage.
    """
    requests_per_day: Optional[int]  # None = unlimited (subscription-based)
    requests_used: int = 0
    window_start: Optional[datetime] = None
    
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        if self.requests_per_day is None:
            return False
        return self.requests_used >= self.requests_per_day
    
    def reset_if_needed(self) -> None:
        """Reset counter if window has passed."""
        if self.window_start is None:
            self.window_start = datetime.now()
            return
        
        # Reset if more than 24 hours have passed
        elapsed = datetime.now() - self.window_start
        if elapsed.total_seconds() >= 86400:  # 24 hours
            self.requests_used = 0
            self.window_start = datetime.now()
    
    def increment(self) -> None:
        """Increment request counter."""
        self.reset_if_needed()
        self.requests_used += 1
    
    def get_remaining(self) -> Optional[int]:
        """Get remaining requests in current window."""
        if self.requests_per_day is None:
            return None
        self.reset_if_needed()
        return max(0, self.requests_per_day - self.requests_used)


@dataclass
class NPRDConfig:
    """
    Global configuration for NPRD system.
    
    Loaded from environment variables and configuration files.
    """
    # Model provider settings
    gemini_api_key: Optional[str] = None
    gemini_yolo_mode: bool = True
    qwen_api_key: Optional[str] = None
    qwen_headless: bool = True
    
    # Rate limits
    qwen_requests_per_day: int = 2000
    
    # Cache settings
    cache_dir: Path = Path("data/nprd/cache")
    cache_max_size_gb: int = 50
    cache_max_age_days: int = 30
    
    # Job queue settings
    max_concurrent_jobs: int = 3
    job_db_path: Path = Path("data/nprd/jobs.db")
    
    # Output settings
    output_dir: Path = Path("docs/knowledge/nprd_outputs")
    
    # Processing settings
    default_frame_interval: int = 30
    default_audio_bitrate: str = "128k"
    default_audio_channels: int = 1
    
    # Retry settings
    max_retry_attempts: int = 3
    base_retry_delay: float = 1.0  # seconds
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = Path("data/nprd/nprd.log")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gemini_api_key": "***" if self.gemini_api_key else None,
            "gemini_yolo_mode": self.gemini_yolo_mode,
            "qwen_api_key": "***" if self.qwen_api_key else None,
            "qwen_headless": self.qwen_headless,
            "qwen_requests_per_day": self.qwen_requests_per_day,
            "cache_dir": str(self.cache_dir),
            "cache_max_size_gb": self.cache_max_size_gb,
            "cache_max_age_days": self.cache_max_age_days,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "job_db_path": str(self.job_db_path),
            "output_dir": str(self.output_dir),
            "default_frame_interval": self.default_frame_interval,
            "default_audio_bitrate": self.default_audio_bitrate,
            "default_audio_channels": self.default_audio_channels,
            "max_retry_attempts": self.max_retry_attempts,
            "base_retry_delay": self.base_retry_delay,
            "log_level": self.log_level,
            "log_file": str(self.log_file) if self.log_file else None,
        }
    
    @classmethod
    def from_env(cls) -> "NPRDConfig":
        """Load configuration from environment variables."""
        import os
        
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY"),
            gemini_yolo_mode=os.getenv("GEMINI_YOLO_MODE", "true").lower() == "true",
            qwen_api_key=os.getenv("QWEN_API_KEY"),
            qwen_headless=os.getenv("QWEN_HEADLESS", "true").lower() == "true",
            qwen_requests_per_day=int(os.getenv("QWEN_REQUESTS_PER_DAY", "2000")),
            cache_dir=Path(os.getenv("NPRD_CACHE_DIR", "data/nprd/cache")),
            cache_max_size_gb=int(os.getenv("NPRD_CACHE_MAX_SIZE_GB", "50")),
            cache_max_age_days=int(os.getenv("NPRD_CACHE_MAX_AGE_DAYS", "30")),
            max_concurrent_jobs=int(os.getenv("NPRD_MAX_CONCURRENT_JOBS", "3")),
            job_db_path=Path(os.getenv("NPRD_JOB_DB_PATH", "data/nprd/jobs.db")),
            output_dir=Path(os.getenv("NPRD_OUTPUT_DIR", "docs/knowledge/nprd_outputs")),
            default_frame_interval=int(os.getenv("NPRD_FRAME_INTERVAL", "30")),
            default_audio_bitrate=os.getenv("NPRD_AUDIO_BITRATE", "128k"),
            default_audio_channels=int(os.getenv("NPRD_AUDIO_CHANNELS", "1")),
            max_retry_attempts=int(os.getenv("NPRD_MAX_RETRY_ATTEMPTS", "3")),
            base_retry_delay=float(os.getenv("NPRD_BASE_RETRY_DELAY", "1.0")),
            log_level=os.getenv("NPRD_LOG_LEVEL", "INFO"),
            log_file=Path(os.getenv("NPRD_LOG_FILE", "data/nprd/nprd.log")) if os.getenv("NPRD_LOG_FILE") else None,
        )
