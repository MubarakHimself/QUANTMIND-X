"""Video Ingest System

A dumb video processing tool that extracts unbiased, objective data from videos.
No trading knowledge, no strategy inference, no downstream awareness.
"""

from src.video_ingest.models import (
    TimelineOutput,
    TimelineClip,
    VideoMetadata,
    JobStatus,
    JobState,
    JobOptions,
    RateLimit,
    VideoIngestConfig,
)
from src.video_ingest.providers import ModelProvider, GeminiCLIProvider, QwenVLProvider
from src.video_ingest.processor import VideoIngestProcessor, PlaylistInfo, ProcessingResult
from src.video_ingest.job_queue import JobQueueManager
from src.video_ingest.downloader import VideoDownloader
from src.video_ingest.extractors import FrameExtractor, AudioExtractor
from src.video_ingest.cache import ArtifactCache
from src.video_ingest.retry import RetryHandler
from src.video_ingest.rate_limiter import RateLimiter
from src.video_ingest.exceptions import (
    VideoIngestError,
    DownloadError,
    ExtractionError,
    ValidationError,
    CacheError,
    ProviderError,
    AuthenticationError,
    NetworkError,
    RateLimitError,
    JobError,
    ConfigurationError,
)
from src.video_ingest.logger import VideoIngestLogger, LogEntry, LogLevel, get_logger, set_log_level
from src.video_ingest.cli import cli, main as cli_main, EXIT_SUCCESS, EXIT_ERROR, EXIT_VALIDATION_ERROR, EXIT_AUTH_ERROR
from src.video_ingest.api import app as api_app, create_app, run_server
from src.video_ingest.validator import (
    TimelineValidator,
    ValidationResult,
    validate_timeline,
    is_valid_timeline,
    validate_timeline_strict,
)

__all__ = [
    # Models
    "TimelineOutput",
    "TimelineClip",
    "VideoMetadata",
    "JobStatus",
    "JobState",
    "JobOptions",
    "RateLimit",
    "VideoIngestConfig",
    # Processor
    "VideoIngestProcessor",
    "PlaylistInfo",
    "ProcessingResult",
    # Providers
    "ModelProvider",
    "GeminiCLIProvider",
    "QwenVLProvider",
    # Components
    "JobQueueManager",
    "VideoDownloader",
    "FrameExtractor",
    "AudioExtractor",
    "ArtifactCache",
    "RetryHandler",
    "RateLimiter",
    # Exceptions
    "VideoIngestError",
    "DownloadError",
    "ExtractionError",
    "ValidationError",
    "CacheError",
    "ProviderError",
    "AuthenticationError",
    "NetworkError",
    "RateLimitError",
    "JobError",
    "ConfigurationError",
    # Logging
    "VideoIngestLogger",
    "LogEntry",
    "LogLevel",
    "get_logger",
    "set_log_level",
    # CLI
    "cli",
    "cli_main",
    "EXIT_SUCCESS",
    "EXIT_ERROR",
    "EXIT_VALIDATION_ERROR",
    "EXIT_AUTH_ERROR",
    # API
    "api_app",
    "create_app",
    "run_server",
    # Validator
    "TimelineValidator",
    "ValidationResult",
    "validate_timeline",
    "is_valid_timeline",
    "validate_timeline_strict",
]
