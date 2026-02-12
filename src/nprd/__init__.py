"""NPRD (Neutral Processing & Reality Documentation) System

A dumb video processing tool that extracts unbiased, objective data from videos.
No trading knowledge, no strategy inference, no downstream awareness.
"""

from src.nprd.models import (
    TimelineOutput,
    TimelineClip,
    VideoMetadata,
    JobStatus,
    JobState,
    JobOptions,
    RateLimit,
    NPRDConfig,
)
from src.nprd.providers import ModelProvider, GeminiCLIProvider, QwenVLProvider
from src.nprd.processor import NPRDProcessor, PlaylistInfo, ProcessingResult
from src.nprd.job_queue import JobQueueManager
from src.nprd.downloader import VideoDownloader
from src.nprd.extractors import FrameExtractor, AudioExtractor
from src.nprd.cache import ArtifactCache
from src.nprd.retry import RetryHandler
from src.nprd.rate_limiter import RateLimiter
from src.nprd.exceptions import (
    NPRDError,
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
from src.nprd.logger import NPRDLogger, LogEntry, LogLevel, get_logger, set_log_level
from src.nprd.cli import cli, main as cli_main, EXIT_SUCCESS, EXIT_ERROR, EXIT_VALIDATION_ERROR, EXIT_AUTH_ERROR
from src.nprd.api import app as api_app, create_app, run_server
from src.nprd.validator import (
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
    "NPRDConfig",
    # Processor
    "NPRDProcessor",
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
    "NPRDError",
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
    "NPRDLogger",
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
