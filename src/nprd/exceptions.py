"""
Custom exceptions for NPRD system.

This module defines all exception types used throughout the NPRD video processing pipeline.
"""


class NPRDError(Exception):
    """Base exception for all NPRD errors."""
    pass


class DownloadError(NPRDError):
    """
    Raised when video download fails.
    
    Attributes:
        url: The URL that failed to download
        attempts: Number of attempts made
        retryable: Whether the error is retryable
    """
    
    def __init__(self, message: str, url: str = None, attempts: int = 0, retryable: bool = True):
        super().__init__(message)
        self.url = url
        self.attempts = attempts
        self.retryable = retryable


class ExtractionError(NPRDError):
    """
    Raised when frame or audio extraction fails.
    
    This includes failures in ffmpeg operations for extracting frames or audio tracks.
    """
    pass


class ValidationError(NPRDError):
    """
    Raised when input validation fails.
    
    This includes invalid URLs, file formats, or corrupted files.
    """
    pass


class CacheError(NPRDError):
    """
    Raised when cache operations fail.
    
    This includes failures in storing, retrieving, or verifying cached artifacts.
    """
    pass


class ProviderError(NPRDError):
    """
    Base exception for model provider errors.
    
    This is the parent class for all AI model provider-related errors.
    """
    pass


class AuthenticationError(ProviderError):
    """
    Raised when model provider authentication fails.
    
    This includes invalid API keys, expired tokens, or missing credentials.
    """
    pass


class NetworkError(ProviderError):
    """
    Raised when network communication with model provider fails.
    
    This includes connection timeouts, DNS failures, and other network issues.
    """
    pass


class RateLimitError(ProviderError):
    """
    Raised when model provider rate limit is exceeded.
    
    Attributes:
        provider: The provider that hit the rate limit
        limit: The rate limit that was exceeded
        reset_time: When the rate limit will reset (if known)
    """
    
    def __init__(self, message: str, provider: str = None, limit: int = None, reset_time: str = None):
        super().__init__(message)
        self.provider = provider
        self.limit = limit
        self.reset_time = reset_time


class JobError(NPRDError):
    """
    Raised when job queue operations fail.
    
    This includes failures in job submission, status updates, or result retrieval.
    """
    pass


class ConfigurationError(NPRDError):
    """
    Raised when configuration is invalid or missing.
    
    This includes missing required settings, invalid values, or configuration file errors.
    """
    pass
