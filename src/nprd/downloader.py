"""
Video Downloader with yt-dlp integration.

Downloads videos from YouTube, Vimeo, and direct URLs with retry logic.
"""

import logging
from pathlib import Path
from typing import Optional
import time
import yt_dlp

from src.nprd.models import VideoMetadata
from src.nprd.exceptions import DownloadError, ValidationError


logger = logging.getLogger(__name__)


class VideoDownloader:
    """
    Downloads videos from various sources using yt-dlp.
    
    Supports:
    - YouTube
    - Vimeo
    - Direct video URLs (MP4, WebM, MKV)
    
    Features:
    - Retry logic with exponential backoff
    - File integrity validation
    - Metadata extraction
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        """
        Initialize video downloader.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
    
    def download(self, url: str, output_dir: str) -> VideoMetadata:
        """
        Download video from URL.
        
        Args:
            url: Video URL (YouTube, Vimeo, or direct)
            output_dir: Directory to save video
            
        Returns:
            VideoMetadata with file path, duration, format
            
        Raises:
            DownloadError: If download fails after retries
            ValidationError: If URL is invalid or video is corrupted
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Validate URL format
        if not self._is_valid_url(url):
            raise ValidationError(f"Invalid URL format: {url}")
        
        # Attempt download with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading video from {url} (attempt {attempt + 1}/{self.max_retries})")
                metadata = self._download_with_ytdlp(url, output_path)
                
                # Validate downloaded file
                self._validate_file(metadata.file_path)
                
                logger.info(f"Successfully downloaded video to {metadata.file_path}")
                return metadata
                
            except Exception as e:
                last_error = e
                logger.warning(f"Download attempt {attempt + 1} failed: {str(e)}")
                
                # Check if error is retryable
                if not self._is_retryable_error(e):
                    raise DownloadError(
                        f"Permanent download error: {str(e)}",
                        url=url,
                        attempts=attempt + 1,
                        retryable=False
                    )
                
                # Wait before retry with exponential backoff
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    logger.info(f"Waiting {delay}s before retry...")
                    time.sleep(delay)
        
        # All retries exhausted
        raise DownloadError(
            f"Failed to download video after {self.max_retries} attempts: {str(last_error)}",
            url=url,
            attempts=self.max_retries,
            retryable=False
        )
    
    def _download_with_ytdlp(self, url: str, output_dir: Path) -> VideoMetadata:
        """
        Download video using yt-dlp.
        
        Args:
            url: Video URL
            output_dir: Output directory
            
        Returns:
            VideoMetadata with download information
        """
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Prefer MP4 format
            'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info without downloading first
            info = ydl.extract_info(url, download=False)
            
            if info is None:
                raise DownloadError(f"Could not extract video information from {url}")
            
            # Download the video
            ydl.download([url])
            
            # Get the downloaded file path
            video_id = info.get('id', 'video')
            ext = info.get('ext', 'mp4')
            file_path = output_dir / f"{video_id}.{ext}"
            
            # Extract metadata
            metadata = VideoMetadata(
                file_path=file_path,
                duration=float(info.get('duration', 0)),
                resolution=f"{info.get('width', 0)}x{info.get('height', 0)}",
                format=ext,
                file_size=file_path.stat().st_size if file_path.exists() else 0,
                url=url,
                title=info.get('title'),
            )
            
            return metadata
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate URL format.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid
        """
        # Basic URL validation
        if not url or not isinstance(url, str):
            return False
        
        # Check for common URL patterns
        valid_patterns = [
            'http://',
            'https://',
            'youtube.com',
            'youtu.be',
            'vimeo.com',
        ]
        
        return any(pattern in url.lower() for pattern in valid_patterns)
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate downloaded file.
        
        Args:
            file_path: Path to downloaded file
            
        Raises:
            ValidationError: If file is invalid or corrupted
        """
        if not file_path.exists():
            raise ValidationError(f"Downloaded file does not exist: {file_path}")
        
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValidationError(f"Downloaded file is empty: {file_path}")
        
        # Check file extension
        valid_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov']
        if file_path.suffix.lower() not in valid_extensions:
            raise ValidationError(
                f"Unsupported video format: {file_path.suffix}. "
                f"Supported formats: {', '.join(valid_extensions)}"
            )
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: Exception that occurred
            
        Returns:
            True if error is retryable (transient)
        """
        error_str = str(error).lower()
        
        # Retryable errors (transient)
        retryable_patterns = [
            'timeout',
            'connection',
            'network',
            'temporary',
            'rate limit',
            '429',  # HTTP 429 Too Many Requests
            '503',  # HTTP 503 Service Unavailable
            'dns',
        ]
        
        # Non-retryable errors (permanent)
        permanent_patterns = [
            '404',  # HTTP 404 Not Found
            '403',  # HTTP 403 Forbidden
            'not found',
            'access denied',
            'copyright',
            'video unavailable',  # More specific than just 'unavailable'
            'private video',
            'invalid url',
        ]
        
        # Check for permanent errors first
        if any(pattern in error_str for pattern in permanent_patterns):
            return False
        
        # Check for retryable errors
        if any(pattern in error_str for pattern in retryable_patterns):
            return True
        
        # Default to retryable for unknown errors
        return True
    
    def get_video_info(self, url: str) -> dict:
        """
        Get video information without downloading.
        
        Args:
            url: Video URL
            
        Returns:
            Dictionary with video information
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if info is None:
                raise DownloadError(f"Could not extract video information from {url}")
            
            return {
                'title': info.get('title'),
                'duration': info.get('duration'),
                'resolution': f"{info.get('width', 0)}x{info.get('height', 0)}",
                'format': info.get('ext'),
                'uploader': info.get('uploader'),
                'upload_date': info.get('upload_date'),
                'view_count': info.get('view_count'),
                'description': info.get('description'),
            }
