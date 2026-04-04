"""
Video Downloader with pytubefix integration.

Downloads videos from YouTube using pytubefix (maintained fork of pytube).
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from pytubefix import YouTube
from pytubefix.exceptions import VideoUnavailable, RegexMatchError

from src.video_ingest.models import VideoMetadata
from src.video_ingest.exceptions import DownloadError, ValidationError


logger = logging.getLogger(__name__)


class VideoDownloader:
    """
    Downloads videos from YouTube using pytubefix.

    pytubefix is a maintained fork of pytube that handles YouTube's
    new streaming protocols better than yt-dlp in some cases.
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
        Download video from YouTube URL.

        Args:
            url: YouTube video URL
            output_dir: Directory to save video

        Returns:
            VideoMetadata with download information

        Raises:
            DownloadError: If download fails
            ValidationError: If URL is invalid
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading video (attempt {attempt + 1}/{self.max_retries}): {url}")

                # Create YouTube object
                yt = YouTube(url)

                # Get best progressive stream (has both video and audio)
                stream = yt.streams.filter(progressive=True, file_extension='mp4').first()

                if not stream:
                    # Fallback to any stream
                    stream = yt.streams.first()

                if not stream:
                    raise DownloadError(f"No streams available for {url}")

                # Download to file
                logger.info(f"Downloading: {stream.resolution} {stream.mime_type}")
                file_path = stream.download(output_path=str(output_path))

                # Get final file path
                final_path = Path(file_path)

                # Get file size
                file_size = final_path.stat().st_size if final_path.exists() else 0

                # Create metadata
                metadata = VideoMetadata(
                    file_path=final_path,
                    duration=yt.length,
                    resolution=stream.resolution or "unknown",
                    format=stream.mime_type.split('/')[-1] if stream.mime_type else "mp4",
                    file_size=file_size,
                    url=url,
                    title=yt.title,
                )

                logger.info(f"Download complete: {yt.title} ({file_size / 1024 / 1024:.1f} MB)")
                return metadata

            except RegexMatchError as e:
                raise ValidationError(f"Invalid YouTube URL: {url}") from e
            except VideoUnavailable as e:
                raise DownloadError(f"Video unavailable: {url}") from e
            except Exception as e:
                last_error = e
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(self.base_delay * (attempt + 1))

        raise DownloadError(f"Failed to download after {self.max_retries} attempts: {last_error}")

    def get_info(self, url: str) -> dict:
        """
        Get video information without downloading.

        Args:
            url: YouTube video URL

        Returns:
            Dictionary with video metadata
        """
        yt = YouTube(url)

        return {
            "title": yt.title,
            "description": yt.description,
            "length": yt.length,
            "views": yt.views,
            "author": yt.author,
            "publish_date": str(yt.publish_date) if yt.publish_date else None,
            "thumbnail_url": yt.thumbnail_url,
        }

    def download_captions(self, url: str, output_dir: str) -> Optional[Path]:
        """
        Download subtitles/captions using yt-dlp when available.

        Returns the first `.vtt` subtitle file found, or `None` if captions are
        unavailable or yt-dlp fails.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--write-sub",
            "--sub-langs",
            "all,-live_chat",
            "--skip-download",
            "--no-playlist",
            "--output",
            str(output_path / "captions.%(ext)s"),
            url,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                logger.info("Caption download skipped for %s: %s", url, result.stderr.strip())
                return None
        except Exception as exc:
            logger.info("Caption download unavailable for %s: %s", url, exc)
            return None

        caption_files = sorted(output_path.glob("captions*.vtt"))
        return caption_files[0] if caption_files else None
