"""
Frame and Audio extractors using ffmpeg.

Extracts frames at specified intervals and audio tracks from video files.
"""

import logging
import subprocess
import math
from pathlib import Path
from typing import List

from src.nprd.exceptions import ExtractionError, ValidationError


logger = logging.getLogger(__name__)


class FrameExtractor:
    """
    Extracts video frames at specified intervals using ffmpeg.
    
    Default: 1 frame per 30 seconds
    Output format: JPEG (quality 95)
    """
    
    def __init__(self, quality: int = 95):
        """
        Initialize frame extractor.
        
        Args:
            quality: JPEG quality (1-100, default: 95)
        """
        self.quality = quality
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> None:
        """Verify ffmpeg is installed."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ExtractionError(
                "ffmpeg is not installed or not in PATH. "
                "Please install ffmpeg to use frame extraction."
            )
    
    def extract_frames(
        self,
        video_path: Path,
        output_dir: Path,
        interval_seconds: int = 30
    ) -> List[Path]:
        """
        Extract frames from video at specified interval.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save extracted frames
            interval_seconds: Seconds between frames (default: 30)
            
        Returns:
            List of paths to extracted frame images
            
        Raises:
            ExtractionError: If frame extraction fails
            ValidationError: If video file is invalid
        """
        # Validate input
        if not video_path.exists():
            raise ValidationError(f"Video file does not exist: {video_path}")
        
        if interval_seconds <= 0:
            raise ValidationError(f"Interval must be positive: {interval_seconds}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video duration
        duration = self._get_video_duration(video_path)
        
        if duration <= 0:
            raise ValidationError(f"Video duration is zero or invalid: {video_path}")
        
        # Calculate expected frame count
        expected_frames = math.ceil(duration / interval_seconds)
        logger.info(
            f"Extracting {expected_frames} frames from {video_path} "
            f"(duration: {duration}s, interval: {interval_seconds}s)"
        )
        
        # Extract frames using ffmpeg
        frame_paths = []
        for i in range(expected_frames):
            timestamp = i * interval_seconds
            frame_index = i + 1
            
            # Format: frame_0000_001.jpg, frame_0030_002.jpg, etc.
            frame_filename = f"frame_{timestamp:04d}_{frame_index:03d}.jpg"
            frame_path = output_dir / frame_filename
            
            try:
                self._extract_single_frame(video_path, timestamp, frame_path)
                frame_paths.append(frame_path)
                logger.debug(f"Extracted frame {frame_index}/{expected_frames}: {frame_path}")
            except Exception as e:
                logger.warning(f"Failed to extract frame at {timestamp}s: {str(e)}")
                # Continue with next frame
        
        # Validate extraction
        if len(frame_paths) == 0:
            raise ExtractionError(f"No frames were extracted from {video_path}")
        
        if len(frame_paths) < expected_frames:
            logger.warning(
                f"Extracted {len(frame_paths)} frames, expected {expected_frames}. "
                "Some frames may be missing."
            )
        
        logger.info(f"Successfully extracted {len(frame_paths)} frames")
        return frame_paths
    
    def _get_video_duration(self, video_path: Path) -> float:
        """
        Get video duration in seconds using ffprobe.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Duration in seconds
        """
        try:
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(video_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            duration_str = result.stdout.strip()
            return float(duration_str)
            
        except (subprocess.CalledProcessError, ValueError) as e:
            raise ExtractionError(f"Failed to get video duration: {str(e)}")
    
    def _extract_single_frame(
        self,
        video_path: Path,
        timestamp: float,
        output_path: Path
    ) -> None:
        """
        Extract a single frame at specified timestamp.
        
        Args:
            video_path: Path to video file
            timestamp: Timestamp in seconds
            output_path: Path to save frame
        """
        try:
            subprocess.run(
                [
                    'ffmpeg',
                    '-ss', str(timestamp),  # Seek to timestamp
                    '-i', str(video_path),  # Input file
                    '-vframes', '1',        # Extract 1 frame
                    '-q:v', str(100 - self.quality),  # Quality (lower is better)
                    '-y',                   # Overwrite output file
                    str(output_path)
                ],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise ExtractionError(
                f"ffmpeg failed to extract frame at {timestamp}s: {e.stderr.decode()}"
            )


class AudioExtractor:
    """
    Extracts audio track from video using ffmpeg.
    
    Default: MP3, 128kbps, mono
    """
    
    def __init__(self):
        """Initialize audio extractor."""
        self._verify_ffmpeg()
    
    def _verify_ffmpeg(self) -> None:
        """Verify ffmpeg is installed."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise ExtractionError(
                "ffmpeg is not installed or not in PATH. "
                "Please install ffmpeg to use audio extraction."
            )
    
    def extract_audio(
        self,
        video_path: Path,
        output_path: Path,
        bitrate: str = "128k",
        channels: int = 1
    ) -> Path:
        """
        Extract audio track from video.
        
        Args:
            video_path: Path to video file
            output_path: Path to save audio file (should end with .mp3)
            bitrate: Audio bitrate (default: "128k")
            channels: Number of audio channels (1=mono, 2=stereo, default: 1)
            
        Returns:
            Path to extracted MP3 file
            
        Raises:
            ExtractionError: If audio extraction fails
            ValidationError: If video file is invalid
        """
        # Validate input
        if not video_path.exists():
            raise ValidationError(f"Video file does not exist: {video_path}")
        
        if channels not in [1, 2]:
            raise ValidationError(f"Channels must be 1 (mono) or 2 (stereo): {channels}")
        
        # Ensure output path has .mp3 extension
        if output_path.suffix.lower() != '.mp3':
            output_path = output_path.with_suffix('.mp3')
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Extracting audio from {video_path} to {output_path} "
            f"(bitrate: {bitrate}, channels: {channels})"
        )
        
        try:
            # Extract audio using ffmpeg
            subprocess.run(
                [
                    'ffmpeg',
                    '-i', str(video_path),      # Input file
                    '-vn',                       # No video
                    '-acodec', 'libmp3lame',    # MP3 codec
                    '-b:a', bitrate,            # Audio bitrate
                    '-ac', str(channels),       # Audio channels
                    '-y',                        # Overwrite output file
                    str(output_path)
                ],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise ExtractionError(
                f"ffmpeg failed to extract audio: {e.stderr.decode()}"
            )
        
        # Validate output
        if not output_path.exists():
            raise ExtractionError(f"Audio file was not created: {output_path}")
        
        if output_path.stat().st_size == 0:
            raise ExtractionError(f"Extracted audio file is empty: {output_path}")
        
        # Verify audio duration matches video duration (±1 second tolerance)
        video_duration = self._get_duration(video_path)
        audio_duration = self._get_duration(output_path)
        
        duration_diff = abs(video_duration - audio_duration)
        if duration_diff > 1.0:
            logger.warning(
                f"Audio duration ({audio_duration}s) differs from video duration "
                f"({video_duration}s) by {duration_diff}s"
            )
        
        logger.info(f"Successfully extracted audio to {output_path}")
        return output_path
    
    def _get_duration(self, file_path: Path) -> float:
        """
        Get media file duration in seconds using ffprobe.
        
        Args:
            file_path: Path to media file
            
        Returns:
            Duration in seconds
        """
        try:
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(file_path)
                ],
                capture_output=True,
                text=True,
                check=True
            )
            
            duration_str = result.stdout.strip()
            return float(duration_str)
            
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.warning(f"Failed to get duration for {file_path}: {str(e)}")
            return 0.0
    
    def extract_primary_track(
        self,
        video_path: Path,
        output_path: Path,
        bitrate: str = "128k",
        channels: int = 1
    ) -> Path:
        """
        Extract primary audio track from video with multiple audio tracks.
        
        Args:
            video_path: Path to video file
            output_path: Path to save audio file
            bitrate: Audio bitrate (default: "128k")
            channels: Number of audio channels (default: 1)
            
        Returns:
            Path to extracted MP3 file
        """
        # For now, this is the same as extract_audio
        # ffmpeg automatically selects the first audio track
        return self.extract_audio(video_path, output_path, bitrate, channels)
