"""
Model provider interface for NPRD system.

This module defines the abstract interface for AI model providers (Gemini CLI, Qwen-VL)
that analyze video content to generate transcripts and visual descriptions.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import subprocess
import json
import logging

from .models import TimelineOutput, RateLimit, TimelineClip
from .exceptions import (
    ProviderError,
    AuthenticationError,
    NetworkError,
    ValidationError,
    RateLimitError,
)


logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """
    Abstract interface for multimodal AI model providers.
    
    Model providers analyze video frames and audio to generate timeline outputs
    with verbatim transcripts and objective visual descriptions.
    
    Implementations:
    - GeminiCLIProvider: Uses Gemini CLI with YOLO mode
    - QwenVLProvider: Uses Qwen-VL API in headless mode
    """
    
    @abstractmethod
    def analyze(self, frames: List[Path], audio: Path, prompt: str) -> TimelineOutput:
        """
        Analyze video content using multimodal AI.
        
        This method processes extracted video frames and audio to generate a timeline
        with transcripts and visual descriptions. The analysis should be objective
        and unbiased, following the "dumb extraction" philosophy.
        
        Args:
            frames: List of paths to extracted frame images (JPEG format)
            audio: Path to extracted audio file (MP3 format)
            prompt: Analysis prompt instructing the model on extraction methodology
            
        Returns:
            TimelineOutput containing timeline clips with transcripts and descriptions
            
        Raises:
            ProviderError: If analysis fails (authentication, rate limit, etc.)
            NetworkError: If network communication fails
            ValidationError: If input data is invalid
        """
        pass
    
    @abstractmethod
    def get_rate_limit(self) -> RateLimit:
        """
        Get current rate limit status for this provider.
        
        Returns rate limit configuration and current usage statistics.
        This allows the system to track API usage and switch providers
        when limits are approached.
        
        Returns:
            RateLimit object with requests_per_day, requests_used, and window_start
        """
        pass



class GeminiCLIProvider(ModelProvider):
    """
    Gemini CLI provider with YOLO mode support.
    
    This provider uses the @google/gemini-cli tool to analyze video content.
    YOLO mode bypasses permission prompts for automated processing.
    
    Requirements:
    - Gemini CLI installed: npm install -g @google/gemini-cli
    - API key configured: gemini auth or GEMINI_API_KEY env var
    - YOLO mode enabled for automated processing
    
    Rate Limits:
    - Subscription-based (no hard daily limit)
    - Tracks usage for monitoring purposes
    """
    
    def __init__(self, yolo_mode: bool = True, api_key: Optional[str] = None):
        """
        Initialize Gemini CLI provider.
        
        Args:
            yolo_mode: Enable YOLO mode to bypass permission prompts (default: True)
            api_key: Optional API key (uses GEMINI_API_KEY env var if not provided)
        """
        self.yolo_mode = yolo_mode
        self.api_key = api_key
        self.rate_limit = RateLimit(requests_per_day=None)  # Subscription-based, no hard limit
        
        logger.info(f"Initialized GeminiCLIProvider with yolo_mode={yolo_mode}")
    
    def analyze(self, frames: List[Path], audio: Path, prompt: str) -> TimelineOutput:
        """
        Analyze video content using Gemini CLI.
        
        This method processes extracted frames and audio using the Gemini CLI tool.
        It runs the CLI with YOLO mode enabled to bypass permission prompts.
        
        Args:
            frames: List of paths to extracted frame images (JPEG format)
            audio: Path to extracted audio file (MP3 format)
            prompt: Analysis prompt instructing the model on extraction methodology
            
        Returns:
            TimelineOutput containing timeline clips with transcripts and descriptions
            
        Raises:
            AuthenticationError: If Gemini CLI authentication fails
            NetworkError: If network communication fails
            ValidationError: If input data is invalid
            ProviderError: For other provider-specific errors
        """
        # Validate inputs
        if not frames:
            logger.warning("No frames provided for analysis")
            return self._create_empty_timeline()
        
        if not audio.exists():
            raise ValidationError(f"Audio file not found: {audio}")
        
        for frame in frames:
            if not frame.exists():
                raise ValidationError(f"Frame file not found: {frame}")
        
        logger.info(f"Analyzing {len(frames)} frames with Gemini CLI (YOLO mode: {self.yolo_mode})")
        
        try:
            # Build Gemini CLI command
            cmd = self._build_command(frames, audio, prompt)
            
            # Execute Gemini CLI
            result = self._execute_command(cmd)
            
            # Parse JSON output
            timeline = self._parse_output(result, frames)
            
            # Increment rate limit counter
            self.rate_limit.increment()
            
            logger.info(f"Successfully analyzed video with {len(timeline.timeline)} clips")
            return timeline
            
        except subprocess.CalledProcessError as e:
            # Check for authentication errors
            if "authentication" in str(e.stderr).lower() or "api key" in str(e.stderr).lower():
                error_msg = "Gemini CLI authentication failed. Please run 'gemini auth' or set GEMINI_API_KEY environment variable."
                logger.error(error_msg)
                raise AuthenticationError(error_msg) from e
            
            # Check for network errors
            if "network" in str(e.stderr).lower() or "connection" in str(e.stderr).lower():
                error_msg = f"Network error communicating with Gemini API: {e.stderr}"
                logger.error(error_msg)
                raise NetworkError(error_msg) from e
            
            # Generic provider error
            error_msg = f"Gemini CLI execution failed: {e.stderr}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse Gemini CLI JSON output: {e}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e
        
        except Exception as e:
            error_msg = f"Unexpected error during Gemini CLI analysis: {e}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e
    
    def get_rate_limit(self) -> RateLimit:
        """
        Get current rate limit status for Gemini CLI.
        
        Gemini CLI is subscription-based with no hard daily limit.
        This method returns the current usage statistics for monitoring.
        
        Returns:
            RateLimit object with requests_per_day=None (unlimited) and current usage
        """
        return self.rate_limit
    
    def _build_command(self, frames: List[Path], audio: Path, prompt: str) -> List[str]:
        """
        Build Gemini CLI command with YOLO mode.
        
        Args:
            frames: List of frame paths
            audio: Audio file path
            prompt: Analysis prompt
            
        Returns:
            Command as list of strings
        """
        cmd = ["gemini", "run"]
        
        # Add YOLO mode flag if enabled
        if self.yolo_mode:
            cmd.append("--yolo")
        
        # Add API key if provided
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        
        # Add prompt
        cmd.append(prompt)
        
        # Add audio file
        cmd.extend(["-f", str(audio)])
        
        # Add frame files
        for frame in frames:
            cmd.extend(["-f", str(frame)])
        
        # Request JSON output
        cmd.extend(["--format", "json"])
        
        return cmd
    
    def _execute_command(self, cmd: List[str]) -> str:
        """
        Execute Gemini CLI command and return output.
        
        Args:
            cmd: Command to execute
            
        Returns:
            Command stdout as string
            
        Raises:
            subprocess.CalledProcessError: If command fails
        """
        logger.debug(f"Executing command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=300  # 5 minute timeout
        )
        
        return result.stdout
    
    def _parse_output(self, output: str, frames: List[Path]) -> TimelineOutput:
        """
        Parse Gemini CLI JSON output into TimelineOutput.
        
        Args:
            output: JSON output from Gemini CLI
            frames: List of frame paths (for reference)
            
        Returns:
            TimelineOutput object
            
        Raises:
            json.JSONDecodeError: If output is not valid JSON
            KeyError: If required fields are missing
        """
        data = json.loads(output)
        
        # Extract timeline clips from Gemini response
        # The exact structure depends on Gemini CLI output format
        # This is a generic implementation that can be adjusted
        clips = []
        
        # Gemini CLI may return different formats, handle common cases
        if "timeline" in data:
            timeline_data = data["timeline"]
        elif "clips" in data:
            timeline_data = data["clips"]
        elif "segments" in data:
            timeline_data = data["segments"]
        else:
            # If no structured timeline, create clips from frames
            timeline_data = self._create_clips_from_frames(data, frames)
        
        # Parse clips
        for i, clip_data in enumerate(timeline_data):
            clip = self._parse_clip(clip_data, i + 1, frames)
            clips.append(clip)
        
        # Create TimelineOutput
        return TimelineOutput(
            video_url=data.get("video_url", "unknown"),
            title=data.get("title", "Untitled"),
            duration_seconds=len(frames) * 30,  # Estimate from frame count
            processed_at=datetime.now().isoformat(),
            model_provider="gemini",
            timeline=clips
        )
    
    def _parse_clip(self, clip_data: dict, clip_id: int, frames: List[Path]) -> TimelineClip:
        """
        Parse a single clip from Gemini output.
        
        Args:
            clip_data: Clip data from Gemini
            clip_id: Clip ID (1-indexed)
            frames: List of frame paths
            
        Returns:
            TimelineClip object
        """
        # Calculate timestamps based on clip ID
        start_seconds = (clip_id - 1) * 30
        end_seconds = clip_id * 30
        
        # Format timestamps as HH:MM:SS
        timestamp_start = self._format_timestamp(start_seconds)
        timestamp_end = self._format_timestamp(end_seconds)
        
        # Extract transcript and visual description
        transcript = clip_data.get("transcript", clip_data.get("text", ""))
        visual_description = clip_data.get("visual_description", clip_data.get("description", ""))
        
        # Get frame path
        frame_index = clip_id - 1
        frame_path = str(frames[frame_index]) if frame_index < len(frames) else ""
        
        return TimelineClip(
            clip_id=clip_id,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
            transcript=transcript,
            visual_description=visual_description,
            frame_path=frame_path
        )
    
    def _create_clips_from_frames(self, data: dict, frames: List[Path]) -> List[dict]:
        """
        Create clip data from frames when Gemini doesn't return structured timeline.
        
        Args:
            data: Gemini response data
            frames: List of frame paths
            
        Returns:
            List of clip dictionaries
        """
        clips = []
        
        # Extract text content (may be a single string or list)
        text_content = data.get("text", data.get("content", ""))
        
        if isinstance(text_content, str):
            # Split text into segments (one per frame)
            segments = text_content.split("\n\n")
            for i, segment in enumerate(segments[:len(frames)]):
                clips.append({
                    "transcript": segment,
                    "visual_description": f"Frame {i+1} content"
                })
        elif isinstance(text_content, list):
            # Use list items as clips
            for item in text_content[:len(frames)]:
                if isinstance(item, dict):
                    clips.append(item)
                else:
                    clips.append({
                        "transcript": str(item),
                        "visual_description": ""
                    })
        
        # Ensure we have at least one clip per frame
        while len(clips) < len(frames):
            clips.append({
                "transcript": "",
                "visual_description": ""
            })
        
        return clips
    
    def _format_timestamp(self, seconds: int) -> str:
        """
        Format seconds as HH:MM:SS timestamp.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _create_empty_timeline(self) -> TimelineOutput:
        """
        Create an empty timeline output.
        
        Returns:
            Empty TimelineOutput object
        """
        return TimelineOutput(
            video_url="unknown",
            title="Empty",
            duration_seconds=0,
            processed_at=datetime.now().isoformat(),
            model_provider="gemini",
            timeline=[]
        )


class QwenVLProvider(ModelProvider):
    """
    Qwen-VL provider with headless mode support.
    
    This provider uses the Qwen-VL API (OpenAI-compatible) to analyze video content.
    Headless mode allows automated processing without GUI dependencies.
    
    Requirements:
    - Qwen API key: QWEN_API_KEY env var
    - OpenAI-compatible API endpoint
    - Headless mode for non-interactive processing
    
    Rate Limits:
    - Free tier: 2000 requests/day
    - Tracks usage to stay within limits
    """
    
    def __init__(
        self,
        api_key: str,
        headless: bool = True,
        api_endpoint: str = "https://api.qwen.ai/v1",
    ):
        """
        Initialize Qwen-VL provider.
        
        Args:
            api_key: Qwen API key
            headless: Run in headless mode (no GUI) (default: True)
            api_endpoint: API endpoint URL (default: Qwen API)
        """
        self.api_key = api_key
        self.headless = headless
        self.api_endpoint = api_endpoint
        self.rate_limit = RateLimit(requests_per_day=2000)  # Free tier limit
        
        logger.info(f"Initialized QwenVLProvider with headless={headless}")
    
    def analyze(self, frames: List[Path], audio: Path, prompt: str) -> TimelineOutput:
        """
        Analyze video content using Qwen-VL API.
        
        This method processes extracted frames and audio using the Qwen-VL API
        in headless mode (OpenAI-compatible endpoint).
        
        Args:
            frames: List of paths to extracted frame images (JPEG format)
            audio: Path to extracted audio file (MP3 format)
            prompt: Analysis prompt instructing the model on extraction methodology
            
        Returns:
            TimelineOutput containing timeline clips with transcripts and descriptions
            
        Raises:
            AuthenticationError: If Qwen API authentication fails
            NetworkError: If network communication fails
            ValidationError: If input data is invalid
            RateLimitError: If rate limit is exceeded
            ProviderError: For other provider-specific errors
        """
        # Validate inputs
        if not frames:
            logger.warning("No frames provided for analysis")
            return self._create_empty_timeline()
        
        if not audio.exists():
            raise ValidationError(f"Audio file not found: {audio}")
        
        for frame in frames:
            if not frame.exists():
                raise ValidationError(f"Frame file not found: {frame}")
        
        # Check rate limit
        if self.rate_limit.is_exceeded():
            remaining_time = self._get_reset_time()
            raise RateLimitError(
                f"Qwen rate limit exceeded (2000 requests/day). Resets in {remaining_time}",
                provider="qwen",
                limit=2000,
                reset_time=remaining_time,
            )
        
        logger.info(f"Analyzing {len(frames)} frames with Qwen-VL (headless mode: {self.headless})")
        
        try:
            # Call Qwen-VL API
            timeline = self._call_qwen_api(frames, audio, prompt)
            
            # Increment rate limit counter
            self.rate_limit.increment()
            
            logger.info(f"Successfully analyzed video with {len(timeline.timeline)} clips")
            return timeline
            
        except Exception as e:
            # Check for authentication errors
            if "authentication" in str(e).lower() or "api key" in str(e).lower() or "401" in str(e):
                error_msg = "Qwen API authentication failed. Please check QWEN_API_KEY environment variable."
                logger.error(error_msg)
                raise AuthenticationError(error_msg) from e
            
            # Check for rate limit errors
            if "rate limit" in str(e).lower() or "429" in str(e):
                error_msg = f"Qwen API rate limit exceeded: {str(e)}"
                logger.error(error_msg)
                raise RateLimitError(error_msg, provider="qwen", limit=2000) from e
            
            # Check for network errors
            if "network" in str(e).lower() or "connection" in str(e).lower():
                error_msg = f"Network error communicating with Qwen API: {str(e)}"
                logger.error(error_msg)
                raise NetworkError(error_msg) from e
            
            # Generic provider error
            error_msg = f"Qwen API call failed: {str(e)}"
            logger.error(error_msg)
            raise ProviderError(error_msg) from e
    
    def get_rate_limit(self) -> RateLimit:
        """
        Get current rate limit status for Qwen-VL.
        
        Qwen free tier has 2000 requests/day limit.
        This method returns the current usage statistics.
        
        Returns:
            RateLimit object with requests_per_day=2000 and current usage
        """
        return self.rate_limit
    
    def _call_qwen_api(self, frames: List[Path], audio: Path, prompt: str) -> TimelineOutput:
        """
        Call Qwen-VL API using OpenAI-compatible endpoint.
        
        Args:
            frames: List of frame paths
            audio: Audio file path
            prompt: Analysis prompt
            
        Returns:
            TimelineOutput object
        """
        import requests
        import base64
        
        # Prepare multimodal content
        messages = [
            {
                "role": "system",
                "content": "You are a video analysis assistant that extracts verbatim transcripts and objective visual descriptions."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ]
        
        # Add frames as images
        for i, frame in enumerate(frames):
            with open(frame, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}"
                }
            })
        
        # Add audio (if API supports it, otherwise skip)
        # Note: Qwen-VL may not support audio directly, so we focus on visual analysis
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": "qwen-vl-plus",  # or "qwen-vl-max" for better quality
            "messages": messages,
            "temperature": 0.1,  # Low temperature for objective extraction
        }
        
        response = requests.post(
            f"{self.api_endpoint}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300,  # 5 minute timeout
        )
        
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # Parse content into timeline clips
        clips = self._parse_qwen_response(content, frames)
        
        # Create TimelineOutput
        return TimelineOutput(
            video_url="unknown",
            title="Qwen Analysis",
            duration_seconds=len(frames) * 30,  # Estimate from frame count
            processed_at=datetime.now().isoformat(),
            model_provider="qwen",
            timeline=clips
        )
    
    def _parse_qwen_response(self, content: str, frames: List[Path]) -> List[TimelineClip]:
        """
        Parse Qwen API response into timeline clips.
        
        Args:
            content: Response content from Qwen API
            frames: List of frame paths
            
        Returns:
            List of TimelineClip objects
        """
        clips = []
        
        # Split content into segments (one per frame)
        # Qwen may return structured or unstructured content
        segments = content.split("\n\n")
        
        for i, frame in enumerate(frames):
            clip_id = i + 1
            start_seconds = i * 30
            end_seconds = (i + 1) * 30
            
            # Format timestamps
            timestamp_start = self._format_timestamp(start_seconds)
            timestamp_end = self._format_timestamp(end_seconds)
            
            # Extract segment content
            segment_content = segments[i] if i < len(segments) else ""
            
            # Try to parse structured content
            # Format: "Transcript: ... | Visual: ..."
            if "|" in segment_content:
                parts = segment_content.split("|")
                transcript = parts[0].replace("Transcript:", "").strip()
                visual_description = parts[1].replace("Visual:", "").strip() if len(parts) > 1 else ""
            else:
                # Unstructured content - use as transcript
                transcript = segment_content.strip()
                visual_description = f"Frame {clip_id} content"
            
            clip = TimelineClip(
                clip_id=clip_id,
                timestamp_start=timestamp_start,
                timestamp_end=timestamp_end,
                transcript=transcript,
                visual_description=visual_description,
                frame_path=str(frame)
            )
            
            clips.append(clip)
        
        return clips
    
    def _format_timestamp(self, seconds: int) -> str:
        """
        Format seconds as HH:MM:SS timestamp.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _create_empty_timeline(self) -> TimelineOutput:
        """
        Create an empty timeline output.
        
        Returns:
            Empty TimelineOutput object
        """
        return TimelineOutput(
            video_url="unknown",
            title="Empty",
            duration_seconds=0,
            processed_at=datetime.now().isoformat(),
            model_provider="qwen",
            timeline=[]
        )
    
    def _get_reset_time(self) -> str:
        """
        Get time until rate limit resets.
        
        Returns:
            Human-readable time string
        """
        if self.rate_limit.window_start is None:
            return "unknown"
        
        reset_time = self.rate_limit.window_start + timedelta(days=1)
        now = datetime.now()
        
        if reset_time <= now:
            return "now"
        
        delta = reset_time - now
        hours = delta.seconds // 3600
        minutes = (delta.seconds % 3600) // 60
        
        return f"{hours}h {minutes}m"
