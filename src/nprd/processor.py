"""
NPRD Processing Pipeline.

This module provides the main NPRDProcessor class that orchestrates the entire video
processing pipeline: download → frame extraction → audio extraction → AI analysis.
"""

import logging
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from .models import (
    TimelineOutput,
    TimelineClip,
    VideoMetadata,
    JobOptions,
    JobState,
    NPRDConfig,
)
from .downloader import VideoDownloader
from .extractors import FrameExtractor, AudioExtractor
from .providers import ModelProvider, GeminiCLIProvider, QwenVLProvider
from .cache import ArtifactCache
from .retry import RetryHandler
from .rate_limiter import RateLimiter
from .exceptions import (
    NPRDError,
    DownloadError,
    ExtractionError,
    ValidationError,
    ProviderError,
    RateLimitError,
)


logger = logging.getLogger(__name__)


# Default analysis prompt for video content extraction
DEFAULT_ANALYSIS_PROMPT = """
Analyze this video content and extract information for each 30-second segment.

For each segment, provide:
1. **Transcript**: Verbatim speech/narration (if any)
2. **Visual Description**: Objective description of what is shown on screen

Guidelines:
- Be objective and factual in descriptions
- Do not add interpretation or analysis
- Focus on observable content only
- Use neutral language

Output Format:
Return a JSON object with a "timeline" array, where each element contains:
- "transcript": verbatim text of any speech
- "visual_description": objective description of visuals
"""


@dataclass
class PlaylistInfo:
    """
    Information about a video playlist.
    
    Contains metadata and video URLs extracted from a playlist.
    """
    url: str
    title: str
    video_count: int
    video_urls: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "video_count": self.video_count,
            "video_urls": self.video_urls,
            "metadata": self.metadata,
        }


@dataclass
class ProcessingResult:
    """
    Result of processing a single video or playlist.
    
    Contains the timeline output and processing metadata.
    """
    timeline: TimelineOutput
    processing_time_seconds: float
    cached: bool = False
    provider_used: str = ""
    video_metadata: Optional[VideoMetadata] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeline": self.timeline.to_json(),
            "processing_time_seconds": self.processing_time_seconds,
            "cached": self.cached,
            "provider_used": self.provider_used,
            "video_metadata": self.video_metadata.to_dict() if self.video_metadata else None,
        }


class NPRDProcessor:
    """
    Main NPRD processing pipeline orchestrator.
    
    This class coordinates the entire video processing workflow:
    1. Download video from URL
    2. Extract frames at 30-second intervals
    3. Extract audio track
    4. Analyze content using AI model provider
    5. Generate timeline JSON output
    
    Features:
    - Automatic provider selection and fallback
    - Caching of downloaded videos and extracted artifacts
    - Retry handling for transient failures
    - Rate limiting across providers
    - Playlist support for batch processing
    """
    
    def __init__(
        self,
        config: Optional[NPRDConfig] = None,
        downloader: Optional[VideoDownloader] = None,
        frame_extractor: Optional[FrameExtractor] = None,
        audio_extractor: Optional[AudioExtractor] = None,
        providers: Optional[List[ModelProvider]] = None,
        cache: Optional[ArtifactCache] = None,
        retry_handler: Optional[RetryHandler] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        Initialize NPRD Processor.
        
        Args:
            config: NPRD configuration (uses defaults if not provided)
            downloader: Video downloader instance
            frame_extractor: Frame extractor instance
            audio_extractor: Audio extractor instance
            providers: List of model providers (in priority order)
            cache: Artifact cache instance
            retry_handler: Retry handler instance
            rate_limiter: Rate limiter instance
        """
        self.config = config or NPRDConfig()
        
        # Initialize components
        self.downloader = downloader or VideoDownloader(
            max_retries=self.config.max_retry_attempts,
            base_delay=self.config.base_retry_delay
        )
        self.frame_extractor = frame_extractor or FrameExtractor()
        self.audio_extractor = audio_extractor or AudioExtractor()
        
        # Initialize providers (Gemini first, then Qwen as fallback)
        if providers:
            self.providers = providers
        else:
            self.providers = self._init_default_providers()
        
        # Initialize cache
        self.cache = cache or ArtifactCache(
            cache_dir=self.config.cache_dir,
            max_size_gb=self.config.cache_max_size_gb,
            max_age_days=self.config.cache_max_age_days
        )
        
        # Initialize retry handler
        self.retry_handler = retry_handler or RetryHandler(
            max_attempts=self.config.max_retry_attempts,
            base_delay=self.config.base_retry_delay
        )
        
        # Initialize rate limiter
        self.rate_limiter = rate_limiter or RateLimiter()
        
        # Job status update callback
        self._status_callback: Optional[callable] = None
        
        logger.info(
            f"NPRDProcessor initialized with {len(self.providers)} providers, "
            f"cache at {self.config.cache_dir}"
        )
    
    def _init_default_providers(self) -> List[ModelProvider]:
        """
        Initialize default model providers based on configuration.
        
        Returns:
            List of ModelProvider instances
        """
        providers = []
        
        # Add Gemini provider if API key available
        if self.config.gemini_api_key:
            providers.append(GeminiCLIProvider(
                yolo_mode=self.config.gemini_yolo_mode,
                api_key=self.config.gemini_api_key
            ))
            logger.info("Gemini CLI provider initialized")
        
        # Add Qwen provider if API key available
        if self.config.qwen_api_key:
            providers.append(QwenVLProvider(
                api_key=self.config.qwen_api_key,
                headless=self.config.qwen_headless
            ))
            logger.info("Qwen-VL provider initialized")
        
        # Fallback: Initialize Gemini without explicit key (uses env var)
        if not providers:
            providers.append(GeminiCLIProvider(yolo_mode=True))
            logger.warning("No API keys configured, using Gemini CLI with env var")
        
        return providers
    
    def set_status_callback(self, callback: callable) -> None:
        """
        Set callback for status updates during processing.
        
        The callback receives (job_id, status, progress, message) parameters.
        
        Args:
            callback: Status update callback function
        """
        self._status_callback = callback
    
    def _update_status(
        self,
        job_id: str,
        status: JobState,
        progress: int,
        message: str
    ) -> None:
        """
        Update processing status via callback.
        
        Args:
            job_id: Job identifier
            status: Current job state
            progress: Progress percentage (0-100)
            message: Status message
        """
        if self._status_callback:
            try:
                self._status_callback(job_id, status, progress, message)
            except Exception as e:
                logger.error(f"Status callback failed: {e}")
    
    def process(
        self,
        url: str,
        job_id: Optional[str] = None,
        options: Optional[JobOptions] = None
    ) -> ProcessingResult:
        """
        Process a single video URL.
        
        This is the main entry point for video processing. It orchestrates:
        1. Video download
        2. Frame extraction
        3. Audio extraction
        4. AI analysis
        5. Timeline generation
        
        Args:
            url: Video URL to process
            job_id: Optional job identifier for status updates
            options: Optional processing options
            
        Returns:
            ProcessingResult with timeline and metadata
            
        Raises:
            DownloadError: If video download fails
            ExtractionError: If frame/audio extraction fails
            ProviderError: If AI analysis fails
            ValidationError: If input is invalid
        """
        options = options or JobOptions()
        job_id = job_id or f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        logger.info(f"Starting processing for URL: {url} (job_id: {job_id})")
        
        # Check cache first
        cache_key = self.cache.generate_key(url)
        cached_result = self._check_cache(cache_key, options)
        if cached_result:
            logger.info(f"Cache hit for URL: {url}")
            return cached_result
        
        # Create temporary working directory
        work_dir = Path(tempfile.mkdtemp(prefix=f"nprd_{job_id}_"))
        
        try:
            # Step 1: Download video
            self._update_status(job_id, JobState.DOWNLOADING, 10, "Downloading video...")
            video_metadata = self._download_video(url, work_dir, options)
            logger.info(f"Video downloaded: {video_metadata.file_path}")
            
            # Step 2: Extract frames
            self._update_status(job_id, JobState.PROCESSING, 30, "Extracting frames...")
            frames = self._extract_frames(video_metadata.file_path, work_dir, options)
            logger.info(f"Extracted {len(frames)} frames")
            
            # Step 3: Extract audio
            self._update_status(job_id, JobState.PROCESSING, 50, "Extracting audio...")
            audio_path = self._extract_audio(video_metadata.file_path, work_dir, options)
            logger.info(f"Audio extracted: {audio_path}")
            
            # Step 4: Analyze content
            self._update_status(job_id, JobState.ANALYZING, 70, "Analyzing content...")
            timeline, provider_used = self._analyze_content(
                frames, audio_path, video_metadata, options
            )
            logger.info(f"Analysis complete using {provider_used}")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = ProcessingResult(
                timeline=timeline,
                processing_time_seconds=processing_time,
                cached=False,
                provider_used=provider_used,
                video_metadata=video_metadata,
            )
            
            # Cache result if enabled
            if options.cache_enabled:
                self._cache_result(cache_key, result)
            
            self._update_status(job_id, JobState.COMPLETED, 100, "Processing complete")
            logger.info(f"Processing complete in {processing_time:.2f}s")
            
            return result
            
        finally:
            # Cleanup temporary directory
            try:
                shutil.rmtree(work_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup work directory: {e}")
    
    def _check_cache(
        self,
        cache_key: str,
        options: JobOptions
    ) -> Optional[ProcessingResult]:
        """
        Check cache for existing result.
        
        Args:
            cache_key: Cache key for the URL
            options: Processing options
            
        Returns:
            ProcessingResult if cached, None otherwise
        """
        if not options.cache_enabled:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                # Reconstruct ProcessingResult from cached data
                # This is a simplified version - full implementation would
                # serialize/deserialize the complete result
                return None  # For now, don't use cached timeline results
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: ProcessingResult) -> None:
        """
        Cache processing result.
        
        Args:
            cache_key: Cache key
            result: Processing result to cache
        """
        try:
            # Store timeline JSON in cache
            timeline_json = result.timeline.to_json()
            self.cache.put(cache_key, timeline_json.encode('utf-8'))
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def _download_video(
        self,
        url: str,
        work_dir: Path,
        options: JobOptions
    ) -> VideoMetadata:
        """
        Download video with retry handling.
        
        Args:
            url: Video URL
            work_dir: Working directory
            options: Job options
            
        Returns:
            VideoMetadata with download information
        """
        def download_attempt():
            return self.downloader.download(url, str(work_dir / "video"))
        
        return self.retry_handler.execute(
            download_attempt,
            error_type=DownloadError
        )
    
    def _extract_frames(
        self,
        video_path: Path,
        work_dir: Path,
        options: JobOptions
    ) -> List[Path]:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            work_dir: Working directory
            options: Job options
            
        Returns:
            List of frame paths
        """
        interval = options.frame_interval or self.config.default_frame_interval
        frames_dir = work_dir / "frames"
        
        return self.frame_extractor.extract_frames(
            video_path,
            frames_dir,
            interval_seconds=interval
        )
    
    def _extract_audio(
        self,
        video_path: Path,
        work_dir: Path,
        options: JobOptions
    ) -> Path:
        """
        Extract audio from video.
        
        Args:
            video_path: Path to video file
            work_dir: Working directory
            options: Job options
            
        Returns:
            Path to extracted audio file
        """
        bitrate = options.audio_bitrate or self.config.default_audio_bitrate
        channels = options.audio_channels or self.config.default_audio_channels
        audio_path = work_dir / "audio" / "audio.mp3"
        
        return self.audio_extractor.extract_audio(
            video_path,
            audio_path,
            bitrate=bitrate,
            channels=channels
        )
    
    def _analyze_content(
        self,
        frames: List[Path],
        audio_path: Path,
        video_metadata: VideoMetadata,
        options: JobOptions
    ) -> tuple[TimelineOutput, str]:
        """
        Analyze content using AI provider with fallback.
        
        Args:
            frames: List of frame paths
            audio_path: Path to audio file
            video_metadata: Video metadata
            options: Job options
            
        Returns:
            Tuple of (TimelineOutput, provider_name)
        """
        # Select provider (respect options if specified)
        providers_to_try = self._get_providers_to_try(options)
        
        last_error = None
        
        for provider in providers_to_try:
            provider_name = provider.__class__.__name__
            
            try:
                # Check rate limit
                if not self.rate_limiter.can_proceed(provider_name):
                    logger.warning(f"Rate limit exceeded for {provider_name}, trying next")
                    continue
                
                logger.info(f"Trying provider: {provider_name}")
                
                # Analyze content
                timeline = provider.analyze(frames, audio_path, DEFAULT_ANALYSIS_PROMPT)
                
                # Update timeline metadata
                timeline.video_url = video_metadata.url
                timeline.title = video_metadata.title or "Untitled"
                timeline.duration_seconds = int(video_metadata.duration)
                
                # Record rate limit usage
                self.rate_limiter.record_usage(provider_name)
                
                return timeline, provider_name
                
            except RateLimitError as e:
                logger.warning(f"Rate limit error from {provider_name}: {e}")
                last_error = e
                continue
                
            except ProviderError as e:
                logger.warning(f"Provider error from {provider_name}: {e}")
                last_error = e
                continue
        
        # All providers failed
        error_msg = f"All providers failed. Last error: {last_error}"
        logger.error(error_msg)
        raise ProviderError(error_msg)
    
    def _get_providers_to_try(self, options: JobOptions) -> List[ModelProvider]:
        """
        Get list of providers to try based on options.
        
        Args:
            options: Job options with optional provider preference
            
        Returns:
            List of providers in order of preference
        """
        if options.model_provider:
            # Filter to specified provider
            target = options.model_provider.lower()
            for provider in self.providers:
                provider_name = provider.__class__.__name__.lower()
                if target in provider_name:
                    return [provider]
            
            logger.warning(f"Specified provider '{options.model_provider}' not found, using all")
        
        return self.providers
    
    # =========================================================================
    # Playlist Support
    # =========================================================================
    
    def extract_playlist(self, playlist_url: str) -> PlaylistInfo:
        """
        Extract video URLs from a playlist.
        
        Supports YouTube playlists and other playlist formats supported by yt-dlp.
        
        Args:
            playlist_url: URL of the playlist
            
        Returns:
            PlaylistInfo with video URLs and metadata
            
        Raises:
            ValidationError: If URL is not a valid playlist
            DownloadError: If playlist extraction fails
        """
        import yt_dlp
        
        logger.info(f"Extracting playlist: {playlist_url}")
        
        # Configure yt-dlp for playlist extraction (no download)
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',  # Don't download individual videos
            'skip_download': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist_url, download=False)
                
                if info is None:
                    raise ValidationError(f"Could not extract playlist info from {playlist_url}")
                
                # Check if this is actually a playlist
                if info.get('_type') != 'playlist' and 'entries' not in info:
                    raise ValidationError(
                        f"URL is not a playlist: {playlist_url}. "
                        "Use process() for single videos."
                    )
                
                # Extract video URLs from entries
                entries = info.get('entries', [])
                video_urls = []
                
                for entry in entries:
                    if entry is None:
                        continue
                    
                    # Get video URL
                    video_url = entry.get('url')
                    if not video_url:
                        # Try to construct URL from video ID
                        video_id = entry.get('id')
                        if video_id:
                            # Assume YouTube format
                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                    
                    if video_url:
                        video_urls.append(video_url)
                
                playlist_info = PlaylistInfo(
                    url=playlist_url,
                    title=info.get('title', 'Unknown Playlist'),
                    video_count=len(video_urls),
                    video_urls=video_urls,
                    metadata={
                        'uploader': info.get('uploader'),
                        'uploader_id': info.get('uploader_id'),
                        'description': info.get('description'),
                        'playlist_id': info.get('id'),
                    }
                )
                
                logger.info(
                    f"Extracted playlist '{playlist_info.title}' with "
                    f"{playlist_info.video_count} videos"
                )
                
                return playlist_info
                
        except yt_dlp.DownloadError as e:
            raise DownloadError(
                f"Failed to extract playlist: {e}",
                url=playlist_url,
                retryable=False
            )
        except Exception as e:
            if isinstance(e, (ValidationError, DownloadError)):
                raise
            raise DownloadError(
                f"Unexpected error extracting playlist: {e}",
                url=playlist_url,
                retryable=False
            )
    
    def process_playlist(
        self,
        playlist_url: str,
        job_id_prefix: Optional[str] = None,
        options: Optional[JobOptions] = None,
        on_video_complete: Optional[callable] = None,
    ) -> List[ProcessingResult]:
        """
        Process all videos in a playlist.
        
        Creates separate jobs for each video in the playlist and maintains
        playlist ordering.
        
        Args:
            playlist_url: URL of the playlist
            job_id_prefix: Prefix for job IDs (default: "playlist")
            options: Processing options for each video
            on_video_complete: Callback when each video completes
            
        Returns:
            List of ProcessingResult objects (in playlist order)
            
        Raises:
            ValidationError: If playlist URL is invalid
            DownloadError: If playlist extraction fails
        """
        options = options or JobOptions()
        job_id_prefix = job_id_prefix or "playlist"
        
        # Extract playlist
        playlist_info = self.extract_playlist(playlist_url)
        
        logger.info(
            f"Processing playlist '{playlist_info.title}' with "
            f"{playlist_info.video_count} videos"
        )
        
        results = []
        
        for i, video_url in enumerate(playlist_info.video_urls):
            job_id = f"{job_id_prefix}_{i+1:03d}"
            
            logger.info(
                f"Processing video {i+1}/{playlist_info.video_count}: {video_url}"
            )
            
            try:
                result = self.process(video_url, job_id=job_id, options=options)
                results.append(result)
                
                # Call completion callback
                if on_video_complete:
                    try:
                        on_video_complete(i, video_url, result, None)
                    except Exception as e:
                        logger.warning(f"Video completion callback failed: {e}")
                        
            except Exception as e:
                logger.error(f"Failed to process video {i+1}: {e}")
                
                # Call completion callback with error
                if on_video_complete:
                    try:
                        on_video_complete(i, video_url, None, e)
                    except Exception as cb_error:
                        logger.warning(f"Video completion callback failed: {cb_error}")
                
                # Create error result placeholder
                results.append(ProcessingResult(
                    timeline=TimelineOutput(
                        video_url=video_url,
                        title=f"Error: {video_url}",
                        duration_seconds=0,
                        processed_at=datetime.now().isoformat(),
                        model_provider="error",
                        timeline=[]
                    ),
                    processing_time_seconds=0,
                    cached=False,
                    provider_used="error",
                ))
        
        logger.info(
            f"Playlist processing complete: {len(results)} videos processed"
        )
        
        return results
    
    def create_playlist_jobs(
        self,
        playlist_url: str,
        job_queue,
        options: Optional[JobOptions] = None,
    ) -> List[str]:
        """
        Create separate jobs for each video in a playlist.
        
        This method extracts videos from the playlist and submits them
        to the job queue as individual jobs.
        
        Args:
            playlist_url: URL of the playlist
            job_queue: JobQueueManager instance
            options: Processing options for each video
            
        Returns:
            List of created job IDs (in playlist order)
        """
        options = options or JobOptions()
        
        # Extract playlist
        playlist_info = self.extract_playlist(playlist_url)
        
        logger.info(
            f"Creating jobs for playlist '{playlist_info.title}' with "
            f"{playlist_info.video_count} videos"
        )
        
        job_ids = []
        
        for i, video_url in enumerate(playlist_info.video_urls):
            try:
                job_id = job_queue.submit_job(video_url, options)
                job_ids.append(job_id)
                logger.info(
                    f"Created job {job_id} for video {i+1}/{playlist_info.video_count}"
                )
            except Exception as e:
                logger.error(f"Failed to create job for video {i+1}: {e}")
                job_ids.append(None)
        
        return job_ids
