"""
VideoIngest Processing Pipeline.

This module provides the main VideoIngestProcessor class that orchestrates the entire video
processing pipeline: download → frame extraction → audio extraction → AI analysis.
"""

import logging
import tempfile
import shutil
import subprocess
import json
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
    VideoIngestConfig,
)
from .downloader import VideoDownloader
from .extractors import FrameExtractor, AudioExtractor
from .providers import (
    ModelProvider,
    GeminiCLIProvider,
    QwenVLProvider,
    OpenRouterProvider,
    QwenCodeCLIProvider,
)
from .cache import ArtifactCache
from .retry import RetryHandler
from .rate_limiter import RateLimiter, MultiProviderRateLimiter
from .exceptions import (
    VideoIngestError,
    DownloadError,
    ExtractionError,
    ValidationError,
    ProviderError,
    RateLimitError,
)


logger = logging.getLogger(__name__)

CHUNK_ANALYSIS_THRESHOLD_SECONDS = 20 * 60


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


class VideoIngestProcessor:
    """
    Main VideoIngest processing pipeline orchestrator.
    
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
        config: Optional[VideoIngestConfig] = None,
        downloader: Optional[VideoDownloader] = None,
        frame_extractor: Optional[FrameExtractor] = None,
        audio_extractor: Optional[AudioExtractor] = None,
        providers: Optional[List[ModelProvider]] = None,
        cache: Optional[ArtifactCache] = None,
        retry_handler: Optional[RetryHandler] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        Initialize VideoIngest Processor.
        
        Args:
            config: VideoIngest configuration (uses defaults if not provided)
            downloader: Video downloader instance
            frame_extractor: Frame extractor instance
            audio_extractor: Audio extractor instance
            providers: List of model providers (in priority order)
            cache: Artifact cache instance
            retry_handler: Retry handler instance
            rate_limiter: Rate limiter instance
        """
        self.config = config or VideoIngestConfig()
        
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
            max_retries=self.config.max_retry_attempts,
            base_delay=self.config.base_retry_delay
        )
        
        # Initialize rate limiter
        self.rate_limiter = rate_limiter or MultiProviderRateLimiter()
        
        # Job status update callback
        self._status_callback: Optional[callable] = None
        
        logger.info(
            f"VideoIngestProcessor initialized with {len(self.providers)} providers, "
            f"cache at {self.config.cache_dir}"
        )
    
    def _init_default_providers(self) -> List[ModelProvider]:
        """
        Initialize default model providers based on configuration.
        
        Provider priority order:
        1. OpenRouter (primary) - Multi-model access with subscription-based limits
        2. Qwen Code CLI (fallback) - Headless mode with 2000 requests/day free tier
        3. Gemini CLI (tertiary) - YOLO mode if configured
        
        Returns:
            List of ModelProvider instances in priority order
        """
        providers = []
        
        # Primary: OpenRouter provider if API key available
        if self.config.openrouter_api_key:
            providers.append(OpenRouterProvider(
                api_key=self.config.openrouter_api_key,
                model=self.config.openrouter_model
            ))
            logger.info("OpenRouter provider initialized (primary)")
        
        # Fallback: Qwen Code CLI provider if API key available
        if self.config.qwen_api_key:
            providers.append(QwenCodeCLIProvider(
                api_key=self.config.qwen_api_key,
                headless=self.config.qwen_headless,
                model=self.config.qwen_model
            ))
            logger.info("Qwen Code CLI provider initialized (fallback)")
        
        # Tertiary: Gemini CLI provider if authenticated via API key or OAuth.
        gemini_provider = GeminiCLIProvider(
            yolo_mode=self.config.gemini_yolo_mode,
            api_key=self.config.gemini_api_key,
        )
        if gemini_provider.is_authenticated():
            providers.append(gemini_provider)
            logger.info("Gemini CLI provider initialized (tertiary fallback)")

        if not providers:
            logger.warning("No authenticated video ingest providers configured")

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
        work_dir = Path(tempfile.mkdtemp(prefix=f"video_ingest_{job_id}_"))
        
        try:
            # Step 1: Download video
            self._update_status(job_id, JobState.DOWNLOADING, 10, "Downloading video...")
            video_metadata = self._download_video(url, work_dir, options)
            logger.info(f"Video downloaded: {video_metadata.file_path}")

            self._update_status(job_id, JobState.PROCESSING, 20, "Extracting source audio...")
            source_audio_path = self._extract_audio(video_metadata.file_path, work_dir, options)
            self._persist_source_audio(source_audio_path, options)

            self._update_status(job_id, JobState.PROCESSING, 25, "Fetching captions...")
            captions_path = self._download_captions(url, work_dir, options)
            caption_segments = self._parse_caption_file(captions_path) if captions_path else []

            chunk_plan = self._build_chunk_plan(int(video_metadata.duration or 0), options)
            self._persist_chunk_plan(job_id, chunk_plan, options)

            if len(chunk_plan) == 1:
                # Step 2: Extract frames
                self._update_status(job_id, JobState.PROCESSING, 30, "Extracting frames...")
                frames = self._extract_frames(video_metadata.file_path, work_dir, options)
                logger.info(f"Extracted {len(frames)} frames")

                # Step 4: Analyze content
                self._update_status(job_id, JobState.ANALYZING, 70, "Analyzing content...")
                captions_text = self._captions_for_window(
                    caption_segments,
                    0,
                    int(video_metadata.duration or 0),
                )
                timeline, provider_used = self._analyze_content(
                    frames,
                    None if captions_text else source_audio_path,
                    video_metadata,
                    options,
                    captions_text=captions_text,
                )
            else:
                timeline, provider_used = self._process_chunked_video(
                    job_id=job_id,
                    video_metadata=video_metadata,
                    chunk_plan=chunk_plan,
                    work_dir=work_dir,
                    options=options,
                    caption_segments=caption_segments,
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

    def _resolve_chunk_budget_seconds(self, options: Optional[JobOptions] = None) -> int:
        options = options or JobOptions()
        model_name = ""
        if options.model_provider and options.model_provider.lower() == "openrouter":
            model_name = (self.config.openrouter_model or "").lower()
        elif options.model_provider and "qwen" in options.model_provider.lower():
            model_name = (self.config.qwen_model or "").lower()
        else:
            model_name = (self.config.openrouter_model or "").lower()

        model_budgets = {
            "google/gemini-2.0-flash-lite-001": 30 * 60,
            "google/gemini-2.5-flash-lite": 40 * 60,
            "google/gemini-2.0-flash-001": 35 * 60,
            "qwen-vl-plus": 25 * 60,
            "qwen-vl-max": 30 * 60,
        }
        return model_budgets.get(model_name, 30 * 60)

    def _build_chunk_plan(self, duration_seconds: int, options: Optional[JobOptions] = None) -> List[Dict[str, Any]]:
        """Create chunk boundaries for long videos."""
        chunk_budget_seconds = self._resolve_chunk_budget_seconds(options)
        threshold_seconds = min(CHUNK_ANALYSIS_THRESHOLD_SECONDS, chunk_budget_seconds)
        if duration_seconds <= 0 or duration_seconds <= threshold_seconds:
            return [{"start": 0, "end": max(duration_seconds, 0), "label": "full"}]

        chunk_count = max((duration_seconds + chunk_budget_seconds - 1) // chunk_budget_seconds, 2)
        chunk_duration = max(duration_seconds // chunk_count, 1)
        chunks: List[Dict[str, Any]] = []
        for index in range(chunk_count):
            start = index * chunk_duration
            end = duration_seconds if index == chunk_count - 1 else min((index + 1) * chunk_duration, duration_seconds)
            chunks.append({"start": start, "end": end, "label": f"part_{index + 1}"})
        return chunks

    def _chunk_manifest_path(self, job_id: str, options: JobOptions) -> Optional[Path]:
        if not options.artifact_root:
            return None
        return options.artifact_root / "source" / "chunks" / f"{job_id}.json"

    def _persist_chunk_plan(
        self,
        job_id: str,
        chunk_plan: List[Dict[str, Any]],
        options: JobOptions,
    ) -> None:
        chunk_path = self._chunk_manifest_path(job_id, options)
        if chunk_path is None:
            return
        try:
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "job_id": job_id,
                "workflow_id": options.workflow_id,
                "strategy_id": options.strategy_id,
                "strategy_family": options.strategy_family,
                "source_bucket": options.source_bucket,
                "chunk_count": len(chunk_plan),
                "total_duration_seconds": max((int(chunk.get("end", 0) or 0) for chunk in chunk_plan), default=0),
                "updated_at": datetime.now().isoformat(),
                "chunks": [
                    {
                        "chunk_id": index,
                        "label": chunk["label"],
                        "start_seconds": int(chunk["start"]),
                        "end_seconds": int(chunk["end"]),
                        "duration_seconds": max(int(chunk["end"]) - int(chunk["start"]), 0),
                        "status": "pending",
                        "timeline_path": None,
                    }
                    for index, chunk in enumerate(chunk_plan, start=1)
                ],
            }
            chunk_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to persist chunk plan: %s", exc)

    def _update_chunk_manifest_entry(
        self,
        job_id: str,
        chunk_label: str,
        options: JobOptions,
        **updates: Any,
    ) -> None:
        chunk_path = self._chunk_manifest_path(job_id, options)
        if chunk_path is None or not chunk_path.exists():
            return
        try:
            payload = json.loads(chunk_path.read_text(encoding="utf-8"))
            for chunk in payload.get("chunks", []):
                if chunk.get("label") == chunk_label:
                    chunk.update(updates)
                    break
            payload["updated_at"] = datetime.now().isoformat()
            chunk_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to update chunk manifest for %s: %s", chunk_label, exc)

    def _persist_chunk_timeline(
        self,
        job_id: str,
        chunk_label: str,
        timeline: TimelineOutput,
        options: JobOptions,
    ) -> None:
        if not options.artifact_root:
            return
        try:
            chunk_timeline_path = (
                options.artifact_root
                / "source"
                / "timelines"
                / "chunks"
                / job_id
                / f"{chunk_label}.json"
            )
            chunk_timeline_path.parent.mkdir(parents=True, exist_ok=True)
            timeline.save_to_file(chunk_timeline_path)
            relative_path = str(chunk_timeline_path.relative_to(options.artifact_root)).replace("\\", "/")
            self._update_chunk_manifest_entry(
                job_id,
                chunk_label,
                options,
                status="completed",
                timeline_path=relative_path,
            )
        except Exception as exc:
            logger.warning("Failed to persist chunk timeline for %s: %s", chunk_label, exc)

    def _extract_video_chunk(
        self,
        video_path: Path,
        start_seconds: int,
        end_seconds: int,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video_path),
                "-ss",
                str(start_seconds),
                "-to",
                str(end_seconds),
                "-c",
                "copy",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise ExtractionError(f"Failed to extract chunk {start_seconds}-{end_seconds}: {result.stderr}")
        return output_path

    def _process_chunked_video(
        self,
        job_id: str,
        video_metadata: VideoMetadata,
        chunk_plan: List[Dict[str, Any]],
        work_dir: Path,
        options: JobOptions,
        caption_segments: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[TimelineOutput, str]:
        combined_clips: List[TimelineClip] = []
        provider_used = ""
        chunk_root = work_dir / "chunks"
        total_chunks = len(chunk_plan)

        for index, chunk in enumerate(chunk_plan):
            progress_floor = 25 + int((index / total_chunks) * 55)
            label = chunk["label"]
            self._update_status(job_id, JobState.PROCESSING, progress_floor, f"Preparing chunk {label}...")
            self._update_chunk_manifest_entry(job_id, label, options, status="processing")

            chunk_dir = chunk_root / label
            chunk_video = self._extract_video_chunk(
                video_metadata.file_path,
                chunk["start"],
                chunk["end"],
                chunk_dir / "segment.mp4",
            )

            frames = self._extract_frames(chunk_video, chunk_dir, options)
            captions_text = self._captions_for_window(
                caption_segments or [],
                int(chunk["start"]),
                int(chunk["end"]),
            )
            audio_path = None
            if not captions_text:
                audio_path = self._extract_audio(chunk_video, chunk_dir, options)

            self._update_status(job_id, JobState.ANALYZING, progress_floor + 10, f"Analyzing chunk {label}...")
            chunk_metadata = VideoMetadata(
                file_path=chunk_video,
                duration=max(chunk["end"] - chunk["start"], 0),
                resolution=video_metadata.resolution,
                format=video_metadata.format,
                file_size=chunk_video.stat().st_size if chunk_video.exists() else video_metadata.file_size,
                url=video_metadata.url,
                title=video_metadata.title,
            )
            timeline, provider_used = self._analyze_content(
                frames,
                audio_path,
                chunk_metadata,
                options,
                captions_text=captions_text,
            )
            self._persist_chunk_timeline(job_id, label, timeline, options)
            rebased = self._rebase_timeline(timeline, seconds_offset=chunk["start"], clip_offset=len(combined_clips))
            combined_clips.extend(rebased.timeline)

        return (
            TimelineOutput(
                video_url=video_metadata.url,
                title=video_metadata.title or "Untitled",
                duration_seconds=int(video_metadata.duration),
                processed_at=datetime.now().isoformat(),
                model_provider=provider_used.lower() if provider_used else "unknown",
                timeline=combined_clips,
            ),
            provider_used,
        )

    def _rebase_timeline(
        self,
        timeline: TimelineOutput,
        seconds_offset: int,
        clip_offset: int = 0,
    ) -> TimelineOutput:
        rebased_clips: List[TimelineClip] = []
        for index, clip in enumerate(timeline.timeline, start=1):
            rebased_clips.append(
                TimelineClip(
                    clip_id=clip_offset + index,
                    timestamp_start=self._format_timestamp(self._parse_timestamp(clip.timestamp_start) + seconds_offset),
                    timestamp_end=self._format_timestamp(self._parse_timestamp(clip.timestamp_end) + seconds_offset),
                    transcript=clip.transcript,
                    visual_description=clip.visual_description,
                    frame_path=clip.frame_path,
                )
            )

        return TimelineOutput(
            video_url=timeline.video_url,
            title=timeline.title,
            duration_seconds=timeline.duration_seconds,
            processed_at=timeline.processed_at,
            model_provider=timeline.model_provider,
            version=timeline.version,
            timeline=rebased_clips,
        )

    def _parse_timestamp(self, value: str) -> int:
        try:
            hours, minutes, seconds = value.split(":", 2)
            seconds = seconds.split(".", 1)[0]
            hours, minutes, seconds = int(hours), int(minutes), int(seconds)
        except Exception:
            return 0
        return hours * 3600 + minutes * 60 + seconds

    def _format_timestamp(self, total_seconds: int) -> str:
        total_seconds = max(int(total_seconds), 0)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
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
            # Skip cache checking for now - simplified implementation
            # Full cache support requires artifact_type parameter
            return None
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

        return self.retry_handler.execute(download_attempt)
    
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

    def _persist_source_audio(self, audio_path: Path, options: JobOptions) -> Optional[Path]:
        if not options.artifact_root or not audio_path.exists():
            return None
        destination = options.artifact_root / "source" / "audio" / audio_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(audio_path, destination)
        return destination

    def _download_captions(
        self,
        url: str,
        work_dir: Path,
        options: JobOptions,
    ) -> Optional[Path]:
        try:
            caption_path = self.downloader.download_captions(url, str(work_dir / "captions"))
        except Exception as exc:
            logger.info("Caption download unavailable for %s: %s", url, exc)
            return None
        if caption_path is None or not caption_path.exists():
            return None
        if options.artifact_root:
            destination = options.artifact_root / "source" / "captions" / caption_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(caption_path, destination)
            return destination
        return caption_path

    def _parse_caption_file(self, caption_path: Optional[Path]) -> List[Dict[str, Any]]:
        if caption_path is None or not caption_path.exists():
            return []
        lines = caption_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        segments: List[Dict[str, Any]] = []
        start_seconds: Optional[int] = None
        end_seconds: Optional[int] = None
        text_lines: List[str] = []

        def flush() -> None:
            nonlocal start_seconds, end_seconds, text_lines
            if start_seconds is None or end_seconds is None or not text_lines:
                start_seconds = None
                end_seconds = None
                text_lines = []
                return
            text = "\n".join(line.strip() for line in text_lines if line.strip()).strip()
            if text:
                segments.append(
                    {
                        "start_seconds": start_seconds,
                        "end_seconds": end_seconds,
                        "text": text,
                    }
                )
            start_seconds = None
            end_seconds = None
            text_lines = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                flush()
                continue
            if line == "WEBVTT" or line.startswith("NOTE") or line.isdigit():
                continue
            if "-->" in line:
                flush()
                start_text, end_text = [part.strip() for part in line.split("-->", 1)]
                start_seconds = self._parse_timestamp(start_text)
                end_seconds = self._parse_timestamp(end_text)
                continue
            if start_seconds is not None and end_seconds is not None:
                text_lines.append(line)

        flush()
        return segments

    def _captions_for_window(
        self,
        segments: List[Dict[str, Any]],
        start_seconds: int,
        end_seconds: int,
    ) -> Optional[str]:
        matching = [
            segment["text"]
            for segment in segments
            if int(segment["end_seconds"]) > int(start_seconds)
            and int(segment["start_seconds"]) < int(end_seconds)
        ]
        if not matching:
            return None
        return "\n".join(text for text in matching if text).strip() or None

    def _build_analysis_prompt(self, captions_text: Optional[str] = None) -> str:
        if not captions_text:
            return DEFAULT_ANALYSIS_PROMPT
        return (
            DEFAULT_ANALYSIS_PROMPT.strip()
            + "\n\nCaption transcript:\n"
            + captions_text.strip()
            + "\n\nUse the caption transcript as the primary speech source. "
              "Do not invent missing spoken words; rely on visuals for visual descriptions only."
        )
    
    def _analyze_content(
        self,
        frames: List[Path],
        audio_path: Optional[Path],
        video_metadata: VideoMetadata,
        options: JobOptions,
        captions_text: Optional[str] = None,
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

        if not providers_to_try:
            error_msg = "No authenticated video ingest providers configured"
            logger.error(error_msg)
            raise ProviderError(error_msg)

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
                timeline = provider.analyze(
                    frames,
                    None if captions_text else audio_path,
                    self._build_analysis_prompt(captions_text),
                )
                
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
