"""
Unit tests for NPRD model provider interface.

Tests the abstract ModelProvider interface and validates that implementations
follow the contract for analyze() and get_rate_limit() methods.
"""

import pytest
import json
import subprocess
from pathlib import Path
from typing import List
from datetime import datetime
from unittest.mock import MagicMock

from src.nprd.providers import ModelProvider
from src.nprd.models import TimelineOutput, TimelineClip, RateLimit


class MockModelProvider(ModelProvider):
    """Mock implementation of ModelProvider for testing."""
    
    def __init__(self, rate_limit: RateLimit):
        self._rate_limit = rate_limit
        self.analyze_called = False
        self.get_rate_limit_called = False
    
    def analyze(self, frames: List[Path], audio: Path, prompt: str) -> TimelineOutput:
        """Mock analyze method that returns a simple timeline."""
        self.analyze_called = True
        
        # Create a simple timeline with one clip per frame
        clips = []
        for i, frame in enumerate(frames):
            clip = TimelineClip(
                clip_id=i + 1,
                timestamp_start=f"00:00:{i*30:02d}",
                timestamp_end=f"00:00:{(i+1)*30:02d}",
                transcript=f"Mock transcript for clip {i+1}",
                visual_description=f"Mock visual description for clip {i+1}",
                frame_path=str(frame)
            )
            clips.append(clip)
        
        return TimelineOutput(
            video_url="https://example.com/test_video",
            title="Test Video",
            duration_seconds=len(frames) * 30,
            processed_at=datetime.now().isoformat(),
            model_provider="mock",
            timeline=clips
        )
    
    def get_rate_limit(self) -> RateLimit:
        """Mock get_rate_limit method."""
        self.get_rate_limit_called = True
        return self._rate_limit


class TestModelProviderInterface:
    """Test suite for ModelProvider abstract interface."""
    
    def test_model_provider_is_abstract(self):
        """Test that ModelProvider cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ModelProvider()
    
    def test_mock_provider_implements_interface(self):
        """Test that mock provider correctly implements the interface."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        # Verify it's an instance of ModelProvider
        assert isinstance(provider, ModelProvider)
    
    def test_analyze_method_signature(self):
        """Test that analyze method has correct signature."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        # Create mock frames and audio
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(3)]
        audio = Path("/tmp/audio.mp3")
        prompt = "Extract verbatim transcripts and objective visual descriptions"
        
        # Call analyze
        result = provider.analyze(frames, audio, prompt)
        
        # Verify method was called
        assert provider.analyze_called
        
        # Verify return type
        assert isinstance(result, TimelineOutput)
        assert len(result.timeline) == 3
        assert result.model_provider == "mock"
    
    def test_analyze_returns_timeline_output(self):
        """Test that analyze returns a valid TimelineOutput."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(2)]
        audio = Path("/tmp/audio.mp3")
        prompt = "Test prompt"
        
        result = provider.analyze(frames, audio, prompt)
        
        # Verify TimelineOutput structure
        assert result.video_url == "https://example.com/test_video"
        assert result.title == "Test Video"
        assert result.duration_seconds == 60  # 2 frames * 30 seconds
        assert result.model_provider == "mock"
        assert len(result.timeline) == 2
        
        # Verify timeline clips
        for i, clip in enumerate(result.timeline):
            assert isinstance(clip, TimelineClip)
            assert clip.clip_id == i + 1
            assert clip.transcript.startswith("Mock transcript")
            assert clip.visual_description.startswith("Mock visual description")
    
    def test_get_rate_limit_method_signature(self):
        """Test that get_rate_limit method has correct signature."""
        rate_limit = RateLimit(requests_per_day=2000, requests_used=100)
        provider = MockModelProvider(rate_limit)
        
        # Call get_rate_limit
        result = provider.get_rate_limit()
        
        # Verify method was called
        assert provider.get_rate_limit_called
        
        # Verify return type
        assert isinstance(result, RateLimit)
        assert result.requests_per_day == 2000
        assert result.requests_used == 100
    
    def test_get_rate_limit_returns_rate_limit(self):
        """Test that get_rate_limit returns a valid RateLimit object."""
        rate_limit = RateLimit(requests_per_day=1500, requests_used=750)
        provider = MockModelProvider(rate_limit)
        
        result = provider.get_rate_limit()
        
        # Verify RateLimit structure
        assert result.requests_per_day == 1500
        assert result.requests_used == 750
        assert result.get_remaining() == 750
    
    def test_unlimited_rate_limit(self):
        """Test provider with unlimited rate limit (subscription-based)."""
        rate_limit = RateLimit(requests_per_day=None)  # Unlimited
        provider = MockModelProvider(rate_limit)
        
        result = provider.get_rate_limit()
        
        # Verify unlimited rate limit
        assert result.requests_per_day is None
        assert not result.is_exceeded()
        assert result.get_remaining() is None
    
    def test_analyze_with_empty_frames(self):
        """Test analyze with empty frames list."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        frames = []
        audio = Path("/tmp/audio.mp3")
        prompt = "Test prompt"
        
        result = provider.analyze(frames, audio, prompt)
        
        # Should return timeline with no clips
        assert len(result.timeline) == 0
        assert result.duration_seconds == 0
    
    def test_analyze_with_single_frame(self):
        """Test analyze with single frame."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        frames = [Path("/tmp/frame_0.jpg")]
        audio = Path("/tmp/audio.mp3")
        prompt = "Test prompt"
        
        result = provider.analyze(frames, audio, prompt)
        
        # Should return timeline with one clip
        assert len(result.timeline) == 1
        assert result.timeline[0].clip_id == 1
        assert result.duration_seconds == 30
    
    def test_analyze_with_multiple_frames(self):
        """Test analyze with multiple frames."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(5)]
        audio = Path("/tmp/audio.mp3")
        prompt = "Test prompt"
        
        result = provider.analyze(frames, audio, prompt)
        
        # Should return timeline with 5 clips
        assert len(result.timeline) == 5
        assert result.duration_seconds == 150  # 5 * 30 seconds
        
        # Verify clip IDs are sequential
        for i, clip in enumerate(result.timeline):
            assert clip.clip_id == i + 1


class TestRateLimitIntegration:
    """Test rate limit integration with model providers."""
    
    def test_rate_limit_tracking(self):
        """Test that rate limit can be tracked across multiple calls."""
        rate_limit = RateLimit(requests_per_day=10, requests_used=0)
        provider = MockModelProvider(rate_limit)
        
        # Make multiple calls and track usage
        for i in range(5):
            rate_limit.increment()
            current_limit = provider.get_rate_limit()
            assert current_limit.requests_used == i + 1
            assert current_limit.get_remaining() == 10 - (i + 1)
    
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded detection."""
        rate_limit = RateLimit(requests_per_day=5, requests_used=5)
        provider = MockModelProvider(rate_limit)
        
        current_limit = provider.get_rate_limit()
        
        # Should be at limit
        assert current_limit.is_exceeded()
        assert current_limit.get_remaining() == 0
    
    def test_rate_limit_not_exceeded(self):
        """Test rate limit not exceeded."""
        rate_limit = RateLimit(requests_per_day=100, requests_used=50)
        provider = MockModelProvider(rate_limit)
        
        current_limit = provider.get_rate_limit()
        
        # Should not be at limit
        assert not current_limit.is_exceeded()
        assert current_limit.get_remaining() == 50


class TestModelProviderContract:
    """Test that model provider implementations follow the contract."""
    
    def test_analyze_must_return_timeline_output(self):
        """Test that analyze must return TimelineOutput type."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        frames = [Path("/tmp/frame_0.jpg")]
        audio = Path("/tmp/audio.mp3")
        prompt = "Test"
        
        result = provider.analyze(frames, audio, prompt)
        
        # Must return TimelineOutput
        assert isinstance(result, TimelineOutput)
        assert hasattr(result, 'video_url')
        assert hasattr(result, 'title')
        assert hasattr(result, 'duration_seconds')
        assert hasattr(result, 'processed_at')
        assert hasattr(result, 'model_provider')
        assert hasattr(result, 'timeline')
    
    def test_get_rate_limit_must_return_rate_limit(self):
        """Test that get_rate_limit must return RateLimit type."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        result = provider.get_rate_limit()
        
        # Must return RateLimit
        assert isinstance(result, RateLimit)
        assert hasattr(result, 'requests_per_day')
        assert hasattr(result, 'requests_used')
        assert hasattr(result, 'window_start')
    
    def test_timeline_clips_have_required_fields(self):
        """Test that timeline clips have all required fields."""
        rate_limit = RateLimit(requests_per_day=1000)
        provider = MockModelProvider(rate_limit)
        
        frames = [Path("/tmp/frame_0.jpg")]
        audio = Path("/tmp/audio.mp3")
        prompt = "Test"
        
        result = provider.analyze(frames, audio, prompt)
        
        # Verify each clip has required fields
        for clip in result.timeline:
            assert isinstance(clip, TimelineClip)
            assert hasattr(clip, 'clip_id')
            assert hasattr(clip, 'timestamp_start')
            assert hasattr(clip, 'timestamp_end')
            assert hasattr(clip, 'transcript')
            assert hasattr(clip, 'visual_description')
            assert hasattr(clip, 'frame_path')
            
            # Verify types
            assert isinstance(clip.clip_id, int)
            assert isinstance(clip.timestamp_start, str)
            assert isinstance(clip.timestamp_end, str)
            assert isinstance(clip.transcript, str)
            assert isinstance(clip.visual_description, str)
            assert isinstance(clip.frame_path, str)



class TestGeminiCLIProvider:
    """Test suite for GeminiCLIProvider implementation."""
    
    def test_gemini_provider_initialization(self):
        """Test GeminiCLIProvider initialization with default settings."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        # Verify default settings
        assert provider.yolo_mode is True
        assert provider.api_key is None
        assert isinstance(provider, ModelProvider)
    
    def test_gemini_provider_initialization_with_api_key(self):
        """Test GeminiCLIProvider initialization with API key."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider(yolo_mode=False, api_key="test-api-key")
        
        # Verify custom settings
        assert provider.yolo_mode is False
        assert provider.api_key == "test-api-key"
    
    def test_gemini_provider_rate_limit_unlimited(self):
        """Test that Gemini provider has unlimited rate limit (subscription-based)."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        rate_limit = provider.get_rate_limit()
        
        # Verify unlimited rate limit
        assert rate_limit.requests_per_day is None
        assert not rate_limit.is_exceeded()
        assert rate_limit.get_remaining() is None
    
    def test_gemini_build_command_with_yolo_mode(self):
        """Test command building with YOLO mode enabled."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider(yolo_mode=True)
        
        frames = [Path("/tmp/frame_0.jpg"), Path("/tmp/frame_1.jpg")]
        audio = Path("/tmp/audio.mp3")
        prompt = "Extract verbatim transcripts"
        
        cmd = provider._build_command(frames, audio, prompt)
        
        # Verify command structure
        assert "gemini" in cmd
        assert "run" in cmd
        assert "--yolo" in cmd
        assert prompt in cmd
        assert "--format" in cmd
        assert "json" in cmd
        
        # Verify files are included
        assert str(audio) in cmd
        assert str(frames[0]) in cmd
        assert str(frames[1]) in cmd
    
    def test_gemini_build_command_without_yolo_mode(self):
        """Test command building with YOLO mode disabled."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider(yolo_mode=False)
        
        frames = [Path("/tmp/frame_0.jpg")]
        audio = Path("/tmp/audio.mp3")
        prompt = "Test prompt"
        
        cmd = provider._build_command(frames, audio, prompt)
        
        # Verify YOLO flag is not present
        assert "--yolo" not in cmd
        
        # Verify other elements are present
        assert "gemini" in cmd
        assert "run" in cmd
        assert prompt in cmd
    
    def test_gemini_build_command_with_api_key(self):
        """Test command building with API key."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider(api_key="test-key-123")
        
        frames = [Path("/tmp/frame_0.jpg")]
        audio = Path("/tmp/audio.mp3")
        prompt = "Test"
        
        cmd = provider._build_command(frames, audio, prompt)
        
        # Verify API key is included
        assert "--api-key" in cmd
        assert "test-key-123" in cmd
    
    def test_gemini_format_timestamp(self):
        """Test timestamp formatting."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        # Test various timestamps
        assert provider._format_timestamp(0) == "00:00:00"
        assert provider._format_timestamp(30) == "00:00:30"
        assert provider._format_timestamp(60) == "00:01:00"
        assert provider._format_timestamp(90) == "00:01:30"
        assert provider._format_timestamp(3600) == "01:00:00"
        assert provider._format_timestamp(3661) == "01:01:01"
    
    def test_gemini_create_empty_timeline(self):
        """Test creating empty timeline."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        timeline = provider._create_empty_timeline()
        
        # Verify empty timeline structure
        assert isinstance(timeline, TimelineOutput)
        assert timeline.model_provider == "gemini"
        assert timeline.duration_seconds == 0
        assert len(timeline.timeline) == 0
    
    def test_gemini_parse_clip(self):
        """Test parsing a single clip."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        clip_data = {
            "transcript": "This is a test transcript",
            "visual_description": "Test visual description"
        }
        frames = [Path("/tmp/frame_0.jpg"), Path("/tmp/frame_1.jpg")]
        
        clip = provider._parse_clip(clip_data, 1, frames)
        
        # Verify clip structure
        assert isinstance(clip, TimelineClip)
        assert clip.clip_id == 1
        assert clip.timestamp_start == "00:00:00"
        assert clip.timestamp_end == "00:00:30"
        assert clip.transcript == "This is a test transcript"
        assert clip.visual_description == "Test visual description"
        assert clip.frame_path == str(frames[0])
    
    def test_gemini_parse_clip_second_clip(self):
        """Test parsing second clip with correct timestamps."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        clip_data = {
            "transcript": "Second clip",
            "visual_description": "Second description"
        }
        frames = [Path("/tmp/frame_0.jpg"), Path("/tmp/frame_1.jpg")]
        
        clip = provider._parse_clip(clip_data, 2, frames)
        
        # Verify timestamps for second clip
        assert clip.clip_id == 2
        assert clip.timestamp_start == "00:00:30"
        assert clip.timestamp_end == "00:01:00"
        assert clip.frame_path == str(frames[1])
    
    def test_gemini_parse_output_with_timeline(self):
        """Test parsing Gemini output with timeline structure."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        output = json.dumps({
            "video_url": "https://example.com/video",
            "title": "Test Video",
            "timeline": [
                {
                    "transcript": "First clip transcript",
                    "visual_description": "First clip visuals"
                },
                {
                    "transcript": "Second clip transcript",
                    "visual_description": "Second clip visuals"
                }
            ]
        })
        
        frames = [Path("/tmp/frame_0.jpg"), Path("/tmp/frame_1.jpg")]
        timeline = provider._parse_output(output, frames)
        
        # Verify timeline structure
        assert isinstance(timeline, TimelineOutput)
        assert timeline.video_url == "https://example.com/video"
        assert timeline.title == "Test Video"
        assert timeline.model_provider == "gemini"
        assert len(timeline.timeline) == 2
        
        # Verify clips
        assert timeline.timeline[0].transcript == "First clip transcript"
        assert timeline.timeline[1].transcript == "Second clip transcript"
    
    def test_gemini_parse_output_with_clips_key(self):
        """Test parsing Gemini output with 'clips' key instead of 'timeline'."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        output = json.dumps({
            "video_url": "https://example.com/video",
            "title": "Test Video",
            "clips": [
                {
                    "transcript": "Clip 1",
                    "visual_description": "Visual 1"
                }
            ]
        })
        
        frames = [Path("/tmp/frame_0.jpg")]
        timeline = provider._parse_output(output, frames)
        
        # Verify timeline was created from clips
        assert len(timeline.timeline) == 1
        assert timeline.timeline[0].transcript == "Clip 1"
    
    def test_gemini_create_clips_from_frames_with_string(self):
        """Test creating clips from frames when text is a string."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        data = {
            "text": "First segment\n\nSecond segment\n\nThird segment"
        }
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(3)]
        
        clips = provider._create_clips_from_frames(data, frames)
        
        # Verify clips were created
        assert len(clips) == 3
        assert clips[0]["transcript"] == "First segment"
        assert clips[1]["transcript"] == "Second segment"
        assert clips[2]["transcript"] == "Third segment"
    
    def test_gemini_create_clips_from_frames_with_list(self):
        """Test creating clips from frames when text is a list."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        data = {
            "text": [
                {"transcript": "Item 1", "visual_description": "Visual 1"},
                {"transcript": "Item 2", "visual_description": "Visual 2"}
            ]
        }
        frames = [Path(f"/tmp/frame_{i}.jpg") for i in range(2)]
        
        clips = provider._create_clips_from_frames(data, frames)
        
        # Verify clips were created
        assert len(clips) == 2
        assert clips[0]["transcript"] == "Item 1"
        assert clips[1]["transcript"] == "Item 2"
    
    def test_gemini_analyze_validation_error_missing_audio(self):
        """Test that analyze raises ValidationError when audio file is missing."""
        from src.nprd.providers import GeminiCLIProvider, ValidationError
        
        provider = GeminiCLIProvider()
        
        frames = [Path("/tmp/frame_0.jpg")]
        audio = Path("/tmp/nonexistent_audio.mp3")
        prompt = "Test"
        
        # Should raise ValidationError
        with pytest.raises(ValidationError, match="Audio file not found"):
            provider.analyze(frames, audio, prompt)
    
    def test_gemini_analyze_validation_error_missing_frame(self):
        """Test that analyze raises ValidationError when frame file is missing."""
        from src.nprd.providers import GeminiCLIProvider, ValidationError
        import tempfile
        
        provider = GeminiCLIProvider()
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio = Path(f.name)
        
        try:
            frames = [Path("/tmp/nonexistent_frame.jpg")]
            prompt = "Test"
            
            # Should raise ValidationError
            with pytest.raises(ValidationError, match="Frame file not found"):
                provider.analyze(frames, audio, prompt)
        finally:
            audio.unlink()
    
    def test_gemini_analyze_empty_frames_returns_empty_timeline(self):
        """Test that analyze with empty frames returns empty timeline."""
        from src.nprd.providers import GeminiCLIProvider
        import tempfile
        
        provider = GeminiCLIProvider()
        
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio = Path(f.name)
        
        try:
            frames = []
            prompt = "Test"
            
            # Should return empty timeline
            timeline = provider.analyze(frames, audio, prompt)
            
            assert isinstance(timeline, TimelineOutput)
            assert len(timeline.timeline) == 0
            assert timeline.duration_seconds == 0
        finally:
            audio.unlink()
    
    def test_gemini_rate_limit_increments(self):
        """Test that rate limit counter increments after successful analysis."""
        from src.nprd.providers import GeminiCLIProvider
        
        provider = GeminiCLIProvider()
        
        # Initial state
        initial_count = provider.get_rate_limit().requests_used
        
        # Increment manually (simulating successful analysis)
        provider.rate_limit.increment()
        
        # Verify increment
        new_count = provider.get_rate_limit().requests_used
        assert new_count == initial_count + 1


class TestGeminiCLIProviderErrorHandling:
    """Test error handling in GeminiCLIProvider."""
    
    def test_gemini_authentication_error_detection(self):
        """Test that authentication errors are properly detected and raised."""
        from src.nprd.providers import GeminiCLIProvider, AuthenticationError
        import tempfile
        from unittest.mock import patch, MagicMock
        
        provider = GeminiCLIProvider()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            frame = Path(f.name)
        
        try:
            frames = [frame]
            prompt = "Test"
            
            # Mock subprocess to raise authentication error
            with patch('subprocess.run') as mock_run:
                error = subprocess.CalledProcessError(1, "gemini")
                error.stderr = "Authentication failed: Invalid API key"
                mock_run.side_effect = error
                
                # Should raise AuthenticationError
                with pytest.raises(AuthenticationError, match="authentication failed"):
                    provider.analyze(frames, audio, prompt)
        finally:
            audio.unlink()
            frame.unlink()
    
    def test_gemini_network_error_detection(self):
        """Test that network errors are properly detected and raised."""
        from src.nprd.providers import GeminiCLIProvider, NetworkError
        import tempfile
        from unittest.mock import patch
        
        provider = GeminiCLIProvider()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            frame = Path(f.name)
        
        try:
            frames = [frame]
            prompt = "Test"
            
            # Mock subprocess to raise network error
            with patch('subprocess.run') as mock_run:
                error = subprocess.CalledProcessError(1, "gemini")
                error.stderr = "Network connection failed"
                mock_run.side_effect = error
                
                # Should raise NetworkError
                with pytest.raises(NetworkError, match="Network error"):
                    provider.analyze(frames, audio, prompt)
        finally:
            audio.unlink()
            frame.unlink()
    
    def test_gemini_json_parse_error(self):
        """Test that JSON parse errors are properly handled."""
        from src.nprd.providers import GeminiCLIProvider, ProviderError
        import tempfile
        from unittest.mock import patch
        
        provider = GeminiCLIProvider()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            frame = Path(f.name)
        
        try:
            frames = [frame]
            prompt = "Test"
            
            # Mock subprocess to return invalid JSON
            with patch('subprocess.run') as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "This is not valid JSON"
                mock_run.return_value = mock_result
                
                # Should raise ProviderError
                with pytest.raises(ProviderError, match="Failed to parse.*JSON"):
                    provider.analyze(frames, audio, prompt)
        finally:
            audio.unlink()
            frame.unlink()
    
    def test_gemini_generic_provider_error(self):
        """Test that generic provider errors are properly handled."""
        from src.nprd.providers import GeminiCLIProvider, ProviderError
        import tempfile
        from unittest.mock import patch
        
        provider = GeminiCLIProvider()
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            audio = Path(f.name)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            frame = Path(f.name)
        
        try:
            frames = [frame]
            prompt = "Test"
            
            # Mock subprocess to raise generic error
            with patch('subprocess.run') as mock_run:
                error = subprocess.CalledProcessError(1, "gemini")
                error.stderr = "Unknown error occurred"
                mock_run.side_effect = error
                
                # Should raise ProviderError
                with pytest.raises(ProviderError, match="Gemini CLI execution failed"):
                    provider.analyze(frames, audio, prompt)
        finally:
            audio.unlink()
            frame.unlink()
