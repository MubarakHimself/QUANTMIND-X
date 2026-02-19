"""
Tests for OpenRouter provider in NPRD system.

This module tests the OpenRouter provider functionality including:
- Provider initialization
- Video content analysis
- Multimodal content handling
- Rate limit tracking
- Error handling and fallback
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile

from src.nprd.providers import OpenRouterProvider, QwenCodeCLIProvider
from src.nprd.models import TimelineOutput, RateLimit
from src.nprd.exceptions import (
    AuthenticationError,
    NetworkError,
    ValidationError,
    RateLimitError,
    ProviderError,
)


class TestOpenRouterProvider:
    """Test cases for OpenRouter provider."""
    
    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit API key."""
        provider = OpenRouterProvider(
            api_key="test-api-key",
            model="anthropic/claude-sonnet-4"
        )
        
        assert provider.api_key == "test-api-key"
        assert provider.model == "anthropic/claude-sonnet-4"
        assert provider.base_url == "https://openrouter.ai/api/v1"
        assert provider.rate_limit.requests_per_day is None  # Subscription-based
    
    def test_initialization_with_env_var(self):
        """Test provider initialization using environment variable."""
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'env-api-key'}):
            provider = OpenRouterProvider()
            assert provider.api_key == "env-api-key"
    
    def test_supported_models(self):
        """Test that supported models are defined."""
        assert "anthropic/claude-sonnet-4" in OpenRouterProvider.SUPPORTED_MODELS
        assert "google/gemini-2.0-flash-exp" in OpenRouterProvider.SUPPORTED_MODELS
        assert "deepseek/deepseek-coder" in OpenRouterProvider.SUPPORTED_MODELS
        assert "zhipu/glm-4-plus" in OpenRouterProvider.SUPPORTED_MODELS
    
    def test_model_capabilities(self):
        """Test model capability flags."""
        # Gemini should support audio
        gemini_info = OpenRouterProvider.SUPPORTED_MODELS["google/gemini-2.0-flash-exp"]
        assert gemini_info["audio"] is True
        assert gemini_info["multimodal"] is True
        
        # Claude should not support audio
        claude_info = OpenRouterProvider.SUPPORTED_MODELS["anthropic/claude-sonnet-4"]
        assert claude_info["audio"] is False
        assert claude_info["multimodal"] is True
    
    def test_get_rate_limit(self):
        """Test rate limit retrieval."""
        provider = OpenRouterProvider(api_key="test-key")
        rate_limit = provider.get_rate_limit()
        
        assert isinstance(rate_limit, RateLimit)
        assert rate_limit.requests_per_day is None  # Unlimited
    
    def test_analyze_empty_frames(self):
        """Test analysis with no frames returns empty timeline."""
        provider = OpenRouterProvider(api_key="test-key")
        
        # Create temp audio file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio:
            audio.write(b"fake audio")
            audio_path = Path(audio.name)
        
        try:
            result = provider.analyze([], audio_path, "test prompt")
            assert len(result.timeline) == 0
            assert result.model_provider == "openrouter"
        finally:
            audio_path.unlink()
    
    def test_analyze_missing_audio(self):
        """Test analysis with missing audio file raises ValidationError."""
        provider = OpenRouterProvider(api_key="test-key")
        
        with pytest.raises(ValidationError) as exc_info:
            provider.analyze(
                [Path("/nonexistent/frame.jpg")],
                Path("/nonexistent/audio.mp3"),
                "test prompt"
            )
        
        assert "Audio file not found" in str(exc_info.value)
    
    def test_analyze_missing_frame(self):
        """Test analysis with missing frame file raises ValidationError."""
        provider = OpenRouterProvider(api_key="test-key")
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as audio:
            audio.write(b"fake audio")
            audio_path = Path(audio.name)
        
        try:
            with pytest.raises(ValidationError) as exc_info:
                provider.analyze(
                    [Path("/nonexistent/frame.jpg")],
                    audio_path,
                    "test prompt"
                )
            
            assert "Frame file not found" in str(exc_info.value)
        finally:
            audio_path.unlink()
    
    @patch('src.nprd.providers.OpenRouterProvider._call_openrouter_api')
    def test_analyze_success(self, mock_call_api):
        """Test successful analysis with frames and audio."""
        provider = OpenRouterProvider(api_key="test-key")
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create frame files
            frames = []
            for i in range(3):
                frame_path = tmpdir / f"frame_{i}.jpg"
                frame_path.write_bytes(b"fake image")
                frames.append(frame_path)
            
            # Create audio file
            audio_path = tmpdir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            
            # Mock API response
            mock_timeline = TimelineOutput(
                video_url="test",
                title="Test",
                duration_seconds=90,
                processed_at="2024-01-01T00:00:00",
                model_provider="openrouter",
                timeline=[]
            )
            mock_call_api.return_value = mock_timeline
            
            result = provider.analyze(frames, audio_path, "test prompt")
            
            assert result.model_provider == "openrouter"
            mock_call_api.assert_called_once()
    
    @patch('requests.post')
    def test_call_api_direct_success(self, mock_post):
        """Test direct API call success."""
        provider = OpenRouterProvider(api_key="test-key")
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create frame
            frame_path = tmpdir / "frame_0.jpg"
            frame_path.write_bytes(b"fake image")
            
            # Create audio
            audio_path = tmpdir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            
            # Mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Transcript: Test content | Visual: Test visual"
                    }
                }]
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            result = provider._call_openrouter_api_direct([frame_path], audio_path, "test prompt")
            
            assert result.model_provider == "openrouter"
            assert len(result.timeline) == 1
    
    @patch('requests.post')
    def test_authentication_error(self, mock_post):
        """Test authentication error handling."""
        provider = OpenRouterProvider(api_key="invalid-key")
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_path = tmpdir / "frame_0.jpg"
            frame_path.write_bytes(b"fake image")
            audio_path = tmpdir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            
            # Mock 401 response
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")
            mock_post.return_value = mock_response
            
            with pytest.raises(AuthenticationError):
                provider.analyze([frame_path], audio_path, "test prompt")
    
    @patch('requests.post')
    def test_rate_limit_error(self, mock_post):
        """Test rate limit error handling."""
        provider = OpenRouterProvider(api_key="test-key")
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_path = tmpdir / "frame_0.jpg"
            frame_path.write_bytes(b"fake image")
            audio_path = tmpdir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            
            # Mock 429 response
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("429 Rate limit exceeded")
            mock_post.return_value = mock_response
            
            with pytest.raises(RateLimitError):
                provider.analyze([frame_path], audio_path, "test prompt")
    
    @patch('requests.post')
    def test_network_error(self, mock_post):
        """Test network error handling."""
        provider = OpenRouterProvider(api_key="test-key")
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_path = tmpdir / "frame_0.jpg"
            frame_path.write_bytes(b"fake image")
            audio_path = tmpdir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            
            # Mock network error
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("Network connection failed")
            mock_post.return_value = mock_response
            
            with pytest.raises(NetworkError):
                provider.analyze([frame_path], audio_path, "test prompt")


class TestQwenCodeCLIProvider:
    """Test cases for Qwen Code CLI provider."""
    
    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit API key."""
        provider = QwenCodeCLIProvider(
            api_key="test-api-key",
            headless=True,
            model="qwen-vl-plus"
        )
        
        assert provider.api_key == "test-api-key"
        assert provider.headless is True
        assert provider.model == "qwen-vl-plus"
        assert provider.rate_limit.requests_per_day == 2000
    
    def test_initialization_with_env_var(self):
        """Test provider initialization using environment variable."""
        with patch.dict('os.environ', {'QWEN_API_KEY': 'env-api-key'}):
            provider = QwenCodeCLIProvider()
            assert provider.api_key == "env-api-key"
    
    def test_get_rate_limit(self):
        """Test rate limit retrieval."""
        provider = QwenCodeCLIProvider(api_key="test-key")
        rate_limit = provider.get_rate_limit()
        
        assert isinstance(rate_limit, RateLimit)
        assert rate_limit.requests_per_day == 2000
    
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded error."""
        provider = QwenCodeCLIProvider(api_key="test-key")
        
        # Exhaust rate limit
        provider.rate_limit.requests_used = 2001
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_path = tmpdir / "frame_0.jpg"
            frame_path.write_bytes(b"fake image")
            audio_path = tmpdir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            
            with pytest.raises(RateLimitError) as exc_info:
                provider.analyze([frame_path], audio_path, "test prompt")
            
            assert "rate limit exceeded" in str(exc_info.value).lower()
    
    def test_build_command(self):
        """Test CLI command building."""
        provider = QwenCodeCLIProvider(
            api_key="test-key",
            headless=True,
            model="qwen-vl-plus"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_path = tmpdir / "frame_0.jpg"
            audio_path = tmpdir / "audio.mp3"
            
            cmd = provider._build_command([frame_path], audio_path, "test prompt")
            
            assert "qwen-code" in cmd
            assert "run" in cmd
            assert "--headless" in cmd
            assert "--api-key" in cmd
            assert "test-key" in cmd
            assert "--model" in cmd
            assert "qwen-vl-plus" in cmd
            assert "--format" in cmd
            assert "json" in cmd


class TestProviderFallback:
    """Test cases for provider fallback logic."""
    
    def test_openrouter_primary(self):
        """Test that OpenRouter is used as primary provider."""
        from src.nprd.processor import NPRDProcessor
        from src.nprd.models import NPRDConfig
        
        config = NPRDConfig(
            openrouter_api_key="openrouter-key",
            qwen_api_key="qwen-key",
            gemini_api_key="gemini-key"
        )
        
        processor = NPRDProcessor(config=config)
        
        # Check provider order
        assert len(processor.providers) >= 1
        assert isinstance(processor.providers[0], OpenRouterProvider)
    
    def test_qwen_fallback(self):
        """Test that Qwen Code CLI is used as fallback."""
        from src.nprd.processor import NPRDProcessor
        from src.nprd.models import NPRDConfig
        
        config = NPRDConfig(
            openrouter_api_key="openrouter-key",
            qwen_api_key="qwen-key"
        )
        
        processor = NPRDProcessor(config=config)
        
        # Check provider order
        assert len(processor.providers) >= 2
        assert isinstance(processor.providers[0], OpenRouterProvider)
        assert isinstance(processor.providers[1], QwenCodeCLIProvider)
    
    def test_gemini_tertiary(self):
        """Test that Gemini CLI is used as tertiary fallback."""
        from src.nprd.processor import NPRDProcessor
        from src.nprd.models import NPRDConfig
        
        config = NPRDConfig(
            openrouter_api_key="openrouter-key",
            qwen_api_key="qwen-key",
            gemini_api_key="gemini-key"
        )
        
        processor = NPRDProcessor(config=config)
        
        # Check provider order
        assert len(processor.providers) >= 3
        assert isinstance(processor.providers[0], OpenRouterProvider)
        assert isinstance(processor.providers[1], QwenCodeCLIProvider)
        assert isinstance(processor.providers[2], GeminiCLIProvider)


class TestMultimodalContent:
    """Test cases for multimodal content handling."""
    
    @patch('requests.post')
    def test_frames_as_base64_images(self, mock_post):
        """Test that frames are encoded as base64 images."""
        provider = OpenRouterProvider(api_key="test-key")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_path = tmpdir / "frame_0.jpg"
            frame_path.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG header
            
            audio_path = tmpdir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            
            # Mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Test content"
                    }
                }]
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            provider._call_openrouter_api_direct([frame_path], audio_path, "test prompt")
            
            # Check that the request was made with image_url content
            call_args = mock_post.call_args
            messages = call_args.kwargs['json']['messages']
            user_content = messages[1]['content']
            
            # Find image content
            image_content = [c for c in user_content if c.get('type') == 'image_url']
            assert len(image_content) > 0
            assert 'url' in image_content[0]['image_url']
            assert image_content[0]['image_url']['url'].startswith('data:image/jpeg;base64,')
    
    @patch('requests.post')
    def test_audio_for_gemini_model(self, mock_post):
        """Test that audio is included for Gemini model."""
        provider = OpenRouterProvider(
            api_key="test-key",
            model="google/gemini-2.0-flash-exp"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            frame_path = tmpdir / "frame_0.jpg"
            frame_path.write_bytes(b"fake image")
            
            audio_path = tmpdir / "audio.mp3"
            audio_path.write_bytes(b"fake audio")
            
            # Mock response
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": "Test content"
                    }
                }]
            }
            mock_response.raise_for_status = Mock()
            mock_post.return_value = mock_response
            
            provider._call_openrouter_api_direct([frame_path], audio_path, "test prompt")
            
            # Check that the request was made with audio content
            call_args = mock_post.call_args
            messages = call_args.kwargs['json']['messages']
            user_content = messages[1]['content']
            
            # Find audio content
            audio_content = [c for c in user_content if c.get('type') == 'audio_url']
            assert len(audio_content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
