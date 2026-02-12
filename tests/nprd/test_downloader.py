"""
Unit tests for Video Downloader.

Tests YouTube, Vimeo, and direct URL downloads with error handling.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.nprd.downloader import VideoDownloader
from src.nprd.exceptions import DownloadError, ValidationError


class TestVideoDownloader:
    """Unit tests for VideoDownloader class."""
    
    def test_init(self):
        """Test VideoDownloader initialization."""
        downloader = VideoDownloader(max_retries=5, base_delay=2.0)
        assert downloader.max_retries == 5
        assert downloader.base_delay == 2.0
    
    def test_is_valid_url_youtube(self):
        """Test URL validation for YouTube URLs."""
        downloader = VideoDownloader()
        
        # Valid YouTube URLs
        assert downloader._is_valid_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        assert downloader._is_valid_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert downloader._is_valid_url("https://youtu.be/dQw4w9WgXcQ")
        assert downloader._is_valid_url("http://youtube.com/watch?v=dQw4w9WgXcQ")
    
    def test_is_valid_url_vimeo(self):
        """Test URL validation for Vimeo URLs."""
        downloader = VideoDownloader()
        
        # Valid Vimeo URLs
        assert downloader._is_valid_url("https://vimeo.com/123456789")
        assert downloader._is_valid_url("https://www.vimeo.com/123456789")
        assert downloader._is_valid_url("http://vimeo.com/123456789")
    
    def test_is_valid_url_direct(self):
        """Test URL validation for direct video URLs."""
        downloader = VideoDownloader()
        
        # Valid direct URLs
        assert downloader._is_valid_url("https://example.com/video.mp4")
        assert downloader._is_valid_url("http://example.com/video.webm")
    
    def test_is_valid_url_invalid(self):
        """Test URL validation rejects invalid URLs."""
        downloader = VideoDownloader()
        
        # Invalid URLs
        assert not downloader._is_valid_url("")
        assert not downloader._is_valid_url(None)
        assert not downloader._is_valid_url("not a url")
        assert not downloader._is_valid_url("ftp://example.com/video.mp4")
    
    def test_validate_file_success(self):
        """Test file validation with valid file."""
        downloader = VideoDownloader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid video file
            video_path = Path(tmpdir) / "test.mp4"
            video_path.write_text("fake video content")
            
            # Should not raise exception
            downloader._validate_file(video_path)
    
    def test_validate_file_not_exists(self):
        """Test file validation with non-existent file."""
        downloader = VideoDownloader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "nonexistent.mp4"
            
            with pytest.raises(ValidationError, match="does not exist"):
                downloader._validate_file(video_path)
    
    def test_validate_file_empty(self):
        """Test file validation with empty file."""
        downloader = VideoDownloader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty file
            video_path = Path(tmpdir) / "empty.mp4"
            video_path.touch()
            
            with pytest.raises(ValidationError, match="empty"):
                downloader._validate_file(video_path)
    
    def test_validate_file_unsupported_format(self):
        """Test file validation with unsupported format."""
        downloader = VideoDownloader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with unsupported extension
            video_path = Path(tmpdir) / "test.txt"
            video_path.write_text("not a video")
            
            with pytest.raises(ValidationError, match="Unsupported video format"):
                downloader._validate_file(video_path)
    
    def test_is_retryable_error_transient(self):
        """Test retryable error detection for transient errors."""
        downloader = VideoDownloader()
        
        # Transient errors (should be retryable)
        assert downloader._is_retryable_error(Exception("Connection timeout"))
        assert downloader._is_retryable_error(Exception("Network error"))
        assert downloader._is_retryable_error(Exception("HTTP 429 rate limit"))
        assert downloader._is_retryable_error(Exception("HTTP 503 service unavailable"))
        assert downloader._is_retryable_error(Exception("DNS resolution failed"))
    
    def test_is_retryable_error_permanent(self):
        """Test retryable error detection for permanent errors."""
        downloader = VideoDownloader()
        
        # Permanent errors (should not be retryable)
        assert not downloader._is_retryable_error(Exception("HTTP 404 not found"))
        assert not downloader._is_retryable_error(Exception("HTTP 403 access denied"))
        assert not downloader._is_retryable_error(Exception("Video not found"))
        assert not downloader._is_retryable_error(Exception("Copyright restriction"))
        assert not downloader._is_retryable_error(Exception("Private video"))
        assert not downloader._is_retryable_error(Exception("Invalid URL"))
    
    @patch('src.nprd.downloader.yt_dlp.YoutubeDL')
    def test_download_youtube_success(self, mock_ytdl_class):
        """Test successful YouTube video download."""
        downloader = VideoDownloader()
        
        # Mock yt-dlp
        mock_ytdl = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        
        # Mock video info
        mock_info = {
            'id': 'test_video',
            'ext': 'mp4',
            'duration': 120,
            'width': 1920,
            'height': 1080,
            'title': 'Test Video',
        }
        mock_ytdl.extract_info.return_value = mock_info
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the expected output file
            output_path = Path(tmpdir) / "test_video.mp4"
            output_path.write_text("fake video content")
            
            # Download
            url = "https://youtube.com/watch?v=test_video"
            metadata = downloader.download(url, tmpdir)
            
            # Verify
            assert metadata.file_path == output_path
            assert metadata.duration == 120
            assert metadata.resolution == "1920x1080"
            assert metadata.format == "mp4"
            assert metadata.url == url
            assert metadata.title == "Test Video"
            
            # Verify yt-dlp was called
            mock_ytdl.extract_info.assert_called_once_with(url, download=False)
            mock_ytdl.download.assert_called_once_with([url])
    
    @patch('src.nprd.downloader.yt_dlp.YoutubeDL')
    def test_download_vimeo_success(self, mock_ytdl_class):
        """Test successful Vimeo video download."""
        downloader = VideoDownloader()
        
        # Mock yt-dlp
        mock_ytdl = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        
        # Mock video info
        mock_info = {
            'id': 'vimeo_123',
            'ext': 'mp4',
            'duration': 180,
            'width': 1280,
            'height': 720,
            'title': 'Vimeo Test Video',
        }
        mock_ytdl.extract_info.return_value = mock_info
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the expected output file
            output_path = Path(tmpdir) / "vimeo_123.mp4"
            output_path.write_text("fake video content")
            
            # Download
            url = "https://vimeo.com/123456789"
            metadata = downloader.download(url, tmpdir)
            
            # Verify
            assert metadata.file_path == output_path
            assert metadata.duration == 180
            assert metadata.resolution == "1280x720"
            assert metadata.format == "mp4"
            assert metadata.url == url
            assert metadata.title == "Vimeo Test Video"
    
    @patch('src.nprd.downloader.yt_dlp.YoutubeDL')
    def test_download_direct_url_success(self, mock_ytdl_class):
        """Test successful direct URL video download."""
        downloader = VideoDownloader()
        
        # Mock yt-dlp
        mock_ytdl = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        
        # Mock video info
        mock_info = {
            'id': 'direct_video',
            'ext': 'webm',
            'duration': 90,
            'width': 640,
            'height': 480,
            'title': 'Direct Video',
        }
        mock_ytdl.extract_info.return_value = mock_info
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the expected output file
            output_path = Path(tmpdir) / "direct_video.webm"
            output_path.write_text("fake video content")
            
            # Download
            url = "https://example.com/video.webm"
            metadata = downloader.download(url, tmpdir)
            
            # Verify
            assert metadata.file_path == output_path
            assert metadata.format == "webm"
    
    def test_download_invalid_url(self):
        """Test download with invalid URL."""
        downloader = VideoDownloader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValidationError, match="Invalid URL format"):
                downloader.download("not a valid url", tmpdir)
    
    @patch('src.nprd.downloader.yt_dlp.YoutubeDL')
    def test_download_unsupported_format_error(self, mock_ytdl_class):
        """Test download error handling for unsupported format."""
        downloader = VideoDownloader()
        
        # Mock yt-dlp
        mock_ytdl = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        
        # Mock video info
        mock_info = {
            'id': 'test_video',
            'ext': 'txt',  # Unsupported format
            'duration': 120,
            'width': 1920,
            'height': 1080,
            'title': 'Test Video',
        }
        mock_ytdl.extract_info.return_value = mock_info
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file with unsupported extension
            output_path = Path(tmpdir) / "test_video.txt"
            output_path.write_text("not a video")
            
            url = "https://youtube.com/watch?v=test_video"
            
            with pytest.raises(DownloadError, match="Unsupported video format"):
                downloader.download(url, tmpdir)
    
    @patch('src.nprd.downloader.yt_dlp.YoutubeDL')
    @patch('time.sleep')  # Mock sleep to speed up test
    def test_download_retry_on_transient_error(self, mock_sleep, mock_ytdl_class):
        """Test download retries on transient errors."""
        downloader = VideoDownloader(max_retries=3, base_delay=1.0)
        
        # Mock yt-dlp to fail twice then succeed
        mock_ytdl = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        
        # First two calls fail with transient error, third succeeds
        mock_ytdl.extract_info.side_effect = [
            Exception("Connection timeout"),
            Exception("Network error"),
            {
                'id': 'test_video',
                'ext': 'mp4',
                'duration': 120,
                'width': 1920,
                'height': 1080,
                'title': 'Test Video',
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the expected output file
            output_path = Path(tmpdir) / "test_video.mp4"
            output_path.write_text("fake video content")
            
            url = "https://youtube.com/watch?v=test_video"
            metadata = downloader.download(url, tmpdir)
            
            # Verify success after retries
            assert metadata.file_path == output_path
            
            # Verify retries occurred
            assert mock_ytdl.extract_info.call_count == 3
            assert mock_sleep.call_count == 2  # Slept between retries
    
    @patch('src.nprd.downloader.yt_dlp.YoutubeDL')
    def test_download_fail_on_permanent_error(self, mock_ytdl_class):
        """Test download fails immediately on permanent errors."""
        downloader = VideoDownloader(max_retries=3)
        
        # Mock yt-dlp to fail with permanent error
        mock_ytdl = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        mock_ytdl.extract_info.side_effect = Exception("HTTP 404 not found")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://youtube.com/watch?v=nonexistent"
            
            with pytest.raises(DownloadError, match="Permanent download error"):
                downloader.download(url, tmpdir)
            
            # Verify only one attempt (no retries for permanent errors)
            assert mock_ytdl.extract_info.call_count == 1
    
    @patch('src.nprd.downloader.yt_dlp.YoutubeDL')
    @patch('time.sleep')
    def test_download_exhausted_retries(self, mock_sleep, mock_ytdl_class):
        """Test download fails after exhausting all retries."""
        downloader = VideoDownloader(max_retries=3, base_delay=1.0)
        
        # Mock yt-dlp to always fail with transient error
        mock_ytdl = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        mock_ytdl.extract_info.side_effect = Exception("Connection timeout")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            url = "https://youtube.com/watch?v=test_video"
            
            with pytest.raises(DownloadError, match="Failed to download video after 3 attempts"):
                downloader.download(url, tmpdir)
            
            # Verify all retries were attempted
            assert mock_ytdl.extract_info.call_count == 3
            assert mock_sleep.call_count == 2  # Slept between retries
    
    @patch('src.nprd.downloader.yt_dlp.YoutubeDL')
    def test_get_video_info(self, mock_ytdl_class):
        """Test getting video info without downloading."""
        downloader = VideoDownloader()
        
        # Mock yt-dlp
        mock_ytdl = MagicMock()
        mock_ytdl_class.return_value.__enter__.return_value = mock_ytdl
        
        # Mock video info
        mock_info = {
            'title': 'Test Video',
            'duration': 120,
            'width': 1920,
            'height': 1080,
            'ext': 'mp4',
            'uploader': 'Test Channel',
            'upload_date': '20240115',
            'view_count': 1000,
            'description': 'Test description',
        }
        mock_ytdl.extract_info.return_value = mock_info
        
        url = "https://youtube.com/watch?v=test_video"
        info = downloader.get_video_info(url)
        
        # Verify
        assert info['title'] == 'Test Video'
        assert info['duration'] == 120
        assert info['resolution'] == '1920x1080'
        assert info['format'] == 'mp4'
        assert info['uploader'] == 'Test Channel'
        assert info['upload_date'] == '20240115'
        assert info['view_count'] == 1000
        assert info['description'] == 'Test description'
        
        # Verify download was not called
        mock_ytdl.download.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
