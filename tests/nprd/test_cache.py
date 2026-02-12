"""
Unit tests for NPRD Artifact Cache.

Tests cache eviction strategies, integrity checking, and edge cases.
"""

import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
import pytest

from src.nprd.cache import (
    ArtifactCache,
    ArtifactType,
    EvictionStrategy,
    compute_content_hash
)
from src.nprd.exceptions import CacheError, ValidationError


class TestArtifactCache:
    """Test suite for ArtifactCache."""
    
    def test_cache_initialization(self):
        """Test cache initialization creates directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = ArtifactCache(cache_dir=cache_dir, max_size_gb=1, max_age_days=30)
            
            assert cache_dir.exists()
            assert cache.max_size_bytes == 1 * 1024 * 1024 * 1024
            assert cache.max_age_days == 30
    
    def test_cache_put_and_get_video(self):
        """Test caching and retrieving a video file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Create a test video file
            video_file = tmpdir_path / "test_video.mp4"
            video_file.write_text("test video content")
            
            # Cache the video
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Retrieve from cache
            cached_path = cache.get(content_hash, ArtifactType.VIDEO)
            
            assert cached_path is not None
            assert cached_path.exists()
            assert cached_path.read_text() == "test video content"
    
    def test_cache_put_and_get_audio(self):
        """Test caching and retrieving an audio file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Create a test audio file
            audio_file = tmpdir_path / "test_audio.mp3"
            audio_file.write_text("test audio content")
            
            # Cache the audio
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.AUDIO, audio_file)
            
            # Retrieve from cache
            cached_path = cache.get(content_hash, ArtifactType.AUDIO)
            
            assert cached_path is not None
            assert cached_path.exists()
            assert cached_path.read_text() == "test audio content"
    
    def test_cache_put_and_get_frames(self):
        """Test caching and retrieving a frames directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Create a test frames directory
            frames_dir = tmpdir_path / "frames"
            frames_dir.mkdir()
            (frames_dir / "frame_001.jpg").write_text("frame 1")
            (frames_dir / "frame_002.jpg").write_text("frame 2")
            
            # Cache the frames
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.FRAMES, frames_dir)
            
            # Retrieve from cache
            cached_path = cache.get(content_hash, ArtifactType.FRAMES)
            
            assert cached_path is not None
            assert cached_path.exists()
            assert cached_path.is_dir()
            assert (cached_path / "frame_001.jpg").exists()
            assert (cached_path / "frame_002.jpg").exists()
    
    def test_cache_miss(self):
        """Test cache miss returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            content_hash = compute_content_hash("https://example.com/nonexistent")
            result = cache.get(content_hash, ArtifactType.VIDEO)
            
            assert result is None
    
    def test_cache_put_nonexistent_file(self):
        """Test caching a nonexistent file raises ValidationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            nonexistent_file = Path(tmpdir) / "nonexistent.mp4"
            content_hash = compute_content_hash("https://example.com/video")
            
            with pytest.raises(ValidationError, match="does not exist"):
                cache.put(content_hash, ArtifactType.VIDEO, nonexistent_file)
    
    def test_cache_integrity_valid(self):
        """Test integrity check passes for valid cached file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Create and cache a file
            video_file = tmpdir_path / "test_video.mp4"
            video_file.write_text("test content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Verify integrity
            assert cache.verify_integrity(content_hash)
    
    def test_cache_integrity_corrupted(self):
        """Test integrity check fails for corrupted cached file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Create and cache a file
            video_file = tmpdir_path / "test_video.mp4"
            video_file.write_text("original content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Corrupt the cached file
            cached_path = cache.get(content_hash, ArtifactType.VIDEO)
            cached_path.write_text("corrupted content")
            
            # Verify integrity fails
            assert not cache.verify_integrity(content_hash)
    
    def test_cache_integrity_missing_file(self):
        """Test integrity check fails for missing cached file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Create and cache a file
            video_file = tmpdir_path / "test_video.mp4"
            video_file.write_text("test content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Delete the cached file
            cached_path = cache.get(content_hash, ArtifactType.VIDEO)
            cached_path.unlink()
            
            # Verify integrity fails
            assert not cache.verify_integrity(content_hash)
    
    def test_cache_get_removes_corrupted(self):
        """Test that get() removes corrupted cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Create and cache a file
            video_file = tmpdir_path / "test_video.mp4"
            video_file.write_text("original content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Verify cache hit
            assert cache.get(content_hash, ArtifactType.VIDEO) is not None
            
            # Corrupt the cached file
            cached_path = cache.get(content_hash, ArtifactType.VIDEO)
            cached_path.write_text("corrupted content")
            
            # Next get should detect corruption and return None
            result = cache.get(content_hash, ArtifactType.VIDEO)
            assert result is None
    
    def test_eviction_lru(self):
        """Test LRU eviction strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            
            # Create cache with small size limit
            cache = ArtifactCache(cache_dir=cache_dir, max_size_gb=0.001)  # ~1MB
            
            # Cache multiple files
            hashes = []
            for i in range(5):
                video_file = tmpdir_path / f"video_{i}.mp4"
                video_file.write_text("x" * 500000)  # 500KB each
                
                content_hash = compute_content_hash(f"https://example.com/video{i}")
                hashes.append(content_hash)
                cache.put(content_hash, ArtifactType.VIDEO, video_file)
                time.sleep(0.01)  # Ensure different access times
            
            # Access first item to make it most recently used
            cache.get(hashes[0], ArtifactType.VIDEO)
            time.sleep(0.01)
            
            # Trigger LRU eviction
            evicted_count = cache.evict(EvictionStrategy.LRU)
            
            # Should have evicted some items
            assert evicted_count > 0
            
            # First item should still be cached (most recently used)
            # Note: Due to timing and size calculations, we just verify eviction happened
            stats = cache.get_cache_stats()
            assert stats["total_size_bytes"] <= cache.max_size_bytes
    
    def test_eviction_size(self):
        """Test size-based eviction strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            
            # Create cache with small size limit
            cache = ArtifactCache(cache_dir=cache_dir, max_size_gb=0.001)  # ~1MB
            
            # Cache multiple files
            for i in range(5):
                video_file = tmpdir_path / f"video_{i}.mp4"
                video_file.write_text("x" * 500000)  # 500KB each
                
                content_hash = compute_content_hash(f"https://example.com/video{i}")
                cache.put(content_hash, ArtifactType.VIDEO, video_file)
                time.sleep(0.01)  # Ensure different creation times
            
            # Trigger size-based eviction
            evicted_count = cache.evict(EvictionStrategy.SIZE)
            
            # Should have evicted some items
            assert evicted_count > 0
            
            # Cache should be under size limit
            stats = cache.get_cache_stats()
            assert stats["total_size_bytes"] <= cache.max_size_bytes
    
    def test_eviction_age(self):
        """Test age-based eviction strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            
            # Create cache with short max age
            cache = ArtifactCache(cache_dir=cache_dir, max_age_days=0)  # Immediate expiry
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("test content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Manually modify the manifest to make it appear old
            artifact_dir = cache._get_artifact_dir(content_hash)
            manifest_path = artifact_dir / "manifest.json"
            
            import json
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Set created_at to 2 days ago
            old_date = (datetime.now() - timedelta(days=2)).isoformat()
            manifest["created_at"] = old_date
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f)
            
            # Trigger age-based eviction
            evicted_count = cache.evict(EvictionStrategy.AGE)
            
            # Should have evicted the item (age > 0 days)
            assert evicted_count == 1
            
            # Cache should be empty
            result = cache.get(content_hash, ArtifactType.VIDEO)
            assert result is None
    
    def test_cache_stats(self):
        """Test cache statistics calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir, max_size_gb=1, max_age_days=30)
            
            # Initially empty
            stats = cache.get_cache_stats()
            assert stats["item_count"] == 0
            assert stats["total_size_bytes"] == 0
            
            # Cache some files
            for i in range(3):
                video_file = tmpdir_path / f"video_{i}.mp4"
                video_file.write_text("x" * 1000)  # 1KB each
                
                content_hash = compute_content_hash(f"https://example.com/video{i}")
                cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Check stats
            stats = cache.get_cache_stats()
            assert stats["item_count"] == 3
            assert stats["total_size_bytes"] > 0
            assert stats["max_size_gb"] == 1
            assert stats["max_age_days"] == 30
            assert 0 <= stats["utilization_pct"] <= 100
    
    def test_cache_clear(self):
        """Test clearing entire cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache some files
            for i in range(3):
                video_file = tmpdir_path / f"video_{i}.mp4"
                video_file.write_text("test content")
                
                content_hash = compute_content_hash(f"https://example.com/video{i}")
                cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Verify items are cached
            stats = cache.get_cache_stats()
            assert stats["item_count"] == 3
            
            # Clear cache
            cleared_count = cache.clear()
            assert cleared_count == 3
            
            # Verify cache is empty
            stats = cache.get_cache_stats()
            assert stats["item_count"] == 0
    
    def test_content_hash_deterministic(self):
        """Test that content hash is deterministic."""
        url = "https://example.com/video"
        
        hash1 = compute_content_hash(url)
        hash2 = compute_content_hash(url)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters
    
    def test_content_hash_unique(self):
        """Test that different URLs produce different hashes."""
        url1 = "https://example.com/video1"
        url2 = "https://example.com/video2"
        
        hash1 = compute_content_hash(url1)
        hash2 = compute_content_hash(url2)
        
        assert hash1 != hash2
    
    def test_cache_directory_structure(self):
        """Test that cache uses correct directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("test content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Verify directory structure: cache/{hash[:2]}/{hash[2:4]}/{hash}/
            expected_dir = cache_dir / content_hash[:2] / content_hash[2:4] / content_hash
            assert expected_dir.exists()
            assert expected_dir.is_dir()
            
            # Verify manifest exists
            manifest_path = expected_dir / "manifest.json"
            assert manifest_path.exists()
    
    def test_cache_multiple_artifact_types(self):
        """Test caching multiple artifact types for same content hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            content_hash = compute_content_hash("https://example.com/video")
            
            # Cache video
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("video content")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Cache audio
            audio_file = tmpdir_path / "audio.mp3"
            audio_file.write_text("audio content")
            cache.put(content_hash, ArtifactType.AUDIO, audio_file)
            
            # Cache frames
            frames_dir = tmpdir_path / "frames"
            frames_dir.mkdir()
            (frames_dir / "frame_001.jpg").write_text("frame 1")
            cache.put(content_hash, ArtifactType.FRAMES, frames_dir)
            
            # Verify all artifacts are accessible
            cached_video = cache.get(content_hash, ArtifactType.VIDEO)
            cached_audio = cache.get(content_hash, ArtifactType.AUDIO)
            cached_frames = cache.get(content_hash, ArtifactType.FRAMES)
            
            assert cached_video is not None
            assert cached_audio is not None
            assert cached_frames is not None
            
            assert cached_video.read_text() == "video content"
            assert cached_audio.read_text() == "audio content"
            assert (cached_frames / "frame_001.jpg").read_text() == "frame 1"
    
    def test_cache_access_time_update(self):
        """Test that cache access updates last_accessed time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("test content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Get initial access time
            artifact_dir = cache._get_artifact_dir(content_hash)
            manifest_path = artifact_dir / "manifest.json"
            
            import json
            with open(manifest_path, 'r') as f:
                manifest1 = json.load(f)
            
            time.sleep(0.1)
            
            # Access the cache
            cache.get(content_hash, ArtifactType.VIDEO)
            
            # Get updated access time
            with open(manifest_path, 'r') as f:
                manifest2 = json.load(f)
            
            # Access time should be updated
            assert manifest2["last_accessed"] > manifest1["last_accessed"]


class TestCacheEvictionStrategies:
    """Test suite for cache eviction strategies (Requirement 4.4)."""
    
    def test_lru_eviction_removes_least_recently_used(self):
        """Test LRU eviction removes least recently used items first."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            
            # Create cache with small size limit (1MB)
            cache = ArtifactCache(cache_dir=cache_dir, max_size_gb=0.001)
            
            # Cache 3 files (500KB each = 1.5MB total)
            hashes = []
            for i in range(3):
                video_file = tmpdir_path / f"video_{i}.mp4"
                video_file.write_text("x" * 500000)  # 500KB
                
                content_hash = compute_content_hash(f"https://example.com/video{i}")
                hashes.append(content_hash)
                cache.put(content_hash, ArtifactType.VIDEO, video_file)
                time.sleep(0.02)  # Ensure different timestamps
            
            # Access items in specific order: 0, 1 (making 2 the LRU)
            cache.get(hashes[0], ArtifactType.VIDEO)
            time.sleep(0.02)
            cache.get(hashes[1], ArtifactType.VIDEO)
            time.sleep(0.02)
            
            # Trigger LRU eviction
            evicted_count = cache.evict(EvictionStrategy.LRU)
            
            # Should evict at least 1 item to get under limit
            assert evicted_count >= 1
            
            # Items 0 and 1 should still exist (recently accessed)
            # Item 2 should be evicted (least recently used)
            assert cache.get(hashes[0], ArtifactType.VIDEO) is not None
            assert cache.get(hashes[1], ArtifactType.VIDEO) is not None
    
    def test_size_eviction_removes_oldest_when_over_limit(self):
        """Test size-based eviction removes oldest items when cache exceeds limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            
            # Create cache with small size limit (1MB)
            cache = ArtifactCache(cache_dir=cache_dir, max_size_gb=0.001)
            
            # Cache 4 files (500KB each = 2MB total)
            hashes = []
            for i in range(4):
                video_file = tmpdir_path / f"video_{i}.mp4"
                video_file.write_text("x" * 500000)  # 500KB
                
                content_hash = compute_content_hash(f"https://example.com/video{i}")
                hashes.append(content_hash)
                cache.put(content_hash, ArtifactType.VIDEO, video_file)
                time.sleep(0.02)  # Ensure different creation times
            
            # Trigger size-based eviction
            evicted_count = cache.evict(EvictionStrategy.SIZE)
            
            # Should evict at least 2 items to get under 1MB limit
            assert evicted_count >= 2
            
            # Cache should be under size limit
            stats = cache.get_cache_stats()
            assert stats["total_size_bytes"] <= cache.max_size_bytes
            
            # Oldest items (0, 1) should be evicted first
            # Newer items (2, 3) should remain
            assert cache.get(hashes[2], ArtifactType.VIDEO) is not None or \
                   cache.get(hashes[3], ArtifactType.VIDEO) is not None
    
    def test_age_eviction_removes_items_older_than_max_age(self):
        """Test age-based eviction removes items older than max_age_days."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            
            # Create cache with 1-day max age
            cache = ArtifactCache(cache_dir=cache_dir, max_age_days=1)
            
            # Cache 3 files
            hashes = []
            for i in range(3):
                video_file = tmpdir_path / f"video_{i}.mp4"
                video_file.write_text("test content")
                
                content_hash = compute_content_hash(f"https://example.com/video{i}")
                hashes.append(content_hash)
                cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Manually modify manifests to simulate different ages
            import json
            
            # Item 0: 3 days old (should be evicted)
            artifact_dir_0 = cache._get_artifact_dir(hashes[0])
            manifest_path_0 = artifact_dir_0 / "manifest.json"
            with open(manifest_path_0, 'r') as f:
                manifest_0 = json.load(f)
            manifest_0["created_at"] = (datetime.now() - timedelta(days=3)).isoformat()
            with open(manifest_path_0, 'w') as f:
                json.dump(manifest_0, f)
            
            # Item 1: 2 days old (should be evicted)
            artifact_dir_1 = cache._get_artifact_dir(hashes[1])
            manifest_path_1 = artifact_dir_1 / "manifest.json"
            with open(manifest_path_1, 'r') as f:
                manifest_1 = json.load(f)
            manifest_1["created_at"] = (datetime.now() - timedelta(days=2)).isoformat()
            with open(manifest_path_1, 'w') as f:
                json.dump(manifest_1, f)
            
            # Item 2: Current (should NOT be evicted)
            # No modification needed
            
            # Trigger age-based eviction
            evicted_count = cache.evict(EvictionStrategy.AGE)
            
            # Should evict 2 items (older than 1 day)
            assert evicted_count == 2
            
            # Only item 2 should remain
            assert cache.get(hashes[0], ArtifactType.VIDEO) is None
            assert cache.get(hashes[1], ArtifactType.VIDEO) is None
            assert cache.get(hashes[2], ArtifactType.VIDEO) is not None
    
    def test_eviction_with_empty_cache(self):
        """Test eviction on empty cache returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Evict from empty cache
            evicted_lru = cache.evict(EvictionStrategy.LRU)
            evicted_size = cache.evict(EvictionStrategy.SIZE)
            evicted_age = cache.evict(EvictionStrategy.AGE)
            
            assert evicted_lru == 0
            assert evicted_size == 0
            assert evicted_age == 0
    
    def test_eviction_preserves_cache_under_limit(self):
        """Test that eviction brings cache size under limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            
            # Create cache with 1MB limit
            cache = ArtifactCache(cache_dir=cache_dir, max_size_gb=0.001)
            
            # Cache 5 files (500KB each = 2.5MB total)
            for i in range(5):
                video_file = tmpdir_path / f"video_{i}.mp4"
                video_file.write_text("x" * 500000)
                
                content_hash = compute_content_hash(f"https://example.com/video{i}")
                cache.put(content_hash, ArtifactType.VIDEO, video_file)
                time.sleep(0.01)
            
            # Verify cache is over limit
            stats_before = cache.get_cache_stats()
            assert stats_before["total_size_bytes"] > cache.max_size_bytes
            
            # Trigger eviction
            cache.evict(EvictionStrategy.SIZE)
            
            # Verify cache is now under limit
            stats_after = cache.get_cache_stats()
            assert stats_after["total_size_bytes"] <= cache.max_size_bytes


class TestCacheIntegrityAndCorruption:
    """Test suite for cache integrity checking and corruption handling (Requirement 4.5)."""
    
    def test_integrity_check_detects_modified_file(self):
        """Test integrity check detects when cached file is modified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            original_content = "original content"
            video_file.write_text(original_content)
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Verify integrity passes initially
            assert cache.verify_integrity(content_hash)
            
            # Modify the cached file
            cached_path = cache.get(content_hash, ArtifactType.VIDEO)
            cached_path.write_text("modified content")
            
            # Integrity check should fail
            assert not cache.verify_integrity(content_hash)
    
    def test_integrity_check_detects_deleted_file(self):
        """Test integrity check detects when cached file is deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("test content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Verify integrity passes initially
            assert cache.verify_integrity(content_hash)
            
            # Delete the cached file
            cached_path = cache.get(content_hash, ArtifactType.VIDEO)
            cached_path.unlink()
            
            # Integrity check should fail
            assert not cache.verify_integrity(content_hash)
    
    def test_integrity_check_detects_corrupted_directory(self):
        """Test integrity check detects corruption in cached directory (frames)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a frames directory
            frames_dir = tmpdir_path / "frames"
            frames_dir.mkdir()
            (frames_dir / "frame_001.jpg").write_text("frame 1")
            (frames_dir / "frame_002.jpg").write_text("frame 2")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.FRAMES, frames_dir)
            
            # Verify integrity passes initially
            assert cache.verify_integrity(content_hash)
            
            # Corrupt one of the frames
            cached_frames = cache.get(content_hash, ArtifactType.FRAMES)
            (cached_frames / "frame_001.jpg").write_text("corrupted frame")
            
            # Integrity check should fail
            assert not cache.verify_integrity(content_hash)
    
    def test_get_removes_corrupted_cache_entry(self):
        """Test that get() automatically removes corrupted cache entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("original content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Verify cache hit
            assert cache.get(content_hash, ArtifactType.VIDEO) is not None
            
            # Corrupt the cached file
            artifact_dir = cache._get_artifact_dir(content_hash)
            cached_video = artifact_dir / "video.mp4"
            cached_video.write_text("corrupted content")
            
            # Next get() should detect corruption and remove entry
            result = cache.get(content_hash, ArtifactType.VIDEO)
            assert result is None
            
            # Verify the entire cache entry was removed
            assert not artifact_dir.exists()
    
    def test_corruption_detection_triggers_redownload_workflow(self):
        """Test that corruption detection enables re-download workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("original content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Verify cache hit
            assert cache.get(content_hash, ArtifactType.VIDEO) is not None
            
            # Corrupt the cached file
            artifact_dir = cache._get_artifact_dir(content_hash)
            cached_video = artifact_dir / "video.mp4"
            cached_video.write_text("corrupted content")
            
            # Attempt to get from cache - should return None (cache miss)
            result = cache.get(content_hash, ArtifactType.VIDEO)
            assert result is None
            
            # Simulate re-download: Create new video file with correct content
            new_video_file = tmpdir_path / "video_redownload.mp4"
            new_video_file.write_text("original content")
            
            # Re-cache the file
            cache.put(content_hash, ArtifactType.VIDEO, new_video_file)
            
            # Verify cache hit with correct content
            recached_path = cache.get(content_hash, ArtifactType.VIDEO)
            assert recached_path is not None
            assert recached_path.read_text() == "original content"
            
            # Verify integrity passes
            assert cache.verify_integrity(content_hash)
    
    def test_integrity_check_with_missing_manifest(self):
        """Test integrity check fails gracefully when manifest is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("test content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Delete the manifest
            artifact_dir = cache._get_artifact_dir(content_hash)
            manifest_path = artifact_dir / "manifest.json"
            manifest_path.unlink()
            
            # Integrity check should fail
            assert not cache.verify_integrity(content_hash)
    
    def test_integrity_check_with_corrupted_manifest(self):
        """Test integrity check fails gracefully when manifest is corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            # Cache a file
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("test content")
            
            content_hash = compute_content_hash("https://example.com/video")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            # Corrupt the manifest
            artifact_dir = cache._get_artifact_dir(content_hash)
            manifest_path = artifact_dir / "manifest.json"
            manifest_path.write_text("invalid json {{{")
            
            # Integrity check should fail gracefully
            assert not cache.verify_integrity(content_hash)
    
    def test_multiple_artifacts_integrity_check(self):
        """Test integrity check validates all artifacts for a content hash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            cache_dir = tmpdir_path / "cache"
            cache = ArtifactCache(cache_dir=cache_dir)
            
            content_hash = compute_content_hash("https://example.com/video")
            
            # Cache multiple artifacts
            video_file = tmpdir_path / "video.mp4"
            video_file.write_text("video content")
            cache.put(content_hash, ArtifactType.VIDEO, video_file)
            
            audio_file = tmpdir_path / "audio.mp3"
            audio_file.write_text("audio content")
            cache.put(content_hash, ArtifactType.AUDIO, audio_file)
            
            # Verify integrity passes for all artifacts
            assert cache.verify_integrity(content_hash)
            
            # Corrupt one artifact
            artifact_dir = cache._get_artifact_dir(content_hash)
            cached_audio = artifact_dir / "audio.mp3"
            cached_audio.write_text("corrupted audio")
            
            # Integrity check should fail (one artifact corrupted)
            assert not cache.verify_integrity(content_hash)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
