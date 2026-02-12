"""
Artifact Cache with content-based addressing.

Caches downloaded videos and extracted artifacts to avoid redundant processing.
Uses SHA-256 hashing for content-based identifiers.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum

from src.nprd.exceptions import CacheError, ValidationError


logger = logging.getLogger(__name__)


class ArtifactType(str, Enum):
    """Types of cached artifacts."""
    VIDEO = "video"
    AUDIO = "audio"
    FRAMES = "frames"
    MANIFEST = "manifest"


class EvictionStrategy(str, Enum):
    """Cache eviction strategies."""
    LRU = "lru"  # Least Recently Used
    SIZE = "size"  # Size-based (oldest when over limit)
    AGE = "age"  # Age-based (older than max age)


class ArtifactCache:
    """
    Cache for video artifacts with content-based addressing.
    
    Storage structure: cache/{hash[:2]}/{hash[2:4]}/{hash}/
    
    Features:
    - Content-based addressing using SHA-256
    - Multiple artifact types: video, audio, frames, manifest
    - Eviction strategies: LRU, size-based, age-based
    - Integrity checking with checksums
    """
    
    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: int = 50,
        max_age_days: int = 30
    ):
        """
        Initialize artifact cache.
        
        Args:
            cache_dir: Root directory for cache storage
            max_size_gb: Maximum cache size in GB (default: 50)
            max_age_days: Maximum age for cached items in days (default: 30)
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.max_age_days = max_age_days
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized artifact cache at {self.cache_dir} "
            f"(max_size: {max_size_gb}GB, max_age: {max_age_days} days)"
        )
    
    def get(self, content_hash: str, artifact_type: ArtifactType) -> Optional[Path]:
        """
        Get cached artifact by content hash.
        
        Args:
            content_hash: SHA-256 hash of content
            artifact_type: Type of artifact to retrieve
            
        Returns:
            Path to cached artifact if exists and valid, None otherwise
        """
        artifact_dir = self._get_artifact_dir(content_hash)
        
        if not artifact_dir.exists():
            logger.debug(f"Cache miss: {content_hash} ({artifact_type})")
            return None
        
        # Get artifact path based on type
        artifact_path = self._get_artifact_path(artifact_dir, artifact_type)
        
        if not artifact_path or not artifact_path.exists():
            logger.debug(f"Cache miss: {content_hash} ({artifact_type}) - artifact not found")
            return None
        
        # Verify integrity
        if not self.verify_integrity(content_hash):
            logger.warning(f"Cache integrity check failed for {content_hash}, removing from cache")
            self._remove_artifact(content_hash)
            return None
        
        # Update access time for LRU
        self._update_access_time(artifact_dir)
        
        logger.debug(f"Cache hit: {content_hash} ({artifact_type})")
        return artifact_path
    
    def put(
        self,
        content_hash: str,
        artifact_type: ArtifactType,
        file_path: Path
    ) -> None:
        """
        Store artifact in cache.
        
        Args:
            content_hash: SHA-256 hash of content
            artifact_type: Type of artifact
            file_path: Path to file to cache
            
        Raises:
            CacheError: If caching fails
            ValidationError: If file doesn't exist
        """
        if not file_path.exists():
            raise ValidationError(f"File to cache does not exist: {file_path}")
        
        # Create artifact directory
        artifact_dir = self._get_artifact_dir(content_hash)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine destination path
        dest_path = self._get_cache_path(artifact_dir, artifact_type, file_path)
        
        try:
            # Copy file to cache
            if file_path.is_dir():
                # For frames directory
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(file_path, dest_path)
            else:
                # For single files
                shutil.copy2(file_path, dest_path)
            
            # Store checksum
            self._store_checksum(artifact_dir, artifact_type, file_path)
            
            # Update manifest
            self._update_manifest(artifact_dir, artifact_type, dest_path)
            
            logger.info(f"Cached artifact: {content_hash} ({artifact_type})")
            
        except Exception as e:
            # Clean up on failure
            if dest_path.exists():
                if dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()
            raise CacheError(f"Failed to cache artifact: {str(e)}")
    
    def verify_integrity(self, content_hash: str) -> bool:
        """
        Verify cached file integrity using checksum.
        
        Args:
            content_hash: SHA-256 hash of content
            
        Returns:
            True if all cached artifacts are valid
        """
        artifact_dir = self._get_artifact_dir(content_hash)
        
        if not artifact_dir.exists():
            return False
        
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            return False
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Verify each artifact
            for artifact_type, artifact_info in manifest.get("artifacts", {}).items():
                artifact_path = Path(artifact_info["path"])
                stored_checksum = artifact_info.get("checksum")
                
                if not artifact_path.exists():
                    logger.warning(f"Artifact missing: {artifact_path}")
                    return False
                
                if stored_checksum:
                    # Calculate current checksum
                    current_checksum = self._calculate_checksum(artifact_path)
                    if current_checksum != stored_checksum:
                        logger.warning(
                            f"Checksum mismatch for {artifact_path}: "
                            f"expected {stored_checksum}, got {current_checksum}"
                        )
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying integrity: {str(e)}")
            return False
    
    def evict(self, strategy: EvictionStrategy) -> int:
        """
        Evict cached items based on strategy.
        
        Args:
            strategy: Eviction strategy to use
            
        Returns:
            Number of items evicted
        """
        if strategy == EvictionStrategy.LRU:
            return self._evict_lru()
        elif strategy == EvictionStrategy.SIZE:
            return self._evict_by_size()
        elif strategy == EvictionStrategy.AGE:
            return self._evict_by_age()
        else:
            raise ValueError(f"Unknown eviction strategy: {strategy}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_size = 0
        item_count = 0
        oldest_item = None
        newest_item = None
        
        for hash_dir in self._iter_cache_items():
            manifest_path = hash_dir / "manifest.json"
            if manifest_path.exists():
                item_count += 1
                
                # Calculate size
                for item in hash_dir.rglob("*"):
                    if item.is_file():
                        total_size += item.stat().st_size
                
                # Track oldest/newest
                created_at = datetime.fromtimestamp(manifest_path.stat().st_ctime)
                if oldest_item is None or created_at < oldest_item:
                    oldest_item = created_at
                if newest_item is None or created_at > newest_item:
                    newest_item = created_at
        
        return {
            "total_size_bytes": total_size,
            "total_size_gb": total_size / (1024 ** 3),
            "item_count": item_count,
            "max_size_gb": self.max_size_bytes / (1024 ** 3),
            "max_age_days": self.max_age_days,
            "oldest_item": oldest_item.isoformat() if oldest_item else None,
            "newest_item": newest_item.isoformat() if newest_item else None,
            "utilization_pct": (total_size / self.max_size_bytes * 100) if self.max_size_bytes > 0 else 0,
        }
    
    def clear(self) -> int:
        """
        Clear entire cache.
        
        Returns:
            Number of items removed
        """
        count = 0
        for hash_dir in self._iter_cache_items():
            try:
                shutil.rmtree(hash_dir)
                count += 1
            except Exception as e:
                logger.error(f"Failed to remove {hash_dir}: {str(e)}")
        
        logger.info(f"Cleared cache: removed {count} items")
        return count
    
    # Private methods
    
    def _get_artifact_dir(self, content_hash: str) -> Path:
        """
        Get artifact directory path using content-based addressing.
        
        Structure: cache/{hash[:2]}/{hash[2:4]}/{hash}/
        
        Args:
            content_hash: SHA-256 hash
            
        Returns:
            Path to artifact directory
        """
        if len(content_hash) < 4:
            raise ValueError(f"Invalid content hash: {content_hash}")
        
        return self.cache_dir / content_hash[:2] / content_hash[2:4] / content_hash
    
    def _get_artifact_path(
        self,
        artifact_dir: Path,
        artifact_type: ArtifactType
    ) -> Optional[Path]:
        """
        Get path to specific artifact from manifest.
        
        Args:
            artifact_dir: Artifact directory
            artifact_type: Type of artifact
            
        Returns:
            Path to artifact if exists
        """
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            artifact_info = manifest.get("artifacts", {}).get(artifact_type.value)
            if artifact_info:
                return Path(artifact_info["path"])
            
            return None
            
        except Exception as e:
            logger.error(f"Error reading manifest: {str(e)}")
            return None
    
    def _get_cache_path(
        self,
        artifact_dir: Path,
        artifact_type: ArtifactType,
        source_path: Path
    ) -> Path:
        """
        Get destination path for caching artifact.
        
        Args:
            artifact_dir: Artifact directory
            artifact_type: Type of artifact
            source_path: Source file path
            
        Returns:
            Destination path in cache
        """
        if artifact_type == ArtifactType.VIDEO:
            return artifact_dir / f"video{source_path.suffix}"
        elif artifact_type == ArtifactType.AUDIO:
            return artifact_dir / f"audio{source_path.suffix}"
        elif artifact_type == ArtifactType.FRAMES:
            return artifact_dir / "frames"
        elif artifact_type == ArtifactType.MANIFEST:
            return artifact_dir / "manifest.json"
        else:
            return artifact_dir / source_path.name
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """
        Calculate SHA-256 checksum of file or directory.
        
        Args:
            file_path: Path to file or directory
            
        Returns:
            SHA-256 checksum as hex string
        """
        sha256 = hashlib.sha256()
        
        if file_path.is_dir():
            # For directories, hash all files in sorted order
            for item in sorted(file_path.rglob("*")):
                if item.is_file():
                    with open(item, 'rb') as f:
                        for chunk in iter(lambda: f.read(8192), b''):
                            sha256.update(chunk)
        else:
            # For single files
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _store_checksum(
        self,
        artifact_dir: Path,
        artifact_type: ArtifactType,
        file_path: Path
    ) -> None:
        """
        Store checksum for artifact.
        
        Args:
            artifact_dir: Artifact directory
            artifact_type: Type of artifact
            file_path: Path to file
        """
        checksum = self._calculate_checksum(file_path)
        checksum_file = artifact_dir / f"{artifact_type.value}.checksum"
        
        with open(checksum_file, 'w') as f:
            f.write(checksum)
    
    def _update_manifest(
        self,
        artifact_dir: Path,
        artifact_type: ArtifactType,
        artifact_path: Path
    ) -> None:
        """
        Update manifest with artifact information.
        
        Args:
            artifact_dir: Artifact directory
            artifact_type: Type of artifact
            artifact_path: Path to cached artifact
        """
        manifest_path = artifact_dir / "manifest.json"
        
        # Load existing manifest or create new
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            manifest = {
                "created_at": datetime.now().isoformat(),
                "artifacts": {}
            }
        
        # Update artifact info
        checksum_file = artifact_dir / f"{artifact_type.value}.checksum"
        checksum = None
        if checksum_file.exists():
            with open(checksum_file, 'r') as f:
                checksum = f.read().strip()
        
        manifest["artifacts"][artifact_type.value] = {
            "path": str(artifact_path),
            "checksum": checksum,
            "cached_at": datetime.now().isoformat(),
            "size_bytes": self._get_size(artifact_path),
        }
        
        manifest["last_accessed"] = datetime.now().isoformat()
        
        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
    
    def _get_size(self, path: Path) -> int:
        """
        Get size of file or directory in bytes.
        
        Args:
            path: Path to file or directory
            
        Returns:
            Size in bytes
        """
        if path.is_dir():
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            return path.stat().st_size
    
    def _update_access_time(self, artifact_dir: Path) -> None:
        """
        Update last access time in manifest.
        
        Args:
            artifact_dir: Artifact directory
        """
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.exists():
            return
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            manifest["last_accessed"] = datetime.now().isoformat()
            
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update access time: {str(e)}")
    
    def _remove_artifact(self, content_hash: str) -> None:
        """
        Remove artifact from cache.
        
        Args:
            content_hash: SHA-256 hash of content
        """
        artifact_dir = self._get_artifact_dir(content_hash)
        if artifact_dir.exists():
            try:
                shutil.rmtree(artifact_dir)
                logger.info(f"Removed artifact from cache: {content_hash}")
            except Exception as e:
                logger.error(f"Failed to remove artifact: {str(e)}")
    
    def _iter_cache_items(self) -> List[Path]:
        """
        Iterate over all cached items.
        
        Yields:
            Path to each artifact directory
        """
        items = []
        
        # Iterate through cache structure: {hash[:2]}/{hash[2:4]}/{hash}/
        for level1 in self.cache_dir.iterdir():
            if not level1.is_dir() or len(level1.name) != 2:
                continue
            
            for level2 in level1.iterdir():
                if not level2.is_dir() or len(level2.name) != 2:
                    continue
                
                for hash_dir in level2.iterdir():
                    if hash_dir.is_dir():
                        items.append(hash_dir)
        
        return items
    
    def _evict_lru(self) -> int:
        """
        Evict least recently used items until under size limit.
        
        Returns:
            Number of items evicted
        """
        # Get all items with access times
        items = []
        for hash_dir in self._iter_cache_items():
            manifest_path = hash_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    
                    last_accessed = manifest.get("last_accessed", manifest.get("created_at"))
                    if last_accessed:
                        access_time = datetime.fromisoformat(last_accessed)
                        size = sum(f.stat().st_size for f in hash_dir.rglob("*") if f.is_file())
                        items.append((access_time, size, hash_dir))
                except Exception as e:
                    logger.error(f"Error reading manifest for {hash_dir}: {str(e)}")
        
        # Sort by access time (oldest first)
        items.sort(key=lambda x: x[0])
        
        # Calculate current size
        current_size = sum(item[1] for item in items)
        
        # Evict until under limit
        evicted = 0
        for access_time, size, hash_dir in items:
            if current_size <= self.max_size_bytes:
                break
            
            try:
                shutil.rmtree(hash_dir)
                current_size -= size
                evicted += 1
                logger.info(f"Evicted (LRU): {hash_dir.name} (last accessed: {access_time})")
            except Exception as e:
                logger.error(f"Failed to evict {hash_dir}: {str(e)}")
        
        return evicted
    
    def _evict_by_size(self) -> int:
        """
        Evict oldest items when cache exceeds size limit.
        
        Returns:
            Number of items evicted
        """
        # Get all items with creation times
        items = []
        for hash_dir in self._iter_cache_items():
            manifest_path = hash_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    
                    created_at = manifest.get("created_at")
                    if created_at:
                        creation_time = datetime.fromisoformat(created_at)
                        size = sum(f.stat().st_size for f in hash_dir.rglob("*") if f.is_file())
                        items.append((creation_time, size, hash_dir))
                except Exception as e:
                    logger.error(f"Error reading manifest for {hash_dir}: {str(e)}")
        
        # Sort by creation time (oldest first)
        items.sort(key=lambda x: x[0])
        
        # Calculate current size
        current_size = sum(item[1] for item in items)
        
        # Evict until under limit
        evicted = 0
        for creation_time, size, hash_dir in items:
            if current_size <= self.max_size_bytes:
                break
            
            try:
                shutil.rmtree(hash_dir)
                current_size -= size
                evicted += 1
                logger.info(f"Evicted (size): {hash_dir.name} (created: {creation_time})")
            except Exception as e:
                logger.error(f"Failed to evict {hash_dir}: {str(e)}")
        
        return evicted
    
    def _evict_by_age(self) -> int:
        """
        Evict items older than max age.
        
        Returns:
            Number of items evicted
        """
        now = datetime.now()
        evicted = 0
        
        for hash_dir in self._iter_cache_items():
            manifest_path = hash_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    
                    created_at = manifest.get("created_at")
                    if created_at:
                        creation_time = datetime.fromisoformat(created_at)
                        age_days = (now - creation_time).days
                        
                        if age_days > self.max_age_days:
                            shutil.rmtree(hash_dir)
                            evicted += 1
                            logger.info(f"Evicted (age): {hash_dir.name} (age: {age_days} days)")
                            
                except Exception as e:
                    logger.error(f"Error processing {hash_dir}: {str(e)}")
        
        return evicted


def compute_content_hash(url: str) -> str:
    """
    Compute SHA-256 hash of URL for content-based addressing.
    
    Args:
        url: Video URL
        
    Returns:
        SHA-256 hash as hex string
    """
    return hashlib.sha256(url.encode('utf-8')).hexdigest()
