"""
SVSS Cache Package

Provides cache stability features:
- Jittered TTL to prevent thundering herd
- Probabilistic early refresh coordination
- Distributed locking for stale-while-revalidate pattern
"""

from svss.cache.cache_manager import SVSSCacheManager, CacheEntry
from svss.cache.lock_manager import DistributedLock, LockAcquisitionError

__all__ = [
    "SVSSCacheManager",
    "CacheEntry",
    "DistributedLock",
    "LockAcquisitionError",
]
