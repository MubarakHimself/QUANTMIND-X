"""
QuantMindX Temporal Decay

Time-based relevance scoring with:
- Exponential decay function
- Configurable half-life
- Boost recent memories
- Decay factor calculation
"""

import logging
from typing import Optional, List
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class DecayType(str, Enum):
    """Types of decay functions."""
    EXPONENTIAL = "exponential"  # Standard exponential decay
    LINEAR = "linear"  # Linear decay over time
    LOGARITHMIC = "logarithmic"  # Logarithmic decay
    STEP = "step"  # Step function decay


@dataclass
class DecayConfig:
    """
    Configuration for temporal decay.
    
    Attributes:
        decay_type: Type of decay function
        half_life: Time for importance to halve (default: 24 hours)
        decay_rate: Decay rate (lambda) for exponential decay
        min_score: Minimum score after decay (default: 0.01)
        boost_recent: Boost factor for recent memories (hours)
        boost_factor: Multiplier for recent memories
    """
    decay_type: DecayType = DecayType.EXPONENTIAL
    half_life: timedelta = timedelta(hours=24)
    decay_rate: Optional[float] = None  # Calculated from half_life if None
    min_score: float = 0.01
    boost_recent: Optional[timedelta] = None  # No boost if None
    boost_factor: float = 1.5
    
    def __post_init__(self):
        """Calculate decay rate from half-life if not set."""
        if self.decay_rate is None and self.half_life:
            # For exponential decay: lambda = ln(2) / half_life
            # Convert half_life to seconds
            half_life_seconds = self.half_life.total_seconds()
            self.decay_rate = 0.693147 / half_life_seconds  # ln(2)
    
    @classmethod
    def default(cls) -> "DecayConfig":
        """Create default decay config (24h half-life)."""
        return cls()
    
    @classmethod
    def fast(cls, half_life_hours: float = 6) -> "DecayConfig":
        """Create fast decay config."""
        return cls(half_life=timedelta(hours=half_life_hours))
    
    @classmethod
    def slow(cls, half_life_hours: float = 168) -> "DecayConfig":
        """Create slow decay config (1 week half-life)."""
        return cls(half_life=timedelta(hours=half_life_hours))


class TemporalDecay:
    """
    Time-based relevance scoring using exponential decay.
    
    Features:
    - Exponential decay with configurable half-life
    - Recent memory boost
    - Multiple decay types
    - Vectorized batch operations
    
    Example:
        >>> decay = TemporalDecay(
        ...     config=DecayConfig(half_life=timedelta(hours=24))
        ... )
        >>> 
        >>> # Calculate decay factor for a memory
        >>> factor = decay.calculate_decay_factor(
        ...     created_at=datetime.now(timezone.utc) - timedelta(hours=12)
        ... )
        >>> print(factor)  # ~0.71 (half decay)
        >>> 
        >>> # Apply decay to importance score
        >>> decayed_score = decay.apply_decay(
        ...     importance=0.8,
        ...     created_at=datetime.now(timezone.utc) - timedelta(hours=12)
        ... )
    """
    
    def __init__(
        self,
        config: Optional[DecayConfig] = None,
        reference_time: Optional[datetime] = None,
    ):
        """
        Initialize temporal decay calculator.
        
        Args:
            config: Decay configuration
            reference_time: Reference time for decay (default: now)
        """
        self.config = config or DecayConfig()
        self.reference_time = reference_time or datetime.now(timezone.utc)
        
        logger.info(
            f"TemporalDecay initialized: type={self.config.decay_type}, "
            f"half_life={self.config.half_life}"
        )
    
    def calculate_decay_factor(
        self,
        created_at: datetime,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate decay factor based on time elapsed.
        
        Args:
            created_at: Creation time of memory
            reference_time: Reference time (default: self.reference_time)
            
        Returns:
            Decay factor (0-1, where 1 is no decay)
        """
        ref = reference_time or self.reference_time
        
        # Ensure timezone awareness
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        
        # Calculate elapsed time
        elapsed = (ref - created_at).total_seconds()
        
        if elapsed <= 0:
            return 1.0  # Future or now, no decay
        
        # Calculate based on decay type
        if self.config.decay_type == DecayType.EXPONENTIAL:
            factor = self._exponential_decay(elapsed)
        elif self.config.decay_type == DecayType.LINEAR:
            factor = self._linear_decay(elapsed)
        elif self.config.decay_type == DecayType.LOGARITHMIC:
            factor = self._logarithmic_decay(elapsed)
        elif self.config.decay_type == DecayType.STEP:
            factor = self._step_decay(elapsed)
        else:
            factor = 1.0
        
        # Apply minimum
        return max(factor, self.config.min_score)
    
    def _exponential_decay(self, elapsed_seconds: float) -> float:
        """
        Exponential decay: exp(-lambda * t)
        
        Where lambda is decay rate and t is elapsed time.
        """
        return np.exp(-self.config.decay_rate * elapsed_seconds)
    
    def _linear_decay(self, elapsed_seconds: float) -> float:
        """
        Linear decay: max(0, 1 - t / half_life)
        """
        half_life_seconds = self.config.half_life.total_seconds()
        return max(0.0, 1.0 - elapsed_seconds / half_life_seconds)
    
    def _logarithmic_decay(self, elapsed_seconds: float) -> float:
        """
        Logarithmic decay: 1 / (1 + log(1 + t))
        """
        return 1.0 / (1.0 + np.log1p(elapsed_seconds / 3600))  # Hours
    
    def _step_decay(self, elapsed_seconds: float) -> float:
        """
        Step function decay: 1.0, 0.5, 0.25, ...
        
        Halves at each half-life interval.
        """
        half_life_seconds = self.config.half_life.total_seconds()
        steps = int(elapsed_seconds / half_life_seconds)
        return 1.0 / (2.0 ** steps)
    
    def apply_decay(
        self,
        importance: float,
        created_at: datetime,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Apply temporal decay to an importance score.
        
        Args:
            importance: Original importance score (0-1)
            created_at: Creation time
            reference_time: Reference time
            
        Returns:
            Decayed importance score
        """
        decay_factor = self.calculate_decay_factor(created_at, reference_time)
        
        # Apply recent boost
        boost = self._calculate_boost(created_at, reference_time)
        
        # Combine: importance * decay * boost
        decayed = importance * decay_factor * boost
        
        # Clamp to 0-1
        return max(0.0, min(1.0, decayed))
    
    def _calculate_boost(
        self,
        created_at: datetime,
        reference_time: Optional[datetime] = None,
    ) -> float:
        """
        Calculate boost factor for recent memories.
        
        Args:
            created_at: Creation time
            reference_time: Reference time
            
        Returns:
            Boost multiplier (1-boost_factor)
        """
        if self.config.boost_recent is None:
            return 1.0
        
        ref = reference_time or self.reference_time
        
        # Ensure timezone awareness
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        
        elapsed = ref - created_at
        
        if elapsed < self.config.boost_recent:
            # Linear boost: decreases from boost_factor to 1
            boost_seconds = self.config.boost_recent.total_seconds()
            elapsed_seconds = elapsed.total_seconds()
            
            # Boost factor ramps down linearly
            ratio = 1.0 - (elapsed_seconds / boost_seconds)
            boost = 1.0 + (self.config.boost_factor - 1.0) * ratio
            
            return boost
        
        return 1.0
    
    def calculate_batch_decay(
        self,
        importances: List[float],
        created_at_times: List[datetime],
        reference_time: Optional[datetime] = None,
    ) -> List[float]:
        """
        Calculate decayed scores for a batch of memories.
        
        Vectorized for efficiency with large batches.
        
        Args:
            importances: List of importance scores
            created_at_times: List of creation times
            reference_time: Reference time
            
        Returns:
            List of decayed scores
        """
        if len(importances) != len(created_at_times):
            raise ValueError("importances and created_at_times must have same length")
        
        ref = reference_time or self.reference_time
        
        # Calculate elapsed times in seconds
        elapsed_times = []
        for created_at in created_at_times:
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
            if ref.tzinfo is None:
                ref_local = ref.replace(tzinfo=timezone.utc)
            else:
                ref_local = ref
            
            elapsed = (ref_local - created_at).total_seconds()
            elapsed_times.append(max(0, elapsed))  # Clamp negative
        
        # Vectorized decay calculation
        if self.config.decay_type == DecayType.EXPONENTIAL:
            decay_factors = np.exp(-self.config.decay_rate * np.array(elapsed_times))
        elif self.config.decay_type == DecayType.LINEAR:
            half_life_sec = self.config.half_life.total_seconds()
            decay_factors = np.clip(
                1.0 - np.array(elapsed_times) / half_life_sec,
                0.0, 1.0
            )
        else:
            # Fallback to individual calculation
            decay_factors = np.array([
                self.calculate_decay_factor(created_at, ref)
                for created_at in created_at_times
            ])
        
        # Apply minimum
        decay_factors = np.maximum(decay_factors, self.config.min_score)
        
        # Apply recent boost
        if self.config.boost_recent is not None:
            boost_seconds = self.config.boost_recent.total_seconds()
            boosts = []
            
            for elapsed in elapsed_times:
                if elapsed < boost_seconds:
                    ratio = 1.0 - (elapsed / boost_seconds)
                    boost = 1.0 + (self.config.boost_factor - 1.0) * ratio
                    boosts.append(boost)
                else:
                    boosts.append(1.0)
            
            boosts = np.array(boosts)
        else:
            boosts = 1.0
        
        # Combine
        decayed = np.array(importances) * decay_factors * boosts
        
        # Clamp and return
        return [max(0.0, min(1.0, float(d))) for d in decayed]
    
    def get_halflife_remaining(
        self,
        created_at: datetime,
        reference_time: Optional[datetime] = None,
    ) -> Optional[float]:
        """
        Get remaining half-life percentage.
        
        Args:
            created_at: Creation time
            reference_time: Reference time
            
        Returns:
            Remaining half-life as percentage (0-100), or None if past all half-lives
        """
        decay_factor = self.calculate_decay_factor(created_at, reference_time)
        
        if decay_factor <= self.config.min_score:
            return None
        
        # Convert decay factor to percentage
        return decay_factor * 100


def create_decay_config(
    half_life_hours: float = 24,
    decay_type: DecayType = DecayType.EXPONENTIAL,
    boost_recent_hours: Optional[float] = None,
    boost_factor: float = 1.5,
) -> DecayConfig:
    """
    Create a decay configuration.
    
    Args:
        half_life_hours: Half-life in hours
        decay_type: Type of decay function
        boost_recent_hours: Boost recent memories (hours)
        boost_factor: Boost multiplier
        
    Returns:
        DecayConfig instance
    """
    config = DecayConfig(
        decay_type=decay_type,
        half_life=timedelta(hours=half_life_hours),
        boost_recent=timedelta(hours=boost_recent_hours) if boost_recent_hours else None,
        boost_factor=boost_factor,
    )
    
    return config


# Preset decay configurations

DECAY_FAST = DecayConfig(
    decay_type=DecayType.EXPONENTIAL,
    half_life=timedelta(hours=6),
    boost_recent=timedelta(hours=1),
    boost_factor=2.0,
)

DECAY_DEFAULT = DecayConfig(
    decay_type=DecayType.EXPONENTIAL,
    half_life=timedelta(hours=24),
    boost_recent=timedelta(hours=2),
    boost_factor=1.5,
)

DECAY_SLOW = DecayConfig(
    decay_type=DecayType.EXPONENTIAL,
    half_life=timedelta(hours=168),  # 1 week
    boost_recent=timedelta(hours=12),
    boost_factor=1.3,
)

DECAY_VERY_SLOW = DecayConfig(
    decay_type=DecayType.EXPONENTIAL,
    half_life=timedelta(hours=720),  # 30 days
    boost_recent=timedelta(hours=24),
    boost_factor=1.2,
)
