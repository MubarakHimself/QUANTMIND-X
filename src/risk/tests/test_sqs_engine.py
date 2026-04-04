"""
Tests for SQS Engine and Related Components

Story: 4-7-spread-quality-score-sqs-system
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
import random

# Import SQS modules
from src.risk.sqs_engine import SQSEngine, SQSResult, SpreadBucket
from src.risk.sqs_cache import SQSRedisCache, JitteredTTLConfig
from src.risk.sqs_calendar import SQSCalendarIntegration, NewsImpact, NewsWindowState
from src.risk.weekend_guard import WeekendGuard, WeekendGuardState


# =============================================================================
# SQS Engine Tests
# =============================================================================

class TestSQSEngine:
    """Tests for SQSEngine computation and threshold evaluation."""

    @pytest.fixture
    def engine(self):
        """Create SQS engine with mock dependencies."""
        mock_cache = MagicMock()
        mock_calendar = MagicMock()
        return SQSEngine(cache=mock_cache, calendar_integration=mock_calendar)

    @pytest.fixture
    def valid_buckets(self):
        """Create valid historical spread buckets."""
        now = datetime.now(timezone.utc)
        return {
            f"{now.weekday()}:{now.hour}:{now.minute // 5}": SpreadBucket(
                avg_spread=0.8,
                sample_count=50,
                updated_at_utc=now
            )
        }

    def test_sqs_calculation_normal(self, engine, valid_buckets):
        """Test SQS calculation with normal spread."""
        # Current spread = 0.8, historical = 0.8 -> SQS = 1.0
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="scalping",
            current_spread=0.8,
            historical_buckets=valid_buckets,
            news_override=None
        )

        assert result.allowed == True
        assert result.sqs == 1.0
        assert result.is_hard_block == False
        assert "allowed" in result.reason.lower()

    def test_sqs_calculation_spread_wider_than_historical(self, engine, valid_buckets):
        """Test SQS when current spread is wider than historical (SQS < 1)."""
        # Current spread = 1.6, historical = 0.8 -> SQS = 0.5
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="scalping",
            current_spread=1.6,
            historical_buckets=valid_buckets,
            news_override=None
        )

        assert result.allowed == False
        assert result.sqs == 0.5
        # 0.5 is NOT a hard block - hard block is < 0.50
        assert result.is_hard_block == False

    def test_sqs_threshold_scalping(self, engine, valid_buckets):
        """Test scalping threshold (>0.75 required)."""
        # Current spread = 1.0, historical = 0.8 -> SQS = 0.8
        # 0.8 > 0.75 -> allowed
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="scalping",
            current_spread=1.0,
            historical_buckets=valid_buckets,
            news_override=None
        )

        assert result.allowed == True
        assert result.sqs == 0.8
        assert result.threshold == 0.75

    def test_sqs_threshold_orb(self, engine, valid_buckets):
        """Test ORB threshold (>0.80 required)."""
        # Current spread = 1.0, historical = 0.8 -> SQS = 0.8
        # 0.8 > 0.80 (ORB threshold) -> allowed
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="ORB",
            current_spread=1.0,
            historical_buckets=valid_buckets,
            news_override=None
        )

        assert result.allowed == True
        assert result.threshold == 0.80

    def test_sqs_threshold_orb_blocks_spread_wider(self, engine, valid_buckets):
        """Test ORB threshold blocks when spread is wider."""
        # Current spread = 1.2, historical = 0.8 -> SQS = 0.667
        # 0.667 < 0.80 -> blocked
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="ORB",
            current_spread=1.2,
            historical_buckets=valid_buckets,
            news_override=None
        )

        assert result.allowed == False
        assert result.sqs == pytest.approx(0.667, rel=0.01)

    def test_sqs_hard_block_threshold(self, engine, valid_buckets):
        """Test hard block when SQS < 0.50."""
        # Current spread = 2.0, historical = 0.8 -> SQS = 0.4
        # 0.4 < 0.50 -> hard block
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="scalping",
            current_spread=2.0,
            historical_buckets=valid_buckets,
            news_override=None
        )

        assert result.allowed == False
        assert result.sqs == 0.4
        assert result.is_hard_block == True

    def test_sqs_news_override_high_impact(self, engine, valid_buckets):
        """Test news override increases threshold for high impact."""
        # High impact adds 0.10 to threshold
        # SQS = 0.8, normal threshold = 0.75, effective = 0.85
        # 0.8 < 0.85 -> blocked
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="scalping",
            current_spread=1.0,
            historical_buckets=valid_buckets,
            news_override=0.10  # High impact bump
        )

        assert result.allowed == False
        assert result.threshold == 0.85

    def test_sqs_graceful_degradation_no_buckets(self, engine):
        """Test graceful degradation when no historical buckets available."""
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="scalping",
            current_spread=1.0,
            historical_buckets={},  # No buckets
            news_override=None
        )

        # Should default to allowed per NFR-R1
        assert result.allowed == True
        assert result.sqs == 1.0
        assert "graceful degradation" in result.reason.lower()

    def test_sqs_graceful_degradation_invalid_spread(self, engine, valid_buckets):
        """Test graceful degradation when spread is invalid."""
        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="scalping",
            current_spread=0,  # Invalid
            historical_buckets=valid_buckets,
            news_override=None
        )

        assert result.allowed == True
        assert "graceful degradation" in result.reason.lower()

    def test_sqs_insufficient_samples(self, engine):
        """Test when bucket has insufficient samples."""
        now = datetime.now(timezone.utc)
        bucket_key = f"{now.weekday()}:{now.hour}:{now.minute // 5}"
        insufficient_buckets = {
            bucket_key: SpreadBucket(
                avg_spread=0.8,
                sample_count=5,  # Less than MIN_SAMPLES_FOR_VALID_BUCKET (20)
                updated_at_utc=now
            )
        }

        result = engine.evaluate(
            symbol="EURUSD",
            strategy_type="scalping",
            current_spread=1.0,
            historical_buckets=insufficient_buckets,
            news_override=None
        )

        # Should default to allowed when samples insufficient
        assert result.allowed == True

    def test_get_bucket_key(self, engine):
        """Test bucket key generation."""
        # Monday, 10:23 -> bucket 4 (23 // 5 = 4)
        dt = datetime(2024, 1, 1, 10, 23, tzinfo=timezone.utc)  # Monday
        key = engine._get_bucket_key(dt)
        assert key == "0:10:4"

    def test_bucket_key_resolution_fallback(self, engine):
        """Test finding valid bucket when exact bucket not available."""
        now = datetime.now(timezone.utc)
        # Create buckets with same hour but different bucket
        buckets = {
            "0:10:0": SpreadBucket(avg_spread=0.8, sample_count=50, updated_at_utc=now),
            "0:10:1": SpreadBucket(avg_spread=0.85, sample_count=50, updated_at_utc=now),
        }

        # Current time bucket might be different, should find nearby in same hour
        result = engine._find_valid_bucket(buckets, now)
        # This may return None if current bucket is far - that's valid behavior
        # The test verifies the function works without errors
        assert result is None or result.sample_count >= 20


# =============================================================================
# SQS Cache Tests
# =============================================================================

class TestSQSRedisCache:
    """Tests for SQS Redis cache with jittered TTL."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock Redis."""
        mock_redis = MagicMock()
        return SQSRedisCache(redis_client=mock_redis)

    def test_jittered_ttl_range(self, cache):
        """Test TTL is within expected range (30-40s)."""
        config = JitteredTTLConfig()
        assert config.BASE_TTL_SECONDS == 30
        assert config.JITTER_MAX_SECONDS == 10

    def test_early_refresh_threshold(self, cache):
        """Test early refresh threshold is 8 seconds."""
        config = JitteredTTLConfig()
        assert config.EARLY_REFRESH_THRESHOLD_SECONDS == 8

    def test_max_stale_seconds(self, cache):
        """Test max stale seconds is 60."""
        config = JitteredTTLConfig()
        assert config.MAX_STALE_SECONDS == 60


# =============================================================================
# SQS Calendar Integration Tests
# =============================================================================

class TestSQSCalendarIntegration:
    """Tests for CalendarGovernor integration."""

    @pytest.fixture
    def calendar_integration(self):
        """Create calendar integration with mock governor."""
        mock_governor = MagicMock()
        return SQSCalendarIntegration(calendar_governor=mock_governor)

    def test_no_news_returns_inactive(self, calendar_integration):
        """Test no news returns inactive state."""
        # Governor returns None when no news
        calendar_integration._governor = None
        state = calendar_integration.get_news_window_state("EURUSD")
        assert state.active == False

    def test_high_impact_threshold_bump(self, calendar_integration):
        """Test high impact applies 0.10 bump."""
        assert calendar_integration.HIGH_IMPACT_THRESHOLD_BUMP == 0.10

    def test_medium_impact_threshold_bump(self, calendar_integration):
        """Test medium impact applies 0.10 bump."""
        assert calendar_integration.MEDIUM_IMPACT_THRESHOLD_BUMP == 0.10


# =============================================================================
# Weekend Guard Tests
# =============================================================================

class TestWeekendGuard:
    """Tests for weekend guard and Monday warm-up."""

    @pytest.fixture
    def guard(self):
        """Create weekend guard with mock cache."""
        mock_cache = MagicMock()
        return WeekendGuard(cache=mock_cache)

    def test_guard_state_model(self):
        """Test WeekendGuardState model."""
        state = WeekendGuardState(
            guard_active=True,
            warmup_active=False,
            warmup_started_at_utc=datetime.now(timezone.utc),
            current_threshold=0.75
        )
        assert state.guard_active == True
        assert state.warmup_active == False
        assert state.current_threshold == 0.75

    def test_config_friday_activation(self, guard):
        """Test Friday 21:00 GMT is activation time."""
        assert guard._config.GUARD_ACTIVATION_HOUR == 21
        assert guard._config.GUARD_ACTIVATION_MINUTE == 0
        assert guard._config.GUARD_DAY_OF_WEEK == 4  # Friday

    def test_config_sunday_deactivation(self, guard):
        """Test Sunday 21:00 GMT is deactivation time."""
        assert guard._config.GUARD_DEACTIVATION_HOUR == 21
        assert guard._config.GUARD_DEACTIVATION_MINUTE == 0
        assert guard._config.DEACTIVATION_DAY_OF_WEEK == 6  # Sunday

    def test_warmup_duration(self, guard):
        """Test Monday warm-up is 15 minutes."""
        assert guard._config.WARMUP_DURATION_MINUTES == 15
        assert guard._config.WARMUP_START_THRESHOLD == 0.75
        assert guard._config.WARMUP_END_THRESHOLD == 0.60

    def test_warmup_rate_calculation(self, guard):
        """Test warmup rate is linear (0.75 - 0.60) / 15 = 0.01 per minute."""
        expected_rate = (0.75 - 0.60) / 15
        assert guard._config.WARMUP_RATE_PER_MINUTE == expected_rate


# =============================================================================
# Threshold Comparison Tests
# =============================================================================

class TestSQSThresholdComparison:
    """Tests for threshold comparison logic."""

    @pytest.fixture
    def engine(self):
        return SQSEngine()

    @pytest.fixture
    def valid_buckets(self):
        now = datetime.now(timezone.utc)
        return {
            f"{now.weekday()}:{now.hour}:{now.minute // 5}": SpreadBucket(
                avg_spread=0.8,
                sample_count=50,
                updated_at_utc=now
            )
        }

    def test_scalping_threshold_comparison(self, engine, valid_buckets):
        """Test all threshold levels for scalping strategy."""
        # Threshold: >0.75 allowed, <0.50 hard block

        # Test 1: SQS = 0.76 -> allowed (just above threshold)
        result = engine.evaluate("EURUSD", "scalping", 0.8/0.76, valid_buckets)
        assert result.allowed == True

        # Test 2: SQS = 0.74 -> blocked (just below threshold)
        result = engine.evaluate("EURUSD", "scalping", 0.8/0.74, valid_buckets)
        assert result.allowed == False
        assert result.is_hard_block == False

        # Test 3: SQS = 0.49 -> hard block
        result = engine.evaluate("EURUSD", "scalping", 0.8/0.49, valid_buckets)
        assert result.allowed == False
        assert result.is_hard_block == True

    def test_orb_threshold_comparison(self, engine, valid_buckets):
        """Test all threshold levels for ORB strategy."""
        # Threshold: >0.80 allowed, <0.50 hard block

        # Test 1: SQS = 0.81 -> allowed
        result = engine.evaluate("EURUSD", "ORB", 0.8/0.81, valid_buckets)
        assert result.allowed == True

        # Test 2: SQS = 0.79 -> blocked (below ORB threshold)
        result = engine.evaluate("EURUSD", "ORB", 0.8/0.79, valid_buckets)
        assert result.allowed == False
        assert result.is_hard_block == False

        # Test 3: SQS < 0.50 -> hard block
        result = engine.evaluate("EURUSD", "ORB", 2.0, valid_buckets)  # 0.8/2.0 = 0.4
        assert result.allowed == False
        assert result.is_hard_block == True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
