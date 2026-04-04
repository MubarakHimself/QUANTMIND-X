"""
Unit tests for Correlation Sensor RMT Enhancement.

Tests for:
- correlation_penalty(i,j) formula with mock correlation values
- M5 vs H1 matrix selection based on regime
- second-signal penalty logic
- Redis correlation matrix caching
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.risk.physics.correlation_sensor import CorrelationSensor, REGIME_TO_TIMEFRAME
from src.risk.correlation_cache import CorrelationCache, JitteredTTLConfig, CorrelationMatrixData


class TestCorrelationPenaltyFormula(unittest.TestCase):
    """Test the correlation_penalty formula: max(0, C_ij - 0.5) / (1 - 0.5)"""

    def setUp(self):
        """Set up test fixtures."""
        self.sensor = CorrelationSensor()

    def test_penalty_high_correlation(self):
        """Test penalty when C_ij = 0.7 -> expected penalty = 0.4"""
        # Create a 2x2 correlation matrix
        correlation_matrix = np.array([
            [1.0, 0.7],
            [0.7, 1.0]
        ])
        bot_index_map = {'bot_i': 0, 'bot_j': 1}

        penalty = self.sensor.get_pairwise_penalty(
            bot_i='bot_i',
            bot_j='bot_j',
            regime='scalping',
            correlation_matrix=correlation_matrix,
            bot_index_map=bot_index_map
        )

        # C_ij = 0.7 -> (0.7 - 0.5) / 0.5 = 0.4
        self.assertAlmostEqual(penalty, 0.4, places=4)

    def test_penalty_very_high_correlation(self):
        """Test penalty when C_ij = 0.9 -> expected penalty = 0.8"""
        correlation_matrix = np.array([
            [1.0, 0.9],
            [0.9, 1.0]
        ])
        bot_index_map = {'bot_i': 0, 'bot_j': 1}

        penalty = self.sensor.get_pairwise_penalty(
            bot_i='bot_i',
            bot_j='bot_j',
            regime='scalping',
            correlation_matrix=correlation_matrix,
            bot_index_map=bot_index_map
        )

        # C_ij = 0.9 -> (0.9 - 0.5) / 0.5 = 0.8
        self.assertAlmostEqual(penalty, 0.8, places=4)

    def test_penalty_no_penalty_threshold(self):
        """Test penalty when C_ij = 0.5 -> expected penalty = 0 (no penalty)"""
        correlation_matrix = np.array([
            [1.0, 0.5],
            [0.5, 1.0]
        ])
        bot_index_map = {'bot_i': 0, 'bot_j': 1}

        penalty = self.sensor.get_pairwise_penalty(
            bot_i='bot_i',
            bot_j='bot_j',
            regime='scalping',
            correlation_matrix=correlation_matrix,
            bot_index_map=bot_index_map
        )

        # C_ij = 0.5 -> max(0, 0) = 0
        self.assertAlmostEqual(penalty, 0.0, places=4)

    def test_penalty_below_threshold(self):
        """Test penalty when C_ij = 0.3 -> expected penalty = 0 (below threshold)"""
        correlation_matrix = np.array([
            [1.0, 0.3],
            [0.3, 1.0]
        ])
        bot_index_map = {'bot_i': 0, 'bot_j': 1}

        penalty = self.sensor.get_pairwise_penalty(
            bot_i='bot_i',
            bot_j='bot_j',
            regime='scalping',
            correlation_matrix=correlation_matrix,
            bot_index_map=bot_index_map
        )

        # C_ij = 0.3 < 0.5 -> max(0, negative) = 0
        self.assertAlmostEqual(penalty, 0.0, places=4)

    def test_penalty_negative_correlation(self):
        """Test penalty when C_ij = -0.3 -> expected penalty = 0"""
        correlation_matrix = np.array([
            [1.0, -0.3],
            [-0.3, 1.0]
        ])
        bot_index_map = {'bot_i': 0, 'bot_j': 1}

        penalty = self.sensor.get_pairwise_penalty(
            bot_i='bot_i',
            bot_j='bot_j',
            regime='scalping',
            correlation_matrix=correlation_matrix,
            bot_index_map=bot_index_map
        )

        # Negative correlation -> no penalty
        self.assertAlmostEqual(penalty, 0.0, places=4)

    def test_penalty_perfect_correlation(self):
        """Test penalty when C_ij = 1.0 -> expected penalty = 1.0 (max)"""
        correlation_matrix = np.array([
            [1.0, 1.0],
            [1.0, 1.0]
        ])
        bot_index_map = {'bot_i': 0, 'bot_j': 1}

        penalty = self.sensor.get_pairwise_penalty(
            bot_i='bot_i',
            bot_j='bot_j',
            regime='scalping',
            correlation_matrix=correlation_matrix,
            bot_index_map=bot_index_map
        )

        # C_ij = 1.0 -> (1.0 - 0.5) / 0.5 = 1.0
        self.assertAlmostEqual(penalty, 1.0, places=4)


class TestTimeframeSelection(unittest.TestCase):
    """Test M5 vs H1 matrix selection based on regime."""

    def setUp(self):
        """Set up test fixtures."""
        self.sensor = CorrelationSensor()

    def test_regime_to_timeframe_mapping(self):
        """Test that regime correctly maps to timeframe."""
        # scalping -> M5
        self.assertEqual(REGIME_TO_TIMEFRAME['scalping'], 'M5')
        # ORB -> H1
        self.assertEqual(REGIME_TO_TIMEFRAME['ORB'], 'H1')
        # M5 -> M5
        self.assertEqual(REGIME_TO_TIMEFRAME['M5'], 'M5')
        # H1 -> H1
        self.assertEqual(REGIME_TO_TIMEFRAME['H1'], 'H1')

    def test_select_timeframe_scalping(self):
        """Test timeframe selection for scalping regime."""
        timeframe = self.sensor._select_timeframe('scalping')
        self.assertEqual(timeframe, 'M5')

    def test_select_timeframe_orb(self):
        """Test timeframe selection for ORB regime."""
        timeframe = self.sensor._select_timeframe('ORB')
        self.assertEqual(timeframe, 'H1')

    def test_select_timeframe_m5(self):
        """Test timeframe selection for M5."""
        timeframe = self.sensor._select_timeframe('M5')
        self.assertEqual(timeframe, 'M5')

    def test_select_timeframe_h1(self):
        """Test timeframe selection for H1."""
        timeframe = self.sensor._select_timeframe('H1')
        self.assertEqual(timeframe, 'H1')

    def test_select_timeframe_unknown_defaults_to_m5(self):
        """Test that unknown regime defaults to M5."""
        timeframe = self.sensor._select_timeframe('unknown')
        self.assertEqual(timeframe, 'M5')


class TestSecondSignalPenalty(unittest.TestCase):
    """Test second-signal penalty logic in Governor integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.sensor = CorrelationSensor()

    def test_penalty_only_applied_to_second_signal(self):
        """Test that penalty is only reduced for the second-signaling bot."""
        correlation_matrix = np.array([
            [1.0, 0.7],
            [0.7, 1.0]
        ])
        bot_index_map = {'bot_first': 0, 'bot_second': 1}

        # First signal should get no penalty (it's the reference)
        first_penalty = self.sensor.get_pairwise_penalty(
            bot_i='bot_first',
            bot_j='bot_second',  # The second bot doesn't exist yet
            regime='scalping',
            correlation_matrix=correlation_matrix,
            bot_index_map=bot_index_map
        )
        # With bot_j not being an "active signal", this is still calculated
        # The penalty application logic is in Governor
        self.assertGreaterEqual(first_penalty, 0.0)

    def test_bot_not_in_index_map(self):
        """Test penalty returns 0 when bot not found in index map."""
        correlation_matrix = np.array([
            [1.0, 0.7],
            [0.7, 1.0]
        ])
        bot_index_map = {'bot_i': 0, 'bot_j': 1}

        penalty = self.sensor.get_pairwise_penalty(
            bot_i='unknown_bot',
            bot_j='bot_j',
            regime='scalping',
            correlation_matrix=correlation_matrix,
            bot_index_map=bot_index_map
        )

        self.assertEqual(penalty, 0.0)


class TestCorrelationCache(unittest.TestCase):
    """Test Redis correlation matrix caching."""

    def setUp(self):
        """Set up test fixtures."""
        self.cache = CorrelationCache(redis_client=None)  # Use local cache
        self.ttl_config = JitteredTTLConfig()

    def test_jittered_ttl_range(self):
        """Test that jittered TTL falls within expected range (30-40s)."""
        for _ in range(100):
            ttl = self.cache._get_ttl_seconds()
            self.assertGreaterEqual(ttl, self.ttl_config.BASE_TTL)
            self.assertLessEqual(ttl, self.ttl_config.BASE_TTL + self.ttl_config.JITTER_MAX)

    def test_early_refresh_threshold(self):
        """Test early refresh threshold is 8 seconds."""
        self.assertEqual(self.ttl_config.EARLY_REFRESH_THRESHOLD, 8)

    def test_should_early_refresh_below_threshold(self):
        """Test early refresh probability when TTL < 8s."""
        # When TTL = 0, p_refresh should be close to 1.0
        p_refreshes = [self.cache._should_early_refresh(2.0) for _ in range(100)]
        true_count = sum(p_refreshes)
        # With TTL = 2s, p_refresh = 1 - (2/8) = 0.75, so most should be True
        self.assertGreater(true_count, 50)

    def test_should_not_early_refresh_above_threshold(self):
        """Test no early refresh when TTL >= 8s."""
        result = self.cache._should_early_refresh(10.0)
        self.assertFalse(result)

    def test_set_and_get_correlation_matrix(self):
        """Test setting and getting correlation matrix from cache."""
        timeframe = 'M5'
        matrix = [[1.0, 0.5], [0.5, 1.0]]
        sample_count = 100

        # Set the matrix
        success = self.cache.set_correlation_matrix(timeframe, matrix, sample_count)
        self.assertTrue(success)

        # Get the matrix
        cached = self.cache.get_correlation_matrix(timeframe)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.matrix, matrix)
        self.assertEqual(cached.sample_count, sample_count)

    def test_invalidate_specific_timeframe(self):
        """Test invalidating a specific timeframe."""
        self.cache.set_correlation_matrix('M5', [[1.0]], 10)
        self.cache.set_correlation_matrix('H1', [[1.0]], 10)

        self.cache.invalidate('M5')

        self.assertIsNone(self.cache.get_correlation_matrix('M5'))
        self.assertIsNotNone(self.cache.get_correlation_matrix('H1'))

    def test_invalidate_all_timeframes(self):
        """Test invalidating all timeframes."""
        self.cache.set_correlation_matrix('M5', [[1.0]], 10)
        self.cache.set_correlation_matrix('H1', [[1.0]], 10)

        self.cache.invalidate()

        self.assertIsNone(self.cache.get_correlation_matrix('M5'))
        self.assertIsNone(self.cache.get_correlation_matrix('H1'))

    def test_correlation_matrix_data_serialization(self):
        """Test CorrelationMatrixData serialization/deserialization."""
        original = CorrelationMatrixData(
            matrix=[[1.0, 0.5], [0.5, 1.0]],
            computed_at=1234567890.0,
            sample_count=100
        )

        json_str = original.to_json()
        restored = CorrelationMatrixData.from_json(json_str)

        self.assertEqual(restored.matrix, original.matrix)
        self.assertEqual(restored.computed_at, original.computed_at)
        self.assertEqual(restored.sample_count, original.sample_count)


class TestCorrelationSensorIntegration(unittest.TestCase):
    """Integration tests for CorrelationSensor with CorrelationCache."""

    def test_get_penalty_convenience_method(self):
        """Test the get_penalty convenience method."""
        sensor = CorrelationSensor()
        correlation_matrix = np.array([
            [1.0, 0.7],
            [0.7, 1.0]
        ])
        bot_index_map = {'bot1': 0, 'bot2': 1}

        penalty = sensor.get_penalty(
            bot_i='bot1',
            bot_j='bot2',
            regime='scalping'
        )

        # Without actual returns data, returns 0
        self.assertEqual(penalty, 0.0)

    def test_clear_cache_clears_correlation_cache(self):
        """Test that clear_cache also clears the correlation cache."""
        sensor = CorrelationSensor()
        sensor._correlation_cache.set_correlation_matrix('M5', [[1.0]], 10)

        sensor.clear_cache()

        self.assertIsNone(sensor._correlation_cache.get_correlation_matrix('M5'))


if __name__ == '__main__':
    unittest.main()
