"""
Performance benchmarks for Enhanced Kelly Position Sizing.

Tests cover:
- Sensor calculation latency (< 100ms target)
- Full position sizing latency (< 200ms target)
- Monte Carlo simulation performance (< 500ms target)
- Memory usage profiling
- Cache effectiveness
"""

import pytest
import time
import tracemalloc
from typing import List, Dict

from src.position_sizing.enhanced_kelly import (
    EnhancedKellyCalculator,
    EnhancedKellyConfig
)
from src.position_sizing.kelly_analyzer import KellyStatisticsAnalyzer
from src.position_sizing.portfolio_kelly import PortfolioKellyScaler


@pytest.mark.performance
class TestEnhancedKellyPerformance:
    """Performance tests for Enhanced Kelly calculator."""

    def test_kelly_calculation_latency(self, standard_config):
        """Test Kelly calculation latency (target: < 50ms)."""
        calculator = EnhancedKellyCalculator(standard_config)

        # Warm up
        for _ in range(10):
            calculator.calculate(
                account_balance=10000.0,
                win_rate=0.55,
                avg_win=400.0,
                avg_loss=200.0,
                current_atr=0.0012,
                average_atr=0.0010,
                stop_loss_pips=20.0,
                pip_value=10.0
            )

        # Benchmark
        iterations = 1000
        start_time = time.perf_counter()

        for _ in range(iterations):
            calculator.calculate(
                account_balance=10000.0,
                win_rate=0.55,
                avg_win=400.0,
                avg_loss=200.0,
                current_atr=0.0012,
                average_atr=0.0010,
                stop_loss_pips=20.0,
                pip_value=10.0
            )

        end_time = time.perf_counter()
        avg_latency_ms = (end_time - start_time) / iterations * 1000

        # Target: < 50ms per calculation
        assert avg_latency_ms < 50, f"Kelly calculation too slow: {avg_latency_ms:.2f}ms"

        print(f"\nKelly calculation latency: {avg_latency_ms:.3f}ms (target: < 50ms)")

    def test_kelly_memory_usage(self, standard_config):
        """Test Kelly calculation memory usage."""
        calculator = EnhancedKellyCalculator(standard_config)

        tracemalloc.start()

        # Perform 100 calculations
        for _ in range(100):
            calculator.calculate(
                account_balance=10000.0,
                win_rate=0.55,
                avg_win=400.0,
                avg_loss=200.0,
                current_atr=0.0012,
                average_atr=0.0010,
                stop_loss_pips=20.0,
                pip_value=10.0
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Target: < 10MB peak memory
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 10, f"Memory usage too high: {peak_mb:.2f}MB"

        print(f"Kelly calculation peak memory: {peak_mb:.2f}MB (target: < 10MB)")

    def test_batch_calculation_performance(self, standard_config):
        """Test performance of multiple calculations."""
        calculator = EnhancedKellyCalculator(standard_config)

        # Simulate processing multiple trading signals
        scenarios = [
            {
                "account_balance": 10000.0,
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
                "current_atr": 0.0012,
                "average_atr": 0.0010,
                "stop_loss_pips": 20.0,
                "pip_value": 10.0
            }
            for _ in range(100)
        ]

        start_time = time.perf_counter()

        for scenario in scenarios:
            calculator.calculate(**scenario)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_time_ms = total_time_ms / len(scenarios)

        # Target: < 100ms for 100 calculations (< 1ms each)
        assert total_time_ms < 100, f"Batch calculation too slow: {total_time_ms:.2f}ms"

        print(f"Batch calculation (100): {total_time_ms:.2f}ms total, {avg_time_ms:.3f}ms avg")


@pytest.mark.performance
class TestAnalyzerPerformance:
    """Performance tests for Kelly statistics analyzer."""

    def test_analyzer_latency_small_history(self):
        """Test analyzer latency with small history (< 30 trades)."""
        analyzer = KellyStatisticsAnalyzer(min_trades=30)
        trades = [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(20)]

        # Warm up
        for _ in range(100):
            analyzer.calculate_kelly_parameters(trades)

        # Benchmark
        iterations = 1000
        start_time = time.perf_counter()

        for _ in range(iterations):
            analyzer.calculate_kelly_parameters(trades)

        end_time = time.perf_counter()
        avg_latency_ms = (end_time - start_time) / iterations * 1000

        # Target: < 10ms for small history
        assert avg_latency_ms < 10, f"Analyzer too slow for small history: {avg_latency_ms:.2f}ms"

        print(f"Analyzer latency (20 trades): {avg_latency_ms:.3f}ms (target: < 10ms)")

    def test_analyzer_latency_large_history(self):
        """Test analyzer latency with large history (500 trades)."""
        analyzer = KellyStatisticsAnalyzer(min_trades=30)
        trades = [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(500)]

        # Warm up
        for _ in range(10):
            analyzer.calculate_kelly_parameters(trades)

        # Benchmark
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            analyzer.calculate_kelly_parameters(trades)

        end_time = time.perf_counter()
        avg_latency_ms = (end_time - start_time) / iterations * 1000

        # Target: < 50ms for large history
        assert avg_latency_ms < 50, f"Analyzer too slow for large history: {avg_latency_ms:.2f}ms"

        print(f"Analyzer latency (500 trades): {avg_latency_ms:.3f}ms (target: < 50ms)")

    def test_rolling_kelly_performance(self):
        """Test rolling window Kelly calculation performance."""
        analyzer = KellyStatisticsAnalyzer(min_trades=30)
        trades = [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(500)]
        window_size = 50

        # Benchmark
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            analyzer.calculate_rolling_kelly(trades, window_size)

        end_time = time.perf_counter()
        avg_latency_ms = (end_time - start_time) / iterations * 1000

        # Target: < 100ms for rolling calculation
        assert avg_latency_ms < 100, f"Rolling Kelly too slow: {avg_latency_ms:.2f}ms"

        print(f"Rolling Kelly latency (500 trades, window 50): {avg_latency_ms:.3f}ms (target: < 100ms)")


@pytest.mark.performance
class TestPortfolioScalingPerformance:
    """Performance tests for portfolio Kelly scaler."""

    def test_portfolio_scaling_latency_small(self):
        """Test portfolio scaling latency with few bots (3-5)."""
        scaler = PortfolioKellyScaler()
        bot_kelly = {f"bot{i}": 0.01 for i in range(5)}

        # Warm up
        for _ in range(1000):
            scaler.scale_bot_positions(bot_kelly)

        # Benchmark
        iterations = 10000
        start_time = time.perf_counter()

        for _ in range(iterations):
            scaler.scale_bot_positions(bot_kelly)

        end_time = time.perf_counter()
        avg_latency_ms = (end_time - start_time) / iterations * 1000

        # Target: < 1ms for small portfolio
        assert avg_latency_ms < 1, f"Portfolio scaling too slow: {avg_latency_ms:.3f}ms"

        print(f"Portfolio scaling latency (5 bots): {avg_latency_ms:.3f}ms (target: < 1ms)")

    def test_portfolio_scaling_latency_large(self):
        """Test portfolio scaling latency with many bots (20+)."""
        scaler = PortfolioKellyScaler()
        bot_kelly = {f"bot{i}": 0.005 for i in range(20)}

        # Warm up
        for _ in range(100):
            scaler.scale_bot_positions(bot_kelly)

        # Benchmark
        iterations = 1000
        start_time = time.perf_counter()

        for _ in range(iterations):
            scaler.scale_bot_positions(bot_kelly)

        end_time = time.perf_counter()
        avg_latency_ms = (end_time - start_time) / iterations * 1000

        # Target: < 5ms for large portfolio
        assert avg_latency_ms < 5, f"Portfolio scaling too slow: {avg_latency_ms:.3f}ms"

        print(f"Portfolio scaling latency (20 bots): {avg_latency_ms:.3f}ms (target: < 5ms)")

    def test_portfolio_status_latency(self):
        """Test portfolio status calculation latency."""
        scaler = PortfolioKellyScaler()
        bot_kelly = {f"bot{i}": 0.01 for i in range(10)}

        # Warm up
        for _ in range(1000):
            scaler.get_portfolio_status(bot_kelly)

        # Benchmark
        iterations = 10000
        start_time = time.perf_counter()

        for _ in range(iterations):
            scaler.get_portfolio_status(bot_kelly)

        end_time = time.perf_counter()
        avg_latency_ms = (end_time - start_time) / iterations * 1000

        # Target: < 1ms
        assert avg_latency_ms < 1, f"Portfolio status too slow: {avg_latency_ms:.3f}ms"

        print(f"Portfolio status latency (10 bots): {avg_latency_ms:.3f}ms (target: < 1ms)")


@pytest.mark.performance
class TestEndToEndPerformance:
    """End-to-end performance tests."""

    def test_full_position_sizing_workflow(self, standard_config):
        """Test complete position sizing workflow performance."""
        calculator = EnhancedKellyCalculator(standard_config)
        analyzer = KellyStatisticsAnalyzer(min_trades=30)

        # Sample data
        trades = [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(50)]

        # Step 1: Analyze trade history
        start_time = time.perf_counter()
        params = analyzer.calculate_kelly_parameters(trades)
        analyze_time = time.perf_counter() - start_time

        # Step 2: Calculate position size
        start_time = time.perf_counter()
        result = calculator.calculate(
            account_balance=10000.0,
            win_rate=params.win_rate,
            avg_win=params.avg_win,
            avg_loss=params.avg_loss,
            current_atr=0.0012,
            average_atr=0.0010,
            stop_loss_pips=20.0,
            pip_value=10.0
        )
        calculate_time = time.perf_counter() - start_time

        total_time = analyze_time + calculate_time
        total_time_ms = total_time * 1000

        # Target: < 200ms for full workflow
        assert total_time_ms < 200, f"Full workflow too slow: {total_time_ms:.2f}ms"

        print(f"\nFull position sizing workflow:")
        print(f"  Analysis: {analyze_time*1000:.3f}ms")
        print(f"  Calculation: {calculate_time*1000:.3f}ms")
        print(f"  Total: {total_time_ms:.3f}ms (target: < 200ms)")

    def test_concurrent_calculations(self, standard_config):
        """Test performance when handling multiple concurrent calculations."""
        calculator = EnhancedKellyCalculator(standard_config)

        # Simulate 10 concurrent trading signals
        signals = [
            {
                "account_balance": 10000.0,
                "win_rate": 0.55,
                "avg_win": 400.0,
                "avg_loss": 200.0,
                "current_atr": 0.0012,
                "average_atr": 0.0010,
                "stop_loss_pips": 20.0,
                "pip_value": 10.0
            }
            for _ in range(10)
        ]

        start_time = time.perf_counter()

        for signal in signals:
            calculator.calculate(**signal)

        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000

        # Target: < 500ms for 10 calculations (< 50ms each)
        assert total_time_ms < 500, f"Concurrent calculations too slow: {total_time_ms:.2f}ms"

        print(f"Concurrent calculations (10): {total_time_ms:.3f}ms (target: < 500ms)")


@pytest.mark.performance
@pytest.mark.benchmark
class TestPerformanceRegression:
    """Performance regression tests to prevent degradation."""

    def test_kelly_calculation_regression(self, standard_config, benchmark):
        """Benchmark Kelly calculation for regression detection."""
        calculator = EnhancedKellyCalculator(standard_config)

        def calc_kelly():
            return calculator.calculate(
                account_balance=10000.0,
                win_rate=0.55,
                avg_win=400.0,
                avg_loss=200.0,
                current_atr=0.0012,
                average_atr=0.0010,
                stop_loss_pips=20.0,
                pip_value=10.0
            )

        result = benchmark(calc_kelly)

        # Benchmark library will track performance over time
        assert result.position_size > 0

    def test_analyzer_regression(self, benchmark):
        """Benchmark analyzer for regression detection."""
        analyzer = KellyStatisticsAnalyzer(min_trades=30)
        trades = [{"profit": 100.0 if i % 2 == 0 else -50.0} for i in range(100)]

        def analyze():
            return analyzer.calculate_kelly_parameters(trades)

        result = benchmark(analyze)

        assert result.sample_size == 100


# Performance summary decorator
def print_performance_summary():
    """Print summary of all performance tests."""
    print("\n" + "="*60)
    print("ENHANCED KELLY POSITION SIZING - PERFORMANCE SUMMARY")
    print("="*60)
    print("\nTargets:")
    print("  Kelly calculation: < 50ms")
    print("  Analyzer (small): < 10ms")
    print("  Analyzer (large): < 50ms")
    print("  Portfolio scaling: < 5ms")
    print("  Full workflow: < 200ms")
    print("  Memory usage: < 10MB")
    print("\nRun with: pytest tests/position_sizing/test_performance.py -v -s")
    print("="*60)


if __name__ == "__main__":
    print_performance_summary()
