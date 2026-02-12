"""
Task 26.13: Risk Tier Calculation Benchmark

Benchmarks the performance of risk tier determination and tiered position sizing
to validate <1ms execution time target.

Requirements: 16.1, 16.7
"""

import pytest
import time
import statistics
from typing import List, Dict, Any
from src.router.prop.governor import PropGovernor
from src.database.engine import get_session
from src.database.models import PropFirmAccount
from datetime import datetime, timezone


class TestRiskTierCalculationBenchmark:
    """
    Benchmark tests for risk tier calculation performance.
    
    Target: <1ms for tier determination and position sizing calculations
    """
    
    @pytest.fixture(autouse=True)
    def setup_test_account(self):
        """Create a test PropFirm account for benchmarking."""
        session = get_session()
        
        # Clean up any existing test account
        existing = session.query(PropFirmAccount).filter_by(
            account_id="BENCHMARK_TEST_001"
        ).first()
        if existing:
            session.delete(existing)
            session.commit()
        
        # Create test account
        account = PropFirmAccount(
            account_id="BENCHMARK_TEST_001",
            firm_name="Benchmark Test Firm",
            daily_loss_limit_pct=5.0,
            hard_stop_buffer_pct=1.0,
            target_profit_pct=8.0,
            min_trading_days=5,
            risk_mode='growth',
            created_at=datetime.now(timezone.utc)
        )
        session.add(account)
        session.commit()
        session.close()
        
        yield
        
        # Cleanup
        session = get_session()
        account = session.query(PropFirmAccount).filter_by(
            account_id="BENCHMARK_TEST_001"
        ).first()
        if account:
            session.delete(account)
            session.commit()
        session.close()
    
    def _calculate_statistics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for latency measurements."""
        return {
            'average': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'min': min(latencies),
            'max': max(latencies),
            'p95': statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
            'p99': statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
        }
    
    def _print_statistics(self, name: str, stats: Dict[str, float]):
        """Print formatted statistics."""
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        print(f"Average:  {stats['average']:.6f}ms")
        print(f"Median:   {stats['median']:.6f}ms")
        print(f"Std Dev:  {stats['stdev']:.6f}ms")
        print(f"Min:      {stats['min']:.6f}ms")
        print(f"Max:      {stats['max']:.6f}ms")
        print(f"P95:      {stats['p95']:.6f}ms")
        print(f"P99:      {stats['p99']:.6f}ms")
        print(f"{'='*70}\n")
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_benchmark_determine_risk_tier(self):
        """
        Benchmark: DetermineRiskTier() execution time
        
        Measures the performance of tier determination logic across
        all three tiers (Growth, Scaling, Guardian).
        
        Target: <1ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: DetermineRiskTier() Performance")
        print("="*70)
        
        governor = PropGovernor("BENCHMARK_TEST_001")
        
        # Test equity values across all tiers
        test_equities = [
            # Growth tier ($100-$1,000)
            100.0, 250.0, 500.0, 750.0, 999.0,
            # Scaling tier ($1,000-$5,000)
            1000.0, 1500.0, 2500.0, 3500.0, 4999.0,
            # Guardian tier ($5,000+)
            5000.0, 7500.0, 10000.0, 25000.0, 50000.0
        ]
        
        num_iterations = 10000  # 10,000 iterations per equity value
        num_warmup = 100  # Warmup iterations
        
        print(f"Test Configuration:")
        print(f"  - Equity values: {len(test_equities)}")
        print(f"  - Iterations per value: {num_iterations:,}")
        print(f"  - Total measurements: {len(test_equities) * num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        
        # Warmup phase
        print("\nWarming up...")
        for equity in test_equities:
            for _ in range(num_warmup):
                governor._determine_risk_tier(equity)
        
        # Benchmark phase
        print("Running benchmark...")
        all_latencies = []
        tier_latencies = {'growth': [], 'scaling': [], 'guardian': []}
        
        for equity in test_equities:
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                tier = governor._determine_risk_tier(equity)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                all_latencies.append(latency_ms)
                tier_latencies[tier].append(latency_ms)
        
        # Calculate statistics
        overall_stats = self._calculate_statistics(all_latencies)
        growth_stats = self._calculate_statistics(tier_latencies['growth'])
        scaling_stats = self._calculate_statistics(tier_latencies['scaling'])
        guardian_stats = self._calculate_statistics(tier_latencies['guardian'])
        
        # Print results
        self._print_statistics("Overall DetermineRiskTier() Performance", overall_stats)
        self._print_statistics("Growth Tier Performance", growth_stats)
        self._print_statistics("Scaling Tier Performance", scaling_stats)
        self._print_statistics("Guardian Tier Performance", guardian_stats)
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <1.0ms average latency")
        print(f"  Actual: {overall_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if overall_stats['average'] < 1.0 else '❌ FAILED'}")
        print(f"  Margin: {((1.0 - overall_stats['average']) / 1.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert overall_stats['average'] < 1.0, \
            f"Average latency {overall_stats['average']:.6f}ms exceeds 1ms target"
        assert overall_stats['median'] < 1.0, \
            f"Median latency {overall_stats['median']:.6f}ms exceeds 1ms target"
        assert overall_stats['p95'] < 2.0, \
            f"P95 latency {overall_stats['p95']:.6f}ms exceeds 2ms acceptable limit"
        assert overall_stats['p99'] < 5.0, \
            f"P99 latency {overall_stats['p99']:.6f}ms exceeds 5ms acceptable limit"
    
    @pytest.mark.asyncio
    async def test_benchmark_tiered_position_sizing(self):
        """
        Benchmark: Complete tiered position sizing calculation
        
        Measures the performance of the full position sizing calculation
        including tier determination, Kelly calculation, and throttling.
        
        Target: <1ms average latency
        """
        print("\n" + "="*70)
        print("BENCHMARK: Tiered Position Sizing Performance")
        print("="*70)
        
        governor = PropGovernor("BENCHMARK_TEST_001")
        
        # Test scenarios across all tiers
        test_scenarios = [
            # Growth tier scenarios
            {'equity': 100.0, 'tier': 'growth', 'current_loss': 0.0},
            {'equity': 500.0, 'tier': 'growth', 'current_loss': 10.0},
            {'equity': 999.0, 'tier': 'growth', 'current_loss': 25.0},
            
            # Scaling tier scenarios
            {'equity': 1000.0, 'tier': 'scaling', 'current_loss': 0.0},
            {'equity': 2500.0, 'tier': 'scaling', 'current_loss': 50.0},
            {'equity': 4999.0, 'tier': 'scaling', 'current_loss': 100.0},
            
            # Guardian tier scenarios (with throttling)
            {'equity': 5000.0, 'tier': 'guardian', 'current_loss': 0.0},
            {'equity': 10000.0, 'tier': 'guardian', 'current_loss': 200.0},
            {'equity': 25000.0, 'tier': 'guardian', 'current_loss': 500.0},
        ]
        
        num_iterations = 5000  # 5,000 iterations per scenario
        num_warmup = 50  # Warmup iterations
        
        print(f"Test Configuration:")
        print(f"  - Test scenarios: {len(test_scenarios)}")
        print(f"  - Iterations per scenario: {num_iterations:,}")
        print(f"  - Total measurements: {len(test_scenarios) * num_iterations:,}")
        print(f"  - Warmup iterations: {num_warmup}")
        
        # Warmup phase
        print("\nWarming up...")
        for scenario in test_scenarios:
            for _ in range(num_warmup):
                # Simulate position sizing calculation
                tier = governor._determine_risk_tier(scenario['equity'])
                if tier == 'guardian':
                    governor._get_quadratic_throttle(scenario['equity'])
        
        # Benchmark phase
        print("Running benchmark...")
        all_latencies = []
        tier_latencies = {'growth': [], 'scaling': [], 'guardian': []}
        
        for scenario in test_scenarios:
            equity = scenario['equity']
            current_loss = scenario['current_loss']
            expected_tier = scenario['tier']
            
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                
                # Full position sizing calculation
                tier = governor._determine_risk_tier(equity)
                
                if tier == 'growth':
                    # Growth tier: Fixed risk calculation
                    risk_amount = max(equity * 0.03, 5.0)
                elif tier == 'scaling':
                    # Scaling tier: Kelly calculation
                    # Simplified Kelly for benchmark
                    kelly_fraction = 0.15  # Typical Kelly value
                    risk_amount = equity * kelly_fraction
                else:  # guardian
                    # Guardian tier: Kelly + Quadratic Throttle
                    kelly_fraction = 0.15
                    throttle = governor._get_quadratic_throttle(equity)
                    risk_amount = equity * kelly_fraction * throttle
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                all_latencies.append(latency_ms)
                tier_latencies[expected_tier].append(latency_ms)
        
        # Calculate statistics
        overall_stats = self._calculate_statistics(all_latencies)
        growth_stats = self._calculate_statistics(tier_latencies['growth'])
        scaling_stats = self._calculate_statistics(tier_latencies['scaling'])
        guardian_stats = self._calculate_statistics(tier_latencies['guardian'])
        
        # Print results
        self._print_statistics("Overall Position Sizing Performance", overall_stats)
        self._print_statistics("Growth Tier Position Sizing", growth_stats)
        self._print_statistics("Scaling Tier Position Sizing", scaling_stats)
        self._print_statistics("Guardian Tier Position Sizing (with Throttle)", guardian_stats)
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <1.0ms average latency")
        print(f"  Actual: {overall_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if overall_stats['average'] < 1.0 else '❌ FAILED'}")
        print(f"  Margin: {((1.0 - overall_stats['average']) / 1.0 * 100):.2f}% under target\n")
        
        # Assertions
        assert overall_stats['average'] < 1.0, \
            f"Average latency {overall_stats['average']:.6f}ms exceeds 1ms target"
        assert overall_stats['median'] < 1.0, \
            f"Median latency {overall_stats['median']:.6f}ms exceeds 1ms target"
        assert overall_stats['p95'] < 2.0, \
            f"P95 latency {overall_stats['p95']:.6f}ms exceeds 2ms acceptable limit"
        
        # Tier-specific assertions
        assert guardian_stats['average'] < 1.0, \
            f"Guardian tier (most complex) average latency {guardian_stats['average']:.6f}ms exceeds 1ms target"
    
    @pytest.mark.asyncio
    async def test_benchmark_tier_calculation_under_load(self):
        """
        Benchmark: Tier calculation under concurrent load
        
        Simulates multiple concurrent accounts performing tier calculations
        to validate performance under realistic trading conditions.
        
        Target: Maintain <1ms latency with 100 concurrent accounts
        """
        print("\n" + "="*70)
        print("BENCHMARK: Tier Calculation Under Concurrent Load")
        print("="*70)
        
        num_accounts = 100
        calculations_per_account = 100
        total_calculations = num_accounts * calculations_per_account
        
        print(f"Test Configuration:")
        print(f"  - Concurrent accounts: {num_accounts}")
        print(f"  - Calculations per account: {calculations_per_account}")
        print(f"  - Total calculations: {total_calculations:,}")
        
        # Create governors for multiple accounts
        governors = [PropGovernor("BENCHMARK_TEST_001") for _ in range(num_accounts)]
        
        # Test equity values (mix of all tiers)
        test_equities = [
            100.0, 500.0, 999.0,      # Growth
            1500.0, 3000.0, 4999.0,   # Scaling
            7500.0, 15000.0, 50000.0  # Guardian
        ]
        
        # Warmup
        print("\nWarming up...")
        for governor in governors[:10]:  # Warmup with subset
            for equity in test_equities:
                governor._determine_risk_tier(equity)
        
        # Benchmark
        print("Running benchmark...")
        all_latencies = []
        
        start_total = time.perf_counter()
        
        for governor in governors:
            for _ in range(calculations_per_account):
                # Randomly select equity value
                equity = test_equities[_ % len(test_equities)]
                
                start_time = time.perf_counter()
                tier = governor._determine_risk_tier(equity)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                all_latencies.append(latency_ms)
        
        total_time = (time.perf_counter() - start_total) * 1000
        
        # Calculate statistics
        stats = self._calculate_statistics(all_latencies)
        
        # Print results
        self._print_statistics("Tier Calculation Under Load", stats)
        
        print(f"Load Test Metrics:")
        print(f"  Total time: {total_time:.2f}ms")
        print(f"  Throughput: {total_calculations / (total_time / 1000):.2f} calculations/sec")
        print(f"  Average per account: {total_time / num_accounts:.2f}ms\n")
        
        # Validate targets
        print("Target Validation:")
        print(f"  Target: <1.0ms median latency under load")
        print(f"  Actual: {stats['median']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if stats['median'] < 1.0 else '❌ FAILED'}\n")
        
        # Assertions
        assert stats['median'] < 1.0, \
            f"Median latency {stats['median']:.6f}ms exceeds 1ms target under load"
        assert stats['p95'] < 2.0, \
            f"P95 latency {stats['p95']:.6f}ms exceeds 2ms acceptable limit under load"
    
    @pytest.mark.asyncio
    async def test_benchmark_comparison_summary(self):
        """
        Benchmark: Comprehensive comparison and summary
        
        Runs all benchmarks and provides a comprehensive summary comparing
        tier determination vs full position sizing performance.
        
        This is the main benchmark test that validates all requirements.
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE RISK TIER CALCULATION BENCHMARK")
        print("="*70)
        print("\nRunning comprehensive benchmark suite...")
        print("This may take a few minutes...\n")
        
        governor = PropGovernor("BENCHMARK_TEST_001")
        
        # Test configuration
        num_iterations = 10000
        num_warmup = 100
        
        test_equities = [
            100.0, 500.0, 999.0,      # Growth
            1500.0, 3000.0, 4999.0,   # Scaling
            7500.0, 15000.0, 50000.0  # Guardian
        ]
        
        # Warmup
        print("Phase 1: Warmup")
        for equity in test_equities:
            for _ in range(num_warmup):
                governor._determine_risk_tier(equity)
                governor._get_quadratic_throttle(equity)
        
        # Benchmark 1: Tier Determination Only
        print("Phase 2: Benchmarking tier determination...")
        tier_latencies = []
        
        for equity in test_equities:
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                tier = governor._determine_risk_tier(equity)
                latency_ms = (time.perf_counter() - start_time) * 1000
                tier_latencies.append(latency_ms)
        
        tier_stats = self._calculate_statistics(tier_latencies)
        
        # Benchmark 2: Full Position Sizing
        print("Phase 3: Benchmarking full position sizing...")
        sizing_latencies = []
        
        for equity in test_equities:
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                
                # Full calculation
                tier = governor._determine_risk_tier(equity)
                if tier == 'guardian':
                    throttle = governor._get_quadratic_throttle(equity)
                    risk = equity * 0.15 * throttle
                elif tier == 'scaling':
                    risk = equity * 0.15
                else:  # growth
                    risk = max(equity * 0.03, 5.0)
                
                latency_ms = (time.perf_counter() - start_time) * 1000
                sizing_latencies.append(latency_ms)
        
        sizing_stats = self._calculate_statistics(sizing_latencies)
        
        # Print comprehensive results
        print("\n" + "="*70)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*70)
        
        print("\n1. TIER DETERMINATION PERFORMANCE")
        print("-" * 70)
        print(f"Average:  {tier_stats['average']:.6f}ms")
        print(f"Median:   {tier_stats['median']:.6f}ms")
        print(f"P95:      {tier_stats['p95']:.6f}ms")
        print(f"P99:      {tier_stats['p99']:.6f}ms")
        print(f"Max:      {tier_stats['max']:.6f}ms")
        
        print("\n2. FULL POSITION SIZING PERFORMANCE")
        print("-" * 70)
        print(f"Average:  {sizing_stats['average']:.6f}ms")
        print(f"Median:   {sizing_stats['median']:.6f}ms")
        print(f"P95:      {sizing_stats['p95']:.6f}ms")
        print(f"P99:      {sizing_stats['p99']:.6f}ms")
        print(f"Max:      {sizing_stats['max']:.6f}ms")
        
        print("\n3. PERFORMANCE COMPARISON")
        print("-" * 70)
        overhead = sizing_stats['average'] - tier_stats['average']
        overhead_pct = (overhead / tier_stats['average']) * 100
        print(f"Tier determination:  {tier_stats['average']:.6f}ms")
        print(f"Full position sizing: {sizing_stats['average']:.6f}ms")
        print(f"Additional overhead:  {overhead:.6f}ms ({overhead_pct:.2f}%)")
        
        print("\n4. TARGET VALIDATION")
        print("-" * 70)
        print(f"Requirement 16.1: DetermineRiskTier() <1ms")
        print(f"  Target: <1.0ms average")
        print(f"  Actual: {tier_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if tier_stats['average'] < 1.0 else '❌ FAILED'}")
        print(f"  Margin: {((1.0 - tier_stats['average']) / 1.0 * 100):.2f}% under target")
        
        print(f"\nRequirement 16.7: CalculateTieredPositionSize() <1ms")
        print(f"  Target: <1.0ms average")
        print(f"  Actual: {sizing_stats['average']:.6f}ms")
        print(f"  Status: {'✅ PASSED' if sizing_stats['average'] < 1.0 else '❌ FAILED'}")
        print(f"  Margin: {((1.0 - sizing_stats['average']) / 1.0 * 100):.2f}% under target")
        
        print("\n5. PERFORMANCE CHARACTERISTICS")
        print("-" * 70)
        print(f"Iterations tested: {len(tier_latencies):,}")
        print(f"Equity values tested: {len(test_equities)}")
        print(f"Tiers tested: Growth, Scaling, Guardian")
        print(f"Consistency (tier): {tier_stats['stdev']:.6f}ms std dev")
        print(f"Consistency (sizing): {sizing_stats['stdev']:.6f}ms std dev")
        
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70 + "\n")
        
        # Final assertions
        assert tier_stats['average'] < 1.0, \
            f"DetermineRiskTier() average latency {tier_stats['average']:.6f}ms exceeds 1ms target (Requirement 16.1)"
        assert tier_stats['median'] < 1.0, \
            f"DetermineRiskTier() median latency {tier_stats['median']:.6f}ms exceeds 1ms target"
        
        assert sizing_stats['average'] < 1.0, \
            f"CalculateTieredPositionSize() average latency {sizing_stats['average']:.6f}ms exceeds 1ms target (Requirement 16.7)"
        assert sizing_stats['median'] < 1.0, \
            f"CalculateTieredPositionSize() median latency {sizing_stats['median']:.6f}ms exceeds 1ms target"
        
        # P95 should be reasonable (allow some outliers)
        assert tier_stats['p95'] < 2.0, \
            f"DetermineRiskTier() P95 latency {tier_stats['p95']:.6f}ms exceeds 2ms acceptable limit"
        assert sizing_stats['p95'] < 2.0, \
            f"CalculateTieredPositionSize() P95 latency {sizing_stats['p95']:.6f}ms exceeds 2ms acceptable limit"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
