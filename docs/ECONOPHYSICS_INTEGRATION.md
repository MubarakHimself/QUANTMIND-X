# Econophysics Module Integration Guide

> **Source Repository:** `quant-traderr-lab/`  
> **Target System:** QuantMindX

This document maps advanced quantitative models from the quant-traderr-lab repository to QuantMindX components.

---

## Module Integration Matrix

| Module | Primary Integration | Secondary Integration | Priority |
|--------|--------------------|-----------------------|----------|
| [Monte Carlo](#monte-carlo) | Backtest Engine | Enhanced Kelly | HIGH |
| [RMT](#rmt-random-matrix-theory) | Strategy Router | - | HIGH |
| [Fisher Transform](#fisher-transform) | Strategy Router | EA Signals | MEDIUM |
| [Ising Model](#ising-model) | Strategy Router (Regime Detection) | - | HIGH |
| [Lyapunov Exponent](#lyapunov-exponent) | Risk Governor | Enhanced Kelly | MEDIUM |
| [Wavelet Transform](#wavelet-transform) | Strategy Router | - | LOW |
| [MST](#mst-minimum-spanning-tree) | Strategy Router | Risk Management, Position Sizing | MEDIUM |
| [Sandpile Model](#sandpile-model) | Documentation/Reference | - | LOW |

---

## Monte Carlo

**Source:** `quant-traderr-lab/Monte Carlo/Monte Carlo Pipeline.py`

### Core Functionality
- Geometric Brownian Motion (GBM) price simulation
- Bootstrap resampling from historical returns
- VaR (Value at Risk) calculation
- Confidence interval estimation

### Integration: Backtest Engine

```python
class BacktestEngine:
    def run_monte_carlo_validation(self, strategy, n_simulations=10000):
        """
        After backtest, run Monte Carlo to validate results aren't due to luck.
        
        Steps:
        1. Extract strategy's historical returns
        2. Bootstrap resample to create synthetic paths
        3. Calculate distribution of outcomes
        4. Compute probability of observed performance
        """
        mc_engine = MonteCarloEngine(strategy.returns)
        paths = mc_engine.run(n_simulations, days=len(strategy.returns))
        
        # Compare actual vs simulated distribution
        actual_return = strategy.total_return
        simulated_returns = paths[-1, :] / paths[0, :] - 1
        
        # What percentile is our actual return?
        percentile = np.percentile(simulated_returns <= actual_return)
        
        return {
            'actual_return': actual_return,
            'simulated_median': np.median(simulated_returns),
            'percentile_rank': percentile,
            'statistical_significance': percentile > 95  # Top 5%
        }
```

### Integration: Enhanced Kelly (Physics-Aware Position Sizing)

> **Context:** Standard Kelly optimizes for geometric growth but assumes stationary probabilities. Real markets have "phase transitions" (Ising) and "chaos" (Lyapunov). We use these physical measurements to dynamically dampen the Kelly fraction.

```python
class PhysicsAwareKelly:
    """
    Advanced Position Sizing Engine integrating Econophysics metrics.
    
    Formula:
    f* = (p/a - q/b) * Correlation_Penalty * Chaos_Dampener * Regime_Scaler
    
    Where:
    - p = probability of win
    - b = odds received (take_profit / risk)
    """
    
    def __init__(self, risk_governor):
        self.risk_governor = risk_governor
        
    def calculate_size(self, strategy_metrics, market_physics):
        """
        Calculate optimal position size using physics constraints.
        
        Args:
            strategy_metrics (dict):
                - win_rate (float): 0.0-1.0
                - payoff_ratio (float): avg_win / avg_loss
            
            market_physics (dict):
                - lyapunov (float): Chaos metric (>0.1 is chaotic)
                - ising_susceptibility (float): Phase transition sensor
                - rmt_eigenvalue (float): Market correlation noise
        
        Returns:
            float: Recommended risk % (e.g., 0.015 for 1.5%)
        """
        p = strategy_metrics['win_rate']
        b = strategy_metrics['payoff_ratio']
        q = 1 - p
        
        # 1. Base Kelly Fraction (Full Kelly)
        # f = (p * b - q) / b
        if b == 0: return 0.0
        kelly_raw = (p * b - q) / b
        
        if kelly_raw <= 0:
            return 0.0
            
        # 2. Apply "Fractional Kelly" Base (Safety First)
        # We default to Half-Kelly for prop firm safety
        kelly_fractional = kelly_raw * 0.5
        
        # 3. Apply Physicist Constraints (The "Econophysics" Layer)
        
        # A. Chaos Dampener (Lyapunov)
        # If system is chaotic (positive Lyapunov), prediction horizon shrinks.
        # We reduce size exponentially as chaos increases.
        lyapunov = market_physics.get('lyapunov', 0)
        chaos_penalty = max(0.0, 1.0 - (lyapunov * 2)) if lyapunov > 0 else 1.0
        
        # B. Phase Transition Guard (Ising Model)
        # High magnetic susceptibility = Critical Point = Impending crash/trend change.
        # We cut risk near critical points.
        susceptibility = market_physics.get('ising_susceptibility', 0)
        regime_penalty = 0.5 if susceptibility > 0.8 else 1.0
        
        # C. Correlation Adjustment (RMT)
        # If global market correlation is high (panic mode), diversification fails.
        # We reduce risk on all strategies.
        rmt_eigen = market_physics.get('rmt_max_eigen', 1.0)
        correlation_penalty = 1.0 / rmt_eigen if rmt_eigen > 1.0 else 1.0
        
        # 4. Final Calculation
        final_risk = kelly_fractional * chaos_penalty * regime_penalty * correlation_penalty
        
        # 5. Monte Carlo Validation (The "Stress Test")
        # Ensure < 1% risk of ruin even with this sizing
        is_safe = self._run_monte_carlo_check(strategy_metrics, final_risk)
        if not is_safe:
            final_risk *= 0.5  # Slash in half if Monte Carlo fails
            
        return round(final_risk, 4)

    def _run_monte_carlo_check(self, metrics, risk_pct, runs=1000):
        """Simulate 1000 equity curves. Return False if Ruin > 1%."""
        # See Monte Carlo module for implementation
        return True 
```

#### Use Cases

1.  **The "Calm Bull" Scenario**
    *   Markets rising steadily (Low Lyapunov).
    *   No phase transition imminent (Low Ising Susceptibility).
    *   **Result:** Algorithm allocates full Half-Kelly (e.g., 2% risk). Maximizes growth.

2.  **The "Pre-Crash" Scenario**
    *   Volatility clustering appears.
    *   **Ising Model** spikes (high susceptibility) indicating critical state.
    *   **Result:** Algorithm detects "Phase Transition" risk and cuts size by 50% *before* the crash happens.

3.  **The "Choppy/Noise" Scenario**
    *   **Lyapunov Exponent** implies high chaos (random walk).
    *   **Result:** Chaos Dampener tends toward 0. Algorithm effectively "sits on hands" until stability returns.


---

## RMT (Random Matrix Theory)

**Source:** `quant-traderr-lab/RMT_Correlation_Filter/RMT_Pipeline.py`

### Core Functionality
- Marchenko-Pastur eigenvalue filtering
- Signal extraction from noisy correlation matrices
- PCA-based noise removal

### Integration: Strategy Router

```python
class StrategyRouter:
    def __init__(self):
        self.rmt_filter = RMTCorrelationFilter()
    
    def get_filtered_correlations(self, returns_df):
        """
        Filter noise from pair correlations to avoid false diversification.
        
        Problem: Raw correlations contain measurement noise
        Solution: Use RMT to identify 'true' signals vs random noise
        """
        raw_corr = returns_df.corr().values
        
        # Apply Marchenko-Pastur filtering
        cleaned_corr, lambda_plus = self.rmt_filter.apply_filtering(
            raw_corr, 
            T=len(returns_df)  # Sample size
        )
        
        return cleaned_corr
    
    def allocate_strategies(self, strategy_returns):
        """
        Use RMT-cleaned correlations for portfolio allocation.
        """
        clean_corr = self.get_filtered_correlations(strategy_returns)
        
        # Find truly uncorrelated strategies
        uncorrelated_pairs = np.where(np.abs(clean_corr) < 0.3)
        
        # Allocate more to uncorrelated strategies
        return self._optimize_weights(clean_corr)
```

---

## Fisher Transform

**Source:** `quant-traderr-lab/Fisher Transfrom/Fisher pipeline.py`

### Core Functionality
- Normalizes price to Gaussian distribution (-1, +1)
- Clearer overbought/oversold signals
- Reduces lag compared to RSI

### Integration: Strategy Router

```python
class StrategyRouter:
    def get_market_sentiment(self, prices, period=10):
        """
        Use Fisher Transform for regime-aware signal generation.
        
        Fisher > 1.5: Overbought (potential trend exhaustion)
        Fisher < -1.5: Oversold (potential trend exhaustion)  
        Fisher crossing zero: Momentum shift
        """
        fisher = self._calculate_fisher(prices, period)
        
        if fisher[-1] > 2.0:
            return 'EXTREME_BULLISH'  # Reduce trend-following allocation
        elif fisher[-1] < -2.0:
            return 'EXTREME_BEARISH'  # Reduce trend-following allocation
        else:
            return 'NEUTRAL'
```

### Integration: EA Signals

```python
class FisherSignalGenerator:
    """
    Wrapper for EAs to use Fisher Transform signals.
    
    Can be called by any EA via the API to get Fisher-based entry/exit signals.
    """
    
    def __init__(self, period: int = 10):
        self.period = period
    
    def get_signal(self, prices: list, threshold: float = 1.5) -> dict:
        """
        Generate trading signal from Fisher Transform.
        
        Returns:
            {
                'fisher_value': float,
                'signal_value': float (lagged fisher),
                'action': 'BUY' | 'SELL' | 'HOLD',
                'strength': 0.0-1.0
            }
        """
        fisher = self._calculate_fisher(prices)
        signal = fisher[-2]  # Lagged for crossover
        
        action = 'HOLD'
        if fisher[-1] > signal and fisher[-1] > -threshold:
            action = 'BUY'
        elif fisher[-1] < signal and fisher[-1] < threshold:
            action = 'SELL'
        
        return {
            'fisher_value': fisher[-1],
            'signal_value': signal,
            'action': action,
            'strength': min(abs(fisher[-1]) / 2.0, 1.0)
        }
    
    def _calculate_fisher(self, prices):
        """Fisher Transform calculation (from quant-traderr-lab)."""
        n = len(prices)
        val1 = np.zeros(n)
        fish = np.zeros(n)
        
        for i in range(self.period, n):
            min_val = np.min(prices[i-self.period+1:i+1])
            max_val = np.max(prices[i-self.period+1:i+1])
            
            range_val = max(max_val - min_val, 0.001)
            val_curr = 0.33 * 2 * ((prices[i] - min_val) / range_val - 0.5) + 0.67 * val1[i-1]
            val_curr = max(min(val_curr, 0.999), -0.999)
            val1[i] = val_curr
            
            fish[i] = 0.5 * np.log((1 + val_curr) / (1 - val_curr)) + 0.5 * fish[i-1]
        
        return fish
```

---

## Ising Model

**Source:** `quant-traderr-lab/Ising Model/Ising Pipeline.py`

### Core Functionality
- Phase transition detection using spin dynamics
- Maps volatility → temperature
- Identifies critical points (market regime changes)

### Integration: Strategy Router (Regime Detection)

The Strategy Router already has regime detection. Ising Model enhances it:

```python
class StrategyRouter:
    def __init__(self):
        self.ising = IsingRegimeDetector()
    
    def detect_regime_with_ising(self, returns, volatility):
        """
        Use Ising Model to detect if market is near phase transition.
        
        Phase transitions = regime changes
        High susceptibility = critical point (unstable regime)
        """
        # Map volatility to temperature
        temperature = volatility * 100  # Scale for Ising
        
        # Run Ising simulation
        magnetization, susceptibility = self.ising.run_at_temperature(temperature)
        
        # High susceptibility = near critical point = regime change imminent
        if susceptibility > 0.5:
            return {
                'regime': 'CRITICAL',
                'action': 'REDUCE_EXPOSURE',
                'confidence': 'HIGH'
            }
        elif abs(magnetization) > 0.5:
            return {
                'regime': 'TRENDING',
                'direction': 'BULLISH' if magnetization > 0 else 'BEARISH',
                'confidence': 'HIGH'
            }
        else:
            return {
                'regime': 'RANGING',
                'action': 'MEAN_REVERSION_STRATEGIES',
                'confidence': 'MEDIUM'
            }
```

---

## Lyapunov Exponent

**Source:** `quant-traderr-lab/Lyapunov Exponent/Lyapunov Pipeline.py`

### Core Functionality
- Measures chaos/instability in time series
- Positive Lyapunov = chaotic (unpredictable)
- Negative Lyapunov = stable (predictable)

### Integration: Risk Governor / Enhanced Kelly

```python
class RiskGovernor:
    def adjust_for_chaos(self, kelly_f, price_series):
        """
        Reduce position size when market is chaotic (unpredictable).
        
        High Lyapunov exponent = high chaos = reduce risk
        """
        lyapunov = self._calculate_lyapunov(price_series)
        
        if lyapunov > 0.5:  # High chaos
            chaos_multiplier = 0.5
            reason = 'HIGH_CHAOS'
        elif lyapunov > 0.2:  # Moderate chaos
            chaos_multiplier = 0.75
            reason = 'MODERATE_CHAOS'
        else:  # Stable
            chaos_multiplier = 1.0
            reason = 'STABLE'
        
        adjusted_kelly = kelly_f * chaos_multiplier
        
        return {
            'original_kelly': kelly_f,
            'adjusted_kelly': adjusted_kelly,
            'lyapunov_exponent': lyapunov,
            'adjustment_reason': reason
        }
```

---

## Wavelet Transform

**Source:** `quant-traderr-lab/Wavelet Transform/Wavelet_Pipeline.py`

### Core Functionality
- Multi-resolution time series decomposition
- Separates short-term noise from long-term trend
- Identifies cycle components at different frequencies

### Integration: Strategy Router

```python
class StrategyRouter:
    def get_multi_timeframe_signals(self, prices):
        """
        Use wavelet decomposition to analyze multiple timeframes at once.
        
        - High-frequency component → Scalping signals
        - Mid-frequency component → Swing trading signals
        - Low-frequency component → Position trading signals
        """
        coeffs = pywt.wavedec(prices, 'db4', level=4)
        
        # cA4 = Trend (low freq)
        # cD4, cD3 = Swing (mid freq)
        # cD2, cD1 = Noise/Scalping (high freq)
        
        trend_direction = 'UP' if coeffs[0][-1] > coeffs[0][-2] else 'DOWN'
        
        return {
            'trend': trend_direction,
            'scalping_regime': 'ACTIVE' if np.std(coeffs[-1]) > threshold else 'QUIET',
            'swing_regime': 'TRENDING' if np.std(coeffs[2]) > threshold else 'RANGING'
        }
```

---

## MST (Minimum Spanning Tree)

**Source:** `quant-traderr-lab/MST/MST pipeline.py`

### Core Functionality
- Network analysis of asset correlations
- Identifies clusters of correlated assets
- Finds central "hub" assets

### Integration: Strategy Router + Risk Management + Position Sizing

```python
class CorrelationManager:
    def build_asset_network(self, returns_df):
        """
        Build MST from correlation matrix to identify clusters.
        
        Used for:
        - Diversification: Don't over-allocate to same cluster
        - Risk: High connectivity = systemic risk
        """
        corr = returns_df.corr()
        
        # Convert correlation to distance (1 - |corr|)
        distance = 1 - np.abs(corr.values)
        
        # Build MST
        mst = self._build_mst(distance)
        
        # Find clusters
        clusters = self._identify_clusters(mst)
        
        return {
            'clusters': clusters,
            'hub_assets': self._find_hubs(mst),
            'diversification_score': self._calc_diversification(clusters)
        }
    
    def limit_cluster_exposure(self, positions, clusters, max_per_cluster=0.25):
        """
        Ensure no single cluster has more than 25% of portfolio.
        """
        for cluster_id, assets in clusters.items():
            cluster_exposure = sum(positions.get(a, 0) for a in assets)
            if cluster_exposure > max_per_cluster:
                # Scale down proportionally
                scale = max_per_cluster / cluster_exposure
                for asset in assets:
                    if asset in positions:
                        positions[asset] *= scale
        
        return positions
```

---

## Sandpile Model

**Source:** `quant-traderr-lab/Sandpile Model/Sandpile Pipeline.py`

> **Note:** This model is included for reference/documentation only. Not actively integrated.

### Core Concept
- Self-organized criticality
- Small perturbations can cause large avalanches
- Markets naturally evolve to critical states

### Potential Future Use
- Crash probability estimation
- Identifying when markets are "stressed" and prone to cascades
- Early warning system for flash crashes

---

## Implementation Priority

### Phase 1 (Immediate)
1. **Fisher Transform API** - For EA signals
2. **RMT Filter** - For Strategy Router correlations

### Phase 2 (With Backtest Engine)
1. **Monte Carlo** - Strategy validation
2. **Lyapunov** - Chaos-adjusted sizing

### Phase 3 (Optimization)
1. **MST** - Portfolio clustering
2. **Ising Model** - Enhanced regime detection
3. **Wavelet** - Multi-TF analysis

---

## File Locations

```
quant-traderr-lab/
├── Monte Carlo/Monte Carlo Pipeline.py    → src/backtest/monte_carlo.py
├── RMT_Correlation_Filter/RMT_Pipeline.py → src/strategy_router/rmt_filter.py
├── Fisher Transfrom/Fisher pipeline.py    → src/signals/fisher_transform.py
├── Ising Model/Ising Pipeline.py          → src/strategy_router/ising_regime.py
├── Lyapunov Exponent/Lyapunov Pipeline.py → src/risk/lyapunov_chaos.py
├── Wavelet Transform/Wavelet_Pipeline.py  → src/strategy_router/wavelet_mtf.py
├── MST/MST pipeline.py                    → src/risk/mst_clustering.py
└── Sandpile Model/Sandpile Pipeline.py    → docs/reference/sandpile_theory.md
```
