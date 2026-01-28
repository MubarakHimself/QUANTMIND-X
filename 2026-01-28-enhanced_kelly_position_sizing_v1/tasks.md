# Task Breakdown: Enhanced Kelly Position Sizing with Econophysics Integration

## Overview
**Total Tasks:** 72 tasks across 9 major task groups
**Estimated Implementation Time:** 3-4 weeks with proper testing
**Total Estimated Tests:** 38-62 tests (focused testing approach)

This tasks list breaks down the implementation of a production-ready position sizing system that integrates econophysics models (Ising phase transitions, Lyapunov chaos detection, RMT correlation filtering) with the Kelly Criterion for superior risk-adjusted returns.

**Key Features:**
- Modular physics sensors for market regime detection
- Monte Carlo validation for risk confirmation
- Physics-aware Kelly engine with weakest-link aggregation
- End-to-end lot calculation with broker constraints
- Prop firm preset support (FTMO, The5ers, etc.)
- MT5/MCP integration for live account data
- Performance optimized (< 200ms calculation time)

---

## Execution Order

**Recommended implementation sequence:**
1. **Foundation Layer** (Task Groups 1-3) - Data models, config, directory structure
2. **Physics Sensor Layer** (Task Groups 4-6) - Ising, Chaos, Correlation sensors
3. **Risk Engine Layer** (Task Groups 7-8) - Monte Carlo validator, Kelly engine
4. **Integration Layer** (Task Groups 9-10) - Governor, MT5 integration
5. **Quality Assurance** (Task Groups 11-13) - Testing, edge cases, documentation

**Parallel Development Opportunities:**
- After Task Groups 1-3 complete, sensors (4-6) can be developed independently
- Monte Carlo validator (7) can be developed in parallel with sensors
- Kelly engine (8) requires all sensors but can be tested with mocks

---

## Task List

### Foundation Layer

#### Task Group 1: Data Models and Pydantic Schemas
**Dependencies:** None

- [ ] 1.0 Complete data models layer
  - [ ] 1.1 Write 2-6 focused tests for Pydantic models
    - Test StrategyPerformance validation (win_rate edge cases, negative values)
    - Test MarketPhysics timestamp validation and staleness detection
    - Test SizingRecommendation constraint tracking
    - Test PositionSizingResult lot calculation accuracy
    - Skip exhaustive field validation tests
  - [ ] 1.2 Create `src/risk/models/strategy_performance.py`
    - Fields: win_rate (0-1), avg_win, avg_loss, total_trades, profit_factor, k_fraction
    - Validators: win_rate bounds, positive values for monetary fields
    - Computed property: payoff_ratio = avg_win / avg_loss
    - Method: expectancy() -> float (win_rate * avg_win - (1-win_rate) * avg_loss)
  - [ ] 1.3 Create `src/risk/models/market_physics.py`
    - Fields: lyapunov_exponent, ising_susceptibility, ising_magnetization
    - Fields: rmt_max_eigenvalue, rmt_noise_threshold
    - Fields: calculated_at (datetime), is_stale (bool)
    - Method: is_fresh(max_age_seconds: int) -> bool
    - Method: risk_level() -> str ("LOW", "MODERATE", "HIGH")
  - [ ] 1.4 Create `src/risk/models/sizing_recommendation.py`
    - Fields: raw_kelly, physics_multiplier, final_risk_pct, position_size_lots
    - Fields: constraint_source (str), validation_passed (bool), adjustments_applied (list[str])
    - Method: apply_penalty(penalty_type: str, multiplier: float) -> None
    - Method: is_constrained() -> bool
  - [ ] 1.5 Create `src/risk/models/position_sizing_result.py`
    - Fields: account_balance, risk_amount, stop_loss_pips, pip_value, lot_size
    - Fields: estimated_margin, remaining_margin, calculation_steps (list[str])
    - Method: to_dict() -> dict for serialization
  - [ ] 1.6 Create `src/risk/models/__init__.py`
    - Export all models: StrategyPerformance, MarketPhysics, SizingRecommendation, PositionSizingResult
    - Export validation exceptions
  - [ ] 1.7 Ensure data model tests pass
    - Run ONLY the 2-6 tests written in 1.1
    - Verify Pydantic validation works correctly
    - Do NOT run entire test suite

**Acceptance Criteria:**
- All Pydantic models validate input correctly
- Edge cases (negative values, out-of-bounds) raise proper errors
- Models are serializable for logging and API responses
- The 2-6 tests written in 1.1 pass

---

#### Task Group 2: Configuration Management
**Dependencies:** None

- [ ] 2.0 Complete configuration layer
  - [ ] 2.1 Write 2-4 focused tests for configuration
    - Test prop firm preset loading (FTMO, The5ers)
    - Test configuration validation (thresholds, limits)
    - Test default values are reasonable
  - [ ] 2.2 Create `src/risk/config.py`
    - Physics thresholds:
      - LYAPUNOV_CHAOS_THRESHOLD = 0.5
      - ISING_CRITICAL_SUSCEPTIBILITY = 0.8
      - RMT_MAX_EIGEN_THRESHOLD = 1.5
      - RMT_CRITICAL_EIGEN_THRESHOLD = 2.0
    - Cache settings:
      - PHYSICS_CACHE_TTL = 300 (5 minutes)
      - ACCOUNT_CACHE_TTL = 10 (seconds)
    - Monte Carlo settings:
      - MC_SIMULATION_RUNS = 2000
      - MC_MAX_DRAWDOWN = 0.10 (10%)
      - MC_RUIN_THRESHOLD = 0.005 (0.5%)
    - Kelly settings:
      - DEFAULT_K_FRACTION = 0.5 (half-Kelly)
      - MAX_RISK_PCT = 0.02 (2%)
    - Broker constraints:
      - MIN_LOT = 0.01
      - LOT_STEP = 0.01
      - MAX_LOT = 100.0
  - [ ] 2.3 Define prop firm preset classes
    - Base class: PropFirmPreset (name, max_drawdown, max_daily_loss, profit_target)
    - FTMO preset: 10% max drawdown, 5% daily loss
    - The5ers preset: 8% max drawdown, 4% daily loss
    - FundingPips preset: 12% max drawdown, 6% daily loss
    - Method: get_max_risk_pct() -> float
  - [ ] 2.4 Create configuration validator
    - Function: validate_config() -> None (raises ValueError for invalid settings)
    - Check: thresholds within reasonable ranges (0.0-1.0 or 0.0-10.0)
    - Check: cache TTL positive values
    - Check: Monte Carlo runs >= 100 (minimum statistical significance)
    - Check: prop firm preset consistency
  - [ ] 2.5 Ensure configuration tests pass
    - Run ONLY the 2-4 tests written in 2.1
    - Verify all presets load correctly

**Acceptance Criteria:**
- All configuration constants are centralized
- Prop firm presets match firm requirements
- Configuration validation catches invalid settings
- The 2-4 tests written in 2.1 pass

---

#### Task Group 3: Directory Structure and Package Setup
**Dependencies:** Task Groups 1-2

- [ ] 3.0 Complete project structure
  - [ ] 3.1 Create directory structure
    - Create `src/risk/` root package
    - Create `src/risk/physics/` for sensor modules
    - Create `src/risk/sizing/` for position sizing logic
    - Create `src/risk/models/` for Pydantic models (already done in 1.6)
    - Create `tests/risk/` for test files
    - Create `tests/risk/physics/`, `tests/risk/sizing/`, `tests/risk/integration/`
  - [ ] 3.2 Create `src/risk/__init__.py`
    - Export RiskGovernor as main entry point
    - Export all models for convenience
    - Export exceptions for error handling
    - Version: __version__ = "1.0.0"
  - [ ] 3.3 Create `src/risk/physics/__init__.py`
    - Export all sensor classes
    - Export sensor exceptions
  - [ ] 3.4 Create `src/risk/sizing/__init__.py`
    - Export Kelly engine and validator classes
  - [ ] 3.5 Add requirements to project requirements.txt
    - numpy>=1.24.0 (vectorized calculations)
    - scipy>=1.10.0 (KD-tree, statistical functions)
    - pydantic>=2.0.0 (data validation)
    - Add to existing requirements.txt file
  - [ ] 3.6 Create package setup validation
    - Test: import risk; from risk import RiskGovernor
    - Verify all submodules import correctly

**Acceptance Criteria:**
- All directories created with proper __init__.py files
- Package imports work: `from risk import RiskGovernor`
- Dependencies are listed in requirements.txt

---

### Physics Sensor Layer

#### Task Group 4: Ising Regime Sensor
**Dependencies:** Task Group 3

- [ ] 4.0 Complete Ising regime sensor
  - [ ] 4.1 Write 2-6 focused tests for Ising sensor
    - Test phase transition detection at critical susceptibility (χ > 0.8)
    - Test temperature mapping from volatility
    - Test Metropolis-Hastings algorithm acceptance rate
    - Test magnetization calculation
    - Skip visualization and rendering tests
  - [ ] 4.2 Create `src/risk/physics/ising_sensor.py`
    - Extract core IsingSystem logic from quant-traderr-lab/Ising Model
    - Remove visualization, market fetching, video code
    - Class: IsingRegimeSensor
  - [ ] 4.3 Implement IsingSystem simulation core
    - 3D lattice: 12x12x12 grid (1728 spins)
    - Periodic boundary conditions
    - Metropolis-Hastings Monte Carlo algorithm:
      - Energy: E = -J * sum(s_i * s_j)
      - Acceptance probability: min(1, exp(-ΔE/T))
    - Magnetization: M = |sum(s_i)| / N
    - Susceptibility: χ = (⟨M²⟩ - ⟨M⟩²) / T
  - [ ] 4.4 Implement detect_regime method
    - Input: volatility (float) mapped to temperature (0.1 to 10.0)
    - Simulation steps: 5000 for equilibrium, 1000 for measurement
    - Output: dict with susceptibility, magnetization, regime_label
    - Regime labels:
      - "ORDERED" (χ < 0.3, M > 0.5)
      - "TRANSITIONAL" (0.3 <= χ < 0.8)
      - "CHAOTIC" (χ >= 0.8)
  - [ ] 4.5 Add temperature mapping
    - Low volatility (< 0.5%) → T = 0.1 (ordered)
    - Medium volatility (0.5% - 2%) → T = 2.0 - 5.0 (transitional)
    - High volatility (> 2%) → T = 10.0 (chaotic)
  - [ ] 4.6 Add caching layer
    - Cache results with timestamp
    - Method: is_cache_valid(max_age_seconds: int) -> bool
    - Method: get_cached_result(volatility: float) -> dict or None
  - [ ] 4.7 Ensure Ising sensor tests pass
    - Run ONLY the 2-6 tests written in 4.1
    - Verify phase transition detection works

**Acceptance Criteria:**
- Ising sensor detects phase transitions (χ > 0.8)
- Simulation completes in < 100ms
- Magnetization and susceptibility calculations accurate
- The 2-6 tests written in 4.1 pass

---

#### Task Group 5: Chaos Sensor (Lyapunov)
**Dependencies:** Task Group 3

- [ ] 5.0 Complete Lyapunov chaos sensor
  - [ ] 5.1 Write 2-6 focused tests for Chaos sensor
    - Test Lyapunov exponent calculation on synthetic chaotic data (logistic map)
    - Test periodic time series returns λ ≈ 0
    - Test phase space embedding dimension
    - Test KD-tree nearest neighbor search
    - Skip visualization tests
  - [ ] 5.2 Create `src/risk/physics/chaos_sensor.py`
    - Extract logic from quant-traderr-lab/Lyapunov Exponent
    - Remove visualization and plotting code
    - Class: ChaosSensor
  - [ ] 5.3 Implement phase space reconstruction
    - Time delay embedding: dimension = 3, tau = 12
    - Embedding: X(t) = [x(t), x(t-τ), x(t-2τ), ..., x(t-(d-1)τ)]
    - Input: log-returns array (N >= 300 points)
  - [ ] 5.4 Implement method of analogues
    - Build KD-tree from embedded trajectory (scipy.spatial.KDTree)
    - Find nearest neighbor to current state
    - Track divergence over k steps forward (k=10)
    - Lyapunov exponent: λ = (1/k) * sum(log(d(t+i) / d(t)))
  - [ ] 5.5 Implement analyze_chaos method
    - Input: returns (np.ndarray), lookback = 300
    - Output: dict with lyapunov_exponent, match_distance, match_index, chaos_level
    - Chaos labels:
      - "STABLE" (λ < 0.2)
      - "MODERATE" (0.2 <= λ < 0.5)
      - "CHAOTIC" (λ >= 0.5)
  - [ ] 5.6 Add validation for input length
    - Raise ValueError if N < 300 (insufficient data)
    - Warn if N < 500 (reduced accuracy)
    - Pad with zeros if N < 300 (with warning)
  - [ ] 5.7 Add caching layer
    - Cache results by input hash
    - Method: is_cache_valid(max_age_seconds: int) -> bool
  - [ ] 5.8 Ensure Chaos sensor tests pass
    - Run ONLY the 2-6 tests written in 5.1
    - Verify Lyapunov calculation accuracy

**Acceptance Criteria:**
- Lyapunov exponent detects chaos (λ > 0.5)
- Phase space embedding preserves dynamics
- KD-tree search is efficient (O(log N))
- The 2-6 tests written in 5.1 pass

---

#### Task Group 6: Correlation Sensor (RMT)
**Dependencies:** Task Group 3

- [ ] 6.0 Complete RMT correlation sensor
  - [ ] 6.1 Write 2-6 focused tests for Correlation sensor
    - Test Marchenko-Pastur distribution calculation
    - Test eigenvalue decomposition (PCA)
    - Test noise threshold detection
    - Test systemic risk detection (λ_max > 2.0)
    - Skip 3D rendering tests
  - [ ] 6.2 Create `src/risk/physics/correlation_sensor.py`
    - Extract logic from quant-traderr-lab/RMT_Correlation_Filter
    - Remove 3D rendering, video compilation, matplotlib code
    - Class: CorrelationSensor
  - [ ] 6.3 Implement Marchenko-Pastur distribution
    - Formula: λ_max = σ² * (1 + sqrt(Q))² where Q = T/N
    - Input: correlation matrix (N assets × N assets)
    - Calculate theoretical eigenvalue bounds for random matrix
    - σ² = 1 for normalized correlation matrix
  - [ ] 6.4 Implement eigenvalue decomposition
    - Use scipy.linalg.eigh for symmetric correlation matrix
    - Sort eigenvalues descending
    - Extract λ_max (largest eigenvalue)
  - [ ] 6.5 Implement detect_systemic_risk method
    - Input: returns_matrix (np.ndarray) shape (N_assets, T_periods)
    - Calculate correlation matrix: C = corr(returns) using np.corrcoef
    - Compute eigenvalues via eigh(C)
    - Calculate Marchenko-Pastur threshold
    - Output: dict with max_eigenvalue, noise_threshold, denoised_matrix, risk_level
    - Risk labels:
      - "LOW" (λ_max < 1.5)
      - "MODERATE" (1.5 <= λ_max < 2.0)
      - "HIGH" (λ_max >= 2.0)
  - [ ] 6.6 Add denoising (optional)
    - Remove eigenvalues below noise threshold
    - Reconstruct correlation matrix: C_denoised = Q_signal * Λ_signal * Q_signal^T
  - [ ] 6.7 Handle edge cases
    - Validate minimum dimensions (N >= 2, T >= 20)
    - Handle singular matrices (add regularization: C += 1e-6 * I)
    - Handle missing data (use mean imputation)
  - [ ] 6.8 Add caching layer
    - Cache results by matrix hash
    - Method: is_cache_valid(max_age_seconds: int) -> bool
  - [ ] 6.9 Ensure Correlation sensor tests pass
    - Run ONLY the 2-6 tests written in 6.1
    - Verify RMT filtering works correctly

**Acceptance Criteria:**
- Marchenko-Pastur distribution correctly identifies noise
- Eigenvalue decomposition accurate
- Systemic risk detected when λ_max > 2.0
- The 2-6 tests written in 6.1 pass

---

### Risk Engine Layer

#### Task Group 7: Monte Carlo Validator
**Dependencies:** Task Group 3

- [ ] 7.0 Complete Monte Carlo validator
  - [ ] 7.1 Write 2-6 focused tests for Monte Carlo validator
    - Test risk of ruin calculation (< 0.5% threshold)
    - Test confidence interval calculation (95% CI)
    - Test bootstrap resampling accuracy
    - Test validation failure handling (halved risk)
    - Skip visualization tests
  - [ ] 7.2 Create `src/risk/sizing/monte_carlo_validator.py`
    - Extract logic from quant-traderr-lab/Monte Carlo
    - Remove visualization and plotting code
    - Class: MonteCarloValidator
  - [ ] 7.3 Implement bootstrap resampling
    - Input: historical_returns (np.ndarray), risk_pct (float), runs (int = 2000)
    - For each run:
      - Sample with replacement from historical returns
      - Simulate trading sequence with risk_pct per trade
      - Track: final equity, max drawdown, ruin events
    - Use vectorized numpy operations for performance
  - [ ] 7.4 Implement validate_risk method
    - Input: StrategyPerformance, risk_pct (float), runs (int)
    - Calculate: avg_return, std_return from strategy metrics
    - Generate return distribution (bootstrap from historical if available)
    - Simulate runs: equity curve with risk_pct per trade
    - Output: ValidationResult (passed: bool, risk_of_ruin: float, adjusted_risk: float, ci_95: tuple)
  - [ ] 7.5 Calculate risk metrics
    - Risk of ruin: ruin_count / runs
    - 95% confidence interval: np.percentile(final_equities, [2.5, 97.5])
    - Expected drawdown: avg(max_drawdowns)
    - VaR (95%): percentile of final equities
  - [ ] 7.6 Implement decision logic
    - Pass if: risk_of_ruin < MC_RUIN_THRESHOLD (0.5%)
    - Fail if: risk_of_ruin >= threshold
    - Adjusted risk: risk_pct / 2 if failed (conservative halving)
    - Log validation result with metrics
  - [ ] 7.7 Add vectorization for performance
    - Use numpy operations instead of Python loops
    - Pre-allocate arrays: equities = np.zeros(runs)
    - Use np.random.choice for bootstrapping
  - [ ] 7.8 Ensure Monte Carlo validator tests pass
    - Run ONLY the 2-6 tests written in 7.1
    - Verify risk calculations accurate

**Acceptance Criteria:**
- Risk of ruin calculated correctly
- 95% CI accurate for final equity
- Simulation completes in < 500ms for 2000 runs
- The 2-6 tests written in 7.1 pass

---

#### Task Group 8: Physics-Aware Kelly Engine
**Dependencies:** Task Groups 1, 2, 4-6

- [ ] 8.0 Complete Kelly engine
  - [ ] 8.1 Write 2-8 focused tests for Kelly engine
    - Test Kelly calculation: f* = (bp - q) / b (win_rate, payoff_ratio)
    - Test half-Kelly safety scalar (k_fraction = 0.5)
    - Test Lyapunov penalty: P_λ = max(0, 1.0 - (2.0 × λ))
    - Test Ising penalty: P_χ = 0.5 if χ > 0.8 else 1.0
    - Test Eigen penalty: P_E = min(1.0, 1.5 / λ_max)
    - Test weakest link aggregation: M_physics = min(P_λ, P_χ, P_E)
    - Test negative expectancy handling (zero position)
    - Skip exhaustive edge case tests
  - [ ] 8.2 Create `src/risk/sizing/kelly_engine.py`
    - Class: PhysicsAwareKellyEngine
    - Dependencies: StrategyPerformance, MarketPhysics models
  - [ ] 8.3 Implement base Kelly calculation
    - Formula: f* = (p * b - q) / b
      - p = win_rate
      - q = 1 - p
      - b = payoff_ratio = avg_win / avg_loss
    - Half-Kelly: f_base = f* * k_fraction (default 0.5)
    - Constraint: 0 <= f_base <= MAX_RISK_PCT (2%)
    - Handle: division by zero (avg_loss = 0), negative expectancy
  - [ ] 8.4 Implement physics multipliers
    - Lyapunov: P_λ = max(0, 1.0 - (2.0 × λ))
      - λ > 0.5 → P_λ < 0 (chaos penalty)
      - λ < 0.2 → P_λ = 1.0 (stable)
    - Ising: P_χ = 0.5 if χ > ISING_CRITICAL_SUSCEPTIBILITY else 1.0
      - χ > 0.8 → phase transition, reduce risk by 50%
    - Eigen: P_E = min(1.0, RMT_MAX_EIGEN_THRESHOLD / λ_max) if λ_max > RMT_MAX_EIGEN_THRESHOLD else 1.0
      - λ_max > 1.5 → correlation penalty
  - [ ] 8.5 Implement weakest link aggregation
    - M_physics = min(P_λ, P_χ, P_E)
    - Track constraint_source: which penalty was minimum
    - Add to adjustments_applied list with details
  - [ ] 8.6 Implement calculate_position_size method
    - Input: StrategyPerformance, MarketPhysics, optional: MonteCarloValidator
    - Calculate: raw_kelly, base_kelly (half-Kelly), physics_multiplier
    - Final risk: final_risk_pct = base_kelly * M_physics
    - Optional: Monte Carlo validation if final_risk_pct > 0.5%
    - Return: SizingRecommendation with full breakdown
  - [ ] 8.7 Add edge case handling
    - Zero/negative payoff: return zero position with reason
    - Missing physics data: default multiplier = 0.5 (safety mode)
    - All penalties zero: log critical error, use 0.1 minimum
    - Extreme payoff ratios (> 10.0): cap at 10.0
  - [ ] 8.8 Ensure Kelly engine tests pass
    - Run ONLY the 2-8 tests written in 8.1
    - Verify Kelly formula correct
    - Verify physics penalties applied

**Acceptance Criteria:**
- Kelly formula implemented correctly
- Physics multipliers match specification
- Weakest link principle enforced
- Monte Carlo integration works
- The 2-8 tests written in 8.1 pass

---

### Integration Layer

#### Task Group 9: RiskGovernor Main Entry Point
**Dependencies:** Task Groups 4-8

- [ ] 9.0 Complete RiskGovernor orchestrator
  - [ ] 9.1 Write 2-8 focused integration tests
    - Test end-to-end calculation with mock data
    - Test lot calculation: Lots = (Balance * Risk%) / (SL_Pips * Pip_Value)
    - Test broker constraint enforcement (min_lot, max_lot, lot_step)
    - Test prop firm preset application (FTMO, The5ers)
    - Test cache invalidation (TTL expiry)
    - Test missing sensor data handling
    - Test zero position edge cases
  - [ ] 9.2 Create `src/risk/governor.py`
    - Class: RiskGovernor (main entry point)
    - Initialize: IsingRegimeSensor, ChaosSensor, CorrelationSensor, KellyEngine, MonteCarloValidator
  - [ ] 9.3 Implement calculate_position_size method
    - Signature: calculate_position_size(account_info, strategy_perf, market_state, stop_loss_pips, pip_value) -> PositionSizingResult
    - Steps:
      1. Fetch account balance (from AccountManager or cached)
      2. Fetch/refresh physics sensor data (check cache)
      3. Calculate Kelly recommendation
      4. Monte Carlo validation (optional)
      5. Convert risk% to lots
      6. Apply broker constraints
      7. Calculate margin requirements
    - Return: PositionSizingResult with all details
  - [ ] 9.4 Implement lot calculation
    - Formula: lots = (balance * risk_pct) / (sl_pips * pip_value)
    - Rounding: floor to lot_step (e.g., 0.01)
    - Constraints: min_lot <= lots <= max_lot
    - Handle: lots < min_lot → return 0.0 or min_lot (configurable)
  - [ ] 9.5 Implement caching layer
    - Physics sensor cache: TTL = 300 seconds (5 minutes)
    - Account info cache: TTL = 10 seconds
    - Method: _get_cached_physics() -> MarketPhysics or None
    - Method: _get_cached_account() -> AccountInfo or None
    - Method: _invalidate_cache() -> None
  - [ ] 9.6 Add prop firm preset support
    - Method: apply_preset(preset: PropFirmPreset) -> None
    - Override max_risk_pct based on preset
    - Apply preset-specific drawdown limits
    - Log preset application
  - [ ] 9.7 Implement error handling
    - Sensor failure: log warning, use default multiplier = 0.5
    - MT5 connection failure: log error, use provided account_info
    - Validation failure: log, continue with conservative sizing
    - Missing data: log, use fallback values
  - [ ] 9.8 Add logging and observability
    - Log all calculation steps (input, output, adjustments)
    - Log constraint source and penalty breakdown
    - Log cache hits/misses
    - Structured JSON format for aggregation
    - Use Python logging module with proper levels
  - [ ] 9.9 Ensure RiskGovernor tests pass
    - Run ONLY the 2-8 tests written in 9.1
    - Verify end-to-end workflow

**Acceptance Criteria:**
- End-to-end position calculation works
- Lot formula accurate for various inputs
- Broker constraints enforced
- Caching reduces redundant calculations
- The 2-8 tests written in 9.1 pass

---

#### Task Group 10: MT5 Integration
**Dependencies:** Task Group 9

- [ ] 10.0 Complete MT5/MCP integration
  - [ ] 10.1 Write 2-4 focused tests for MT5 integration
    - Test AccountManager import and usage
    - Test balance fetching from MT5
    - Test pip value calculation for various symbols
    - Test connection error handling
  - [ ] 10.2 Create `src/risk/integrations/mt5_client.py`
    - Wrapper class for mcp-metatrader5-server AccountManager
    - Handle: import errors, connection failures
    - Class: MT5AccountClient
  - [ ] 10.3 Implement AccountManager integration
    - Import: from mcp_mt5 import AccountManager
    - Method: get_account_balance() -> float
    - Method: get_account_equity() -> float
    - Method: get_margin_info() -> dict (free_margin, margin_level)
    - Cache results with 10-second TTL
    - Handle connection errors gracefully
  - [ ] 10.4 Implement pip value calculation
    - Standard lot (1.00) = $10/pip for EURUSD, GBPUSD, etc.
    - Cross pairs: calculate via base currency exchange rate
    - Gold (XAUUSD): $1/pip per standard lot
    - Indices: config override (e.g., NAS100 = $1/pip, varies by broker)
    - Method: calculate_pip_value(symbol: str, lot_size: float) -> float
  - [ ] 10.5 Add symbol info lookup
    - Use MT5 symbol_info() for contract size
    - Extract: point size, tick value, lot step
    - Handle: exotic symbols with custom pip_value parameter
    - Cache symbol info to reduce MT5 calls
  - [ ] 10.6 Add graceful degradation
    - If MT5 unavailable: use provided account_info dict
    - Log warning when using fallback data
    - Validate: balance > 0, equity >= balance
  - [ ] 10.7 Ensure MT5 integration tests pass
    - Run ONLY the 2-4 tests written in 10.1
    - Verify AccountManager integration works

**Acceptance Criteria:**
- AccountManager integration functional
- Pip value calculations accurate for major pairs
- Fallback mode works when MT5 unavailable
- The 2-4 tests written in 10.1 pass

---

### Quality Assurance Layer

#### Task Group 11: Comprehensive Testing
**Dependencies:** Task Groups 1-10

- [ ] 11.0 Review existing tests and fill critical gaps
  - [ ] 11.1 Review all tests from Task Groups 1-10
    - Review 2-6 tests from data models (1.1)
    - Review 2-4 tests from configuration (2.1)
    - Review 2-6 tests from Ising sensor (4.1)
    - Review 2-6 tests from Chaos sensor (5.1)
    - Review 2-6 tests from Correlation sensor (6.1)
    - Review 2-6 tests from Monte Carlo validator (7.1)
    - Review 2-8 tests from Kelly engine (8.1)
    - Review 2-8 tests from RiskGovernor (9.1)
    - Review 2-4 tests from MT5 integration (10.1)
    - Total existing tests: approximately 28-52 tests
  - [ ] 11.2 Analyze test coverage gaps for THIS feature only
    - Identify critical integration paths lacking coverage
    - Focus on end-to-end workflows: data → sensors → Kelly → lots
    - Check for missing edge cases in business logic
    - Prioritize integration points over unit test gaps
    - Skip exhaustive edge case coverage
  - [ ] 11.3 Write up to 10 additional strategic tests maximum
    - Test physics sensor orchestration (all sensors together)
    - Test cache invalidation across multiple calls
    - Test prop firm preset switching
    - Test concurrent calls to RiskGovernor
    - Test margin calculation accuracy
    - Focus on integration and workflows
    - Add maximum of 10 new tests total
  - [ ] 11.4 Create backtesting comparison test
    - Compare Physics Kelly vs Standard Kelly on synthetic data
    - Test crisis period simulation (high volatility, high chaos)
    - Verify: Physics Kelly reduces position size during chaos
    - Metric: max drawdown reduction > 20% during crisis
  - [ ] 11.5 Run feature-specific tests only
    - Run ONLY tests related to this spec's feature
    - Expected total: approximately 38-62 tests maximum (existing 28-52 + up to 10 new)
    - Do NOT run entire application test suite
    - Verify critical workflows pass

**Acceptance Criteria:**
- All feature-specific tests pass (approximately 38-62 tests total)
- Critical integration workflows covered
- No more than 10 additional tests added
- Backtesting shows Physics Kelly superiority

---

#### Task Group 12: Edge Case Handling
**Dependencies:** Task Groups 1-10

- [ ] 12.0 Implement edge case handling
  - [ ] 12.1 Handle insufficient trade history
    - If total_trades < 30: use conservative fallback (0.5-1% risk)
    - Log warning about statistical uncertainty
  - [ ] 12.2 Handle missing physics sensor data
    - Default multiplier = 0.5 (safety mode)
    - Log which sensor failed
  - [ ] 12.3 Handle stale physics metrics
    - Check age vs PHYSICS_CACHE_TTL
    - Re-calculate if stale, or use decay factor
  - [ ] 12.4 Handle zero/negative payoff
    - Return zero position with clear reason
    - Log: "Negative expectancy, no position recommended"
  - [ ] 12.5 Handle zero average loss
    - Return zero position (division by zero protection)
    - Log: "Zero loss history, cannot calculate Kelly"
  - [ ] 12.6 Handle flash crash events
    - Ising susceptibility spikes (> 1.5): immediate 50% risk reduction
    - Log critical warning
  - [ ] 12.7 Handle gap openings
    - Skip trade if gap > 2 * ATR
    - Log: "Gap detected, position sizing skipped"
  - [ ] 12.8 Handle position size below minimum lot
    - If allow_zero=True: return 0.0 lots
    - Else: return min_lot with warning
  - [ ] 12.9 Handle position size above maximum lot
    - Cap at max_lot
    - Log warning: "Position size capped at broker maximum"
  - [ ] 12.10 Handle insufficient margin
    - Reduce size to fit free margin
    - Log: "Position reduced due to margin constraints"

**Acceptance Criteria:**
- All documented edge cases handled gracefully
- Clear error messages and logging
- No uncaught exceptions on edge cases

---

#### Task Group 13: Performance Optimization
**Dependencies:** Task Groups 4-8

- [ ] 13.0 Optimize performance
  - [ ] 13.1 Profile physics sensor calculations
    - Target: Ising sensor < 100ms
    - Target: Chaos sensor < 100ms
    - Target: Correlation sensor < 150ms
  - [ ] 13.2 Optimize NumPy vectorization
    - Replace Python loops with vectorized operations
    - Use in-place operations where possible
  - [ ] 13.3 Optimize Monte Carlo simulation
    - Pre-allocate arrays for results
    - Use numpy.random.choice for bootstrapping
    - Target: < 500ms for 2000 runs
  - [ ] 13.4 Implement parallel processing (optional)
    - Use multiprocessing.Pool for Monte Carlo
    - Parallelize independent sensor calculations
  - [ ] 13.5 Optimize memory usage
    - Limit time series length (300 points max)
    - Use generators for Monte Carlo
    - Clear cache after TTL expiry
  - [ ] 13.6 Add performance benchmarks
    - Benchmark: sensor calculation latency
    - Benchmark: full position sizing latency
    - Benchmark: cache hit rate
  - [ ] 13.7 Create performance test suite
    - Use pytest-benchmark or custom timing
    - Regression tests for performance
    - Target: < 200ms full calculation (with cache)

**Acceptance Criteria:**
- Ising sensor < 100ms
- Full calculation < 200ms (with cache)
- Monte Carlo < 500ms
- Memory usage < 100MB

---

#### Task Group 14: Documentation and Logging
**Dependencies:** Task Groups 1-10

- [ ] 14.0 Complete documentation
  - [ ] 14.1 Add code docstrings
    - All classes: purpose, usage example
    - All public methods: parameters, returns, raises
    - Include LaTeX formulas in docstrings
    - Follow NumPy docstring style guide
  - [ ] 14.2 Add inline comments
    - Explain complex physics calculations
    - Reference papers (Ising, Lyapunov, RMT)
  - [ ] 14.3 Add type hints
    - All function parameters and returns
    - Use typing module (List, Dict, Optional, etc.)
  - [ ] 14.4 Create user guide
    - Quick start example
    - Prop firm configuration guide
    - Physics metrics interpretation
    - Troubleshooting common issues
  - [ ] 14.5 Add logging configuration
    - Structured JSON logging
    - Log levels: DEBUG (calculations), INFO (decisions), WARNING (errors)
    - Log key metrics: Kelly fraction, physics multipliers, constraint source
  - [ ] 14.6 Create API documentation
    - RiskGovernor public API
    - Model schemas
    - Configuration reference
  - [ ] 14.7 Create README for risk package
    - Package overview and features
    - Installation instructions
    - Quick start example
    - API reference
  - [ ] 14.8 Update main project README
    - Add section on risk management system
    - Link to enhanced Kelly specification
    - Include architecture diagram

**Acceptance Criteria:**
- All code documented with docstrings
- Type hints on all functions
- User guide covers main use cases
- Logging provides audit trail

---

## Summary

**Total Task Groups:** 14
**Total Individual Tasks:** ~94 tasks (including subtasks)
**Estimated Implementation Time:** 3-4 weeks

**Key Deliverables:**
- Production-ready position sizing system
- 3 modular physics sensors (Ising, Chaos, Correlation)
- Monte Carlo validator for risk confirmation
- Physics-aware Kelly engine
- RiskGovernor orchestrator with caching
- MT5/MCP integration
- Comprehensive test suite (38-62 tests)
- Performance benchmarks
- Complete documentation

**Success Criteria:**
- All unit and integration tests pass
- Physics sensors detect market regimes accurately
- Position sizes respect broker and prop firm constraints
- Performance targets met (< 200ms calculation)
- Backtesting shows risk-adjusted return improvement vs standard Kelly

---

## Implementation Notes

**This tasks list follows the agent-OS standards:**
- Limited test writing during development (2-8 tests per task group)
- Focused on critical behaviors, not exhaustive coverage
- Strategic test gap analysis in Task Group 11
- Maximum of 10 additional tests to fill critical gaps
- Total test count kept reasonable (38-62 tests)

**Dependencies:**
- Quant-traderr-lab code will be refactored (not copied)
- mcp-metatrader5-server integration required
- Pydantic, NumPy, SciPy dependencies added

**Testing Strategy:**
- Development: 2-8 focused tests per task group (28-52 total)
- Gap analysis: Up to 10 additional strategic tests
- Total: 38-62 tests maximum
- Focus on critical workflows, not exhaustive edge cases

**Out of Scope (per spec):**
- Real-time data feed integration
- Order execution
- ML model training
- GUI/web interface
- Multi-asset portfolio optimization
- Options or derivatives pricing
- Hedging strategies
- Live paper trading integration
