# Specification: Enhanced Kelly Position Sizing with Econophysics Integration

## Goal

Create a production-ready position sizing system that integrates econophysics models (Ising phase transitions, Lyapunov chaos detection, RMT correlation filtering) with the Kelly Criterion to dynamically adjust risk based on market structure, providing superior risk-adjusted returns compared to standard Kelly implementations.

## User Stories

- As a quantitative trader, I want the system to automatically reduce position sizes during market phase transitions and chaotic periods so that I can protect capital during regime shifts
- As a prop firm trader, I want position sizes calculated with end-to-end lot conversion (Balance * Risk% / (SL_Pips * Pip_Value)) so that I can respect strict drawdown limits
- As a system architect, I want modular physics sensors that can be independently tested and swapped so that I can maintain and extend the system easily

## Specific Requirements

### 1. Directory Structure and Module Organization

- Create `src/risk/` as the root package for all risk management components
- Implement `src/risk/physics/` subdirectory containing econophysics sensor modules
- Implement `src/risk/sizing/` subdirectory containing position sizing logic
- Create `src/risk/governor.py` as the main entry point for position size calculation
- Create `src/risk/config.py` for all configuration constants and settings
- Add `src/risk/__init__.py` to expose the public API (RiskGovernor class)
- Follow Python package structure with proper relative imports

### 2. Physics Sensor Modules (Refactored from quant-traderr-lab)

**IsingRegimeSensor** (from Ising Pipeline.py)
- Extract the IsingSystem class and refactor into a modular sensor class
- Remove visualization and market data fetching code (keep physics core only)
- Input: Volatility stream (mapped to temperature)
- Output: Magnetic susceptibility (χ) and magnetization (M)
- Key method: `detect_regime(volatility: float) -> Dict[str, float]`
- Preserve the 3D lattice simulation with periodic boundary conditions
- Maintain the Metropolis-Hastings Monte Carlo algorithm
- Grid size: 12x12x12 (1728 agents) for performance
- Temperature range: 10.0 (chaos) to 0.1 (order)
- Critical threshold: χ > 0.8 indicates phase transition

**ChaosSensor** (from Lyapunov Pipeline.py)
- Extract phase space reconstruction and method of analogues logic
- Remove visualization code, keep only analysis metrics
- Input: Log-returns time series (N=300+ points)
- Output: Maximum Lyapunov exponent (λ), match distance, match index
- Key method: `analyze_chaos(returns: np.ndarray) -> Dict[str, float]`
- Preserve time delay embedding (dim=3, tau=12)
- Maintain KD-tree nearest neighbor search
- Calculate phase space trajectory similarity
- Critical threshold: λ > 0.5 indicates high chaos

**CorrelationSensor** (from RMT Pipeline.py)
- Extract RMT filtering and eigenvalue analysis logic
- Remove 3D rendering and video compilation code
- Input: Portfolio returns matrix (N assets × T periods)
- Output: Maximum eigenvalue (λ_max), noise threshold, denoised matrix
- Key method: `detect_systemic_risk(returns_matrix: np.ndarray) -> Dict[str, float]`
- Preserve Marchenko-Pastur distribution calculation
- Maintain PCA-based eigenvalue decomposition
- Critical threshold: λ_max > 2.0 indicates high correlation risk

### 3. Monte Carlo Validator (from Monte Carlo Pipeline.py)

**MonteCarloValidator** class
- Extract the MonteCarloEngine and bootstrap resampling logic
- Remove visualization code, keep statistical calculations only
- Input: Strategy performance metrics, proposed risk fraction
- Output: Validation result (pass/fail), risk of ruin probability
- Key method: `validate_risk(perf: StrategyPerformance, risk_pct: float, runs: int = 2000) -> ValidationResult`
- Preserve vectorized bootstrap resampling from historical returns
- Calculate 95% confidence intervals and VaR
- Count simulations hitting max drawdown limit (e.g., 10%)
- Risk threshold: Reject if probability of ruin > 0.5%
- Return adjusted risk (halved) if validation fails

### 4. Physics-Aware Kelly Engine

**PhysicsAwareKellyEngine** class
- Implement the master formula: S = B × f_base × M_physics
- Calculate base Kelly fraction from win rate and payoff ratio
- Apply half-Kelly safety scalar (K_fraction = 0.5)
- Calculate physics multipliers:
  - Lyapunov penalty: P_λ = max(0, 1.0 - (2.0 × λ))
  - Ising penalty: P_χ = 0.5 if χ > 0.8 else 1.0
  - Eigen penalty: P_E = min(1.0, 1.5 / λ_max) for λ_max > 1.5
- Aggregate penalties: M_physics = min(P_λ, P_χ, P_E) (weakest link principle)
- Track constraint source for logging (which physical law limited size)
- Optional Monte Carlo validation for meaningful sizes (>0.5%)
- Return SizingRecommendation with complete breakdown

### 5. RiskGovernor Main Entry Point

**RiskGovernor** class (governor.py)
- Orchestrates all sensors and the Kelly engine
- Provides simple API for position size calculation
- Key method: `calculate_position_size(account_info, strategy_perf, market_state) -> PositionSizingResult`
- End-to-end lot calculation: Lots = (Balance × Risk%) / (SL_Pips × Pip_Value)
- Integrate with mcp-metatrader5-server AccountManager for balance data
- Cache physics sensor results with TTL (e.g., 5 minutes) to avoid redundant calculations
- Handle edge cases: missing data, failed sensors, stale metrics
- Log all adjustments and constraint sources for audit trail
- Support multiple prop firm presets (FTMO, The5ers, etc.)
- Validate against broker constraints (min lot, max lot, lot step)

### 6. Configuration Management (config.py)

**Configuration constants and settings**
- Define LYAPUNOV_CHAOS_THRESHOLD = 0.5
- Define ISING_CRITICAL_SUSCEPTIBILITY = 0.8
- Define RMT_MAX_EIGEN_THRESHOLD = 1.5
- Define RMT_CRITICAL_EIGEN_THRESHOLD = 2.0
- Physics sensor cache TTL: 300 seconds (5 minutes)
- Monte Carlo simulation runs: 2000 (default)
- Max drawdown limit for validation: 10% (prop firm standard)
- Risk of ruin threshold: 0.5%
- Default Kelly fraction: 0.5 (half-Kelly)
- Per-trade risk cap: 2% (configurable)
- Broker lot constraints: min_lot=0.01, lot_step=0.01, max_lot=100.0
- Prop firm presets as configuration classes

### 7. Data Models (Pydantic)

**StrategyPerformance** model
- win_rate: float (0 to 1)
- avg_win: float (in currency)
- avg_loss: float (positive number in currency)
- total_trades: int
- profit_factor: float (avg_win / avg_loss)
- k_fraction: float = 0.5 (configurable safety scalar)

**MarketPhysics** model
- lyapunov_exponent: float (chaos metric)
- ising_susceptibility: float (phase transition metric)
- ising_magnetization: float (trend strength)
- rmt_max_eigenvalue: float (correlation metric)
- rmt_noise_threshold: float (Marchenko-Pastur bound)
- calculated_at: datetime (timestamp)
- is_stale: bool (age check)

**SizingRecommendation** model
- raw_kelly: float (pure Kelly % before adjustments)
- physics_multiplier: float (aggregate penalty 0.0-1.0)
- final_risk_pct: float (executable risk %)
- position_size_lots: float (end-to-end lot calculation)
- constraint_source: str (which physical law limited size)
- validation_passed: bool (Monte Carlo gatekeeper)
- adjustments_applied: list[str] (audit trail)

**PositionSizingResult** model (Governor output)
- Combine SizingRecommendation with execution details
- Include: account_balance, risk_amount, stop_loss_pips, pip_value
- Final rounded lot size respecting broker constraints
- Estimated margin requirement
- Remaining margin after trade
- All calculation steps for transparency

### 8. Integration with mcp-metatrader5-server

**AccountManager integration**
- Import AccountManager from mcp-metatrader5-server
- Use for fetching live account balance and equity
- Retrieve currency info for pip value calculations
- Support account switching via AccountManager.switch_account()
- Handle connection errors gracefully with fallback values
- Cache account info with short TTL (e.g., 10 seconds)

**Pip value calculation**
- Standard lot (1.00) = $10/pip for most forex pairs
- Cross pairs require calculation via base currency
- Gold (XAUUSD) = $1/pip for standard lot
- Indices vary by broker (include config override)
- Support custom pip_value parameter for exotic instruments

### 9. Testing Strategy

**Unit tests for each sensor**
- Test IsingRegimeSensor with known volatility inputs
- Verify phase transition detection at critical susceptibility
- Test ChaosSensor with synthetic chaotic vs periodic time series
- Verify Lyapunov exponent calculation accuracy
- Test CorrelationSensor with known correlation matrices
- Verify Marchenko-Pastur threshold calculation

**Unit tests for Kelly engine**
- Test Kelly calculation with various win rates and payoffs
- Verify negative expectancy handling (return zero or fallback)
- Test physics multiplier calculations independently
- Verify weakest link aggregation (min of all penalties)
- Test edge cases: zero loss history, zero payoff, extreme values

**Integration tests for Governor**
- Test end-to-end calculation with mock data
- Verify lot calculation formula accuracy
- Test broker constraint enforcement (min/max lot, rounding)
- Test Monte Carlo validation gatekeeper
- Verify prop firm preset application

**Backtesting framework**
- Compare Physics Kelly vs Standard Kelly on historical data
- Focus on crisis periods: COVID 2020, 2008, rate hikes
- Metric: Max drawdown reduction
- Metric: Risk-adjusted return (Sharpe) improvement
- Metric: Trade frequency during chaos (should reduce)

### 10. Edge Case Handling

**Data quality issues**
- Insufficient trade history (< 30 trades): Use conservative fallback (0.5-1% risk)
- Missing physics sensor data: Default multiplier to 0.5 (safety mode)
- Stale physics metrics (> cache TTL): Re-calculate or use safety mode
- Zero/negative payoff ratio: Return zero position with clear reason
- Zero average loss: Return zero position (division by zero protection)

**Market extreme events**
- Flash crash: Ising susceptibility spikes → immediate 50% risk reduction
- Gap opening: Skip trade if gap exceeds ATR threshold
- Low liquidity: Reduce position by correlation penalty if λ_max spikes
- Data feed outage: Use last known physics state with decay factor

**Broker constraints**
- Position size below minimum lot: Return 0 if allow_zero=True, else min_lot
- Position size exceeds maximum lot: Cap at max_lot and log warning
- Lot step rounding: Round down to avoid exceeding risk limit
- Insufficient margin: Reduce size to fit free margin

**Configuration validation**
- Validate all constants on module import
- Check prop firm preset consistency (max_risk_pct must be reasonable)
- Validate physics thresholds are within expected ranges
- Raise errors for invalid configurations during initialization

### 11. Performance Considerations

**Computational efficiency**
- Cache physics sensor results to avoid redundant simulations
- Use vectorized NumPy operations for all calculations
- Limit Ising grid size to 12x12x12 (sufficient accuracy, fast simulation)
- Use KD-tree for nearest neighbor search (O(log N) complexity)
- Parallelize Monte Carlo runs (embarrassingly parallel)

**Memory optimization**
- Limit time series length for phase space embedding (300 points max)
- Use generators for Monte Carlo simulations (avoid storing all paths)
- Clear cached physics data after TTL expires
- Release large arrays after calculations complete

**Latency targets**
- Physics sensor calculation: < 100ms per call
- Full position size calculation: < 200ms (with cached physics)
- Monte Carlo validation: < 500ms for 2000 runs
- Governor orchestrator overhead: < 50ms

### 12. Logging and Observability

**Structured logging**
- Log all calculation steps with input/output values
- Include constraint source and physics penalty breakdown
- Log Monte Carlo validation results
- Track cache hits/misses for sensors
- Use JSON format for log aggregation

**Metrics to expose**
- Current Kelly fraction (before and after physics)
- Physics multiplier values (P_λ, P_χ, P_E)
- Constraint source distribution (which limits most often)
- Monte Carlo pass/fail rate
- Average position size over time
- Risk reduction during chaotic periods

### 13. Documentation Requirements

**Code documentation**
- Docstrings for all classes and public methods
- Type hints for all function parameters and returns
- Inline comments for complex physics calculations
- Reference formulas with LaTeX in docstrings

**User documentation**
- Quick start guide with examples
- Prop firm configuration guide
- Physics metrics interpretation guide
- Troubleshooting common issues

## Out of Scope

- Real-time data feed integration (use existing MCP server or manual input)
- Order execution (position sizing only, not trade placement)
- Machine learning model training (only statistical inference)
- Multi-asset portfolio optimization (single strategy focus)
- GUI or web interface (API/CLI only)
- Backtesting engine implementation (use existing tools)
- Alternative position sizing methods (fixed ratio, etc.)
- Options or derivatives pricing (spot forex/CFD focus)
- Hedging strategies and correlated position sizing
- Live paper trading integration
- Performance benchmarking against brokers
