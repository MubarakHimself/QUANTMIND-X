# Enhanced Kelly Position Sizing - API Reference

## Overview

The Enhanced Kelly Position Sizing API provides scientifically optimal position sizing with multi-layer safety protections. This document describes all public APIs, data models, and configuration options.

---

## Table of Contents

1. [Main API](#main-api)
2. [Kelly Calculator](#kelly-calculator)
3. [Kelly Analyzer](#kelly-analyzer)
4. [Portfolio Scaler](#portfolio-scaler)
5. [Data Models](#data-models)
6. [Configuration](#configuration)
7. [Edge Cases](#edge-cases)
8. [Logging](#logging)

---

## Main API

### `enhanced_kelly_position_size()`

Calculate optimal position size using Enhanced Kelly formula.

**Signature**:
```python
def enhanced_kelly_position_size(
    account_balance: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    current_atr: float,
    average_atr: float,
    stop_loss_pips: float,
    pip_value: float = 10.0,
    config: Optional[EnhancedKellyConfig] = None
) -> float
```

**Parameters**:
| Name | Type | Description | Example |
|------|------|-------------|---------|
| `account_balance` | `float` | Current account balance | `10000.0` |
| `win_rate` | `float` | Historical win rate (0-1) | `0.55` |
| `avg_win` | `float` | Average win in currency | `400.0` |
| `avg_loss` | `float` | Average loss in currency (positive) | `200.0` |
| `current_atr` | `float` | Current ATR value | `0.0012` |
| `average_atr` | `float` | 20-period ATR average | `0.0010` |
| `stop_loss_pips` | `float` | Stop loss distance | `20.0` |
| `pip_value` | `float` | Pip value per standard lot | `10.0` |
| `config` | `Optional[EnhancedKellyConfig]` | Configuration object | `None` |

**Returns**:
- `float`: Position size in lots

**Raises**:
- `ValueError`: If inputs are invalid

**Example**:
```python
from src.position_sizing.enhanced_kelly import enhanced_kelly_position_size

position = enhanced_kelly_position_size(
    account_balance=10000.0,
    win_rate=0.55,
    avg_win=400.0,
    avg_loss=200.0,
    current_atr=0.0012,
    average_atr=0.0010,
    stop_loss_pips=20.0,
    pip_value=10.0
)
print(f"Position: {position} lots")  # Output: 1.0 lots
```

---

## Kelly Calculator

### `EnhancedKellyCalculator`

Main calculator class with full result breakdown.

**Constructor**:
```python
def __init__(self, config: Optional[EnhancedKellyConfig] = None)
```

**Methods**:

#### `calculate()`

Calculate position size with detailed breakdown.

**Signature**:
```python
def calculate(
    self,
    account_balance: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    current_atr: float,
    average_atr: float,
    stop_loss_pips: float,
    pip_value: float = 10.0
) -> KellyResult
```

**Returns**: `KellyResult` object

**Example**:
```python
from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator

calculator = EnhancedKellyCalculator()
result = calculator.calculate(
    account_balance=10000.0,
    win_rate=0.55,
    avg_win=400.0,
    avg_loss=200.0,
    current_atr=0.0012,
    average_atr=0.0010,
    stop_loss_pips=20.0,
    pip_value=10.0
)

print(f"Position: {result.position_size} lots")
print(f"Risk: {result.kelly_f:.2%}")
print(f"Amount: ${result.risk_amount:.2f}")
```

---

## Kelly Analyzer

### `KellyStatisticsAnalyzer`

Extract Kelly parameters from trade history.

**Constructor**:
```python
def __init__(self, min_trades: int = 30)
```

**Methods**:

#### `calculate_kelly_parameters()`

Extract Kelly parameters from trade history.

**Signature**:
```python
def calculate_kelly_parameters(
    self,
    trade_history: List[Dict[str, Any]]
) -> KellyParameters
```

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `trade_history` | `List[Dict]` | List of trades with 'profit' or 'pnl' field |

**Returns**: `KellyParameters` object

**Example**:
```python
from src.position_sizing.kelly_analyzer import KellyStatisticsAnalyzer

analyzer = KellyStatisticsAnalyzer(min_trades=30)
trades = [
    {"profit": 500.0},
    {"profit": -200.0},
    {"profit": 600.0},
    # ... more trades
]

params = analyzer.calculate_kelly_parameters(trades)
print(f"Win rate: {params.win_rate:.1%}")
print(f"Base Kelly: {params.base_kelly_f:.2%}")
```

#### `calculate_rolling_kelly()`

Calculate Kelly parameters over rolling windows.

**Signature**:
```python
def calculate_rolling_kelly(
    self,
    trade_history: List[Dict[str, Any]],
    window_size: int = 50
) -> List[KellyParameters]
```

**Returns**: List of `KellyParameters` for each window

#### `detect_edge_decay()`

Detect if trading edge is decaying over time.

**Signature**:
```python
def detect_edge_decay(
    self,
    rolling_kelly: List[KellyParameters],
    decay_threshold: float = 0.15
) -> Dict[str, Any]
```

**Returns**: Dictionary with decay analysis

---

## Portfolio Scaler

### `PortfolioKellyScaler`

Manage position sizing across multiple bots.

**Constructor**:
```python
def __init__(
    self,
    max_portfolio_risk_pct: float = 0.03,
    correlation_adjustment: float = 1.5
)
```

**Methods**:

#### `scale_bot_positions()`

Scale down position sizes when multiple bots are active.

**Signature**:
```python
def scale_bot_positions(
    self,
    bot_kelly_factors: Dict[str, float],
    bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
) -> Dict[str, float]
```

**Parameters**:
| Name | Type | Description |
|------|------|-------------|
| `bot_kelly_factors` | `Dict[str, float]` | Bot ID → Kelly fraction |
| `bot_correlations` | `Optional[Dict]` | (bot1, bot2) → correlation |

**Returns**: Scaled Kelly fractions

**Example**:
```python
from src.position_sizing.portfolio_kelly import PortfolioKellyScaler

scaler = PortfolioKellyScaler(max_portfolio_risk_pct=0.03)

bot_requests = {
    "trend_bot": 0.015,
    "scalper_bot": 0.010,
    "swing_bot": 0.020
}

scaled = scaler.scale_bot_positions(bot_requests)
print(f"Scaled: {scaled}")
```

#### `get_portfolio_status()`

Return portfolio risk metrics for monitoring.

**Signature**:
```python
def get_portfolio_status(
    self,
    bot_kelly_factors: Dict[str, float],
    bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
) -> PortfolioStatus
```

**Returns**: `PortfolioStatus` object

#### `allocate_risk_equally()`

Allocate risk equally across all bots.

**Signature**:
```python
def allocate_risk_equally(
    self,
    bot_ids: List[str],
    total_risk_budget: Optional[float] = None
) -> Dict[str, float]
```

#### `allocate_risk_by_performance()`

Allocate risk based on bot performance.

**Signature**:
```python
def allocate_risk_by_performance(
    self,
    bot_performance: Dict[str, float],
    total_risk_budget: Optional[float] = None,
    min_allocation: float = 0.005
) -> Dict[str, float]
```

---

## Data Models

### `KellyResult`

Result of Enhanced Kelly calculation.

**Attributes**:
```python
@dataclass
class KellyResult:
    position_size: float           # Final position size in lots
    kelly_f: float                 # Final adjusted Kelly fraction
    base_kelly_f: float            # Raw Kelly before adjustments
    risk_amount: float             # Dollar amount at risk
    adjustments_applied: List[str] # List of calculation steps
    status: str                    # 'calculated', 'fallback', 'zero'
```

**Methods**:
- `to_dict() -> Dict`: Convert to dictionary for serialization

### `KellyParameters`

Extracted Kelly parameters from trade history.

**Attributes**:
```python
@dataclass
class KellyParameters:
    win_rate: float                # Win rate (0-1)
    avg_win: float                 # Average win amount
    avg_loss: float                # Average loss amount
    risk_reward_ratio: float       # R:R ratio
    base_kelly_f: float            # Base Kelly fraction
    sample_size: int               # Number of trades
    expectancy: float              # Average profit per trade
    profit_factor: float           # Total wins / Total losses
    is_reliable: bool              # Sufficient sample size
    confidence_note: str           # Confidence assessment
```

### `PortfolioStatus`

Portfolio risk status report.

**Attributes**:
```python
@dataclass
class PortfolioStatus:
    total_raw_risk: float          # Sum of all bot Kelly fractions
    total_scaled_risk: float       # After portfolio scaling
    risk_utilization: float        # Percentage of max allowed
    bot_count: int                 # Number of active bots
    status: str                    # 'safe', 'caution', 'danger'
    scale_factor: float            # Scaling factor applied
    recommendation: str            # Text recommendation
```

---

## Configuration

### `EnhancedKellyConfig`

Configuration for Enhanced Kelly position sizing.

**Attributes**:
```python
@dataclass
class EnhancedKellyConfig:
    # Layer 1: Kelly Fraction
    kelly_fraction: float = 0.50         # 50% of full Kelly

    # Layer 2: Hard Risk Cap
    max_risk_pct: float = 0.02           # 2% max per trade

    # Layer 3: Volatility Thresholds
    high_vol_threshold: float = 1.3      # ATR ratio for reduction
    low_vol_threshold: float = 0.7       # ATR ratio for increase
    low_vol_boost: float = 1.2           # Multiplier in calm markets

    # Data Requirements
    min_trade_history: int = 30          # Minimum trades for Kelly
    atr_period: int = 20                 # ATR averaging period
    confidence_interval: float = 0.95    # Statistical significance

    # Broker Constraints
    min_lot_size: float = 0.01
    lot_step: float = 0.01
    max_lot_size: float = 100.0

    # Safety
    allow_zero_position: bool = False    # Return min_lot or 0
    fallback_risk_pct: float = 0.01      # Fallback when insufficient data

    # Portfolio
    enable_portfolio_scaling: bool = True
    max_portfolio_risk_pct: float = 0.03  # 3% total daily risk
```

**Methods**:
- `validate() -> bool`: Validate configuration parameters

### `PropFirmPresets`

Pre-configured settings for popular prop firms.

**Methods**:
- `ftmo_challenge() -> EnhancedKellyConfig`
- `ftmo_funded() -> EnhancedKellyConfig`
- `the5ers() -> EnhancedKellyConfig`
- `personal_aggressive() -> EnhancedKellyConfig`
- `paper_trading() -> EnhancedKellyConfig`

---

## Edge Cases

### `EdgeCaseHandler`

Handles edge cases for Enhanced Kelly position sizing.

**Constructor**:
```python
def __init__(self, config: EnhancedKellyConfig)
```

**Methods**:

#### `handle_insufficient_history()`

Handle insufficient trade history.

**Signature**:
```python
def handle_insufficient_history(
    self,
    trade_count: int,
    fallback_risk_pct: Optional[float] = None
) -> tuple[float, str]
```

**Returns**: (risk_percentage, warning_message)

#### `handle_negative_expectancy()`

Handle negative or zero expectancy scenarios.

**Signature**:
```python
def handle_negative_expectancy(
    self,
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> tuple[float, str]
```

**Returns**: (position_size, reason_message)

#### `handle_flash_crash()`

Handle flash crash detection.

**Signature**:
```python
def handle_flash_crash(
    self,
    current_atr: float,
    average_atr: float,
    atr_suspicion_threshold: float = 1.5
) -> tuple[Optional[float], str]
```

**Returns**: (penalty_multiplier, warning_message)

### `SafetyValidator`

Validates position sizes against safety constraints.

**Constructor**:
```python
def __init__(self, config: EnhancedKellyConfig)
```

**Methods**:

#### `validate_position_size()`

Validate position size against all safety constraints.

**Signature**:
```python
def validate_position_size(
    self,
    position_size: float,
    account_balance: float,
    risk_amount: float,
    stop_loss_pips: float,
    pip_value: float
) -> tuple[bool, list[str]]
```

**Returns**: (is_valid, list_of_warnings)

#### `validate_account_state()`

Validate account state for safe trading.

**Signature**:
```python
def validate_account_state(
    self,
    balance: float,
    equity: float,
    margin: float,
    free_margin: float
) -> tuple[bool, list[str]]
```

**Returns**: (is_safe, list_of_warnings)

---

## Logging

### `PositionSizingLogger`

Logger for Enhanced Kelly operations.

**Constructor**:
```python
def __init__(self, name: str = "enhanced_kelly")
```

**Methods**:

#### `log_calculation()`

Log position sizing calculation.

**Signature**:
```python
def log_calculation(
    self,
    account_balance: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_f: float,
    position_size: float,
    risk_amount: float
)
```

#### `log_edge_case()`

Log edge case handling.

**Signature**:
```python
def log_edge_case(
    self,
    edge_case: str,
    details: Dict[str, Any],
    level: str = "warning"
)
```

#### `log_performance()`

Log performance metrics.

**Signature**:
```python
def log_performance(
    self,
    operation: str,
    duration_ms: float,
    details: Dict[str, Any] = None
)
```

### `setup_logging()`

Setup logging configuration.

**Signature**:
```python
def setup_logging(
    level: str = "INFO",
    log_file: str = "logs/enhanced_kelly.log",
    json_output: bool = True
)
```

---

## Constants

### Default Configuration Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KELLY_FRACTION` | `0.50` | 50% of full Kelly |
| `MAX_RISK_PCT` | `0.02` | 2% maximum per trade |
| `HIGH_VOL_THRESHOLD` | `1.3` | ATR ratio for high vol |
| `LOW_VOL_THRESHOLD` | `0.7` | ATR ratio for low vol |
| `LOW_VOL_BOOST` | `1.2` | 20% boost in calm markets |
| `MIN_TRADE_HISTORY` | `30` | Minimum trades |
| `MIN_LOT` | `0.01` | Minimum lot size |
| `LOT_STEP` | `0.01` | Lot step size |
| `MAX_LOT` | `100.0` | Maximum lot size |
| `FALLBACK_RISK_PCT` | `0.01` | 1% fallback |
| `MAX_PORTFOLIO_RISK_PCT` | `0.03` | 3% total daily risk |

### Performance Targets

| Metric | Target | Unit |
|--------|--------|------|
| Kelly calculation | < 50 | ms |
| Analyzer (small) | < 10 | ms |
| Analyzer (large) | < 50 | ms |
| Portfolio scaling | < 5 | ms |
| Full workflow | < 200 | ms |
| Memory usage | < 10 | MB |

---

## Type Hints

All functions and methods use Python type hints:

```python
from typing import Optional, List, Dict, Tuple, Any

def example_function(
    param1: float,
    param2: Optional[str] = None,
    param3: List[Dict[str, Any]] = None
) -> tuple[bool, str]:
    ...
```

---

## Exceptions

| Exception | When Raised |
|-----------|-------------|
| `ValueError` | Invalid input parameters |
| `ValidationError` | Configuration validation fails |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-28 | Initial release |

---

**For more information**, see:
- User Guide: `docs/user_guides/enhanced_kelly_user_guide.md`
- Test Suite: `tests/position_sizing/`
- Source Code: `src/position_sizing/`
