# Enhanced Kelly Position Sizing - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Prop Firm Presets](#prop-firm-presets)
7. [Edge Case Handling](#edge-case-handling)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [API Reference](#api-reference)

---

## Introduction

The Enhanced Kelly Position Sizing system provides **scientifically optimal position sizing** for trading strategies. It combines the **Kelly Criterion** (maximizing long-term growth) with **multi-layer safety protections** suitable for prop firm trading and retail accounts.

### Key Features

- **3-Layer Protection System**:
  1. Kelly Fraction (50% of full Kelly for safety)
  2. Hard Risk Cap (2% maximum per trade)
  3. Dynamic Volatility Adjustment (ATR-based scaling)

- **Portfolio Risk Management**: Auto-scales positions when multiple bots trade the same account
- **Prop Firm Presets**: Built-in configurations for FTMO, The5%ers, and more
- **Edge Case Handling**: Graceful handling of insufficient data, missing signals, and extreme market conditions
- **Performance Optimized**: < 200ms calculation time, < 10MB memory usage

### Mathematical Foundation

The Kelly Criterion formula:

$$f^* = \frac{bp - q}{b} = \frac{p(b+1) - 1}{b}$$

Where:
- $f^*$ = Fraction of capital to wager
- $p$ = Probability of winning (win rate)
- $q$ = Probability of losing ($1 - p$)
- $b$ = Ratio of average win to average loss (payoff ratio)

**Enhanced Kelly applies**:
1. Half-Kelly multiplier: $f_{enhanced} = f^* \times 0.5$
2. Hard cap: $f_{final} = \min(f_{enhanced}, 0.02)$ (2% max)
3. Volatility adjustment: Scale by ATR ratio

---

## Quick Start

### Installation

```bash
# The Enhanced Kelly system is part of QuantMindX
# No additional installation required
cd /path/to/QUANTMINDX
source venv/bin/activate
```

### Basic Usage

```python
from src.position_sizing.enhanced_kelly import enhanced_kelly_position_size

# Calculate position size
position_size = enhanced_kelly_position_size(
    account_balance=10000.0,      # $10,000 account
    win_rate=0.55,                 # 55% win rate
    avg_win=400.0,                 # $400 average win
    avg_loss=200.0,                # $200 average loss
    current_atr=0.0012,            # Current ATR (12 pips)
    average_atr=0.0010,            # Average ATR (10 pips)
    stop_loss_pips=20.0,           # 20 pip stop loss
    pip_value=10.0                 # $10 per pip (standard lot)
)

print(f"Position size: {position_size:.2f} lots")
# Output: Position size: 1.00 lots
```

### From Trade History

```python
from src.position_sizing.kelly_analyzer import KellyStatisticsAnalyzer
from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator

# 1. Analyze trade history
analyzer = KellyStatisticsAnalyzer(min_trades=30)
trade_history = [
    {"profit": 500.0},
    {"profit": -200.0},
    {"profit": 600.0},
    # ... more trades
]

params = analyzer.calculate_kelly_parameters(trade_history)
print(f"Win rate: {params.win_rate:.1%}")
print(f"Base Kelly: {params.base_kelly_f:.2%}")

# 2. Calculate position size
calculator = EnhancedKellyCalculator()
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

print(f"Position size: {result.position_size:.2f} lots")
print(f"Risk amount: ${result.risk_amount:.2f}")
print(f"Risk percentage: {result.kelly_f:.2%}")
```

---

## Core Concepts

### 1. The Kelly Criterion

The Kelly Criterion maximizes **long-term growth rate** while preventing ruin.

**Example**: If your strategy has:
- Win rate: 55%
- Average win: $400
- Average loss: $200

Calculation:
- Payoff ratio ($b$) = 400 / 200 = 2
- Kelly fraction = $(0.55 \times 3 - 1) / 2 = 0.325$ (32.5%)

**Problem**: Full Kelly is **too aggressive** for most traders:
- 30-40% drawdowns common
- Psychological stress
- Violates prop firm limits

**Solution**: **Half-Kelly** (16.25% in this example) captures ~70-80% of growth with 30-40% less drawdown.

### 2. 3-Layer Protection System

#### Layer 1: Kelly Fraction (50%)
```
f_base = kelly_f × 0.5
```
- Reduces full Kelly by 50%
- Balances growth vs. drawdown
- Configurable (0.4 - 0.7 typical)

#### Layer 2: Hard Risk Cap (2%)
```
f_capped = min(f_base, 0.02)
```
- Never exceeds 2% per trade
- Protects against over-leverage
- Configurable (1% - 3%)

#### Layer 3: Dynamic Volatility Adjustment
```
atr_ratio = current_atr / average_atr

if atr_ratio > 1.3:
    f_final = f_capped / atr_ratio  # Reduce position
elif atr_ratio < 0.7:
    f_final = f_capped × 1.2       # Increase position (conservative)
```
- Reduces risk in volatile markets
- Increases size in calm markets
- Prevents over-trading during chaos

### 3. Portfolio Scaling

When multiple bots trade one account:

```
Total Risk = Σ (bot_kelly_f)

if Total Risk > 3%:
    Scale Factor = 3% / Total Risk
    Each Bot: kelly_f × Scale Factor
```

**Example**:
- 5 bots each want 1% risk
- Total = 5% (dangerous!)
- Scale factor = 3% / 5% = 0.6
- Each bot gets: 1% × 0.6 = 0.6%

---

## Configuration

### Standard Configuration

```python
from src.position_sizing.kelly_config import EnhancedKellyConfig

config = EnhancedKellyConfig(
    # Layer 1: Kelly Fraction
    kelly_fraction=0.50,        # 50% of full Kelly

    # Layer 2: Hard Risk Cap
    max_risk_pct=0.02,          # 2% maximum per trade

    # Layer 3: Volatility Thresholds
    high_vol_threshold=1.3,     # ATR ratio > 1.3 = reduce
    low_vol_threshold=0.7,      # ATR ratio < 0.7 = increase
    low_vol_boost=1.2,          # 20% boost in calm markets

    # Data Requirements
    min_trade_history=30,       # Minimum trades for Kelly
    atr_period=20,              # ATR averaging period

    # Broker Constraints
    min_lot_size=0.01,
    lot_step=0.01,
    max_lot_size=100.0,

    # Safety
    allow_zero_position=False,  # Return min_lot if sizing fails
    fallback_risk_pct=0.01,     # 1% when insufficient data

    # Portfolio
    enable_portfolio_scaling=True,
    max_portfolio_risk_pct=0.03 # 3% total daily risk
)
```

### Prop Firm Presets

```python
from src.position_sizing.kelly_config import PropFirmPresets

# FTMO Challenge (ultra conservative)
ftmo_challenge = PropFirmPresets.ftmo_challenge()

# FTMO Funded (slightly more aggressive)
ftmo_funded = PropFirmPresets.ftmo_funded()

# The5%ers
the5ers = PropFirmPresets.the5ers()

# Personal account (aggressive)
personal = PropFirmPresets.personal_aggressive()

# Paper trading (test full Kelly)
paper = PropFirmPresets.paper_trading()
```

---

## Usage Examples

### Example 1: Profitable Strategy

```python
# Strategy: 55% win rate, 2:1 reward-risk
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

print(f"Position size: {result.position_size:.2f} lots")
print(f"Risk percentage: {result.kelly_f:.2%}")
print(f"Risk amount: ${result.risk_amount:.2f}")

# View calculation steps
for step in result.adjustments_applied:
    print(f"  {step}")
```

**Output**:
```
Position size: 1.00 lots
Risk percentage: 2.00%
Risk amount: $200.00
  R:R ratio = 2.00
  Base Kelly = 0.3250 (32.50%)
  Layer 1 (×0.5): 0.1625
  Layer 2 (capped at 2.0%): 0.0200
  Layer 3 (normal vol): 0.0200
  Risk amount = $200.00
  Raw position = 1.0000 lots
  Rounded position = 1.00 lots
```

### Example 2: High Volatility

```python
# Market volatility increased (ATR ratio = 2.0)
result = calculator.calculate(
    account_balance=10000.0,
    win_rate=0.55,
    avg_win=400.0,
    avg_loss=200.0,
    current_atr=0.0020,  # 20 pips (double normal)
    average_atr=0.0010,  # 10 pips
    stop_loss_pips=20.0,
    pip_value=10.0
)

print(f"Position size: {result.position_size:.2f} lots")
# Output: Position size: 0.50 lots (reduced by 50%)
```

### Example 3: Portfolio Management

```python
from src.position_sizing.portfolio_kelly import PortfolioKellyScaler

scaler = PortfolioKellyScaler(max_portfolio_risk_pct=0.03)

# 3 bots want to trade
bot_requests = {
    "trend_bot": 0.015,      # 1.5%
    "scalper_bot": 0.010,    # 1.0%
    "swing_bot": 0.020       # 2.0%
}
# Total = 4.5% (exceeds 3% limit)

scaled = scaler.scale_bot_positions(bot_requests)
print(f"Scaled positions: {scaled}")

# Get portfolio status
status = scaler.get_portfolio_status(bot_requests)
print(f"Status: {status.status}")
print(f"Risk utilization: {status.risk_utilization:.1%}")
print(f"Recommendation: {status.recommendation}")
```

**Output**:
```
Scaled positions: {'trend_bot': 0.010, 'scalper_bot': 0.007, 'swing_bot': 0.013}
Status: danger
Risk utilization: 100.0%
Recommendation: At risk limit - consider reducing bot count
```

### Example 4: Trade History Analysis

```python
analyzer = KellyStatisticsAnalyzer(min_trades=30)

# Load your trade history
trades = [
    {"profit": 500.0},
    {"profit": -200.0},
    {"profit": 600.0},
    # ... more trades
]

params = analyzer.calculate_kelly_parameters(trades)

print(f"Win Rate: {params.win_rate:.1%}")
print(f"Avg Win: ${params.avg_win:.2f}")
print(f"Avg Loss: ${params.avg_loss:.2f}")
print(f"R:R Ratio: {params.risk_reward_ratio:.2f}")
print(f"Base Kelly: {params.base_kelly_f:.2%}")
print(f"Expectancy: ${params.expectancy:.2f}")
print(f"Profit Factor: {params.profit_factor:.2f}")
print(f"Reliability: {params.confidence_note}")
```

---

## Prop Firm Presets

### FTMO Challenge

```python
config = PropFirmPresets.ftmo_challenge()
```

**Parameters**:
- Kelly Fraction: 40%
- Max Risk: 1% per trade
- High Vol Threshold: 1.2
- Low Vol Boost: 1.1
- Allow Zero Position: Yes

**Best For**: Passing challenge phase with conservative risk

### FTMO Funded

```python
config = PropFirmPresets.ftmo_funded()
```

**Parameters**:
- Kelly Fraction: 55%
- Max Risk: 1.5% per trade
- High Vol Threshold: 1.3
- Low Vol Boost: 1.2

**Best For**: Growing funded account with moderate risk

### The5%ers

```python
config = PropFirmPresets.the5ers()
```

**Parameters**:
- Kelly Fraction: 50%
- Max Risk: 2% per trade
- High Vol Threshold: 1.3
- Portfolio Risk: 4%

**Best For**: The5%ers account rules

### Personal Aggressive

```python
config = PropFirmPresets.personal_aggressive()
```

**Parameters**:
- Kelly Fraction: 60%
- Max Risk: 2.5% per trade
- High Vol Threshold: 1.5
- Low Vol Boost: 1.3

**Best For**: Personal accounts (experienced traders only)

---

## Edge Case Handling

### Insufficient Trade History

```python
# < 10 trades: VERY LOW confidence
# Use 0.5% conservative risk

# 10-29 trades: LOW confidence
# Use fallback risk (default 1%)

# 30+ trades: Reliable
# Use calculated Kelly
```

### Missing Physics Data

```python
# If physics sensors fail:
# - Apply 0.5x safety multiplier
# - Log warning
# - Continue with conservative sizing
```

### Flash Crashes

```python
# If ATR ratio > 1.5:
# - Immediate 50% risk reduction
# - Log critical warning
# - Revert to safety mode
```

### Negative Expectancy

```python
# If win rate × avg_win < loss rate × avg_loss:
# - Return zero position
# - Log: "No edge detected"
# - Do not trade
```

### Broker Constraints

```python
# Min Lot: Return 0 or min_lot (configurable)
# Max Lot: Cap at max_lot
# Lot Step: Round down to step size
# Margin: Reduce to fit free margin
```

---

## Performance Optimization

### Targets

- **Kelly Calculation**: < 50ms
- **Analyzer (small)**: < 10ms
- **Analyzer (large)**: < 50ms
- **Portfolio Scaling**: < 5ms
- **Full Workflow**: < 200ms
- **Memory Usage**: < 10MB

### Optimization Tips

1. **Cache Kelly Parameters**: Reuse analyzer results
2. **Batch Calculations**: Process multiple signals together
3. **Use Portfolio Scaler**: Auto-scale multiple bots
4. **Monitor Performance**: Run benchmarks regularly

### Benchmarking

```bash
# Run performance tests
pytest tests/position_sizing/test_performance.py -v -s
```

---

## Troubleshooting

### Issue: Position Size Always Zero

**Causes**:
- Negative expectancy (losing strategy)
- Insufficient trade history
- Configuration too conservative

**Solutions**:
- Check win rate and R:R ratio
- Collect more trade history (30+ trades)
- Adjust `allow_zero_position=False`

### Issue: Position Size Too Small

**Causes**:
- Low win rate
- Wide stop loss
- High volatility

**Solutions**:
- Improve strategy edge
- Tighten stop loss
- Wait for lower volatility
- Adjust `kelly_fraction` (carefully!)

### Issue: Portfolio Scaling Too Aggressive

**Causes**:
- Too many bots active
- High correlation between bots
- Max portfolio risk too low

**Solutions**:
- Reduce number of active bots
- Check correlation matrix
- Increase `max_portfolio_risk_pct`

### Issue: Performance Degradation

**Causes**:
- Large trade history (> 1000 trades)
- Rolling window calculations
- Concurrent requests

**Solutions**:
- Limit history size
- Use caching
- Profile with pytest-benchmark

---

## API Reference

### EnhancedKellyCalculator

```python
class EnhancedKellyCalculator:
    def __init__(self, config: EnhancedKellyConfig = None)
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

### KellyStatisticsAnalyzer

```python
class KellyStatisticsAnalyzer:
    def __init__(self, min_trades: int = 30)
    def calculate_kelly_parameters(
        self,
        trade_history: List[Dict[str, Any]]
    ) -> KellyParameters
    def calculate_rolling_kelly(
        self,
        trade_history: List[Dict[str, Any]],
        window_size: int = 50
    ) -> List[KellyParameters]
    def detect_edge_decay(
        self,
        rolling_kelly: List[KellyParameters],
        decay_threshold: float = 0.15
    ) -> Dict[str, Any]
```

### PortfolioKellyScaler

```python
class PortfolioKellyScaler:
    def __init__(
        self,
        max_portfolio_risk_pct: float = 0.03,
        correlation_adjustment: float = 1.5
    )
    def scale_bot_positions(
        self,
        bot_kelly_factors: Dict[str, float],
        bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[str, float]
    def get_portfolio_status(
        self,
        bot_kelly_factors: Dict[str, float],
        bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
    ) -> PortfolioStatus
```

---

## Additional Resources

- **Technical Documentation**: `docs/specs/enhanced_kelly_position_sizing_v1.md`
- **Test Suite**: `tests/position_sizing/`
- **Edge Case Handler**: `src/position_sizing/edge_cases.py`
- **Configuration**: `src/position_sizing/kelly_config.py`

---

**Version**: 1.0.0
**Last Updated**: 2026-01-28
**Author**: QuantMindX Team
