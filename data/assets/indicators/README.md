# Indicators Documentation

## Overview

This directory contains a collection of technical indicators implemented in MQL5 for the QuantMindX shared assets library. All indicators follow strict coding standards and are designed for robust, production-ready use.

## Available Indicators

### 1. Relative Strength Index (RSI) - `rsi.mq5`
- **Purpose**: Measures overbought/oversold conditions using price momentum
- **Key Features**:
  - Configurable period (default: 14)
  - Overbought/oversold levels (default: 70/30)
  - Multiple applied price options
  - Divergence detection
  - Zero-lag calculations
- **Signals**: 
  - Overbought/Oversold conditions
  - Bullish/Bearish divergence
  - Trend momentum analysis
- **Recommended Usage**: 
  - Confirming entry/exit points
  - Filtering trades during extreme market conditions
  - Divergence-based strategies

### 2. Moving Average Convergence Divergence (MACD) - `macd.mq5`
- **Purpose**: Identifies trend changes and momentum using moving averages
- **Key Features**:
  - Configurable fast (default: 12) and slow (default: 26) EMAs
  - Signal line (SMA) smoothing period (default: 9)
  - Histogram visualization
  - Crossover detection
  - Divergence analysis
- **Signals**:
  - MACD/Signal crossovers
  - Bullish/Bearish divergence
  - Trend strength measurement
  - Momentum confirmation
- **Recommended Usage**:
  - Trend following strategies
  - Momentum-based entry points
  - Filter for trend direction

### 3. Bollinger Bands - `bollinger_bands.mq5`
- **Purpose**: Measures volatility and identifies overbought/oversold conditions
- **Key Features**:
  - Configurable period (default: 20) and deviations (default: 2.0)
  - Multiple MA methods (SMA, EMA, etc.)
  - %B and bandwidth calculations
  - Breakout detection
  - Squeeze identification
- **Signals**:
  - Price touching/extreme bands
  - Breakouts above/below bands
  - Volatility contraction/expansion
  - %B for mean reversion timing
- **Recommended Usage**:
  - Mean reversion strategies
  - Breakout confirmation
  - Volatility-based position sizing
  - Squeeze trading strategies

### 4. Average True Range (ATR) - `atr.mq5`
- **Purpose**: Measures market volatility and assists with risk management
- **Key Features**:
  - Configurable period (default: 14)
  - True Range calculation using all three methods
  - Volatility percentile ranking
  - ATR-based stop loss calculation
  - Position sizing integration
- **Signals**:
  - Volatility level assessment
  - Volatility contraction/expansion
  - Dynamic stop loss placement
  - Position size optimization
- **Recommended Usage**:
  - Dynamic risk management
  - Stop loss calculation
  - Volatility filtering
  - Position sizing models

## Implementation Standards

### Code Quality
- **Zero Warnings**: All indicators compile with 0 warnings in MetaEditor
- **Proper Error Handling**: Comprehensive input validation
- **Resource Management**: Efficient memory usage
- **Performance**: Optimized calculations avoiding unnecessary repetition

### Documentation
- **Inline Comments**: Detailed explanation of logic and calculations
- **Function Documentation**: Clear parameter descriptions and return values
- **Usage Examples**: Practical implementation guidance
- **Edge Cases**: Handling of boundary conditions

### Integration Features
- **Standard Interface**: Consistent function naming and parameters
- **Configurable Parameters**: Flexible input settings via `input` variables
- **Multiple Timeframes**: Support for different chart timeframes
- **Applied Prices**: Support for various price types (close, open, high, low, etc.)

## Usage Examples

### Loading an Indicator
```mql5
// In your strategy EA
#property strict
#include <indicators/rsi.mq5>

// Create indicator handle
int rsi_handle = iCustom(NULL, 0, "rsi", 14, 70, 30, PRICE_CLOSE);

// Get current RSI value
double rsi_value = iCustom(NULL, 0, "rsi", 14, 70, 30, PRICE_CLOSE, 0, 0);
```

### Using Indicator Functions
```mql5
// Direct function calls (if indicator is included)
double current_rsi = GetRSIValue();
bool is_overbought = IsOverbought();
bool is_oversold = IsOversold();

// MACD example
int crossover_signal = GetCrossoverSignal();
double trend_strength = GetTrendStrength();

// Bollinger Bands example
double bandwidth = GetBandwidth();
bool breakout_above = IsBreakoutAbove(Bid);
```

## Risk Management Integration

### ATR-Based Position Sizing
```mql5
#include <indicators/atr.mq5>

double CalculatePositionSize(double account_balance, 
                           double risk_percent = 1.0,
                           double atr_multiplier = 1.5)
{
    return CalculatePositionSize(account_balance, risk_percent, atr_multiplier);
}
```

### Volatility Filtering
```mql5
#include <indicators/bollinger_bands.mq5>
#include <indicators/atr.mq5>

// Only trade when volatility is within acceptable range
if(GetVolatilityLevel() == 2) // Normal volatility
{
    // Execute trading logic
}
```

## Best Practices

1. **Always validate inputs** before using indicator values
2. **Check for sufficient historical data** before calculations
3. **Use appropriate timeframes** for your strategy
4. **Combine multiple indicators** for confirmation
5. **Test thoroughly** with different market conditions
6. **Monitor performance** and adjust parameters as needed

## Version History

- **v1.0.0** (2026-02-02): Initial release with 4 core indicators
  - RSI with divergence detection
  - MACD with histogram and crossover signals
  - Bollinger Bands with %B and squeeze detection
  - ATR with volatility analysis and position sizing

## Contributing

To add new indicators:
1. Follow the existing code structure and standards
2. Include comprehensive documentation
3. Add proper error handling
4. Test with multiple symbols and timeframes
5. Update registry.json with new indicator metadata