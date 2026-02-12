# Strategy Templates Documentation

## Overview

This directory contains strategy templates for the QuantMindX shared assets library. Each template provides a complete, production-ready trading strategy that can be customized for specific trading approaches.

## Available Strategy Templates

### 1. Trend Following Strategy - `trend_following.mq5`
- **Approach**: Moving average crossovers with momentum confirmation
- **Key Features**:
  - Configurable fast/slow moving averages (default: 10/20 EMA)
  - Multiple MA types (SMA, EMA, SMMA, LWMA)
  - Trailing stop functionality
  - Automatic opposite position closing
  - Risk-based position sizing
- **Best For**: 
  - Strong trending markets
  - Medium to long-term positions
  - Traders who prefer following market direction
- **Parameters**:
  - `InpMAPeriodFast`: Fast MA period (default: 10)
  - `InpMAPeriodSlow`: Slow MA period (default: 20)
  - `InpRiskPercent`: Risk percentage per trade (default: 1.0%)
  - `InpTakeProfitPips`: Take profit in pips (default: 50)
  - `InpStopLossPips`: Stop loss in pips (default: 25)

### 2. Mean Reversion Strategy - `mean_reversion.mq5`
- **Approach**: Bollinger Bands with RSI confirmation for overbought/oversold conditions
- **Key Features**:
  - Bollinger Bands for volatility measurement
  - RSI for momentum confirmation
  - Consecutive candle requirement for entry
  - Dynamic stop placement based on bands
  - Risk-based position sizing
- **Best For**:
  - Range-bound/ranging markets
  - Short to medium-term positions
  - Traders who profit from price returning to mean
- **Parameters**:
  - `InpBBPeriod`: Bollinger Bands period (default: 20)
  - `InpBBDeviations`: Standard deviations (default: 2.0)
  - `InpRSIPeriod`: RSI period (default: 14)
  - `InpOverbought`: RSI overbought level (default: 70)
  - `InpOversold`: RSI oversold level (default: 30)
  - `InpMinConsecutive`: Minimum consecutive candles (default: 3)

### 3. Breakout Strategy - `breakout.mq5`
- **Approach**: Range breakouts with volume confirmation
- **Key Features**:
  - Dynamic range calculation (high/low over N periods)
  - Volume confirmation for breakout validity
  - Cooldown period between trades
  - Automatic range reset after each breakout
  - Risk-based position sizing
- **Best For**:
  - Volatile markets with clear support/resistance
  - Medium-term positions
  - Traders who capitalize on momentum breakouts
- **Parameters**:
  - `InpRangePeriod`: Range calculation period (default: 20)
  - `InpVolumeMultiplier`: Volume confirmation multiplier (default: 1.5)
  - `InpRiskPercent`: Risk percentage per trade (default: 1.0%)
  - `InpCooldownPeriod`: Time between trades (default: 3600 seconds)

### 4. Scalping Strategy - `scalping.mq5`
- **Approach**: High-frequency trading with tight stops and quick profits
- **Key Features**:
  - Very tight stop losses and take profits
  - Price action confirmation with indicators
  - Maximum position limits
  - Tick-based cooldown
  - Consecutive win tracking for risk management
- **Best For**:
  - M1-M5 timeframes only
  - High-frequency trading
  - Traders with fast execution and low spreads
- **Parameters**:
  - `InpTimeframe`: Recommended timeframe (M1-M5)
  - `InpPipTarget`: Profit target in pips (default: 5)
  - `InpStopLossPips`: Stop loss in pips (default: 3)
  - `InpRiskPercent`: Risk percentage (default: 0.5%)
  - `InpMaxPositions`: Maximum concurrent positions (default: 3)

## Implementation Standards

### Code Quality
- **Zero Warnings**: All strategies compile with 0 warnings
- **Proper Error Handling**: Comprehensive validation and error checking
- **Resource Management**: Efficient memory usage and handle management
- **Performance**: Optimized execution with minimal CPU usage

### Risk Management
- **Position Sizing**: All strategies implement risk-based position sizing
- **Stop Losses**: Every order includes stop loss for risk control
- **Take Profits**: Profit targets defined for each strategy
- **Maximum Limits**: Position limits and cooldown periods to prevent over-trading

### Trading Logic
- **Clear Entry Rules**: Well-defined conditions for entering trades
- **Exit Management**: Automatic stop loss and take profit execution
- **Position Management**: Handling of multiple positions and reversals
- **Market Filtering**: Spread checks and market hour validation

## Usage Examples

### Basic Implementation
```mql5
// Simply attach any strategy template to a chart
// All parameters are configurable via Inputs tab
// Strategy will automatically begin trading based on its logic
```

### Customization
```mql5
// Modify parameters in the Inputs section:
// - Adjust risk percentage based on account size
// - Change timeframes for different market conditions
// - Modify indicator parameters for sensitivity
// - Adjust SL/TP levels based on volatility
```

### Integration with Indicators
```mql5
// All strategies can be enhanced with additional indicators
// from the indicators/ directory:

#include <indicators/rsi.mq5>
#include <indicators/atr.mq5>

// Use indicator functions for enhanced logic
double current_rsi = GetRSIValue();
double volatility = GetATRValue();
```

## Risk Management Features

### Universal Risk Controls
1. **Spread Filtering**: All strategies check spread before trading
2. **Market Hours**: Weekend and holiday filtering
3. **Position Limits**: Maximum concurrent positions
4. **Cooldown Periods**: Prevent over-trading with time delays

### Strategy-Specific Risk Features
- **Trend Following**: Trailing stops for profit protection
- **Mean Reversion**: Dynamic SL placement based on volatility
- **Breakout**: Volume confirmation to filter false breakouts
- **Scalping**: Consecutive win/loss tracking for position sizing adjustment

## Best Practices

### Before Live Trading
1. **Backtest thoroughly** with historical data
2. **Forward test** on demo account for 2-4 weeks
3. **Start with small positions** on live account
4. **Monitor performance** and adjust parameters
5. **Use appropriate risk management** (1-2% per trade max)

### Parameter Optimization
1. **Don't over-optimize** - use robust parameters
2. **Test across different market conditions**
3. **Validate with out-of-sample data**
4. **Consider transaction costs** in performance metrics
5. **Monitor for curve fitting**

### Market Conditions
- **Trend Following**: Best in strong trending markets
- **Mean Reversion**: Best in ranging/consolidation markets
- **Breakout**: Best in volatile markets with clear ranges
- **Scalping**: Best in liquid markets with tight spreads

## Version History

- **v1.0.0** (2026-02-02): Initial release with 4 strategy templates
  - Complete trend following strategy
  - Mean reversion with Bollinger Bands + RSI
  - Breakout strategy with volume confirmation
  - High-frequency scalping strategy

## Contributing

To add new strategy templates:
1. Follow the existing code structure and standards
2. Include comprehensive risk management
3. Add proper documentation and comments
4. Test with multiple symbols and timeframes
5. Update registry.json with new strategy metadata
6. Include performance benchmarks and recommendations