#!/bin/bash
# Indicator Template Tool
# Retrieves MQL5 indicator code templates
#
# Usage:
#   echo '{"indicator": "RSI"}' | ./indicator_template.sh
#   ./indicator_template.sh "MACD"
#
# Output: JSON with indicator template code
# Exit codes: 0 = success, 1 = error

set -euo pipefail

# Function to output JSON error
error_json() {
    echo "{\"error\": \"$1\", \"template\": null}"
    exit 1
}

# Indicator templates
get_indicator_template() {
    local indicator="$1"
    local indicator_upper=$(echo "$indicator" | tr '[:lower:]' '[:upper:]')
    
    case "$indicator_upper" in
        RSI)
            cat <<EOF
{
  "indicator": "RSI",
  "description": "Relative Strength Index - momentum oscillator measuring speed and change of price movements",
  "declaration": "int rsiHandle;\ndouble rsiBuffer[];",
  "initialization": "rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);\nArraySetAsSeries(rsiBuffer, true);",
  "on_tick": "CopyBuffer(rsiHandle, 0, 0, 3, rsiBuffer);\ndouble currentRSI = rsiBuffer[0];\ndouble previousRSI = rsiBuffer[1];",
  "parameters": {
    "period": {"default": 14, "type": "int", "description": "RSI period"},
    "applied_price": {"default": "PRICE_CLOSE", "type": "ENUM_APPLIED_PRICE", "description": "Price type for calculation"}
  },
  "usage_example": "// Check oversold\nif(currentRSI < 30) { /* Long signal */ }\n// Check overbought\nif(currentRSI > 70) { /* Short signal */ }",
  "signals": {
    "oversold": "RSI < 30",
    "overbought": "RSI > 70",
    "divergence": "Price makes new high/low but RSI doesn't"
  }
}
EOF
            ;;
        MACD)
            cat <<EOF
{
  "indicator": "MACD",
  "description": "Moving Average Convergence Divergence - trend-following momentum indicator",
  "declaration": "int macdHandle;\ndouble macdMainBuffer[], macdSignalBuffer[];",
  "initialization": "macdHandle = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);\nArraySetAsSeries(macdMainBuffer, true);\nArraySetAsSeries(macdSignalBuffer, true);",
  "on_tick": "CopyBuffer(macdHandle, 0, 0, 3, macdMainBuffer);\nCopyBuffer(macdHandle, 1, 0, 3, macdSignalBuffer);\ndouble macdMain = macdMainBuffer[0];\ndouble macdSignal = macdSignalBuffer[0];",
  "parameters": {
    "fast_period": {"default": 12, "type": "int", "description": "Fast EMA period"},
    "slow_period": {"default": 26, "type": "int", "description": "Slow EMA period"},
    "signal_period": {"default": 9, "type": "int", "description": "Signal line period"}
  },
  "usage_example": "// Bullish crossover\nif(macdMainBuffer[0] > macdSignalBuffer[0] && macdMainBuffer[1] <= macdSignalBuffer[1]) { /* Long signal */ }\n// Bearish crossover\nif(macdMainBuffer[0] < macdSignalBuffer[0] && macdMainBuffer[1] >= macdSignalBuffer[1]) { /* Short signal */ }",
  "signals": {
    "bullish_crossover": "MACD line crosses above signal line",
    "bearish_crossover": "MACD line crosses below signal line",
    "zero_line_cross": "MACD crosses zero line - trend confirmation"
  }
}
EOF
            ;;
        MA|MOVING_AVERAGE|SMA|EMA)
            cat <<EOF
{
  "indicator": "Moving Average",
  "description": "Moving Average - smooths price data to identify trend direction",
  "declaration": "int maHandle;\ndouble maBuffer[];",
  "initialization": "maHandle = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);\nArraySetAsSeries(maBuffer, true);",
  "on_tick": "CopyBuffer(maHandle, 0, 0, 3, maBuffer);\ndouble maValue = maBuffer[0];",
  "parameters": {
    "period": {"default": 20, "type": "int", "description": "MA period"},
    "shift": {"default": 0, "type": "int", "description": "Horizontal shift"},
    "method": {"default": "MODE_EMA", "type": "ENUM_MA_METHOD", "description": "SMA, EMA, SMMA, LWMA"},
    "applied_price": {"default": "PRICE_CLOSE", "type": "ENUM_APPLIED_PRICE", "description": "Price type"}
  },
  "usage_example": "// Price above MA - uptrend\nif(close > maValue) { /* Long bias */ }\n// Price below MA - downtrend\nif(close < maValue) { /* Short bias */ }",
  "signals": {
    "trend_direction": "Price above MA = uptrend, below = downtrend",
    "crossover": "Price crosses MA - potential trend change",
    "support_resistance": "MA acts as dynamic support/resistance"
  }
}
EOF
            ;;
        BB|BOLLINGER)
            cat <<EOF
{
  "indicator": "Bollinger Bands",
  "description": "Bollinger Bands - volatility indicator with upper/lower bands around moving average",
  "declaration": "int bbHandle;\ndouble bbUpperBuffer[], bbMiddleBuffer[], bbLowerBuffer[];",
  "initialization": "bbHandle = iBands(_Symbol, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE);\nArraySetAsSeries(bbUpperBuffer, true);\nArraySetAsSeries(bbMiddleBuffer, true);\nArraySetAsSeries(bbLowerBuffer, true);",
  "on_tick": "CopyBuffer(bbHandle, 0, 0, 3, bbMiddleBuffer);\nCopyBuffer(bbHandle, 1, 0, 3, bbUpperBuffer);\nCopyBuffer(bbHandle, 2, 0, 3, bbLowerBuffer);",
  "parameters": {
    "period": {"default": 20, "type": "int", "description": "MA period"},
    "deviation": {"default": 2.0, "type": "double", "description": "Standard deviation multiplier"}
  },
  "usage_example": "// Price at lower band - potential long\nif(close <= bbLowerBuffer[0]) { /* Long signal */ }\n// Price at upper band - potential short\nif(close >= bbUpperBuffer[0]) { /* Short signal */ }",
  "signals": {
    "oversold": "Price at or below lower band",
    "overbought": "Price at or above upper band",
    "squeeze": "Bands narrow - low volatility, breakout coming",
    "expansion": "Bands widen - high volatility"
  }
}
EOF
            ;;
        ATR)
            cat <<EOF
{
  "indicator": "ATR",
  "description": "Average True Range - measures market volatility",
  "declaration": "int atrHandle;\ndouble atrBuffer[];",
  "initialization": "atrHandle = iATR(_Symbol, PERIOD_CURRENT, 14);\nArraySetAsSeries(atrBuffer, true);",
  "on_tick": "CopyBuffer(atrHandle, 0, 0, 3, atrBuffer);\ndouble atrValue = atrBuffer[0];",
  "parameters": {
    "period": {"default": 14, "type": "int", "description": "ATR period"}
  },
  "usage_example": "// Calculate stop loss distance\ndouble stopLossDistance = atrValue * 2; // 2x ATR\nint stopLossPips = (int)(stopLossDistance / _Point / 10);",
  "signals": {
    "high_volatility": "ATR above average - larger stops needed",
    "low_volatility": "ATR below average - smaller stops possible",
    "stop_loss": "Use ATR multiplier for dynamic stops"
  }
}
EOF
            ;;
        ADX)
            cat <<EOF
{
  "indicator": "ADX",
  "description": "Average Directional Index - measures trend strength",
  "declaration": "int adxHandle;\ndouble adxBuffer[], adxPlusBuffer[], adxMinusBuffer[];",
  "initialization": "adxHandle = iADX(_Symbol, PERIOD_CURRENT, 14);\nArraySetAsSeries(adxBuffer, true);\nArraySetAsSeries(adxPlusBuffer, true);\nArraySetAsSeries(adxMinusBuffer, true);",
  "on_tick": "CopyBuffer(adxHandle, 0, 0, 3, adxBuffer);\nCopyBuffer(adxHandle, 1, 0, 3, adxPlusBuffer);\nCopyBuffer(adxHandle, 2, 0, 3, adxMinusBuffer);\ndouble adx = adxBuffer[0];",
  "parameters": {
    "period": {"default": 14, "type": "int", "description": "ADX period"}
  },
  "usage_example": "// Strong trend\nif(adx > 25) { /* Trend following strategy */ }\n// Weak/no trend\nif(adx < 20) { /* Range/mean reversion strategy */ }",
  "signals": {
    "strong_trend": "ADX > 25",
    "weak_trend": "ADX < 20",
    "strengthening": "ADX rising",
    "weakening": "ADX falling"
  }
}
EOF
            ;;
        STOCHASTIC|STOCH)
            cat <<EOF
{
  "indicator": "Stochastic",
  "description": "Stochastic Oscillator - momentum indicator comparing closing price to price range",
  "declaration": "int stochHandle;\ndouble stochMainBuffer[], stochSignalBuffer[];",
  "initialization": "stochHandle = iStochastic(_Symbol, PERIOD_CURRENT, 14, 3, 3, MODE_SMA, STO_LOWHIGH);\nArraySetAsSeries(stochMainBuffer, true);\nArraySetAsSeries(stochSignalBuffer, true);",
  "on_tick": "CopyBuffer(stochHandle, 0, 0, 3, stochMainBuffer);\nCopyBuffer(stochHandle, 1, 0, 3, stochSignalBuffer);",
  "parameters": {
    "k_period": {"default": 14, "type": "int", "description": "%K period"},
    "d_period": {"default": 3, "type": "int", "description": "%D period"},
    "slowing": {"default": 3, "type": "int", "description": "Slowing value"}
  },
  "usage_example": "// Oversold\nif(stochMainBuffer[0] < 20 && stochMainBuffer[1] >= 20) { /* Long signal */ }\n// Overbought\nif(stochMainBuffer[0] > 80 && stochMainBuffer[1] <= 80) { /* Short signal */ }",
  "signals": {
    "oversold": "%K < 20",
    "overbought": "%K > 80",
    "bullish_signal": "%K crosses above %D in oversold zone",
    "bearish_signal": "%K crosses below %D in overbought zone"
  }
}
EOF
            ;;
        *)
            cat <<EOF
{
  "error": "Unknown indicator: $indicator",
  "available_indicators": ["RSI", "MACD", "MA", "Bollinger", "ATR", "ADX", "Stochastic"]
}
EOF
            ;;
    esac
}

# Main logic
if [ $# -ge 1 ]; then
    indicator="$1"
    get_indicator_template "$indicator"
elif [ ! -t 0 ]; then
    input=$(cat)
    indicator=$(echo "$input" | jq -r '.indicator // empty')
    
    if [ -z "$indicator" ]; then
        error_json "Missing 'indicator' field in input"
    fi
    
    get_indicator_template "$indicator"
else
    echo "{\"error\": \"Usage: $0 'indicator' OR echo '{\"indicator\": \"...\"}' | $0\", \"available_indicators\": [\"RSI\", \"MACD\", \"MA\", \"Bollinger\", \"ATR\", \"ADX\", \"Stochastic\"]}"
    exit 1
fi

exit 0