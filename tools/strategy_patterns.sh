#!/bin/bash
# Strategy Patterns Tool
# Retrieves common trading strategy patterns from the knowledge base
#
# Usage:
#   echo '{"pattern_type": "trend_following"}' | ./strategy_patterns.sh
#   ./strategy_patterns.sh "trend_following"
#
# Output: JSON with strategy pattern details
# Exit codes: 0 = success, 1 = error

set -euo pipefail

# Configuration
PAGEINDEX_URL="${PAGEINDEX_ARTICLES_URL:-http://localhost:3000}"

# Function to output JSON error
error_json() {
    echo "{\"error\": \"$1\", \"pattern\": null}"
    exit 1
}

# Strategy patterns database (fallback)
get_pattern_fallback() {
    local pattern_type="$1"
    
    case "$pattern_type" in
        trend_following)
            cat <<EOF
{
  "name": "Trend Following",
  "description": "Follow the direction of the market trend using moving averages or momentum indicators",
  "indicators": ["Moving Average", "MACD", "ADX"],
  "entry_conditions": [
    "Price above/below moving average",
    "MACD crossover in trend direction",
    "ADX > 25 indicating strong trend"
  ],
  "exit_conditions": [
    "Price crosses moving average opposite direction",
    "MACD zero line crossover",
    "Trailing stop hit"
  ],
  "risk_management": {
    "stop_loss": "ATR multiplier (2-3x)",
    "take_profit": "Risk:Reward 1:2 or higher",
    "position_size": "1-2% of account"
  },
  "timeframes": ["H1", "H4", "D1"],
  "best_for": "Strong trending markets"
}
EOF
            ;;
        mean_reversion)
            cat <<EOF
{
  "name": "Mean Reversion",
  "description": "Trade price returning to mean value using overbought/oversold indicators",
  "indicators": ["RSI", "Bollinger Bands", "Stochastic"],
  "entry_conditions": [
    "RSI < 30 for long / RSI > 70 for short",
    "Price at lower/upper Bollinger Band",
    "Stochastic oversold/overbought"
  ],
  "exit_conditions": [
    "RSI returns to neutral (50)",
    "Price returns to middle Bollinger Band",
    "Fixed stop loss/take profit"
  ],
  "risk_management": {
    "stop_loss": "Beyond recent swing high/low",
    "take_profit": "At middle band or mean price",
    "position_size": "1-2% of account"
  },
  "timeframes": ["M15", "H1", "H4"],
  "best_for": "Ranging or consolidating markets"
}
EOF
            ;;
        breakout)
            cat <<EOF
{
  "name": "Breakout",
  "description": "Trade price breaking through support or resistance levels",
  "indicators": ["Support/Resistance", "Volume", "ATR"],
  "entry_conditions": [
    "Price breaks above resistance for long",
    "Price breaks below support for short",
    "Volume confirmation on breakout",
    "Close beyond the level"
  ],
  "exit_conditions": [
    "Measured move target reached",
    "Price re-enters previous range",
    "Trailing stop after move"
  ],
  "risk_management": {
    "stop_loss": "Just inside the breakout level",
    "take_profit": "Equal to range width (measured move)",
    "position_size": "1% of account due to false breakout risk"
  },
  "timeframes": ["H1", "H4", "D1"],
  "best_for": "Consolidation followed by news or catalyst"
}
EOF
            ;;
        scalping)
            cat <<EOF
{
  "name": "Scalping",
  "description": "Quick trades capturing small price movements",
  "indicators": ["EMA", "RSI", "Order Flow"],
  "entry_conditions": [
    "EMA crossover for direction",
    "RSI extreme for entry timing",
    "Tight spread environment"
  ],
  "exit_conditions": [
    "Small profit target (5-10 pips)",
    "Quick stop out if trade goes against",
    "Time-based exit if no movement"
  ],
  "risk_management": {
    "stop_loss": "Tight (5-10 pips)",
    "take_profit": "Small (5-15 pips)",
    "position_size": "Consistent lot size"
  },
  "timeframes": ["M1", "M5", "M15"],
  "best_for": "High liquidity, low spread pairs"
}
EOF
            ;;
        *)
            cat <<EOF
{
  "error": "Unknown pattern type: $pattern_type",
  "available_patterns": ["trend_following", "mean_reversion", "breakout", "scalping"]
}
EOF
            ;;
    esac
}

# Function to get pattern from knowledge base or fallback
get_pattern() {
    local pattern_type="$1"
    
    # Try knowledge base first
    local response
    response=$(curl -s -X POST \
        "${PAGEINDEX_URL}/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"${pattern_type} strategy pattern\", \"namespace\": \"strategies\", \"limit\": 1}" \
        2>/dev/null) || {
        # Fallback to local patterns
        get_pattern_fallback "$pattern_type"
        return
    }
    
    # Check if we got results
    if echo "$response" | jq -e '.results | length > 0' >/dev/null 2>&1; then
        echo "$response" | jq '.results[0]'
    else
        # Fallback to local patterns
        get_pattern_fallback "$pattern_type"
    fi
}

# Main logic
if [ $# -ge 1 ]; then
    pattern_type="$1"
    get_pattern "$pattern_type"
elif [ ! -t 0 ]; then
    input=$(cat)
    pattern_type=$(echo "$input" | jq -r '.pattern_type // empty')
    
    if [ -z "$pattern_type" ]; then
        error_json "Missing 'pattern_type' field in input"
    fi
    
    get_pattern "$pattern_type"
else
    echo "{\"error\": \"Usage: $0 'pattern_type' OR echo '{\"pattern_type\": \"...\"}' | $0\", \"available_patterns\": [\"trend_following\", \"mean_reversion\", \"breakout\", \"scalping\"]}"
    exit 1
fi

exit 0