"""
Helper functions for QuantCode agent.

Contains all extraction and generation helper functions.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _extract_strategy_type(trd_content: str) -> str:
    """Extract strategy type from TRD content."""
    if not trd_content:
        return "Momentum"

    content_lower = trd_content.lower()
    if "mean reversion" in content_lower:
        return "Mean Reversion"
    elif "trend following" in content_lower:
        return "Trend Following"
    elif "breakout" in content_lower:
        return "Breakout"
    elif "scalping" in content_lower:
        return "Scalping"
    else:
        return "Momentum"


def _extract_indicators(trd_content: str) -> List[str]:
    """Extract indicators from TRD content."""
    if not trd_content:
        return ["RSI", "MACD", "Moving Average"]

    indicators = []
    content_lower = trd_content.lower()

    indicator_keywords = {
        "rsi": "RSI",
        "relative strength": "RSI",
        "macd": "MACD",
        "moving average": "Moving Average",
        "ma ": "Moving Average",
        "bollinger": "Bollinger Bands",
        "stochastic": "Stochastic",
        "atr": "ATR",
        "average true range": "ATR",
        "cci": "CCI",
        "adx": "ADX"
    }

    for keyword, indicator in indicator_keywords.items():
        if keyword in content_lower and indicator not in indicators:
            indicators.append(indicator)

    return indicators if indicators else ["RSI", "MACD", "Moving Average"]


def _extract_entry_rules(trd_content: str) -> List[str]:
    """Extract entry rules from TRD content."""
    if not trd_content:
        return ["RSI < 30", "MACD crossover"]

    # Simple extraction - look for common patterns
    rules = []
    content_lower = trd_content.lower()

    if "oversold" in content_lower or "rsi" in content_lower:
        rules.append("RSI < 30 (oversold)")
    if "overbought" in content_lower:
        rules.append("RSI > 70 (overbought)")
    if "crossover" in content_lower or "cross" in content_lower:
        rules.append("MACD signal crossover")
    if "breakout" in content_lower:
        rules.append("Price breaks resistance/support")

    return rules if rules else ["RSI < 30", "MACD crossover"]


def _extract_exit_rules(trd_content: str) -> List[str]:
    """Extract exit rules from TRD content."""
    if not trd_content:
        return ["RSI > 70", "Stop loss at 2%"]

    rules = []
    content_lower = trd_content.lower()

    if "take profit" in content_lower:
        rules.append("Take profit target reached")
    if "stop loss" in content_lower:
        rules.append("Stop loss triggered")
    if "trailing" in content_lower:
        rules.append("Trailing stop activated")

    return rules if rules else ["RSI > 70", "Stop loss at 2%"]


def _extract_risk_management(trd_content: str) -> str:
    """Extract risk management rules from TRD content."""
    if not trd_content:
        return "Kelly Criterion with 0.8 threshold"

    content_lower = trd_content.lower()

    if "kelly" in content_lower:
        return "Kelly Criterion position sizing"
    elif "fixed lot" in content_lower:
        return "Fixed lot size"
    elif "percentage" in content_lower:
        return "Percentage of equity"
    else:
        return "Risk per trade: 1-2% of account"


def _extract_timeframes(trd_content: str) -> List[str]:
    """Extract timeframes from TRD content."""
    if not trd_content:
        return ["H1"]

    timeframes = []
    content = trd_content.upper()

    tf_patterns = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
    for tf in tf_patterns:
        if tf in content:
            timeframes.append(tf)

    return timeframes if timeframes else ["H1"]


def _extract_symbols(trd_content: str) -> List[str]:
    """Extract trading symbols from TRD content."""
    if not trd_content:
        return ["EURUSD"]

    symbols = []
    content_upper = trd_content.upper()

    common_symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD"]
    for symbol in common_symbols:
        if symbol in content_upper:
            symbols.append(symbol)

    return symbols if symbols else ["EURUSD"]


def _generate_mql5_code(plan: Dict[str, Any], trd_content: str) -> str:
    """Generate MQL5 code from strategy plan."""
    strategy_type = plan.get('strategy_type', 'ExpertAdvisor') if isinstance(plan, dict) else 'ExpertAdvisor'
    indicators = plan.get('indicators', ['RSI', 'MACD']) if isinstance(plan, dict) else ['RSI', 'MACD']
    entry_rules = plan.get('entry_rules', []) if isinstance(plan, dict) else []
    exit_rules = plan.get('exit_rules', []) if isinstance(plan, dict) else []
    risk_mgmt = plan.get('risk_management', '') if isinstance(plan, dict) else ''

    # Generate indicator declarations
    indicator_code = ""
    indicator_init = ""
    indicator_on_tick = ""

    for ind in indicators:
        if ind == "RSI":
            indicator_code += "int rsiHandle;\ndouble rsiBuffer[];\n"
            indicator_init += "rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);\nArraySetAsSeries(rsiBuffer, true);\n"
            indicator_on_tick += "CopyBuffer(rsiHandle, 0, 0, 3, rsiBuffer);\n"
        elif ind == "MACD":
            indicator_code += "int macdHandle;\ndouble macdMainBuffer[], macdSignalBuffer[];\n"
            indicator_init += "macdHandle = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);\nArraySetAsSeries(macdMainBuffer, true);\nArraySetAsSeries(macdSignalBuffer, true);\n"
            indicator_on_tick += "CopyBuffer(macdHandle, 0, 0, 3, macdMainBuffer);\nCopyBuffer(macdHandle, 1, 0, 3, macdSignalBuffer);\n"
        elif ind == "Moving Average":
            indicator_code += "int maHandle;\ndouble maBuffer[];\n"
            indicator_init += "maHandle = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);\nArraySetAsSeries(maBuffer, true);\n"
            indicator_on_tick += "CopyBuffer(maHandle, 0, 0, 3, maBuffer);\n"

    # Generate entry/exit logic
    entry_logic = "// Entry conditions\n"
    for rule in entry_rules:
        entry_logic += f"// {rule}\n"

    exit_logic = "// Exit conditions\n"
    for rule in exit_rules:
        exit_logic += f"// {rule}\n"

    code = f"""
//+------------------------------------------------------------------+
//|                                    {strategy_type.replace(' ', '_')}.mq5 |
//|                                  Generated by QuantMind QuantCode |
//+------------------------------------------------------------------+
#property copyright "QuantMind"
#property version   "1.00"

#include <Trade\\Trade.mqh>

// Input parameters
input double LotSize = 0.1;
input int StopLossPips = 50;
input int TakeProfitPips = 100;
input double RiskPercent = 1.0;

// Indicator handles and buffers
{indicator_code}

// Trade object
CTrade trade;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{{
    // Initialize indicators
    {indicator_init}

    Print("{strategy_type} initialized successfully");
    return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
    // Release indicator handles
    if(rsiHandle != INVALID_HANDLE) IndicatorRelease(rsiHandle);
    if(macdHandle != INVALID_HANDLE) IndicatorRelease(macdHandle);
    if(maHandle != INVALID_HANDLE) IndicatorRelease(maHandle);
}}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{{
    // Check for new bar
    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);

    if(lastBarTime == currentBarTime) return;
    lastBarTime = currentBarTime;

    // Update indicator values
    {indicator_on_tick}

    // Get current values
    double currentRSI = rsiBuffer[0];
    double previousRSI = rsiBuffer[1];
    double currentMACD = macdMainBuffer[0];
    double currentSignal = macdSignalBuffer[0];

    // Check if we have an open position
    if(PositionsTotal() > 0)
    {{
        {exit_logic}

        // Exit logic: RSI overbought for long positions
        if(currentRSI > 70)
        {{
            trade.PositionClose(_Symbol);
            Print("Exit signal: RSI overbought");
        }}
    }}
    else
    {{
        {entry_logic}

        // Entry logic: RSI oversold + MACD bullish
        if(currentRSI < 30 && previousRSI >= 30)
        {{
            // Calculate stop loss and take profit
            double sl = SymbolInfoDouble(_Symbol, SYMBOL_BID) - StopLossPips * _Point;
            double tp = SymbolInfoDouble(_Symbol, SYMBOL_BID) + TakeProfitPips * _Point;

            if(trade.Buy(LotSize, _Symbol, 0, sl, tp, "Entry: RSI oversold"))
            {{
                Print("Buy order placed successfully");
            }}
        }}

        // Entry logic: MACD crossover
        if(currentMACD > currentSignal && macdMainBuffer[1] <= macdSignalBuffer[1])
        {{
            double sl = SymbolInfoDouble(_Symbol, SYMBOL_ASK) - StopLossPips * _Point;
            double tp = SymbolInfoDouble(_Symbol, SYMBOL_ASK) + TakeProfitPips * _Point;

            if(trade.Buy(LotSize, _Symbol, 0, sl, tp, "Entry: MACD crossover"))
            {{
                Print("Buy order placed successfully");
            }}
        }}
    }}
}}

//+------------------------------------------------------------------+
//| Calculate position size based on risk                              |
//+------------------------------------------------------------------+
double CalculatePositionSize(double riskPercent, double stopLossPips)
{{
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * riskPercent / 100;
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);

    if(tickValue == 0 || tickSize == 0) return LotSize;

    double pipValue = tickValue * 10; // Approximate pip value
    double lotSize = riskAmount / (stopLossPips * pipValue);

    // Normalize to broker's lot step
    double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    lotSize = MathFloor(lotSize / lotStep) * lotStep;

    // Check min/max lots
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);

    if(lotSize < minLot) lotSize = minLot;
    if(lotSize > maxLot) lotSize = maxLot;

    return lotSize;
}}
"""
    return code


def _attempt_code_fix(code: str, errors: List[str]) -> str:
    """Attempt to fix compilation errors in code."""
    # Simple error fixing - add missing semicolons, fix common issues
    fixed_code = code

    for error in errors:
        if "semicolon" in error.lower():
            # Try to add missing semicolons
            pass
        elif "undeclared" in error.lower():
            # Try to add missing declarations
            pass

    return fixed_code


def _calculate_kelly_score(win_rate: float, max_drawdown: float, sharpe_ratio: float) -> float:
    """Calculate Kelly score for strategy validation."""
    if win_rate <= 0 or max_drawdown >= 1:
        return 0.0

    # Simplified Kelly calculation
    # Kelly = W - (1-W)/R where W is win rate and R is win/loss ratio
    # We use a modified version that considers drawdown and sharpe

    base_kelly = win_rate - (1 - win_rate) / max(0.1, sharpe_ratio)

    # Adjust for drawdown
    drawdown_penalty = max_drawdown * 0.5

    # Final score
    kelly_score = max(0, min(1, base_kelly - drawdown_penalty))

    return kelly_score
