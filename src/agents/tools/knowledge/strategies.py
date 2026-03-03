# src/agents/tools/knowledge/strategies.py
"""Strategy-Specific Knowledge Tools."""

import logging
from typing import Dict, Any, Optional
from src.agents.tools.knowledge.client import call_pageindex_tool

logger = logging.getLogger(__name__)


async def search_strategy_patterns(
    pattern_type: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for trading strategy patterns and examples using PageIndex MCP.

    Args:
        pattern_type: Type of pattern (e.g., "entry", "exit", "risk_management")
        context: Optional context for more specific results

    Returns:
        Dictionary containing matching patterns
    """
    logger.info(f"Searching strategy patterns: {pattern_type}")

    # Build search query
    query = f"{pattern_type} trading strategy pattern"
    if context:
        query = f"{query} {context}"

    try:
        # Search strategies namespace
        result = await call_pageindex_tool("search", {
            "query": query,
            "namespace": "strategies",
            "max_results": 10,
            "include_content": True
        })

        patterns = []
        if isinstance(result, dict):
            for item in result.get("results", []):
                patterns.append({
                    "name": item.get("title", "Pattern"),
                    "description": item.get("content", "")[:200] if item.get("content") else "",
                    "code_snippet": item.get("code", ""),
                    "relevance": item.get("relevance", 0.0)
                })

        return {
            "success": True,
            "pattern_type": pattern_type,
            "patterns": patterns,
            "context": context,
            "total": len(patterns)
        }

    except Exception as e:
        logger.error(f"Strategy pattern search failed: {e}")
        # Return default patterns on error
        default_patterns = {
            "entry": [
                {
                    "name": "Moving Average Crossover",
                    "description": "Enter when fast MA crosses above slow MA",
                    "code_snippet": "if(fastMA > slowMA && previousFastMA <= previousSlowMA)",
                    "relevance": 0.9
                },
                {
                    "name": "RSI Oversold",
                    "description": "Enter when RSI crosses above 30 from below",
                    "code_snippet": "if(rsi > 30 && previousRSI <= 30)",
                    "relevance": 0.85
                }
            ],
            "exit": [
                {
                    "name": "Take Profit Target",
                    "description": "Exit at predefined price target",
                    "code_snippet": "if(currentPrice >= entryPrice + takeProfit)",
                    "relevance": 0.9
                }
            ],
            "risk_management": [
                {
                    "name": "Fixed Percentage Risk",
                    "description": "Risk fixed percentage of account per trade",
                    "code_snippet": "lotSize = (accountBalance * riskPercent) / (stopLossPips * pipValue)",
                    "relevance": 0.9
                }
            ]
        }

        return {
            "success": True,
            "pattern_type": pattern_type,
            "patterns": default_patterns.get(pattern_type, []),
            "context": context,
            "total": len(default_patterns.get(pattern_type, []))
        }


async def get_indicator_template(
    indicator_name: str
) -> Dict[str, Any]:
    """
    Get MQL5 code template for a specific indicator.

    First searches the MQL5 book, then falls back to built-in templates.

    Args:
        indicator_name: Name of the indicator (e.g., "RSI", "MACD", "MovingAverage")

    Returns:
        Dictionary containing indicator template code
    """
    logger.info(f"Getting indicator template: {indicator_name}")

    # Try to find in MQL5 book first
    try:
        result = await call_pageindex_tool("search", {
            "query": f"{indicator_name} indicator code example",
            "namespace": "mql5_book",
            "max_results": 3,
            "include_content": True
        })

        if isinstance(result, dict) and result.get("results"):
            # Found in MQL5 book
            best_match = result["results"][0]
            return {
                "success": True,
                "name": indicator_name,
                "code": best_match.get("content", ""),
                "source": "mql5_book",
                "parameters": []  # Would be extracted from content
            }
    except Exception as e:
        logger.warning(f"MQL5 book search for {indicator_name} failed: {e}")

    # Fall back to built-in templates
    templates = {
        "RSI": {
            "name": "Relative Strength Index",
            "code": """
int rsiHandle;
double rsiBuffer[];

int OnInit()
{
    rsiHandle = iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE);
    ArraySetAsSeries(rsiBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(rsiHandle, 0, 0, 3, rsiBuffer);
    double currentRSI = rsiBuffer[0];
    double previousRSI = rsiBuffer[1];
}
""",
            "parameters": ["period", "applied_price"]
        },
        "MACD": {
            "name": "Moving Average Convergence Divergence",
            "code": """
int macdHandle;
double macdMainBuffer[];
double macdSignalBuffer[];

int OnInit()
{
    macdHandle = iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE);
    ArraySetAsSeries(macdMainBuffer, true);
    ArraySetAsSeries(macdSignalBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(macdHandle, 0, 0, 3, macdMainBuffer);
    CopyBuffer(macdHandle, 1, 0, 3, macdSignalBuffer);
}
""",
            "parameters": ["fast_period", "slow_period", "signal_period", "applied_price"]
        },
        "MovingAverage": {
            "name": "Moving Average",
            "code": """
int maHandle;
double maBuffer[];

int OnInit()
{
    maHandle = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_EMA, PRICE_CLOSE);
    ArraySetAsSeries(maBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(maHandle, 0, 0, 3, maBuffer);
    double currentMA = maBuffer[0];
    double previousMA = maBuffer[1];
}
""",
            "parameters": ["period", "ma_shift", "ma_method", "applied_price"]
        },
        "BollingerBands": {
            "name": "Bollinger Bands",
            "code": """
int bbHandle;
double bbUpperBuffer[];
double bbMiddleBuffer[];
double bbLowerBuffer[];

int OnInit()
{
    bbHandle = iBands(_Symbol, PERIOD_CURRENT, 20, 0, 2.0, PRICE_CLOSE);
    ArraySetAsSeries(bbUpperBuffer, true);
    ArraySetAsSeries(bbMiddleBuffer, true);
    ArraySetAsSeries(bbLowerBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(bbHandle, 0, 0, 3, bbMiddleBuffer);
    CopyBuffer(bbHandle, 1, 0, 3, bbUpperBuffer);
    CopyBuffer(bbHandle, 2, 0, 3, bbLowerBuffer);
}
""",
            "parameters": ["period", "deviation", "applied_price"]
        },
        "ATR": {
            "name": "Average True Range",
            "code": """
int atrHandle;
double atrBuffer[];

int OnInit()
{
    atrHandle = iATR(_Symbol, PERIOD_CURRENT, 14);
    ArraySetAsSeries(atrBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(atrHandle, 0, 0, 3, atrBuffer);
    double currentATR = atrBuffer[0];
}
""",
            "parameters": ["period"]
        },
        "Stochastic": {
            "name": "Stochastic Oscillator",
            "code": """
int stochHandle;
double stochMainBuffer[];
double stochSignalBuffer[];

int OnInit()
{
    stochHandle = iStochastic(_Symbol, PERIOD_CURRENT, 5, 3, 3, MODE_SMA, STO_LOWHIGH);
    ArraySetAsSeries(stochMainBuffer, true);
    ArraySetAsSeries(stochSignalBuffer, true);
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    CopyBuffer(stochHandle, 0, 0, 3, stochMainBuffer);
    CopyBuffer(stochHandle, 1, 0, 3, stochSignalBuffer);
}
""",
            "parameters": ["k_period", "d_period", "slowing", "ma_method", "price_field"]
        }
    }

    # Find matching template
    for key, template in templates.items():
        if key.lower() == indicator_name.lower():
            return {
                "success": True,
                "name": template["name"],
                "code": template["code"],
                "source": "builtin",
                "parameters": template["parameters"]
            }

    # Not found
    return {
        "success": False,
        "name": indicator_name,
        "code": "",
        "source": "none",
        "error": f"Indicator template not found: {indicator_name}"
    }
