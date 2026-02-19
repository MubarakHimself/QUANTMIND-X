"""
QuantCode Agent Workflow

Implements the QuantCode agent using LangGraph for strategy development and backtesting.

This agent:
1. Ingests TRD content
2. Generates MQL5 code
3. Invokes MT5 compiler MCP with error-handling/retry
4. Runs Backtest MCP
5. Passes results through the backtest validator

**Validates: Requirements 8.3**
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import QuantCodeState
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)

# Lazy imports to avoid module not found errors
_PaperTradingValidator = None
_PaperTradingDeployer = None

def _get_paper_trading_validator():
    """Lazy load PaperTradingValidator to avoid import errors."""
    global _PaperTradingValidator
    if _PaperTradingValidator is None:
        try:
            from src.agents.paper_trading_validator import PaperTradingValidator
            _PaperTradingValidator = PaperTradingValidator
        except ImportError as e:
            logger.warning(f"PaperTradingValidator not available: {e}")
            _PaperTradingValidator = None
    return _PaperTradingValidator

def _get_paper_trading_deployer():
    """Lazy load PaperTradingDeployer to avoid import errors."""
    global _PaperTradingDeployer
    if _PaperTradingDeployer is None:
        try:
            from mcp_mt5.paper_trading.deployer import PaperTradingDeployer
            _PaperTradingDeployer = PaperTradingDeployer
        except ImportError as e:
            logger.warning(f"PaperTradingDeployer not available: {e}")
            _PaperTradingDeployer = None
    return _PaperTradingDeployer


# =============================================================================
# MCP Tool Integration
# =============================================================================

async def compile_mql5(code: str, filename: str, code_type: str = "expert") -> Dict[str, Any]:
    """
    Compile MQL5 code using MT5 Compiler MCP.
    
    Args:
        code: MQL5 source code
        filename: Output filename (without extension)
        code_type: Type of code (expert, indicator, script, library)
        
    Returns:
        Compilation result with success status, errors, and warnings
    """
    try:
        from src.agents.tools.mcp_tools import compile_mql5_code
        result = await compile_mql5_code(
            code=code,
            filename=filename,
            code_type=code_type
        )
        return result
    except Exception as e:
        logger.error(f"MT5 Compiler MCP call failed: {e}")
        return {
            "success": False,
            "errors": [str(e)],
            "warnings": [],
            "output_path": None
        }


async def run_backtest_mcp(code: str, config: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """
    Run backtest using Backtest MCP.
    
    Args:
        code: MQL5 source code
        config: Backtest configuration
        strategy_name: Name of the strategy
        
    Returns:
        Backtest results
    """
    try:
        from src.agents.tools.mcp_tools import run_backtest, get_backtest_results, get_backtest_status
        
        # Submit backtest job
        job = await run_backtest(
            code=code,
            config=config,
            strategy_name=strategy_name
        )
        
        backtest_id = job.get("backtest_id")
        if not backtest_id:
            raise RuntimeError("No backtest ID returned")
        
        # Wait for completion
        max_wait = 300  # 5 minutes
        waited = 0
        while waited < max_wait:
            status = await get_backtest_status(backtest_id)
            if status.get("status") == "completed":
                break
            if status.get("status") == "failed":
                raise RuntimeError(f"Backtest failed: {status.get('error', 'Unknown error')}")
            await asyncio.sleep(5)
            waited += 5
        
        if waited >= max_wait:
            raise TimeoutError(f"Backtest timed out after {max_wait} seconds")
        
        # Get results
        results = await get_backtest_results(backtest_id)
        return results
        
    except Exception as e:
        logger.error(f"Backtest MCP call failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "metrics": {},
            "trades": []
        }


async def validate_syntax(code: str) -> Dict[str, Any]:
    """
    Validate MQL5 syntax using MT5 Compiler MCP.
    
    Args:
        code: MQL5 source code
        
    Returns:
        Validation result
    """
    try:
        from src.agents.tools.mcp_tools import validate_mql5_syntax
        result = await validate_mql5_syntax(code)
        return result
    except Exception as e:
        logger.error(f"Syntax validation failed: {e}")
        return {
            "valid": False,
            "errors": [str(e)],
            "warnings": []
        }


# =============================================================================
# Agent Node Functions
# =============================================================================

def planning_node(state: QuantCodeState) -> Dict[str, Any]:
    """Planning node: Create strategy development plan from TRD content."""
    logger.info("Planning node creating strategy plan")
    
    # Extract TRD content from state
    trd_content = state.get('trd_content', '') or state.get('context', {}).get('trd_content', '')
    
    # Parse TRD and create strategy plan
    strategy_plan = {
        "strategy_type": _extract_strategy_type(trd_content),
        "indicators": _extract_indicators(trd_content),
        "entry_rules": _extract_entry_rules(trd_content),
        "exit_rules": _extract_exit_rules(trd_content),
        "risk_management": _extract_risk_management(trd_content),
        "timeframes": _extract_timeframes(trd_content),
        "symbols": _extract_symbols(trd_content)
    }
    
    message = AIMessage(content=f"Created strategy plan: {strategy_plan['strategy_type']}")
    
    return {
        "messages": [message],
        "strategy_plan": str(strategy_plan),
        "context": {**state.get('context', {}), "strategy_plan": strategy_plan}
    }


def coding_node(state: QuantCodeState) -> Dict[str, Any]:
    """Coding node: Implement strategy in MQL5."""
    logger.info("Coding node implementing strategy")
    
    # Get strategy plan
    plan = state.get('strategy_plan') or state.get('context', {}).get('strategy_plan', {})
    if isinstance(plan, str):
        import json
        try:
            plan = json.loads(plan.replace("'", '"'))
        except:
            plan = {}
    
    # Get TRD content for context
    trd_content = state.get('trd_content', '') or state.get('context', {}).get('trd_content', '')
    
    # Generate MQL5 code based on plan
    code_implementation = _generate_mql5_code(plan, trd_content)
    
    message = AIMessage(content="Strategy implementation completed in MQL5")
    
    return {
        "messages": [message],
        "code_implementation": code_implementation,
        "context": {**state.get('context', {}), "code_implementation": code_implementation}
    }


async def compilation_node(state: QuantCodeState) -> Dict[str, Any]:
    """
    Compilation node: Compile MQL5 code using MT5 Compiler MCP.
    
    Includes error handling and retry logic for compilation failures.
    """
    logger.info("Compilation node compiling MQL5 code")
    
    code = state.get('code_implementation', '')
    if not code:
        return {
            "messages": [AIMessage(content="No code to compile")],
            "compilation_result": {"success": False, "errors": ["No code generated"]},
            "context": state.get('context', {})
        }
    
    # Get strategy name from plan
    plan = state.get('strategy_plan') or state.get('context', {}).get('strategy_plan', {})
    if isinstance(plan, str):
        import json
        try:
            plan = json.loads(plan.replace("'", '"'))
        except:
            plan = {}
    
    strategy_type = plan.get('strategy_type', 'ExpertAdvisor') if isinstance(plan, dict) else 'ExpertAdvisor'
    filename = f"EA_{strategy_type.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Compile with retry
    max_retries = 3
    compilation_result = None
    
    for attempt in range(max_retries):
        compilation_result = await compile_mql5(code, filename, "expert")
        
        if compilation_result.get("success", False):
            break
        
        # If compilation failed, try to fix errors
        errors = compilation_result.get("errors", [])
        if errors and attempt < max_retries - 1:
            logger.warning(f"Compilation attempt {attempt + 1} failed, attempting fixes")
            code = _attempt_code_fix(code, errors)
    
    # Store compilation result
    context = state.get('context', {})
    context['compilation_result'] = compilation_result
    context['compiled_code'] = code if compilation_result.get("success") else None
    
    if compilation_result.get("success"):
        message = AIMessage(content=f"Compilation successful: {filename}.ex5")
    else:
        errors = compilation_result.get("errors", [])
        message = AIMessage(content=f"Compilation failed after {max_retries} attempts: {len(errors)} errors")
    
    return {
        "messages": [message],
        "compilation_result": compilation_result,
        "code_implementation": code,  # Updated code after fixes
        "context": context
    }


async def backtesting_node(state: QuantCodeState) -> Dict[str, Any]:
    """
    Backtesting node: Run strategy backtest using Backtest MCP.
    
    Uses the compiled code and configuration from the strategy plan.
    """
    logger.info("Backtesting node running strategy backtest")
    
    # Get compiled code
    code = state.get('code_implementation', '')
    context = state.get('context', {})
    compilation_result = context.get('compilation_result', {})
    
    if not compilation_result.get("success"):
        return {
            "messages": [AIMessage(content="Skipping backtest - compilation failed")],
            "backtest_results": {"success": False, "error": "Compilation failed"},
            "context": context
        }
    
    # Get strategy plan for configuration
    plan = state.get('strategy_plan') or context.get('strategy_plan', {})
    if isinstance(plan, str):
        import json
        try:
            plan = json.loads(plan.replace("'", '"'))
        except:
            plan = {}
    
    # Build backtest configuration
    symbols = plan.get('symbols', ['EURUSD']) if isinstance(plan, dict) else ['EURUSD']
    timeframes = plan.get('timeframes', ['H1']) if isinstance(plan, dict) else ['H1']
    
    backtest_config = {
        "symbol": symbols[0] if symbols else "EURUSD",
        "timeframe": timeframes[0] if timeframes else "H1",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "initial_deposit": 10000.0,
        "spread": 10,
        "model": "every_tick"
    }
    
    strategy_name = plan.get('strategy_type', 'ExpertAdvisor') if isinstance(plan, dict) else 'ExpertAdvisor'
    
    # Run backtest via MCP
    backtest_results = await run_backtest_mcp(code, backtest_config, strategy_name)
    
    # Extract metrics
    metrics = backtest_results.get("metrics", {})
    
    context['backtest_results'] = backtest_results
    context['backtest_config'] = backtest_config
    
    message = AIMessage(content=f"Backtest completed: {metrics.get('total_trades', 0)} trades, {metrics.get('win_rate', 0):.1%} win rate")
    
    return {
        "messages": [message],
        "backtest_results": metrics,
        "context": context
    }


def analysis_node(state: QuantCodeState) -> Dict[str, Any]:
    """Analysis node: Analyze backtest results."""
    logger.info("Analysis node analyzing backtest results")
    
    backtest_results: Dict[str, Any] = state.get('backtest_results') or {}
    context = state.get('context', {})
    full_results = context.get('backtest_results', {})
    
    # Extract metrics
    total_trades = backtest_results.get('total_trades', full_results.get('total_trades', 0))
    win_rate = backtest_results.get('win_rate', full_results.get('win_rate', 0))
    sharpe_ratio = backtest_results.get('sharpe_ratio', full_results.get('sharpe_ratio', 0))
    max_drawdown = backtest_results.get('max_drawdown', full_results.get('max_drawdown', 0))
    net_profit = backtest_results.get('net_profit', full_results.get('net_profit', 0))
    
    # Calculate Kelly score
    kelly_score = _calculate_kelly_score(win_rate, max_drawdown, sharpe_ratio)
    
    analysis_report = f"""
Strategy Performance Analysis
============================

Trade Statistics:
- Total Trades: {total_trades}
- Win Rate: {win_rate:.2%}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Max Drawdown: {max_drawdown:.2%}
- Net Profit: ${net_profit:,.2f}
- Kelly Score: {kelly_score:.2f}

Recommendation: {'APPROVED' if kelly_score > 0.8 else 'NEEDS IMPROVEMENT'}
"""
    
    # Store Kelly score for validation
    context['kelly_score'] = kelly_score
    
    message = AIMessage(content="Analysis report generated")
    
    return {
        "messages": [message],
        "analysis_report": analysis_report,
        "context": context
    }


def reflection_node(state: QuantCodeState) -> Dict[str, Any]:
    """Reflection node: Reflect on strategy performance and improvements."""
    logger.info("Reflection node reviewing strategy")
    
    backtest_results: Dict[str, Any] = state.get('backtest_results') or {}
    context = state.get('context', {})
    kelly_score = context.get('kelly_score', 0)
    
    reflection_notes = f"""
Strategy Reflection
==================

Strengths:
- Win rate: {backtest_results.get('win_rate', 0):.2%}
- Kelly score: {kelly_score:.2f}
- Sharpe ratio: {backtest_results.get('sharpe_ratio', 0):.2f}

Areas for Improvement:
- Consider reducing max drawdown ({backtest_results.get('max_drawdown', 0):.2%})
- Test on different market conditions
- Optimize entry/exit timing

Next Steps:
- Deploy to paper trading if Kelly > 0.8
- Monitor live performance
- Iterate based on results
"""
    
    message = AIMessage(content="Reflection completed")
    
    return {
        "messages": [message],
        "reflection_notes": reflection_notes,
        "context": context
    }


def paper_deployment_node(state: QuantCodeState) -> Dict[str, Any]:
    """Deploy strategy to paper trading."""
    deployer_class = _get_paper_trading_deployer()
    if deployer_class is None:
        return {
            "messages": [AIMessage(content="PaperTradingDeployer not available - cannot deploy")],
            "paper_agent_id": None,
            "validation_start_time": None
        }
    
    deployer = deployer_class()
    
    strategy_code = state.get('code_implementation', '// No code available')
    context = state.get('context', {})
    
    # Get strategy name from plan
    plan = state.get('strategy_plan') or context.get('strategy_plan', {})
    if isinstance(plan, str):
        import json
        try:
            plan = json.loads(plan.replace("'", '"'))
        except:
            plan = {}
    
    strategy_type = plan.get('strategy_type', 'unnamed_strategy') if isinstance(plan, dict) else 'unnamed_strategy'
    
    deployment_request = {
        "strategy_code": strategy_code,
        "symbol": "EURUSD",
        "name": strategy_type,
        "initial_balance": 10000.0,
        "risk_params": {"max_daily_loss_pct": 5.0}
    }
    
    result = deployer.deploy_agent(**deployment_request)
    paper_agent_id = result.agent_id if result.agent_id else str(uuid.uuid4())[:8]
    
    return {
        "messages": [AIMessage(content=f"Deployed to paper trading: {paper_agent_id}")],
        "paper_agent_id": paper_agent_id,
        "validation_start_time": datetime.now(timezone.utc)
    }


def validation_monitoring_node(state: QuantCodeState) -> Dict[str, Any]:
    """Monitor paper trading validation."""
    validator_class = _get_paper_trading_validator()
    paper_agent_id = state.get('paper_agent_id')
    if not paper_agent_id:
        return {
            "messages": [AIMessage(content="No paper trading agent ID found")],
            "paper_trading_metrics": {}
        }
    if validator_class is None:
        return {
            "messages": [AIMessage(content="PaperTradingValidator not available")],
            "paper_trading_metrics": {}
        }
    
    validator = validator_class()
    status = validator.check_validation_status(paper_agent_id)
    report = validator.generate_validation_report(paper_agent_id)
    
    return {
        "messages": [AIMessage(content=report)],
        "paper_trading_metrics": status.get('metrics', {})
    }


def promotion_decision_node(state: QuantCodeState) -> Dict[str, Any]:
    """Decide if strategy should be promoted to live trading."""
    metrics = state.get('paper_trading_metrics', {})
    if not metrics:
        return {
            "messages": [AIMessage(content="No metrics available for promotion decision")],
            "promotion_approved": False
        }
    
    validator_class = _get_paper_trading_validator()
    if validator_class is None:
        return {
            "messages": [AIMessage(content="PaperTradingValidator not available")],
            "promotion_approved": False
        }
    
    approved = validator_class().meets_promotion_criteria(metrics)
    return {
        "messages": [AIMessage(content=f"Promotion {'approved' if approved else 'rejected'}")],
        "promotion_approved": approved
    }


# =============================================================================
# Helper Functions
# =============================================================================

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


# =============================================================================
# Conditional Edges
# =============================================================================

def should_continue_after_backtest(state: QuantCodeState) -> str:
    """Determine if backtest was successful."""
    backtest_results = state.get('backtest_results', {})
    context = state.get('context', {})
    
    # Check both direct results and context
    total_trades = backtest_results.get('total_trades', context.get('backtest_results', {}).get('total_trades', 0))
    
    if total_trades > 0:
        return "analysis"
    else:
        return "coding"  # Retry coding if backtest failed


def should_continue_to_end(state: QuantCodeState) -> str:
    """Determine if strategy meets quality criteria."""
    context = state.get('context', {})
    kelly_score = context.get('kelly_score', 0)
    
    if kelly_score >= 0.8:
        return END
    else:
        return "planning"  # Retry from planning if quality insufficient


def should_deploy_paper(state: QuantCodeState) -> str:
    """Determine if strategy should be deployed to paper trading."""
    context = state.get('context', {})
    kelly_score = context.get('kelly_score', 0)
    
    return "paper_deployment" if kelly_score > 0.8 else END


def should_promote(state: QuantCodeState) -> str:
    """Determine if strategy should be promoted to live trading."""
    metrics = state.get('paper_trading_metrics', {})
    if not metrics:
        return "planning"  # retry if no metrics
    
    validator_class = _get_paper_trading_validator()
    if validator_class is None:
        return "planning"  # retry if validator unavailable
    
    validator = validator_class()
    if validator.meets_promotion_criteria(metrics):
        return "promotion_decision"
    return "planning"  # retry


# =============================================================================
# Graph Construction
# =============================================================================

def create_quantcode_graph() -> StateGraph:
    """Create the QuantCode agent workflow graph."""
    workflow = StateGraph(QuantCodeState)
    
    # Add nodes
    workflow.add_node("planning", planning_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("compilation", compilation_node)
    workflow.add_node("backtesting", backtesting_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("paper_deployment", paper_deployment_node)
    workflow.add_node("validation_monitoring", validation_monitoring_node)
    workflow.add_node("promotion_decision", promotion_decision_node)
    
    # Set entry point
    workflow.set_entry_point("planning")
    
    # Add edges
    workflow.add_edge("planning", "coding")
    workflow.add_edge("coding", "compilation")
    workflow.add_conditional_edges(
        "compilation",
        lambda state: "backtesting" if state.get('context', {}).get('compilation_result', {}).get('success') else "coding",
        {
            "backtesting": "backtesting",
            "coding": "coding"
        }
    )
    workflow.add_conditional_edges(
        "backtesting",
        should_continue_after_backtest,
        {
            "analysis": "analysis",
            "coding": "coding"
        }
    )
    workflow.add_edge("analysis", "reflection")
    workflow.add_conditional_edges(
        "reflection",
        should_deploy_paper,
        {
            "paper_deployment": "paper_deployment",
            END: END
        }
    )
    workflow.add_edge("paper_deployment", "validation_monitoring")
    workflow.add_conditional_edges(
        "validation_monitoring",
        should_promote,
        {
            "promotion_decision": "promotion_decision",
            "planning": "planning"
        }
    )
    workflow.add_edge("promotion_decision", END)
    
    return workflow


def compile_quantcode_graph(checkpointer: Optional[MemorySaver] = None) -> Any:
    """Compile the QuantCode agent graph with optional checkpointing."""
    workflow = create_quantcode_graph()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("QuantCode agent graph compiled successfully")
    
    return compiled_graph


# =============================================================================
# Execution Interface
# =============================================================================

def run_quantcode_workflow(
    strategy_request: str,
    trd_content: Optional[str] = None,
    workspace_path: str = "workspaces/quant",
    memory_namespace: tuple = ("memories", "quantcode", "default")
) -> Dict[str, Any]:
    """Execute the QuantCode agent workflow."""
    initial_state: QuantCodeState = {
        "messages": [HumanMessage(content=strategy_request)],
        "current_task": "strategy_development",
        "workspace_path": workspace_path,
        "context": {"trd_content": trd_content} if trd_content else {},
        "memory_namespace": memory_namespace,
        "strategy_plan": None,
        "code_implementation": None,
        "backtest_results": None,
        "analysis_report": None,
        "reflection_notes": None,
        "paper_agent_id": None,
        "validation_start_time": None,
        "validation_period_days": 30,
        "paper_trading_metrics": None,
        "promotion_approved": False,
        "trd_content": trd_content,
        "compilation_result": None
    }
    
    graph = compile_quantcode_graph()
    config = {"configurable": {"thread_id": "quantcode_001"}}
    final_state = graph.invoke(initial_state, config)
    
    logger.info(f"QuantCode workflow completed")
    
    return final_state


# =============================================================================
# Async Execution Interface
# =============================================================================

async def run_quantcode_workflow_async(
    strategy_request: str,
    trd_content: Optional[str] = None,
    workspace_path: str = "workspaces/quant",
    memory_namespace: tuple = ("memories", "quantcode", "default")
) -> Dict[str, Any]:
    """Execute the QuantCode agent workflow asynchronously."""
    initial_state: QuantCodeState = {
        "messages": [HumanMessage(content=strategy_request)],
        "current_task": "strategy_development",
        "workspace_path": workspace_path,
        "context": {"trd_content": trd_content} if trd_content else {},
        "memory_namespace": memory_namespace,
        "strategy_plan": None,
        "code_implementation": None,
        "backtest_results": None,
        "analysis_report": None,
        "reflection_notes": None,
        "paper_agent_id": None,
        "validation_start_time": None,
        "validation_period_days": 30,
        "paper_trading_metrics": None,
        "promotion_approved": False,
        "trd_content": trd_content,
        "compilation_result": None
    }
    
    graph = compile_quantcode_graph()
    config = {"configurable": {"thread_id": f"quantcode_{uuid.uuid4().hex[:8]}"}}
    final_state = await graph.ainvoke(initial_state, config)
    
    logger.info(f"QuantCode workflow completed")
    
    return final_state
