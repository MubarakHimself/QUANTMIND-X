"""
Agent Node Functions for QuantCode workflow.

Contains all LangGraph node functions for the QuantCode agent.
"""

import logging
import json
import asyncio
from typing import Dict, Any
from datetime import datetime, timezone
import uuid

from langchain_core.messages import HumanMessage, AIMessage

from src.agents.state import QuantCodeState
from src.agents.quantcode.helpers import (
    _extract_strategy_type,
    _extract_indicators,
    _extract_entry_rules,
    _extract_exit_rules,
    _extract_risk_management,
    _extract_timeframes,
    _extract_symbols,
    _generate_mql5_code,
    _attempt_code_fix,
    _calculate_kelly_score
)
from src.agents.quantcode.mcp_tools import (
    compile_mql5,
    run_backtest_mcp,
    _get_paper_trading_validator,
    _get_paper_trading_deployer
)

logger = logging.getLogger(__name__)


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
