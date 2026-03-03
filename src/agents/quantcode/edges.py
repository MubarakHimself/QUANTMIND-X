"""
Conditional Edge Functions for QuantCode workflow.

Contains all conditional edge functions that determine workflow flow.
"""

import logging
from typing import Dict, Any

from src.agents.state import QuantCodeState
from src.agents.quantcode.mcp_tools import _get_paper_trading_validator

logger = logging.getLogger(__name__)


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

    from langgraph.graph import END

    if kelly_score >= 0.8:
        return END
    else:
        return "planning"  # Retry from planning if quality insufficient


def should_deploy_paper(state: QuantCodeState) -> str:
    """Determine if strategy should be deployed to paper trading."""
    from langgraph.graph import END

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
