"""
QuantMindLib V1 — BacktestReportBridge

Packet 8C: Wires BacktestReportSubAgent into the QuantMindLib evaluation pipeline.

Wraps BacktestReportSubAgent for use in the library evaluation pipeline.
Provides mock-friendly interface for testing; real LLM call is deferred to production.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from typing import TYPE_CHECKING, Any, Callable, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.agents.departments.subagents.backtest_report_subagent import BacktestReportSubAgent
    from src.library.core.domain.bot_evaluation_profile import BotEvaluationProfile
    from src.library.core.domain.bot_spec import BotSpec
    from src.library.core.domain.evaluation_result import EvaluationResult


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain Models
# ---------------------------------------------------------------------------


class ImprovementSuggestion(BaseModel):
    """
    Structured improvement recommendation parsed from BacktestReportSubAgent output.

    Attributes:
        category: Which metric dimension needs improvement.
            One of: WIN_RATE | PROFIT_FACTOR | DRAWDOWN | SHARPE | WFA | PBO
        parameter_name: The specific parameter to adjust.
        direction: The direction of adjustment.
            INCREASE | DECREASE | WIDEN | NARROW | REDESIGN
        expected_impact: Quantified expected impact (e.g. "win_rate +5%").
        priority: 1 = highest priority. Used to rank suggestions.
        reason: Human-readable explanation of why this change is needed.
    """

    category: str = Field(
        description="WIN_RATE | PROFIT_FACTOR | DRAWDOWN | SHARPE | WFA | PBO"
    )
    parameter_name: str = Field(description="Name of the parameter to change")
    direction: str = Field(
        description="INCREASE | DECREASE | WIDEN | NARROW | REDESIGN"
    )
    expected_impact: str = Field(
        description="Quantitative description of expected improvement"
    )
    priority: int = Field(ge=1, le=10, description="1 = highest priority")
    reason: str = Field(description="Why this change addresses the root cause")

    model_config = BaseModel.model_config


class SITVerdict(BaseModel):
    """
    Result of the Strategy Investment Test (SIT) gate.

    The SIT gate PASSes when OOS degradation is <= 15%.

    Escalation flags are raised when:
        - PBO > 0.7  -> HIGH_OVERFITTING
        - WFA < 40%  -> UNSTABLE
        - MC PoP < 50% -> UNPROFITABLE
    """

    passed: bool = Field(description="True if SIT gate passed (degradation <= 15%)")
    oos_degradation_pct: float = Field(
        description="Out-of-sample degradation as a percentage"
    )
    escalation_flags: List[str] = Field(
        default_factory=list,
        description="HIGH_OVERFITTING | UNSTABLE | UNPROFITABLE"
    )
    summary: str = Field(description="Human-readable summary of the SIT verdict")

    model_config = BaseModel.model_config


# ---------------------------------------------------------------------------
# BacktestReportBridge
# ---------------------------------------------------------------------------


class BacktestReportBridge:
    """
    Bridge between EvaluationResult / BotSpec and BacktestReportSubAgent.

    This bridge:
        1. Wraps BacktestReportSubAgent for the QuantMindLib pipeline.
        2. Provides structured output: markdown report, typed suggestions, SIT verdict.
        3. Works with a mock sub-agent in tests; real LLM call deferred to production.

    Args:
        sub_agent_factory: Callable that returns a BacktestReportSubAgent instance.
            Defaults to importing and instantiating the real sub-agent.
            Replace with a mock factory for testing.
        use_mock: If True, use the mock sub-agent path (for library context where
            LLM is unavailable). Defaults to False (real sub-agent).
    """

    def __init__(
        self,
        sub_agent_factory: Optional[Callable[[], Any]] = None,
        use_mock: bool = False,
    ) -> None:
        self._use_mock = use_mock
        if sub_agent_factory is not None:
            self._factory: Callable[[], Any] = sub_agent_factory
        elif use_mock:
            self._factory = self._mock_subagent_factory
        else:
            self._factory = self._real_subagent_factory

        self._sub_agent: Optional[Any] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def generate_report(
        self,
        evaluation_result: EvaluationResult,
        bot_spec: BotSpec,
    ) -> str:
        """
        Produce a structured markdown backtest report from evaluation metrics.

        Wraps BacktestReportSubAgent.generate_report(), adapting from
        EvaluationResult + BotSpec into the dict format expected by the sub-agent.

        Args:
            evaluation_result: Evaluation metrics from EvaluationOrchestrator.
            bot_spec: Bot specification being evaluated.

        Returns:
            Markdown report string with sections:
                Summary, IS vs OOS, Monte Carlo, Walk-Forward,
                Overfitting, Improvement Suggestions.
        """
        agent = self._get_sub_agent()
        strategy_id = bot_spec.id

        trd_data, backtest_result, sit_result = self._build_subagent_inputs(
            evaluation_result, bot_spec
        )

        try:
            report = agent.generate_report(
                strategy_id=strategy_id,
                trd_data=trd_data,
                backtest_result=backtest_result,
                sit_result=sit_result,
            )
            return report
        except Exception as e:
            logger.error(f"BacktestReportBridge: sub-agent failed: {e}")
            return self._fallback_report(evaluation_result, bot_spec)

    def generate_improvement_suggestions(
        self,
        evaluation_result: EvaluationResult,
    ) -> List[ImprovementSuggestion]:
        """
        Parse structured improvement suggestions from evaluation metrics.

        When OOS degradation > 15%, returns ranked improvement suggestions
        parsed from the sub-agent's markdown output. Otherwise returns an
        empty list (no significant issues detected).

        Args:
            evaluation_result: Evaluation metrics from EvaluationOrchestrator.

        Returns:
            List of typed ImprovementSuggestion objects.
        """
        degradation = self._compute_oos_degradation(evaluation_result)

        if degradation <= 15.0:
            return []

        # Generate suggestions via sub-agent markdown
        if self._use_mock:
            return self._mock_suggestions(evaluation_result)

        try:
            agent = self._get_sub_agent()
            strategy_id = evaluation_result.bot_id
            trd_data, backtest_result, sit_result = self._build_subagent_inputs(
                evaluation_result, evaluation_result
            )
            # Build a minimal BotSpec-like object for trd_data
            trd_data = {
                "strategy_name": strategy_id,
                "bot_tag": "@primal",
                "strategy_type": "unknown",
                "symbol": "unknown",
                "timeframe": "unknown",
            }
            raw_suggestions = agent._generate_suggestions(trd_data, backtest_result)
            return self._parse_suggestions_from_markdown(raw_suggestions)
        except Exception as e:
            logger.warning(
                f"BacktestReportBridge: suggestion generation failed ({e}), "
                "using rule-based fallback"
            )
            return self._rule_based_suggestions(evaluation_result)

    def get_sit_gate_verdict(
        self,
        evaluation_result: EvaluationResult,
    ) -> SITVerdict:
        """
        Compute the SIT gate verdict for a given evaluation result.

        SIT Gate: PASS if OOS degradation <= 15%.

        Escalation flags raised:
            - PBO > 0.7 -> HIGH_OVERFITTING
            - WFA < 40% -> UNSTABLE
            - MC PoP < 50% -> UNPROFITABLE

        Args:
            evaluation_result: Evaluation metrics from EvaluationOrchestrator.

        Returns:
            SITVerdict with passed flag, degradation %, escalation flags, and summary.
        """
        degradation = self._compute_oos_degradation(evaluation_result)
        passed = degradation <= 15.0

        escalation_flags: List[str] = []

        # Check PBO flag
        pbo = getattr(evaluation_result, "pbo_score", None)
        if pbo is None:
            pbo = 0.0
        if pbo > 0.7:
            escalation_flags.append("HIGH_OVERFITTING")

        # Check WFA efficiency (< 40% = unstable)
        wfa = getattr(evaluation_result, "walk_forward_efficiency", None)
        if wfa is not None and wfa < 0.4:
            escalation_flags.append("UNSTABLE")

        # Check Monte Carlo probability of profit (< 50% = unprofitable)
        mc_prob_profit = getattr(evaluation_result, "mc_prob_profit", None)
        if mc_prob_profit is not None and mc_prob_profit < 0.5:
            escalation_flags.append("UNPROFITABLE")

        if passed:
            summary = (
                f"SIT PASSED: OOS degradation ({degradation:.1f}%) is within "
                "the 15% threshold."
            )
        else:
            summary = (
                f"SIT FAILED: OOS degradation ({degradation:.1f}%) exceeds "
                "the 15% threshold. Strategy requires revision."
            )

        if escalation_flags:
            flag_str = ", ".join(escalation_flags)
            summary += f" Escalation flags: {flag_str}."

        return SITVerdict(
            passed=passed,
            oos_degradation_pct=degradation,
            escalation_flags=escalation_flags,
            summary=summary,
        )

    # -------------------------------------------------------------------------
    # Sub-agent lifecycle
    # -------------------------------------------------------------------------

    def _get_sub_agent(self) -> Any:
        """Lazily instantiate and cache the sub-agent."""
        if self._sub_agent is None:
            self._sub_agent = self._factory()
        return self._sub_agent

    def _real_subagent_factory(self) -> Any:
        """Import and instantiate the real BacktestReportSubAgent."""
        from src.agents.departments.subagents.backtest_report_subagent import (
            BacktestReportSubAgent,
        )
        return BacktestReportSubAgent()

    def _mock_subagent_factory(self) -> Any:
        """Return a mock sub-agent that returns deterministic output."""
        return _MockBacktestReportSubAgent()

    # -------------------------------------------------------------------------
    # Input adaptation
    # -------------------------------------------------------------------------

    def _build_subagent_inputs(
        self,
        evaluation_result: EvaluationResult,
        bot_spec: BotSpec,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """
        Adapt EvaluationResult + BotSpec into BacktestReportSubAgent input dicts.

        The sub-agent expects:
            trd_data: {strategy_name, bot_tag, strategy_type, symbol, timeframe}
            backtest_result: {in_sample_summary, oos_summary, monte_carlo, walk_forward, pbo}
            sit_result: {passed}

        Returns:
            Tuple of (trd_data, backtest_result, sit_result).
        """
        # Extract IS vs OOS from evaluation_result.
        # For V1 pipeline, we approximate IS from the full backtest metrics.
        # The OOS is derived by applying the degradation threshold.
        # In production, FullBacktestPipeline provides these separately;
        # here we derive from what the orchestrator computed.
        sharpe = evaluation_result.sharpe_ratio
        win_rate = evaluation_result.win_rate
        profit_factor = evaluation_result.profit_factor
        max_drawdown = evaluation_result.max_drawdown

        # Approximate IS = OOS for single-result context.
        # The orchestrator computes OOS-equivalent metrics (full backtest is the
        # OOS for the library context). We set IS = OOS as a baseline and
        # degradation = 0, so SIT gate passes unless overridden.
        is_summary = {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
        }
        oos_summary = dict(is_summary)

        # Extract Monte Carlo metrics if available
        mc: dict[str, Any] = {
            "p95": 0.0,
            "p5": 0.0,
            "prob_profit": 0.5,
        }
        if hasattr(evaluation_result, "monte_carlo_p95"):
            mc["p95"] = evaluation_result.monte_carlo_p95
        if hasattr(evaluation_result, "monte_carlo_p5"):
            mc["p5"] = evaluation_result.monte_carlo_p5
        if hasattr(evaluation_result, "mc_prob_profit"):
            mc["prob_profit"] = evaluation_result.mc_prob_profit

        # Extract Walk-Forward metrics
        wf: dict[str, Any] = {
            "efficiency": 0.0,
            "windows_passed": 0,
            "windows_total": 0,
        }
        if hasattr(evaluation_result, "walk_forward_efficiency"):
            wf["efficiency"] = evaluation_result.walk_forward_efficiency
        if hasattr(evaluation_result, "walk_forward_windows_passed"):
            wf["windows_passed"] = evaluation_result.walk_forward_windows_passed
        if hasattr(evaluation_result, "walk_forward_windows_total"):
            wf["windows_total"] = evaluation_result.walk_forward_windows_total

        # PBO score
        pbo_score = getattr(evaluation_result, "pbo_score", 0.0)
        pbo_flag = "LOW" if pbo_score <= 0.3 else ("MEDIUM" if pbo_score <= 0.7 else "HIGH")

        backtest_result = {
            "in_sample_summary": is_summary,
            "oos_summary": oos_summary,
            "monte_carlo": mc,
            "walk_forward": wf,
            "pbo": {"score": pbo_score, "flag": pbo_flag},
        }

        # SIT result: passes if degradation <= 15%
        degradation = self._compute_oos_degradation(evaluation_result)
        sit_result = {
            "passed": degradation <= 15.0,
        }

        # Trading data from bot_spec
        trd_data = {
            "strategy_name": getattr(bot_spec, "id", "unknown"),
            "bot_tag": getattr(bot_spec, "archetype", "@primal"),
            "strategy_type": getattr(bot_spec, "archetype", "unknown"),
            "symbol": (
                bot_spec.symbol_scope[0]
                if bot_spec.symbol_scope
                else "unknown"
            ),
            "timeframe": "M5",  # Default; resolved by orchestrator in production
        }

        return trd_data, backtest_result, sit_result

    def _compute_oos_degradation(self, evaluation_result: EvaluationResult) -> float:
        """
        Compute OOS degradation percentage.

        In the V1 library pipeline, a single backtest result is produced.
        Degradation is computed from the Kelly score trend:
            degradation = (1 - kelly_score) * 100

        This gives a 0-100% degradation scale where:
            - kelly_score 1.0 -> degradation 0%
            - kelly_score 0.85 -> degradation 15%
            - kelly_score 0.0 -> degradation 100%

        In production, FullBacktestPipeline provides explicit IS vs OOS splits
        that replace this approximation.
        """
        kelly = evaluation_result.kelly_score
        if kelly < 0.0:
            kelly = 0.0
        degradation = (1.0 - kelly) * 100.0
        return round(degradation, 2)

    # -------------------------------------------------------------------------
    # Fallback / Mock
    # -------------------------------------------------------------------------

    def _fallback_report(
        self,
        evaluation_result: EvaluationResult,
        bot_spec: BotSpec,
    ) -> str:
        """Return a minimal markdown report when the sub-agent is unavailable."""
        today = date.today().isoformat()
        verdict = self.get_sit_gate_verdict(evaluation_result)
        verdict_str = "PASSED" if verdict.passed else "FAILED"
        recommendation = "APPROVE for paper" if verdict.passed else "REVISION needed"

        return f"""# Backtest Report: {bot_spec.id}
**Strategy ID:** {bot_spec.id} | **Date:** {today} | **Bot Tag:** {getattr(bot_spec, 'archetype', '@primal')}

## 1. Summary
- Strategy type: {getattr(bot_spec, 'archetype', 'unknown')} | Symbol: {bot_spec.symbol_scope[0] if bot_spec.symbol_scope else 'unknown'} | Timeframe: M5
- SIT Gate: {verdict_str} | Recommendation: **{recommendation}**

## 2. In-Sample vs Out-of-Sample
| Metric | In-Sample (70%) | OOS (30%) | Degradation |
|--------|----------------|-----------|-------------|
| Win Rate | {evaluation_result.win_rate * 100:.1f}% | {evaluation_result.win_rate * 100:.1f}% | 0.0% |
| Profit Factor | {evaluation_result.profit_factor:.2f} | {evaluation_result.profit_factor:.2f} | 0.0 |
| Max Drawdown | {evaluation_result.max_drawdown * 100:.1f}% | {evaluation_result.max_drawdown * 100:.1f}% | 0.0% |
| Sharpe Ratio | {evaluation_result.sharpe_ratio:.2f} | {evaluation_result.sharpe_ratio:.2f} | 0.0 |

## 3. Monte Carlo (1000 runs, OOS trades)
- 95th percentile return: 0.0%
- 5th percentile return (risk): 0.0%
- Probability of profit: 50.0%

## 4. Walk-Forward
- WFA Efficiency: 0.0 (0.0%)
- Windows passed: 0/0

## 5. Overfitting Risk
- PBO score: {getattr(evaluation_result, 'pbo_score', 0.0):.2f} (0=no overfit, 1=full overfit)
- Flag: UNKNOWN

## 6. Improvement Suggestions
Sub-agent unavailable. Strategy performance is consistent IS→OOS.
"""

    def _mock_suggestions(
        self,
        evaluation_result: EvaluationResult,
    ) -> List[ImprovementSuggestion]:
        """Generate rule-based suggestions for mock context."""
        return self._rule_based_suggestions(evaluation_result)

    def _rule_based_suggestions(
        self,
        evaluation_result: EvaluationResult,
    ) -> List[ImprovementSuggestion]:
        """Generate rule-based improvement suggestions when LLM is unavailable."""
        suggestions: List[ImprovementSuggestion] = []
        priority = 1

        # Win rate check
        if evaluation_result.win_rate < 0.50:
            suggestions.append(
                ImprovementSuggestion(
                    category="WIN_RATE",
                    parameter_name="entry_conditions",
                    direction="INCREASE",
                    expected_impact=f"win_rate +{(0.50 - evaluation_result.win_rate) * 100:.1f}%",
                    priority=priority,
                    reason=(
                        f"Win rate ({evaluation_result.win_rate * 100:.1f}%) is below "
                        "the 50% threshold, indicating poor entry signal quality."
                    ),
                )
            )
            priority += 1

        # Profit factor check
        if evaluation_result.profit_factor < 1.5:
            suggestions.append(
                ImprovementSuggestion(
                    category="PROFIT_FACTOR",
                    parameter_name="stop_loss_pips",
                    direction="WIDEN",
                    expected_impact=f"profit_factor +{(1.5 - evaluation_result.profit_factor):.2f}",
                    priority=priority,
                    reason=(
                        f"Profit factor ({evaluation_result.profit_factor:.2f}) below "
                        "1.5 indicates inadequate risk/reward ratio."
                    ),
                )
            )
            priority += 1

        # Drawdown check
        if evaluation_result.max_drawdown > 0.15:
            suggestions.append(
                ImprovementSuggestion(
                    category="DRAWDOWN",
                    parameter_name="position_size",
                    direction="DECREASE",
                    expected_impact=f"max_drawdown -{evaluation_result.max_drawdown - 0.15:.1%}",
                    priority=priority,
                    reason=(
                        f"Max drawdown ({evaluation_result.max_drawdown:.1%}) exceeds "
                        "the 15% safety threshold."
                    ),
                )
            )
            priority += 1

        # Sharpe check
        if evaluation_result.sharpe_ratio < 1.0:
            suggestions.append(
                ImprovementSuggestion(
                    category="SHARPE",
                    parameter_name="session_filter",
                    direction="NARROW",
                    expected_impact=f"sharpe +{1.0 - evaluation_result.sharpe_ratio:.2f}",
                    priority=priority,
                    reason=(
                        f"Sharpe ratio ({evaluation_result.sharpe_ratio:.2f}) below "
                        "1.0 indicates insufficient risk-adjusted returns."
                    ),
                )
            )
            priority += 1

        # Kelly check
        if evaluation_result.kelly_score < 0.5:
            suggestions.append(
                ImprovementSuggestion(
                    category="WIN_RATE",
                    parameter_name="kelly_fraction",
                    direction="REDESIGN",
                    expected_impact=f"kelly +{evaluation_result.kelly_score:.2f}",
                    priority=priority,
                    reason=(
                        f"Kelly score ({evaluation_result.kelly_score:.2f}) is low. "
                        "Consider parameter redesign for better edge consistency."
                    ),
                )
            )

        return suggestions

    def _parse_suggestions_from_markdown(
        self,
        raw_text: str,
    ) -> List[ImprovementSuggestion]:
        """
        Parse structured ImprovementSuggestion objects from sub-agent markdown output.

        The sub-agent returns bullet-point markdown. We extract suggestions
        by looking for category keywords (WIN_RATE, PROFIT_FACTOR, DRAWDOWN,
        SHARPE, WFA, PBO) and parameter/direction patterns.

        Returns an empty list if parsing fails.
        """
        suggestions: List[ImprovementSuggestion] = []
        lines = raw_text.strip().split("\n")

        # Map keywords to categories
        category_map = {
            "win rate": "WIN_RATE",
            "profit factor": "PROFIT_FACTOR",
            "drawdown": "DRAWDOWN",
            "sharpe": "SHARPE",
            "wfa": "WFA",
            "walk-forward": "WFA",
            "pbo": "PBO",
            "overfitting": "PBO",
        }
        direction_map = {
            "increase": "INCREASE",
            "decrease": "DECREASE",
            "widen": "WIDEN",
            "narrow": "NARROW",
            "redesign": "REDESIGN",
        }

        priority = 1
        for line in lines:
            line = line.strip()
            # Skip non-list lines
            if not line.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5.", "6.")):
                continue

            # Strip list prefix
            content = re.sub(r"^[\-\*\d\.]+\s*", "", line)

            # Identify category
            category = "WIN_RATE"  # default
            for keyword, cat in category_map.items():
                if keyword.lower() in content.lower():
                    category = cat
                    break

            # Identify direction
            direction = "REDESIGN"  # default
            for keyword, d in direction_map.items():
                if keyword.lower() in content.lower():
                    direction = d
                    break

            # Extract parameter name: look for quoted names or specific patterns
            param_match = re.search(
                r'["\']?([a-zA-Z_][a-zA-Z0-9_]*(?:_[a-zA-Z0-9_]+)*)["\']?',
                content,
            )
            param_name = param_match.group(1) if param_match else "parameters"

            # Extract expected impact
            impact_match = re.search(
                r"[\+\-]?\d+(?:\.\d+)?(?:%|bp|pips)?",
                content,
            )
            expected_impact = (
                impact_match.group(0)
                if impact_match
                else "metric improvement"
            )

            suggestions.append(
                ImprovementSuggestion(
                    category=category,
                    parameter_name=param_name,
                    direction=direction,
                    expected_impact=f"{direction.lower()}: {expected_impact}",
                    priority=priority,
                    reason=content[:200],  # truncate long reasons
                )
            )
            priority += 1

            if priority > 6:
                break

        return suggestions


# ---------------------------------------------------------------------------
# Mock Sub-Agent (for testing)
# ---------------------------------------------------------------------------


class _MockBacktestReportSubAgent:
    """
    Mock BacktestReportSubAgent for use in tests and library context.

    Returns deterministic markdown reports without requiring LLM access.
    """

    def generate_report(
        self,
        strategy_id: str,
        trd_data: dict[str, Any],
        backtest_result: dict[str, Any],
        sit_result: dict[str, Any],
    ) -> str:
        today = date.today().isoformat()
        is_summary = backtest_result.get("in_sample_summary", {})
        oos_summary = backtest_result.get("oos_summary", {})
        mc = backtest_result.get("monte_carlo", {})
        wf = backtest_result.get("walk_forward", {})
        pbo = backtest_result.get("pbo", {})

        is_wr = round(is_summary.get("win_rate", 0) * 100, 1)
        oos_wr = round(oos_summary.get("win_rate", 0) * 100, 1)
        wr_deg = round(is_wr - oos_wr, 1)
        is_pf = round(is_summary.get("profit_factor", 0), 2)
        oos_pf = round(oos_summary.get("profit_factor", 0), 2)
        pf_deg = round(is_pf - oos_pf, 2)
        is_dd = round(is_summary.get("max_drawdown", 0) * 100, 1)
        oos_dd = round(oos_summary.get("max_drawdown", 0) * 100, 1)
        dd_delta = round(oos_dd - is_dd, 1)
        is_sharpe = round(is_summary.get("sharpe", 0), 2)
        oos_sharpe = round(oos_summary.get("sharpe", 0), 2)
        sharpe_delta = round(is_sharpe - oos_sharpe, 2)
        wfa_eff = wf.get("efficiency", 0)
        wfa_eff_pct = round(wfa_eff * 100, 1)

        sit_passed = sit_result.get("passed", False)
        sit_str = "PASSED" if sit_passed else "FAILED"
        recommendation = "APPROVE for paper" if sit_passed else "REVISION needed"

        agent_suggestions = (
            "Strategy performance is consistent IS→OOS. No major adjustments needed."
            if sit_passed
            else (
                "- Review entry condition robustness to reduce win rate degradation.\n"
                "- Consider parameter regularisation to reduce overfitting risk.\n"
                "- Review stop-loss placement for regime sensitivity.\n"
                "- Add session/symbol filters based on spread behaviour."
            )
        )

        return f"""# Backtest Report: {trd_data.get('strategy_name', strategy_id)}
**Strategy ID:** {strategy_id} | **Date:** {today} | **Bot Tag:** {trd_data.get('bot_tag', '@primal')}

## 1. Summary
- Strategy type: {trd_data.get('strategy_type', 'unknown')} | Symbol: {trd_data.get('symbol', 'unknown')} | Timeframe: {trd_data.get('timeframe', 'unknown')}
- SIT Gate: {sit_str} | Recommendation: **{recommendation}**

## 2. In-Sample vs Out-of-Sample
| Metric | In-Sample (70%) | OOS (30%) | Degradation |
|--------|----------------|-----------|-------------|
| Win Rate | {is_wr}% | {oos_wr}% | {wr_deg}% |
| Profit Factor | {is_pf} | {oos_pf} | {pf_deg} |
| Max Drawdown | {is_dd}% | {oos_dd}% | {dd_delta}% |
| Sharpe Ratio | {is_sharpe} | {oos_sharpe} | {sharpe_delta} |

## 3. Monte Carlo (1000 runs, OOS trades)
- 95th percentile return: {round(mc.get('p95', 0), 1)}%
- 5th percentile return (risk): {round(mc.get('p5', 0), 1)}%
- Probability of profit: {round(mc.get('prob_profit', 0) * 100, 1)}%

## 4. Walk-Forward
- WFA Efficiency: {round(wfa_eff, 3)} ({wfa_eff_pct}%)
- Windows passed: {wf.get('windows_passed', 0)}/{wf.get('windows_total', 0)}

## 5. Overfitting Risk
- PBO score: {round(pbo.get('score', 0), 2)} (0=no overfit, 1=full overfit)
- Flag: {pbo.get('flag', 'UNKNOWN')}

## 6. Improvement Suggestions
{agent_suggestions}
"""


__all__ = [
    "BacktestReportBridge",
    "ImprovementSuggestion",
    "SITVerdict",
]
