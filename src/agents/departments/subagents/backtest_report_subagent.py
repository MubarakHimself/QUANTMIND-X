"""
BacktestReportSubAgent — Report generation only.

Fills a structured template from backtest result data, then adds
AI-generated improvement suggestions for strategies that need work.
This sub-agent does NOT write code. It produces markdown reports only.
"""

import logging

logger = logging.getLogger(__name__)


BACKTEST_REPORT_SYSTEM_PROMPT = """You are the Backtest Report Analyst at QUANTMINDX — a specialist sub-agent in the Research/Risk pipeline.

## IDENTITY & ROLE
You generate structured, quantitative backtest reports and improvement suggestions.
You do NOT write code. You produce markdown analysis reports only.
Your parent departments are Research (when generating post-backtest reports) and Risk (when evaluating strategy fitness).

## CORE RESPONSIBILITIES
- Fill structured report templates from backtest result data (sections 1-5: summary, IS vs OOS, Monte Carlo, Walk-Forward, Overfitting)
- Generate AI-powered improvement suggestions (section 6) when OOS degradation exceeds 15%
- Identify the root cause of performance degradation (regime sensitivity, overfitting, parameter fragility)
- Provide actionable, quantitative recommendations (not vague advice)

## ANALYSIS FRAMEWORK FOR IMPROVEMENT SUGGESTIONS
When OOS degradation > 15%, analyse these dimensions in order:
1. **Win Rate Degradation** — Entry condition robustness, signal quality decay
2. **Profit Factor Collapse** — Risk/reward ratio shift, stop placement issues
3. **Drawdown Expansion** — Regime mismatch, volatility sensitivity
4. **Sharpe Ratio Drop** — Return consistency, tail risk exposure
5. **WFA Efficiency** — Parameter stability across windows (< 60% = fragile)
6. **PBO Score** — Overfitting probability (> 0.5 = significant concern)

## OUTPUT RULES
- Suggestions must be specific: name parameters, directions (increase/decrease), and regimes
- Include quantitative targets where possible (e.g., "widen SL from 15 → 25 pips")
- Never suggest vague actions like "optimize parameters" — specify WHICH parameters
- If data is insufficient, state what additional data is needed
- Maximum 6 suggestions, minimum 4 when degradation is significant
- Each suggestion must include: what to change, why, and expected impact

## QUALITY CRITERIA
- SIT Gate PASS: Strategy performance is consistent IS→OOS (degradation ≤ 15%)
- SIT Gate FAIL: Significant degradation detected — must include root cause analysis
- Reports must be self-contained: a reader with no prior context should understand the verdict

## ESCALATION
- PBO > 0.7: Flag as HIGH overfitting risk, recommend complete parameter redesign
- WFA efficiency < 40%: Flag as UNSTABLE, recommend longer optimisation windows
- Monte Carlo probability of profit < 50%: Flag as UNPROFITABLE, recommend strategy retirement
"""


REPORT_TEMPLATE = """
# Backtest Report: {strategy_name}
**Strategy ID:** {strategy_id} | **Date:** {date} | **Bot Tag:** {bot_tag}

## 1. Summary
- Strategy type: {strategy_type} | Symbol: {symbol} | Timeframe: {timeframe}
- SIT Gate: {sit_result} | Recommendation: **{recommendation}**

## 2. In-Sample vs Out-of-Sample
| Metric | In-Sample (70%) | OOS (30%) | Degradation |
|--------|----------------|-----------|-------------|
| Win Rate | {is_win_rate}% | {oos_win_rate}% | {wr_degradation}% |
| Profit Factor | {is_pf} | {oos_pf} | {pf_degradation} |
| Max Drawdown | {is_dd}% | {oos_dd}% | {dd_delta}% |
| Sharpe Ratio | {is_sharpe} | {oos_sharpe} | {sharpe_delta} |

## 3. Monte Carlo (1000 runs, OOS trades)
- 95th percentile return: {mc_p95}%
- 5th percentile return (risk): {mc_p5}%
- Probability of profit: {mc_prob_profit}%

## 4. Walk-Forward
- WFA Efficiency: {wfa_efficiency} ({wfa_efficiency_pct}%)
- Windows passed: {wfa_windows_passed}/{wfa_windows_total}

## 5. Overfitting Risk
- PBO score: {pbo_score} (0=no overfit, 1=full overfit)
- Flag: {pbo_flag}

## 6. Improvement Suggestions ← Agent-generated
{agent_suggestions}
"""

class BacktestReportSubAgent:
    """Fills report template + generates improvement suggestions."""

    def __init__(self):
        pass

    def generate_report(
        self, strategy_id: str, trd_data: dict,
        backtest_result: dict, sit_result: dict
    ) -> str:
        """
        Step 1: Fill template sections 1-5 from structured data (no AI needed).
        Step 2: Agent generates section 6 (improvement suggestions).
        """
        # Fill sections 1-5 from data
        filled = self._fill_template(strategy_id, trd_data, backtest_result, sit_result)

        # Section 6: only call agent if strategy needs improvement (OOS degradation > 15%)
        is_summary = backtest_result.get("in_sample_summary", {})
        oos_summary = backtest_result.get("oos_summary", {})

        is_wr = is_summary.get("win_rate", 0)
        oos_wr = oos_summary.get("win_rate", 0)
        degradation = (is_wr - oos_wr) * 100 if is_wr else 0

        if (degradation > 15 or not sit_result.get("passed")) and sit_result.get("passed") is not None:
            suggestions = self._generate_suggestions(trd_data, backtest_result)
        else:
            suggestions = "Strategy performance is consistent IS→OOS. No major adjustments needed."

        return filled.replace("{agent_suggestions}", suggestions)

    def _fill_template(self, strategy_id: str, trd_data: dict, backtest_result: dict, sit_result: dict) -> str:
        """Fill sections 1-5 from structured data."""
        is_summary = backtest_result.get("in_sample_summary", {})
        oos_summary = backtest_result.get("oos_summary", {})
        mc = backtest_result.get("monte_carlo", {})
        wf = backtest_result.get("walk_forward", {})
        pbo = backtest_result.get("pbo", {})

        sit_passed = sit_result.get("passed", False)
        sit_result_str = "PASSED" if sit_passed else "FAILED"
        recommendation = "APPROVE for paper" if sit_passed else "REVISION needed before paper"

        is_wr = is_summary.get("win_rate", 0)
        oos_wr = oos_summary.get("win_rate", 0)
        wr_deg = round((is_wr - oos_wr) * 100, 1) if is_wr else 0

        is_pf = is_summary.get("profit_factor", 0)
        oos_pf = oos_summary.get("profit_factor", 0)
        pf_deg = round(is_pf - oos_pf, 2) if is_pf else 0

        is_dd = round(is_summary.get("max_drawdown", 0) * 100, 1)
        oos_dd = round(oos_summary.get("max_drawdown", 0) * 100, 1)
        dd_delta = round(oos_dd - is_dd, 1)

        is_sharpe = is_summary.get("sharpe", 0)
        oos_sharpe = oos_summary.get("sharpe", 0)
        sharpe_delta = round(is_sharpe - oos_sharpe, 2)

        wfa_eff = wf.get("efficiency", 0)
        wfa_eff_pct = round(wfa_eff * 100, 1) if wfa_eff else 0

        return REPORT_TEMPLATE.format(
            strategy_name=trd_data.get("strategy_name", strategy_id),
            strategy_id=strategy_id,
            date=self._today(),
            bot_tag=trd_data.get("bot_tag", "@primal"),
            strategy_type=trd_data.get("strategy_type", "unknown"),
            symbol=trd_data.get("symbol", "unknown"),
            timeframe=trd_data.get("timeframe", "unknown"),
            sit_result=sit_result_str,
            recommendation=recommendation,
            is_win_rate=round(is_wr * 100, 1) if is_wr else 0,
            oos_win_rate=round(oos_wr * 100, 1) if oos_wr else 0,
            wr_degradation=wr_deg,
            is_pf=round(is_pf, 2) if is_pf else 0,
            oos_pf=round(oos_pf, 2) if oos_pf else 0,
            pf_degradation=pf_deg,
            is_dd=is_dd,
            oos_dd=oos_dd,
            dd_delta=dd_delta,
            is_sharpe=round(is_sharpe, 2) if is_sharpe else 0,
            oos_sharpe=round(oos_sharpe, 2) if oos_sharpe else 0,
            sharpe_delta=sharpe_delta,
            mc_p95=round(mc.get("p95", 0), 1),
            mc_p5=round(mc.get("p5", 0), 1),
            mc_prob_profit=round(mc.get("prob_profit", 0) * 100, 1),
            wfa_efficiency=round(wfa_eff, 3) if wfa_eff else 0,
            wfa_efficiency_pct=wfa_eff_pct,
            wfa_windows_passed=wf.get("windows_passed", 0),
            wfa_windows_total=wf.get("windows_total", 0),
            pbo_score=round(pbo.get("score", 0), 2),
            pbo_flag=pbo.get("flag", "UNKNOWN"),
            agent_suggestions="{agent_suggestions}",  # placeholder
        )

    def _generate_suggestions(self, trd_data: dict, backtest_result: dict) -> str:
        """
        Generate improvement suggestions using Haiku LLM.

        Called when OOS degradation > 15% to provide AI-generated
        specific improvement recommendations.
        """
        try:
            from anthropic import Anthropic

            is_summary = backtest_result.get("in_sample_summary", {})
            oos_summary = backtest_result.get("oos_summary", {})
            mc = backtest_result.get("monte_carlo", {})
            wf = backtest_result.get("walk_forward", {})
            pbo = backtest_result.get("pbo", {})

            is_wr = round(is_summary.get("win_rate", 0) * 100, 1)
            oos_wr = round(oos_summary.get("win_rate", 0) * 100, 1)
            is_pf = round(is_summary.get("profit_factor", 0), 2)
            oos_pf = round(oos_summary.get("profit_factor", 0), 2)
            is_dd = round(is_summary.get("max_drawdown", 0) * 100, 1)
            oos_dd = round(oos_summary.get("max_drawdown", 0) * 100, 1)
            is_sharpe = round(is_summary.get("sharpe", 0), 2)
            oos_sharpe = round(oos_summary.get("sharpe", 0), 2)
            wfa_eff = round(wf.get("efficiency", 0) * 100, 1)
            pbo_score = round(pbo.get("score", 0), 2)

            user_prompt = f"""You are a quant trading strategy analyst. Based on the following backtest degradation data, provide 4-6 specific, actionable improvement suggestions for the trader.

Strategy: {trd_data.get("strategy_name", "unknown")}
Symbol: {trd_data.get("symbol", "unknown")} | Timeframe: {trd_data.get("timeframe", "unknown")}

IS vs OOS Degradation:
- Win Rate: {is_wr}% (IS) → {oos_wr}% (OOS), degradation = {is_wr - oos_wr}%
- Profit Factor: {is_pf} (IS) → {oos_pf} (OOS)
- Max Drawdown: {is_dd}% (IS) → {oos_dd}% (OOS)
- Sharpe: {is_sharpe} (IS) → {oos_sharpe} (OOS)
- WFA Efficiency: {wfa_eff}%
- PBO Score: {pbo_score} (0=no overfit, 1=full overfit)
- Monte Carlo Prob of Profit: {round(mc.get('prob_profit', 0) * 100, 1)}%

Provide suggestions as a markdown bullet list. Be specific about parameter names and regimes where relevant. Focus on:
1. Which metrics degraded most and what that implies
2. Specific parameter adjustments (with direction: increase/decrease/tune)
3. Regime or session conditions to filter
4. Overfitting mitigations given the PBO score"""

            from src.agents.departments.subagents.llm_utils import get_subagent_client
            client, model = get_subagent_client()
            response = client.messages.create(
                model=model,
                max_tokens=800,
                system=BACKTEST_REPORT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            suggestions = response.content[0].text.strip()
            logger.info("BacktestReportSubAgent: Haiku suggestions generated successfully")
            return suggestions

        except ImportError:
            logger.warning("BacktestReportSubAgent: Anthropic SDK not available, using fallback suggestions")
            return self._fallback_suggestions(backtest_result)
        except Exception as e:
            logger.error(f"BacktestReportSubAgent: Haiku call failed: {e}")
            return self._fallback_suggestions(backtest_result)

    def _fallback_suggestions(self, backtest_result: dict) -> str:
        """Fallback suggestions when Haiku is unavailable."""
        is_summary = backtest_result.get("in_sample_summary", {})
        oos_summary = backtest_result.get("oos_summary", {})
        is_wr = is_summary.get("win_rate", 0)
        oos_wr = oos_summary.get("win_rate", 0)
        degradation = (is_wr - oos_wr) * 100 if is_wr else 0

        suggestions = []
        if degradation > 15:
            suggestions.append("- Significant OOS win rate degradation detected: review entry condition robustness")
        suggestions.append("- Consider parameter regularisation to reduce overfitting")
        suggestions.append("- Review stop-loss placement for regime sensitivity")
        suggestions.append("- Add session/symbol filters based on spread behaviour")
        suggestions.append("- Run walk-forward analysis to validate parameter stability")
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(suggestions))

    def _today(self) -> str:
        from datetime import date
        return date.today().isoformat()