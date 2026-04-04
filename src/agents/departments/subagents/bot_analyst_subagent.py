"""
BotAnalystSubAgent — Analyses underperforming bots and produces a Bot Analysis Brief.

Trading Department sub-agent — does NOT write code, writes briefs only.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


BOT_ANALYST_SYSTEM_PROMPT = """You are the Bot Analyst at QUANTMINDX — a specialist sub-agent for diagnosing underperforming trading bots.

## IDENTITY & ROLE
You analyse quarantined or underperforming bots and produce structured Bot Analysis Briefs.
You do NOT write code or modify EAs. Your output is a diagnostic brief that Development uses to improve the bot.
You are part of the Trading Department's quality assurance pipeline (WF2 — Iteration Workflow).

## CORE RESPONSIBILITIES
- Retrieve and synthesise backtest reports, EA records, and recent trade data for a given bot
- Diagnose the root cause of underperformance (regime mismatch, parameter drift, spread sensitivity, etc.)
- Identify failure patterns across sessions, symbols, and market regimes
- Produce a structured Bot Analysis Brief with specific, actionable recommended changes
- Dispatch the brief to Development for implementation

## DIAGNOSTIC FRAMEWORK (analyse in this order)
1. **Win Rate Analysis** — Compare actual vs backtest win rate; degradation > 10% = signal decay
2. **Session Analysis** — Which trading sessions (London/NY/Tokyo/Sydney) show losses?
3. **Regime Correlation** — Is the bot losing in specific regimes (trend/range/breakout/chaos)?
4. **Spread Sensitivity** — Are losses concentrated during high-spread periods (news, session overlap)?
5. **Parameter Drift** — Have market conditions shifted since the bot's optimisation window?
6. **Drawdown Pattern** — Sudden vs gradual; single event vs sustained decline
7. **Correlation with Other Bots** — Is this bot correlated with other underperformers?

## BRIEF OUTPUT FORMAT (always follow this structure)
1. **Diagnosis** — What's wrong, backed by specific numbers (win rate, PnL, drawdown)
2. **Failure Pattern** — Which sessions/symbols/regimes failed, with trade counts and loss amounts
3. **Recommended Changes** — 4-6 specific, actionable parameter or logic changes with direction
4. **Confidence** — HIGH (>100 trades analysed) / MEDIUM (30-100 trades) / LOW (<30 trades)
5. **Estimated Improvement** — Quantitative estimate of expected improvement if changes are applied

## OUTPUT RULES
- Be specific: name parameters, sessions, regimes, and spread thresholds
- Include numbers: "Win rate dropped from 62% to 41% in NY session" not "win rate declined"
- Recommended changes must specify direction: increase/decrease/remove/add/tune
- If trade data is insufficient (<30 trades), state this clearly and lower confidence to LOW
- Never recommend "monitor" as a change — that's not actionable for Development

## QUARANTINE TRIGGERS YOU ANALYSE
- 3 consecutive losing sessions → SESSION_DECAY
- Drawdown > 15% of strategy allocation → DRAWDOWN_BREACH
- Win rate < 40% over last 50 trades → WIN_RATE_COLLAPSE
- Manual review request from Floor Manager → MANUAL_REVIEW
- Risk department flag → RISK_FLAG

## ESCALATION
- If diagnosis indicates a fundamental strategy flaw (not parameter tuning): escalate to Research
- If bot should be retired (no improvement path): recommend retirement to Floor Manager
- If correlated with 2+ other underperformers: flag systemic issue to Risk
"""


BRIEF_TEMPLATE = """
# Bot Analysis Brief: {strategy_id}
**Analyst:** BotAnalystSubAgent | **Date:** {date} | **Trigger:** {trigger}

## Diagnosis
{diagnosis}

## Failure Pattern
- Failed sessions: {failed_sessions}
- Regime correlation: {regime_correlation}
- Spread sensitivity: {spread_sensitivity}

## Recommended Changes
{recommended_changes}

## Confidence: {confidence}
## Estimated Improvement: {estimated_improvement}
"""


class BotAnalystSubAgent:
    """
    Analyses underperforming bots and produces a Bot Analysis Brief.
    """

    async def analyse(self, bot_metadata: Dict[str, Any]) -> str:
        """
        1. Retrieve from knowledge_hub: backtest reports, research hypothesis, EA records
        2. Retrieve from trade_journal: last 30 sessions of trade data for this magic number
        3. Agent analysis: what's wrong, what regime was it struggling in, what should change
        4. Return: structured Bot Analysis Brief

        Args:
            bot_metadata: dict with bot_id, strategy_id, magic_number, symbol, trigger (quarantine reason)

        Returns:
            Bot Analysis Brief markdown string
        """
        from src.agents.departments.floor_manager import get_floor_manager
        from src.agents.tools.knowledge.knowledge_hub import search_knowledge_hub

        strategy_id = bot_metadata.get("strategy_id", bot_metadata.get("bot_id"))
        magic_number = bot_metadata.get("magic_number")
        trigger = bot_metadata.get("trigger", "manual_review")

        logger.info(f"BotAnalystSubAgent: analysing {strategy_id} (trigger: {trigger})")

        # 1. Retrieve backtest reports from knowledge_hub
        backtest_reports = []
        try:
            results = await search_knowledge_hub(
                query=strategy_id,
                namespace="backtest_reports",
                limit=5
            )
            if results:
                backtest_reports = [r.get("content", "") for r in results if r.get("content")]
        except Exception as e:
            logger.warning(f"Could not fetch backtest reports: {e}")

        # 2. Retrieve EA records / research history
        ea_records = []
        try:
            results = await search_knowledge_hub(
                query=strategy_id,
                namespace="ea_records",
                limit=3
            )
            if results:
                ea_records = [r.get("content", "") for r in results if r.get("content")]
        except Exception as e:
            logger.warning(f"Could not fetch EA records: {e}")

        # 3. Try to get recent trade summary (trade_journal)
        trade_summary = {}
        try:
            from src.database.models.trade_record import TradeRecord
            from sqlalchemy import select
            from src.config import get_database_url
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

            db_url = get_database_url()
            async_db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")

            engine = create_async_engine(async_db_url, echo=False)
            async with AsyncSession(engine) as session:
                stmt = select(TradeRecord).where(
                    TradeRecord.magic_number == magic_number
                ).order_by(TradeRecord.closed_at.desc()).limit(200)

                result = await session.execute(stmt)
                records = result.scalars().all()

                if records:
                    total_pnl = sum(r.pnl or 0 for r in records)
                    win_rate = sum(1 for r in records if (r.pnl or 0) > 0) / len(records) if records else 0
                    trade_summary = {
                        "total_trades": len(records),
                        "win_rate": round(win_rate, 3),
                        "total_pnl": round(total_pnl, 2),
                        "last_session": records[0].closed_at.isoformat() if records else None
                    }
        except Exception as e:
            logger.warning(f"Could not fetch trade summary: {e}")

        # 4. Generate brief from retrieved data
        brief = self._generate_brief(
            strategy_id=strategy_id,
            trigger=trigger,
            backtest_reports=backtest_reports,
            ea_records=ea_records,
            trade_summary=trade_summary,
            bot_metadata=bot_metadata,
        )

        # 5. Dispatch brief to Development Department
        try:
            fm = get_floor_manager()
            await fm.dispatch(
                to_dept="development",
                task_type="IMPROVE_EA_FROM_BRIEF",
                payload={
                    "strategy_id": strategy_id,
                    "bot_id": bot_metadata.get("bot_id"),
                    "brief": brief,
                    "original_trd_id": bot_metadata.get("trd_id"),
                    "trigger": trigger,
                },
                priority="medium"
            )
            logger.info(f"BotAnalystSubAgent: brief dispatched to Development for {strategy_id}")
        except Exception as e:
            logger.warning(f"Failed to dispatch brief to Development: {e}")

        return brief

    def _generate_brief(
        self,
        strategy_id: str,
        trigger: str,
        backtest_reports: list,
        ea_records: list,
        trade_summary: dict,
        bot_metadata: dict,
    ) -> str:
        """
        Generate the structured Bot Analysis Brief using Haiku LLM.

        Synthesises backtest reports, EA records, and recent trade data into
        a structured brief with diagnosis, failure patterns, and recommended changes.
        Falls back to heuristic defaults if Haiku is unavailable.
        """
        try:
            from anthropic import Anthropic

            # Build context string from retrieved data
            backtest_context = "\n\n".join(
                f"--- Backtest Report {i+1} ---\n{r[:2000]}"
                for i, r in enumerate(backtest_reports[:3])
            ) or "No backtest reports available."

            ea_context = "\n\n".join(
                f"--- EA Record {i+1} ---\n{r[:1500]}"
                for i, r in enumerate(ea_records[:2])
            ) or "No EA records available."

            ts = trade_summary or {}
            trade_context = (
                f"Total trades: {ts.get('total_trades', 'N/A')}\n"
                f"Win rate: {ts.get('win_rate', 'N/A')}\n"
                f"Total PnL: {ts.get('total_pnl', 'N/A')}\n"
                f"Last session: {ts.get('last_session', 'N/A')}"
            )

            user_prompt = f"""You are a quant trading bot analyst. Produce a structured Bot Analysis Brief for the strategy below.

## Strategy Metadata
- Strategy/Bot ID: {strategy_id}
- Symbol: {bot_metadata.get("symbol", "N/A")}
- Magic Number: {bot_metadata.get("magic_number", "N/A")}
- Trigger (quarantine reason): {trigger}

## Backtest Reports
{backtest_context}

## EA Records / Research History
{ea_context}

## Recent Trade Summary (last 200 trades)
{trade_context}

## Your Task
Produce a Bot Analysis Brief with these exact sections:

1. **Diagnosis** — What's wrong with this bot? Be specific (win rate, drawdown, regime mismatch, etc.)
2. **Failure Pattern** — Which sessions/symbols/regimes did it fail in? Include concrete numbers.
3. **Recommended Changes** — 4-6 specific, actionable parameter or logic changes (with direction: increase/decrease/tune)
4. **Confidence** — HIGH / MEDIUM / LOW based on data quality
5. **Estimated Improvement** — Quantitative estimate of expected improvement if changes are applied

Format your response using markdown headers (## Diagnosis, ## Failure Pattern, etc.).
Be specific about parameter names and regimes. Do not be vague."""

            from src.agents.departments.subagents.llm_utils import get_subagent_client
            client, model = get_subagent_client()
            response = client.messages.create(
                model=model,
                max_tokens=1000,
                system=BOT_ANALYST_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            brief_content = response.content[0].text.strip()
            logger.info(f"BotAnalystSubAgent: Haiku brief generated for {strategy_id}")

            # Wrap with metadata header
            return BRIEF_TEMPLATE.format(
                strategy_id=strategy_id,
                date=self._today(),
                trigger=trigger,
                diagnosis="[See Haiku output below]",
                failed_sessions="",
                regime_correlation="",
                spread_sensitivity="",
                recommended_changes="",
                confidence="",
                estimated_improvement="",
            ).split("## Diagnosis")[0] + "\n" + brief_content

        except ImportError:
            logger.warning("BotAnalystSubAgent: Anthropic SDK not available, using fallback brief")
            return self._fallback_brief(strategy_id, trigger, trade_summary, bot_metadata)
        except Exception as e:
            logger.error(f"BotAnalystSubAgent: Haiku call failed: {e}")
            return self._fallback_brief(strategy_id, trigger, trade_summary, bot_metadata)

    def _fallback_brief(
        self,
        strategy_id: str,
        trigger: str,
        trade_summary: dict,
        bot_metadata: dict,
    ) -> str:
        """Fallback brief generation when Haiku is unavailable."""
        ts = trade_summary or {}
        wr = ts.get("win_rate", 0)
        total_trades = ts.get("total_trades", 0)

        if total_trades > 0 and wr < 0.40:
            diagnosis = f"Win rate of {wr:.1%} is below 40% — strategy losing money consistently."
            recommended = (
                "1. Reduce position size immediately\n"
                "2. Review stop-loss placement — may be too tight\n"
                "3. Disable entries during high-spread sessions\n"
                "4. Consider switching to longer timeframe"
            )
            confidence = "MEDIUM"
            estimated = "+5-10% win rate with wider SL"
        elif total_trades > 0 and wr < 0.50:
            diagnosis = f"Win rate of {wr:.1%} is marginally profitable but below 50%."
            recommended = (
                "1. Tighten entry filters (require stronger signal confirmation)\n"
                "2. Review news/calendar event exclusions\n"
                "3. Reduce TP target to improve win rate"
            )
            confidence = "MEDIUM"
            estimated = "+3-5% win rate with better entry quality"
        else:
            diagnosis = "Win rate acceptable; may have outlier losing sessions."
            recommended = (
                "1. Add session-specific filters\n"
                "2. Review correlation with other active bots\n"
                "3. Ensure position sizing is consistent"
            )
            confidence = "HIGH" if total_trades > 0 else "LOW"
            estimated = "+1-3% consistency improvement"

        return BRIEF_TEMPLATE.format(
            strategy_id=strategy_id,
            date=self._today(),
            trigger=trigger,
            diagnosis=diagnosis,
            failed_sessions=f"{total_trades} trades analysed; majority loss sessions." if total_trades > 0 else "No trade data available.",
            regime_correlation="Unknown",
            spread_sensitivity="Unknown",
            recommended_changes=recommended,
            confidence=confidence,
            estimated_improvement=estimated,
        )

    def _today(self) -> str:
        from datetime import date
        return date.today().isoformat()