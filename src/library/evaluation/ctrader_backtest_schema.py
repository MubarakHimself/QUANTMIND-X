"""
QuantMindLib V1 — cTrader Backtest Schema

CTRADER-008 contract: Pydantic schema for cTrader Open API backtest results.
This model is the canonical contract that the cTrader backtest adapter
(CTRADER-008) must implement. It maps field-for-field to EvaluationResult
using cTrader naming conventions, with bidirectional conversion via
to_evaluation_result() / from_evaluation_result().
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, Field

from src.library.core.domain.evaluation_result import EvaluationResult


def _fa(camel: str, snake: str) -> AliasChoices:
    """Create a validation alias that accepts either camelCase or snake_case."""
    return AliasChoices(camel, snake)


class CTraderBacktestResult(BaseModel):
    """
    Pydantic schema representing a backtest result produced by the
    cTrader Open API backtest engine (CTRADER-008).

    Field names use snake_case internally but accept camelCase input
    (cTrader API convention) via AliasChoices validation aliases.
    Values are semantically equivalent to EvaluationResult fields.

    CTRADER-008 implementers must produce data conforming to this schema.
    """

    model_config = {"populate_by_name": True}

    # ── Core metrics (equivalent to EvaluationResult) ────────────────────────

    bot_id: str = Field(
        validation_alias=_fa("botId", "bot_id"),
        description="Bot identifier, mirrors EvaluationResult.bot_id"
    )
    mode: str = Field(
        default="BACKTEST",
        validation_alias=_fa("mode", "mode"),
        description="Mode string, mirrors EvaluationResult.mode"
    )

    # Sharpe ratio
    sharpe_ratio: float = Field(
        validation_alias=_fa("sharpeRatio", "sharpe_ratio"),
        description="Sharpe ratio, mirrors EvaluationResult.sharpe_ratio"
    )

    # Maximum drawdown
    max_drawdown: float = Field(
        validation_alias=_fa("maxDrawdown", "max_drawdown"),
        description="Max drawdown as fraction [0-1], mirrors EvaluationResult.max_drawdown"
    )

    # Win rate
    win_rate: float = Field(
        ge=0.0, le=1.0,
        validation_alias=_fa("winRate", "win_rate"),
        description="Win rate as fraction [0-1], mirrors EvaluationResult.win_rate"
    )

    # Profit factor
    profit_factor: float = Field(
        validation_alias=_fa("profitFactor", "profit_factor"),
        description="Profit factor, mirrors EvaluationResult.profit_factor"
    )

    # Expectancy
    expectancy: float = Field(
        default=0.0,
        validation_alias=_fa("expectancy", "expectancy"),
        description="Average profit per trade, mirrors EvaluationResult.expectancy"
    )

    # Total trades
    total_trades: int = Field(
        ge=0,
        validation_alias=_fa("totalTrades", "total_trades"),
        description="Number of completed trades, mirrors EvaluationResult.total_trades"
    )

    # Return percentage
    return_pct: float = Field(
        validation_alias=_fa("returnPercent", "return_pct"),
        description="Total return as percentage, mirrors EvaluationResult.return_pct"
    )

    # Kelly criterion score
    kelly_score: float = Field(
        validation_alias=_fa("kellyScore", "kelly_score"),
        description="Kelly criterion score [0-1], mirrors EvaluationResult.kelly_score"
    )

    # Gate pass/fail
    passes_gate: bool = Field(
        validation_alias=_fa("gateStatus", "passes_gate"),
        description="True if strategy passes evaluation gate, mirrors EvaluationResult.passes_gate"
    )

    # ── Regime analytics (optional, mirrors EvaluationResult) ─────────────────

    regime_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        validation_alias=_fa("regimeDistribution", "regime_distribution"),
        description="Regime-to-trade-count mapping, mirrors EvaluationResult.regime_distribution"
    )

    # Filtered trades count
    filtered_trades: Optional[int] = Field(
        default=None,
        ge=0,
        validation_alias=_fa("filteredTrades", "filtered_trades"),
        description="Number of trades filtered by regime detection, mirrors EvaluationResult.filtered_trades"
    )

    # ── cTrader-specific fields (used internally, not in EvaluationResult) ────

    # Trade history for derived metric computation
    trade_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        validation_alias=_fa("tradeHistory", "trade_history"),
        description="List of trade records from cTrader, each with 'profit' key"
    )

    # In-sample / out-of-sample split
    is_regime_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        validation_alias=_fa("isRegimeDistribution", "is_regime_distribution"),
        description="Regime distribution for in-sample period"
    )
    oos_regime_distribution: Optional[Dict[str, int]] = Field(
        default=None,
        validation_alias=_fa("oosRegimeDistribution", "oos_regime_distribution"),
        description="Regime distribution for out-of-sample period"
    )

    # ── Field aliases for cTrader API compatibility ─────────────────────────

    @classmethod
    def from_ctrader_api(
        cls,
        data: Dict[str, Any],
        bot_id: str,
    ) -> CTraderBacktestResult:
        """
        Parse a raw cTrader Open API response dict into CTraderBacktestResult.

        This handles the cTrader API's camelCase naming and normalises values
        to the schema's conventions (win_rate as fraction, max_drawdown as
        fraction, etc.).

        Args:
            data: Raw cTrader API response dictionary.
            bot_id: Bot identifier to attach to the result.

        Returns:
            CTraderBacktestResult instance.
        """
        # Normalise win_rate: cTrader returns percentage (0-100) or fraction (0-1)
        raw_win_rate = data.get("winRate", 0.0)
        win_rate = raw_win_rate / 100.0 if raw_win_rate > 1.0 else raw_win_rate

        # Normalise max_drawdown: cTrader may return as percentage or fraction
        raw_dd = data.get("maxDrawdown", 0.0)
        max_drawdown = raw_dd / 100.0 if raw_dd > 1.0 else raw_dd

        return cls(
            bot_id=bot_id,
            mode=data.get("mode", "BACKTEST"),
            sharpe_ratio=data.get("sharpeRatio", 0.0),
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=data.get("profitFactor", 1.0),
            expectancy=data.get("expectancy", 0.0),
            total_trades=data.get("totalTrades", 0),
            return_pct=data.get("returnPercent", 0.0),
            kelly_score=data.get("kellyScore", 0.0),
            passes_gate=data.get("gateStatus", False),
            regime_distribution=data.get("regimeDistribution"),
            filtered_trades=data.get("filteredTrades"),
            trade_history=data.get("tradeHistory"),
            is_regime_distribution=data.get("isRegimeDistribution"),
            oos_regime_distribution=data.get("oosRegimeDistribution"),
        )

    # ── Conversion to canonical EvaluationResult ────────────────────────────

    def to_evaluation_result(self) -> EvaluationResult:
        """
        Convert cTrader backtest result to the canonical EvaluationResult.

        Computes regime_distribution by merging IS and OOS regime distributions
        if both are present. Falls back to the top-level regime_distribution
        field if IS/OOS fields are absent.

        Returns:
            EvaluationResult with all fields populated from this instance.
        """
        merged_regime: Optional[Dict[str, int]] = self.regime_distribution

        # Merge IS and OOS regime distributions when available
        if self.is_regime_distribution or self.oos_regime_distribution:
            merged_regime = {}
            for dist in [self.is_regime_distribution, self.oos_regime_distribution]:
                if dist:
                    for regime, count in dist.items():
                        merged_regime[regime] = merged_regime.get(regime, 0) + count

        return EvaluationResult(
            bot_id=self.bot_id,
            mode=self.mode,
            sharpe_ratio=self.sharpe_ratio,
            max_drawdown=self.max_drawdown,
            win_rate=self.win_rate,
            profit_factor=self.profit_factor,
            expectancy=self.expectancy,
            total_trades=self.total_trades,
            return_pct=self.return_pct,
            kelly_score=self.kelly_score,
            passes_gate=self.passes_gate,
            regime_distribution=merged_regime,
            filtered_trades=self.filtered_trades,
        )

    # ── Conversion from canonical EvaluationResult ──────────────────────────

    @classmethod
    def from_evaluation_result(
        cls,
        result: EvaluationResult,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        is_regime_distribution: Optional[Dict[str, int]] = None,
        oos_regime_distribution: Optional[Dict[str, int]] = None,
    ) -> CTraderBacktestResult:
        """
        Create a CTraderBacktestResult from an EvaluationResult.

        This is the inverse of to_evaluation_result(). Useful for round-trip
        testing or when a canonical EvaluationResult needs to be serialised
        in cTrader-compatible form.

        Args:
            result: Canonical EvaluationResult to convert.
            trade_history: Optional trade history from the source engine.
            is_regime_distribution: Optional IS-period regime distribution.
            oos_regime_distribution: Optional OOS-period regime distribution.

        Returns:
            CTraderBacktestResult equivalent to the input EvaluationResult.
        """
        return cls(
            bot_id=result.bot_id,
            mode=result.mode,
            sharpe_ratio=result.sharpe_ratio,
            max_drawdown=result.max_drawdown,
            win_rate=result.win_rate,
            profit_factor=result.profit_factor,
            expectancy=result.expectancy,
            total_trades=result.total_trades,
            return_pct=result.return_pct,
            kelly_score=result.kelly_score,
            passes_gate=result.passes_gate,
            regime_distribution=result.regime_distribution,
            filtered_trades=result.filtered_trades,
            trade_history=trade_history,
            is_regime_distribution=is_regime_distribution,
            oos_regime_distribution=oos_regime_distribution,
        )


__all__ = ["CTraderBacktestResult"]
