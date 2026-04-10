"""
QuantMindLib V1 — BotSpec Multi-Profile Contract
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict

from src.library.core.types.enums import ActivationState, BotHealth


class SessionScore(BaseModel):
    """Nested model for per-session performance metrics."""
    session_id: str
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    expectancy: float


class WalkForwardMetrics(BaseModel):
    """Nested model for walk-forward analysis aggregate results."""
    n_splits: int
    avg_sharpe: float
    avg_return: float
    stability: float


class MonteCarloMetrics(BaseModel):
    """Nested model for Monte Carlo confidence interval results."""
    n_simulations: int
    percentile_5_return: float
    percentile_95_return: float
    max_drawdown_95: float
    sharpe_confidence_width: float


class BacktestMetrics(BaseModel):
    """Nested model for backtest performance metrics."""
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_bars_held: float
    total_trades: int


class BotEvaluationProfile(BaseModel):
    """Mutable profile for evaluation results across all evaluation modes."""
    bot_id: str
    backtest: BacktestMetrics
    monte_carlo: MonteCarloMetrics
    walk_forward: WalkForwardMetrics
    pbo_score: float
    robustness_score: float
    spread_sensitivity: float
    session_scores: Dict[str, SessionScore]
    report: Optional[str] = None  # Markdown backtest report from BacktestReportBridge


class BotMutationProfile(BaseModel):
    """Mutable profile for mutation lineage and safety boundaries."""
    bot_id: str
    allowed_mutation_areas: List[str]
    locked_components: List[str]
    lineage: List[str]
    parent_variant_id: Optional[str]
    generation: int
    unsafe_areas: List[str]


class BotRuntimeProfile(BaseModel):
    """Mutable profile for runtime state — activation, health, DPR rank, eligibility."""
    bot_id: str
    activation_state: ActivationState
    deployment_target: str
    health: BotHealth
    session_eligibility: Dict[str, bool]
    dpr_ranking: int
    dpr_score: float
    journal_id: Optional[str] = None
    report_ids: List[str]


class BotSpec(BaseModel):
    """
    Frozen canonical bot specification — the primary spec object consumed by all systems.
    """
    model_config = ConfigDict(frozen=True)

    id: str
    archetype: str
    symbol_scope: List[str]
    sessions: List[str]
    features: List[str]
    confirmations: List[str]
    execution_profile: str
    capability_spec: Optional[Any] = None
    runtime: Optional[BotRuntimeProfile] = None
    evaluation: Optional[BotEvaluationProfile] = None
    mutation: Optional[BotMutationProfile] = None


__all__ = [
    "SessionScore",
    "WalkForwardMetrics",
    "MonteCarloMetrics",
    "BacktestMetrics",
    "BotEvaluationProfile",
    "BotMutationProfile",
    "BotRuntimeProfile",
    "BotSpec",
]
