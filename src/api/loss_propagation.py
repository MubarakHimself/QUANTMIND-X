"""
Cross-Strategy Loss Propagation API

REST API for managing correlated risk when a strategy hits daily loss cap.
FR76: Cross-strategy loss propagation
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk", tags=["loss-propagation"])


# ============================================================================
# Request/Response Models
# ============================================================================


class LossCapBreachEvent(BaseModel):
    """Event when a strategy hits its daily loss cap."""
    strategy_id: str = Field(..., description="Strategy that breached loss cap")
    breach_time: str = Field(..., description="When the breach occurred")
    loss_amount: float = Field(..., description="Total loss incurred")
    daily_loss_cap: float = Field(..., description="The cap that was breached")
    correlation_threshold: float = Field(0.5, description="Correlation threshold for propagation")


class AffectedStrategy(BaseModel):
    """A strategy affected by loss propagation."""
    strategy_id: str
    original_kelly: float
    adjusted_kelly: float
    correlation: float
    reason: str


class LossPropagationEvent(BaseModel):
    """Record of loss propagation event (FR76 audit)."""
    event_id: str
    source_strategy: str
    breach_time: str
    affected_strategies: List[AffectedStrategy]
    total_affected: int
    audit_code: str = "FR76"


class LossPropagationResponse(BaseModel):
    """Response to loss propagation trigger."""
    success: bool
    source_strategy: str
    breach_time: str
    affected_strategies: List[AffectedStrategy]
    event_id: str
    message: str


# ============================================================================
# Loss Propagation Service
# ============================================================================


class LossPropagationService:
    """
    Service for cross-strategy loss propagation.

    When a strategy hits its daily loss cap:
    1. Get correlation matrix for all strategies
    2. Find strategies with |correlation| >= 0.5 to the breaching strategy
    3. Adjust Kelly fraction (multiply by 0.75)
    4. Log FR76 audit event
    5. Notify Copilot of affected strategies
    """

    def __init__(self):
        self._correlation_cache: Dict[str, np.ndarray] = {}
        self._kelly_adjustments: Dict[str, float] = {}

    async def get_correlation_matrix(self) -> Dict[str, np.ndarray]:
        """
        Get correlation matrix from correlation sensor.

        In production, this would query src/risk/physics/correlation_sensor.py
        """
        raise NotImplementedError(
            "Loss propagation requires real correlation data from correlation sensor. "
            "Not wired to production data."
        )

    async def get_correlated_strategies(
        self,
        source_strategy: str,
        correlation_matrix: Dict[str, np.ndarray],
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Find strategies with correlation above threshold to source.

        Returns list of {strategy_id, correlation} where |correlation| >= threshold.
        """
        if source_strategy not in correlation_matrix:
            logger.warning(f"Strategy {source_strategy} not in correlation matrix")
            return []

        source_corrs = correlation_matrix[source_strategy]
        strategies = list(correlation_matrix.keys())

        correlated = []
        for i, strategy in enumerate(strategies):
            if strategy == source_strategy:
                continue

            corr = source_corrs[i]
            if abs(corr) >= threshold:
                correlated.append({
                    "strategy_id": strategy,
                    "correlation": float(corr),
                })

        return correlated

    async def get_current_kelly_fraction(
        self,
        strategy_id: str,
    ) -> float:
        """
        Get current Kelly fraction for a strategy.

        In production: query risk_params table or Kelly engine
        """
        # Mock: return default Kelly fraction
        return 0.10  # 10% base Kelly

    async def adjust_kelly_fraction(
        self,
        strategy_id: str,
        original_kelly: float,
        adjustment_factor: float = 0.75,
    ) -> float:
        """
        Adjust Kelly fraction by multiplication factor.

        AC 3: Kelly × 0.75 for correlated strategies
        """
        adjusted = original_kelly * adjustment_factor

        # Store adjustment for audit
        self._kelly_adjustments[strategy_id] = adjusted

        logger.info(
            f"Adjusted Kelly for {strategy_id}: "
            f"{original_kelly:.4f} → {adjusted:.4f} (×{adjustment_factor})"
        )

        return adjusted

    async def log_audit_event(
        self,
        event: LossPropagationEvent,
    ) -> str:
        """
        Log FR76 audit event.

        In production: write to risk_audit_log table
        """
        # Mock: generate audit ID
        audit_id = f"FR76-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        logger.info(
            f"AUDIT {audit_id}: Loss propagation from {event.source_strategy} "
            f"affected {event.total_affected} strategies"
        )

        return audit_id

    async def trigger_loss_propagation(
        self,
        breach: LossCapBreachEvent,
    ) -> LossPropagationResponse:
        """
        Process loss cap breach and propagate risk adjustments.

        AC 3:
        - Given Strategy A hits its daily loss cap,
        - When the loss event fires,
        - Then all strategies with correlation >= 0.5 to Strategy A
          receive tightened risk params (Kelly fraction × 0.75),
        - And a loss propagation event records in the risk audit log (FR76).
        """
        logger.info(f"Loss cap breach for {breach.strategy_id}, triggering propagation")

        # Get correlation matrix
        correlation_matrix = await self.get_correlation_matrix()

        # Find correlated strategies
        correlated = await self.get_correlated_strategies(
            breach.strategy_id,
            correlation_matrix,
            threshold=breach.correlation_threshold,
        )

        # Adjust Kelly for each correlated strategy
        affected_strategies = []
        for item in correlated:
            strategy_id = item["strategy_id"]
            correlation = item["correlation"]

            original_kelly = await self.get_current_kelly_fraction(strategy_id)
            adjusted_kelly = await self.adjust_kelly_fraction(
                strategy_id,
                original_kelly,
                adjustment_factor=0.75,
            )

            affected_strategies.append(AffectedStrategy(
                strategy_id=strategy_id,
                original_kelly=original_kelly,
                adjusted_kelly=adjusted_kelly,
                correlation=correlation,
                reason=f"Correlation {correlation:.2f} >= {breach.correlation_threshold}",
            ))

        # Create audit event
        event = LossPropagationEvent(
            event_id="",  # Will be generated
            source_strategy=breach.strategy_id,
            breach_time=breach.breach_time,
            affected_strategies=affected_strategies,
            total_affected=len(affected_strategies),
        )

        # Log audit event
        event_id = await self.log_audit_event(event)

        return LossPropagationResponse(
            success=True,
            source_strategy=breach.strategy_id,
            breach_time=breach.breach_time,
            affected_strategies=affected_strategies,
            event_id=event_id,
            message=f"Loss propagation triggered. {len(affected_strategies)} strategies adjusted.",
        )

    async def get_propagation_history(
        self,
        limit: int = 50,
    ) -> List[LossPropagationEvent]:
        """
        Get recent loss propagation events for audit.

        In production: query risk_audit_log table for FR76 events
        """
        # Mock: return empty list for demo
        return []


# Singleton instance
_loss_propagation_service = LossPropagationService()


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/loss-propagation/trigger")
async def trigger_loss_propagation(
    breach: LossCapBreachEvent,
) -> LossPropagationResponse:
    """
    Trigger loss propagation for a strategy that hit daily loss cap.

    POST /api/risk/loss-propagation/trigger

    Body: {
        "strategy_id": "STRAT_A",
        "breach_time": "2026-03-20T10:30:00Z",
        "loss_amount": 500.0,
        "daily_loss_cap": 500.0,
        "correlation_threshold": 0.5
    }

    Returns: List of affected strategies with adjusted Kelly fractions.
    """
    logger.info(f"Loss propagation triggered for {breach.strategy_id}")
    try:
        return await _loss_propagation_service.trigger_loss_propagation(breach)
    except NotImplementedError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/loss-propagation/history")
async def get_propagation_history(
    limit: int = Query(50, ge=1, le=100, description="Max events to return"),
) -> List[LossPropagationEvent]:
    """
    Get recent loss propagation events.

    GET /api/risk/loss-propagation/history?limit=50
    """
    return await _loss_propagation_service.get_propagation_history(limit)


@router.get("/loss-propagation/status/{strategy_id}")
async def get_strategy_kelly_adjustment(
    strategy_id: str,
) -> Dict[str, Any]:
    """
    Get current Kelly adjustment status for a strategy.

    GET /api/risk/loss-propagation/status/{strategy_id}
    """
    current_kelly = await _loss_propagation_service.get_current_kelly_fraction(strategy_id)

    # Check if there's an active adjustment
    adjusted = _loss_propagation_service._kelly_adjustments.get(strategy_id)

    return {
        "strategy_id": strategy_id,
        "base_kelly": current_kelly,
        "current_kelly": adjusted or current_kelly,
        "is_adjusted": adjusted is not None and adjusted != current_kelly,
        "adjustment_factor": adjusted / current_kelly if adjusted else 1.0,
    }


@router.post("/loss-propagation/reset/{strategy_id}")
async def reset_kelly_adjustment(
    strategy_id: str,
) -> Dict[str, Any]:
    """
    Reset Kelly adjustment for a strategy (manual override).

    POST /api/risk/loss-propagation/reset/{strategy_id}
    """
    original_kelly = await _loss_propagation_service.get_current_kelly_fraction(strategy_id)

    # Reset in service
    if strategy_id in _loss_propagation_service._kelly_adjustments:
        del _loss_propagation_service._kelly_adjustments[strategy_id]

    return {
        "success": True,
        "strategy_id": strategy_id,
        "kelly_fraction": original_kelly,
        "message": "Kelly adjustment reset to base value",
    }