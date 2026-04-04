"""
Paper Trading Promotion Endpoints

Handles agent promotion and trade recording endpoints.
"""

import logging
import os
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from .runtime import (
    PaperTradingUnavailableDeployer,
    ensure_paper_trading_runtime,
)

logger = logging.getLogger(__name__)


def get_deployer():
    """Dependency injection for PaperTradingDeployer."""
    try:
        from mcp_mt5.paper_trading.deployer import PaperTradingDeployer
    except ImportError as e:
        logger.warning("PaperTradingDeployer unavailable: %s", e)
        return PaperTradingUnavailableDeployer(
            "Paper trading runtime unavailable on this host. "
            "Install and configure the MT5 paper trading runtime on the target VPS."
        )

    deployer = PaperTradingDeployer()
    setattr(deployer, "available", True)
    return deployer


def get_validator():
    """Dependency injection for PaperTradingValidator."""
    try:
        from src.agents.paper_trading_validator import PaperTradingValidator
        return PaperTradingValidator()
    except ImportError as e:
        logger.warning(f"PaperTradingValidator not available: {e}")
        return None


def get_promotion_manager():
    """Dependency injection for PromotionManager."""
    from src.router.promotion_manager import PromotionManager
    return PromotionManager()


async def broadcast_promotion(agent_id: str, bot_id: str, target_account: str, performance_summary: dict):
    """Broadcast promotion event."""
    from src.api.websocket_endpoints import broadcast_paper_trading_promotion
    await broadcast_paper_trading_promotion(
        agent_id=agent_id,
        bot_id=bot_id,
        target_account=target_account,
        performance_summary=performance_summary
    )


async def broadcast_update(agent_id: str, status: str, **kwargs):
    """Broadcast update to UI."""
    from src.api.websocket_endpoints import broadcast_paper_trading_update
    await broadcast_paper_trading_update(
        agent_id=agent_id,
        status=status,
        **kwargs
    )


def setup_promotion_routes(router: APIRouter):
    """Setup promotion-related routes."""

    @router.post("/agents/{agent_id}/promote")
    async def promote_agent_endpoint(
        agent_id: str,
        request,  # PromotionRequest
        validator=Depends(get_validator),
        deployer=Depends(get_deployer),
        promotion_manager=Depends(get_promotion_manager),
    ):
        """
        Promote a validated paper trading agent to live trading.
        """
        from .models import PromotionResult

        ensure_paper_trading_runtime(deployer)

        # Check if agent exists
        agent = deployer.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        if validator is None:
            raise HTTPException(
                status_code=503,
                detail="Promotion service unavailable - validator not initialized"
            )

        try:
            # Step 1: Validate promotion criteria
            validation_result = validator.check_validation_status(agent_id)

            validation_status = validation_result.get("status", "pending")
            days_validated = validation_result.get("days_validated", 0)
            meets_criteria = validation_result.get("meets_criteria", False)
            metrics = validation_result.get("metrics", {})

            # Check validation requirements
            if validation_status != "VALIDATED":
                if days_validated < 30:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Validation period incomplete: {days_validated}/30 days"
                    )
                if not meets_criteria:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "Performance criteria not met",
                            "metrics": metrics,
                            "thresholds": {
                                "sharpe_ratio": 1.5,
                                "win_rate": 0.55
                            }
                        }
                    )

            # Step 2-6: Promotion logic (simplified for modular structure)
            # Full implementation would be similar to original paper_trading_endpoints.py

            return PromotionResult(
                promoted=False,
                agent_id=agent_id,
                error="Promotion logic moved to promotion module"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Promotion failed for {agent_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Promotion failed: {str(e)}"
            )

    @router.post("/bots/{bot_id}/trades")
    async def record_trade_for_bot(
        bot_id: str,
        request,  # TradeRecordRequest
        promotion_manager=Depends(get_promotion_manager),
    ) -> dict:
        """
        Record a trade for a bot to update performance tracking.
        """
        try:
            trade = {
                "symbol": request.symbol,
                "direction": request.direction,
                "entry_price": request.entry_price,
                "exit_price": request.exit_price,
                "pnl": request.pnl,
                "timestamp": request.timestamp or datetime.now(timezone.utc).isoformat(),
            }

            # Record through PromotionManager
            promotion_manager.record_trade_for_bot(bot_id, trade)

            logger.info(f"Recorded trade for bot {bot_id}: pnl={request.pnl}")

            return {
                "success": True,
                "bot_id": bot_id,
                "message": "Trade recorded successfully"
            }

        except Exception as e:
            logger.error(f"Failed to record trade for bot {bot_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to record trade: {str(e)}"
            )

    @router.post("/promotion/daily-check")
    async def run_daily_promotion_check(
        promotion_manager=Depends(get_promotion_manager),
    ) -> dict:
        """
        Run daily promotion eligibility check for all bots.
        """
        try:
            results = promotion_manager.run_daily_promotion_check()

            eligible_count = sum(1 for r in results if r.eligible)

            logger.info(f"Daily promotion check complete: {eligible_count}/{len(results)} bots eligible")

            return {
                "success": True,
                "total_bots": len(results),
                "eligible_for_promotion": eligible_count,
                "results": [
                    {
                        "bot_id": r.bot_id,
                        "current_mode": r.current_mode,
                        "eligible": r.eligible,
                        "next_mode": r.next_mode,
                        "missing_criteria": r.missing_criteria,
                    }
                    for r in results
                ]
            }

        except Exception as e:
            logger.error(f"Daily promotion check failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Daily promotion check failed: {str(e)}"
            )

    @router.get("/bots/{bot_id}/promotion-status")
    async def get_bot_promotion_status(
        bot_id: str,
        promotion_manager=Depends(get_promotion_manager),
    ) -> dict:
        """
        Get promotion status for a specific bot.
        """
        try:
            status = promotion_manager.get_promotion_status(bot_id)
            return status
        except Exception as e:
            logger.error(f"Failed to get promotion status for {bot_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get promotion status: {str(e)}"
            )

    return router
