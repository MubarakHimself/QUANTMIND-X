"""
Paper Trading Agent Management Endpoints

Handles agent listing, status, stop, logs, and performance endpoints.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from .runtime import (
    PaperTradingUnavailableDeployer,
    ensure_paper_trading_runtime,
    is_paper_trading_runtime_available,
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


def get_enhanced_deployer():
    """Dependency injection for EnhancedPaperTradingDeployer."""
    from src.agents.enhanced_paper_trading_deployer import EnhancedPaperTradingDeployer
    return EnhancedPaperTradingDeployer()


def get_validator():
    """Dependency injection for PaperTradingValidator."""
    try:
        from src.agents.paper_trading_validator import PaperTradingValidator
        return PaperTradingValidator()
    except ImportError as e:
        logger.warning(f"PaperTradingValidator not available: {e}")
        return None


async def broadcast_update(agent_id: str, status: str, **kwargs):
    """Broadcast update to UI."""
    from src.api.websocket_endpoints import broadcast_paper_trading_update
    await broadcast_paper_trading_update(
        agent_id=agent_id,
        status=status,
        **kwargs
    )


def setup_agent_routes(router: APIRouter):
    """Setup agent-related routes."""

    @router.get("/agents")
    async def list_agents(
        deployer=Depends(get_deployer),
    ) -> List:
        """
        List all paper trading agents.

        Returns status information for all deployed agent containers.
        """
        agents = deployer.list_agents()

        # Broadcast current status of all agents to UI
        for agent in agents:
            await broadcast_update(
                agent_id=agent.agent_id,
                status=agent.status,
                symbol=agent.symbol,
                timeframe=agent.timeframe,
                uptime_seconds=agent.uptime_seconds
            )

        return agents

    @router.get("/agents/{agent_id}")
    async def get_agent_endpoint(
        agent_id: str,
        deployer=Depends(get_deployer),
    ):
        """
        Get status of a specific paper trading agent.
        """
        ensure_paper_trading_runtime(deployer)
        agent = deployer.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")
        return agent

    @router.post("/agents/{agent_id}/stop")
    async def stop_agent_endpoint(
        agent_id: str,
        deployer=Depends(get_deployer),
    ) -> dict:
        """
        Stop a paper trading agent.
        """
        ensure_paper_trading_runtime(deployer)
        success = deployer.stop_agent(agent_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to stop agent")
        logger.info(f"Stopped agent {agent_id}")

        # Broadcast stop update to UI
        await broadcast_update(
            agent_id=agent_id,
            status="stopped"
        )

        return {"success": True, "message": f"Agent {agent_id} stopped successfully"}

    @router.get("/agents/{agent_id}/logs")
    async def get_agent_logs_endpoint(
        agent_id: str,
        tail_lines: int = Query(default=100, ge=1, le=10000),
        deployer=Depends(get_deployer),
    ):
        """
        Get logs from a paper trading agent.
        """
        ensure_paper_trading_runtime(deployer)
        try:
            logs = deployer.get_agent_logs(agent_id, tail_lines)
            return logs
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Failed to get logs for {agent_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")

    @router.get("/agents/{agent_id}/performance")
    async def get_agent_performance_endpoint(
        agent_id: str,
        validator=Depends(get_validator),
        deployer=Depends(get_deployer),
    ):
        """
        Get performance metrics for a paper trading agent.
        """
        from .models import AgentPerformanceResponse

        ensure_paper_trading_runtime(deployer)

        # Check if agent exists
        agent = deployer.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        if validator is None:
            # Return default response if validator unavailable
            return AgentPerformanceResponse(
                agent_id=agent_id,
                validation_status="unavailable",
                error="Performance validation not available"
            )

        try:
            # Get validation status with metrics
            validation_result = validator.check_validation_status(agent_id)
            metrics = validation_result.get("metrics", {})

            return AgentPerformanceResponse(
                agent_id=agent_id,
                total_trades=metrics.get("total_trades", 0),
                winning_trades=metrics.get("winning_trades", 0),
                losing_trades=metrics.get("losing_trades", 0),
                win_rate=metrics.get("win_rate", 0.0),
                total_pnl=metrics.get("total_pnl", 0.0),
                average_pnl=metrics.get("average_pnl", 0.0),
                max_drawdown=metrics.get("max_drawdown", 0.0),
                profit_factor=metrics.get("profit_factor", 0.0),
                sharpe_ratio=metrics.get("sharpe", None),
                validation_status=validation_result.get("status", "pending").lower(),
                days_validated=validation_result.get("days_validated", 0),
                meets_criteria=validation_result.get("meets_criteria", False)
            )
        except Exception as e:
            logger.error(f"Failed to get performance for {agent_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve performance metrics: {str(e)}"
            )

    @router.get("/agents/{agent_id}/validation-status")
    async def get_validation_status_endpoint(
        agent_id: str,
        validator=Depends(get_validator),
        deployer=Depends(get_deployer),
    ) -> dict:
        """
        Get validation status for a paper trading agent.
        """
        ensure_paper_trading_runtime(deployer)

        # Check if agent exists
        agent = deployer.get_agent(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="Agent not found")

        if validator is None:
            return {
                "agent_id": agent_id,
                "validation_status": "unavailable",
                "error": "Validation service not available"
            }

        try:
            validation_result = validator.check_validation_status(agent_id)
            metrics = validation_result.get("metrics", {})

            return {
                "agent_id": agent_id,
                "validation_status": validation_result.get("status", "pending").lower(),
                "days_validated": validation_result.get("days_validated", 0),
                "required_days": 30,
                "meets_criteria": validation_result.get("meets_criteria", False),
                "metrics": {
                    "sharpe_ratio": metrics.get("sharpe", 0.0),
                    "win_rate": metrics.get("win_rate", 0.0),
                    "total_trades": metrics.get("total_trades", 0),
                    "profit_factor": metrics.get("profit_factor", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0)
                },
                "thresholds": {
                    "min_sharpe_ratio": 1.5,
                    "min_win_rate": 0.55,
                    "min_validation_days": 30
                },
                "promotion_eligible": (
                    validation_result.get("status") == "VALIDATED" and
                    validation_result.get("meets_criteria", False) and
                    validation_result.get("days_validated", 0) >= 30
                )
            }
        except Exception as e:
            logger.error(f"Failed to get validation status for {agent_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve validation status: {str(e)}"
            )

    return router
