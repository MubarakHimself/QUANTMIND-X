"""
Paper Trading Deployment Endpoints

Handles agent deployment endpoints.
"""

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from .runtime import (
    PaperTradingUnavailableDeployer,
    ensure_paper_trading_runtime,
)

logger = logging.getLogger(__name__)


# Dependency Injection for Deployer
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


def get_demo_account_manager():
    """Dependency injection for DemoAccountManager."""
    from src.agents.demo_account_manager import DemoAccountManager
    return DemoAccountManager()


async def broadcast_deployment_update(agent_id: str, status: str, **kwargs):
    """Broadcast deployment update to UI."""
    from src.api.websocket_endpoints import broadcast_paper_trading_update
    await broadcast_paper_trading_update(
        agent_id=agent_id,
        status=status,
        **kwargs
    )


def setup_deploy_routes(router: APIRouter):
    """Setup deployment-related routes."""

    @router.post("/deploy")
    async def deploy_agent_endpoint(
        request,  # AgentDeploymentRequest
        deployer=Depends(get_deployer),
    ):
        """
        Deploy a new paper trading agent.

        Creates a Docker container with the specified strategy code and MT5 credentials.
        """
        ensure_paper_trading_runtime(deployer)
        try:
            # Note: symbol and timeframe can be added to config if needed
            config = getattr(request, 'config', {})
            symbol = getattr(request, 'symbol', None)
            timeframe = getattr(request, 'timeframe', None)
            if symbol:
                config['symbol'] = symbol
            if timeframe:
                config['timeframe'] = timeframe

            result = deployer.deploy_agent(
                strategy_name=request.strategy_name,
                strategy_code=request.strategy_code,
                config=config,
                mt5_credentials=request.mt5_credentials,
                magic_number=request.magic_number,
            )
            logger.info(f"Deployed agent {result.agent_id}")

            # Broadcast deployment update to UI
            await broadcast_deployment_update(
                agent_id=result.agent_id,
                status="starting",
                symbol=symbol,
                timeframe=timeframe
            )

            return result
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

    @router.get("/deploy/enhanced")
    async def deploy_enhanced_endpoint(
        request,  # EnhancedDeploymentRequest
        deployer=Depends(get_enhanced_deployer),
    ):
        """
        Deploy a bot with enhanced paper trading (multiple formats).
        """
        try:
            result = deployer.deploy_bot(request)

            # Broadcast deployment update
            await broadcast_deployment_update(
                agent_id=result.bot_id,
                status=result.status,
                symbol=request.symbol
            )

            return result
        except Exception as e:
            logger.error(f"Enhanced deployment failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Deployment failed: {str(e)}"
            )

    return router
