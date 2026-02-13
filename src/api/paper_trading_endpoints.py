"""
Paper Trading API Endpoints

Exposes paper trading deployer functionality via REST API endpoints.
"""

import os
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from mcp_mt5.paper_trading.deployer import (
    PaperTradingDeployer,
    AgentDeploymentRequest,
    AgentDeploymentResult,
    PaperAgentStatus,
    AgentLogsResult,
)
from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
from src.api.tick_stream_handler import get_tick_handler
from src.api.websocket_endpoints import broadcast_paper_trading_update

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/paper-trading", tags=["paper-trading"])


def get_deployer() -> PaperTradingDeployer:
    """Dependency injection for PaperTradingDeployer."""
    return PaperTradingDeployer()


@router.post("/deploy", response_model=AgentDeploymentResult)
async def deploy_agent_endpoint(
    request: AgentDeploymentRequest,
    deployer: PaperTradingDeployer = Depends(get_deployer),
) -> AgentDeploymentResult:
    """
    Deploy a new paper trading agent.

    Creates a Docker container with the specified strategy code and MT5 credentials.
    """
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
        await broadcast_paper_trading_update(
            agent_id=result.agent_id,
            status="starting",
            symbol=symbol,
            timeframe=timeframe
        )
        
        return result
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")


@router.get("/agents", response_model=List[PaperAgentStatus])
async def list_agents(
    deployer: PaperTradingDeployer = Depends(get_deployer),
) -> List[PaperAgentStatus]:
    """
    List all paper trading agents.

    Returns status information for all deployed agent containers.
    """
    agents = deployer.list_agents()
    
    # Broadcast current status of all agents to UI
    for agent in agents:
        await broadcast_paper_trading_update(
            agent_id=agent.agent_id,
            status=agent.status,
            symbol=agent.symbol,
            timeframe=agent.timeframe,
            uptime_seconds=agent.uptime_seconds
        )
    
    return agents


@router.get("/agents/{agent_id}", response_model=PaperAgentStatus)
async def get_agent_endpoint(
    agent_id: str,
    deployer: PaperTradingDeployer = Depends(get_deployer),
) -> PaperAgentStatus:
    """
    Get status of a specific paper trading agent.

    Returns detailed status information for the specified agent.
    """
    agent = deployer.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("/agents/{agent_id}/stop")
async def stop_agent_endpoint(
    agent_id: str,
    deployer: PaperTradingDeployer = Depends(get_deployer),
) -> dict:
    """
    Stop a paper trading agent.

    Gracefully stops the specified agent container.
    """
    success = deployer.stop_agent(agent_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop agent")
    logger.info(f"Stopped agent {agent_id}")
    
    # Broadcast stop update to UI
    await broadcast_paper_trading_update(
        agent_id=agent_id,
        status="stopped"
    )
    
    return {"success": True, "message": f"Agent {agent_id} stopped successfully"}


@router.get("/agents/{agent_id}/logs", response_model=AgentLogsResult)
async def get_agent_logs_endpoint(
    agent_id: str,
    tail_lines: int = Query(default=100, ge=1, le=10000),
    deployer: PaperTradingDeployer = Depends(get_deployer),
) -> AgentLogsResult:
    """
    Get logs from a paper trading agent.

    Returns the last N log lines from the agent container.
    """
    try:
        logs = deployer.get_agent_logs(agent_id, tail_lines)
        return logs
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get logs for {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")


@router.post("/tick-data/subscribe")
async def subscribe_tick_data(
    symbol: str = Query(..., description="Symbol to subscribe to tick data (e.g., EURUSD)"),
):
    """
    Subscribe to live tick data for a symbol.

    Starts streaming real-time bid/ask prices via WebSocket.
    """
    try:
        config = {
            "vps_host": os.getenv("MT5_VPS_HOST", "localhost"),
            "vps_port": int(os.getenv("MT5_VPS_PORT", "5555")),
            "account_id": os.getenv("MT5_ACCOUNT_ID"),
        }
        mt5_adapter = MT5SocketAdapter(config)
        tick_handler = get_tick_handler(mt5_adapter)
        await tick_handler.subscribe(symbol)
        return {"success": True, "message": f"Subscribed to tick data for {symbol}"}
    except Exception as e:
        logger.error(f"Failed to subscribe to tick data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to subscribe: {str(e)}")


@router.post("/tick-data/unsubscribe")
async def unsubscribe_tick_data(
    symbol: str = Query(..., description="Symbol to unsubscribe from tick data"),
):
    """
    Unsubscribe from live tick data for a symbol.

    Stops streaming tick prices for the specified symbol.
    """
    try:
        config = {
            "vps_host": os.getenv("MT5_VPS_HOST", "localhost"),
            "vps_port": int(os.getenv("MT5_VPS_PORT", "5555")),
            "account_id": os.getenv("MT5_ACCOUNT_ID"),
        }
        mt5_adapter = MT5SocketAdapter(config)
        tick_handler = get_tick_handler(mt5_adapter)
        await tick_handler.unsubscribe(symbol)
        return {"success": True, "message": f"Unsubscribed from tick data for {symbol}"}
    except Exception as e:
        logger.error(f"Failed to unsubscribe from tick data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unsubscribe: {str(e)}")
