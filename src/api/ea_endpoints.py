"""
EA Lifecycle API Endpoints.

Provides REST API endpoints for managing EA lifecycle operations.
Live trading deployment is explicitly prohibited.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging

from src.agents.tools.ea_lifecycle import EALifecycleTools, EALifecycleStatus
from src.api.dependencies import get_ea_lifecycle_tools

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ea", tags=["EA Lifecycle"])


class EACreationRequest(BaseModel):
    """Request model for EA creation"""
    strategy_code: str
    parameters: Dict[str, Any]


class EACreationResponse(BaseModel):
    """Response model for EA creation"""
    ea_id: str
    status: str
    created_at: float


class EAValidationRequest(BaseModel):
    """Request model for EA validation"""
    pass


class EAValidationResponse(BaseModel):
    """Response model for EA validation"""
    ea_id: str
    status: str
    validation_errors: List[str]
    validated_at: float


class EABacktestRequest(BaseModel):
    """Request model for EA backtest"""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_balance: float = 10000.0


class EABacktestResponse(BaseModel):
    """Response model for EA backtest"""
    ea_id: str
    metrics: Dict[str, Any]
    equity_curve: List[Dict[str, float]]
    drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    status: str
    completed_at: float


class EAStressTestRequest(BaseModel):
    """Request model for EA stress test"""
    volatility_multiplier: float = 2.0
    liquidity_multiplier: float = 0.5


class EAStressTestResponse(BaseModel):
    """Response model for EA stress test"""
    ea_id: str
    status: str
    stress_test_results: Dict[str, Any]
    completed_at: float


class EAMonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulation"""
    num_simulations: int = 1000
    confidence_level: float = 0.95


class EAMonteCarloResponse(BaseModel):
    """Response model for Monte Carlo simulation"""
    ea_id: str
    simulation_results: List[Dict[str, Any]]
    confidence_intervals: Dict[str, float]
    probability_of_ruin: float
    status: str
    completed_at: float


class EADeployPaperRequest(BaseModel):
    """Request model for paper trading deployment"""
    initial_balance: float = 10000.0
    leverage: int = 50


class EADeployPaperResponse(BaseModel):
    """Response model for paper trading deployment"""
    ea_id: str
    status: str
    paper_trading_id: str
    deployed_at: float
    initial_balance: float


class EAMonitorResponse(BaseModel):
    """Response model for EA monitoring"""
    ea_id: str
    status: str
    current_equity: float
    unrealized_pnl: float
    trades_executed: int
    last_update: float


class EAOptimizationRequest(BaseModel):
    """Request model for EA optimization"""
    optimization_method: str = "genetic"
    population_size: int = 50
    generations: int = 20


class EAOptimizationResponse(BaseModel):
    """Response model for EA optimization"""
    ea_id: str
    status: str
    optimization_results: Dict[str, Any]
    optimized_at: float


class EAStopResponse(BaseModel):
    """Response model for stopping EA"""
    ea_id: str
    status: str
    stopped_at: float
    final_equity: float
    total_pnl: float


class EAListResponse(BaseModel):
    """Response model for listing EAs"""
    eas: List[Dict[str, Any]]


@router.post("/create", response_model=EACreationResponse, status_code=status.HTTP_201_CREATED)
async def create_ea(
    request: EACreationRequest,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EACreationResponse:
    """
    Create a new EA (Expert Advisor)
    """
    try:
        result = tools.create_ea(request.strategy_code, request.parameters)
        return EACreationResponse(
            ea_id=result.ea_id,
            status=result.status,
            created_at=result.created_at
        )
    except Exception as e:
        logger.error(f"Failed to create EA: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create EA: {str(e)}"
        )


@router.get("/list", response_model=EAListResponse)
async def list_eas(
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EAListResponse:
    """
    List all EAs
    """
    try:
        eas = tools.list_eas()
        return EAListResponse(eas=eas)
    except Exception as e:
        logger.error(f"Failed to list EAs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list EAs: {str(e)}"
        )


@router.post("/{ea_id}/validate", response_model=EAValidationResponse)
async def validate_ea(
    ea_id: str,
    request: EAValidationRequest,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EAValidationResponse:
    """
    Validate an EA
    """
    try:
        result = tools.validate_ea(ea_id)
        return EAValidationResponse(
            ea_id=result.ea_id,
            status=result.status,
            validation_errors=result.validation_errors,
            validated_at=result.validated_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to validate EA {ea_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate EA: {str(e)}"
        )


@router.post("/{ea_id}/backtest", response_model=EABacktestResponse)
async def backtest_ea(
    ea_id: str,
    request: EABacktestRequest,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EABacktestResponse:
    """
    Run backtest on an EA
    """
    try:
        # First validate the EA
        tools.validate_ea(ea_id)

        result = tools.backtest_ea(
            ea_id,
            {
                'symbol': request.symbol,
                'timeframe': request.timeframe,
                'start_date': request.start_date,
                'end_date': request.end_date,
                'initial_balance': request.initial_balance
            }
        )
        return EABacktestResponse(
            ea_id=result.ea_id,
            metrics=result.metrics,
            equity_curve=result.equity_curve,
            drawdown=result.drawdown,
            sharpe_ratio=result.sharpe_ratio,
            sortino_ratio=result.sortino_ratio,
            status=result.status,
            completed_at=result.completed_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to backtest EA {ea_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to backtest EA: {str(e)}"
        )


@router.post("/{ea_id}/stress-test", response_model=EAStressTestResponse)
async def stress_test_ea(
    ea_id: str,
    request: EAStressTestRequest,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EAStressTestResponse:
    """
    Run stress test on an EA
    """
    try:
        # First validate and backtest the EA
        tools.validate_ea(ea_id)
        tools.backtest_ea(ea_id, {})  # Use default backtest params

        result = tools.stress_test_ea(
            ea_id,
            {
                'volatility_multiplier': request.volatility_multiplier,
                'liquidity_multiplier': request.liquidity_multiplier
            }
        )
        return EAStressTestResponse(
            ea_id=result.ea_id,
            status=result.status,
            stress_test_results=result.stress_test_results,
            completed_at=result.completed_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to stress test EA {ea_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stress test EA: {str(e)}"
        )


@router.post("/{ea_id}/monte-carlo", response_model=EAMonteCarloResponse)
async def monte_carlo_sim(
    ea_id: str,
    request: EAMonteCarloRequest,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EAMonteCarloResponse:
    """
    Run Monte Carlo simulation on an EA
    """
    try:
        # First validate and backtest the EA
        tools.validate_ea(ea_id)
        tools.backtest_ea(ea_id, {})  # Use default backtest params

        result = tools.monte_carlo_sim(
            ea_id,
            {
                'num_simulations': request.num_simulations,
                'confidence_level': request.confidence_level
            }
        )
        return EAMonteCarloResponse(
            ea_id=result.ea_id,
            simulation_results=result.simulation_results,
            confidence_intervals=result.confidence_intervals,
            probability_of_ruin=result.probability_of_ruin,
            status=result.status,
            completed_at=result.completed_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to run Monte Carlo simulation on EA {ea_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run Monte Carlo simulation: {str(e)}"
        )


@router.post("/{ea_id}/deploy-paper", response_model=EADeployPaperResponse)
async def deploy_paper(
    ea_id: str,
    request: EADeployPaperRequest,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EADeployPaperResponse:
    """
    Deploy EA to paper trading environment
    """
    try:
        # First validate and backtest the EA
        tools.validate_ea(ea_id)
        tools.backtest_ea(ea_id, {})  # Use default backtest params

        result = tools.deploy_paper(
            ea_id,
            {
                'initial_balance': request.initial_balance,
                'leverage': request.leverage
            }
        )
        return EADeployPaperResponse(
            ea_id=result.ea_id,
            status=result.status,
            paper_trading_id=result.paper_trading_id,
            deployed_at=result.deployed_at,
            initial_balance=result.initial_balance
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to deploy EA {ea_id} to paper trading: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy EA to paper trading: {str(e)}"
        )


@router.get("/{ea_id}/monitor", response_model=EAMonitorResponse)
async def monitor_ea(
    ea_id: str,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EAMonitorResponse:
    """
    Monitor EA in paper trading
    """
    try:
        result = tools.monitor_ea(ea_id)
        return EAMonitorResponse(
            ea_id=result.ea_id,
            status=result.status,
            current_equity=result.current_equity,
            unrealized_pnl=result.unrealized_pnl,
            trades_executed=result.trades_executed,
            last_update=result.last_update
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to monitor EA {ea_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to monitor EA: {str(e)}"
        )


@router.post("/{ea_id}/optimize", response_model=EAOptimizationResponse)
async def optimize_ea(
    ea_id: str,
    request: EAOptimizationRequest,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EAOptimizationResponse:
    """
    Optimize EA parameters
    """
    try:
        # First validate and backtest the EA
        tools.validate_ea(ea_id)
        tools.backtest_ea(ea_id, {})  # Use default backtest params

        result = tools.optimize_ea(
            ea_id,
            {
                'optimization_method': request.optimization_method,
                'population_size': request.population_size,
                'generations': request.generations
            }
        )
        return EAOptimizationResponse(
            ea_id=result.ea_id,
            status=result.status,
            optimization_results=result.optimization_results,
            optimized_at=result.optimization_results.optimized_at
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to optimize EA {ea_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize EA: {str(e)}"
        )


@router.post("/{ea_id}/stop", response_model=EAStopResponse)
async def stop_ea(
    ea_id: str,
    tools: EALifecycleTools = Depends(get_ea_lifecycle_tools)
) -> EAStopResponse:
    """
    Stop EA (paper trading or monitoring)
    """
    try:
        result = tools.stop_ea(ea_id)
        return EAStopResponse(
            ea_id=result.ea_id,
            status=result.status,
            stopped_at=result.stopped_at,
            final_equity=result.final_equity,
            total_pnl=result.total_pnl
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to stop EA {ea_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop EA: {str(e)}"
        )