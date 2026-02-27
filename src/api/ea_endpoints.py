"""
EA Lifecycle API Endpoints

FastAPI router for Expert Advisor lifecycle management.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
import logging

from src.agents.tools.ea_lifecycle import EALifecycleTools

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ea", tags=["ea"])

# Initialize EA tools
ea_tools = EALifecycleTools()


@router.post("/create")
async def create_ea(
    strategy_name: str,
    ea_name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create an Expert Advisor from a strategy.

    Args:
        strategy_name: Name of the strategy to convert
        ea_name: Name for the EA (optional)
        parameters: Trading parameters (optional)

    Returns:
        EA creation result with file path
    """
    try:
        result = ea_tools.create_ea(
            strategy_name=strategy_name,
            ea_name=ea_name,
            parameters=parameters
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error creating EA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_eas() -> Dict[str, Any]:
    """
    List all available Expert Advisors.

    Returns:
        List of EAs with metadata
    """
    try:
        from pathlib import Path

        ea_output_path = Path("/home/mubarkahimself/Desktop/QUANTMINDX/output/expert_advisors")
        ea_files = []

        if ea_output_path.exists():
            for ea_file in ea_output_path.glob("*.mq5"):
                ea_files.append({
                    "name": ea_file.stem,
                    "file_path": str(ea_file),
                    "size": ea_file.stat().st_size,
                    "modified": ea_file.stat().st_mtime
                })

        return {
            "success": True,
            "eas": ea_files,
            "count": len(ea_files)
        }

    except Exception as e:
        logger.error(f"Error listing EAs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate")
async def validate_ea(ea_name: str) -> Dict[str, Any]:
    """
    Validate an Expert Advisor.

    Args:
        ea_name: Name of the EA to validate

    Returns:
        Validation results
    """
    try:
        result = ea_tools.validate_ea(ea_name=ea_name)

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error validating EA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def backtest_ea(
    ea_name: str,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    date_from: str = "2023-01-01",
    date_to: str = "2023-12-31",
    deposit: int = 10000
) -> Dict[str, Any]:
    """
    Run backtest for an Expert Advisor.

    Args:
        ea_name: Name of the EA to backtest
        symbol: Trading symbol
        timeframe: Timeframe
        date_from: Start date
        date_to: End date
        deposit: Initial deposit

    Returns:
        Backtest results
    """
    try:
        result = ea_tools.backtest_ea(
            ea_name=ea_name,
            symbol=symbol,
            timeframe=timeframe,
            date_from=date_from,
            date_to=date_to,
            deposit=deposit
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deploy-paper")
async def deploy_paper(
    ea_name: str,
    account_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Deploy EA to paper trading.

    Args:
        ea_name: Name of the EA to deploy
        account_id: Paper trading account ID

    Returns:
        Deployment status
    """
    try:
        result = ea_tools.deploy_paper(
            ea_name=ea_name,
            account_id=account_id
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error deploying to paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_ea(
    ea_name: str,
    environment: str = "paper"
) -> Dict[str, Any]:
    """
    Stop a running EA.

    Args:
        ea_name: Name of the EA to stop
        environment: Environment where EA is running

    Returns:
        Stop status
    """
    try:
        result = ea_tools.stop_ea(
            ea_name=ea_name,
            environment=environment
        )

        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error"))

        return result

    except Exception as e:
        logger.error(f"Error stopping EA: {e}")
        raise HTTPException(status_code=500, detail=str(e))
