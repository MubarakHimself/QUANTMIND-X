"""
Correlation Sensor API Endpoints

REST API for the Correlation Sensor which provides:
- max_eigenvalue and RMT threshold analysis
- Correlation matrix for M5 and H1 timeframes
- Regime classification: CORRELATED | UNCORRELATED | NEUTRAL

Endpoints:
- GET /api/risk/correlation/sensor - Full sensor data
- GET /api/risk/correlation/matrix/{timeframe} - Correlation matrix for M5 or H1
"""

from __future__ import annotations

import logging
from typing import Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/risk/correlation", tags=["risk", "correlation"])


class CorrelationSensorResponse(BaseModel):
    """Response model for correlation sensor endpoint."""

    max_eigenvalue: float
    rmt_threshold: float
    is_correlated: bool
    regime: str
    m5_matrix: list
    h1_matrix: list
    eigenvalues: list
    symbols: list
    timestamp: float


class CorrelationMatrixResponse(BaseModel):
    """Response model for correlation matrix endpoint."""

    timeframe: str
    matrix: list
    symbols: list
    timestamp: float


def _compute_sensor_data() -> Dict:
    """
    Compute correlation sensor data.

    Returns correlation sensor data from the physics module,
    with error handling for missing dependencies.
    """
    try:
        from src.risk.physics.correlation_sensor import compute_correlation_sensor
        return compute_correlation_sensor()
    except ImportError as e:
        logger.error(f"Failed to import correlation_sensor module: {e}")
        raise HTTPException(
            status_code=503,
            detail="Correlation sensor module not available"
        )
    except Exception as e:
        logger.error(f"Failed to compute correlation sensor data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Correlation sensor computation failed: {str(e)}"
        )


def _get_matrix_data(timeframe: str) -> Dict:
    """
    Get correlation matrix for a specific timeframe.

    Args:
        timeframe: "M5" or "H1"

    Returns:
        Matrix data from the correlation sensor
    """
    try:
        from src.risk.physics.correlation_sensor import get_correlation_matrix
        return get_correlation_matrix(timeframe)
    except ImportError as e:
        logger.error(f"Failed to import correlation_sensor module: {e}")
        raise HTTPException(
            status_code=503,
            detail="Correlation sensor module not available"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get correlation matrix: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get correlation matrix: {str(e)}"
        )


@router.get("/sensor", response_model=CorrelationSensorResponse)
async def get_correlation_sensor_data() -> CorrelationSensorResponse:
    """
    Get full correlation sensor data.

    Returns:
        CorrelationSensorResponse with:
        - max_eigenvalue: Largest eigenvalue from correlation matrix
        - rmt_threshold: Random Matrix Theory threshold
        - is_correlated: True if market is in correlated regime
        - regime: "CORRELATED" | "UNCORRELATED" | "NEUTRAL"
        - m5_matrix: 10x10 correlation matrix for M5 timeframe
        - h1_matrix: 10x10 correlation matrix for H1 timeframe
        - eigenvalues: Top 10 eigenvalues
        - symbols: List of 10 forex symbols
        - timestamp: Unix timestamp of computation
    """
    data = _compute_sensor_data()

    return CorrelationSensorResponse(
        max_eigenvalue=data["max_eigenvalue"],
        rmt_threshold=data["rmt_threshold"],
        is_correlated=data["is_correlated"],
        regime=data["regime"],
        m5_matrix=data["m5_matrix"],
        h1_matrix=data["h1_matrix"],
        eigenvalues=data["eigenvalues"],
        symbols=data["symbols"],
        timestamp=data["timestamp"]
    )


@router.get("/matrix/{timeframe}", response_model=CorrelationMatrixResponse)
async def get_correlation_matrix_by_timeframe(
    timeframe: str
) -> CorrelationMatrixResponse:
    """
    Get correlation matrix for a specific timeframe.

    Args:
        timeframe: Timeframe identifier - "M5" or "H1"

    Returns:
        CorrelationMatrixResponse with:
        - timeframe: The requested timeframe
        - matrix: NxN correlation matrix
        - symbols: List of symbols in the matrix
        - timestamp: Unix timestamp of computation

    Raises:
        HTTPException 400: If timeframe is not "M5" or "H1"
        HTTPException 500: If computation fails
    """
    # Validate timeframe
    timeframe_upper = timeframe.upper()
    if timeframe_upper not in ("M5", "H1"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe: {timeframe}. Use 'M5' or 'H1'."
        )

    data = _get_matrix_data(timeframe_upper)

    return CorrelationMatrixResponse(
        timeframe=data["timeframe"],
        matrix=data["matrix"],
        symbols=data["symbols"],
        timestamp=data["timestamp"]
    )
