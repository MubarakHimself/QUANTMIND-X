"""
Backtest API Handler

Handles backtest API endpoints.
"""

import logging
import uuid
from typing import Dict

from .models import (
    BacktestRunRequest,
    BacktestRunResponse,
    BacktestResultResponse,
)

logger = logging.getLogger(__name__)


class BacktestAPIHandler:
    """
    Handles backtest API endpoints.

    Integrates with:
    - src/backtesting/mt5_engine.py for backtest execution
    - src/database/duckdb_connection.py for result storage
    - src/router/sentinel.py for regime filtering (spiced variants)
    """

    def __init__(self):
        """Initialize backtest API handler."""
        self._backtest_results: Dict[str, BacktestResultResponse] = {}

    def run_backtest(self, request: BacktestRunRequest) -> BacktestRunResponse:
        """
        Queue and run a backtest.

        Args:
            request: Backtest run request

        Returns:
            Backtest run response with backtest ID
        """
        try:
            # Generate unique backtest ID
            backtest_id = str(uuid.uuid4())

            logger.info(
                f"Queuing backtest {backtest_id}: "
                f"{request.symbol} {request.timeframe.value} "
                f"{request.variant.value} "
                f"{request.start_date} to {request.end_date}"
            )

            # Create response
            response = BacktestRunResponse(
                success=True,
                backtest_id=backtest_id,
                status="queued",
                message=f"Backtest queued for {request.symbol}",
                estimated_time_seconds=60  # Estimate
            )

            # In a real implementation, this would:
            # 1. Store request in database
            # 2. Queue backtest job
            # 3. Return immediately with backtest_id

            return response

        except Exception as e:
            logger.error(f"Error queuing backtest: {e}")
            return BacktestRunResponse(
                success=False,
                backtest_id="",
                status="failed",
                message=f"Error: {str(e)}"
            )

    def get_backtest_results(self, backtest_id: str) -> BacktestResultResponse:
        """
        Retrieve backtest results by ID.

        Args:
            backtest_id: Backtest identifier

        Returns:
            Backtest result response with metrics and trade history
        """
        try:
            # Check if we have cached results
            if backtest_id in self._backtest_results:
                return self._backtest_results[backtest_id]

            # In a real implementation, this would:
            # 1. Query DuckDB for backtest results
            # 2. Return metrics, equity curve, trade history

            # Return placeholder for now
            return BacktestResultResponse(
                backtest_id=backtest_id,
                status="not_found",
                error_message=f"Backtest {backtest_id} not found"
            )

        except Exception as e:
            logger.error(f"Error retrieving backtest results: {e}")
            return BacktestResultResponse(
                backtest_id=backtest_id,
                status="error",
                error_message=str(e)
            )
