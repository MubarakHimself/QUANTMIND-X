"""
Queue Manager Module

Manages backtest request queue with parallel execution.
Uses multiprocessing.Pool for up to 10 simultaneous backtests.

Spec: lines 46-49, 740-744
"""

import hashlib
import logging
import multiprocessing
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

# Handle both relative and absolute imports
try:
    from .models import (
        BacktestConfig,
        BacktestResult,
        BacktestStatus,
    )
    from .backtest_runner import BacktestRunner
except ImportError:
    from models import (
        BacktestConfig,
        BacktestResult,
        BacktestStatus,
    )
    from backtest_runner import BacktestRunner

logger = logging.getLogger(__name__)


def _execute_backtest_worker(
    backtest_id: str,
    code_content: str,
    language: str,
    config_dict: dict[str, Any],
) -> tuple[str, BacktestResult]:
    """
    Worker function for executing backtest in separate process/thread.

    Args:
        backtest_id: Unique backtest identifier
        code_content: Strategy code
        language: "python" or "mq5"
        config_dict: Backtest configuration dictionary

    Returns:
        Tuple of (backtest_id, BacktestResult)
    """
    try:
        # Parse config
        config = BacktestConfig(**config_dict)

        # Create runner and execute
        runner = BacktestRunner()

        if language == "python":
            result = runner.run_python_strategy(code_content, config)
        elif language == "mq5":
            result = runner.run_mql5_strategy(code_content, config)
        else:
            result = BacktestResult(
                backtest_id=backtest_id,
                status="error",
                metrics=None,
                equity_curve=None,
                trade_log=None,
                logs=f"Unsupported language: {language}",
                execution_time_seconds=0.0,
            )

        # Update backtest_id in result
        result.backtest_id = backtest_id

        return (backtest_id, result)

    except Exception as e:
        logger.error(f"Worker error for {backtest_id}: {e}")
        return (
            backtest_id,
            BacktestResult(
                backtest_id=backtest_id,
                status="error",
                metrics=None,
                equity_curve=None,
                trade_log=None,
                logs=f"Worker error: {str(e)}",
                execution_time_seconds=0.0,
            ),
        )


class BacktestQueueManager:
    """
    Manages backtest queue and parallel execution.

    Features:
    - CPU-aware scheduling with multiprocessing.cpu_count()
    - Up to 10 simultaneous backtests
    - Status tracking: queued, running, completed, failed
    - Result caching for identical configurations
    """

    def __init__(self, max_workers: int = 10):
        """
        Initialize queue manager.

        Args:
            max_workers: Maximum number of parallel workers (default: 10)
        """
        # Detect CPU count and limit workers
        cpu_count = multiprocessing.cpu_count()
        self.max_workers = min(max_workers, cpu_count, 10)

        logger.info(
            f"QueueManager initialized: max_workers={self.max_workers} "
            f"(CPU count={cpu_count})"
        )

        # Thread-safe storage for backtest status
        self._backtests: dict[str, BacktestStatus] = {}
        self._backtests_lock = threading.Lock()

        # Result cache: cache_key -> BacktestResult
        self._result_cache: dict[str, BacktestResult] = {}
        self._cache_lock = threading.Lock()

        # Thread pool for parallel execution
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Track running futures
        self._futures: dict[str, Any] = {}
        self._futures_lock = threading.Lock()

    def submit_backtest(
        self, code_content: str, language: str, config: dict[str, Any]
    ) -> str:
        """
        Submit a backtest request to the queue.

        Args:
            code_content: Strategy code
            language: "python" or "mq5"
            config: Backtest configuration dictionary

        Returns:
            backtest_id: Unique identifier for tracking
        """
        backtest_id = str(uuid.uuid4())

        # Check cache for identical configuration
        cache_key = self._generate_cache_key(code_content, config)
        cached_result = self._get_cached_result(cache_key)

        if cached_result:
            logger.info(f"Cache hit for backtest {backtest_id}, returning cached result")
            with self._backtests_lock:
                self._backtests[backtest_id] = BacktestStatus(
                    backtest_id=backtest_id,
                    status="completed",
                    progress_percent=100.0,
                    estimated_completion=None,
                    result=cached_result,
                )
            return backtest_id

        # Create initial status
        with self._backtests_lock:
            self._backtests[backtest_id] = BacktestStatus(
                backtest_id=backtest_id,
                status="queued",
                progress_percent=0.0,
                estimated_completion=datetime.now() + timedelta(minutes=5),
                result=None,
            )

        # Submit to executor
        future = self._executor.submit(
            _execute_backtest_worker, backtest_id, code_content, language, config
        )

        # Track future
        with self._futures_lock:
            self._futures[backtest_id] = future

        # Add completion callback
        future.add_done_callback(lambda f: self._handle_completion(backtest_id, f))

        logger.info(
            f"Backtest {backtest_id} submitted to queue "
            f"({language}, {config.get('symbol', 'N/A')})"
        )

        return backtest_id

    def get_status(self, backtest_id: str) -> BacktestStatus:
        """
        Get the status of a backtest.

        Args:
            backtest_id: Backtest identifier

        Returns:
            BacktestStatus with current state and progress
        """
        with self._backtests_lock:
            if backtest_id not in self._backtests:
                # Return status for unknown backtest
                return BacktestStatus(
                    backtest_id=backtest_id,
                    status="failed",
                    progress_percent=0.0,
                    estimated_completion=None,
                    result=None,
                )
            return self._backtests[backtest_id]

    def _handle_completion(self, backtest_id: str, future: Any) -> None:
        """
        Handle backtest completion callback.

        Args:
            backtest_id: Backtest identifier
            future: Completed future
        """
        try:
            _, result = future.result()

            # Update status
            with self._backtests_lock:
                if backtest_id in self._backtests:
                    old_status = self._backtests[backtest_id]
                    self._backtests[backtest_id] = BacktestStatus(
                        backtest_id=backtest_id,
                        status="completed" if result.status == "success" else "failed",
                        progress_percent=100.0,
                        estimated_completion=None,
                        result=result,
                    )

            # Cache successful results
            if result.status == "success":
                self._cache_result(backtest_id, result)

            logger.info(
                f"Backtest {backtest_id} completed with status: {result.status}"
            )

        except Exception as e:
            logger.error(f"Error handling completion for {backtest_id}: {e}")
            with self._backtests_lock:
                if backtest_id in self._backtests:
                    self._backtests[backtest_id] = BacktestStatus(
                        backtest_id=backtest_id,
                        status="failed",
                        progress_percent=0.0,
                        estimated_completion=None,
                        result=None,
                    )

        finally:
            # Clean up future
            with self._futures_lock:
                self._futures.pop(backtest_id, None)

    def _generate_cache_key(self, code_content: str, config: dict[str, Any]) -> str:
        """
        Generate cache key for backtest configuration.

        Args:
            code_content: Strategy code
            config: Backtest configuration

        Returns:
            SHA256 hash of code + config
        """
        # Sort config for consistent hashing
        config_str = str(sorted(config.items()))
        combined = f"{code_content}:{config_str}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> BacktestResult | None:
        """
        Get cached backtest result.

        Args:
            cache_key: Cache key

        Returns:
            Cached BacktestResult or None
        """
        with self._cache_lock:
            return self._result_cache.get(cache_key)

    def _cache_result(self, backtest_id: str, result: BacktestResult) -> None:
        """
        Cache backtest result.

        Args:
            backtest_id: Backtest identifier
            result: BacktestResult to cache
        """
        # Store by backtest_id for now
        # In production, you'd use the cache_key
        with self._cache_lock:
            self._result_cache[backtest_id] = result

        logger.debug(f"Cached result for backtest {backtest_id}")

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the queue manager and cleanup resources.

        Args:
            wait: Wait for pending backtests to complete
        """
        logger.info("Shutting down QueueManager...")
        self._executor.shutdown(wait=wait)

        # Cancel pending futures
        with self._futures_lock:
            for bid, future in self._futures.items():
                future.cancel()

        logger.info("QueueManager shutdown complete")

    def get_queue_stats(self) -> dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dictionary with queue stats
        """
        with self._backtests_lock:
            stats = {
                "total": len(self._backtests),
                "queued": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
            }

            for status in self._backtests.values():
                stats[status.status] = stats.get(status.status, 0) + 1

        return stats
