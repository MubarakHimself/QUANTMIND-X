"""
HMM Walk-Forward Analysis Calibrator
=====================================

Dynamically calibrates WFA window size based on the average regime
transition interval from the Kamatera T2 HMM.

The WFA window is calculated as f(avg_regime_transition_interval):
- Uses rolling 1-month windows as baseline for scalping variants
- Dynamic window sizing ensures sufficient regime coverage
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# Default base URL for Kamatera T2 HMM API
HMM_T2_BASE_URL = "http://kamatera-t2:8080"


@dataclass
class WfaWindowConfig:
    """
    Configuration for Walk-Forward Analysis window.

    Attributes:
        window_days: Number of days in the WFA window
        window_type: Type of window (rolling_1month or rolling_adaptive)
        baseline: Baseline strategy type (scalping_variants)
        avg_regime_interval_used: The avg_regime_interval used to calculate window_days
    """
    window_days: int
    window_type: str  # "rolling_1month" or "rolling_adaptive"
    baseline: str  # "scalping_variants"
    avg_regime_interval_used: float


class WfaCalibrator:
    """
    Dynamically calibrates Walk-Forward Analysis window based on regime transitions.

    The optimal WFA window is calculated as f(avg_regime_transition_interval):
    - Window size = avg_interval * 4 (provides sufficient regime coverage)
    - Clamped between 7 and 30 days
    - Uses rolling 1-month windows as baseline for scalping variants

    This ensures the WFA always has enough data to capture at least one
    full regime cycle while remaining responsive to regime changes.
    """

    # Window calculation constants
    WINDOW_MULTIPLIER: float = 4.0
    MIN_WINDOW_DAYS: int = 7
    MAX_WINDOW_DAYS: int = 30
    ROLLING_1MONTH_THRESHOLD: int = 28  # 1 month ~ 28 days

    def __init__(self, hmm_t2_base_url: Optional[str] = None):
        """
        Initialize the WFA calibrator.

        Args:
            hmm_t2_base_url: Base URL for Kamatera T2 HMM API.
                           Defaults to HMM_T2_BASE_URL constant.
        """
        self.hmm_t2_base_url = hmm_t2_base_url or HMM_T2_BASE_URL
        logger.info(f"WfaCalibrator initialized with T2 base URL: {self.hmm_t2_base_url}")

    async def get_avg_regime_transition_interval(self) -> float:
        """
        Query Kamatera T2 HMM for average regime transition interval in days.

        Makes an HTTP GET request to the HMM metrics endpoint to retrieve
        the avg_regime_transition_interval metric.

        Returns:
            Average regime transition interval in days

        Raises:
            Exception: If the API call fails
        """
        import aiohttp

        url = f"{self.hmm_t2_base_url}/api/hmm/metrics/regime_transition_interval"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        avg_interval = data.get("avg_interval_days", 7.0)
                        logger.info(f"Retrieved avg_regime_transition_interval: {avg_interval} days")
                        return float(avg_interval)
                    else:
                        logger.warning(
                            f"API returned status {response.status}, using default interval"
                        )
                        return 7.0  # Default to 7 days on error
        except Exception as e:
            logger.warning(f"Failed to query T2 HMM: {e}, using default interval")
            return 7.0  # Default to 7 days on error

    async def calibrate_wfa_window(self) -> WfaWindowConfig:
        """
        Calculate optimal WFA window size from avg_regime_transition_interval.

        The window is calculated as:
        - window_days = avg_interval * WINDOW_MULTIPLIER
        - Clamped between MIN_WINDOW_DAYS (7) and MAX_WINDOW_DAYS (30)
        - window_type = "rolling_1month" if window_days >= 28, else "rolling_adaptive"

        Returns:
            WfaWindowConfig with calculated window parameters
        """
        avg_interval = await self.get_avg_regime_transition_interval()

        # Calculate window size: avg_interval * 4, clamped to [7, 30]
        window_days = int(avg_interval * self.WINDOW_MULTIPLIER)
        window_days = max(self.MIN_WINDOW_DAYS, min(self.MAX_WINDOW_DAYS, window_days))

        # Determine window type based on size
        if window_days >= self.ROLLING_1MONTH_THRESHOLD:
            window_type = "rolling_1month"
        else:
            window_type = "rolling_adaptive"

        config = WfaWindowConfig(
            window_days=window_days,
            window_type=window_type,
            baseline="scalping_variants",
            avg_regime_interval_used=avg_interval
        )

        logger.info(
            f"WFA window calibrated: {window_days} days ({window_type}) "
            f"based on avg_interval={avg_interval}"
        )

        return config

    def calculate_window_sync(self, avg_interval: float) -> WfaWindowConfig:
        """
        Synchronously calculate WFA window (no API call).

        Useful for testing or when avg_interval is already known.

        Args:
            avg_interval: Average regime transition interval in days

        Returns:
            WfaWindowConfig with calculated window parameters
        """
        window_days = int(avg_interval * self.WINDOW_MULTIPLIER)
        window_days = max(self.MIN_WINDOW_DAYS, min(self.MAX_WINDOW_DAYS, window_days))

        if window_days >= self.ROLLING_1MONTH_THRESHOLD:
            window_type = "rolling_1month"
        else:
            window_type = "rolling_adaptive"

        return WfaWindowConfig(
            window_days=window_days,
            window_type=window_type,
            baseline="scalping_variants",
            avg_regime_interval_used=avg_interval
        )


# Global singleton instance
_wfa_calibrator_instance: Optional[WfaCalibrator] = None


def get_wfa_calibrator() -> WfaCalibrator:
    """Get the global WfaCalibrator singleton instance."""
    global _wfa_calibrator_instance
    if _wfa_calibrator_instance is None:
        _wfa_calibrator_instance = WfaCalibrator()
    return _wfa_calibrator_instance
