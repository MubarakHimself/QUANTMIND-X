"""
Edge Case Handling for Enhanced Kelly Position Sizing.

Provides comprehensive edge case handling for:
- Insufficient trade history
- Missing or stale market data
- Zero/negative payoff scenarios
- Extreme market events (flash crashes, gap openings)
- Broker constraints
- Margin requirements
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from .kelly_config import EnhancedKellyConfig

logger = logging.getLogger(__name__)


class EdgeCaseHandler:
    """
    Handles edge cases for Enhanced Kelly position sizing.

    Provides fallback values and safety mechanisms when:
    - Data is missing or insufficient
    - Market conditions are extreme
    - Broker constraints are violated
    - Margin is insufficient
    """

    def __init__(self, config: EnhancedKellyConfig):
        """
        Initialize edge case handler.

        Args:
            config: Enhanced Kelly configuration
        """
        self.config = config
        self.last_known_state: Dict[str, Any] = {}

    def handle_insufficient_history(
        self,
        trade_count: int,
        fallback_risk_pct: Optional[float] = None
    ) -> tuple[float, str]:
        """
        Handle insufficient trade history.

        Args:
            trade_count: Number of trades available
            fallback_risk_pct: Optional fallback risk percentage

        Returns:
            (risk_percentage, warning_message)
        """
        if trade_count == 0:
            msg = "NO DATA: No trade history available - using conservative fallback"
            risk_pct = fallback_risk_pct or self.config.fallback_risk_pct
            logger.warning(msg)
            return risk_pct, msg

        elif trade_count < 10:
            msg = f"VERY LOW: Only {trade_count} trades - Kelly unreliable, using 0.5% risk"
            logger.warning(msg)
            return 0.005, msg

        elif trade_count < self.config.min_trade_history:
            fallback = fallback_risk_pct or self.config.fallback_risk_pct
            msg = f"LOW: {trade_count}/{self.config.min_trade_history} trades - using {fallback*100:.1f}% fallback"
            logger.warning(msg)
            return fallback, msg

        else:
            # Sufficient history
            return None, ""

    def handle_missing_physics_data(
        self,
        sensor_name: str,
        last_value: Optional[float] = None
    ) -> tuple[float, str]:
        """
        Handle missing physics sensor data.

        Args:
            sensor_name: Name of the sensor that failed
            last_value: Last known value (for decay calculation)

        Returns:
            (safety_multiplier, warning_message)
        """
        msg = f"MISSING DATA: {sensor_name} failed - using safety mode (0.5x multiplier)"
        logger.warning(msg)

        # Safety mode: 50% position reduction
        return 0.5, msg

    def handle_stale_physics_data(
        self,
        sensor_name: str,
        data_age_seconds: float,
        max_age_seconds: float = 300.0
    ) -> tuple[Optional[float], str]:
        """
        Handle stale physics data.

        Args:
            sensor_name: Name of the sensor
            data_age_seconds: Age of the data in seconds
            max_age_seconds: Maximum allowed age

        Returns:
            (decay_factor, warning_message) or (None, "") if data is fresh
        """
        if data_age_seconds < max_age_seconds:
            return None, ""

        # Calculate decay factor based on staleness
        # Decay linearly from 1.0 to 0.5 over max_age_seconds
        excess_age = data_age_seconds - max_age_seconds
        decay_factor = max(0.5, 1.0 - (excess_age / max_age_seconds) * 0.5)

        msg = f"STALE DATA: {sensor_name} is {data_age_seconds:.0f}s old - applying {decay_factor:.2f}x decay"
        logger.warning(msg)

        return decay_factor, msg

    def handle_negative_expectancy(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> tuple[float, str]:
        """
        Handle negative or zero expectancy scenarios.

        Args:
            win_rate: Current win rate
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            (position_size, reason_message)
        """
        if avg_loss == 0:
            msg = "ZERO LOSS: No loss history - cannot calculate Kelly"
            logger.error(msg)
            return 0.0, msg

        risk_reward = avg_win / avg_loss
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        if expectancy <= 0:
            msg = f"NEGATIVE EXPECTANCY: Win rate {win_rate:.1%}, R:R {risk_reward:.2f} - no edge detected"
            logger.warning(msg)
            return 0.0, msg

        return None, ""

    def handle_flash_crash(
        self,
        current_atr: float,
        average_atr: float,
        atr_suspicion_threshold: float = 1.5
    ) -> tuple[Optional[float], str]:
        """
        Handle flash crash detection.

        Args:
            current_atr: Current ATR value
            average_atr: Average ATR value
            atr_suspicion_threshold: ATR ratio threshold for flash crash

        Returns:
            (penalty_multiplier, warning_message) or (None, "") if normal
        """
        atr_ratio = current_atr / average_atr

        if atr_ratio > atr_suspicion_threshold:
            # Immediate 50% risk reduction for flash crashes
            msg = f"FLASH CRASH: ATR ratio {atr_ratio:.2f} - applying 50% risk reduction"
            logger.critical(msg)
            return 0.5, msg

        return None, ""

    def handle_gap_opening(
        self,
        open_price: float,
        previous_close: float,
        average_atr_pips: float,
        gap_threshold_multiple: float = 2.0
    ) -> tuple[bool, str]:
        """
        Handle gap opening detection.

        Args:
            open_price: Current opening price
            previous_close: Previous session close price
            average_atr_pips: Average ATR in pips
            gap_threshold_multiple: Gap size as multiple of ATR

        Returns:
            (skip_trade, reason_message)
        """
        gap_size = abs(open_price - previous_close)
        gap_threshold = average_atr_pips * gap_threshold_multiple

        if gap_size > gap_threshold:
            msg = f"GAP OPENING: Gap of {gap_size:.1f} pips exceeds threshold ({gap_threshold:.1f}) - skipping trade"
            logger.warning(msg)
            return True, msg

        return False, ""

    def handle_broker_constraints(
        self,
        position_size: float,
        min_lot: float,
        max_lot: float,
        lot_step: float
    ) -> tuple[float, list[str]]:
        """
        Handle broker lot size constraints.

        Args:
            position_size: Calculated position size
            min_lot: Broker minimum lot size
            max_lot: Broker maximum lot size
            lot_step: Broker lot step size

        Returns:
            (adjusted_position_size, list_of_adjustments)
        """
        adjustments = []
        adjusted_size = position_size

        # Check minimum lot
        if adjusted_size < min_lot:
            if self.config.allow_zero_position:
                adjustments.append(f"Position below minimum lot ({min_lot}) - returning 0")
                return 0.0, adjustments
            else:
                adjustments.append(f"Position below minimum lot ({min_lot}) - using min lot")
                adjusted_size = min_lot

        # Check maximum lot
        elif adjusted_size > max_lot:
            adjustments.append(f"Position exceeds maximum lot ({max_lot}) - capping at max")
            adjusted_size = max_lot

        # Round to lot step
        steps = int(adjusted_size / lot_step)
        rounded_size = steps * lot_step

        if rounded_size != adjusted_size and adjusted_size >= min_lot:
            adjustments.append(f"Rounded to lot step ({lot_step}): {adjusted_size:.3f} -> {rounded_size:.3f}")
            adjusted_size = rounded_size

        return adjusted_size, adjustments

    def handle_margin_constraints(
        self,
        position_size_lots: float,
        required_margin_per_lot: float,
        free_margin: float,
        margin_available: float
    ) -> tuple[float, str]:
        """
        Handle margin requirement constraints.

        Args:
            position_size_lots: Position size in lots
            required_margin_per_lot: Margin required per lot
            free_margin: Available free margin
            margin_available: Margin available for trading

        Returns:
            (adjusted_position_size, warning_message)
        """
        required_margin = position_size_lots * required_margin_per_lot

        if required_margin > free_margin:
            # Calculate maximum lots that fit in free margin
            max_lots = int(free_margin / required_margin_per_lot)
            max_lots = max(0, max_lots)  # Ensure non-negative

            msg = f"INSUFFICIENT MARGIN: Required ${required_margin:.2f}, available ${free_margin:.2f} - reducing to {max_lots} lots"
            logger.warning(msg)

            return float(max_lots), msg

        return position_size_lots, ""

    def handle_extreme_payoff_ratio(
        self,
        payoff_ratio: float,
        max_payoff: float = 10.0
    ) -> tuple[float, str]:
        """
        Handle extreme payoff ratios.

        Args:
            payoff_ratio: Calculated payoff ratio (avg_win / avg_loss)
            max_payoff: Maximum reasonable payoff ratio

        Returns:
            (adjusted_payoff, warning_message) or (original, "") if normal
        """
        if payoff_ratio > max_payoff:
            msg = f"EXTREME PAYOFF: R:R {payoff_ratio:.2f} capped at {max_payoff:.2f}"
            logger.warning(msg)
            return max_payoff, msg

        return payoff_ratio, ""


class SafetyValidator:
    """
    Validates position sizes against multiple safety constraints.

    Ensures positions are safe before execution.
    """

    def __init__(self, config: EnhancedKellyConfig):
        """
        Initialize safety validator.

        Args:
            config: Enhanced Kelly configuration
        """
        self.config = config

    def validate_position_size(
        self,
        position_size: float,
        account_balance: float,
        risk_amount: float,
        stop_loss_pips: float,
        pip_value: float
    ) -> tuple[bool, list[str]]:
        """
        Validate position size against all safety constraints.

        Args:
            position_size: Calculated position size in lots
            account_balance: Current account balance
            risk_amount: Dollar amount at risk
            stop_loss_pips: Stop loss distance in pips
            pip_value: Pip value per lot

        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        is_valid = True

        # Check 1: Position size limits
        if position_size < self.config.min_lot_size and position_size > 0:
            warnings.append(f"Position below minimum lot: {position_size:.3f} < {self.config.min_lot_size}")
            if not self.config.allow_zero_position:
                is_valid = False

        if position_size > self.config.max_lot_size:
            warnings.append(f"Position exceeds maximum lot: {position_size:.3f} > {self.config.max_lot_size}")
            is_valid = False

        # Check 2: Risk percentage limits
        risk_pct = risk_amount / account_balance
        if risk_pct > self.config.max_risk_pct:
            warnings.append(f"Risk exceeds maximum: {risk_pct:.2%} > {self.config.max_risk_pct:.2%}")
            is_valid = False

        # Check 3: Reasonable stop loss
        if stop_loss_pips < 5:
            warnings.append(f"Stop loss very tight: {stop_loss_pips:.1f} pips")

        if stop_loss_pips > 200:
            warnings.append(f"Stop loss very wide: {stop_loss_pips:.1f} pips")

        # Check 4: Risk amount is reasonable
        if risk_amount < 1.0:
            warnings.append(f"Risk amount very low: ${risk_amount:.2f}")

        if risk_amount > account_balance * 0.10:
            warnings.append(f"Risk amount very high: ${risk_amount:.2f} ({risk_pct:.1%} of balance)")

        return is_valid, warnings

    def validate_account_state(
        self,
        balance: float,
        equity: float,
        margin: float,
        free_margin: float
    ) -> tuple[bool, list[str]]:
        """
        Validate account state for safe trading.

        Args:
            balance: Account balance
            equity: Account equity
            margin: Used margin
            free_margin: Free margin

        Returns:
            (is_safe, list_of_warnings)
        """
        warnings = []
        is_safe = True

        # Check balance vs equity (drawdown)
        drawdown_pct = (equity - balance) / balance * 100
        if drawdown_pct < -10:
            warnings.append(f"Significant drawdown: {drawdown_pct:.1f}%")
            is_safe = False

        # Check margin level
        if margin > 0:
            margin_level = (equity / margin) * 100
            if margin_level < 150:
                warnings.append(f"Low margin level: {margin_level:.1f}%")
                is_safe = False
            elif margin_level < 200:
                warnings.append(f"Margin level getting low: {margin_level:.1f}%")

        # Check free margin
        free_margin_pct = (free_margin / equity) * 100
        if free_margin_pct < 20:
            warnings.append(f"Low free margin: {free_margin_pct:.1f}% of equity")
            is_safe = False
        elif free_margin_pct < 50:
            warnings.append(f"Free margin moderate: {free_margin_pct:.1f}% of equity")

        return is_safe, warnings
