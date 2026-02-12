"""
Enhanced Kelly Position Sizing Algorithm

Implements the 3-layer protection system for scientifically optimal position sizing:
1. Kelly Fraction (50% of full Kelly by default)
2. Hard Risk Cap (2% max per trade)
3. Dynamic Volatility Adjustment (ATR-based scaling)

Mathematical Foundation:
    The Kelly Criterion maximizes long-term growth rate while preventing ruin:

    $$f^* = \\frac{bp - q}{b} = \\frac{p(b+1) - 1}{b}$$

    Where:
    - $f^*$ = Fraction of capital to wager
    - $p$ = Probability of winning (win rate)
    - $q$ = Probability of losing ($1 - p$)
    - $b$ = Ratio of average win to average loss (payoff ratio)

Enhanced Kelly applies:
1. Half-Kelly multiplier: $f_{enhanced} = f^* \\times 0.5$
2. Hard cap: $f_{final} = \\min(f_{enhanced}, 0.02)$ (2% max)
3. Volatility adjustment: Scale by ATR ratio

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, List

from .kelly_config import EnhancedKellyConfig

logger = logging.getLogger(__name__)


@dataclass
class KellyResult:
    """
    Result of Enhanced Kelly calculation with full breakdown.

    Attributes:
        position_size: Final position size in lots (rounded to broker precision)
        kelly_f: Final adjusted Kelly fraction (after all 3 layers)
        base_kelly_f: Raw Kelly fraction before adjustments
        risk_amount: Dollar amount at risk (account_balance × kelly_f)
        adjustments_applied: List of adjustment descriptions for audit trail
        status: Calculation status - 'calculated', 'fallback', or 'zero'

    Example:
        >>> result = KellyResult(
        ...     position_size=1.0,
        ...     kelly_f=0.02,
        ...     base_kelly_f=0.325,
        ...     risk_amount=200.0,
        ...     adjustments_applied=["Layer 1: 0.1625", "Layer 2: 0.0200"],
        ...     status='calculated'
        ... )
        >>> print(f"Trade {result.position_size} lots (${result.risk_amount} risk)")
    """
    position_size: float
    kelly_f: float
    base_kelly_f: float
    risk_amount: float
    adjustments_applied: List[str] = field(default_factory=list)
    status: str = 'calculated'

    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "position_size": self.position_size,
            "kelly_fraction": self.kelly_f,
            "base_kelly": self.base_kelly_f,
            "risk_amount": self.risk_amount,
            "adjustments": self.adjustments_applied,
            "status": self.status
        }


def enhanced_kelly_position_size(
    account_balance: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    current_atr: float,
    average_atr: float,
    stop_loss_pips: float,
    pip_value: float = 10.0,     # Default pip value for standard lot
    config: Optional[EnhancedKellyConfig] = None,
    regime_quality: float = 1.0   # Regime quality from Sentinel (1.0=STABLE, 0.0=CHAOTIC)
) -> float:
    """
    Calculate optimal position size using Enhanced Kelly with all safety layers.

    Args:
        account_balance: Current account balance in base currency
        win_rate: Historical win rate (0 to 1, e.g., 0.55 for 55%)
        avg_win: Average winning trade profit in currency
        avg_loss: Average losing trade loss in currency (positive number)
        current_atr: Current ATR value (same units as price)
        average_atr: 20-period ATR average
        stop_loss_pips: Stop loss distance in pips/points
        pip_value: Value per pip for 1 standard lot (default $10 for forex majors)
        config: EnhancedKellyConfig instance (uses defaults if None)

    Returns:
        Position size in lots (rounded to broker precision)

    Raises:
        ValueError: If inputs are invalid
    """
    calculator = EnhancedKellyCalculator(config)
    result = calculator.calculate(
        account_balance=account_balance,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        current_atr=current_atr,
        average_atr=average_atr,
        stop_loss_pips=stop_loss_pips,
        pip_value=pip_value,
        regime_quality=regime_quality
    )
    return result.position_size


class EnhancedKellyCalculator:
    """
    Enhanced Kelly Position Sizing Calculator.

    Implements the 3-layer protection system for scientifically optimal
    and safe position sizing suitable for prop firm trading.

    Layer 1 - Kelly Fraction:
        Reduces full Kelly by a safety factor (default 50%).
        Captures 70-80% of growth with 30-40% less drawdown.

    Layer 2 - Hard Risk Cap:
        Never exceeds maximum risk percentage (default 2%).
        Prevents over-leverage and protects account.

    Layer 3 - Dynamic Volatility Adjustment:
        - High volatility (ATR ratio > 1.3): Reduce position by ATR ratio
        - Low volatility (ATR ratio < 0.7): Increase position conservatively (×1.2)
        - Normal volatility: No adjustment

    Attributes:
        config: EnhancedKellyConfig instance with all parameters

    Example:
        >>> calculator = EnhancedKellyCalculator()
        >>> result = calculator.calculate(
        ...     account_balance=10000.0,
        ...     win_rate=0.55,
        ...     avg_win=400.0,
        ...     avg_loss=200.0,
        ...     current_atr=0.0012,
        ...     average_atr=0.0010,
        ...     stop_loss_pips=20.0,
        ...     pip_value=10.0
        ... )
        >>> print(f"Position: {result.position_size} lots")
    """

    def __init__(self, config: Optional[EnhancedKellyConfig] = None):
        """
        Initialize Enhanced Kelly Calculator.

        Args:
            config: Optional configuration. Uses defaults if None.

        Raises:
            ValueError: If configuration is invalid
        """
        self.config = config or EnhancedKellyConfig()
        self.config.validate()
        self._last_kelly_f: Optional[float] = None

        logger.info(
            "EnhancedKellyCalculator initialized",
            extra={
                "kelly_fraction": self.config.kelly_fraction,
                "max_risk_pct": self.config.max_risk_pct,
                "high_vol_threshold": self.config.high_vol_threshold,
                "low_vol_threshold": self.config.low_vol_threshold
            }
        )

    def calculate(
        self,
        account_balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_atr: float,
        average_atr: float,
        stop_loss_pips: float,
        pip_value: float = 10.0,
        regime_quality: float = 1.0,
        commission_per_lot: float = 0.0,
        spread_pips: float = 0.0,
        broker_id: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> KellyResult:
        """
        Calculate optimal position size using Enhanced Kelly with all safety layers.

        The calculation follows these steps:

        1. **Validate Inputs**: Ensure all parameters are valid
        2. **Broker Auto-Lookup**: If broker_id provided, auto-fetch pip_value, commission, spread
        3. **Calculate Risk-Reward Ratio**: $B = \\text{avg\\_win} / \\text{avg\\_loss}$
        4. **Fee Adjustment**: Deduct fees from avg_win, add to avg_loss, recalculate ratio
        5. **Fee Kill Switch**: Block trades if fees >= avg_win
        6. **Calculate Base Kelly**: $f^* = (B+1)P - 1) / B$
        7. **Apply Layer 1**: Multiply by Kelly fraction (default 50%)
        8. **Apply Layer 2**: Cap at maximum risk percentage (default 2%)
        9. **Apply Layer 3**: Adjust for volatility using ATR ratio
        10. **Calculate Risk Amount**: $\\text{risk} = \\text{balance} \\times f$
        11. **Calculate Position Size**: $\\text{lots} = \\text{risk} / (\\text{SL} \\times \\text{pip\\_value})$
        12. **Round to Broker Precision**: Round down to lot step

        Args:
            account_balance: Current account balance in base currency (e.g., 10000.0)
            win_rate: Historical win rate from 0 to 1 (e.g., 0.55 for 55%)
            avg_win: Average winning trade profit in currency (e.g., 400.0)
            avg_loss: Average losing trade loss in currency, positive (e.g., 200.0)
            current_atr: Current ATR value in price units (e.g., 0.0012 for 12 pips)
            average_atr: 20-period average ATR in price units (e.g., 0.0010)
            stop_loss_pips: Stop loss distance in pips/points (e.g., 20.0)
            pip_value: Value per pip for 1 standard lot (default $10 for forex majors)
            regime_quality: Regime quality from Sentinel (1.0=STABLE, 0.0=CHAOTIC)
            commission_per_lot: Commission per standard lot (default 0.0)
            spread_pips: Average spread in pips (default 0.0)
            broker_id: Broker identifier for auto-lookup (optional)
            symbol: Trading symbol for pip value lookup (required if broker_id provided)

        Returns:
            KellyResult containing:
            - position_size: Final position size in lots
            - kelly_f: Final adjusted Kelly fraction
            - base_kelly_f: Raw Kelly before adjustments
            - risk_amount: Dollar amount at risk
            - adjustments_applied: List of calculation steps
            - status: 'calculated', 'fallback', or 'zero'

        Raises:
            ValueError: If inputs are invalid (negative, out of range, etc.)

        Example:
            >>> calculator = EnhancedKellyCalculator()
            >>> result = calculator.calculate(
            ...     account_balance=10000.0,
            ...     win_rate=0.55,
            ...     avg_win=400.0,
            ...     avg_loss=200.0,
            ...     current_atr=0.0012,
            ...     average_atr=0.0010,
            ...     stop_loss_pips=20.0,
            ...     pip_value=10.0,
            ...     commission_per_lot=7.0,
            ...     spread_pips=0.1,
            ...     broker_id="icmarkets_raw",
            ...     symbol="XAUUSD"
            ... )
            >>> print(f"Position: {result.position_size} lots at {result.kelly_f:.1%} risk")
        """
        adjustments = []

        # Step 0: Broker Auto-Lookup for fee parameters
        if broker_id is not None and symbol is not None:
            try:
                from src.router.broker_registry import BrokerRegistryManager
                broker_mgr = BrokerRegistryManager()

                # Auto-fetch pip_value from broker registry (overrides passed value)
                pip_value = broker_mgr.get_pip_value(symbol, broker_id)
                adjustments.append(f"Auto-lookup: pip_value={pip_value} from {broker_id}")

                # Auto-fetch commission if not explicitly provided
                if commission_per_lot == 0.0:
                    commission_per_lot = broker_mgr.get_commission(broker_id)
                    adjustments.append(f"Auto-lookup: commission=${commission_per_lot}/lot from {broker_id}")

                # Auto-fetch spread if not explicitly provided
                if spread_pips == 0.0:
                    spread_pips = broker_mgr.get_spread(broker_id)
                    adjustments.append(f"Auto-lookup: spread={spread_pips} pips from {broker_id}")

                logger.info(
                    "Broker auto-lookup complete",
                    extra={
                        "broker_id": broker_id,
                        "symbol": symbol,
                        "pip_value": pip_value,
                        "commission_per_lot": commission_per_lot,
                        "spread_pips": spread_pips
                    }
                )
            except Exception as e:
                # Gracefully degrade when database is unavailable or any error occurs
                # Continue using passed/default pip_value, commission_per_lot, and spread_pips
                logger.warning(
                    f"Broker auto-lookup failed for broker_id={broker_id}, symbol={symbol}: {e}. "
                    f"Using provided/default values: pip_value={pip_value}, "
                    f"commission_per_lot={commission_per_lot}, spread_pips={spread_pips}"
                )

        # Step 1: Input Validation
        self._validate_inputs(win_rate, avg_win, avg_loss, stop_loss_pips, average_atr)

        # Step 2: Calculate Risk-Reward Ratio (B)
        risk_reward_ratio = avg_win / avg_loss
        adjustments.append(f"R:R ratio = {risk_reward_ratio:.2f}")

        # Step 2.5: Fee-Adjusted Returns Calculation
        # Calculate total fee per lot: commission + (spread * pip_value)
        fee_per_lot = commission_per_lot + (spread_pips * pip_value)
        adjustments.append(f"Fee per lot = ${fee_per_lot:.2f} (comm: ${commission_per_lot}, spread: {spread_pips} pips × ${pip_value})")

        # Adjust average win and loss with fees
        # Win is reduced by fees paid on entry and exit
        avg_win_net = avg_win - fee_per_lot
        # Loss is increased by fees paid (losses still incur fees)
        avg_loss_net = avg_loss + fee_per_lot
        adjustments.append(f"Net avg_win = ${avg_win_net:.2f}, Net avg_loss = ${avg_loss_net:.2f}")

        # Recalculate risk-reward ratio using net values
        if avg_loss_net > 0:
            risk_reward_ratio = avg_win_net / avg_loss_net
            adjustments.append(f"Fee-adjusted R:R = {risk_reward_ratio:.2f}")
        else:
            adjustments.append("WARNING: avg_loss_net is zero or negative, using original ratio")

        # Step 3: Calculate Base Kelly Formula
        # f = ((B + 1) × P - 1) / B
        base_kelly_f = ((risk_reward_ratio + 1) * win_rate - 1) / risk_reward_ratio
        adjustments.append(f"Base Kelly = {base_kelly_f:.4f} ({base_kelly_f*100:.2f}%)")

        # Step 3.5: Fee Kill Switch
        # If fees exceed expected win, block the trade
        if fee_per_lot >= avg_win:
            logger.warning(
                "FEE KILL SWITCH ACTIVATED",
                extra={
                    "fee_per_lot": fee_per_lot,
                    "avg_win": avg_win,
                    "symbol": symbol,
                    "broker_id": broker_id
                }
            )
            return KellyResult(
                position_size=0.0,
                kelly_f=0.0,
                base_kelly_f=base_kelly_f,
                risk_amount=0.0,
                adjustments_applied=adjustments + ["FEE KILL SWITCH - Fees exceed expected profit"],
                status='fee_blocked'
            )

        # Handle negative or zero expectancy - always return zero position
        if base_kelly_f <= 0:
            adjustments.append("NEGATIVE EXPECTANCY - No edge detected")
            logger.warning(
                "Negative expectancy detected",
                extra={
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "base_kelly": base_kelly_f
                }
            )
            return KellyResult(
                position_size=0.0,
                kelly_f=0.0,
                base_kelly_f=base_kelly_f,
                risk_amount=0.0,
                adjustments_applied=adjustments,
                status='zero'
            )

        kelly_f = base_kelly_f

        # Step 4: Apply Layer 1 - Kelly Fraction
        kelly_f = kelly_f * self.config.kelly_fraction
        adjustments.append(f"Layer 1 (×{self.config.kelly_fraction}): {kelly_f:.4f}")

        # Step 5: Apply Layer 2 - Hard Risk Cap
        if kelly_f > self.config.max_risk_pct:
            kelly_f = self.config.max_risk_pct
            adjustments.append(f"Layer 2 (capped at {self.config.max_risk_pct*100:.1f}%): {kelly_f:.4f}")
        else:
            adjustments.append(f"Layer 2 (no cap needed): {kelly_f:.4f}")

        # Step 6: Apply Layer 3 - Physics-Aware Volatility Adjustment
        # Combines Regime Quality (from Sentinel) with ATR volatility
        atr_ratio = current_atr / average_atr

        # Physics Scalar: regime_quality (1.0 = STABLE, 0.0 = CHAOTIC)
        physics_scalar = regime_quality

        # Volatility Scalar: inverse relationship (high ATR = low scalar)
        vol_scalar = average_atr / current_atr if atr_ratio > 0 else 1.0

        adjustments.append(f"Regime Quality = {physics_scalar:.2f} (from Sentinel)")
        adjustments.append(f"ATR ratio = {atr_ratio:.2f}, Vol Scalar = {vol_scalar:.2f}")

        # Combined adjustment: multiply Kelly by both scalars
        kelly_f_before = kelly_f
        kelly_f = kelly_f * physics_scalar * vol_scalar

        # Apply traditional Layer 3 adjustments for edge cases
        if atr_ratio > self.config.high_vol_threshold:
            # Additional reduction for extreme volatility
            kelly_f = kelly_f / atr_ratio
            adjustments.append(f"Layer 3 (high vol ÷{atr_ratio:.2f}): {kelly_f:.4f}")
        elif atr_ratio < self.config.low_vol_threshold:
            # Conservative boost for low volatility (capped by physics)
            kelly_f = kelly_f * self.config.low_vol_boost
            # Re-apply cap after boost
            kelly_f = min(kelly_f, self.config.max_risk_pct)
            adjustments.append(f"Layer 3 (low vol ×{self.config.low_vol_boost}): {kelly_f:.4f}")
        else:
            adjustments.append(f"Layer 3 (physics × vol): {kelly_f_before:.4f} → {kelly_f:.4f}")

        logger.info(
            "Physics-aware Layer 3 applied",
            extra={
                "regime_quality": regime_quality,
                "atr_ratio": atr_ratio,
                "physics_scalar": physics_scalar,
                "vol_scalar": vol_scalar,
                "adjusted_kelly": kelly_f
            }
        )

        # Store for portfolio scaling
        self._last_kelly_f = kelly_f

        # Step 7: Calculate Risk Amount
        risk_amount = account_balance * kelly_f
        adjustments.append(f"Risk amount = ${risk_amount:.2f}")

        # Step 8: Calculate Position Size
        # Risk per pip = stop_loss_pips * pip_value_per_lot
        risk_per_lot = stop_loss_pips * pip_value
        position_size = risk_amount / risk_per_lot
        adjustments.append(f"Raw position = {position_size:.4f} lots")

        # Step 9: Round to Broker Precision
        position_size = self._round_to_lot_size(position_size)
        adjustments.append(f"Rounded position = {position_size:.2f} lots")

        logger.info(
            "Enhanced Kelly calculation complete",
            extra={
                "position_size": position_size,
                "kelly_fraction": kelly_f,
                "risk_amount": risk_amount,
                "account_balance": account_balance
            }
        )

        return KellyResult(
            position_size=position_size,
            kelly_f=kelly_f,
            base_kelly_f=base_kelly_f,
            risk_amount=risk_amount,
            adjustments_applied=adjustments,
            status='calculated'
        )

    def _validate_inputs(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        stop_loss_pips: float,
        average_atr: float
    ) -> None:
        """Validate all input parameters."""
        if not 0 < win_rate <= 1:
            raise ValueError(f"win_rate must be between 0 (exclusive) and 1 (inclusive), got {win_rate}")
        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")
        if stop_loss_pips <= 0:
            raise ValueError(f"stop_loss_pips must be positive, got {stop_loss_pips}")
        if average_atr <= 0:
            raise ValueError(f"average_atr must be positive, got {average_atr}")

    def _round_to_lot_size(self, position_size: float) -> float:
        """Round position size to broker lot step."""
        if position_size < self.config.min_lot_size:
            return self.config.min_lot_size if not self.config.allow_zero_position else 0.0

        if position_size > self.config.max_lot_size:
            return self.config.max_lot_size

        # Round down to nearest lot step
        steps = math.floor(position_size / self.config.lot_step)
        return round(steps * self.config.lot_step, 2)

    def get_current_f(self) -> float:
        """Get the last calculated Kelly fraction (for portfolio scaling)."""
        return getattr(self, '_last_kelly_f', self.config.fallback_risk_pct)

    def set_fraction(self, scaled_f: float) -> None:
        """Set Kelly fraction from portfolio scaler."""
        self._scaled_kelly_f = scaled_f
