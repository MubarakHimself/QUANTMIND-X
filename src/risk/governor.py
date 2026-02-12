"""
Risk Governor Orchestrator

Main entry point for the Enhanced Kelly Position Sizing system that coordinates
all risk management components including physics sensors, portfolio scaling,
and position sizing calculations.

This orchestrator:
1. Integrates physics-based market state analysis
2. Applies prop firm constraints and presets
3. Manages caching for performance
4. Provides unified position sizing API
5. Handles error cases and logging
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any

from .models.position_sizing_result import PositionSizingResult
from .models.sizing_recommendation import SizingRecommendation
from .physics.chaos_sensor import ChaosSensor
from .physics.correlation_sensor import CorrelationSensor
from .physics.ising_sensor import IsingRegimeSensor
from .config import (
    PROP_FIRM_PRESETS,
    get_preset,
    MAX_RISK_PCT,
    PHYSICS_CACHE_TTL,
    ACCOUNT_CACHE_TTL,
    FTPreset,
    The5ersPreset,
    FundingPipsPreset
)
from .portfolio_kelly import PortfolioKellyScaler

logger = logging.getLogger(__name__)


class RiskGovernor:
    """
    Risk Governor Orchestrator - Main entry point for Enhanced Kelly Position Sizing.

    Coordinates all risk management components:
    - Physics-based market state analysis (chaos, correlation, phase transitions)
    - Prop firm constraint enforcement
    - Portfolio risk scaling
    - Caching for performance optimization
    - Position sizing calculations with error handling

    Example:
        >>> governor = RiskGovernor()
        >>> result = governor.calculate_position_size(
        ...     account_balance=10000.0,
        ...     win_rate=0.55,
        ...     avg_win=400.0,
        ...     avg_loss=200.0,
        ...     current_atr=0.0012,
        ...     average_atr=0.0010,
        ...     stop_loss_pips=20.0,
        ...     pip_value=10.0
        ... )
        >>> print(f"Position size: {result.lot_size} lots")
    """

    def __init__(self, prop_firm_preset: Optional[str] = None):
        """
        Initialize Risk Governor with optional prop firm preset.

        Args:
            prop_firm_preset: Name of prop firm preset (e.g., "ftmo", "the5ers")
        """
        self._cache = {
            'physics': {},  # Physics sensor cache
            'account': {}   # Account info cache
        }
        self._cache_timestamps = {
            'physics': {},
            'account': {}
        }

        # Initialize physics sensors
        self.chaos_sensor = ChaosSensor()
        self.correlation_sensor = CorrelationSensor()
        self.ising_sensor = IsingRegimeSensor()

        # Initialize portfolio scaler
        self.portfolio_scaler = PortfolioKellyScaler()

        # Set prop firm preset if provided
        self.prop_firm_preset = None
        if prop_firm_preset:
            self.set_prop_firm_preset(prop_firm_preset)

        logger.info(
            "RiskGovernor initialized",
            extra={
                'prop_firm_preset': prop_firm_preset,
                'max_risk_pct': self.get_max_risk_pct()
            }
        )

    def calculate_position_size(
        self,
        account_balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_atr: float,
        average_atr: float,
        stop_loss_pips: float,
        pip_value: float = 10.0,
        bot_id: Optional[str] = None,
        portfolio_risk: Optional[Dict[str, float]] = None
    ) -> PositionSizingResult:
        """
        Calculate optimal position size using Enhanced Kelly with all safety layers.

        This method orchestrates the entire position sizing process:
        1. Validate inputs
        2. Calculate base Kelly fraction
        3. Apply physics-based adjustments
        4. Apply prop firm constraints
        5. Apply portfolio scaling
        6. Calculate final position size
        7. Return comprehensive result with audit trail

        Args:
            account_balance: Current account balance in base currency
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade profit in currency
            avg_loss: Average losing trade loss in currency (positive)
            current_atr: Current ATR value in price units
            average_atr: 20-period average ATR in price units
            stop_loss_pips: Stop loss distance in pips/points
            pip_value: Value per pip for 1 standard lot (default $10)
            bot_id: Optional bot identifier for portfolio scaling
            portfolio_risk: Optional portfolio risk data for scaling

        Returns:
            PositionSizingResult with complete calculation details

        Raises:
            ValueError: If inputs are invalid
        """
        # Step 0: Input Validation (must happen before PositionSizingResult creation)
        self._validate_inputs(
            account_balance, win_rate, avg_win, avg_loss, stop_loss_pips, average_atr
        )

        start_time = time.time()
        result = PositionSizingResult(
            account_balance=account_balance,
            risk_amount=0.0,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            lot_size=0.0,
            estimated_margin=None,
            remaining_margin=None,
            calculation_steps=[]
        )

        try:
            result.add_step("Input validation passed")

            # Step 2: Calculate base Kelly fraction
            base_kelly_f = self._calculate_base_kelly(win_rate, avg_win, avg_loss)
            result.add_step(f"Base Kelly fraction: {base_kelly_f:.4f} ({base_kelly_f*100:.2f}%)")

            # Step 3: Apply physics-based adjustments
            physics_multiplier = self._apply_physics_adjustments()
            result.add_step(f"Physics multiplier: {physics_multiplier:.2f}")

            # Step 4: Apply prop firm constraints
            prop_firm_multiplier = self._apply_prop_firm_constraints(base_kelly_f)
            result.add_step(f"Prop firm multiplier: {prop_firm_multiplier:.2f}")

            # Step 5: Apply portfolio scaling
            portfolio_multiplier = self._apply_portfolio_scaling(
                base_kelly_f, bot_id, portfolio_risk
            )
            result.add_step(f"Portfolio multiplier: {portfolio_multiplier:.2f}")

            # Step 6: Calculate final Kelly fraction
            final_kelly_f = base_kelly_f * physics_multiplier * prop_firm_multiplier * portfolio_multiplier
            result.add_step(f"Final Kelly fraction: {final_kelly_f:.4f} ({final_kelly_f*100:.2f}%)")

            # Step 7: Calculate risk amount and position size
            risk_amount = account_balance * final_kelly_f
            lot_size = risk_amount / (stop_loss_pips * pip_value)

            # Step 8: Round to broker precision
            lot_size = self._round_to_lot_size(lot_size)

            result.risk_amount = risk_amount
            result.lot_size = lot_size

            # Add final calculation steps
            result.add_step(f"Risk amount: ${risk_amount:,.2f}")
            result.add_step(f"Raw position size: {lot_size:.4f} lots")
            result.add_step(f"Rounded position: {lot_size:.2f} lots")

            # Add performance metrics
            calculation_time = time.time() - start_time
            result.add_step(f"Calculation completed in {calculation_time:.4f} seconds")

            logger.info(
                "Position sizing calculation complete",
                extra={
                    'account_balance': account_balance,
                    'final_kelly_f': final_kelly_f,
                    'risk_amount': risk_amount,
                    'lot_size': lot_size,
                    'calculation_time': calculation_time
                }
            )

            return result

        except Exception as e:
            error_msg = f"Position sizing failed: {str(e)}"
            result.add_step(error_msg)
            logger.error(error_msg, exc_info=True)
            raise

    def _validate_inputs(
        self,
        account_balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        stop_loss_pips: float,
        average_atr: float
    ) -> None:
        """Validate all input parameters."""
        if account_balance <= 0:
            raise ValueError(f"account_balance must be positive, got {account_balance}")
        if not 0 <= win_rate <= 1:
            raise ValueError(f"win_rate must be between 0 and 1, got {win_rate}")
        if avg_win <= 0:
            raise ValueError(f"avg_win must be positive, got {avg_win}")
        if avg_loss <= 0:
            raise ValueError(f"avg_loss must be positive, got {avg_loss}")
        if stop_loss_pips <= 0:
            raise ValueError(f"stop_loss_pips must be positive, got {stop_loss_pips}")
        if average_atr <= 0:
            raise ValueError(f"average_atr must be positive, got {average_atr}")

    def _calculate_base_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate base Kelly fraction using the Kelly formula."""
        risk_reward_ratio = avg_win / avg_loss

        # Kelly formula: f = ((B + 1) × P - 1) / B
        base_kelly_f = ((risk_reward_ratio + 1) * win_rate - 1) / risk_reward_ratio

        # Handle negative expectancy
        if base_kelly_f <= 0:
            logger.warning("Negative expectancy detected, using fallback risk")
            return MAX_RISK_PCT  # Use max risk as fallback

        return base_kelly_f

    def _apply_physics_adjustments(self) -> float:
        """Apply physics-based adjustments to position sizing."""
        # Check cache first
        cache_key = "physics_adjustment"
        if self._is_cache_valid('physics', cache_key):
            return self._cache['physics'][cache_key]

        # Get physics sensor readings
        chaos_reading = self.chaos_sensor.get_reading()
        correlation_reading = self.correlation_sensor.get_reading()
        ising_reading = self.ising_sensor.get_reading()

        # Calculate physics multiplier (0.25 to 1.0)
        # Higher chaos/correlation = lower multiplier
        chaos_penalty = min(chaos_reading * 0.75, 0.75)  # Max 75% penalty
        correlation_penalty = min(correlation_reading * 0.5, 0.5)  # Max 50% penalty
        ising_penalty = min(ising_reading * 0.25, 0.25)  # Max 25% penalty

        total_penalty = chaos_penalty + correlation_penalty + ising_penalty
        physics_multiplier = max(0.25, 1.0 - total_penalty)

        # Cache the result
        self._cache['physics'][cache_key] = physics_multiplier
        self._cache_timestamps['physics'][cache_key] = time.time()

        logger.info(
            "Physics adjustment applied",
            extra={
                'chaos_reading': chaos_reading,
                'correlation_reading': correlation_reading,
                'ising_reading': ising_reading,
                'physics_multiplier': physics_multiplier
            }
        )

        return physics_multiplier

    def _apply_prop_firm_constraints(self, base_kelly_f: float) -> float:
        """Apply prop firm constraints to position sizing."""
        if not self.prop_firm_preset:
            return 1.0  # No constraints if no preset

        # Get prop firm max risk percentage
        max_risk_pct = self.get_max_risk_pct()

        # Calculate multiplier to cap at prop firm limit
        if base_kelly_f > max_risk_pct:
            return max_risk_pct / base_kelly_f
        return 1.0

    def _apply_portfolio_scaling(
        self,
        base_kelly_f: float,
        bot_id: Optional[str],
        portfolio_risk: Optional[Dict[str, float]]
    ) -> float:
        """Apply portfolio scaling to prevent over-leverage."""
        if not bot_id or not portfolio_risk:
            return 1.0  # No scaling if no bot ID or portfolio data

        # Create portfolio risk data
        bot_kelly_factors = {bot_id: base_kelly_f}

        # Scale positions - portfolio_risk is used as bot_correlations here
        # Convert Dict[str, float] to Dict[Tuple[str, str], float] if needed
        # For now, pass None since we don't have pairwise correlations
        scaled_factors = self.portfolio_scaler.scale_bot_positions(bot_kelly_factors, None)

        # Get scaled Kelly factor for this bot
        scaled_kelly_f = scaled_factors.get(bot_id, base_kelly_f)

        # Calculate multiplier
        return scaled_kelly_f / base_kelly_f if base_kelly_f > 0 else 1.0

    def _round_to_lot_size(self, position_size: float) -> float:
        """Round position size to broker lot step."""
        # Use constants from config
        from .config import MIN_LOT, LOT_STEP, MAX_LOT

        if position_size < MIN_LOT:
            return MIN_LOT

        if position_size > MAX_LOT:
            return MAX_LOT

        # Round down to nearest lot step
        steps = int(position_size / LOT_STEP)
        return steps * LOT_STEP

    def _is_cache_valid(self, cache_type: str, key: str) -> bool:
        """Check if cache entry is still valid."""
        if key not in self._cache[cache_type]:
            return False

        cache_time = self._cache_timestamps[cache_type][key]
        ttl = PHYSICS_CACHE_TTL if cache_type == 'physics' else ACCOUNT_CACHE_TTL

        return time.time() - cache_time < ttl

    def get_max_risk_pct(self) -> float:
        """Get maximum risk percentage based on prop firm preset."""
        if not self.prop_firm_preset:
            return MAX_RISK_PCT

        return self.prop_firm_preset.get_max_risk_pct()

    def get_allocation_scalar(self) -> float:
        """Get the Kelly allocation scalar for position sizing.
        
        Returns a float value representing the Kelly fraction to use
        for position sizing based on current risk settings.
        """
        # Calculate physics adjustment
        physics_multiplier = self._apply_physics_adjustments()
        
        # Get base Kelly (assuming default parameters)
        base_kelly = 0.5  # Half Kelly as conservative default
        
        # Apply prop firm constraints
        prop_firm_multiplier = self._apply_prop_firm_constraints(base_kelly)
        
        # Return the final scalar
        return base_kelly * physics_multiplier * prop_firm_multiplier

    def set_prop_firm_preset(self, preset_name: str) -> None:
        """Set prop firm preset by name."""
        self.prop_firm_preset = get_preset(preset_name)
        logger.info(f"Prop firm preset set to {preset_name}")

    def get_portfolio_status(
        self,
        bot_kelly_factors: Dict[str, float],
        bot_correlations: Optional[Dict[Tuple[str, str], float]] = None
    ) -> Dict[str, Any]:
        """Get portfolio risk status for monitoring."""
        status = self.portfolio_scaler.get_portfolio_status(
            bot_kelly_factors, bot_correlations
        )

        return {
            'total_raw_risk': status.total_raw_risk,
            'total_scaled_risk': status.total_scaled_risk,
            'risk_utilization': status.risk_utilization,
            'bot_count': status.bot_count,
            'status': status.status,
            'scale_factor': status.scale_factor,
            'recommendation': status.recommendation
        }

    def get_sizing_recommendation(
        self,
        account_balance: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_atr: float,
        average_atr: float,
        stop_loss_pips: float,
        pip_value: float = 10.0
    ) -> SizingRecommendation:
        """
        Get detailed sizing recommendation with constraint tracking.

        Returns a SizingRecommendation object with full adjustment history.
        """
        # Calculate base Kelly
        base_kelly_f = self._calculate_base_kelly(win_rate, avg_win, avg_loss)

        # Apply physics adjustment
        physics_multiplier = self._apply_physics_adjustments()

        # Apply prop firm constraints
        prop_firm_multiplier = self._apply_prop_firm_constraints(base_kelly_f)

        # Calculate final values
        final_kelly_f = base_kelly_f * physics_multiplier * prop_firm_multiplier
        position_size = (account_balance * final_kelly_f) / (stop_loss_pips * pip_value)

        # Create recommendation
        recommendation = SizingRecommendation(
            raw_kelly=base_kelly_f,
            physics_multiplier=physics_multiplier,
            final_risk_pct=final_kelly_f,
            position_size_lots=position_size,
            constraint_source=None
        )

        # Add adjustment descriptions
        recommendation.add_adjustment(f"Base Kelly: {base_kelly_f:.4f}")
        recommendation.add_adjustment(f"Physics adjustment: ×{physics_multiplier:.2f}")
        recommendation.add_adjustment(f"Prop firm constraint: ×{prop_firm_multiplier:.2f}")
        recommendation.add_adjustment(f"Final Kelly: {final_kelly_f:.4f}")

        return recommendation

    def to_json(self) -> str:
        """Serialize governor state to JSON for logging or monitoring."""
        return json.dumps({
            'prop_firm_preset': self.prop_firm_preset.name if self.prop_firm_preset else None,
            'max_risk_pct': self.get_max_risk_pct(),
            'physics_cache_ttl': PHYSICS_CACHE_TTL,
            'account_cache_ttl': ACCOUNT_CACHE_TTL
        }, indent=2)

    def __str__(self) -> str:
        """String representation of governor state."""
        return f"RiskGovernor(prop_firm_preset={self.prop_firm_preset})"