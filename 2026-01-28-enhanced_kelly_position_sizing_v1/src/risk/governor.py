"""
RiskGovernor - Main Entry Point for Enhanced Kelly Position Sizing.

This module implements the main orchestrator that coordinates all components
of the Enhanced Kelly Position Sizing system:

1. IsingRegimeSensor - Detects market phase transitions
2. ChaosSensor - Measures chaos via Lyapunov exponents
3. CorrelationSensor - Detects systemic risk via RMT
4. PhysicsAwareKellyEngine - Calculates Kelly with physics adjustments
5. MonteCarloValidator - Validates risk via bootstrap resampling

The RiskGovernor provides:
- Unified position sizing API
- Caching for performance (300s physics, 10s account)
- Prop firm preset support
- Comprehensive error handling
- Structured JSON logging

Reference: docs/trds/enhanced_kelly_position_sizing_v1.md
"""

import logging
import time
import json
from datetime import datetime
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field

# Import models and config from main risk package (shared across project)
import sys
from pathlib import Path

# Add main project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.risk.models.market_physics import MarketPhysics
from src.risk.models.strategy_performance import StrategyPerformance
from src.risk.models.sizing_recommendation import SizingRecommendation

# Import config - try relative first, then fall back to absolute
try:
    from .config import (
        PHYSICS_CACHE_TTL,
        ACCOUNT_CACHE_TTL,
        MAX_RISK_PCT,
        MIN_LOT,
        LOT_STEP,
        MAX_LOT,
        get_preset,
        PropFirmPreset,
    )
except ImportError:
    # When loaded directly, use absolute import with enhanced kelly path
    enhanced_kelly_src = Path(__file__).parent
    if str(enhanced_kelly_src) not in sys.path:
        sys.path.insert(0, str(enhanced_kelly_src))
    from config import (
        PHYSICS_CACHE_TTL,
        ACCOUNT_CACHE_TTL,
        MAX_RISK_PCT,
        MIN_LOT,
        LOT_STEP,
        MAX_LOT,
        get_preset,
        PropFirmPreset,
    )


# Configure structured JSON logging
class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


@dataclass
class PositionSizingResult:
    """
    Result of position sizing calculation.

    Attributes:
        account_balance: Current account balance in base currency
        risk_amount: Dollar amount at risk for this trade
        stop_loss_pips: Stop loss distance in pips/points
        pip_value: Value per pip per lot
        lot_size: Recommended position size in lots
        raw_kelly: Base Kelly fraction before adjustments
        physics_multiplier: Physics-based multiplier applied
        final_risk_pct: Final risk percentage after all adjustments
        constraint_source: Source of any constraint applied
        calculation_steps: List of calculation steps for audit trail
        calculation_time: Time taken for calculation in seconds
    """
    account_balance: float
    risk_amount: float
    stop_loss_pips: float
    pip_value: float
    lot_size: float
    raw_kelly: float = 0.0
    physics_multiplier: float = 1.0
    final_risk_pct: float = 0.0
    constraint_source: Optional[str] = None
    calculation_steps: list[str] = field(default_factory=list)
    calculation_time: float = 0.0

    def add_step(self, step_description: str) -> None:
        """Add a calculation step to the audit trail."""
        self.calculation_steps.append(step_description)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "account_balance": self.account_balance,
            "risk_amount": self.risk_amount,
            "stop_loss_pips": self.stop_loss_pips,
            "pip_value": self.pip_value,
            "lot_size": self.lot_size,
            "raw_kelly": self.raw_kelly,
            "physics_multiplier": self.physics_multiplier,
            "final_risk_pct": self.final_risk_pct,
            "constraint_source": self.constraint_source,
            "calculation_steps": self.calculation_steps,
            "calculation_time": self.calculation_time,
            "risk_percentage": (self.risk_amount / self.account_balance * 100) if self.account_balance > 0 else 0.0,
        }

    def to_json(self) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class RiskGovernor:
    """
    Risk Governor - Main entry point for Enhanced Kelly Position Sizing.

    This class orchestrates all risk management components:
    - Physics sensors (Ising, Chaos, Correlation)
    - Kelly engine with physics adjustments
    - Monte Carlo validator
    - Prop firm constraint enforcement
    - Caching for performance optimization

    Example:
        >>> governor = RiskGovernor(prop_firm_preset="ftmo")
        >>> result = governor.calculate_position_size(
        ...     account_info={"balance": 10000.0, "equity": 10500.0},
        ...     strategy_perf={"win_rate": 0.55, "avg_win": 400.0, "avg_loss": 200.0},
        ...     market_state={"lyapunov": -0.5, "ising_susceptibility": 0.5},
        ...     stop_loss_pips=20.0,
        ...     pip_value=10.0
        ... )
        >>> print(f"Position size: {result.lot_size} lots")
    """

    def __init__(
        self,
        prop_firm_preset: Optional[str] = None,
        enable_monte_carlo: bool = True,
        physics_cache_ttl: int = PHYSICS_CACHE_TTL,
        account_cache_ttl: int = ACCOUNT_CACHE_TTL,
    ):
        """
        Initialize Risk Governor with sensors and engines.

        Args:
            prop_firm_preset: Name of prop firm preset (e.g., "ftmo", "the5ers")
            enable_monte_carlo: Whether to enable Monte Carlo validation
            physics_cache_ttl: Cache TTL for physics sensor results (seconds)
            account_cache_ttl: Cache TTL for account info (seconds)

        Raises:
            ValueError: If prop_firm_preset name is invalid
        """
        self._logger = self._setup_logger()
        self._enable_monte_carlo = enable_monte_carlo

        # Cache configuration
        self._physics_cache_ttl = physics_cache_ttl
        self._account_cache_ttl = account_cache_ttl

        # Initialize caches
        self._physics_cache: Dict[str, Tuple[Any, float]] = {}
        self._account_cache: Dict[str, Tuple[Any, float]] = {}

        # Import and initialize sensors and engines
        self._init_components()

        # Set prop firm preset
        self._prop_firm_preset: Optional[PropFirmPreset] = None
        if prop_firm_preset:
            self.set_prop_firm_preset(prop_firm_preset)

        self._logger.info(
            "RiskGovernor initialized",
            extra={
                "prop_firm_preset": prop_firm_preset,
                "enable_monte_carlo": enable_monte_carlo,
                "physics_cache_ttl": physics_cache_ttl,
                "account_cache_ttl": account_cache_ttl,
                "max_risk_pct": self.get_max_risk_pct(),
            }
        )

    def _setup_logger(self) -> logging.Logger:
        """Setup structured JSON logger."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Avoid adding duplicate handlers
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            logger.addHandler(handler)

        return logger

    def _init_components(self) -> None:
        """Initialize physics sensors and Kelly engine."""
        try:
            # Import from main project source
            from src.risk.physics.chaos_sensor import ChaosSensor
            from src.risk.physics.correlation_sensor import CorrelationSensor
            from src.risk.physics.ising_sensor import IsingRegimeSensor
            from src.risk.sizing.kelly_engine import PhysicsAwareKellyEngine
            from src.risk.sizing.monte_carlo_validator import MonteCarloValidator

            # Initialize physics sensors
            self.chaos_sensor = ChaosSensor()
            self.correlation_sensor = CorrelationSensor()
            self.ising_regime_sensor = IsingRegimeSensor()

            # Initialize Kelly engine
            self.kelly_engine = PhysicsAwareKellyEngine(
                kelly_fraction=0.5,
                enable_monte_carlo=self._enable_monte_carlo,
            )

            # Initialize Monte Carlo validator
            self.mc_validator = MonteCarloValidator(runs=2000)

            self._logger.info("All components initialized successfully")

        except ImportError as e:
            self._logger.error(f"Failed to import components: {e}")
            raise RuntimeError(f"Failed to initialize RiskGovernor components: {e}")

    def calculate_position_size(
        self,
        account_info: Dict[str, float],
        strategy_perf: Dict[str, float],
        market_state: Dict[str, Any],
        stop_loss_pips: float,
        pip_value: float,
    ) -> PositionSizingResult:
        """
        Calculate optimal position size using Enhanced Kelly with all safety layers.

        Calculation flow:
        1. Validate inputs
        2. Get/calculate physics indicators (with caching)
        3. Build StrategyPerformance model
        4. Build MarketPhysics model
        5. Calculate base Kelly with physics adjustments
        6. Apply prop firm constraints
        7. Calculate position size: lots = (balance * risk_pct) / (sl_pips * pip_value)
        8. Round to broker lot precision
        9. Return PositionSizingResult with audit trail

        Args:
            account_info: Account info dict with 'balance' (and optionally 'equity')
            strategy_perf: Strategy performance dict with 'win_rate', 'avg_win', 'avg_loss'
            market_state: Market state dict with physics indicators
            stop_loss_pips: Stop loss distance in pips/points
            pip_value: Value per pip for 1 standard lot

        Returns:
            PositionSizingResult with complete calculation details

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If sensor failures occur
        """
        start_time = time.time()

        # Initialize result with defaults
        result = PositionSizingResult(
            account_balance=account_info.get("balance", 0.0),
            risk_amount=0.0,
            stop_loss_pips=stop_loss_pips,
            pip_value=pip_value,
            lot_size=0.0,
        )

        try:
            # Step 1: Input validation
            self._validate_inputs(account_info, strategy_perf, market_state, stop_loss_pips, pip_value)
            result.add_step("Input validation passed")

            # Step 2: Get account balance from cache or input
            account_balance = self._get_account_balance(account_info)
            result.account_balance = account_balance
            result.add_step(f"Account balance: ${account_balance:,.2f}")

            # Step 3: Get physics state (with caching)
            physics = self._get_physics_state(market_state)
            result.add_step(f"Market physics: risk_level={physics.risk_level().value}, " +
                          f"lyapunov={physics.lyapunov_exponent:.2f}")

            # Step 4: Build strategy performance model
            perf = self._build_strategy_performance(strategy_perf)
            result.add_step(f"Strategy: win_rate={perf.win_rate:.2%}, " +
                          f"payoff_ratio={perf.payoff_ratio:.2f}, expectancy=${perf.expectancy():.2f}")

            # Step 5: Calculate Kelly with physics adjustments
            sizing_rec = self.kelly_engine.calculate_position_size(
                perf=perf,
                physics=physics,
                validator=self.mc_validator if self._enable_monte_carlo else None,
            )

            result.raw_kelly = sizing_rec.raw_kelly
            result.physics_multiplier = sizing_rec.physics_multiplier
            result.final_risk_pct = sizing_rec.final_risk_pct
            result.constraint_source = sizing_rec.constraint_source

            result.add_step(f"Base Kelly: {sizing_rec.raw_kelly:.4f} ({sizing_rec.raw_kelly*100:.2f}%)")
            result.add_step(f"Physics multiplier: {sizing_rec.physics_multiplier:.2f}")
            result.add_step(f"Final Kelly: {sizing_rec.final_risk_pct:.4f} ({sizing_rec.final_risk_pct*100:.2f}%)")

            # Step 6: Apply prop firm constraints
            final_risk_pct = self._apply_prop_firm_constraints(sizing_rec.final_risk_pct)
            result.final_risk_pct = final_risk_pct

            # Step 7: Calculate position size
            risk_amount = account_balance * final_risk_pct
            lot_size = risk_amount / (stop_loss_pips * pip_value)

            # Step 8: Round to broker precision
            lot_size = self._round_to_lot_size(lot_size)

            result.risk_amount = risk_amount
            result.lot_size = lot_size

            result.add_step(f"Risk amount: ${risk_amount:,.2f} ({final_risk_pct*100:.2f}%)")
            result.add_step(f"Position size: {lot_size:.2f} lots")

            # Step 9: Update constraint source if prop firm constraint was applied
            if final_risk_pct < sizing_rec.final_risk_pct:
                result.constraint_source = "prop_firm_limit"
                result.add_step(f"Prop firm constraint applied: capped at {final_risk_pct*100:.2f}%")

            result.calculation_time = time.time() - start_time
            result.add_step(f"Calculation completed in {result.calculation_time:.4f}s")

            self._logger.info(
                "Position sizing calculation complete",
                extra={
                    "account_balance": account_balance,
                    "final_risk_pct": final_risk_pct,
                    "risk_amount": risk_amount,
                    "lot_size": lot_size,
                    "constraint_source": result.constraint_source,
                    "calculation_time": result.calculation_time,
                }
            )

            return result

        except Exception as e:
            result.calculation_time = time.time() - start_time
            result.add_step(f"Calculation failed: {str(e)}")

            self._logger.error(
                "Position sizing calculation failed",
                extra={"error": str(e), "error_type": type(e).__name__},
                exc_info=True,
            )

            raise RuntimeError(f"Position sizing calculation failed: {e}") from e

    def _validate_inputs(
        self,
        account_info: Dict[str, float],
        strategy_perf: Dict[str, float],
        market_state: Dict[str, Any],
        stop_loss_pips: float,
        pip_value: float,
    ) -> None:
        """Validate all input parameters."""
        # Validate account info
        if "balance" not in account_info:
            raise ValueError("account_info must contain 'balance'")
        if account_info["balance"] <= 0:
            raise ValueError(f"account_balance must be positive, got {account_info['balance']}")

        # Validate strategy performance
        required_perf_keys = {"win_rate", "avg_win", "avg_loss"}
        missing_perf_keys = required_perf_keys - strategy_perf.keys()
        if missing_perf_keys:
            raise ValueError(f"strategy_perf missing required keys: {missing_perf_keys}")

        if not 0 <= strategy_perf["win_rate"] <= 1:
            raise ValueError(f"win_rate must be between 0 and 1, got {strategy_perf['win_rate']}")
        if strategy_perf["avg_win"] <= 0:
            raise ValueError(f"avg_win must be positive, got {strategy_perf['avg_win']}")
        if strategy_perf["avg_loss"] <= 0:
            raise ValueError(f"avg_loss must be positive, got {strategy_perf['avg_loss']}")

        # Validate trade parameters
        if stop_loss_pips <= 0:
            raise ValueError(f"stop_loss_pips must be positive, got {stop_loss_pips}")
        if pip_value <= 0:
            raise ValueError(f"pip_value must be positive, got {pip_value}")

    def _get_account_balance(self, account_info: Dict[str, float]) -> float:
        """Get account balance from cache or input."""
        balance = account_info.get("balance", 0.0)

        # Cache account info (use equity if available, otherwise balance)
        cache_key = "account_balance"
        current_time = time.time()

        if cache_key in self._account_cache:
            cached_value, cached_time = self._account_cache[cache_key]
            if current_time - cached_time < self._account_cache_ttl:
                return cached_value

        # Update cache
        self._account_cache[cache_key] = (balance, current_time)
        return balance

    def _get_physics_state(self, market_state: Dict[str, Any]) -> MarketPhysics:
        """Get or calculate market physics state with caching."""
        cache_key = "physics_state"
        current_time = time.time()

        # Check cache
        if cache_key in self._physics_cache:
            cached_physics, cached_time = self._physics_cache[cache_key]
            if current_time - cached_time < self._physics_cache_ttl:
                self._logger.debug("Using cached physics state")
                return cached_physics

        # Build MarketPhysics from market_state
        try:
            physics = MarketPhysics(
                lyapunov_exponent=market_state.get("lyapunov", 0.0),
                ising_susceptibility=market_state.get("ising_susceptibility", 0.5),
                ising_magnetization=market_state.get("ising_magnetization", 0.0),
                rmt_max_eigenvalue=market_state.get("rmt_max_eigenvalue"),
                rmt_noise_threshold=market_state.get("rmt_noise_threshold"),
            )

            # Cache the result
            self._physics_cache[cache_key] = (physics, current_time)

            return physics

        except Exception as e:
            self._logger.error(f"Failed to build MarketPhysics: {e}")
            # Return default safe physics state
            return MarketPhysics(
                lyapunov_exponent=-0.5,
                ising_susceptibility=0.5,
                ising_magnetization=0.0,
                rmt_max_eigenvalue=None,
                rmt_noise_threshold=None,
            )

    def _build_strategy_performance(self, strategy_perf: Dict[str, float]) -> StrategyPerformance:
        """Build StrategyPerformance model from dict."""
        return StrategyPerformance(
            win_rate=strategy_perf["win_rate"],
            avg_win=strategy_perf["avg_win"],
            avg_loss=strategy_perf["avg_loss"],
            total_trades=strategy_perf.get("total_trades", 100),
            k_fraction=strategy_perf.get("k_fraction", 0.5),
        )

    def _apply_prop_firm_constraints(self, risk_pct: float) -> float:
        """Apply prop firm constraints to risk percentage."""
        if self._prop_firm_preset is None:
            return risk_pct

        max_risk = self._prop_firm_preset.get_max_risk_pct()
        return min(risk_pct, max_risk)

    def _round_to_lot_size(self, position_size: float) -> float:
        """Round position size to broker lot step."""
        if position_size < MIN_LOT:
            return MIN_LOT

        if position_size > MAX_LOT:
            return MAX_LOT

        # Round down to nearest lot step
        steps = int(position_size / LOT_STEP)
        return max(MIN_LOT, steps * LOT_STEP)

    def get_max_risk_pct(self) -> float:
        """Get maximum risk percentage based on prop firm preset."""
        if self._prop_firm_preset is None:
            return MAX_RISK_PCT
        return self._prop_firm_preset.get_max_risk_pct()

    def set_prop_firm_preset(self, preset_name: str) -> None:
        """Set prop firm preset by name."""
        try:
            self._prop_firm_preset = get_preset(preset_name)
            self._logger.info(f"Prop firm preset set to {preset_name}")
        except ValueError as e:
            self._logger.error(f"Failed to set prop firm preset: {e}")
            raise

    def clear_caches(self) -> None:
        """Clear all caches (physics and account)."""
        self._physics_cache.clear()
        self._account_cache.clear()
        self._logger.info("All caches cleared")

    def to_json(self) -> str:
        """Serialize governor state to JSON."""
        return json.dumps({
            "prop_firm_preset": self._prop_firm_preset.name if self._prop_firm_preset else None,
            "max_risk_pct": self.get_max_risk_pct(),
            "enable_monte_carlo": self._enable_monte_carlo,
            "physics_cache_ttl": self._physics_cache_ttl,
            "account_cache_ttl": self._account_cache_ttl,
            "physics_cache_size": len(self._physics_cache),
            "account_cache_size": len(self._account_cache),
        }, indent=2)

    def __str__(self) -> str:
        """String representation of governor state."""
        preset_name = self._prop_firm_preset.name if self._prop_firm_preset else "none"
        return f"RiskGovernor(prop_firm={preset_name}, max_risk={self.get_max_risk_pct():.2%})"
