"""
Strategy Family Monitor for Progressive Kill Switch System

Implements Tier 2 protection by monitoring strategy family performance
and quarantining entire families when multiple bots fail.

**Validates: Phase 2 - Tier 2 Strategy-Level Protection**
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Set, List, Any, TYPE_CHECKING

from src.router.bot_manifest import StrategyType, BotManifest
from src.router.alert_manager import AlertManager, AlertLevel, get_alert_manager

if TYPE_CHECKING:
    from src.router.bot_circuit_breaker import BotCircuitBreakerManager

logger = logging.getLogger(__name__)

# Comment 4: DB Session for database access
_DB_SESSION = None


def _get_db_session():
    """Get or create database session for DB access."""
    global _DB_SESSION
    if _DB_SESSION is None:
        try:
            from src.database.engine import SessionLocal
            _DB_SESSION = SessionLocal()
        except Exception as e:
            logger.warning(f"Could not load DB session: {e}")
    return _DB_SESSION


@dataclass
class FamilyState:
    """
    Tracks the state of a strategy family.

    Attributes:
        family: The strategy type (SCALPER, STRUCTURAL, SWING, HFT)
        failed_bots: Set of bot IDs that are currently failed
        total_pnl: Combined P&L for the family
        initial_capital: Starting capital allocation
        is_quarantined: Whether the family is quarantined
        quarantine_time: When the family was quarantined
        quarantine_reason: Why the family was quarantined
    """
    family: StrategyType
    failed_bots: Set[str] = field(default_factory=set)
    total_pnl: float = 0.0
    initial_capital: float = 10000.0
    is_quarantined: bool = False
    quarantine_time: Optional[datetime] = None
    quarantine_reason: Optional[str] = None

    @property
    def loss_pct(self) -> float:
        """Calculate loss percentage for the family."""
        if self.initial_capital <= 0:
            return 0.0
        return abs(self.total_pnl) / self.initial_capital if self.total_pnl < 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "family": self.family.value,
            "failed_bots": list(self.failed_bots),
            "total_pnl": self.total_pnl,
            "initial_capital": self.initial_capital,
            "loss_pct": self.loss_pct,
            "is_quarantined": self.is_quarantined,
            "quarantine_time": self.quarantine_time.isoformat() if self.quarantine_time else None,
            "quarantine_reason": self.quarantine_reason
        }


class StrategyFamilyMonitor:
    """
    Monitors strategy family performance and triggers family-level quarantines.

    Tier 2 Protection:
    - Quarantines family when MAX_FAILED_BOTS bots fail
    - Quarantines family when loss exceeds MAX_FAMILY_LOSS_PCT
    - Coordinates with Tier 1 BotCircuitBreaker for bot failure notifications

    Thresholds:
    - MAX_FAILED_BOTS = 3 (default)
    - MAX_FAMILY_LOSS_PCT = 0.20 (20%)

    Usage:
        monitor = StrategyFamilyMonitor(alert_manager)
        monitor.record_bot_failure(bot_manifest)
        allowed = monitor.is_family_allowed(StrategyType.SCALPER)
    """

    # Default thresholds (can be overridden via config)
    MAX_FAILED_BOTS = 3
    MAX_FAMILY_LOSS_PCT = 0.20

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        max_failed_bots: int = MAX_FAILED_BOTS,
        max_family_loss_pct: float = MAX_FAMILY_LOSS_PCT
    ):
        """
        Initialize StrategyFamilyMonitor.

        Args:
            alert_manager: AlertManager for raising alerts
            max_failed_bots: Maximum failed bots before quarantine
            max_family_loss_pct: Maximum family loss percentage
        """
        self.alert_manager = alert_manager or get_alert_manager()
        self.max_failed_bots = max_failed_bots
        self.max_family_loss_pct = max_family_loss_pct

        # Family states indexed by StrategyType
        self.family_states: Dict[StrategyType, FamilyState] = {
            family: FamilyState(family=family)
            for family in StrategyType
        }
        
        # Comment 4: Load states from DB
        self._load_all_from_db()

        logger.info(
            f"StrategyFamilyMonitor initialized: "
            f"max_failed_bots={max_failed_bots}, "
            f"max_loss_pct={max_family_loss_pct:.1%}"
        )
    
    def _load_all_from_db(self) -> None:
        """
        Load all family states from database.
        
        Comment 4: Read state from DB instead of in-memory only.
        """
        try:
            session = _get_db_session()
            if session is None:
                return
                
            from src.database.models import StrategyFamilyState
            db_records = session.query(StrategyFamilyState).all()
            
            for db_record in db_records:
                try:
                    family = StrategyType(db_record.family)
                    if family in self.family_states:
                        state = self.family_states[family]
                        state.failed_bots = set(db_record.failed_bots) if db_record.failed_bots else set()
                        state.total_pnl = db_record.total_pnl
                        state.initial_capital = db_record.initial_capital
                        state.is_quarantined = db_record.is_quarantined
                        state.quarantine_time = db_record.quarantine_time
                        state.quarantine_reason = db_record.quarantine_reason
                except ValueError:
                    pass  # Unknown strategy type
        except Exception as e:
            logger.debug(f"Could not load family states from DB: {e}")
    
    def _save_to_db(self, family: StrategyType) -> None:
        """
        Save family state to database.
        
        Comment 4: Write state to DB for persistence.
        """
        state = self.family_states.get(family)
        if state is None:
            return
            
        try:
            session = _get_db_session()
            if session is None:
                return
                
            from src.database.models import StrategyFamilyState
            db_record = session.query(StrategyFamilyState).filter_by(
                family=family.value
            ).first()
            
            if db_record:
                db_record.failed_bots = list(state.failed_bots)
                db_record.total_pnl = state.total_pnl
                db_record.initial_capital = state.initial_capital
                db_record.is_quarantined = state.is_quarantined
                db_record.quarantine_time = state.quarantine_time
                db_record.quarantine_reason = state.quarantine_reason
            else:
                db_record = StrategyFamilyState(
                    family=family.value,
                    failed_bots=list(state.failed_bots),
                    total_pnl=state.total_pnl,
                    initial_capital=state.initial_capital,
                    is_quarantined=state.is_quarantined,
                    quarantine_time=state.quarantine_time,
                    quarantine_reason=state.quarantine_reason
                )
                session.add(db_record)
            
            session.commit()
        except Exception as e:
            logger.warning(f"Could not save family state to DB: {e}")

    def record_bot_failure(self, bot_manifest: BotManifest, reason: str = "") -> bool:
        """
        Record a bot failure for its strategy family.

        Args:
            bot_manifest: The failed bot's manifest
            reason: Reason for the bot failure

        Returns:
            True if family was quarantined, False otherwise
        """
        family = bot_manifest.strategy_type
        state = self.family_states[family]

        # Skip if already quarantined
        if state.is_quarantined:
            logger.debug(f"Bot {bot_manifest.bot_id} failed but family {family.value} already quarantined")
            return True

        # Add to failed bots
        state.failed_bots.add(bot_manifest.bot_id)

        # Calculate threshold percentage
        threshold_pct = (len(state.failed_bots) / self.max_failed_bots) * 100

        logger.warning(
            f"Bot failure recorded: {bot_manifest.bot_id} in family {family.value} "
            f"({len(state.failed_bots)}/{self.max_failed_bots} failed)"
        )

        # Check if we should quarantine
        if len(state.failed_bots) >= self.max_failed_bots:
            self._quarantine_family(
                family,
                state,
                reason=f"{len(state.failed_bots)} bots failed in family"
            )
            return True

        # Raise alert based on threshold
        level = self.alert_manager.calculate_level(threshold_pct)
        if level in [AlertLevel.YELLOW, AlertLevel.ORANGE]:
            self.alert_manager.raise_alert(
                tier=2,
                message=f"{len(state.failed_bots)}/{self.max_failed_bots} bots failed in {family.value} family",
                threshold_pct=threshold_pct,
                source="strategy",
                metadata={
                    "family": family.value,
                    "failed_bots": list(state.failed_bots),
                    "last_failure_reason": reason
                }
            )

        return False

    def record_family_pnl(
        self,
        family: StrategyType,
        pnl: float,
        initial_capital: Optional[float] = None
    ) -> bool:
        """
        Record P&L for a strategy family.

        Args:
            family: Strategy type
            pnl: Profit/loss amount
            initial_capital: Optional update to initial capital

        Returns:
            True if family was quarantined, False otherwise
        """
        state = self.family_states[family]

        # Update capital if provided
        if initial_capital is not None:
            state.initial_capital = initial_capital

        # Skip if already quarantined
        if state.is_quarantined:
            return True

        # Update P&L
        state.total_pnl += pnl

        # Check loss threshold
        loss_pct = state.loss_pct
        if loss_pct >= self.max_family_loss_pct:
            threshold_pct = 100.0
            self._quarantine_family(
                family,
                state,
                reason=f"Family loss {loss_pct:.1%} exceeds limit {self.max_family_loss_pct:.1%}"
            )
            return True

        # Calculate threshold percentage
        threshold_pct = (loss_pct / self.max_family_loss_pct) * 100

        # Raise alert if significant loss
        if threshold_pct >= 50:  # YELLOW or higher
            self.alert_manager.raise_alert(
                tier=2,
                message=f"{family.value} family loss at {loss_pct:.1%}",
                threshold_pct=threshold_pct,
                source="strategy",
                metadata={
                    "family": family.value,
                    "total_pnl": state.total_pnl,
                    "loss_pct": loss_pct
                }
            )

        return False

    def _quarantine_family(
        self,
        family: StrategyType,
        state: FamilyState,
        reason: str
    ) -> None:
        """Quarantine a strategy family."""
        state.is_quarantined = True
        state.quarantine_time = datetime.now(timezone.utc)
        state.quarantine_reason = reason

        # Raise BLACK/RED alert
        threshold_pct = 100.0
        self.alert_manager.raise_alert(
            tier=2,
            message=f"Family {family.value} QUARANTINED: {reason}",
            threshold_pct=threshold_pct,
            source="strategy",
            metadata={
                "family": family.value,
                "reason": reason,
                "failed_bots": list(state.failed_bots),
                "total_pnl": state.total_pnl
            }
        )

        # Comment 4: Save to DB after quarantine
        self._save_to_db(family)

        logger.critical(
            f"🚨 Strategy family QUARANTINED: {family.value} - {reason}"
        )

    def is_family_allowed(self, family: StrategyType) -> bool:
        """
        Check if a strategy family is allowed to trade.

        Args:
            family: Strategy type to check

        Returns:
            True if family can trade, False if quarantined
        """
        state = self.family_states.get(family)
        if state is None:
            return True
        return not state.is_quarantined

    def reactivate_family(self, family: StrategyType) -> bool:
        """
        Reactivate a quarantined strategy family.

        Args:
            family: Strategy type to reactivate

        Returns:
            True if reactivated, False if not quarantined
        """
        state = self.family_states.get(family)
        if state is None or not state.is_quarantined:
            return False

        state.is_quarantined = False
        state.quarantine_time = None
        state.quarantine_reason = None
        state.failed_bots.clear()
        state.total_pnl = 0.0

        # Clear related alerts
        self.alert_manager.clear_alerts_by_source("strategy")

        logger.info(f"Strategy family reactivated: {family.value}")
        return True

    def clear_bot_from_family(self, bot_id: str, family: StrategyType) -> None:
        """Clear a specific bot from family's failed list."""
        state = self.family_states.get(family)
        if state and bot_id in state.failed_bots:
            state.failed_bots.discard(bot_id)
            logger.info(f"Bot {bot_id} cleared from {family.value} family failures")

    def get_family_state(self, family: StrategyType) -> Optional[Dict[str, Any]]:
        """Get state for a specific family."""
        state = self.family_states.get(family)
        return state.to_dict() if state else None

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all family states."""
        return {
            family.value: state.to_dict()
            for family, state in self.family_states.items()
        }

    def get_quarantined_families(self) -> List[str]:
        """Get list of quarantined family names."""
        return [
            family.value
            for family, state in self.family_states.items()
            if state.is_quarantined
        ]

    def reset_family_pnl(self, family: StrategyType) -> None:
        """Reset P&L tracking for a family (e.g., at start of new period)."""
        state = self.family_states.get(family)
        if state:
            state.total_pnl = 0.0
            logger.info(f"P&L reset for family: {family.value}")


# Global singleton instance
_global_strategy_monitor: Optional[StrategyFamilyMonitor] = None


def get_strategy_monitor() -> StrategyFamilyMonitor:
    """Get or create the global StrategyFamilyMonitor instance."""
    global _global_strategy_monitor
    if _global_strategy_monitor is None:
        _global_strategy_monitor = StrategyFamilyMonitor()
    return _global_strategy_monitor


def reset_strategy_monitor() -> None:
    """Reset the global StrategyFamilyMonitor instance (for testing)."""
    global _global_strategy_monitor
    _global_strategy_monitor = None
