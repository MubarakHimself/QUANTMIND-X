"""
Enhanced Trade Logger: The Black Box

Records detailed context for every trade decision, answering "Why?"

From PDF: "Enhanced Logging (Timestamp, Chaos Lvl, Governor Value, Spread, Confidence)"

**Validates: PDF Requirements - Black Box Recorder**
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class TradeDecision(Enum):
    """Trade decision outcome."""
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    THROTTLED = "THROTTLED"
    MODIFIED = "MODIFIED"


class RejectionReason(Enum):
    """Reasons for trade rejection."""
    SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
    CHAOS_TOO_HIGH = "CHAOS_TOO_HIGH"
    GOVERNOR_THROTTLE = "GOVERNOR_THROTTLE"
    KELLY_REJECTED = "KELLY_REJECTED"
    PRESERVATION_MODE = "PRESERVATION_MODE"
    MAX_POSITIONS = "MAX_POSITIONS"
    MAX_DAILY_TRADES = "MAX_DAILY_TRADES"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PROP_FIRM_RISK = "PROP_FIRM_RISK"
    KILL_SWITCH_ACTIVE = "KILL_SWITCH_ACTIVE"


@dataclass
class MarketContext:
    """Market conditions at time of trade."""
    spread_pips: float
    atr_current: float
    atr_average: float
    volatility_ratio: float
    session: str  # "LONDON", "NY", "ASIA", "OVERLAP"
    news_impact: Optional[str] = None


@dataclass
class RiskContext:
    """Risk engine state at time of trade."""
    chaos_level: str  # "LOW", "MEDIUM", "HIGH", "EXTREME"
    governor_value: float  # 0.0 - 1.0 throttle
    current_tier: str  # "GROWTH", "SCALING", "GUARDIAN"
    account_equity: float
    current_drawdown: float
    daily_pnl: float
    open_positions: int
    daily_trade_count: int


@dataclass
class BotContext:
    """Bot state at time of trade."""
    bot_id: str
    strategy_type: str
    confidence_score: float  # 0.0 - 1.0
    signal_strength: float
    entry_reason: str


@dataclass
class TradeLogEntry:
    """
    Complete trade log entry - the "Black Box" record.
    
    Answers "Why?" for every trade decision.
    """
    # Identity
    log_id: str
    timestamp: datetime
    
    # Trade details
    symbol: str
    direction: str  # "BUY" or "SELL"
    requested_volume: float
    approved_volume: float
    risk_multiplier: float
    
    # Decision
    decision: TradeDecision
    rejection_reason: Optional[RejectionReason] = None
    
    # Context - the "Why?"
    market: Optional[MarketContext] = None
    risk: Optional[RiskContext] = None
    bot: Optional[BotContext] = None
    
    # Execution (filled after trade)
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    execution_time_ms: Optional[float] = None
    
    # Additional notes
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        data = {
            "log_id": self.log_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "requested_volume": self.requested_volume,
            "approved_volume": self.approved_volume,
            "risk_multiplier": self.risk_multiplier,
            "decision": self.decision.value,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "execution_time_ms": self.execution_time_ms,
            "notes": self.notes
        }
        
        if self.market:
            data["market"] = asdict(self.market)
        if self.risk:
            data["risk"] = asdict(self.risk)
        if self.bot:
            data["bot"] = asdict(self.bot)
        
        return data


class TradeLogger:
    """
    Black Box trade logger.
    
    Records all trade decisions with full context for post-analysis.
    """
    
    def __init__(
        self,
        log_directory: Path = None,
        max_entries_per_file: int = 1000
    ):
        self.log_directory = Path(log_directory or "data/logs/trades")
        self.log_directory.mkdir(parents=True, exist_ok=True)
        self.max_entries_per_file = max_entries_per_file
        self._current_entries: List[TradeLogEntry] = []
        self._entry_count = 0
    
    def _generate_log_id(self) -> str:
        """Generate unique log ID."""
        self._entry_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"TL_{timestamp}_{self._entry_count:05d}"
    
    def log_trade(
        self,
        symbol: str,
        direction: str,
        requested_volume: float,
        approved_volume: float,
        risk_multiplier: float,
        decision: TradeDecision,
        rejection_reason: Optional[RejectionReason] = None,
        market: Optional[MarketContext] = None,
        risk: Optional[RiskContext] = None,
        bot: Optional[BotContext] = None,
        notes: Optional[List[str]] = None
    ) -> TradeLogEntry:
        """
        Log a trade decision with full context.
        
        This is the primary logging method, called by socket_server and commander.
        """
        entry = TradeLogEntry(
            log_id=self._generate_log_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            requested_volume=requested_volume,
            approved_volume=approved_volume,
            risk_multiplier=risk_multiplier,
            decision=decision,
            rejection_reason=rejection_reason,
            market=market,
            risk=risk,
            bot=bot,
            notes=notes or []
        )
        
        self._current_entries.append(entry)
        
        # Log to console
        self._log_to_console(entry)
        
        # Persist if threshold reached
        if len(self._current_entries) >= self.max_entries_per_file:
            self._flush_to_file()
        
        return entry
    
    def _log_to_console(self, entry: TradeLogEntry) -> None:
        """Log summary to console."""
        decision_emoji = {
            TradeDecision.APPROVED: "✅",
            TradeDecision.REJECTED: "❌",
            TradeDecision.THROTTLED: "⚠️",
            TradeDecision.MODIFIED: "🔄"
        }
        
        emoji = decision_emoji.get(entry.decision, "❓")
        
        log_msg = (
            f"{emoji} [{entry.log_id}] {entry.symbol} {entry.direction} "
            f"Vol: {entry.requested_volume} → {entry.approved_volume} "
            f"Decision: {entry.decision.value}"
        )
        
        if entry.rejection_reason:
            log_msg += f" ({entry.rejection_reason.value})"
        
        if entry.risk:
            log_msg += f" | Chaos: {entry.risk.chaos_level}, Gov: {entry.risk.governor_value:.2f}"
        
        if entry.bot:
            log_msg += f" | Conf: {entry.bot.confidence_score:.2f}"
        
        logger.info(log_msg)
    
    def _flush_to_file(self) -> None:
        """Write current entries to file."""
        if not self._current_entries:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_directory / f"trades_{timestamp}.json"
        
        data = {
            "entries": [e.to_dict() for e in self._current_entries],
            "count": len(self._current_entries),
            "created_at": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Flushed {len(self._current_entries)} trade logs to {filename}")
        self._current_entries.clear()
    
    def update_execution(
        self,
        log_id: str,
        entry_price: float,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        execution_time_ms: Optional[float] = None
    ) -> None:
        """Update log entry with execution details."""
        for entry in self._current_entries:
            if entry.log_id == log_id:
                entry.entry_price = entry_price
                entry.exit_price = exit_price
                entry.pnl = pnl
                entry.execution_time_ms = execution_time_ms
                break
    
    def get_recent(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent trade logs."""
        return [e.to_dict() for e in self._current_entries[-count:]]
    
    def get_rejections(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent rejected trades."""
        rejections = [
            e for e in self._current_entries
            if e.decision == TradeDecision.REJECTED
        ]
        return [e.to_dict() for e in rejections[-count:]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get trade logging statistics."""
        approved = sum(1 for e in self._current_entries if e.decision == TradeDecision.APPROVED)
        rejected = sum(1 for e in self._current_entries if e.decision == TradeDecision.REJECTED)
        
        return {
            "total_logged": len(self._current_entries),
            "approved": approved,
            "rejected": rejected,
            "approval_rate": approved / len(self._current_entries) if self._current_entries else 0,
            "in_memory": len(self._current_entries),
            "log_directory": str(self.log_directory)
        }
    
    def log_dispatch_context(
        self,
        regime: str,
        chaos_score: float,
        mandate: Any,
        dispatches: List[Dict],
        symbol: str = "MULTI"
    ) -> None:
        """
        Log dispatch context from Strategy Router.
        
        Called by engine.py after bot auction to record why bots were dispatched.
        
        Args:
            regime: Current market regime from Sentinel
            chaos_score: Chaos level (0.0 - 1.0)
            mandate: RiskMandate from Governor
            dispatches: List of dispatched bots
            symbol: Symbol being processed
        """
        if not dispatches:
            return
        
        # Create risk context from mandate
        risk = RiskContext(
            chaos_level=self._chaos_to_level(chaos_score),
            governor_value=getattr(mandate, 'allocation_scalar', 1.0),
            current_tier=getattr(mandate, 'risk_mode', 'STANDARD'),
            account_equity=0.0,  # Would need account data
            current_drawdown=0.0,
            daily_pnl=0.0,
            open_positions=len(dispatches),
            daily_trade_count=0
        )
        
        # Log each dispatch
        for bot in dispatches:
            bot_context = BotContext(
                bot_id=bot.get('bot_id', 'UNKNOWN'),
                strategy_type=bot.get('strategy_type', 'UNKNOWN'),
                confidence_score=bot.get('score', 0.0),
                signal_strength=bot.get('win_rate', 0.0),
                entry_reason=f"Regime: {regime}, Auction Winner"
            )
            
            self.log_trade(
                symbol=symbol,
                direction="DISPATCH",
                requested_volume=0.0,
                approved_volume=0.0,
                risk_multiplier=bot.get('authorized_risk_scalar', 1.0),
                decision=TradeDecision.APPROVED,
                risk=risk,
                bot=bot_context,
                notes=[
                    f"Dispatched in {regime} regime",
                    f"Risk mode: {bot.get('risk_mode', 'STANDARD')}",
                    f"Account: {bot.get('assigned_account', 'DEFAULT')}"
                ]
            )
    
    def _chaos_to_level(self, chaos_score: float) -> str:
        """Convert chaos score to level string."""
        if chaos_score < 0.2:
            return "LOW"
        elif chaos_score < 0.4:
            return "MEDIUM"
        elif chaos_score < 0.6:
            return "HIGH"
        else:
            return "EXTREME"
    
    def flush(self) -> None:
        """Force flush to file."""
        self._flush_to_file()


# Global trade logger instance
_global_trade_logger: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    """Get or create the global trade logger instance."""
    global _global_trade_logger
    if _global_trade_logger is None:
        _global_trade_logger = TradeLogger()
    return _global_trade_logger
