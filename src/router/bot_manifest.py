"""
BotManifest: The Passport System

Each bot in the fleet has a "passport" declaring its requirements and characteristics.
This enables the Routing Matrix to automatically assign bots to appropriate accounts.

From PDF: "Update the BaseBot class to require a BotManifest property."

**Validates: PDF Requirements - Bot Tagging & Routing**

**V3: Paper->Demo->Live Promotion Workflow**
- TradingMode enum for bot lifecycle tracking
- Performance stats per mode (paper_stats, demo_stats, live_stats)
- Promotion eligibility tracking
- Capital allocation per mode
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json
import logging

from src.router.multi_timeframe_sentinel import Timeframe
from src.router.sessions import TradingSession

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Strategy classification for routing decisions."""
    SCALPER = "SCALPER"       # High-frequency, many trades
    STRUCTURAL = "STRUCTURAL" # ICT, AMD, Pattern-based
    SWING = "SWING"           # Multi-day holds
    HFT = "HFT"               # Sub-second execution


class TradeFrequency(Enum):
    """Trade frequency classification."""
    HFT = "HFT"           # >100 trades/day
    HIGH = "HIGH"         # 20-100 trades/day
    MEDIUM = "MEDIUM"     # 5-20 trades/day
    LOW = "LOW"           # <5 trades/day


class BrokerType(Enum):
    """Preferred broker type based on strategy needs."""
    RAW_ECN = "RAW_ECN"       # Low spreads, fast execution (for scalpers)
    STANDARD = "STANDARD"     # Reliability over speed (for structural)
    ANY = "ANY"               # No preference


class TradingMode(Enum):
    """
    Trading mode for bot lifecycle tracking.

    Paper->Live Promotion Workflow:
    - PAPER: Paper trading validation (simulated execution, includes demo)
    - LIVE: Live trading with real capital

    Note: PAPER mode encompasses what was previously separate DEMO mode.
    Demo trading is now considered part of paper trading validation.

    Promotion Criteria (per spec):
    - PAPER->LIVE: 30+ days, Sharpe > 1.5, Win Rate > 55%, Max DD < 10%
    """
    PAPER = "paper"
    LIVE = "live"


class AccountBook(Enum):
    """
    Account book type for the bot.

    Used to distinguish between:
    - PERSONAL: Trader's own capital
    - PROP_FIRM: Prop firm account (requires prop_firm_name)
    """
    PERSONAL = "PERSONAL"
    PROP_FIRM = "PROP_FIRM"


@dataclass
class ModePerformanceStats:
    """
    Performance statistics for a specific trading mode.
    
    Tracks metrics accumulated during paper, demo, or live trading.
    Used for promotion eligibility evaluation.
    """
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    trading_days: int = 0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "total_pnl": self.total_pnl,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "trading_days": self.trading_days,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModePerformanceStats":
        """Deserialize from dictionary."""
        return cls(
            total_trades=data.get("total_trades", 0),
            winning_trades=data.get("winning_trades", 0),
            losing_trades=data.get("losing_trades", 0),
            win_rate=data.get("win_rate", 0.0),
            total_pnl=data.get("total_pnl", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            profit_factor=data.get("profit_factor", 0.0),
            trading_days=data.get("trading_days", 0),
            start_date=datetime.fromisoformat(data["start_date"]) if data.get("start_date") else None,
            end_date=datetime.fromisoformat(data["end_date"]) if data.get("end_date") else None,
        )


@dataclass
class TimeWindow:
    """
    Defines a specific trading time window for ICT-style strategies.

    Used for bots that only trade during specific time periods
    (e.g., 9:50-10:10 AM NY for Silver Bullet).

    Args:
        start: Start time in "HH:MM" format
        end: End time in "HH:MM" format
        timezone: IANA timezone name (e.g., "America/New_York")
    """
    start: str  # Format: "HH:MM"
    end: str    # Format: "HH:MM"
    timezone: str  # IANA timezone (e.g., "America/New_York")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize TimeWindow to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "timezone": self.timezone
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TimeWindow":
        """Deserialize TimeWindow from dictionary."""
        return cls(
            start=data.get("start", "00:00"),
            end=data.get("end", "23:59"),
            timezone=data.get("timezone", "UTC")
        )


@dataclass
class PreferredConditions:
    """
    Session and time window preferences for a bot.

    Enables ICT-style filtering where bots only trade during:
    - Specific sessions (e.g., NEW_YORK only)
    - Custom time windows (e.g., 9:50-10:10 AM)
    - Volatility ranges (min/max ATR thresholds)

    Args:
        sessions: List of preferred TradingSession enums (e.g., [TradingSession.LONDON, TradingSession.NEW_YORK])
        time_windows: List of specific time windows for custom ICT strategies
        min_volatility: Minimum volatility requirement (ATR threshold)
        max_volatility: Maximum volatility threshold (ATR limit)
    """
    sessions: List[TradingSession] = field(default_factory=list)
    time_windows: List[TimeWindow] = field(default_factory=list)
    min_volatility: Optional[float] = None
    max_volatility: Optional[float] = None

    def __post_init__(self):
        """Coerce string entries in sessions to TradingSession enums."""
        coerced_sessions = []
        for s in self.sessions:
            if isinstance(s, TradingSession):
                coerced_sessions.append(s)
            elif isinstance(s, str):
                try:
                    coerced_sessions.append(TradingSession(s))
                except ValueError:
                    logger.warning(f"Invalid session '{s}' in PreferredConditions, skipping")
            else:
                logger.warning(f"Invalid session type '{type(s)}' in PreferredConditions, skipping")
        self.sessions = coerced_sessions

    def _session_to_value(self, s) -> str:
        """Convert a session (enum or string) to its string value."""
        if isinstance(s, TradingSession):
            return s.value
        elif isinstance(s, str):
            # Try to convert string to TradingSession and get value
            try:
                return TradingSession(s).value
            except ValueError:
                return s
        return str(s)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PreferredConditions to dictionary."""
        return {
            "sessions": [self._session_to_value(s) for s in self.sessions],
            "time_windows": [tw.to_dict() for tw in self.time_windows],
            "min_volatility": self.min_volatility,
            "max_volatility": self.max_volatility
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferredConditions":
        """Deserialize PreferredConditions from dictionary."""
        time_windows = []
        for tw_data in data.get("time_windows", []):
            time_windows.append(TimeWindow.from_dict(tw_data))

        # Convert strings to TradingSession enums
        sessions = []
        for session_str in data.get("sessions", []):
            try:
                sessions.append(TradingSession(session_str))
            except ValueError:
                logger.warning(f"Invalid session '{session_str}' in PreferredConditions, skipping")

        return cls(
            sessions=sessions,
            time_windows=time_windows,
            min_volatility=data.get("min_volatility"),
            max_volatility=data.get("max_volatility")
        )


# Build a lookup table for Timeframe by name (e.g., "M15" -> Timeframe.M15)
_TIMEFRAME_BY_NAME: Dict[str, Timeframe] = {tf.name: tf for tf in Timeframe}


def _parse_timeframe(data: Any, key: str, default: Timeframe = Timeframe.H1) -> Timeframe:
    """
    Parse a Timeframe enum from dictionary data with backward compatibility.
    
    Handles cases where:
    - Key exists and contains a valid Timeframe name
    - Key exists but has invalid value (returns default)
    - Key doesn't exist (returns default for backward compatibility)
    """
    if data is None:
        return default
    
    value = data.get(key) if isinstance(data, dict) else None
    if value is None:
        return default
    
    try:
        if isinstance(value, Timeframe):
            return value
        # Handle string names like "M15", "H1", etc.
        if isinstance(value, str) and value in _TIMEFRAME_BY_NAME:
            return _TIMEFRAME_BY_NAME[value]
        # Also try the enum directly (for cases where value is already the enum name)
        return Timeframe[value]
    except (ValueError, KeyError, AttributeError):
        logger.warning(f"Invalid timeframe value '{value}' for key '{key}', using default {default}")
        return default


def _parse_timeframe_list(data: Any, key: str) -> List[Timeframe]:
    """
    Parse a list of Timeframe enums from dictionary data with backward compatibility.
    
    Handles cases where:
    - Key exists and contains a list of valid Timeframe names
    - Key exists but has invalid values (skips invalid ones)
    - Key doesn't exist (returns empty list for backward compatibility)
    """
    if data is None:
        return []
    
    value = data.get(key) if isinstance(data, dict) else None
    if value is None:
        return []
    
    if not isinstance(value, list):
        return []
    
    result = []
    for item in value:
        try:
            if isinstance(item, Timeframe):
                result.append(item)
            elif isinstance(item, str) and item in _TIMEFRAME_BY_NAME:
                result.append(_TIMEFRAME_BY_NAME[item])
            else:
                result.append(Timeframe[item])
        except (ValueError, KeyError, AttributeError):
            logger.warning(f"Invalid timeframe value '{item}' in list, skipping")
            continue
    
    return result


@dataclass
class BotManifest:
    """
    The "Passport" for each trading bot.
    
    Declares the bot's requirements for automatic routing to appropriate accounts.
    
    From PDF:
    - bot_id: Unique identifier
    - strategy_type: "SCALPER", "STRUCTURAL", "SWING"
    - frequency: "HFT" (>20/day), "LOW" (<5/day)
    - min_capital_req: Margin buffer (e.g., $50)
    - preferred_broker_type: "RAW_ECN" or "STANDARD"
    - prop_firm_safe: False if violates 1-min rule or HFT ban
    """
    bot_id: str
    strategy_type: StrategyType
    frequency: TradeFrequency
    min_capital_req: float = 50.0
    preferred_broker_type: BrokerType = BrokerType.ANY
    prop_firm_safe: bool = True
    
    # Additional metadata
    name: str = ""
    description: str = ""
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    preferred_timeframe: Timeframe = Timeframe.H1
    use_multi_timeframe: bool = False
    secondary_timeframes: List[Timeframe] = field(default_factory=list)
    max_positions: int = 1
    max_daily_trades: int = 100
    
    # Runtime tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_trade_at: Optional[datetime] = None
    total_trades: int = 0
    win_rate: float = 0.0

    # Session preferences (ICT-style filtering)
    preferred_conditions: Optional[PreferredConditions] = None
    
    # Tags for lifecycle tracking (from original TRD)
    tags: List[str] = field(default_factory=list)  # @primal, @pending, @perfect, @quarantine, @dead
    
    # V3: Trading mode and promotion workflow
    trading_mode: TradingMode = TradingMode.PAPER
    capital_allocated: float = 0.0
    promotion_eligible: bool = False
    paper_stats: Optional[ModePerformanceStats] = None
    demo_stats: Optional[ModePerformanceStats] = None  # Deprecated: use paper_stats instead
    live_stats: Optional[ModePerformanceStats] = None
    mode_start_date: Optional[datetime] = None  # When current mode started

    # Account book type for routing decisions
    account_book_type: AccountBook = AccountBook.PERSONAL
    prop_firm_name: Optional[str] = None  # Required if account_book_type is PROP_FIRM
    max_drawdown_pct: Optional[float] = None  # Max drawdown percentage (e.g., 10.0 for 10%)
    
    # Source tracking for imported EAs
    source_type: Optional[str] = None  # 'imported_ea', 'native', etc.
    source_path: Optional[str] = None  # Path to source file or repo
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize manifest to dictionary."""
        result = {
            "bot_id": self.bot_id,
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type.value,
            "frequency": self.frequency.value,
            "min_capital_req": self.min_capital_req,
            "preferred_broker_type": self.preferred_broker_type.value,
            "prop_firm_safe": self.prop_firm_safe,
            "symbols": self.symbols,
            "timeframes": self.timeframes,
            "preferred_timeframe": self.preferred_timeframe.name,
            "use_multi_timeframe": self.use_multi_timeframe,
            "secondary_timeframes": [tf.name for tf in self.secondary_timeframes],
            "max_positions": self.max_positions,
            "max_daily_trades": self.max_daily_trades,
            "created_at": self.created_at.isoformat(),
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "tags": self.tags,
            # V3: Trading mode and promotion fields
            "trading_mode": self.trading_mode.value,
            "capital_allocated": self.capital_allocated,
            "promotion_eligible": self.promotion_eligible,
            "mode_start_date": self.mode_start_date.isoformat() if self.mode_start_date else None,
            # Source tracking for imported EAs
            "source_type": self.source_type,
            "source_path": self.source_path,
            # Account book type
            "account_book_type": self.account_book_type.value,
            "prop_firm_name": self.prop_firm_name,
            "max_drawdown_pct": self.max_drawdown_pct,
        }
        # Add preferred_conditions if present
        if self.preferred_conditions is not None:
            result["preferred_conditions"] = self.preferred_conditions.to_dict()
        # V3: Add mode-specific stats
        if self.paper_stats is not None:
            result["paper_stats"] = self.paper_stats.to_dict()
        if self.demo_stats is not None:
            result["demo_stats"] = self.demo_stats.to_dict()
        if self.live_stats is not None:
            result["live_stats"] = self.live_stats.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotManifest":
        """Deserialize manifest from dictionary."""
        # Deserialize preferred_conditions if present
        preferred_conditions = None
        if "preferred_conditions" in data:
            preferred_conditions = PreferredConditions.from_dict(data["preferred_conditions"])

        # Parse timeframe fields with backward compatibility
        preferred_timeframe = _parse_timeframe(data, "preferred_timeframe", Timeframe.H1)
        use_multi_timeframe = data.get("use_multi_timeframe", False) if isinstance(data, dict) else False
        secondary_timeframes = _parse_timeframe_list(data, "secondary_timeframes")

        # V3: Parse trading mode with backward compatibility (default to PAPER for new bots)
        trading_mode_str = data.get("trading_mode", "paper")
        try:
            trading_mode = TradingMode(trading_mode_str)
        except ValueError:
            logger.warning(f"Invalid trading_mode '{trading_mode_str}', defaulting to PAPER")
            trading_mode = TradingMode.PAPER

        # V3: Parse mode-specific stats
        paper_stats = None
        if "paper_stats" in data:
            paper_stats = ModePerformanceStats.from_dict(data["paper_stats"])
        
        demo_stats = None
        if "demo_stats" in data:
            demo_stats = ModePerformanceStats.from_dict(data["demo_stats"])
        
        live_stats = None
        if "live_stats" in data:
            live_stats = ModePerformanceStats.from_dict(data["live_stats"])

        return cls(
            bot_id=data["bot_id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            strategy_type=StrategyType(data["strategy_type"]),
            frequency=TradeFrequency(data["frequency"]),
            min_capital_req=data.get("min_capital_req", 50.0),
            preferred_broker_type=BrokerType(data.get("preferred_broker_type", "ANY")),
            prop_firm_safe=data.get("prop_firm_safe", True),
            symbols=data.get("symbols", []),
            timeframes=data.get("timeframes", []),
            preferred_timeframe=preferred_timeframe,
            use_multi_timeframe=use_multi_timeframe,
            secondary_timeframes=secondary_timeframes,
            max_positions=data.get("max_positions", 1),
            max_daily_trades=data.get("max_daily_trades", 100),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            last_trade_at=datetime.fromisoformat(data["last_trade_at"]) if data.get("last_trade_at") else None,
            total_trades=data.get("total_trades", 0),
            win_rate=data.get("win_rate", 0.0),
            tags=data.get("tags", []),
            preferred_conditions=preferred_conditions,
            # V3: Trading mode and promotion fields
            trading_mode=trading_mode,
            capital_allocated=data.get("capital_allocated", 0.0),
            promotion_eligible=data.get("promotion_eligible", False),
            paper_stats=paper_stats,
            demo_stats=demo_stats,
            live_stats=live_stats,
            mode_start_date=datetime.fromisoformat(data["mode_start_date"]) if data.get("mode_start_date") else None,
            # Source tracking for imported EAs
            source_type=data.get("source_type"),
            source_path=data.get("source_path"),
            # Account book type
            account_book_type=AccountBook(data.get("account_book_type", "PERSONAL")),
            prop_firm_name=data.get("prop_firm_name"),
            max_drawdown_pct=data.get("max_drawdown_pct"),
        )
    
    def is_compatible_with_account(self, account_type: str) -> bool:
        """
        Check if bot is compatible with account type.
        
        Account types from PDF:
        - "machine_gun": For HFT/Scalpers (RoboForex Prime)
        - "sniper": For Structural/ICT (Exness Raw)
        - "prop_firm": For prop firm safe bots only
        """
        if account_type == "machine_gun":
            return self.strategy_type in [StrategyType.SCALPER, StrategyType.HFT]
        elif account_type == "sniper":
            return self.strategy_type in [StrategyType.STRUCTURAL, StrategyType.SWING]
        elif account_type == "prop_firm":
            return self.prop_firm_safe
        return True
    
    def get_current_stats(self) -> Optional[ModePerformanceStats]:
        """
        Get performance stats for the current trading mode.

        Returns:
            ModePerformanceStats for current mode, or None if not available
        """
        stats_map = {
            TradingMode.PAPER: self.paper_stats,
            TradingMode.LIVE: self.live_stats,
        }
        return stats_map.get(self.trading_mode)
    
    def update_stats(self, stats: ModePerformanceStats, mode: Optional[TradingMode] = None) -> None:
        """
        Update performance stats for a specific mode.

        Args:
            stats: Performance stats to save
            mode: Target mode (defaults to current trading_mode)
        """
        target_mode = mode or self.trading_mode

        if target_mode == TradingMode.PAPER:
            self.paper_stats = stats
        elif target_mode == TradingMode.LIVE:
            self.live_stats = stats
    
    def check_promotion_eligibility(self) -> Dict[str, Any]:
        """
        Check if bot is eligible for promotion to the next trading mode.

        Promotion Criteria (per spec):
        - PAPER->LIVE: 30+ days, Sharpe > 1.5, Win Rate > 55%, Max DD < 10%

        Returns:
            Dict with eligibility status and missing criteria
        """
        current_stats = self.get_current_stats()
        
        if current_stats is None:
            return {
                "eligible": False,
                "current_mode": self.trading_mode.value,
                "reason": "No performance stats available for current mode",
                "criteria": {},
            }
        
        # Base criteria for all promotions
        criteria = {
            "min_trading_days": 30,
            "min_sharpe_ratio": 1.5,
            "min_win_rate": 0.55,
        }
        
        # Add max drawdown requirement for PAPER->LIVE
        criteria["max_drawdown"] = 10.0  # Max 10% drawdown
        
        # Check each criterion
        missing = []
        
        if current_stats.trading_days < criteria["min_trading_days"]:
            missing.append(f"trading_days: {current_stats.trading_days}/{criteria['min_trading_days']}")
        
        if current_stats.sharpe_ratio < criteria["min_sharpe_ratio"]:
            missing.append(f"sharpe_ratio: {current_stats.sharpe_ratio:.2f}/{criteria['min_sharpe_ratio']}")
        
        if current_stats.win_rate < criteria["min_win_rate"]:
            missing.append(f"win_rate: {current_stats.win_rate:.2%}/{criteria['min_win_rate']:.0%}")
        
        if "max_drawdown" in criteria:
            if current_stats.max_drawdown > criteria["max_drawdown"]:
                missing.append(f"max_drawdown: {current_stats.max_drawdown:.2%}/{criteria['max_drawdown']:.0%}")
        
        eligible = len(missing) == 0
        
        # Update promotion_eligible field
        self.promotion_eligible = eligible
        
        return {
            "eligible": eligible,
            "current_mode": self.trading_mode.value,
            "next_mode": self._get_next_mode().value if eligible else None,
            "missing_criteria": missing,
            "criteria": criteria,
            "current_stats": current_stats.to_dict(),
        }
    
    def _get_next_mode(self) -> Optional[TradingMode]:
        """Get the next trading mode in the promotion chain."""
        mode_progression = {
            TradingMode.PAPER: TradingMode.LIVE,
            TradingMode.LIVE: None,  # Already at highest level
        }
        return mode_progression.get(self.trading_mode)
    
    def promote(self) -> Dict[str, Any]:
        """
        Promote bot to the next trading mode.
        
        Returns:
            Dict with promotion result
        """
        eligibility = self.check_promotion_eligibility()
        
        if not eligibility["eligible"]:
            return {
                "success": False,
                "reason": "Bot not eligible for promotion",
                "missing_criteria": eligibility["missing_criteria"],
            }
        
        next_mode = self._get_next_mode()
        if next_mode is None:
            return {
                "success": False,
                "reason": "Bot already at highest trading mode (LIVE)",
            }
        
        # Perform promotion
        old_mode = self.trading_mode
        self.trading_mode = next_mode
        self.mode_start_date = datetime.now()
        self.promotion_eligible = False  # Reset for next promotion cycle
        
        logger.info(f"Bot {self.bot_id} promoted from {old_mode.value} to {next_mode.value}")
        
        return {
            "success": True,
            "old_mode": old_mode.value,
            "new_mode": next_mode.value,
            "promoted_at": self.mode_start_date.isoformat(),
        }
    
    def downgrade(self, reason: str = "") -> Dict[str, Any]:
        """
        Downgrade bot to a lower trading mode (e.g., due to performance issues).
        
        Args:
            reason: Reason for downgrade
            
        Returns:
            Dict with downgrade result
        """
        mode_regression = {
            TradingMode.LIVE: TradingMode.PAPER,
            TradingMode.PAPER: TradingMode.PAPER,  # Can't go lower than PAPER
        }
        
        new_mode = mode_regression.get(self.trading_mode, TradingMode.PAPER)
        
        if new_mode == self.trading_mode:
            return {
                "success": False,
                "reason": "Bot already at lowest trading mode (PAPER)",
            }
        
        old_mode = self.trading_mode
        self.trading_mode = new_mode
        self.mode_start_date = datetime.now()
        self.promotion_eligible = False
        
        logger.warning(f"Bot {self.bot_id} downgraded from {old_mode.value} to {new_mode.value}. Reason: {reason}")
        
        return {
            "success": True,
            "old_mode": old_mode.value,
            "new_mode": new_mode.value,
            "reason": reason,
            "downgraded_at": self.mode_start_date.isoformat(),
        }


class BotRegistry:
    """
    Central registry for all bot manifests.
    
    Provides lookup, registration, and persistence for bot passports.
    """
    
    def __init__(self, storage_path: str = "data/bot_registry.json"):
        self.storage_path = storage_path
        self._bots: Dict[str, BotManifest] = {}
        self._load()
    
    def _load(self) -> None:
        """Load registry from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for bot_data in data.get("bots", []):
                    manifest = BotManifest.from_dict(bot_data)
                    self._bots[manifest.bot_id] = manifest
                logger.info(f"Loaded {len(self._bots)} bot manifests")
        except FileNotFoundError:
            logger.info("No existing registry found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
    
    def _save(self) -> None:
        """Persist registry to storage."""
        try:
            data = {
                "bots": [bot.to_dict() for bot in self._bots.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
    
    def register(self, manifest: BotManifest) -> None:
        """Register a new bot manifest."""
        self._bots[manifest.bot_id] = manifest
        self._save()
        logger.info(f"Registered bot: {manifest.bot_id} ({manifest.strategy_type.value})")
    
    def unregister(self, bot_id: str) -> None:
        """Remove a bot from registry."""
        if bot_id in self._bots:
            del self._bots[bot_id]
            self._save()
            logger.info(f"Unregistered bot: {bot_id}")
    
    def get(self, bot_id: str) -> Optional[BotManifest]:
        """Get bot manifest by ID."""
        return self._bots.get(bot_id)
    
    def list_all(self) -> List[BotManifest]:
        """List all registered bots."""
        return list(self._bots.values())
    
    def list_by_type(self, strategy_type: StrategyType) -> List[BotManifest]:
        """List bots by strategy type."""
        return [b for b in self._bots.values() if b.strategy_type == strategy_type]
    
    def list_by_tag(self, tag: str) -> List[BotManifest]:
        """List bots by tag."""
        return [b for b in self._bots.values() if tag in b.tags]
    
    def list_prop_firm_safe(self) -> List[BotManifest]:
        """List only prop firm safe bots."""
        return [b for b in self._bots.values() if b.prop_firm_safe]
    
    def get_compatible_bots(self, account_type: str) -> List[BotManifest]:
        """Get bots compatible with account type."""
        return [b for b in self._bots.values() if b.is_compatible_with_account(account_type)]
    
    def list_by_trading_mode(self, mode: TradingMode) -> List[BotManifest]:
        """List bots by trading mode."""
        return [b for b in self._bots.values() if b.trading_mode == mode]
    
    def list_promotion_eligible(self) -> List[BotManifest]:
        """List bots eligible for promotion to next mode."""
        return [b for b in self._bots.values() if b.promotion_eligible]
    
    def list_paper_trading(self) -> List[BotManifest]:
        """List bots in PAPER trading mode."""
        return self.list_by_trading_mode(TradingMode.PAPER)
    
    def list_demo_trading(self) -> List[BotManifest]:
        """List bots in DEMO trading mode (deprecated: DEMO merged into PAPER)."""
        # DEMO mode has been merged into PAPER
        return self.list_by_trading_mode(TradingMode.PAPER)
    
    def list_live_trading(self) -> List[BotManifest]:
        """List bots in LIVE trading mode."""
        return self.list_by_trading_mode(TradingMode.LIVE)
    
    def update_bot_stats(self, bot_id: str, stats: ModePerformanceStats, mode: Optional[TradingMode] = None) -> bool:
        """
        Update performance stats for a bot.
        
        Args:
            bot_id: Bot identifier
            stats: Performance stats to save
            mode: Target mode (defaults to bot's current mode)
            
        Returns:
            True if update successful, False if bot not found
        """
        bot = self.get(bot_id)
        if bot is None:
            return False
        
        bot.update_stats(stats, mode)
        self._save()
        return True
    
    def check_all_promotions(self) -> List[Dict[str, Any]]:
        """
        Check promotion eligibility for all bots and update their status.
        
        Returns:
            List of promotion eligibility results for all bots
        """
        results = []
        for bot in self._bots.values():
            eligibility = bot.check_promotion_eligibility()
            results.append({
                "bot_id": bot.bot_id,
                **eligibility
            })
        self._save()
        return results
    
    def promote_bot(self, bot_id: str) -> Dict[str, Any]:
        """
        Promote a bot to the next trading mode.
        
        Args:
            bot_id: Bot identifier
            
        Returns:
            Dict with promotion result
        """
        bot = self.get(bot_id)
        if bot is None:
            return {
                "success": False,
                "reason": f"Bot {bot_id} not found",
            }
        
        result = bot.promote()
        if result["success"]:
            self._save()
        
        return result
    
    def downgrade_bot(self, bot_id: str, reason: str = "") -> Dict[str, Any]:
        """
        Downgrade a bot to a lower trading mode.
        
        Args:
            bot_id: Bot identifier
            reason: Reason for downgrade
            
        Returns:
            Dict with downgrade result
        """
        bot = self.get(bot_id)
        if bot is None:
            return {
                "success": False,
                "reason": f"Bot {bot_id} not found",
            }
        
        result = bot.downgrade(reason)
        if result["success"]:
            self._save()
        
        return result


# Example bot manifests from PDF
EXAMPLE_MANIFESTS = [
    BotManifest(
        bot_id="hft_scalper_01",
        name="HFT Flow Scalper",
        description="High-frequency scalper hunting small ticks",
        strategy_type=StrategyType.HFT,
        frequency=TradeFrequency.HFT,
        min_capital_req=200.0,
        preferred_broker_type=BrokerType.RAW_ECN,
        prop_firm_safe=False,
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["M1"],
        max_daily_trades=300,
        tags=["@primal"]
    ),
    BotManifest(
        bot_id="ict_macro_01",
        name="ICT Silver Bullet",
        description="ICT Macro trader for 9:50-10:10 AM window",
        strategy_type=StrategyType.STRUCTURAL,
        frequency=TradeFrequency.LOW,
        min_capital_req=50.0,
        preferred_broker_type=BrokerType.STANDARD,
        prop_firm_safe=True,
        symbols=["EURUSD", "GBPUSD", "NAS100"],
        timeframes=["M15", "H1"],
        max_daily_trades=5,
        tags=["@primal"],
        preferred_conditions=PreferredConditions(
            sessions=[TradingSession.NEW_YORK],
            time_windows=[TimeWindow(start="09:50", end="10:10", timezone="America/New_York")]
        )
    ),
    BotManifest(
        bot_id="london_breakout_01",
        name="London Breakout",
        description="London session breakout trader",
        strategy_type=StrategyType.STRUCTURAL,
        frequency=TradeFrequency.LOW,
        min_capital_req=50.0,
        preferred_broker_type=BrokerType.STANDARD,
        prop_firm_safe=True,
        symbols=["EURUSD", "GBPUSD"],
        timeframes=["M15", "H1"],
        max_daily_trades=3,
        tags=["@primal"],
        preferred_conditions=PreferredConditions(
            sessions=[TradingSession.LONDON]
        )
    ),
    BotManifest(
        bot_id="overlap_scalper_01",
        name="Overlap Scalper",
        description="London/NY overlap high-volatility scalper",
        strategy_type=StrategyType.SCALPER,
        frequency=TradeFrequency.HIGH,
        min_capital_req=100.0,
        preferred_broker_type=BrokerType.RAW_ECN,
        prop_firm_safe=True,
        symbols=["EURUSD", "GBPUSD", "XAUUSD"],
        timeframes=["M1", "M5"],
        max_daily_trades=50,
        tags=["@primal"],
        preferred_conditions=PreferredConditions(
            sessions=[TradingSession.OVERLAP]
        )
    ),
    BotManifest(
        bot_id="amd_hunter_01",
        name="AMD Pattern Hunter",
        description="Accumulation-Manipulation-Distribution detector",
        strategy_type=StrategyType.STRUCTURAL,
        frequency=TradeFrequency.LOW,
        min_capital_req=50.0,
        preferred_broker_type=BrokerType.STANDARD,
        prop_firm_safe=True,
        symbols=["XAUUSD", "EURUSD"],
        timeframes=["M15", "H1"],
        max_daily_trades=3,
        tags=["@primal"]
    ),
    BotManifest(
        bot_id="orb_breakout_01",
        name="ORB Breakout",
        description="Opening Range Breakout trader",
        strategy_type=StrategyType.STRUCTURAL,
        frequency=TradeFrequency.LOW,
        min_capital_req=50.0,
        preferred_broker_type=BrokerType.STANDARD,
        prop_firm_safe=True,
        symbols=["NAS100", "SPX500"],
        timeframes=["M5", "M15"],
        max_daily_trades=2,
        tags=["@pending"]
    )
]
