"""
BotManifest: The Passport System

Each bot in the fleet has a "passport" declaring its requirements and characteristics.
This enables the Routing Matrix to automatically assign bots to appropriate accounts.

From PDF: "Update the BaseBot class to require a BotManifest property."

**Validates: PDF Requirements - Bot Tagging & Routing**
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import json
import logging

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
        sessions: List of preferred session names (e.g., ["LONDON", "NEW_YORK"])
        time_windows: List of specific time windows for custom ICT strategies
        min_volatility: Minimum volatility requirement (ATR threshold)
        max_volatility: Maximum volatility threshold (ATR limit)
    """
    sessions: List[str] = field(default_factory=list)
    time_windows: List[TimeWindow] = field(default_factory=list)
    min_volatility: Optional[float] = None
    max_volatility: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize PreferredConditions to dictionary."""
        return {
            "sessions": self.sessions,
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

        return cls(
            sessions=data.get("sessions", []),
            time_windows=time_windows,
            min_volatility=data.get("min_volatility"),
            max_volatility=data.get("max_volatility")
        )


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
            "max_positions": self.max_positions,
            "max_daily_trades": self.max_daily_trades,
            "created_at": self.created_at.isoformat(),
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "tags": self.tags
        }
        # Add preferred_conditions if present
        if self.preferred_conditions is not None:
            result["preferred_conditions"] = self.preferred_conditions.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotManifest":
        """Deserialize manifest from dictionary."""
        # Deserialize preferred_conditions if present
        preferred_conditions = None
        if "preferred_conditions" in data:
            preferred_conditions = PreferredConditions.from_dict(data["preferred_conditions"])

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
            max_positions=data.get("max_positions", 1),
            max_daily_trades=data.get("max_daily_trades", 100),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            last_trade_at=datetime.fromisoformat(data["last_trade_at"]) if data.get("last_trade_at") else None,
            total_trades=data.get("total_trades", 0),
            win_rate=data.get("win_rate", 0.0),
            tags=data.get("tags", []),
            preferred_conditions=preferred_conditions
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
            sessions=["NEW_YORK"],
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
            sessions=["LONDON"]
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
            sessions=["OVERLAP"]
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
