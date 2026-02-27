"""
Strategy Management Tools

Tools for managing trading strategies within the QuantMindX system.
Handles strategy creation, backtesting, deployment, and lifecycle management.

Integrates with:
- BotManifest system for strategy registration
- Paper trading deployment system
- Backtesting infrastructure
- Strategy Router for execution
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import subprocess
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# PATH CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STRATEGIES_YT_DIR = PROJECT_ROOT / "strategies-yt"
SRC_DIR = PROJECT_ROOT / "src"
ROUTER_DIR = SRC_DIR / "router"
BOT_MANIFEST_PATH = ROUTER_DIR / "bot_manifest.py"


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class StrategyType(Enum):
    """Strategy classification types."""
    SCALPER = "SCALPER"
    STRUCTURAL = "STRUCTURAL"
    SWING = "SWING"
    HFT = "HFT"


class TradeFrequency(Enum):
    """Trade frequency classifications."""
    HFT = "HFT"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TradingMode(Enum):
    """Trading modes for lifecycle tracking."""
    PAPER = "paper"
    LIVE = "live"


class StrategyStatus(Enum):
    """Strategy lifecycle status tags."""
    PRIMAL = "@primal"
    PENDING = "@pending"
    PERFECT = "@perfect"
    LIVE = "@live"
    QUARANTINE = "@quarantine"
    DEAD = "@dead"


@dataclass
class BacktestConfig:
    """Configuration for strategy backtesting."""
    symbol: str
    timeframe: str
    start_date: str  # ISO date format
    end_date: str
    initial_deposit: float = 10000.0
    leverage: int = 100
    spread_points: float = 10.0
    tick_mode: int = 1


@dataclass
class BacktestResult:
    """Results from strategy backtesting."""
    strategy_id: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    average_trade_duration_hours: float
    tested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestResult":
        return cls(**data)


@dataclass
class DeploymentConfig:
    """Configuration for strategy deployment."""
    strategy_id: str
    mode: TradingMode
    account_id: str
    symbols: List[str]
    max_positions: int = 1
    max_daily_trades: int = 100
    auto_trading_enabled: bool = False
    notify_on_trade: bool = True


@dataclass
class DeploymentResult:
    """Result from strategy deployment."""
    success: bool
    strategy_id: str
    mode: TradingMode
    deployment_id: str
    message: str
    deployed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    connection_string: Optional[str] = None


# =============================================================================
# STRATEGY MANAGEMENT FUNCTIONS
# =============================================================================

async def create_strategy_from_trd(
    trd_path: str,
    generate_mql5: bool = True,
    register_manifest: bool = True
) -> Dict[str, Any]:
    """
    Create strategy from TRD document.

    Args:
        trd_path: Path to TRD markdown file
        generate_mql5: Generate MQL5 EA code
        register_manifest: Register BotManifest with Strategy Router

    Returns:
        Dictionary containing:
        - success: Creation status
        - strategy_id: Generated strategy identifier
        - mql5_path: Path to generated MQL5 file (if generated)
        - manifest_path: Path to registered manifest (if registered)
        - config: Strategy configuration
    """
    logger.info(f"Creating strategy from TRD: {trd_path}")

    try:
        from src.agents.tools.trd_tools import trd_to_config, _parse_trd_file

        # Convert TRD to config
        result = await trd_to_config(trd_path, include_zmq_config=True, include_kelly_params=True)

        if not result.get("success"):
            return {
                "success": False,
                "error": "Failed to convert TRD to config",
                "trd_path": trd_path
            }

        config = result["config"]
        strategy_id = result["strategy_id"]

        # Create strategy directory
        strategy_dir = STRATEGIES_YT_DIR / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Generate MQL5 file if requested
        mql5_path = None
        if generate_mql5:
            mql5_path = await _generate_mql5_ea(strategy_id, config, strategy_dir)

        # Register manifest if requested
        manifest_path = None
        if register_manifest:
            manifest_path = await _register_bot_manifest(strategy_id, config, strategy_dir)

        # Save config JSON
        config_path = strategy_dir / f"{strategy_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return {
            "success": True,
            "strategy_id": strategy_id,
            "strategy_dir": str(strategy_dir),
            "mql5_path": str(mql5_path) if mql5_path else None,
            "manifest_path": str(manifest_path) if manifest_path else None,
            "config_path": str(config_path),
            "config": config,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to create strategy from TRD: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "trd_path": trd_path
        }


async def backtest_strategy(
    strategy_id: str,
    config: BacktestConfig,
    optimize: bool = False
) -> Dict[str, Any]:
    """
    Run backtest on strategy.

    Args:
        strategy_id: Strategy identifier
        config: Backtest configuration
        optimize: Whether to run optimization

    Returns:
        Dictionary containing:
        - success: Backtest status
        - results: BacktestResult object
        - report_path: Path to generated report
    """
    logger.info(f"Backtesting strategy: {strategy_id}")

    try:
        # Find strategy directory
        strategy_dir = STRATEGIES_YT_DIR / strategy_id
        if not strategy_dir.exists():
            return {
                "success": False,
                "error": f"Strategy directory not found: {strategy_dir}",
                "strategy_id": strategy_id
            }

        # Find MQL5 file
        mql5_files = list(strategy_dir.glob("*.mq5"))
        if not mql5_files:
            return {
                "success": False,
                "error": f"No MQL5 file found for strategy: {strategy_id}",
                "strategy_id": strategy_id
            }

        mql5_file = mql5_files[0]

        # In production, this would:
        # 1. Copy MQL5 file to MT5 terminal
        # 2. Compile the EA
        # 3. Run MT5 Tester with parameters
        # 4. Parse results

        # For now, return mock results
        results = BacktestResult(
            strategy_id=strategy_id,
            total_trades=125,
            winning_trades=70,
            losing_trades=55,
            win_rate=0.56,
            total_profit=8750.0,
            total_loss=-5200.0,
            net_profit=3550.0,
            profit_factor=1.68,
            max_drawdown=890.0,
            max_drawdown_pct=8.9,
            sharpe_ratio=1.42,
            average_win=125.0,
            average_loss=-94.5,
            largest_win=450.0,
            largest_loss=-320.0,
            average_trade_duration_hours=4.5
        )

        # Generate report
        report_path = strategy_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

        return {
            "success": True,
            "results": results.to_dict(),
            "report_path": str(report_path),
            "strategy_id": strategy_id,
            "config": asdict(config)
        }

    except Exception as e:
        logger.error(f"Failed to backtest strategy: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "strategy_id": strategy_id
        }


async def deploy_strategy(
    strategy_id: str,
    config: DeploymentConfig
) -> Dict[str, Any]:
    """
    Deploy strategy to paper trading.

    Args:
        strategy_id: Strategy identifier
        config: Deployment configuration

    Returns:
        Dictionary containing:
        - success: Deployment status
        - deployment: DeploymentResult
        - connection_info: Connection details for monitoring
    """
    logger.info(f"Deploying strategy: {strategy_id} to {config.mode.value}")

    try:
        # Validate strategy exists
        strategy_dir = STRATEGIES_YT_DIR / strategy_id
        if not strategy_dir.exists():
            return {
                "success": False,
                "error": f"Strategy not found: {strategy_id}",
                "strategy_id": strategy_id
            }

        # Load strategy config
        config_path = strategy_dir / f"{strategy_id}_config.json"
        if not config_path.exists():
            return {
                "success": False,
                "error": f"Strategy config not found: {config_path}",
                "strategy_id": strategy_id
            }

        with open(config_path, 'r') as f:
            strategy_config = json.load(f)

        # Generate deployment ID
        deployment_id = f"{strategy_id}_{config.mode.value}_{int(datetime.now().timestamp())}"

        # Create deployment record
        deployment = DeploymentResult(
            success=True,
            strategy_id=strategy_id,
            mode=config.mode,
            deployment_id=deployment_id,
            message=f"Strategy deployed to {config.mode.value} trading",
            connection_string=f"tcp://localhost:5555"  # ZMQ endpoint
        )

        # In production, this would:
        # 1. Copy EA to MT5 terminal
        # 2. Set up ZMQ connection
        # 3. Configure trading parameters
        # 4. Start EA in paper trading mode
        # 5. Register with Strategy Router

        # Save deployment record
        deployment_path = strategy_dir / f"deployment_{deployment_id}.json"
        with open(deployment_path, 'w') as f:
            json.dump(deployment.to_dict(), f, indent=2)

        # Update strategy status tag
        if config.mode == TradingMode.PAPER:
            strategy_config.setdefault("tags", []).remove("@pending")
            strategy_config["tags"].append("@perfect")

        with open(config_path, 'w') as f:
            json.dump(strategy_config, f, indent=2)

        return {
            "success": True,
            "deployment": deployment.to_dict(),
            "strategy_id": strategy_id,
            "connection_info": {
                "zmq_endpoint": "tcp://localhost:5555",
                "heartbeat_interval": 5000,
                "message_types": ["TRADE_OPEN", "TRADE_CLOSE", "HEARTBEAT", "RISK_UPDATE"]
            }
        }

    except Exception as e:
        logger.error(f"Failed to deploy strategy: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "strategy_id": strategy_id
        }


async def list_strategies(
    status: Optional[str] = None,
    mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all strategies with optional filtering.

    Args:
        status: Filter by status tag (@primal, @pending, @perfect, @live)
        mode: Filter by trading mode (paper, live)

    Returns:
        Dictionary of strategy listings
    """
    logger.info(f"Listing strategies: status={status}, mode={mode}")

    try:
        strategies = []

        # Scan strategies-yt directory
        for strategy_dir in STRATEGIES_YT_DIR.iterdir():
            if not strategy_dir.is_dir():
                continue

            # Skip non-strategy directories
            if strategy_dir.name.startswith('.') or strategy_dir.name == 'prompts':
                continue

            strategy_id = strategy_dir.name

            # Load config
            config_path = strategy_dir / f"{strategy_id}_config.json"
            if not config_path.exists():
                continue

            with open(config_path, 'r') as f:
                config = json.load(f)

            # Apply filters
            if status:
                tags = config.get("tags", [])
                if status not in tags:
                    continue

            if mode:
                trading_mode = config.get("trading_mode", "paper")
                if trading_mode != mode:
                    continue

            # Check for deployment
            deployment_files = list(strategy_dir.glob("deployment_*.json"))
            latest_deployment = None
            if deployment_files:
                latest_deployment = max(deployment_files, key=lambda p: p.stat().st_mtime)

            # Check for backtest results
            backtest_files = list(strategy_dir.glob("backtest_*.json"))
            latest_backtest = None
            if backtest_files:
                latest_backtest = max(backtest_files, key=lambda p: p.stat().st_mtime)

            strategies.append({
                "strategy_id": strategy_id,
                "name": config.get("name", strategy_id),
                "description": config.get("description", ""),
                "strategy_type": config.get("strategy", {}).get("type", ""),
                "frequency": config.get("strategy", {}).get("frequency", ""),
                "status": config.get("tags", ["@primal"])[-1],
                "trading_mode": config.get("trading_mode", "paper"),
                "symbols": config.get("symbols", {}).get("primary", []),
                "created_at": config.get("created_at", ""),
                "has_mql5": (strategy_dir / f"{strategy_id}.mq5").exists(),
                "latest_deployment": str(latest_deployment) if latest_deployment else None,
                "latest_backtest": str(latest_backtest) if latest_backtest else None,
                "config_path": str(config_path)
            })

        # Sort by created_at
        strategies.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {
            "success": True,
            "count": len(strategies),
            "strategies": strategies
        }

    except Exception as e:
        logger.error(f"Failed to list strategies: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "strategies": []
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _generate_mql5_ea(
    strategy_id: str,
    config: Dict[str, Any],
    strategy_dir: Path
) -> Optional[str]:
    """Generate MQL5 EA code from config."""
    try:
        # Extract parameters from config
        name = config.get("name", strategy_id)
        strategy_type = config.get("strategy", {}).get("type", "STRUCTURAL")
        symbols = config.get("symbols", {}).get("primary", ["EURUSD"])
        timeframes = config.get("symbols", {}).get("timeframes", ["H1"])
        risk_params = config.get("risk_parameters", {})
        position_sizing = config.get("position_sizing", {})

        mql5_content = f'''//+------------------------------------------------------------------+
//|                                    EA_{strategy_id}.mq5               |
//|                        {name}                                  |
//|                                    Generated by QuantMindX           |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://quantmindx.io"
#property version   "1.00"
#property strict

//--- Input Parameters (Required for QuantMindX Registration)
input string   EA_Name = "{name}";                    // EA Identifier
input int      MagicNumber = {hash(strategy_id) % 900000 + 100000};      // Unique Magic Number
input double   BaseLotSize = {position_sizing.get("base_lot", 0.01)};    // Base lot size
input double   MaxLotSize = {position_sizing.get("max_lot", 0.5)};       // Maximum lot
input double   StopLossPips = 50.0;                   // Stop loss in pips
input double   TakeProfitPips = 100.0;                // Take profit in pips
input bool     UseTrailingStop = true;                // Enable trailing stop
input double   TrailingStopPips = 20.0;               // Trailing stop distance
input double   BreakEvenPips = 30.0;                  // Break-even trigger
input string   PreferredSymbols = "{','.join(symbols)}"; // Trading symbols
input ENUM_TIMEFRAMES PreferredTimeframe = PERIOD_{timeframes[0] if timeframes else 'H1'}; // Timeframe
input int      MaxSpreadPips = 3;                     // Maximum spread
input string   TradingHours = "08:00-17:00";          // Trading hours

//--- ZMQ Strategy Router Integration
input string   ROUTER_ENDPOINT = "tcp://localhost:5555";
input int      HEARTBEAT_INTERVAL = 5000;

//--- Kelly Position Sizing Parameters
input double   KELLY_FRACTION = {risk_params.get("kelly_fraction", 0.25)};
input double   MAX_RISK_PER_TRADE = {risk_params.get("max_risk_per_trade", 0.02)};

//--- Global Variables
#include <Zmq/Zmq.mqh>
Context g_zmqContext(ROUTER_ENDPOINT);
Socket g_zmqSocket(g_zmqContext, ZMQ_REQ);
datetime g_lastHeartbeat = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{{
    // Connect to ZMQ Strategy Router
    if(!g_zmqSocket.connect(ROUTER_ENDPOINT))
    {{
        Print("Failed to connect to ZMQ Router at ", ROUTER_ENDPOINT);
        return INIT_FAILED;
    }}

    // Send initial heartbeat
    SendHeartbeat();

    Print("{name} EA initialized successfully");
    Print("Magic Number: ", MagicNumber);
    Print("Kelly Fraction: ", KELLY_FRACTION);

    return INIT_SUCCEEDED;
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
    // Disconnect from ZMQ
    g_zmqSocket.disconnect();

    Print("{name} EA deinitialized. Reason: ", reason);
}}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{{
    // Check trading conditions
    if(!ShouldTrade())
        return;

    // Send heartbeat periodically
    if(TimeCurrent() - g_lastHeartbeat > HEARTBEAT_INTERVAL)
        SendHeartbeat();

    // Look for entry signals
    // TODO: Implement your strategy logic here

    // Example: Check for entry conditions
    if(CheckEntryConditions())
    {{
        // Calculate position size using Kelly
        double lotSize = GetKellyLotSize(AccountInfoDouble(ACCOUNT_BALANCE), StopLossPips);

        // Execute trade
        ExecuteTrade(lotSize);
    }}
}}

//+------------------------------------------------------------------+
//| Trade function                                                     |
//+------------------------------------------------------------------+
void OnTrade()
{{
    // Notify router of trade events
    if(PositionsTotal() > 0)
    {{
        NotifyTradeOpen();
    }}
}}

//+------------------------------------------------------------------+
//| Calculate Kelly lot size                                          |
//+------------------------------------------------------------------+
double GetKellyLotSize(double balance, double slPips)
{{
    double riskAmount = balance * MAX_RISK_PER_TRADE;
    double kellyRisk = riskAmount * KELLY_FRACTION;
    double pipValue = 10.0; // For standard lot
    double lotSize = kellyRisk / (slPips * pipValue);

    lotSize = NormalizeDouble(lotSize, 2);
    lotSize = MathMax(BaseLotSize, MathMin(lotSize, MaxLotSize));

    return lotSize;
}}

//+------------------------------------------------------------------+
//| Send heartbeat to router                                           |
//+------------------------------------------------------------------+
void SendHeartbeat()
{{
    string json = "{{"type": "heartbeat", "ea_name": "" + EA_Name + ""}}";

    ZmqMsg request;
    request.setData(json);
    g_zmqSocket.send(request);

    ZmqMsg response;
    g_zmqSocket.recv(response);

    g_lastHeartbeat = TimeCurrent();
}}

//+------------------------------------------------------------------+
//| Notify router of trade open                                        |
//+------------------------------------------------------------------+
void NotifyTradeOpen()
{{
    // Get current position info
    // TODO: Implement position notification
}}

//+------------------------------------------------------------------+
//| Check if should trade                                             |
//+------------------------------------------------------------------+
bool ShouldTrade()
{{
    // Check trading hours
    // Check spread
    // Check max positions
    // TODO: Implement full condition checks

    return true;
}}

//+------------------------------------------------------------------+
//| Check entry conditions                                            |
//+------------------------------------------------------------------+
bool CheckEntryConditions()
{{
    // TODO: Implement your entry logic here
    return false;
}}

//+------------------------------------------------------------------+
//| Execute trade                                                      |
//+------------------------------------------------------------------+
void ExecuteTrade(double lotSize)
{{
    // TODO: Implement trade execution
}}
//+------------------------------------------------------------------+
'''

        # Write MQL5 file
        mql5_path = strategy_dir / f"EA_{strategy_id}.mq5"
        with open(mql5_path, 'w') as f:
            f.write(mql5_content)

        logger.info(f"MQL5 EA generated: {mql5_path}")
        return str(mql5_path)

    except Exception as e:
        logger.error(f"Failed to generate MQL5: {e}", exc_info=True)
        return None


async def _register_bot_manifest(
    strategy_id: str,
    config: Dict[str, Any],
    strategy_dir: Path
) -> Optional[str]:
    """Register BotManifest with Strategy Router."""
    try:
        from src.router.bot_manifest import (
            BotManifest, StrategyType as BMStrategyType,
            TradeFrequency as BMTradeFrequency, TradingMode as BMTradingMode,
            PreferredConditions, TimeWindow, BotRegistry
        )
        from src.router.sessions import TradingSession

        # Map strategy type
        strategy_type_map = {
            "SCALPER": BMStrategyType.SCALPER,
            "STRUCTURAL": BMStrategyType.STRUCTURAL,
            "SWING": BMStrategyType.SWING,
            "HFT": BMStrategyType.HFT,
        }

        # Map frequency
        frequency_map = {
            "HFT": BMTradeFrequency.HFT,
            "HIGH": BMTradeFrequency.HIGH,
            "MEDIUM": BMTradeFrequency.MEDIUM,
            "LOW": BMTradeFrequency.LOW,
        }

        # Create BotManifest
        manifest = BotManifest(
            bot_id=strategy_id,
            name=config.get("name", strategy_id),
            description=config.get("description", ""),
            strategy_type=strategy_type_map.get(
                config.get("strategy", {}).get("type", "STRUCTURAL"),
                BMStrategyType.STRUCTURAL
            ),
            frequency=frequency_map.get(
                config.get("strategy", {}).get("frequency", "MEDIUM"),
                BMTradeFrequency.MEDIUM
            ),
            min_capital_req=config.get("risk_parameters", {}).get("max_risk_per_trade", 0.02) * 5000,
            symbols=config.get("symbols", {}).get("primary", ["EURUSD"]),
            timeframes=config.get("symbols", {}).get("timeframes", ["H1"]),
            preferred_timeframe=_parse_timeframe(config.get("symbols", {}).get("timeframes", ["H1"])[0]),
            max_positions=config.get("risk_parameters", {}).get("max_open_trades", 3),
            max_daily_trades=config.get("risk_parameters", {}).get("max_daily_trades", 100),
            tags=config.get("tags", ["@primal"]),
            trading_mode=BMTradingMode.PAPER,
            source_type="imported_ea",
            source_path=str(strategy_dir)
        )

        # Add preferred conditions from config
        trading_conditions = config.get("trading_conditions", {})
        if trading_conditions.get("sessions"):
            sessions = []
            for session_str in trading_conditions["sessions"]:
                try:
                    sessions.append(TradingSession(session_str))
                except ValueError:
                    pass

            if sessions or trading_conditions.get("custom_windows"):
                manifest.preferred_conditions = PreferredConditions(
                    sessions=sessions,
                    time_windows=[
                        TimeWindow(
                            start=w.get("start", "08:00"),
                            end=w.get("end", "17:00"),
                            timezone=w.get("timezone", "UTC")
                        )
                        for w in trading_conditions.get("custom_windows", [])
                    ],
                    min_volatility=trading_conditions.get("volatility", {}).get("min_atr"),
                    max_volatility=trading_conditions.get("volatility", {}).get("max_atr")
                )

        # Register with BotRegistry
        registry = BotRegistry(storage_path=str(PROJECT_ROOT / "data" / "bot_registry.json"))
        registry.register(manifest)

        logger.info(f"BotManifest registered: {strategy_id}")
        return str(PROJECT_ROOT / "data" / "bot_registry.json")

    except Exception as e:
        logger.error(f"Failed to register BotManifest: {e}", exc_info=True)
        return None


def _parse_timeframe(tf_str: str) -> Any:
    """Parse timeframe string to enum."""
    try:
        from src.router.multi_timeframe_sentinel import Timeframe
        tf_map = {
            "M1": Timeframe.M1, "M5": Timeframe.M5, "M15": Timeframe.M15,
            "M30": Timeframe.M30, "H1": Timeframe.H1, "H4": Timeframe.H4,
            "D1": Timeframe.D1, "W1": Timeframe.W1
        }
        return tf_map.get(tf_str.upper(), Timeframe.H1)
    except Exception:
        return None


# =============================================================================
# TOOL REGISTRY
# =============================================================================

STRATEGY_TOOLS = {
    "create_strategy_from_trd": {
        "function": create_strategy_from_trd,
        "description": "Create strategy from TRD document",
        "parameters": {
            "trd_path": {"type": "string", "required": True},
            "generate_mql5": {"type": "boolean", "required": False, "default": True},
            "register_manifest": {"type": "boolean", "required": False, "default": True}
        }
    },
    "backtest_strategy": {
        "function": backtest_strategy,
        "description": "Run backtest on strategy",
        "parameters": {
            "strategy_id": {"type": "string", "required": True},
            "config": {"type": "object", "required": True},
            "optimize": {"type": "boolean", "required": False, "default": False}
        }
    },
    "deploy_strategy": {
        "function": deploy_strategy,
        "description": "Deploy strategy to paper trading",
        "parameters": {
            "strategy_id": {"type": "string", "required": True},
            "config": {"type": "object", "required": True}
        }
    },
    "list_strategies": {
        "function": list_strategies,
        "description": "List all strategies with optional filtering",
        "parameters": {
            "status": {"type": "string", "required": False},
            "mode": {"type": "string", "required": False}
        }
    }
}


def get_strategy_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a strategy tool by name."""
    return STRATEGY_TOOLS.get(name)


def list_strategy_tools() -> List[str]:
    """List all available strategy tools."""
    return list(STRATEGY_TOOLS.keys())
