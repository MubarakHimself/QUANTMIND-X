"""
TRD (Trading Requirements Document) Tools

Tools for generating, validating, and managing Trading Requirements Documents
from video analysis output (Truth Objects) for the strategies-yt pipeline.

Integrates with:
- Video analysis pipeline (strategies-yt)
- TRD generation agent
- Strategy Router BotManifest system
- ZMQ communication layer
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# PATH CONSTANTS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
STRATEGIES_YT_DIR = PROJECT_ROOT / "strategies-yt"
TRD_DIR = STRATEGIES_YT_DIR
PROMPTS_DIR = STRATEGIES_YT_DIR / "prompts"


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class StrategyType(Enum):
    """Strategy classification for routing decisions."""
    SCALPER = "SCALPER"
    STRUCTURAL = "STRUCTURAL"
    SWING = "SWING"
    HFT = "HFT"


class TradeFrequency(Enum):
    """Trade frequency classification."""
    HFT = "HFT"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class TradingSession(Enum):
    """Trading sessions."""
    ASIAN = "ASIAN"
    LONDON = "LONDON"
    NEW_YORK = "NEW_YORK"
    OVERLAP = "OVERLAP"


@dataclass
class TruthObject:
    """
    Truth Object - Input format for TRD generation.

    Represents a parsed video analysis result containing
    strategy elements extracted from educational content.
    """
    # Core identification
    title: str
    version: str = "1.0.0"
    description: str = ""

    # Strategy classification
    strategy_type: StrategyType = StrategyType.STRUCTURAL
    frequency: TradeFrequency = TradeFrequency.MEDIUM
    direction: str = "BOTH"  # BOTH, LONG, SHORT

    # Trading logic
    entry_conditions: List[str] = field(default_factory=list)
    exit_conditions: List[str] = field(default_factory=list)
    filters: List[str] = field(default_factory=list)

    # Risk parameters
    stop_loss_pips: float = 50.0
    take_profit_pips: float = 100.0
    kelly_fraction: float = 0.25

    # Trading preferences
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD"])
    timeframes: List[str] = field(default_factory=lambda: ["H1"])
    sessions: List[str] = field(default_factory=lambda: ["LONDON", "NEW_YORK"])
    trading_hours: str = ""

    # Performance targets
    target_win_rate: Optional[float] = None
    target_sharpe: Optional[float] = None

    # Metadata
    author: str = "TRD Generation Agent"
    source_url: str = ""
    source_video_id: str = ""
    tags: List[str] = field(default_factory=lambda: ["@primal", "demo"])

    # Custom variables for strategy-specific parameters
    custom_variables: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling enums."""
        data = asdict(self)
        # Convert enums to values
        if isinstance(self.strategy_type, StrategyType):
            data['strategy_type'] = self.strategy_type.value
        if isinstance(self.frequency, TradeFrequency):
            data['frequency'] = self.frequency.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TruthObject":
        """Create from dictionary, handling enums."""
        # Handle enum conversion
        if 'strategy_type' in data and isinstance(data['strategy_type'], str):
            data['strategy_type'] = StrategyType(data['strategy_type'])
        if 'frequency' in data and isinstance(data['frequency'], str):
            data['frequency'] = TradeFrequency(data['frequency'])
        return cls(**data)


@dataclass
class TRDValidationResult:
    """Result of TRD validation."""
    valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    kelly_fraction_adjusted: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# TRD GENERATION FUNCTIONS
# =============================================================================

async def generate_trd_from_video(
    video_id: str,
    truth_object: Optional[Dict[str, Any]] = None,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """
    Generate TRD from video analysis output.

    Args:
        video_id: YouTube video ID or identifier
        truth_object: Optional pre-extracted truth object data
        force_regenerate: Force regeneration even if TRD exists

    Returns:
        Dictionary containing:
        - success: Generation status
        - trd_path: Path to generated TRD
        - config_path: Path to generated config
        - truth_object: The truth object used
        - validation: Validation results
    """
    logger.info(f"Generating TRD for video: {video_id}")

    try:
        # Load or create truth object
        if truth_object is None:
            truth_object = await _extract_truth_from_video(video_id)

        # Validate truth object
        truth = TruthObject.from_dict(truth_object)
        validation = await validate_trd(truth)

        if not validation.valid and not force_regenerate:
            return {
                "success": False,
                "error": "Truth object validation failed",
                "validation": validation.to_dict(),
                "video_id": video_id
            }

        # Generate safe strategy ID from title
        strategy_id = _generate_strategy_id(truth.title)
        strategy_dir = TRD_DIR / strategy_id
        strategy_dir.mkdir(parents=True, exist_ok=True)

        # Generate TRD markdown
        trd_content = _generate_trd_markdown(truth, validation)
        trd_path = strategy_dir / f"{strategy_id}_TRD.md"

        if not trd_path.exists() or force_regenerate:
            with open(trd_path, 'w') as f:
                f.write(trd_content)
            logger.info(f"TRD written to: {trd_path}")

        # Generate config JSON
        config_content = _generate_config_json(truth, validation)
        config_path = strategy_dir / f"{strategy_id}_config.json"

        if not config_path.exists() or force_regenerate:
            with open(config_path, 'w') as f:
                json.dump(config_content, f, indent=2)
            logger.info(f"Config written to: {config_path}")

        return {
            "success": True,
            "trd_path": str(trd_path),
            "config_path": str(config_path),
            "strategy_id": strategy_id,
            "truth_object": truth.to_dict(),
            "validation": validation.to_dict(),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to generate TRD: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "video_id": video_id
        }


async def validate_trd(
    truth_object: TruthObject
) -> TRDValidationResult:
    """
    Validate TRD structure and completeness.

    Args:
        truth_object: Truth object to validate

    Returns:
        TRDValidationResult with issues and warnings
    """
    issues: List[str] = []
    warnings: List[str] = []

    # Required fields
    if not truth_object.title or truth_object.title.strip() == "":
        issues.append("Missing title")

    if not truth_object.description or truth_object.description.strip() == "":
        issues.append("Missing description")

    # Validate strategy type
    valid_types = [t.value for t in StrategyType]
    if truth_object.strategy_type.value not in valid_types:
        issues.append(f"Invalid strategy_type: {truth_object.strategy_type}")

    # Validate entry/exit conditions
    if not truth_object.entry_conditions:
        warnings.append("No entry conditions defined")

    if not truth_object.exit_conditions:
        warnings.append("No exit conditions defined")

    # Validate and adjust Kelly fraction
    recommended_kelly = _get_recommended_kelly(truth_object.strategy_type)
    kelly_adjusted = None

    if truth_object.kelly_fraction < 0.1 or truth_object.kelly_fraction > 0.4:
        warnings.append(
            f"Kelly fraction {truth_object.kelly_fraction} outside recommended range "
            f"[{recommended_kelly[0]:.2f}-{recommended_kelly[1]:.2f}] for {truth_object.strategy_type.value}"
        )
        kelly_adjusted = max(recommended_kelly[0], min(truth_object.kelly_fraction, recommended_kelly[1]))

    # Validate symbols
    if not truth_object.symbols:
        issues.append("No trading symbols specified")

    # Validate SL/TP ratio
    if truth_object.take_profit_pips > 0 and truth_object.stop_loss_pips > 0:
        rr_ratio = truth_object.take_profit_pips / truth_object.stop_loss_pips
        if rr_ratio < 1.0:
            warnings.append(f"Risk/Reward ratio {rr_ratio:.2f} < 1.0 (lose more than win)")

    # Calculate completeness score
    required_fields = [
        truth_object.title,
        truth_object.description,
        truth_object.strategy_type,
        truth_object.symbols,
        truth_object.entry_conditions,
        truth_object.exit_conditions
    ]
    completeness = sum(1 for f in required_fields if f) / len(required_fields) * 100

    return TRDValidationResult(
        valid=len(issues) == 0,
        issues=issues,
        warnings=warnings,
        completeness_score=completeness,
        kelly_fraction_adjusted=kelly_adjusted
    )


async def trd_to_config(
    trd_path: str,
    include_zmq_config: bool = True,
    include_kelly_params: bool = True
) -> Dict[str, Any]:
    """
    Convert TRD to EA config JSON.

    Args:
        trd_path: Path to TRD markdown file
        include_zmq_config: Include ZMQ router configuration
        include_kelly_params: Include fee-aware Kelly parameters

    Returns:
        Config dictionary
    """
    logger.info(f"Converting TRD to config: {trd_path}")

    try:
        trd_file = Path(trd_path)
        if not trd_file.exists():
            return {
                "success": False,
                "error": f"TRD file not found: {trd_path}"
            }

        # Read and parse TRD
        with open(trd_file, 'r') as f:
            trd_content = f.read()

        # Extract metadata from TRD
        metadata = _parse_trd_metadata(trd_content)
        strategy_id = trd_file.stem.replace("_TRD", "")

        # Build config
        config = {
            "ea_id": strategy_id,
            "name": metadata.get("name", strategy_id.replace("_", " ").title()),
            "version": metadata.get("version", "1.0.0"),
            "description": metadata.get("description", ""),

            "strategy": {
                "type": metadata.get("strategy_type", "STRUCTURAL"),
                "frequency": metadata.get("frequency", "MEDIUM"),
                "direction": metadata.get("direction", "BOTH"),
            },

            "symbols": {
                "primary": metadata.get("symbols", ["EURUSD", "GBPUSD"]),
                "timeframes": metadata.get("timeframes", ["H1"]),
                "symbol_groups": ["majors"],
            },

            "trading_conditions": {
                "sessions": metadata.get("sessions", ["LONDON", "NEW_YORK"]),
                "timezone": "UTC",
                "volatility": {
                    "min_atr": 0.0005,
                    "max_atr": 0.003,
                    "atr_period": 14,
                },
                "preferred_regime": "TRENDING",
                "min_spread_pips": 0,
                "max_spread_pips": 3,
            },

            "risk_parameters": {
                "kelly_fraction": metadata.get("kelly_fraction", 0.25),
                "max_risk_per_trade": 0.02,
                "max_daily_loss": 0.05,
                "max_drawdown": 0.15,
                "max_open_trades": 3,
                "correlation_limit": 0.7,
            },

            "position_sizing": {
                "base_lot": 0.01,
                "max_lot": 0.5,
                "scale_with_account": True,
                "respect_prop_limits": True,
            },

            "broker_preferences": {
                "preferred_type": "RAW_ECN",
                "min_leverage": 100,
                "allowed_brokers": [],
                "excluded_brokers": [],
            },

            "prop_firm": {
                "compatible": True,
                "supported_firms": ["FTMO", "The5ers", "FundingPips"],
                "daily_loss_limit": 0.05,
                "max_trailing_drawdown": 0.10,
            },

            "tags": metadata.get("tags", ["@primal"]),
            "author": metadata.get("author", "TRD Generation Agent"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add ZMQ config if requested
        if include_zmq_config:
            config["zmq_router"] = {
                "enabled": True,
                "endpoint": "tcp://localhost:5555",
                "heartbeat_interval_ms": 5000,
                "message_types": ["TRADE_OPEN", "TRADE_CLOSE", "TRADE_MODIFY", "HEARTBEAT", "RISK_UPDATE"],
                "subscription_topics": ["risk_multiplier", "regime_change", "circuit_breaker"]
            }

        # Add Kelly params if requested
        if include_kelly_params:
            config["kelly_position_sizing"] = {
                "enabled": True,
                "kelly_fraction": metadata.get("kelly_fraction", 0.25),
                "fee_adjusted": True,
                "broker_fee_per_lot": 7.0,
                "spread_cost_pips": 1.5,
            }

        # Add custom variables if present
        if "custom_variables" in metadata:
            config["custom_variables"] = metadata["custom_variables"]

        # Add source tracking
        config["source"] = {
            "trd_path": str(trd_path),
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

        return {
            "success": True,
            "config": config,
            "strategy_id": strategy_id
        }

    except Exception as e:
        logger.error(f"Failed to convert TRD to config: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "trd_path": trd_path
        }


async def list_trds(
    strategy_id: Optional[str] = None,
    tag: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all TRDs in strategies-yt folder.

    Args:
        strategy_id: Filter by specific strategy ID
        tag: Filter by tag (e.g., @primal, @pending, @live)

    Returns:
        Dictionary of TRD listings
    """
    logger.info(f"Listing TRDs: strategy_id={strategy_id}, tag={tag}")

    try:
        trds = []

        # Scan strategies-yt directory
        for strategy_dir in STRATEGIES_YT_DIR.iterdir():
            if not strategy_dir.is_dir():
                continue

            # Skip non-strategy directories
            if strategy_dir.name.startswith('.') or strategy_dir.name == 'prompts':
                continue

            # Filter by strategy_id if specified
            if strategy_id and strategy_dir.name != strategy_id:
                continue

            # Find TRD files
            trd_files = list(strategy_dir.glob("*_TRD.md"))
            if not trd_files:
                continue

            for trd_file in trd_files:
                # Parse TRD for metadata
                metadata = _parse_trd_file(trd_file)

                # Filter by tag if specified
                if tag and tag not in metadata.get("tags", []):
                    continue

                trds.append({
                    "strategy_id": strategy_dir.name,
                    "trd_path": str(trd_file),
                    "config_path": str(strategy_dir / f"{strategy_dir.name}_config.json"),
                    "metadata": metadata,
                    "created_at": metadata.get("created_at", ""),
                    "tags": metadata.get("tags", [])
                })

        # Sort by created_at
        trds.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        return {
            "success": True,
            "count": len(trds),
            "trds": trds,
            "strategies_yt_path": str(STRATEGIES_YT_DIR)
        }

    except Exception as e:
        logger.error(f"Failed to list TRDs: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "trds": []
        }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

async def _extract_truth_from_video(video_id: str) -> Dict[str, Any]:
    """Extract truth object from video analysis."""
    # In production, this would call the video analysis pipeline
    # For now, return a minimal truth object
    return {
        "title": f"Strategy from {video_id}",
        "version": "1.0.0",
        "description": f"Trading strategy extracted from video {video_id}",
        "strategy_type": "STRUCTURAL",
        "frequency": "MEDIUM",
        "direction": "BOTH",
        "entry_conditions": ["Define entry conditions from video"],
        "exit_conditions": ["Define exit conditions from video"],
        "stop_loss_pips": 50.0,
        "take_profit_pips": 100.0,
        "kelly_fraction": 0.25,
        "symbols": ["EURUSD"],
        "timeframes": ["H1"],
        "sessions": ["LONDON", "NEW_YORK"],
        "source_video_id": video_id,
        "tags": ["@primal", "demo"]
    }


def _generate_strategy_id(title: str) -> str:
    """Generate safe strategy ID from title."""
    # Remove special characters, lowercase, replace spaces with underscores
    safe = "".join(c.lower() if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    return safe.replace(' ', '_').replace('-', '_')[:50]


def _generate_trd_markdown(truth: TruthObject, validation: TRDValidationResult) -> str:
    """Generate TRD markdown content."""
    kelly = validation.kelly_fraction_adjusted or truth.kelly_fraction

    trd = f"""# Technical Requirements Document (TRD)

## {truth.title}

**Version**: {truth.version}
**Strategy Type**: {truth.strategy_type.value}
**Frequency**: {truth.frequency.value}
**Direction**: {truth.direction}

---

## 1. Strategy Overview

{truth.description}

### Strategy Classification

| Parameter | Value |
|-----------|-------|
| Type | {truth.strategy_type.value} |
| Frequency | {truth.frequency.value} |
| Direction | {truth.direction} |
| Kelly Fraction | {kelly:.3f} |

### Trading Instruments

**Symbols**: {', '.join(truth.symbols)}
**Timeframes**: {', '.join(truth.timeframes)}
**Sessions**: {', '.join(truth.sessions)}
{f"**Trading Hours**: {truth.trading_hours}" if truth.trading_hours else ""}

---

## 2. Entry Conditions

{chr(10).join(f"{i+1}. {cond}" for i, cond in enumerate(truth.entry_conditions))}

---

## 3. Exit Conditions

{chr(10).join(f"{i+1}. {cond}" for i, cond in enumerate(truth.exit_conditions))}

---

## 4. Risk Management

### Stop Loss & Take Profit

- **Stop Loss**: {truth.stop_loss_pips} pips
- **Take Profit**: {truth.take_profit_pips} pips
- **Risk/Reward Ratio**: {(truth.take_profit_pips / truth.stop_loss_pips):.2f}

### Kelly Position Sizing

**Base Kelly Fraction**: {kelly:.3f}

### Risk Limits

| Parameter | Value |
|-----------|-------|
| Max Risk Per Trade | 2% |
| Max Daily Loss | 5% |
| Max Drawdown | 15% |

---

## 5. Filters

{chr(10).join(f"{i+1}. {f}" for i, f in enumerate(truth.filters)) if truth.filters else "No additional filters defined."}

---

## 6. ZMQ Strategy Router Integration

The EA integrates with QuantMindX ZMQ Strategy Router for:

- **Real-time risk multiplier updates**
- **Circuit breaker notifications**
- **Regime change alerts**
- **Trade event logging**

### Connection Details

- **Endpoint**: tcp://localhost:5555
- **Heartbeat Interval**: 5000ms
- **Message Types**: TRADE_OPEN, TRADE_CLOSE, TRADE_MODIFY, HEARTBEAT, RISK_UPDATE

---

## 7. MQL5 Implementation Notes

### Required Inputs

```mql5
input string EA_Name = "{truth.title}";
input int MagicNumber = 100001;
input double BaseLotSize = 0.01;
input double MaxLotSize = 0.5;
input double StopLossPips = {truth.stop_loss_pips};
input double TakeProfitPips = {truth.take_profit_pips};
input string PreferredSymbols = "{','.join(truth.symbols)}";
input ENUM_TIMEFRAMES PreferredTimeframe = PERIOD_{truth.timeframes[0] if truth.timeframes else 'H1'};
```

---

## 8. Backtest Expectations

{f"- **Target Win Rate**: >{truth.target_win_rate * 100}%" if truth.target_win_rate else ""}
{f"- **Target Sharpe Ratio**: >{truth.target_sharpe}" if truth.target_sharpe else ""}

### Validation Criteria (PAPER -> LIVE)

- Minimum 30 days active
- Sharpe ratio > 1.5
- Win rate > 55%
- Maximum drawdown < 10%
- At least 50 trades executed

---

**Generated**: {datetime.now(timezone.utc).isoformat()}
**Author**: {truth.author}
{f"**Source**: {truth.source_url}" if truth.source_url else ""}
**Tags**: {', '.join(truth.tags)}
"""
    return trd


def _generate_config_json(truth: TruthObject, validation: TRDValidationResult) -> Dict[str, Any]:
    """Generate config JSON content."""
    strategy_id = _generate_strategy_id(truth.title)
    kelly = validation.kelly_fraction_adjusted or truth.kelly_fraction

    return {
        "ea_id": strategy_id,
        "name": truth.title,
        "version": truth.version,
        "description": truth.description,

        "strategy": {
            "type": truth.strategy_type.value,
            "frequency": truth.frequency.value,
            "direction": truth.direction,
        },

        "symbols": {
            "primary": truth.symbols,
            "timeframes": truth.timeframes,
            "symbol_groups": ["majors" if any("USD" in s for s in truth.symbols) else "crosses"],
        },

        "trading_conditions": {
            "sessions": truth.sessions,
            "timezone": "UTC",
            "volatility": {
                "min_atr": 0.0005,
                "max_atr": 0.003,
                "atr_period": 14,
            },
            "preferred_regime": "TRENDING",
            "min_spread_pips": 0,
            "max_spread_pips": 3,
        },

        "risk_parameters": {
            "kelly_fraction": kelly,
            "max_risk_per_trade": 0.02,
            "max_daily_loss": 0.05,
            "max_drawdown": 0.15,
            "max_open_trades": 3,
            "correlation_limit": 0.7,
        },

        "position_sizing": {
            "base_lot": 0.01,
            "max_lot": 0.5,
            "scale_with_account": True,
            "respect_prop_limits": True,
        },

        "broker_preferences": {
            "preferred_type": "RAW_ECN",
            "min_leverage": 100,
            "allowed_brokers": [],
            "excluded_brokers": [],
        },

        "prop_firm": {
            "compatible": True,
            "supported_firms": ["FTMO", "The5ers", "FundingPips"],
            "daily_loss_limit": 0.05,
            "max_trailing_drawdown": 0.10,
        },

        "zmq_router": {
            "enabled": True,
            "endpoint": "tcp://localhost:5555",
            "heartbeat_interval_ms": 5000,
            "message_types": ["TRADE_OPEN", "TRADE_CLOSE", "TRADE_MODIFY", "HEARTBEAT", "RISK_UPDATE"],
            "subscription_topics": ["risk_multiplier", "regime_change", "circuit_breaker"]
        },

        "kelly_position_sizing": {
            "enabled": True,
            "kelly_fraction": kelly,
            "fee_adjusted": True,
            "broker_fee_per_lot": 7.0,
            "spread_cost_pips": 1.5,
        },

        "custom_variables": truth.custom_variables,

        "tags": truth.tags,
        "author": truth.author,
        "source": {
            "video_id": truth.source_video_id,
            "source_url": truth.source_url,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _get_recommended_kelly(strategy_type: StrategyType) -> Tuple[float, float]:
    """Get recommended Kelly fraction range for strategy type."""
    kelly_ranges = {
        StrategyType.HFT: (0.10, 0.15),
        StrategyType.SCALPER: (0.15, 0.25),
        StrategyType.STRUCTURAL: (0.20, 0.30),
        StrategyType.SWING: (0.25, 0.40),
    }
    return kelly_ranges.get(strategy_type, (0.20, 0.30))


def _parse_trd_metadata(trd_content: str) -> Dict[str, Any]:
    """Parse metadata from TRD markdown content."""
    metadata = {}

    lines = trd_content.split('\n')
    for i, line in enumerate(lines):
        # Parse title
        if line.startswith('# '):
            metadata['name'] = line[2:].strip()

        # Parse version
        elif line.startswith('**Version**'):
            metadata['version'] = line.split(':', 1)[1].strip() if ':' in line else '1.0.0'

        # Parse strategy type
        elif 'Strategy Type' in line and '|' in line:
            parts = line.split('|')
            if len(parts) > 2:
                metadata['strategy_type'] = parts[2].strip()

        # Parse symbols
        elif '**Symbols**' in line:
            symbols = line.split(':', 1)[1].strip() if ':' in line else ""
            metadata['symbols'] = [s.strip() for s in symbols.split(',')]

        # Parse sessions
        elif '**Sessions**' in line:
            sessions = line.split(':', 1)[1].strip() if ':' in line else ""
            metadata['sessions'] = [s.strip() for s in sessions.split(',')]

        # Parse Kelly
        elif 'Kelly Fraction' in line and '|' in line:
            parts = line.split('|')
            if len(parts) > 2:
                try:
                    metadata['kelly_fraction'] = float(parts[2].strip())
                except ValueError:
                    pass

        # Parse tags
        elif '**Tags**' in line:
            tags = line.split(':', 1)[1].strip() if ':' in line else ""
            metadata['tags'] = [t.strip() for t in tags.split(',')]

        # Parse generated date
        elif '**Generated**' in line:
            date_str = line.split(':', 1)[1].strip() if ':' in line else ""
            metadata['created_at'] = date_str

    return metadata


def _parse_trd_file(trd_path: Path) -> Dict[str, Any]:
    """Parse TRD file and extract metadata."""
    try:
        with open(trd_path, 'r') as f:
            content = f.read()
        return _parse_trd_metadata(content)
    except Exception as e:
        logger.warning(f"Failed to parse TRD file {trd_path}: {e}")
        return {}


# =============================================================================
# TOOL REGISTRY
# =============================================================================

TRD_TOOLS = {
    "generate_trd_from_video": {
        "function": generate_trd_from_video,
        "description": "Generate TRD from video analysis output (Truth Object)",
        "parameters": {
            "video_id": {"type": "string", "required": True},
            "truth_object": {"type": "object", "required": False},
            "force_regenerate": {"type": "boolean", "required": False, "default": False}
        }
    },
    "validate_trd": {
        "function": validate_trd,
        "description": "Validate TRD structure and completeness",
        "parameters": {
            "truth_object": {"type": "object", "required": True}
        }
    },
    "trd_to_config": {
        "function": trd_to_config,
        "description": "Convert TRD to EA config JSON",
        "parameters": {
            "trd_path": {"type": "string", "required": True},
            "include_zmq_config": {"type": "boolean", "required": False, "default": True},
            "include_kelly_params": {"type": "boolean", "required": False, "default": True}
        }
    },
    "list_trds": {
        "function": list_trds,
        "description": "List all TRDs in strategies-yt folder",
        "parameters": {
            "strategy_id": {"type": "string", "required": False},
            "tag": {"type": "string", "required": False}
        }
    }
}


def get_trd_tool(name: str) -> Optional[Dict[str, Any]]:
    """Get a TRD tool by name."""
    return TRD_TOOLS.get(name)


def list_trd_tools() -> List[str]:
    """List all available TRD tools."""
    return list(TRD_TOOLS.keys())
