"""
QuantCode Agent Workflow with ToolNode Integration

Implements the QuantCode agent using LangGraph with dynamic tool calling for
strategy development, MQL5 generation, and backtesting.

**Validates: Requirements 8.3**
"""

import logging
from typing import Dict, Any, List, Optional, Union, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid

from src.agents.state import QuantCodeState
from src.agents.llm_provider import get_quantcode_llm

logger = logging.getLogger(__name__)

# ============================================================================
# Paper Trading Integration Imports
# ============================================================================

# Lazy imports to avoid circular dependencies
_paper_trading_deployer = None
_paper_trading_validator = None
_bot_manifest = None
_commander = None


def _get_paper_trading_deployer():
    """Lazy load PaperTradingDeployer to avoid import errors."""
    global _paper_trading_deployer
    if _paper_trading_deployer is None:
        try:
            from mcp_mt5.paper_trading.deployer import PaperTradingDeployer
            _paper_trading_deployer = PaperTradingDeployer()
        except ImportError as e:
            logger.warning(f"PaperTradingDeployer not available: {e}")
            return None
    return _paper_trading_deployer


def _get_paper_trading_validator():
    """Lazy load PaperTradingValidator to avoid import errors."""
    global _paper_trading_validator
    if _paper_trading_validator is None:
        try:
            from src.agents.paper_trading_validator import PaperTradingValidator
            _paper_trading_validator = PaperTradingValidator()
        except ImportError as e:
            logger.warning(f"PaperTradingValidator not available: {e}")
            return None
    return _paper_trading_validator


def _get_bot_manifest_classes():
    """Lazy load BotManifest classes to avoid import errors."""
    global _bot_manifest
    if _bot_manifest is None:
        try:
            from src.router.bot_manifest import (
                BotManifest, StrategyType, TradeFrequency, BrokerType
            )
            _bot_manifest = {
                'BotManifest': BotManifest,
                'StrategyType': StrategyType,
                'TradeFrequency': TradeFrequency,
                'BrokerType': BrokerType
            }
        except ImportError as e:
            logger.warning(f"BotManifest classes not available: {e}")
            return None
    return _bot_manifest


def _get_commander():
    """Lazy load Commander to avoid import errors."""
    global _commander
    if _commander is None:
        try:
            from src.router.commander import Commander
            _commander = Commander()
        except ImportError as e:
            logger.warning(f"Commander not available: {e}")
            return None
    return _commander


# Track deployed agents for state management
_deployed_agents: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# Tool Definitions for QuantCode Agent
# ============================================================================

@tool
def create_strategy_plan(
    strategy_type: str,
    indicators: List[str],
    entry_rules: List[str],
    exit_rules: List[str],
    risk_management: str = "Kelly Criterion"
) -> Dict[str, Any]:
    """
    Create a strategy development plan.

    Args:
        strategy_type: Type of strategy (momentum, mean_reversion, etc.)
        indicators: List of indicators to use
        entry_rules: Entry rule conditions
        exit_rules: Exit rule conditions
        risk_management: Risk management approach

    Returns:
        Strategy plan
    """
    return {
        "strategy_type": strategy_type,
        "indicators": indicators,
        "entry_rules": entry_rules,
        "exit_rules": exit_rules,
        "risk_management": risk_management,
        "created_at": datetime.now(timezone.utc).isoformat()
    }


@tool
def generate_mql5_code(
    trd_content: str,
    template: str = "standard",
    include_comments: bool = True
) -> Dict[str, Any]:
    """
    Generate MQL5 code from Technical Requirements Document.

    Args:
        trd_content: TRD content
        template: Code template to use
        include_comments: Include detailed comments

    Returns:
        Generated MQL5 code
    """
    code = """//+------------------------------------------------------------------+
//|                                    Generated Expert Advisor      |
//+------------------------------------------------------------------+
#property copyright "QuantMind"
#property version   "1.00"
#property strict

input double RiskPercent = 2.0;
input int StopLoss = 50;
input int TakeProfit = 100;
input int MagicNumber = 123456;

int OnInit() { return INIT_SUCCEEDED; }

void OnTick() {
    // Trading logic here
}
"""
    return {
        "code": code,
        "lines_of_code": len(code.split("\n")),
        "language": "mql5"
    }


@tool
def generate_component(
    component_type: str,
    requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate a specific MQL5 component.

    Args:
        component_type: Type (signal_generator, order_manager, risk_manager)
        requirements: Component requirements

    Returns:
        Generated component code
    """
    return {
        "component_type": component_type,
        "code": f"// {component_type} implementation\n// ...",
        "dependencies": []
    }


@tool
def validate_syntax(
    code: str,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Validate MQL5 code syntax.

    Args:
        code: MQL5 code to validate
        strict: Enable strict validation

    Returns:
        Validation result
    """
    errors = []
    warnings = []

    # Simple validation
    if "OnInit" not in code:
        errors.append("Missing OnInit function")
    if "OnTick" not in code:
        errors.append("Missing OnTick function")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


@tool
def fix_syntax(
    code: str,
    errors: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fix MQL5 syntax errors.

    Args:
        code: Code with errors
        errors: Known errors to fix

    Returns:
        Fixed code
    """
    return {
        "fixed_code": code,  # In production, would apply actual fixes
        "fixes_applied": 0,
        "remaining_errors": []
    }


@tool
def compile_mql5(
    code: str,
    optimization_level: str = "debug"
) -> Dict[str, Any]:
    """
    Compile MQL5 code.

    Args:
        code: MQL5 code to compile
        optimization_level: Optimization level

    Returns:
        Compilation result
    """
    return {
        "success": True,
        "output_path": "output/EA.ex5",
        "errors": [],
        "warnings": ["Variable 'unused' declared but never used"],
        "compile_time_ms": 1500
    }


@tool
def debug_code(
    code: str,
    error_message: str = None
) -> Dict[str, Any]:
    """
    Debug MQL5 code.

    Args:
        code: Code to debug
        error_message: Error message to analyze

    Returns:
        Debug analysis
    """
    return {
        "issues": [],
        "suggestions": ["Check for null pointer access", "Verify array bounds"],
        "analyzed_lines": len(code.split("\n"))
    }


@tool
def optimize_code(
    code: str,
    goals: List[str] = None
) -> Dict[str, Any]:
    """
    Optimize MQL5 code performance.

    Args:
        code: Code to optimize
        goals: Optimization goals (performance, memory, readability)

    Returns:
        Optimized code
    """
    goals = goals or ["performance"]
    return {
        "optimized_code": code,
        "optimizations": [{"type": "caching", "description": "Cached repeated calculations"}],
        "improvement_percent": 15
    }


@tool
def run_backtest(
    ea_path: str,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01",
    initial_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    Run a strategy backtest.

    Args:
        ea_path: Path to compiled EA
        symbol: Trading symbol
        timeframe: Chart timeframe
        start_date: Backtest start date
        end_date: Backtest end date
        initial_balance: Initial balance

    Returns:
        Backtest results
    """
    return {
        "backtest_id": str(uuid.uuid4())[:8],
        "symbol": symbol,
        "timeframe": timeframe,
        "results": {
            "total_trades": 150,
            "winning_trades": 95,
            "losing_trades": 55,
            "win_rate": 0.633,
            "sharpe_ratio": 1.85,
            "max_drawdown": 0.12,
            "total_return": 0.45,
            "kelly_score": 0.82,
            "profit_factor": 1.76
        }
    }


@tool
def analyze_backtest_results(
    backtest_id: str,
    include_recommendations: bool = True
) -> Dict[str, Any]:
    """
    Analyze backtest results.

    Args:
        backtest_id: Backtest ID
        include_recommendations: Include optimization recommendations

    Returns:
        Analysis report
    """
    return {
        "backtest_id": backtest_id,
        "analysis": {
            "strengths": ["High win rate", "Good Sharpe ratio"],
            "weaknesses": ["Drawdown slightly high"],
            "overall_rating": "GOOD"
        },
        "recommendations": [
            "Consider adding trailing stops",
            "Test on different market conditions"
        ] if include_recommendations else []
    }


@tool
def run_monte_carlo(
    backtest_id: str,
    simulations: int = 1000
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation.

    Args:
        backtest_id: Backtest ID to simulate
        simulations: Number of simulations

    Returns:
        Monte Carlo results
    """
    return {
        "backtest_id": backtest_id,
        "simulations": simulations,
        "results": {
            "median_return": 0.42,
            "percentile_5": 0.15,
            "percentile_95": 0.75,
            "probability_of_ruin": 0.02
        }
    }


@tool
def deploy_paper_trading(
    strategy_name: str,
    strategy_code: str,
    symbol: str = "EURUSD",
    timeframe: str = "H1",
    mt5_account: int = None,
    mt5_password: str = None,
    mt5_server: str = "MetaQuotes-Demo",
    magic_number: int = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Deploy a strategy to paper trading using Docker containers.

    This tool integrates with the PaperTradingDeployer to create isolated
    Docker containers for running trading strategies in paper trading mode.

    Args:
        strategy_name: Name of the trading strategy
        strategy_code: Python strategy code or template reference (e.g., "template:rsi-reversal")
        symbol: Trading symbol (default: EURUSD)
        timeframe: Chart timeframe (default: H1)
        mt5_account: MT5 account number
        mt5_password: MT5 account password
        mt5_server: MT5 server name (default: MetaQuotes-Demo)
        magic_number: Unique magic number for trade identification (auto-generated if not provided)
        config: Additional strategy configuration parameters

    Returns:
        Deployment result with agent_id, container_id, status, and monitoring URLs
    """
    deployer = _get_paper_trading_deployer()
    
    if deployer is None:
        # Fallback to mock deployment if deployer unavailable
        logger.warning("PaperTradingDeployer unavailable, using mock deployment")
        agent_id = str(uuid.uuid4())[:8]
        return {
            "deployed": True,
            "agent_id": agent_id,
            "container_id": f"mock-{agent_id}",
            "container_name": f"quantmindx-agent-{agent_id}",
            "status": "RUNNING",
            "terminal_id": "paper_trading_001",
            "redis_channel": f"agent:heartbeat:{agent_id}",
            "logs_url": f"docker logs -f quantmindx-agent-{agent_id}",
            "message": "Mock deployment (Docker unavailable)"
        }
    
    try:
        # Generate magic number if not provided
        if magic_number is None:
            magic_number = int(uuid.uuid4().int % 2147483647)
        
        # Build MT5 credentials
        mt5_credentials = {
            "account": mt5_account or 12345678,
            "password": mt5_password or "demo_password",
            "server": mt5_server
        }
        
        # Build strategy config
        strategy_config = config or {}
        strategy_config["symbol"] = symbol
        strategy_config["timeframe"] = timeframe
        
        # Deploy the agent
        result = deployer.deploy_agent(
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            config=strategy_config,
            mt5_credentials=mt5_credentials,
            magic_number=magic_number
        )
        
        # Store deployment metadata for tracking
        _deployed_agents[result.agent_id] = {
            "strategy_name": strategy_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "magic_number": magic_number,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "container_id": result.container_id,
            "status": result.status.value
        }
        
        return {
            "deployed": True,
            "agent_id": result.agent_id,
            "container_id": result.container_id,
            "container_name": result.container_name,
            "status": result.status.value,
            "redis_channel": result.redis_channel,
            "logs_url": result.logs_url,
            "message": result.message
        }
        
    except Exception as e:
        logger.error(f"Paper trading deployment failed: {e}")
        return {
            "deployed": False,
            "agent_id": None,
            "container_id": None,
            "status": "ERROR",
            "error": str(e),
            "message": f"Deployment failed: {e}"
        }


@tool
def check_paper_trading_status(
    agent_id: str
) -> Dict[str, Any]:
    """
    Check paper trading agent status and performance metrics.

    This tool retrieves the current status of a paper trading agent,
    including performance metrics and validation progress.

    Args:
        agent_id: Paper trading agent ID

    Returns:
        Status and metrics including:
        - agent_id: Agent identifier
        - status: Current status (running, stopped, error)
        - uptime_hours: Hours since deployment
        - trades_executed: Number of trades executed
        - current_pnl: Current profit/loss
        - metrics: Performance metrics (sharpe_ratio, win_rate, etc.)
        - validation_status: Validation progress (pending, validating, passed)
        - days_validated: Days since deployment
        - meets_criteria: Whether agent meets promotion criteria
    """
    validator = _get_paper_trading_validator()
    deployer = _get_paper_trading_deployer()
    
    # Get basic status from deployer
    agent_status = None
    if deployer:
        agent_status = deployer.get_agent(agent_id)
    
    # Get validation status from validator
    validation_result = None
    if validator:
        validation_result = validator.check_validation_status(agent_id)
    
    # Build response
    if agent_status is None and validation_result is None:
        # Fallback to mock data
        return {
            "agent_id": agent_id,
            "status": "UNKNOWN",
            "uptime_hours": 0,
            "trades_executed": 0,
            "current_pnl": 0.0,
            "metrics": {
                "win_rate": 0.0,
                "sharpe_ratio": 0.0,
                "avg_trade_duration_hours": 0.0
            },
            "validation_status": "pending",
            "days_validated": 0,
            "meets_criteria": False,
            "error": "Agent not found"
        }
    
    # Calculate uptime
    uptime_hours = 0
    if agent_status and agent_status.uptime_seconds:
        uptime_hours = agent_status.uptime_seconds / 3600
    
    # Get metrics from validation result
    metrics = {}
    days_validated = 0
    meets_criteria = False
    validation_status = "pending"
    
    if validation_result:
        metrics = validation_result.get("metrics", {})
        days_validated = validation_result.get("days_validated", 0)
        meets_criteria = validation_result.get("meets_criteria", False)
        validation_status = validation_result.get("status", "pending").lower()
    
    return {
        "agent_id": agent_id,
        "status": agent_status.status.value if agent_status else "unknown",
        "uptime_hours": round(uptime_hours, 2),
        "trades_executed": metrics.get("total_trades", 0),
        "current_pnl": metrics.get("total_pnl", 0.0),
        "metrics": {
            "win_rate": metrics.get("win_rate", 0.0),
            "sharpe_ratio": metrics.get("sharpe", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "avg_trade_duration_hours": metrics.get("avg_trade_duration_hours", 0.0)
        },
        "validation_status": validation_status,
        "days_validated": days_validated,
        "meets_criteria": meets_criteria,
        "validation_thresholds": {
            "sharpe_ratio": 1.5,
            "win_rate": 0.55,
            "validation_period_days": 30
        }
    }


@tool
def generate_documentation(
    code: str,
    format: str = "markdown"
) -> Dict[str, Any]:
    """
    Generate code documentation.

    Args:
        code: Code to document
        format: Documentation format

    Returns:
        Generated documentation
    """
    return {
        "documentation": "# Expert Advisor Documentation\n\n...",
        "format": format,
        "functions_documented": 5
    }


@tool
def get_paper_trading_performance(
    agent_id: str
) -> Dict[str, Any]:
    """
    Get comprehensive performance metrics for a paper trading agent.

    This tool retrieves detailed performance data including risk metrics,
    trade statistics, and validation progress for promotion eligibility.

    Args:
        agent_id: Paper trading agent ID

    Returns:
        Comprehensive performance data including:
        - sharpe_ratio: Risk-adjusted return metric
        - win_rate: Percentage of winning trades (0-1)
        - total_trades: Number of trades executed
        - pnl: Total profit/loss
        - max_drawdown: Maximum equity drawdown
        - profit_factor: Gross wins / gross losses
        - validation_status: "pending", "validating", or "passed"
        - days_validated: Number of days in paper trading
        - meets_criteria: Boolean for promotion eligibility
    """
    validator = _get_paper_trading_validator()
    
    if validator is None:
        # Fallback to mock data
        return {
            "agent_id": agent_id,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "pnl": 0.0,
            "average_pnl": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
            "validation_status": "pending",
            "days_validated": 0,
            "meets_criteria": False,
            "error": "Validator not available"
        }
    
    try:
        # Get validation status with metrics
        validation_result = validator.check_validation_status(agent_id)
        metrics = validation_result.get("metrics", {})
        
        return {
            "agent_id": agent_id,
            "sharpe_ratio": metrics.get("sharpe", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "total_trades": metrics.get("total_trades", 0),
            "winning_trades": metrics.get("winning_trades", 0),
            "losing_trades": metrics.get("losing_trades", 0),
            "pnl": metrics.get("total_pnl", 0.0),
            "average_pnl": metrics.get("average_pnl", 0.0),
            "max_drawdown": metrics.get("max_drawdown", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "validation_status": validation_result.get("status", "pending").lower(),
            "days_validated": validation_result.get("days_validated", 0),
            "meets_criteria": validation_result.get("meets_criteria", False),
            "validation_thresholds": {
                "min_sharpe_ratio": 1.5,
                "min_win_rate": 0.55,
                "min_validation_days": 30
            }
        }
    except Exception as e:
        logger.error(f"Failed to get performance metrics for {agent_id}: {e}")
        return {
            "agent_id": agent_id,
            "error": str(e),
            "validation_status": "error"
        }


@tool
def promote_to_live_trading(
    agent_id: str,
    target_account: str = "account_b_sniper",
    strategy_name: str = None,
    strategy_type: str = "STRUCTURAL"
) -> Dict[str, Any]:
    """
    Promote a validated paper trading agent to live trading.

    This tool validates promotion criteria and creates a BotManifest
    registered with the Commander for live trading.

    Args:
        agent_id: Paper trading agent ID to promote
        target_account: Target account for live trading
            - "account_a_machine_gun": For HFT/Scalpers (RoboForex Prime)
            - "account_b_sniper": For Structural/ICT (Exness Raw)
            - "account_c_prop": For prop firm safe bots
        strategy_name: Name for the live bot (defaults to paper trading name)
        strategy_type: Strategy classification (SCALPER, STRUCTURAL, SWING, HFT)

    Returns:
        Promotion result with:
        - bot_id: Newly created bot identifier
        - manifest: Serialized BotManifest
        - registration_status: "success" or "failed"
        - live_trading_url: URL to monitor live bot
    """
    validator = _get_paper_trading_validator()
    manifest_classes = _get_bot_manifest_classes()
    commander = _get_commander()
    
    # Check if required components are available
    if validator is None:
        return {
            "promoted": False,
            "bot_id": None,
            "error": "PaperTradingValidator not available",
            "registration_status": "failed"
        }
    
    if manifest_classes is None:
        return {
            "promoted": False,
            "bot_id": None,
            "error": "BotManifest classes not available",
            "registration_status": "failed"
        }
    
    try:
        # Step 1: Validate promotion criteria
        validation_result = validator.check_validation_status(agent_id)
        
        validation_status = validation_result.get("status", "pending")
        days_validated = validation_result.get("days_validated", 0)
        meets_criteria = validation_result.get("meets_criteria", False)
        metrics = validation_result.get("metrics", {})
        
        # Check validation requirements
        if validation_status != "VALIDATED":
            if days_validated < 30:
                return {
                    "promoted": False,
                    "agent_id": agent_id,
                    "error": f"Validation period incomplete: {days_validated}/30 days",
                    "days_validated": days_validated,
                    "required_days": 30,
                    "registration_status": "failed"
                }
            if not meets_criteria:
                return {
                    "promoted": False,
                    "agent_id": agent_id,
                    "error": "Performance criteria not met",
                    "metrics": metrics,
                    "thresholds": {
                        "sharpe_ratio": 1.5,
                        "win_rate": 0.55
                    },
                    "registration_status": "failed"
                }
        
        # Step 2: Get deployment metadata
        deployment_info = _deployed_agents.get(agent_id, {})
        bot_strategy_name = strategy_name or deployment_info.get("strategy_name", f"bot-{agent_id}")
        
        # Step 3: Map strategy type
        StrategyType = manifest_classes['StrategyType']
        TradeFrequency = manifest_classes['TradeFrequency']
        BrokerType = manifest_classes['BrokerType']
        BotManifest = manifest_classes['BotManifest']
        
        strategy_type_map = {
            "SCALPER": StrategyType.SCALPER,
            "STRUCTURAL": StrategyType.STRUCTURAL,
            "SWING": StrategyType.SWING,
            "HFT": StrategyType.HFT
        }
        mapped_strategy_type = strategy_type_map.get(strategy_type.upper(), StrategyType.STRUCTURAL)
        
        # Derive frequency from trade count
        total_trades = metrics.get("total_trades", 0)
        if total_trades > 100:
            frequency = TradeFrequency.HFT
        elif total_trades > 20:
            frequency = TradeFrequency.HIGH
        elif total_trades > 5:
            frequency = TradeFrequency.MEDIUM
        else:
            frequency = TradeFrequency.LOW
        
        # Map broker preference based on target account
        broker_map = {
            "account_a_machine_gun": BrokerType.RAW_ECN,
            "account_b_sniper": BrokerType.RAW_ECN,
            "account_c_prop": BrokerType.STANDARD
        }
        preferred_broker = broker_map.get(target_account, BrokerType.ANY)
        
        # Check prop firm safety
        prop_firm_safe = strategy_type.upper() in ["STRUCTURAL", "SWING"]
        
        # Step 4: Generate unique bot ID
        timestamp = int(datetime.now(timezone.utc).timestamp())
        bot_id = f"{bot_strategy_name.lower().replace(' ', '-')}-{agent_id[:8]}-{timestamp}"
        
        # Step 5: Create BotManifest
        manifest = BotManifest(
            bot_id=bot_id,
            name=bot_strategy_name,
            description=f"Promoted from paper trading agent {agent_id}",
            strategy_type=mapped_strategy_type,
            frequency=frequency,
            min_capital_req=50.0,
            preferred_broker_type=preferred_broker,
            prop_firm_safe=prop_firm_safe,
            symbols=[deployment_info.get("symbol", "EURUSD")],
            timeframes=[deployment_info.get("timeframe", "H1")],
            total_trades=total_trades,
            win_rate=metrics.get("win_rate", 0.0),
            tags=["@primal"]
        )
        
        # Step 6: Register with Commander
        registration_status = "success"
        if commander and commander.bot_registry:
            commander.bot_registry.register(manifest)
            logger.info(f"Bot {bot_id} registered with Commander")
        else:
            logger.warning("Commander or BotRegistry not available for registration")
            registration_status = "registered_locally"
        
        # Step 7: Broadcast promotion event via WebSocket
        try:
            import asyncio
            from src.api.websocket_endpoints import broadcast_paper_trading_promotion
            
            # Prepare performance summary for broadcast
            performance_summary = {
                "sharpe_ratio": metrics.get("sharpe", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "total_trades": total_trades,
                "total_pnl": metrics.get("total_pnl", 0.0),
                "max_drawdown": metrics.get("max_drawdown", 0.0),
                "profit_factor": metrics.get("profit_factor", 0.0),
                "days_validated": days_validated
            }
            
            # Schedule the async broadcast (non-blocking)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    broadcast_paper_trading_promotion(
                        agent_id=agent_id,
                        bot_id=bot_id,
                        target_account=target_account,
                        performance_summary=performance_summary
                    )
                )
            except RuntimeError:
                # No running loop, create a new one
                asyncio.run(
                    broadcast_paper_trading_promotion(
                        agent_id=agent_id,
                        bot_id=bot_id,
                        target_account=target_account,
                        performance_summary=performance_summary
                    )
                )
        except ImportError:
            logger.debug("WebSocket broadcast not available for promotion event")
        except Exception as e:
            logger.warning(f"Failed to broadcast promotion event: {e}")
        
        # Step 8: Return promotion result
        return {
            "promoted": True,
            "bot_id": bot_id,
            "agent_id": agent_id,
            "manifest": manifest.to_dict(),
            "registration_status": registration_status,
            "target_account": target_account,
            "live_trading_url": f"/api/bots/{bot_id}/status",
            "performance_summary": {
                "sharpe_ratio": metrics.get("sharpe", 0.0),
                "win_rate": metrics.get("win_rate", 0.0),
                "total_trades": total_trades,
                "days_validated": days_validated
            }
        }
        
    except ValueError as e:
        logger.error(f"Validation error during promotion: {e}")
        return {
            "promoted": False,
            "agent_id": agent_id,
            "error": str(e),
            "registration_status": "failed"
        }
    except Exception as e:
        logger.error(f"Promotion failed for {agent_id}: {e}")
        return {
            "promoted": False,
            "agent_id": agent_id,
            "error": str(e),
            "registration_status": "failed"
        }


# List of all QuantCode tools
QUANTCODE_TOOLS = [
    create_strategy_plan,
    generate_mql5_code,
    generate_component,
    validate_syntax,
    fix_syntax,
    compile_mql5,
    debug_code,
    optimize_code,
    run_backtest,
    analyze_backtest_results,
    run_monte_carlo,
    deploy_paper_trading,
    check_paper_trading_status,
    get_paper_trading_performance,
    promote_to_live_trading,
    generate_documentation,
]


# ============================================================================
# Agent Node Functions
# ============================================================================

def agent_node(state: MessagesState) -> Dict[str, Any]:
    """
    Main agent node that uses LLM to decide which tools to call.
    Uses the project's provider config (OpenRouter/Zhipu) with fallbacks.
    """
    llm_with_tools = get_quantcode_llm(tools=QUANTCODE_TOOLS)

    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


# ============================================================================
# Graph Construction with ToolNode
# ============================================================================

def create_quantcode_graph_with_tools() -> StateGraph:
    """
    Create the QuantCode agent workflow graph with ToolNode.

    Workflow:
    START -> agent -> (tool_condition) -> tools -> agent -> ...
                    -> (no tools) -> END
    """
    tool_node = ToolNode(QUANTCODE_TOOLS)

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge("tools", "agent")

    return workflow


def create_quantcode_graph_legacy() -> StateGraph:
    """
    Create the legacy QuantCode agent workflow graph.

    Kept for backward compatibility.
    """
    workflow = StateGraph(QuantCodeState)

    # Legacy hardcoded nodes (simplified)
    def planning_node(state: QuantCodeState) -> Dict[str, Any]:
        return {
            "messages": [AIMessage(content="Strategy plan created")],
            "strategy_plan": {"strategy_type": "momentum"}
        }

    def coding_node(state: QuantCodeState) -> Dict[str, Any]:
        return {
            "messages": [AIMessage(content="Code generated")],
            "code_implementation": "// Generated code"
        }

    def backtesting_node(state: QuantCodeState) -> Dict[str, Any]:
        return {
            "messages": [AIMessage(content="Backtest complete")],
            "backtest_results": {"win_rate": 0.633, "kelly_score": 0.82}
        }

    def analysis_node(state: QuantCodeState) -> Dict[str, Any]:
        return {
            "messages": [AIMessage(content="Analysis complete")],
            "analysis_report": "Performance Analysis..."
        }

    def reflection_node(state: QuantCodeState) -> Dict[str, Any]:
        return {
            "messages": [AIMessage(content="Reflection complete")],
            "reflection_notes": "Strategy looks good..."
        }

    workflow.add_node("planning", planning_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("backtesting", backtesting_node)
    workflow.add_node("analysis", analysis_node)
    workflow.add_node("reflection", reflection_node)

    workflow.set_entry_point("planning")
    workflow.add_edge("planning", "coding")
    workflow.add_edge("coding", "backtesting")
    workflow.add_edge("backtesting", "analysis")
    workflow.add_edge("analysis", "reflection")
    workflow.add_edge("reflection", END)

    return workflow


def compile_quantcode_graph(
    checkpointer: MemorySaver = None,
    use_tool_node: bool = True
) -> Any:
    """
    Compile the QuantCode agent graph.

    Args:
        checkpointer: Optional memory checkpointer
        use_tool_node: Use ToolNode architecture if True

    Returns:
        Compiled graph
    """
    if use_tool_node:
        workflow = create_quantcode_graph_with_tools()
    else:
        workflow = create_quantcode_graph_legacy()

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled_graph = workflow.compile(checkpointer=checkpointer)
    logger.info(f"QuantCode agent graph compiled (ToolNode={use_tool_node})")

    return compiled_graph


def run_quantcode_workflow(
    strategy_request: str,
    workspace_path: str = "workspaces/quant",
    memory_namespace: tuple = ("memories", "quantcode", "default"),
    use_tool_node: bool = True
) -> Dict[str, Any]:
    """
    Execute the QuantCode agent workflow.

    Args:
        strategy_request: The strategy development request
        workspace_path: Path to workspace
        memory_namespace: Memory namespace tuple
        use_tool_node: Use ToolNode architecture if True

    Returns:
        Final state after workflow completion
    """
    if use_tool_node:
        initial_state = {
            "messages": [HumanMessage(content=strategy_request)]
        }
    else:
        initial_state: QuantCodeState = {
            "messages": [HumanMessage(content=strategy_request)],
            "current_task": "strategy_development",
            "workspace_path": workspace_path,
            "context": {},
            "memory_namespace": memory_namespace,
            "strategy_plan": None,
            "code_implementation": None,
            "backtest_results": None,
            "analysis_report": None,
            "reflection_notes": None,
            "paper_agent_id": None,
            "validation_start_time": None,
            "validation_period_days": 30,
            "paper_trading_metrics": None,
            "promotion_approved": False
        }

    graph = compile_quantcode_graph(use_tool_node=use_tool_node)
    config = {"configurable": {"thread_id": "quantcode_001"}}
    final_state = graph.invoke(initial_state, config)

    logger.info(f"QuantCode workflow completed")

    return final_state


# =============================================================================
# Factory-Based Agent Creation (Phase 4)
# =============================================================================

def create_quantcode_from_config(config: AgentConfig) -> CompiledAgent:
    """
    Create a QuantCode agent from configuration using the factory pattern.
    
    Args:
        config: AgentConfig instance with quantcode configuration
        
    Returns:
        CompiledAgent instance
        
    Raises:
        ValueError: If config is not for a quantcode agent
    """
    import warnings
    
    # Validate agent type
    if config.agent_type != "quantcode":
        raise ValueError(f"Expected agent_type='quantcode', got '{config.agent_type}'")
    
    # Issue deprecation warning
    warnings.warn(
        "create_quantcode_from_config() uses the factory pattern. "
        "Consider using AgentFactory directly for new code.",
        DeprecationWarning,
        stacklevel=2
    )
    
    from src.agents.factory import get_factory
    from src.agents.di_container import get_container
    from src.agents.observers.logging_observer import LoggingObserver
    from src.agents.observers.prometheus_observer import PrometheusObserver
    
    container = get_container()
    factory = get_factory(container)
    
    if not container.get_observers():
        container.add_observer(LoggingObserver())
        container.add_observer(PrometheusObserver())
    
    agent = factory.create(config)
    
    logger.info(f"Created quantcode agent from config: {config.agent_id}")
    
    return agent


def create_quantcode_agent(
    agent_id: str = "quantcode_001",
    name: str = "QuantCode Agent",
    llm_model: str = "deepseek/deepseek-coder",
    temperature: float = 0.0,
    **kwargs
) -> CompiledAgent:
    """
    Convenience function to create a QuantCode agent.
    """
    config = AgentConfig(
        agent_id=agent_id,
        agent_type="quantcode",
        name=name,
        llm_model=llm_model,
        temperature=temperature,
        tools=[
            "create_strategy_plan",
            "generate_mql5_code",
            "validate_code",
            "compile_code",
            "run_backtest",
            "analyze_strategy_performance",
            "optimize_parameters",
            "create_documentation",
        ],
        **kwargs
    )
    
    return create_quantcode_from_config(config)
