"""
Strategies-YT Tools Integration Layer

Exposes strategies-yt pipeline tools as copilot-compatible tools.
Provides SDK-compatible tool definitions for the agent system.

Modules:
- trd_tools: TRD generation, validation, and config conversion
- video_analysis_tools: Video analysis and strategy element extraction
- strategy_tools: Strategy creation, backtesting, and deployment
- zmq_tools: ZMQ communication with Strategy Router

Usage:
    from src.agents.tools.strategies_yt import (
        list_strategies_yt_tools,
        get_strategies_yt_tool,
        STRATEGIES_YT_TOOLS
    )

    # Get tool by name
    tool = get_strategies_yt_tool("generate_trd_from_video")

    # Invoke tool
    result = await tool["function"](video_id="abc123")

    # List all tools
    all_tools = list_strategies_yt_tools()
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from .trd_tools import (
    generate_trd_from_video,
    validate_trd,
    trd_to_config,
    list_trds,
    TRD_TOOLS,
    TruthObject,
    TRDValidationResult
)

from .video_analysis_tools import (
    analyze_trading_video,
    analyze_playlist,
    extract_indicators,
    extract_entry_rules,
    extract_exit_rules,
    extract_risk_parameters,
    VIDEO_ANALYSIS_TOOLS,
    VideoClip,
    VideoAnalysisResult,
    IndicatorExtraction,
    RuleExtraction,
    RiskParameterExtraction
)

from .strategy_tools import (
    create_strategy_from_trd,
    backtest_strategy,
    deploy_strategy,
    list_strategies,
    STRATEGY_TOOLS,
    BacktestConfig,
    BacktestResult,
    DeploymentConfig,
    DeploymentResult,
    StrategyStatus,
    TradingMode
)

from .zmq_tools import (
    send_to_router,
    receive_from_router,
    register_ea,
    send_heartbeat,
    ZMQ_TOOLS,
    ZMQMessage,
    ZMQResponse,
    ZMQClient,
    MessageType,
    ConnectionStatus
)

logger = logging.getLogger(__name__)


# =============================================================================
# COMBINED TOOL REGISTRY
# =============================================================================

STRATEGIES_YT_TOOLS: Dict[str, Dict[str, Any]] = {
    **TRD_TOOLS,
    **VIDEO_ANALYSIS_TOOLS,
    **STRATEGY_TOOLS,
    **ZMQ_TOOLS
}


# =============================================================================
# TOOL CATEGORIES
# =============================================================================

TRD_TOOL_NAMES = list(TRD_TOOLS.keys())
VIDEO_ANALYSIS_TOOL_NAMES = list(VIDEO_ANALYSIS_TOOLS.keys())
STRATEGY_TOOL_NAMES = list(STRATEGY_TOOLS.keys())
ZMQ_TOOL_NAMES = list(ZMQ_TOOLS.keys())


# =============================================================================
# PUBLIC API
# =============================================================================

def list_strategies_yt_tools() -> List[str]:
    """
    List all available strategies-yt tools.

    Returns:
        List of tool names
    """
    return list(STRATEGIES_YT_TOOLS.keys())


def get_strategies_yt_tool(name: str) -> Optional[Dict[str, Any]]:
    """
    Get a strategies-yt tool by name.

    Args:
        name: Tool name

    Returns:
        Tool definition dictionary or None if not found
    """
    return STRATEGIES_YT_TOOLS.get(name)


def get_trd_tools() -> List[str]:
    """List all TRD-related tools."""
    return TRD_TOOL_NAMES


def get_video_analysis_tools() -> List[str]:
    """List all video analysis tools."""
    return VIDEO_ANALYSIS_TOOL_NAMES


def get_strategy_tools() -> List[str]:
    """List all strategy management tools."""
    return STRATEGY_TOOL_NAMES


def get_zmq_tools() -> List[str]:
    """List all ZMQ communication tools."""
    return ZMQ_TOOL_NAMES


def get_tools_by_category(category: str) -> List[str]:
    """
    Get tools by category.

    Args:
        category: Category name (trd, video_analysis, strategy, zmq)

    Returns:
        List of tool names in category
    """
    categories = {
        "trd": TRD_TOOL_NAMES,
        "video_analysis": VIDEO_ANALYSIS_TOOL_NAMES,
        "video": VIDEO_ANALYSIS_TOOL_NAMES,
        "strategy": STRATEGY_TOOL_NAMES,
        "zmq": ZMQ_TOOL_NAMES,
        "communication": ZMQ_TOOL_NAMES
    }
    return categories.get(category.lower(), [])


async def invoke_tool(name: str, **kwargs) -> Dict[str, Any]:
    """
    Invoke a strategies-yt tool by name.

    Args:
        name: Tool name
        **kwargs: Tool parameters

    Returns:
        Tool result dictionary
    """
    tool = get_strategies_yt_tool(name)
    if not tool:
        return {
            "success": False,
            "error": f"Unknown tool: {name}"
        }

    try:
        func = tool["function"]
        if asyncio.iscoroutinefunction(func):
            return await func(**kwargs)
        else:
            return func(**kwargs)
    except Exception as e:
        logger.error(f"Error invoking tool {name}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "tool": name
        }


# =============================================================================
# SDK COMPATIBILITY
# =============================================================================

def get_sdk_tool_definitions() -> List[Dict[str, Any]]:
    """
    Get all tools in SDK-compatible format.

    Returns:
        List of tool definitions compatible with agent SDK
    """
    tools = []

    for name, tool_def in STRATEGIES_YT_TOOLS.items():
        parameters = tool_def.get("parameters", {})

        # Convert to SDK format
        sdk_tool = {
            "name": name,
            "description": tool_def.get("description", ""),
            "input_schema": {
                "type": "object",
                "properties": {
                    k: {
                        "type": v.get("type", "string"),
                        "description": k.replace("_", " ").title()
                    }
                    for k, v in parameters.items()
                },
                "required": [
                    k for k, v in parameters.items()
                    if v.get("required", False)
                ]
            }
        }
        tools.append(sdk_tool)

    return tools


# =============================================================================
# WORKFLOW HELPERS
# =============================================================================

async def video_to_strategy_workflow(
    video_id: str,
    generate_code: bool = True,
    register: bool = True
) -> Dict[str, Any]:
    """
    Complete workflow: Video -> TRD -> Strategy -> Registration.

    Args:
        video_id: YouTube video ID
        generate_code: Generate MQL5 EA code
        register: Register BotManifest

    Returns:
        Workflow result with all artifacts
    """
    logger.info(f"Starting video-to-strategy workflow: {video_id}")

    artifacts = {
        "video_id": video_id,
        "steps": []
    }

    try:
        # Step 1: Analyze video
        logger.info("Step 1: Analyzing video...")
        analysis_result = await analyze_trading_video(video_id)
        artifacts["steps"].append({"step": "analyze_video", "result": analysis_result})

        if not analysis_result.get("success"):
            return {
                "success": False,
                "error": "Video analysis failed",
                "artifacts": artifacts
            }

        # Step 2: Generate TRD
        logger.info("Step 2: Generating TRD...")
        trd_result = await generate_trd_from_video(video_id)
        artifacts["steps"].append({"step": "generate_trd", "result": trd_result})

        if not trd_result.get("success"):
            return {
                "success": False,
                "error": "TRD generation failed",
                "artifacts": artifacts
            }

        trd_path = trd_result.get("trd_path")
        artifacts["trd_path"] = trd_path

        # Step 3: Create strategy
        logger.info("Step 3: Creating strategy...")
        strategy_result = await create_strategy_from_trd(
            trd_path,
            generate_mql5=generate_code,
            register_manifest=register
        )
        artifacts["steps"].append({"step": "create_strategy", "result": strategy_result})

        if not strategy_result.get("success"):
            return {
                "success": False,
                "error": "Strategy creation failed",
                "artifacts": artifacts
            }

        artifacts["strategy_id"] = strategy_result.get("strategy_id")
        artifacts["strategy_dir"] = strategy_result.get("strategy_dir")
        artifacts["mql5_path"] = strategy_result.get("mql5_path")
        artifacts["manifest_path"] = strategy_result.get("manifest_path")

        logger.info(f"Workflow completed successfully: {artifacts.get('strategy_id')}")

        return {
            "success": True,
            "message": "Video-to-strategy workflow completed",
            "artifacts": artifacts
        }

    except Exception as e:
        logger.error(f"Workflow error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "artifacts": artifacts
        }


async def validate_and_deploy_workflow(
    strategy_id: str,
    run_backtest: bool = True,
    deploy_to_paper: bool = True
) -> Dict[str, Any]:
    """
    Complete workflow: Validate -> Backtest -> Deploy.

    Args:
        strategy_id: Strategy identifier
        run_backtest: Run backtest before deployment
        deploy_to_paper: Deploy to paper trading

    Returns:
        Workflow result
    """
    logger.info(f"Starting validation workflow: {strategy_id}")

    artifacts = {
        "strategy_id": strategy_id,
        "steps": []
    }

    try:
        # Step 1: List strategy info
        strategies = await list_strategies()
        strategy_info = next(
            (s for s in strategies.get("strategies", []) if s["strategy_id"] == strategy_id),
            None
        )

        if not strategy_info:
            return {
                "success": False,
                "error": f"Strategy not found: {strategy_id}",
                "artifacts": artifacts
            }

        artifacts["strategy_info"] = strategy_info

        # Step 2: Run backtest
        if run_backtest:
            logger.info("Step 2: Running backtest...")
            backtest_config = BacktestConfig(
                symbol=strategy_info["symbols"][0] if strategy_info["symbols"] else "EURUSD",
                timeframe=strategy_info.get("timeframes", ["H1"])[0] if strategy_info.get("timeframes") else "H1",
                start_date="2023-01-01",
                end_date="2024-01-01"
            )

            backtest_result = await backtest_strategy(strategy_id, backtest_config)
            artifacts["steps"].append({"step": "backtest", "result": backtest_result})
            artifacts["backtest_results"] = backtest_result.get("results")

        # Step 3: Deploy to paper
        if deploy_to_paper:
            logger.info("Step 3: Deploying to paper trading...")
            deployment_config = DeploymentConfig(
                strategy_id=strategy_id,
                mode=TradingMode.PAPER,
                account_id="paper_demo",
                symbols=strategy_info["symbols"],
                max_positions=3
            )

            deployment_result = await deploy_strategy(strategy_id, deployment_config)
            artifacts["steps"].append({"step": "deploy", "result": deployment_result})
            artifacts["deployment"] = deployment_result.get("deployment")

        return {
            "success": True,
            "message": "Validation workflow completed",
            "artifacts": artifacts
        }

    except Exception as e:
        logger.error(f"Validation workflow error: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "artifacts": artifacts
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Tool lists
    "STRATEGIES_YT_TOOLS",
    "TRD_TOOL_NAMES",
    "VIDEO_ANALYSIS_TOOL_NAMES",
    "STRATEGY_TOOL_NAMES",
    "ZMQ_TOOL_NAMES",

    # Accessor functions
    "list_strategies_yt_tools",
    "get_strategies_yt_tool",
    "get_trd_tools",
    "get_video_analysis_tools",
    "get_strategy_tools",
    "get_zmq_tools",
    "get_tools_by_category",
    "invoke_tool",

    # SDK compatibility
    "get_sdk_tool_definitions",

    # Workflows
    "video_to_strategy_workflow",
    "validate_and_deploy_workflow",

    # TRD tools
    "generate_trd_from_video",
    "validate_trd",
    "trd_to_config",
    "list_trds",
    "TruthObject",
    "TRDValidationResult",

    # Video analysis tools
    "analyze_trading_video",
    "extract_indicators",
    "extract_entry_rules",
    "extract_exit_rules",
    "extract_risk_parameters",
    "VideoClip",
    "VideoAnalysisResult",
    "IndicatorExtraction",
    "RuleExtraction",
    "RiskParameterExtraction",

    # Strategy tools
    "create_strategy_from_trd",
    "backtest_strategy",
    "deploy_strategy",
    "list_strategies",
    "BacktestConfig",
    "BacktestResult",
    "DeploymentConfig",
    "DeploymentResult",
    "StrategyStatus",
    "TradingMode",

    # ZMQ tools
    "send_to_router",
    "receive_from_router",
    "register_ea",
    "send_heartbeat",
    "ZMQMessage",
    "ZMQResponse",
    "ZMQClient",
    "MessageType",
    "ConnectionStatus"
]
