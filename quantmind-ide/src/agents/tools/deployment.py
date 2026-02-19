"""
Deployment Tools for QuantMind agents.

These tools provide EA deployment and management:
- deploy_ea: Compile and deploy EA to MT5 terminal
- stop_ea: Stop running EA
- get_ea_status: Get EA runtime status
- compile_ea: Compile EA without deploying
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .base import (
    QuantMindTool,
    ToolCategory,
    ToolError,
    ToolPriority,
    ToolResult,
    register_tool,
)
from .registry import AgentType


logger = logging.getLogger(__name__)


class DeploymentStatus(str, Enum):
    """EA deployment status."""
    IDLE = "idle"
    COMPILING = "compiling"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class CompilationResult:
    """Result of EA compilation."""
    success: bool
    errors: List[str]
    warnings: List[str]
    output_path: Optional[str] = None
    compile_time_ms: float = 0


class DeployEAInput(BaseModel):
    """Input schema for deploy_ea tool."""
    ea_path: str = Field(
        description="Path to the EA source file (.mq5) or compiled file (.ex5)"
    )
    terminal_id: Optional[str] = Field(
        default=None,
        description="Target MT5 terminal ID (uses default if not specified)"
    )
    symbol: str = Field(
        default="EURUSD",
        description="Trading symbol for the EA"
    )
    timeframe: str = Field(
        default="H1",
        description="Chart timeframe (M1, M5, M15, M30, H1, H4, D1, W1, MN)"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="EA input parameters"
    )
    auto_start: bool = Field(
        default=True,
        description="Whether to start the EA automatically after deployment"
    )


class StopEAInput(BaseModel):
    """Input schema for stop_ea tool."""
    ea_name: Optional[str] = Field(
        default=None,
        description="Name of the EA to stop (stops all if not specified)"
    )
    terminal_id: Optional[str] = Field(
        default=None,
        description="MT5 terminal ID (uses default if not specified)"
    )
    force: bool = Field(
        default=False,
        description="Force stop even if EA is in mid-operation"
    )


class GetEaStatusInput(BaseModel):
    """Input schema for get_ea_status tool."""
    ea_name: Optional[str] = Field(
        default=None,
        description="Name of the EA to check (checks all if not specified)"
    )
    terminal_id: Optional[str] = Field(
        default=None,
        description="MT5 terminal ID (uses default if not specified)"
    )


class CompileEAInput(BaseModel):
    """Input schema for compile_ea tool."""
    ea_path: str = Field(
        description="Path to the EA source file (.mq5)"
    )
    log_level: str = Field(
        default="warning",
        description="Compiler log level (quiet, warning, info, debug)"
    )


@register_tool(
    agent_types=[AgentType.COPILOT],
    tags=["deployment", "mt5", "ea"],
)
class DeployEATool(QuantMindTool):
    """Compile and deploy an EA to MT5 terminal."""

    name: str = "deploy_ea"
    description: str = """Compile and deploy an Expert Advisor to MetaTrader 5.
    If source file (.mq5) is provided, it will be compiled first.
    Deploys to specified terminal and optionally starts the EA.
    Returns deployment status and any compilation results."""

    args_schema: type[BaseModel] = DeployEAInput
    category: ToolCategory = ToolCategory.DEPLOYMENT
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        ea_path: str,
        terminal_id: Optional[str] = None,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        parameters: Optional[Dict[str, Any]] = None,
        auto_start: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute EA deployment."""
        # Validate EA path
        validated_path = self.validate_workspace_path(ea_path)

        if not validated_path.exists():
            raise ToolError(
                f"EA file '{ea_path}' does not exist",
                tool_name=self.name,
                error_code="EA_NOT_FOUND"
            )

        # Check if compilation needed
        if validated_path.suffix == ".mq5":
            # Compile first
            compile_result = self._compile_ea(validated_path)
            if not compile_result.success:
                return ToolResult.error(
                    error="Compilation failed",
                    metadata={
                        "compilation_errors": compile_result.errors,
                        "compilation_warnings": compile_result.warnings,
                    }
                )
            ex5_path = compile_result.output_path
        else:
            ex5_path = str(validated_path)

        # Deploy to MT5 (would call MCP server in production)
        deployment_result = self._deploy_to_mt5(
            ex5_path=ex5_path,
            terminal_id=terminal_id,
            symbol=symbol,
            timeframe=timeframe,
            parameters=parameters,
            auto_start=auto_start,
        )

        return ToolResult.ok(
            data={
                "status": deployment_result["status"],
                "ea_path": ea_path,
                "ex5_path": ex5_path,
                "terminal_id": deployment_result.get("terminal_id"),
            },
            metadata={
                "symbol": symbol,
                "timeframe": timeframe,
                "auto_start": auto_start,
                "deployed_at": datetime.now().isoformat(),
            }
        )

    def _compile_ea(self, source_path: Path) -> CompilationResult:
        """Compile EA source file."""
        # In production, this would call MT5 compiler via MCP
        # For now, simulate compilation
        logger.info(f"Compiling EA: {source_path}")

        # Simulate compilation success
        ex5_path = source_path.with_suffix(".ex5")

        return CompilationResult(
            success=True,
            errors=[],
            warnings=[],
            output_path=str(ex5_path),
            compile_time_ms=1500
        )

    def _deploy_to_mt5(
        self,
        ex5_path: str,
        terminal_id: Optional[str],
        symbol: str,
        timeframe: str,
        parameters: Optional[Dict[str, Any]],
        auto_start: bool,
    ) -> Dict[str, Any]:
        """Deploy compiled EA to MT5 terminal."""
        # In production, this would call MCP MT5 server
        # For now, simulate deployment
        logger.info(f"Deploying EA to MT5: {ex5_path}")

        return {
            "status": DeploymentStatus.RUNNING.value if auto_start else DeploymentStatus.DEPLOYING.value,
            "terminal_id": terminal_id or "default",
            "chart_id": "chart_001",
        }


@register_tool(
    agent_types=[AgentType.COPILOT],
    tags=["deployment", "mt5", "ea", "stop"],
)
class StopEATool(QuantMindTool):
    """Stop a running EA in MT5 terminal."""

    name: str = "stop_ea"
    description: str = """Stop a running Expert Advisor in MetaTrader 5.
    Can stop a specific EA by name or all EAs.
    Optionally force stop if EA is mid-operation."""

    args_schema: type[BaseModel] = StopEAInput
    category: ToolCategory = ToolCategory.DEPLOYMENT
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        ea_name: Optional[str] = None,
        terminal_id: Optional[str] = None,
        force: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute EA stop operation."""
        # In production, this would call MCP MT5 server
        logger.info(f"Stopping EA: {ea_name or 'all'}")

        return ToolResult.ok(
            data={
                "stopped": True,
                "ea_name": ea_name,
                "terminal_id": terminal_id or "default",
            },
            metadata={
                "force": force,
                "stopped_at": datetime.now().isoformat(),
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT],
    tags=["deployment", "mt5", "ea", "status"],
)
class GetEaStatusTool(QuantMindTool):
    """Get EA runtime status from MT5 terminal."""

    name: str = "get_ea_status"
    description: str = """Get the runtime status of Expert Advisors in MetaTrader 5.
    Can check a specific EA or all EAs.
    Returns status, performance metrics, and any error information."""

    args_schema: type[BaseModel] = GetEaStatusInput
    category: ToolCategory = ToolCategory.DEPLOYMENT
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        ea_name: Optional[str] = None,
        terminal_id: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute EA status check."""
        # In production, this would call MCP MT5 server
        logger.info(f"Getting EA status: {ea_name or 'all'}")

        # Simulate status response
        if ea_name:
            status_data = {
                "name": ea_name,
                "status": DeploymentStatus.RUNNING.value,
                "symbol": "EURUSD",
                "timeframe": "H1",
                "running_since": datetime.now().isoformat(),
                "trades_today": 5,
                "profit": 125.50,
            }
        else:
            status_data = {
                "eas": [
                    {
                        "name": "MovingAverage",
                        "status": DeploymentStatus.RUNNING.value,
                        "symbol": "EURUSD",
                    },
                    {
                        "name": "RSI_Trader",
                        "status": DeploymentStatus.STOPPED.value,
                        "symbol": "GBPUSD",
                    }
                ]
            }

        return ToolResult.ok(
            data=status_data,
            metadata={
                "terminal_id": terminal_id or "default",
                "checked_at": datetime.now().isoformat(),
            }
        )


@register_tool(
    agent_types=[AgentType.COPILOT, AgentType.QUANTCODE],
    tags=["deployment", "mt5", "compile", "ea"],
)
class CompileEATool(QuantMindTool):
    """Compile EA source file without deploying."""

    name: str = "compile_ea"
    description: str = """Compile an Expert Advisor source file (.mq5) to .ex5.
    Returns compilation result with any errors or warnings.
    Does not deploy the compiled EA."""

    args_schema: type[BaseModel] = CompileEAInput
    category: ToolCategory = ToolCategory.DEPLOYMENT
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        ea_path: str,
        log_level: str = "warning",
        **kwargs
    ) -> ToolResult:
        """Execute EA compilation."""
        # Validate EA path
        validated_path = self.validate_workspace_path(ea_path)

        if not validated_path.exists():
            raise ToolError(
                f"EA file '{ea_path}' does not exist",
                tool_name=self.name,
                error_code="EA_NOT_FOUND"
            )

        if validated_path.suffix != ".mq5":
            raise ToolError(
                f"File '{ea_path}' is not a source file (.mq5 required)",
                tool_name=self.name,
                error_code="INVALID_FILE_TYPE"
            )

        # Compile (in production, would call MT5 compiler)
        logger.info(f"Compiling EA: {ea_path}")

        # Simulate compilation
        ex5_path = validated_path.with_suffix(".ex5")

        result = CompilationResult(
            success=True,
            errors=[],
            warnings=["Variable 'unused_var' is declared but never used"],
            output_path=str(ex5_path),
            compile_time_ms=1500
        )

        return ToolResult.ok(
            data={
                "success": result.success,
                "output_path": result.output_path,
                "errors": result.errors,
                "warnings": result.warnings,
            },
            metadata={
                "source_path": ea_path,
                "compile_time_ms": result.compile_time_ms,
                "log_level": log_level,
            }
        )


# Export all tools
__all__ = [
    "DeployEATool",
    "StopEATool",
    "GetEaStatusTool",
    "CompileEATool",
    "DeploymentStatus",
    "CompilationResult",
]
