"""
MT5 Compiler MCP Tools Module.

Provides tools for MQL5 code compilation and validation via MT5 Compiler MCP server.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from src.agents.tools.mcp.manager import get_mcp_manager

logger = logging.getLogger(__name__)


async def compile_mql5_code(
    code: str,
    filename: str,
    code_type: str = "expert"
) -> Dict[str, Any]:
    """
    Compile MQL5 code using MT5 Compiler MCP.

    This tool compiles MQL5 code and returns compilation results
    including any errors or warnings.

    Args:
        code: MQL5 source code to compile
        filename: Output filename (without extension)
        code_type: Type of code ("expert", "indicator", "script", "library")

    Returns:
        Dictionary containing:
        - success: Whether compilation succeeded
        - errors: List of compilation errors
        - warnings: List of compilation warnings
        - output_path: Path to compiled file (if successful)
    """
    logger.info(f"Compiling MQL5 code: {filename}")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "metatrader5",
            "compile",
            {
                "code": code,
                "filename": filename,
                "code_type": code_type
            }
        )

        if isinstance(result, dict):
            return {
                "success": result.get("success", False),
                "filename": filename,
                "code_type": code_type,
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),
                "output_path": result.get("output_path", f"/MT5/MQL5/Experts/{filename}.ex5"),
                "compiled_at": result.get("compiled_at", datetime.now().isoformat())
            }
        return {
            "success": False,
            "filename": filename,
            "code_type": code_type,
            "errors": ["Unknown compilation result"],
            "warnings": [],
            "output_path": None,
            "compiled_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        raise RuntimeError(f"Failed to compile MQL5 code: {e}")


async def validate_mql5_syntax(code: str) -> Dict[str, Any]:
    """
    Validate MQL5 code syntax without full compilation.

    Args:
        code: MQL5 source code to validate

    Returns:
        Dictionary containing validation results
    """
    logger.info("Validating MQL5 syntax")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "metatrader5",
            "validate-syntax",
            {"code": code}
        )

        if isinstance(result, dict):
            return {
                "valid": result.get("valid", False),
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),
                "suggestions": result.get("suggestions", [])
            }
        return {
            "valid": False,
            "errors": ["Unknown validation result"],
            "warnings": [],
            "suggestions": []
        }

    except Exception as e:
        logger.error(f"Syntax validation failed: {e}")
        raise RuntimeError(f"Failed to validate MQL5 syntax: {e}")


async def get_compilation_errors(
    error_codes: List[str]
) -> Dict[str, Any]:
    """
    Get detailed information about MQL5 compilation errors.

    Args:
        error_codes: List of MQL5 error codes

    Returns:
        Dictionary containing error details and suggested fixes
    """
    logger.info(f"Getting details for {len(error_codes)} error codes")

    manager = await get_mcp_manager()

    try:
        result = await manager.call_tool(
            "metatrader5",
            "get-error-details",
            {"error_codes": error_codes}
        )

        if isinstance(result, dict):
            return {
                "errors": result.get("errors", [])
            }
        return {"errors": []}

    except Exception as e:
        logger.error(f"Failed to get error details: {e}")
        raise RuntimeError(f"Failed to get compilation error details: {e}")
