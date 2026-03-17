"""
Pine Script Agent - DEPRECATED

Use floor_manager /api/floor-manager endpoints instead.
This module used LangGraph which has been removed.

**Validates: Property 17: Pine Script Agent**

NOTE: This file is now a STUB pending migration to Anthropic Agent SDK (Epic 7).
All LangGraph/LangChain imports have been removed.
"""

import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Deprecation warning
logger.warning(
    "pinescript.py is deprecated. "
    "Use /api/floor-manager endpoints with 'development' department instead."
)


# Stub for AgentState
class AgentState(dict):
    """Stub for langgraph AgentState - pending migration to Anthropic Agent SDK."""
    pass


# Stub for END
END = "END"


def get_llm(*args, **kwargs):
    """Deprecated - use floor_manager instead."""
    raise NotImplementedError(
        "pinescript module is deprecated. Use /api/floor-manager instead."
    )


# Import Pine Script tools for validation and conversion (these don't need langchain)
from src.agents.tools.pinescript_tools import (
    validate_pine_script_syntax,
    validate_pine_script_strategy,
    convert_mql5_to_pine,
    extract_pine_indicators,
    PINESCRIPT_TOOLS,
)


# Pine Script v5 system prompt (kept for reference)
PINESCRIPT_SYSTEM_PROMPT = """You are an expert Pine Script v5 developer for TradingView.

Your task is to generate clean, efficient, and well-documented Pine Script v5 code
based on the user's strategy description.

## Code Style Guidelines:
1. Always use `indicator()` or `strategy()` declaration at the top
2. Use Pine Script v5 syntax only (no legacy syntax)
3. Add descriptive comments for each section
4. Use meaningful variable names
5. Include input parameters for customization
6. Add proper stop-loss and take-profit levels
7. Implement proper entry and exit conditions
8. Use `plot()` for visual confirmation

## Important Notes:
- Never use `varip` unless absolutely necessary
- Avoid repainting indicators
- Use `request.security()` carefully for multi-timeframe analysis
- Always test strategies thoroughly before live trading
"""


class PineScriptState(AgentState):
    """
    State for Pine Script agent workflow.

    **Validates: Requirements 8.7**
    """
    user_query: Optional[str]
    strategy_description: Optional[str]
    pine_script_code: Optional[str]
    validation_errors: List[str]
    status: str  # pending, generating, validating, fixing, complete, error
    mql5_source: Optional[str]  # For MQL5 to Pine Script conversion
    conversion_mode: str  # 'generate' or 'convert'


def generate_pine_script(state: PineScriptState) -> Dict[str, Any]:
    """
    STUB - Generation pending migration to Anthropic Agent SDK.
    """
    logger.warning("PineScript generate_pine_script is a stub - pending Epic 7")
    raise NotImplementedError(
        "PineScript generation is pending migration to Anthropic Agent SDK (Epic 7)"
    )


def validate_syntax(state: PineScriptState) -> Dict[str, Any]:
    """
    Validate Pine Script syntax using pattern matching.
    This function is STUB since validation tools are still available.
    """
    logger.info("Validating Pine Script syntax...")

    code = state.get('pine_script_code', '')

    if not code:
        errors = ["No Pine Script code to validate"]
        return {'validation_errors': errors, 'status': 'error'}

    # Use the validation tool for comprehensive syntax checking
    try:
        validation_result = validate_pine_script_syntax.invoke({"pine_code": code})
        errors = validation_result.get('errors', [])
        warnings = validation_result.get('warnings', [])

        if warnings and not errors:
            logger.info(f"Validation warnings: {warnings}")
    except Exception as e:
        logger.error(f"Validation tool error: {e}")
        errors = [f"Validation error: {str(e)}"]

    if errors:
        return {
            'validation_errors': errors,
            'status': 'fixing'
        }

    return {
        'validation_errors': [],
        'status': 'complete'
    }


def fix_errors(state: PineScriptState) -> Dict[str, Any]:
    """
    STUB - Fix pending migration to Anthropic Agent SDK.
    """
    logger.warning("PineScript fix_errors is a stub - pending Epic 7")
    raise NotImplementedError(
        "PineScript error fixing is pending migration to Anthropic Agent SDK (Epic 7)"
    )


def should_fix_or_complete(state: PineScriptState) -> str:
    """
    STUB - pending migration to Anthropic Agent SDK.
    """
    return END


def compile_pinescript_graph():
    """
    STUB - Compilation pending migration to Anthropic Agent SDK.
    """
    logger.warning("compile_pinescript_graph is a stub - pending Epic 7")
    raise NotImplementedError(
        "PineScript graph compilation is pending migration to Anthropic Agent SDK (Epic 7)"
    )


# Create the compiled graph for langgraph.json
pine_script_graph = None  # Previously: compile_pinescript_graph()


def register_pinescript_tools_with_agent(agent):
    """
    STUB - pending migration to Anthropic Agent SDK.
    """
    logger.warning("register_pinescript_tools_with_agent is a stub - pending Epic 7")
    raise NotImplementedError(
        "Tool registration is pending migration to Anthropic Agent SDK (Epic 7)"
    )


def generate_pine_script_from_query(query: str) -> Dict[str, Any]:
    """
    STUB - pending migration to Anthropic Agent SDK.
    """
    logger.warning("generate_pine_script_from_query is a stub - pending Epic 7")
    raise NotImplementedError(
        "Pine Script generation is pending migration to Anthropic Agent SDK (Epic 7)"
    )


def convert_mql5_to_pinescript(mql5_code: str) -> Dict[str, Any]:
    """
    STUB - pending migration to Anthropic Agent SDK.
    """
    logger.warning("convert_mql5_to_pinescript is a stub - pending Epic 7")
    raise NotImplementedError(
        "MQL5 to Pine Script conversion is pending migration to Anthropic Agent SDK (Epic 7)"
    )


if __name__ == '__main__':
    print("pinescript.py is deprecated. Use /api/floor-manager endpoints instead.")
