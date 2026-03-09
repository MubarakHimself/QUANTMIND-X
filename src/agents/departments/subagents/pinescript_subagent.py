"""
PineScript Subagent

Specialized worker agent for Pine Script development.
Generates Pine Script v5 code from natural language descriptions.

Model: Haiku (fast, cost-effective for code generation)
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# Pine Script v5 system prompt for LLM
PINESCRIPT_SYSTEM_PROMPT = """You are an expert Pine Script v5 developer for TradingView.

Your task is to generate clean, efficient, and well-documented Pine Script v5 code
based on the user's strategy description.

## Code Style Guidelines:
1. Always use `//@version=5` declaration at the top
2. Use `indicator()` or `strategy()` declaration
3. Use Pine Script v5 syntax only (no legacy syntax)
4. Add descriptive comments for each section
5. Use meaningful variable names
6. Include input parameters for customization
7. Add proper stop-loss and take-profit levels
8. Implement proper entry and exit conditions
9. Use `plot()` for visual confirmation
10. Use `strategy()` with proper default_qty_type

## Common Patterns:

### Strategy Template:
```pinescript
//@version=5
strategy("Strategy Name", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// Input parameters
length = input.int(14, "Length")
risk_per_trade = input.float(1.0, "Risk % per trade")

// Indicators
rsi = ta.rsi(close, length)
sma20 = ta.sma(close, 20)
sma50 = ta.sma(close, 50)

// Entry conditions
long_condition = ta.crossover(rsi, 30)
short_condition = ta.crossunder(rsi, 70)

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)

if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop loss and take profit
strategy.exit("Exit Long", from_entry="Long", stop=stop_loss, limit=take_profit)

// Plotting
plot(sma20, "SMA 20", color.blue)
plot(sma50, "SMA 50", color.orange)
```

## Important Notes:
- Never use `varip` unless absolutely necessary
- Avoid repainting indicators
- Use `request.security()` carefully for multi-timeframe analysis
- Always test strategies thoroughly before live trading
- Output ONLY the Pine Script code, no explanations
"""


# MQL5 to Pine conversion prompt
MQL5_TO_PINE_PROMPT = """You are an expert Pine Script v5 developer. Convert the following MQL5 code to Pine Script v5.

Convert:
1. MQL5 functions to Pine Script equivalents
2. MQL5 order logic to Pine Script strategy functions
3. MQL5 indicators to Pine Script's ta namespace
4. Maintain the same strategy logic and parameters

Output ONLY the Pine Script code, no explanations.
"""


@dataclass
class PineScriptTask:
    """PineScript task input data."""
    task_type: str  # generate, convert, validate
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class PineScriptSubAgent:
    """
    PineScript subagent for TradingView Pine Script development.

    Capabilities:
    - Generate Pine Script v5 from natural language
    - Convert MQL5 code to Pine Script
    - Validate Pine Script syntax

    Model: Haiku (fast, cost-effective for code generation)
    """

    def __init__(
        self,
        agent_id: str,
        task: Optional[PineScriptTask] = None,
        available_tools: Optional[List[str]] = None,
    ):
        """
        Initialize PineScript subagent.

        Args:
            agent_id: Unique identifier for this agent
            task: Task configuration
            available_tools: List of tool names available to this agent
        """
        self.agent_id = agent_id
        self.agent_type = "pinescript"
        self.task = task or PineScriptTask(task_type="generate")
        self.available_tools = available_tools or []
        self.model_tier = "haiku"
        self._llm_client = None
        self._initialize_tools()
        self._initialize_llm()

    def _initialize_llm(self) -> None:
        """Initialize LLM client for code generation."""
        try:
            from anthropic import Anthropic
            self._llm_client = Anthropic()
            logger.info("PineScriptSubAgent: LLM client initialized")
        except ImportError:
            logger.warning("PineScriptSubAgent: Anthropic SDK not available")

    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for this agent."""
        tools = {}

        # Import validation tools
        try:
            from src.agents.tools.pinescript_tools import (
                validate_pine_script_syntax,
                validate_pine_script_strategy,
                convert_mql5_to_pine as tool_convert,
                extract_pine_indicators,
            )
            self._validation_tools = {
                "validate_syntax": validate_pine_script_syntax,
                "validate_strategy": validate_pine_script_strategy,
                "convert_mql5": tool_convert,
                "extract_indicators": extract_pine_indicators,
            }
        except ImportError as e:
            logger.warning(f"PineScriptSubAgent: Could not load validation tools: {e}")
            self._validation_tools = {}

        return tools

    def _call_llm(
        self,
        user_prompt: str,
        system_prompt: str = PINESCRIPT_SYSTEM_PROMPT,
    ) -> str:
        """
        Call LLM to generate code.

        Args:
            user_prompt: User's request
            system_prompt: System instructions

        Returns:
            Generated code
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        try:
            response = self._llm_client.messages.create(
                model="claude-3-5-sonnet-20241022",  # Use Sonnet for better code gen
                max_tokens=4000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract code from response
            code = response.content[0].text

            # Extract code from markdown code blocks if present
            code_match = re.search(
                r'```(?:pinescript)?\s*([\s\S]*?)\s*```',
                code,
                re.IGNORECASE
            )
            if code_match:
                code = code_match.group(1).strip()

            return code

        except Exception as e:
            logger.error(f"PineScriptSubAgent: LLM call failed: {e}")
            raise

    def generate(
        self,
        strategy_description: str,
        include_validation: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate Pine Script v5 code from natural language.

        Args:
            strategy_description: Natural language description of the strategy
            include_validation: Whether to validate the generated code

        Returns:
            Dict with code, validation results, and status
        """
        logger.info(f"PineScriptSubAgent: Generating Pine Script for: {strategy_description[:50]}...")

        try:
            # Generate code using LLM
            pine_code = self._call_llm(
                user_prompt=f"Generate Pine Script v5 code for: {strategy_description}",
                system_prompt=PINESCRIPT_SYSTEM_PROMPT,
            )

            result = {
                "code": pine_code,
                "status": "generated",
                "task_type": "generate",
            }

            # Validate if requested
            if include_validation and self._validation_tools:
                validation = self._validation_tools["validate_syntax"](pine_code)
                result["validation"] = validation
                result["is_valid"] = validation.get("is_valid", False)
                result["errors"] = validation.get("errors", [])
                result["warnings"] = validation.get("warnings", [])

                if validation.get("is_valid"):
                    result["status"] = "complete"
                else:
                    result["status"] = "needs_fix"

            return result

        except Exception as e:
            logger.error(f"PineScriptSubAgent: Generation failed: {e}")
            return {
                "code": None,
                "status": "error",
                "errors": [str(e)],
            }

    def convert(
        self,
        mql5_code: str,
        include_validation: bool = True,
    ) -> Dict[str, Any]:
        """
        Convert MQL5 code to Pine Script v5.

        Args:
            mql5_code: MQL5 source code
            include_validation: Whether to validate the converted code

        Returns:
            Dict with code, validation results, and status
        """
        logger.info("PineScriptSubAgent: Converting MQL5 to Pine Script")

        try:
            # Convert using LLM
            pine_code = self._call_llm(
                user_prompt=mql5_code,
                system_prompt=MQL5_TO_PINE_PROMPT,
            )

            result = {
                "code": pine_code,
                "status": "converted",
                "task_type": "convert",
            }

            # Validate if requested
            if include_validation and self._validation_tools:
                validation = self._validation_tools["validate_syntax"](pine_code)
                result["validation"] = validation
                result["is_valid"] = validation.get("is_valid", False)
                result["errors"] = validation.get("errors", [])
                result["warnings"] = validation.get("warnings", [])

                if validation.get("is_valid"):
                    result["status"] = "complete"
                else:
                    result["status"] = "needs_fix"

            return result

        except Exception as e:
            logger.error(f"PineScriptSubAgent: Conversion failed: {e}")
            return {
                "code": None,
                "status": "error",
                "errors": [str(e)],
            }

    def validate(self, pine_code: str) -> Dict[str, Any]:
        """
        Validate Pine Script syntax.

        Args:
            pine_code: Pine Script source code

        Returns:
            Validation results
        """
        if not self._validation_tools:
            return {
                "is_valid": False,
                "errors": ["Validation tools not available"],
                "warnings": [],
            }

        return self._validation_tools["validate_syntax"](pine_code)

    def refine(
        self,
        pine_code: str,
        feedback: str,
    ) -> Dict[str, Any]:
        """
        Refine Pine Script code based on feedback.

        Args:
            pine_code: Current Pine Script code
            feedback: Feedback/instructions for refinement

        Returns:
            Refined code and validation results
        """
        logger.info(f"PineScriptSubAgent: Refining code with feedback: {feedback[:50]}...")

        try:
            # Refine using LLM
            user_prompt = f"""Here is the current Pine Script code:

```
{pine_code}
```

Refine it based on this feedback: {feedback}

Output ONLY the refined Pine Script code, no explanations.
"""

            refined_code = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=PINESCRIPT_SYSTEM_PROMPT,
            )

            result = {
                "code": refined_code,
                "status": "refined",
            }

            # Validate refined code
            if self._validation_tools:
                validation = self._validation_tools["validate_syntax"](refined_code)
                result["validation"] = validation
                result["is_valid"] = validation.get("is_valid", False)
                result["errors"] = validation.get("errors", [])
                result["warnings"] = validation.get("warnings", [])

            return result

        except Exception as e:
            logger.error(f"PineScriptSubAgent: Refinement failed: {e}")
            return {
                "code": None,
                "status": "error",
                "errors": [str(e)],
            }


def create_pinescript_agent(
    agent_id: str = "pinescript-default",
    task_type: str = "generate",
) -> PineScriptSubAgent:
    """
    Factory function to create a PineScript subagent.

    Args:
        agent_id: Unique identifier
        task_type: Type of task (generate, convert, validate)

    Returns:
        PineScriptSubAgent instance
    """
    task = PineScriptTask(
        task_type=task_type,
        parameters={},
        context={},
    )
    return PineScriptSubAgent(agent_id=agent_id, task=task)
