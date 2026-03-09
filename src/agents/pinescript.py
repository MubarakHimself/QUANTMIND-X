"""
Pine Script Agent - DEPRECATED

Use floor_manager /api/floor-manager endpoints instead.
This module used LangGraph which has been removed.

**Validates: Property 17: Pine Script Agent**
"""

import logging

logger = logging.getLogger(__name__)

# Deprecation warning
logger.warning(
    "pinescript.py is deprecated. "
    "Use /api/floor-manager endpoints with 'development' department instead."
)


def get_llm(*args, **kwargs):
    """Deprecated - use floor_manager instead."""
    raise NotImplementedError(
        "pinescript module is deprecated. Use /api/floor-manager instead."
    )

# Import Pine Script tools for validation and conversion
from src.agents.tools.pinescript_tools import (
    validate_pine_script_syntax,
    validate_pine_script_strategy,
    convert_mql5_to_pine,
    extract_pine_indicators,
    PINESCRIPT_TOOLS,
)

logger = logging.getLogger(__name__)


# Pine Script v5 system prompt
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

## Common Patterns:

### Strategy Template:
```pinescript
//@version=5
strategy("Strategy Name", overlay=true, initial_capital=10000, default_qty_type=strategy.percent_of_equity, default_qty_value=10)

// Input parameters
length = input.int(14, "Length")
risk_per_trade = input.float(1.0, "Risk % per trade")

// Indicators
// ... indicator calculations ...

// Entry conditions
long_condition = ...
short_condition = ...

// Execute trades
if (long_condition)
    strategy.entry("Long", strategy.long)
    
if (short_condition)
    strategy.entry("Short", strategy.short)

// Stop loss and take profit
strategy.exit("Exit Long", from_entry="Long", stop=stop_loss, limit=take_profit)
```

## Alert Configuration:
When generating code that needs to send alerts to external webhooks, include:
```pinescript
alertcondition(condition, title="Alert Title", message="Webhook message")
```

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
    Generate Pine Script code from natural language description.
    
    Args:
        state: Current agent state with user query
        
    Returns:
        Updated state with generated Pine Script code
    """
    logger.info("Generating Pine Script code...")
    
    llm = get_llm()
    
    # Determine the prompt based on conversion mode
    if state.get('conversion_mode') == 'convert' and state.get('mql5_source'):
        user_prompt = f"""Convert the following MQL5 code to Pine Script v5:

MQL5 Source Code:
```
{state['mql5_source']}
```

Please:
1. Translate all MQL5 functions to their Pine Script equivalents
2. Convert MQL5 order logic to Pine Script strategy functions
3. Map MQL5 indicators to Pine Script's ta namespace
4. Maintain the same strategy logic and parameters
5. Add proper comments explaining any conversion decisions
"""
    else:
        user_prompt = f"""Generate Pine Script v5 code for the following strategy:

Strategy Description:
{state.get('strategy_description') or state.get('user_query')}

Please:
1. Include all necessary input parameters
2. Implement proper entry and exit conditions
3. Add stop-loss and take-profit logic
4. Include visual plotting for confirmation
5. Add webhook alert conditions if needed
"""
    
    try:
        response = llm.invoke([
            {"role": "system", "content": PINESCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ])
        
        pine_script_code = response.content
        
        # Extract code from markdown code blocks if present
        code_match = re.search(r'```(?:pinescript)?\s*([\s\S]*?)\s*```', pine_script_code)
        if code_match:
            pine_script_code = code_match.group(1).strip()
        
        return {
            'pine_script_code': pine_script_code,
            'status': 'validating',
            'validation_errors': []
        }
        
    except Exception as e:
        logger.error(f"Failed to generate Pine Script: {e}")
        return {
            'status': 'error',
            'validation_errors': [f"Generation error: {str(e)}"]
        }


def validate_syntax(state: PineScriptState) -> Dict[str, Any]:
    """
    Validate Pine Script syntax using pattern matching.
    
    Args:
        state: Current agent state with generated code
        
    Returns:
        Updated state with validation results
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
            # Warnings don't block completion, but let's log them
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
    Fix Pine Script errors using LLM.
    
    Args:
        state: Current agent state with validation errors
        
    Returns:
        Updated state with fixed code
    """
    logger.info("Fixing Pine Script errors...")
    
    llm = get_llm()
    
    errors = state.get('validation_errors', [])
    code = state.get('pine_script_code', '')
    
    fix_prompt = f"""The following Pine Script code has validation errors. Please fix them:

Current Code:
```pinescript
{code}
```

Errors to Fix:
{chr(10).join(f'- {error}' for error in errors)}

Please provide the corrected Pine Script code with all errors fixed.
Only output the corrected code, no explanations.
"""
    
    try:
        response = llm.invoke([
            {"role": "system", "content": PINESCRIPT_SYSTEM_PROMPT},
            {"role": "user", "content": fix_prompt}
        ])
        
        fixed_code = response.content
        
        # Extract code from markdown code blocks if present
        code_match = re.search(r'```(?:pinescript)?\s*([\s\S]*?)\s*```', fixed_code)
        if code_match:
            fixed_code = code_match.group(1).strip()
        
        return {
            'pine_script_code': fixed_code,
            'status': 'validating',
            'validation_errors': []
        }
        
    except Exception as e:
        logger.error(f"Failed to fix Pine Script: {e}")
        return {
            'status': 'error',
            'validation_errors': [f"Fix error: {str(e)}"]
        }


def should_fix_or_complete(state: PineScriptState) -> str:
    """
    Determine next step after validation.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name
    """
    status = state.get('status', '')
    
    if status == 'fixing':
        return 'fix_errors'
    elif status == 'complete':
        return END
    elif status == 'error':
        return END
    else:
        return END


def compile_pinescript_graph() -> StateGraph:
    """
    Compile the Pine Script agent graph.
    
    Returns:
        Compiled StateGraph for Pine Script generation
    """
    workflow = StateGraph(PineScriptState)
    
    # Add nodes
    workflow.add_node('generate', generate_pine_script)
    workflow.add_node('validate', validate_syntax)
    workflow.add_node('fix_errors', fix_errors)
    
    # Set entry point
    workflow.set_entry_point('generate')
    
    # Add edges
    workflow.add_edge('generate', 'validate')
    workflow.add_conditional_edges(
        'validate',
        should_fix_or_complete,
        {
            'fix_errors': 'fix_errors',
            END: END
        }
    )
    workflow.add_edge('fix_errors', 'validate')
    
    return workflow.compile()


# Create the compiled graph for langgraph.json
pine_script_graph = compile_pinescript_graph()


def register_pinescript_tools_with_agent(agent):
    """
    Register Pine Script tools with an agent.
    
    Args:
        agent: Agent instance to register tools with
    
    Returns:
        Agent with Pine Script tools registered
    """
    from langgraph.prebuilt import ToolNode
    
    tool_node = ToolNode(PINESCRIPT_TOOLS)
    logger.info(f"Registered {len(PINESCRIPT_TOOLS)} Pine Script tools with agent")
    return tool_node


def generate_pine_script_from_query(query: str) -> Dict[str, Any]:
    """
    Convenience function to generate Pine Script from a query.
    
    Args:
        query: Natural language strategy description
        
    Returns:
        Dictionary with pine_script, status, and errors
    """
    initial_state = {
        'messages': [],
        'user_query': query,
        'strategy_description': query,
        'pine_script_code': None,
        'validation_errors': [],
        'status': 'pending',
        'workspace_path': '',
        'context': {},
        'memory_namespace': ('pinescript',),
        'conversion_mode': 'generate'
    }
    
    result = pine_script_graph.invoke(initial_state)
    
    return {
        'pine_script': result.get('pine_script_code'),
        'status': result.get('status'),
        'errors': result.get('validation_errors', [])
    }


def convert_mql5_to_pinescript(mql5_code: str) -> Dict[str, Any]:
    """
    Convert MQL5 code to Pine Script.
    
    Args:
        mql5_code: MQL5 source code
        
    Returns:
        Dictionary with pine_script, status, and errors
    """
    initial_state = {
        'messages': [],
        'user_query': 'Convert MQL5 to Pine Script',
        'strategy_description': None,
        'pine_script_code': None,
        'validation_errors': [],
        'status': 'pending',
        'mql5_source': mql5_code,
        'workspace_path': '',
        'context': {},
        'memory_namespace': ('pinescript', 'convert'),
        'conversion_mode': 'convert'
    }
    
    result = pine_script_graph.invoke(initial_state)
    
    return {
        'pine_script': result.get('pine_script_code'),
        'status': result.get('status'),
        'errors': result.get('validation_errors', [])
    }


if __name__ == '__main__':
    # Test the agent
    test_query = "Create an ICT Silver Bullet strategy that identifies optimal entry points during the London and New York session killzones"
    
    result = generate_pine_script_from_query(test_query)
    
    print(f"Status: {result['status']}")
    print(f"Errors: {result['errors']}")
    print(f"\nGenerated Pine Script:\n{result['pine_script']}")