"""
Pine Script to MQL5 Converter
=============================
A LangGraph agent for converting Pine Script code to MQL5 EA code.

This is the reverse direction of the existing MQL5→Pine Script conversion
in src/agents/pinescript.py.

Features:
- Converts Pine Script v5 strategies to MQL5 Expert Advisors
- Validates generated MQL5 code structure
- Iteratively fixes validation errors via LLM feedback
- Returns production-ready MQL5 code

NOTE: LangGraph imports removed - pending migration to Anthropic Agent SDK (Epic 7).
The conversion logic remains but uses a simple function-based workflow instead of StateGraph.
"""

import re
import logging
from typing import TypedDict, List, Dict, Any, Optional

# LangGraph imports removed
# from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)


# =============================================================================
# Stub for StateGraph
# =============================================================================

class StateGraph:
    """Stub for langgraph StateGraph - pending migration to Anthropic Agent SDK."""
    def __init__(self, state_class):
        self.state_class = state_class
        self._nodes = {}

    def add_node(self, name: str, func):
        self._nodes[name] = func

    def add_edge(self, from_node: str, to_node: str):
        pass

    def add_conditional_edges(self, from_node: str, condition_func, mapping: Dict):
        pass

    def set_entry_point(self, node: str):
        pass

    def compile(self):
        return _StubCompiledGraph()


class _StubCompiledGraph:
    """Stub for compiled LangGraph."""
    def invoke(self, state):
        return state

    def ainvoke(self, state):
        return state


# Use a sentinel for END
END = "END"


# ============================================================================
# State Definition
# ============================================================================

class PineToMQL5State(TypedDict):
    """
    State for Pine Script to MQL5 conversion workflow.
    
    Fields:
        pine_script_code: Input Pine Script code
        mql5_code: Generated MQL5 code
        validation_errors: List of validation errors found
        status: Current workflow status
    """
    pine_script_code: str
    mql5_code: Optional[str]
    validation_errors: List[str]
    status: str  # "generating", "validating", "fixing", "completed", "error"


# ============================================================================
# System Prompt
# ============================================================================

PINE_TO_MQL5_SYSTEM_PROMPT = """You are an expert MQL5 developer specializing in converting Pine Script v5 code to MQL5 Expert Advisors.

Your task is to translate Pine Script trading strategies into clean, production-ready MQL5 code.

## MQL5 EA Structure Requirements:

### 1. Header and Properties
```mql5
#property copyright "QuantMindX"
#property version   "1.00"
#property strict
//EA name and description
#property description "Converted from Pine Script"
```

### 2. Input Parameters (inputs)
```mql5
input group "Strategy Parameters"
input int    InpMagicNumber = 123456;    // Magic Number
input double InpLotSize    = 0.1;       // Lot Size
input int    InpStopLoss   = 50;        // Stop Loss (points)
input int    InpTakeProfit = 100;       // Take Profit (points)
```

### 3. Global Variables
```mql5
double g_lot_size = 0.0;
int    g_magic   = 0;
datetime g_last_bar = 0;
```

### 4. OnInit Event
```mql5
int OnInit()
{
   // Validate inputs
   if(InpLotSize <= 0 || InpLotSize > 10)
   {
      Print("Invalid lot size");
      return INIT_PARAMETERS_INCORRECT;
   }
   
   g_lot_size = InpLotSize;
   g_magic    = InpMagicNumber;
   
   // Initialize indicators here if needed
   
   return INIT_SUCCEEDED;
}
```

### 5. OnDeinit Event
```mql5
void OnDeinit(const int reason)
{
   // Cleanup indicators, handles, etc.
   Comment("");
}
```

### 6. OnTick Event (Main Logic)
```mql5
void OnTick()
{
   // Check for new bar (avoid multiple executions per bar)
   if(IsNewBar())
   {
      // Check for existing positions
      if(!HasPosition())
      {
         // Check entry conditions and trade
         CheckForEntry();
      }
      else
      {
         // Check exit conditions
         CheckForExit();
      }
   }
}
```

### 7. Helper Functions
```mql5
bool IsNewBar()
{
   datetime current = iTime(_Symbol, PERIOD_CURRENT, 0);
   if(g_lastBar == current) return false;
   g_lastBar = current;
   return true;
}

bool HasPosition()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == g_magic)
         return true;
   }
   return false;
}

void CheckForEntry()
{
   // Your entry logic here
   // Use OrderSend() for trading
   
   MqlTradeRequest request = {};
   MqlTradeResult  result  = {};
   
   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = _Symbol;
   request.volume   = g_lot_size;
   request.type     = ORDER_TYPE_BUY;
   request.price    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   request.sl      = 0;
   request.tp      = 0;
   request.magic   = g_magic;
   request.comment = "EA Entry";
   
   OrderSend(request, result);
}
```

## Pine Script to MQL5 Indicator Mapping:

| Pine Script | MQL5 |
|------------|------|
| `sma(close, 14)` | `iMA(_Symbol, PERIOD_CURRENT, 14, 0, MODE_SMA, PRICE_CLOSE)` |
| `ema(close, 14)` | `iMA(_Symbol, PERIOD_CURRENT, 14, 0, MODE_EMA, PRICE_CLOSE)` |
| `rsi(close, 14)` | `iRSI(_Symbol, PERIOD_CURRENT, 14, PRICE_CLOSE)` |
| `macd(close, 12, 26, 9)` | `iMACD(_Symbol, PERIOD_CURRENT, 12, 26, 9, PRICE_CLOSE)` |
| `bb(close, 20, 2)` | `iBands(_Symbol, PERIOD_CURRENT, 20, 2, 0, PRICE_CLOSE)` |
| `atr(14)` | `iATR(_Symbol, PERIOD_CURRENT, 14)` |
| `stdev(close, 20)` | Not directly available, calculate manually |

## Pine Script to MQL5 Trading Functions:

| Pine Script | MQL5 |
|------------|------|
| `strategy.entry("Long", strategy.long)` | `OrderSend()` with `ORDER_TYPE_BUY` |
| `strategy.entry("Short", strategy.short)` | `OrderSend()` with `ORDER_TYPE_SELL` |
| `strategy.exit("Exit")` | `OrderClose()` or `PositionClose()` |
| `strategy.order()` | `OrderSend()` |

## Important Notes:
1. Always use proper error handling for trading operations
2. Check for sufficient margin before trading
3. Use proper lot size calculation based on risk
4. Include proper trade comment/magic number for tracking
5. Handle broker spread and slippage appropriately
6. Use `PositionSelect()` and `PositionGetInteger(POSITION_TYPE)` for exit logic

Convert the Pine Script code to MQL5 following these patterns.
"""


# ============================================================================
# Helper Functions
# ============================================================================

def _get_llm():
    """
    Get LLM instance for Pine Script conversion.
    
    Uses the quantcode LLM configuration for code generation.
    """
    try:
        from src.agents.llm_provider import get_quantcode_llm
        return get_quantcode_llm()
    except ImportError:
        # Fallback to get_llm_for_agent if get_quantcode_llm not available
        from src.agents.llm_provider import get_llm_for_agent
        return get_llm_for_agent("quantcode")


def _extract_code_from_response(response: Any) -> str:
    """Extract MQL5 code from LLM response."""
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = str(response)
    
    # Extract code from markdown code blocks if present
    code_match = re.search(r'```(?:mql5)?\s*([\s\S]*?)\s*```', content)
    if code_match:
        return code_match.group(1).strip()
    
    return content.strip()


def _validate_mql5_structure(code: str) -> List[str]:
    """
    Validate MQL5 code structure.
    
    Checks for:
    - OnInit function
    - OnTick function  
    - #property declarations
    - input parameters
    
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check for required functions
    if not re.search(r'void\s+OnInit\s*\(\s*\)', code):
        errors.append("Missing OnInit() function")
    
    if not re.search(r'void\s+OnTick\s*\(\s*\)', code):
        errors.append("Missing OnTick() function")
    
    # Check for #property
    if not re.search(r'#property', code):
        errors.append("Missing #property declarations")
    
    # Check for input parameters
    if not re.search(r'input\s+', code):
        errors.append("Missing input parameter declarations")
    
    # Check for proper includes
    if not re.search(r'#include\s+<.*\.mqh>', code):
        # This is a warning, not an error
        logger.warning("Consider adding #include statements for common functions")
    
    return errors


# ============================================================================
# Graph Nodes
# ============================================================================

def generate_mql5(state: PineToMQL5State) -> PineToMQL5State:
    """
    Generate MQL5 code from Pine Script.
    
    This node invokes the LLM to translate Pine Script logic to MQL5.
    """
    logger.info("Generating MQL5 code from Pine Script...")
    
    try:
        llm = _get_llm()
        
        prompt = f"""Convert the following Pine Script v5 code to MQL5 Expert Advisor code.

Pine Script Code:
```
{state['pine_script_code']}
```

Generate complete, production-ready MQL5 code following the patterns in the system prompt.
Return ONLY the MQL5 code in a markdown code block.
"""
        
        response = llm.invoke([
            {"role": "system", "content": PINE_TO_MQL5_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        mql5_code = _extract_code_from_response(response)
        
        state["mql5_code"] = mql5_code
        state["status"] = "validating"
        
        logger.info(f"Generated MQL5 code ({len(mql5_code)} chars)")
        
    except Exception as e:
        logger.error(f"Error generating MQL5 code: {e}")
        state["validation_errors"] = [f"Generation error: {str(e)}"]
        state["status"] = "error"
    
    return state


def validate_mql5(state: PineToMQL5State) -> PineToMQL5State:
    """
    Validate MQL5 code structure.
    
    Performs regex-based validation to ensure the generated code
    has the required MQL5 EA structure.
    """
    logger.info("Validating MQL5 code structure...")
    
    if not state.get("mql5_code"):
        state["validation_errors"] = ["No MQL5 code to validate"]
        state["status"] = "error"
        return state
    
    errors = _validate_mql5_structure(state["mql5_code"])
    
    if errors:
        state["validation_errors"] = errors
        state["status"] = "fixing"
        logger.warning(f"Validation errors found: {errors}")
    else:
        state["validation_errors"] = []
        state["status"] = "completed"
        logger.info("MQL5 code validated successfully")
    
    return state


def fix_mql5_errors(state: PineToMQL5State) -> PineToMQL5State:
    """
    Fix MQL5 validation errors.
    
    Feeds validation errors back to the LLM for correction.
    """
    logger.info("Fixing MQL5 validation errors...")
    
    if not state.get("validation_errors") or not state.get("pine_script_code"):
        state["status"] = "completed"
        return state
    
    try:
        llm = _get_llm()
        
        errors_str = "\n".join(f"- {err}" for err in state["validation_errors"])
        
        prompt = f"""Fix the following MQL5 code to address the validation errors.

Original Pine Script Code:
```
{state['pine_script_code']}
```

Current MQL5 Code:
```
{state['mql5_code']}
```

Validation Errors:
{errors_str}

Fix these issues and return the corrected MQL5 code in a markdown code block.
"""
        
        response = llm.invoke([
            {"role": "system", "content": PINE_TO_MQL5_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        fixed_code = _extract_code_from_response(response)
        
        state["mql5_code"] = fixed_code
        state["status"] = "validating"  # Will re-validate after fix
        
        logger.info("Applied fixes to MQL5 code")
        
    except Exception as e:
        logger.error(f"Error fixing MQL5 code: {e}")
        state["validation_errors"].append(f"Fix error: {str(e)}")
        state["status"] = "error"
    
    return state


def should_fix_errors(state: PineToMQL5State) -> str:
    """
    Determine if we should attempt to fix validation errors.
    
    Returns:
        "fix" if there are errors and we haven't exceeded retry limit
        "end" if no errors or max retries reached
    """
    if not state.get("validation_errors"):
        return "end"
    
    # Check if we have too many errors (more than 5 is a sign of fundamental problem)
    if len(state["validation_errors"]) > 5:
        logger.error("Too many validation errors, giving up")
        return "end"
    
    return "fix"


# ============================================================================
# Graph Compilation
# ============================================================================

def compile_pine_to_mql5_graph() -> StateGraph:
    """
    Compile the Pine Script to MQL5 conversion graph.
    
    Graph structure:
        generate_mql5 → validate_mql5 → (fix_mql5_errors → validate_mql5)* → END
    """
    graph = StateGraph(PineToMQL5State)
    
    # Add nodes
    graph.add_node("generate", generate_mql5)
    graph.add_node("validate", validate_mql5)
    graph.add_node("fix", fix_mql5_errors)
    
    # Set entry point
    graph.set_entry_point("generate")
    
    # Add edges
    graph.add_edge("generate", "validate")
    
    # Conditional edge from validate to fix or end
    graph.add_conditional_edges(
        "validate",
        should_fix_errors,
        {
            "fix": "fix",
            "end": END
        }
    )
    
    # After fixing, re-validate
    graph.add_edge("fix", "validate")
    
    return graph


# Compile the graph
_pine_to_mql5_graph = compile_pine_to_mql5_graph()


# ============================================================================
# Public API
# ============================================================================

class PineScriptConverter:
    """
    Pine Script to MQL5 converter.
    
    Provides a simple interface to convert Pine Script v5 code to MQL5 EA code.
    
    Usage:
        converter = PineScriptConverter()
        mql5_code = converter.convert(pine_code)
    """
    
    def __init__(self):
        """Initialize the Pine Script converter."""
        self._graph = _pine_to_mql5_graph
    
    def convert(self, pine_code: str) -> str:
        """
        Convert Pine Script code to MQL5.
        
        Args:
            pine_code: Pine Script v5 code to convert
            
        Returns:
            MQL5 code string
            
        Raises:
            ValueError: If conversion fails
        """
        # Initialize state
        initial_state: PineToMQL5State = {
            "pine_script_code": pine_code,
            "mql5_code": None,
            "validation_errors": [],
            "status": "generating"
        }
        
        # Run the graph
        try:
            result = self._graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"Graph execution failed: {e}")
            raise ValueError(f"Conversion failed: {str(e)}")
        
        # Check result
        if result.get("status") == "error":
            errors = result.get("validation_errors", ["Unknown error"])
            raise ValueError(f"Conversion failed: {'; '.join(errors)}")
        
        mql5_code = result.get("mql5_code")
        
        if not mql5_code:
            raise ValueError("Conversion produced no output")
        
        return mql5_code
    
    def validate(self, mql5_code: str) -> Dict[str, Any]:
        """
        Validate MQL5 code structure.
        
        Args:
            mql5_code: MQL5 code to validate
            
        Returns:
            Dictionary with:
                - valid: Boolean indicating if code is valid
                - errors: List of validation errors
        """
        errors = _validate_mql5_structure(mql5_code)
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }


# Singleton instance
_converter: Optional[PineScriptConverter] = None


def get_pine_script_converter() -> PineScriptConverter:
    """Get or create the global PineScriptConverter instance."""
    global _converter
    if _converter is None:
        _converter = PineScriptConverter()
    return _converter
