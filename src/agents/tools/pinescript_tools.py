"""
Pine Script Tools for QuantMind Agents.

Provides validation and MQL5→Pine Script conversion tools for the Pine Script agent.
These tools are exposed to LangGraph and HTTP routes for Pine Script workflow integration.

**Validates: Property 17: Pine Script Agent - Tooling**
"""

import re
import logging
from typing import Dict, Any, List, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# =============================================================================
# Pine Script Validation Tools
# =============================================================================

@tool("validate_pine_script_syntax")
def validate_pine_script_syntax(pine_code: str) -> Dict[str, Any]:
    """
    Validate Pine Script v5 syntax using pattern matching.
    
    Performs comprehensive syntax validation including:
    - Version declaration check
    - Indicator/strategy declaration
    - Balanced parentheses and brackets
    - Deprecated v4 syntax detection
    - Common syntax error patterns
    
    Args:
        pine_code: Pine Script source code to validate
        
    Returns:
        Dictionary with validation results:
        - is_valid: bool
        - errors: List[str] - List of validation errors
        - warnings: List[str] - List of warnings
        - line_count: int
    """
    errors = []
    warnings = []
    
    if not pine_code or not pine_code.strip():
        return {
            "is_valid": False,
            "errors": ["Empty Pine Script code provided"],
            "warnings": [],
            "line_count": 0
        }
    
    lines = pine_code.split('\n')
    line_count = len(lines)
    
    # Check for version declaration
    if not re.search(r'//@version=5', pine_code):
        errors.append("Missing //@version=5 declaration at the top of the script")
    
    # Check for indicator or strategy declaration
    if not (re.search(r'indicator\s*\(', pine_code) or 
            re.search(r'strategy\s*\(', pine_code)):
        errors.append("Missing indicator() or strategy() declaration")
    
    # Check for balanced parentheses
    open_parens = pine_code.count('(')
    close_parens = pine_code.count(')')
    if open_parens != close_parens:
        errors.append(f"Unbalanced parentheses: {open_parens} opening, {close_parens} closing")
    
    # Check for balanced brackets
    open_brackets = pine_code.count('[')
    close_brackets = pine_code.count(']')
    if open_brackets != close_brackets:
        errors.append(f"Unbalanced brackets: {open_brackets} opening, {close_brackets} closing")
    
    # Check for balanced braces
    open_braces = pine_code.count('{')
    close_braces = pine_code.count('}')
    if open_braces != close_braces:
        errors.append(f"Unbalanced braces: {open_braces} opening, {close_braces} closing")
    
    # Check for common v4 syntax (deprecated)
    deprecated_patterns = [
        (r'\bstudy\s*\(', "Deprecated: 'study()' - use 'indicator()' instead"),
        (r'\binput\s*\(\s*\)', "Deprecated: 'input()' without type - use 'input.int()', 'input.float()', etc."),
        (r'\bsma\s*\(', "Deprecated: 'sma()' - use 'ta.sma()' instead"),
        (r'\bema\s*\(', "Deprecated: 'ema()' - use 'ta.ema()' instead"),
        (r'\brsi\s*\(', "Deprecated: 'rsi()' - use 'ta.rsi()' instead"),
        (r'\bcross\s*\(', "Deprecated: 'cross()' - use 'ta.cross()' instead"),
        (r'\bcrossover\s*\(', "Deprecated: 'crossover()' - use 'ta.crossover()' instead"),
        (r'\bcrossunder\s*\(', "Deprecated: 'crossunder()' - use 'ta.crossunder()' instead"),
        (r'\bplotcandle\s*\(', "Deprecated: 'plotcandle()' - use 'plotcandle()' with proper arguments"),
    ]
    
    for pattern, message in deprecated_patterns:
        if re.search(pattern, pine_code):
            warnings.append(message)
    
    # Check for potential issues
    # Check for unclosed strings
    single_quote_count = pine_code.count("'")
    double_quote_count = pine_code.count('"')
    if single_quote_count % 2 != 0:
        warnings.append("Possible unclosed single-quote string")
    if double_quote_count % 2 != 0:
        warnings.append("Possible unclosed double-quote string")
    
    # Check for proper variable declarations
    if re.search(r'^[0-9]', pine_code.split('=')[0].strip() if '=' in pine_code else ''):
        errors.append("Variable names cannot start with a number")
    
    is_valid = len(errors) == 0
    
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "line_count": line_count
    }


@tool("validate_pine_script_strategy")
def validate_pine_script_strategy(pine_code: str) -> Dict[str, Any]:
    """
    Validate Pine Script strategy completeness.
    
    Checks for essential strategy components:
    - Entry conditions
    - Exit conditions (stop loss / take profit)
    - Position management
    - Risk management
    
    Args:
        pine_code: Pine Script source code to validate
        
    Returns:
        Dictionary with strategy validation results:
        - has_entries: bool
        - has_exits: bool
        - has_stop_loss: bool
        - has_take_profit: bool
        - completeness_score: float (0-1)
        - recommendations: List[str]
    """
    recommendations = []
    
    # Check for entry conditions
    has_long_entry = bool(re.search(r'strategy\.entry\s*\(\s*["\']long["\']', pine_code, re.IGNORECASE))
    has_short_entry = bool(re.search(r'strategy\.entry\s*\(\s*["\']short["\']', pine_code, re.IGNORECASE))
    has_entries = has_long_entry or has_short_entry
    
    # Check for exit conditions
    has_exit = bool(re.search(r'strategy\.exit\s*\(', pine_code, re.IGNORECASE))
    has_close = bool(re.search(r'strategy\.close\s*\(', pine_code, re.IGNORECASE))
    has_exits = has_exit or has_close
    
    # Check for stop loss
    has_stop_loss = bool(re.search(r'stop\s*=', pine_code)) or bool(re.search(r'stop=\s*', pine_code))
    
    # Check for take profit
    has_take_profit = bool(re.search(r'limit\s*=', pine_code)) or bool(re.search(r'target\s*=', pine_code))
    
    # Check for risk management
    has_risk = bool(re.search(r'risk|position_size|qty', pine_code, re.IGNORECASE))
    
    # Calculate completeness score
    components = [
        has_entries,
        has_exits,
        has_stop_loss,
        has_take_profit,
        has_risk
    ]
    completeness_score = sum(components) / len(components)
    
    # Generate recommendations
    if not has_entries:
        recommendations.append("Add entry conditions using strategy.entry()")
    if not has_exits:
        recommendations.append("Add exit conditions using strategy.exit() or strategy.close()")
    if not has_stop_loss:
        recommendations.append("Consider adding stop loss for risk management")
    if not has_take_profit:
        recommendations.append("Consider adding take profit levels")
    if not has_risk:
        recommendations.append("Consider adding position sizing or risk management")
    
    if has_long_entry and not has_short_entry:
        recommendations.append("Strategy only has long entries - consider adding short entries for bidirectional trading")
    
    return {
        "has_entries": has_entries,
        "has_exits": has_exits,
        "has_stop_loss": has_stop_loss,
        "has_take_profit": has_take_profit,
        "has_risk_management": has_risk,
        "completeness_score": completeness_score,
        "recommendations": recommendations
    }


# =============================================================================
# MQL5 to Pine Script Conversion Tools
# =============================================================================

# MQL5 to Pine Script function mappings
MQL5_TO_PINE_MAPPING = {
    # Indicator functions
    'iRSI': 'ta.rsi',
    'iMA': 'ta.sma',
    'iEMA': 'ta.ema',
    'iSMA': 'ta.sma',
    'iMACD': 'ta.macd',
    'iBands': 'ta.bands',
    'iStochastic': 'ta.stoch',
    'iATR': 'ta.atr',
    'iCCI': 'ta.cci',
    'iADX': 'ta.adx',
    'iAlligator': 'ta.alligator',
    'iAO': 'ta.awesome_oscillator',
    'iAC': 'ta.ac',
    'iBearsPower': 'ta.bearpower',
    'iBullsPower': 'ta.bullpower',
    'iDeMarker': 'ta.demarker',
    'iEnvelopes': 'ta.envelope',
    'iForce': 'ta.force',
    'iGator': 'ta.gator',
    'iIchimoku': 'ta.ichimoku',
    'iMomentum': 'ta.momentum',
    'iMFI': 'ta.mfi',
    'iOBV': 'ta.obv',
    'iOsMA': 'ta.oscillator',
    'iRVGI': 'ta.rvgi',
    'iStdDev': 'ta.stdev',
    'iTripleEMA': 'ta.tema',
    'iVariance': 'ta.variance',
    'iWPR': 'ta.wpr',
    'iZenWealth': 'ta.zigzag',
    # Order functions
    'OrderSend': 'strategy.entry',
    'OrderClose': 'strategy.close',
    'OrderModify': 'strategy.order',
    # Array functions
    'ArraySetAsSeries': 'array.reverse',
    'ArraySize': 'array.size',
    'ArrayResize': 'array.new_',
    # String functions
    'DoubleToString': 'str.tostring',
    'IntegerToString': 'str.tostring',
    'StringConcatenate': 'str.format',
    # Time functions
    'TimeCurrent': 'timenow',
    'TimeDay': 'dayofmonth',
    'TimeHour': 'hour',
    'TimeMinute': 'minute',
    'TimeSeconds': 'second',
    # Math functions
    'MathAbs': 'math.abs',
    'MathMax': 'math.max',
    'MathMin': 'math.min',
    'MathPow': 'math.pow',
    'MathSqrt': 'math.sqrt',
    'MathRound': 'math.round',
    'MathFloor': 'math.floor',
    'MathCeil': 'math.ceil',
    # Other common functions
    'NormalizeDouble': '0',  # Pine handles this automatically
    'Symbol': 'syminfo.base',
    'Digits': 'syminfo.mintick',
    'Point': 'syminfo.point',
}


@tool("convert_mql5_to_pine")
def convert_mql5_to_pine(mql5_code: str) -> Dict[str, Any]:
    """
    Convert MQL5 code to Pine Script v5.
    
    Performs basic translation of MQL5 syntax to Pine Script:
    - Function name mappings (iRSI -> ta.rsi, etc.)
    - Array access patterns
    - Basic type conversions
    
    Note: This is a basic conversion tool. Complex strategies may require
    manual adjustments. For full conversion, use the Pine Script agent.
    
    Args:
        mql5_code: MQL5 source code to convert
        
    Returns:
        Dictionary with conversion results:
        - pine_code: str - Converted Pine Script code
        - mappings_applied: List[str] - List of function mappings applied
        - warnings: List[str] - Warnings about manual review needed
        - needs_review: bool
    """
    pine_code = mql5_code
    mappings_applied = []
    warnings = []
    
    # Apply function mappings
    for mql5_func, pine_func in MQL5_TO_PINE_MAPPING.items():
        if mql5_func in pine_code:
            # Only replace complete function calls (followed by parenthesis)
            pattern = r'\b' + re.escape(mql5_func) + r'\s*\('
            if re.search(pattern, pine_code):
                pine_code = re.sub(pattern, pine_func + '(', pine_code)
                mappings_applied.append(f"{mql5_func} -> {pine_func}")
    
    # Handle array access patterns: MQL5 uses Array[i], Pine uses array.get()
    # This is complex, add a warning
    if '[]' in mql5_code or 'Array[' in mql5_code:
        warnings.append("Array access patterns detected - manual review required for array indexing")
    
    # Handle input parameter conversions
    # MQL5: input double LotSize = 0.1;
    # Pine: lot_size = input.float(0.1, "Lot Size")
    input_pattern = r'input\s+(\w+)\s+(\w+)\s*=\s*([^;]+);'
    input_matches = re.findall(input_pattern, mql5_code)
    if input_matches:
        warnings.append(f"Found {len(input_matches)} input parameters - may need manual conversion to Pine input functions")
    
    # Handle variable declarations
    # MQL5: double myVar = 1.0;
    # Pine: var myVar = 1.0
    pine_code = re.sub(r'\bdouble\s+', 'var ', pine_code)
    pine_code = re.sub(r'\bint\s+', 'var ', pine_code)
    pine_code = re.sub(r'\bbool\s+', 'var ', pine_code)
    pine_code = re.sub(r'\bstring\s+', 'var ', pine_code)
    
    # Add version declaration if missing
    if not pine_code.startswith('//@version=5'):
        pine_code = '//@version=5\n' + pine_code
    
    # Check if it's a strategy or indicator
    if 'OnTick' in mql5_code or 'OrderSend' in mql5_code:
        # Add strategy declaration
        if 'strategy(' not in pine_code:
            pine_code = pine_code.replace('//@version=5', '//@version=5\nstrategy("Converted Strategy", overlay=true)', 1)
        warnings.append("Converted from EA - ensure strategy.entry() calls are properly configured")
    
    # Mark as needing review if there are complex patterns
    needs_review = len(warnings) > 0 or len(input_matches) > 0
    
    return {
        "pine_code": pine_code,
        "mappings_applied": mappings_applied,
        "warnings": warnings,
        "needs_review": needs_review
    }


@tool("extract_pine_indicators")
def extract_pine_indicators(pine_code: str) -> Dict[str, Any]:
    """
    Extract indicators and their parameters from Pine Script code.
    
    Args:
        pine_code: Pine Script source code
        
    Returns:
        Dictionary with extracted indicators:
        - indicators: List[Dict] - List of indicators with parameters
        - timeframes: List[str] - Timeframes used
        - symbols: List[str] - Symbols referenced
    """
    indicators = []
    timeframes = []
    symbols = []
    
    # Pattern for indicator/function calls with parameters
    # Matches: ta.rsi(close, 14), ta.sma(close, 20), etc.
    indicator_patterns = [
        (r'ta\.(\w+)\s*\(([^)]+)\)', 'ta'),
        (r'(\w+)\s*\(([^)]+)\)', 'other'),  # Generic function calls
    ]
    
    # Known indicator functions
    known_indicators = [
        'sma', 'ema', 'rma', 'vwma', 'wma', 'hma',
        'rsi', 'macd', 'atr', 'stoch', 'cci', 'adx',
        'bb', 'bands', 'keltner', 'donchian', 'pivot',
        'crossover', 'crossunder', 'cross',
        'highest', 'lowest', 'range', 'tr',
        'mom', 'moments', 'stdev', 'variance',
        'sar', 'supertrend', 'zigzag', 'price',
    ]
    
    for line in pine_code.split('\n'):
        # Extract timeframes
        tf_matches = re.findall(r'period(?:_)?(\w+)|timeframe\.(\w+)|\'(\w+)\'', line, re.IGNORECASE)
        for match in tf_matches:
            for m in match:
                if m and m.upper() not in ['D', 'W', 'M', 'H', 'M1', 'M5', 'M15', 'M30', 'H1', 'H4']:
                    continue
                if m and m not in timeframes:
                    timeframes.append(m)
        
        # Extract symbols
        sym_matches = re.findall(r'syminfo\.(\w+)|symbol[:\s]+[\'"](\w+)[\'"]', line, re.IGNORECASE)
        for match in sym_matches:
            for m in match:
                if m and len(m) >= 6 and m.isalpha():
                    if m not in symbols:
                        symbols.append(m)
        
        # Extract indicators
        for known in known_indicators:
            if f'ta.{known}' in line.lower() or f'.{known}(' in line.lower():
                # Try to extract parameters
                param_match = re.search(rf'{known}\s*\(([^)]+)\)', line, re.IGNORECASE)
                params = param_match.group(1).split(',') if param_match else []
                
                indicators.append({
                    "name": known.upper(),
                    "parameters": [p.strip() for p in params[:3]],  # Limit to 3 params
                    "line": line.strip()
                })
    
    return {
        "indicators": indicators,
        "timeframes": list(set(timeframes)),
        "symbols": list(set(symbols))
    }


# =============================================================================
# Tool Registry for LangGraph Integration
# =============================================================================

PINESCRIPT_TOOLS = [
    validate_pine_script_syntax,
    validate_pine_script_strategy,
    convert_mql5_to_pine,
    extract_pine_indicators,
]


def get_pine_script_tool(name: str):
    """
    Get a Pine Script tool by name.
    
    Args:
        name: Tool name (e.g., "validate_pine_script_syntax")
        
    Returns:
        Tool function or None if not found
    """
    tool_map = {tool.name: tool for tool in PINESCRIPT_TOOLS}
    return tool_map.get(name)


def get_all_pine_script_tools() -> List:
    """
    Get all Pine Script tools.
    
    Returns:
        List of Pine Script tools
    """
    return PINESCRIPT_TOOLS


# =============================================================================
# API Integration
# =============================================================================

def register_pine_script_tools(app):
    """
    Register Pine Script tools with FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    try:
        from fastapi import APIRouter
        
        router = APIRouter(prefix="/pinescript", tags=["pinescript"])
        
        @router.post("/validate")
        async def validate_pine(pine_code: str):
            """Validate Pine Script syntax."""
            return validate_pine_script_syntax.invoke({"pine_code": pine_code})
        
        @router.post("/validate-strategy")
        async def validate_strategy(pine_code: str):
            """Validate Pine Script strategy completeness."""
            return validate_pine_script_strategy.invoke({"pine_code": pine_code})
        
        @router.post("/convert")
        async def convert_mql5(mql5_code: str):
            """Convert MQL5 to Pine Script."""
            return convert_mql5_to_pine.invoke({"mql5_code": mql5_code})
        
        @router.post("/extract")
        async def extract_indicators(pine_code: str):
            """Extract indicators from Pine Script."""
            return extract_pine_indicators.invoke({"pine_code": pine_code})
        
        # Include router in app if it exists
        if hasattr(app, 'include_router'):
            app.include_router(router)
        
        logger.info("Registered Pine Script tools with FastAPI")
        
    except ImportError:
        logger.warning("FastAPI not available - skipping tool registration")
    except Exception as e:
        logger.error(f"Failed to register Pine Script tools: {e}")


if __name__ == '__main__':
    # Test the tools
    test_pine = """//@version=5
strategy("Test Strategy", overlay=true)

length = input.int(14, "Length")
source = input(close, "Source")

rsi = ta.rsi(source, length)

if rsi < 30
    strategy.entry("Long", strategy.long)

if rsi > 70
    strategy.close("Long")
"""
    
    print("Testing validate_pine_script_syntax:")
    result = validate_pine_script_syntax.invoke({"pine_code": test_pine})
    print(f"Valid: {result['is_valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")
    
    print("\nTesting validate_pine_script_strategy:")
    result = validate_pine_script_strategy.invoke({"pine_code": test_pine})
    print(f"Completeness: {result['completeness_score']}")
    print(f"Recommendations: {result['recommendations']}")
    
    print("\nTesting extract_pine_indicators:")
    result = extract_pine_indicators.invoke({"pine_code": test_pine})
    print(f"Indicators: {result['indicators']}")
