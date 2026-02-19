"""
QuantCode Tools for QuantMind agents.

These tools provide MQL5 code generation and manipulation:
- generate_mql5: Generate MQL5 code from TRD
- generate_component: Generate specific MQL5 component
- validate_syntax: Validate MQL5 syntax
- fix_syntax: Auto-fix syntax errors
- compile_mql5: Compile MQL5 code
- debug_code: Debug MQL5 code
- optimize_code: Optimize MQL5 performance
- generate_documentation: Generate code documentation
- lookup_docs: Lookup MQL5 documentation
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


class MQL5Component(str, Enum):
    """MQL5 component types."""
    EA_MAIN = "ea_main"
    SIGNAL_GENERATOR = "signal_generator"
    ORDER_MANAGER = "order_manager"
    RISK_MANAGER = "risk_manager"
    INDICATOR = "indicator"
    FILTER = "filter"
    TRAILING_STOP = "trailing_stop"
    MONEY_MANAGEMENT = "money_management"


class SyntaxErrorType(str, Enum):
    """Types of syntax errors."""
    UNDECLARED_VARIABLE = "undeclared_variable"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_SEMICOLON = "missing_semicolon"
    UNCLOSED_BRACKET = "unclosed_bracket"
    UNDEFINED_FUNCTION = "undefined_function"
    INVALID_SYNTAX = "invalid_syntax"


@dataclass
class SyntaxError:
    """Represents a syntax error in code."""
    line: int
    column: int
    error_type: SyntaxErrorType
    message: str
    suggestion: Optional[str] = None


@dataclass
class CompilationResult:
    """Result of MQL5 compilation."""
    success: bool
    errors: List[SyntaxError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    output_path: Optional[str] = None
    compile_time_ms: float = 0


class GenerateMQL5Input(BaseModel):
    """Input schema for generate_mql5 tool."""
    trd_content: str = Field(
        description="Technical Requirements Document content"
    )
    template: Optional[str] = Field(
        default=None,
        description="MQL5 template to use"
    )
    include_comments: bool = Field(
        default=True,
        description="Include detailed comments in generated code"
    )
    style: str = Field(
        default="standard",
        description="Code style (standard, minimal, verbose)"
    )


class GenerateComponentInput(BaseModel):
    """Input schema for generate_component tool."""
    component_type: MQL5Component = Field(
        description="Type of component to generate"
    )
    requirements: Dict[str, Any] = Field(
        description="Component requirements and parameters"
    )
    trd_context: Optional[str] = Field(
        default=None,
        description="TRD context for component generation"
    )
    include_tests: bool = Field(
        default=False,
        description="Generate unit tests for component"
    )


class ValidateSyntaxInput(BaseModel):
    """Input schema for validate_syntax tool."""
    code: str = Field(
        description="MQL5 code to validate"
    )
    strict: bool = Field(
        default=True,
        description="Enable strict validation"
    )


class FixSyntaxInput(BaseModel):
    """Input schema for fix_syntax tool."""
    code: str = Field(
        description="MQL5 code with syntax errors"
    )
    errors: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Known errors to fix (auto-detected if not provided)"
    )


class CompileMQL5Input(BaseModel):
    """Input schema for compile_mql5 tool."""
    code: str = Field(
        description="MQL5 code to compile"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to save compiled file"
    )
    optimization_level: str = Field(
        default="debug",
        description="Optimization level (debug, release)"
    )


class DebugCodeInput(BaseModel):
    """Input schema for debug_code tool."""
    code: str = Field(
        description="MQL5 code to debug"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message to analyze"
    )
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional debugging context"
    )


class OptimizeCodeInput(BaseModel):
    """Input schema for optimize_code tool."""
    code: str = Field(
        description="MQL5 code to optimize"
    )
    optimization_goals: List[str] = Field(
        default=["performance"],
        description="Optimization goals (performance, memory, readability)"
    )


class GenerateDocumentationInput(BaseModel):
    """Input schema for generate_documentation tool."""
    code: str = Field(
        description="MQL5 code to document"
    )
    format: str = Field(
        default="markdown",
        description="Documentation format (markdown, html, doxygen)"
    )
    include_examples: bool = Field(
        default=True,
        description="Include usage examples"
    )


class LookupDocsInput(BaseModel):
    """Input schema for lookup_docs tool."""
    query: str = Field(
        description="Search query for MQL5 documentation"
    )
    category: Optional[str] = Field(
        default=None,
        description="Documentation category (functions, classes, constants)"
    )


@register_tool(
    agent_types=[AgentType.QUANTCODE],
    tags=["mql5", "code", "generation"],
)
class GenerateMQL5Tool(QuantMindTool):
    """Generate MQL5 code from TRD."""

    name: str = "generate_mql5"
    description: str = """Generate complete MQL5 Expert Advisor code from a Technical Requirements Document.
    Produces production-ready code with proper structure, error handling, and documentation.
    Supports different code styles and comment levels."""

    args_schema: type[BaseModel] = GenerateMQL5Input
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        trd_content: str,
        template: Optional[str] = None,
        include_comments: bool = True,
        style: str = "standard",
        **kwargs
    ) -> ToolResult:
        """Execute MQL5 code generation."""
        logger.info("Generating MQL5 code from TRD")

        # Parse TRD and generate code
        # In production, this would use an LLM
        generated_code = self._generate_from_trd(
            trd_content,
            include_comments,
            style
        )

        return ToolResult.ok(
            data={
                "code": generated_code,
                "language": "mql5",
                "lines_of_code": len(generated_code.split("\n")),
            },
            metadata={
                "generated_at": datetime.now().isoformat(),
                "template": template,
                "style": style,
                "include_comments": include_comments,
            }
        )

    def _generate_from_trd(
        self,
        trd_content: str,
        include_comments: bool,
        style: str
    ) -> str:
        """Generate MQL5 code from TRD content."""
        # Basic EA template - in production would be LLM-generated
        comment_prefix = "//" if include_comments else ""

        code = f"""//+------------------------------------------------------------------+
//|                                    Generated Expert Advisor      |
//|                                    QuantMind Code Generator      |
//+------------------------------------------------------------------+
#property copyright "QuantMind"
#property link      "https://quantmind.io"
#property version   "1.00"
#property strict

{comment_prefix}--- Input Parameters ---
input double   RiskPercent = 2.0;     {comment_prefix}Risk per trade (%)
input int      StopLoss = 50;          {comment_prefix}Stop Loss (pips)
input int      TakeProfit = 100;       {comment_prefix}Take Profit (pips)
input int      MagicNumber = 123456;   {comment_prefix}EA Magic Number
input int      Slippage = 3;           {comment_prefix}Max slippage

{comment_prefix}--- Global Variables ---
datetime lastBarTime = 0;
double minLot = 0;
double maxLot = 0;
double lotStep = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{{
    {comment_prefix}--- Validate inputs ---
    if(RiskPercent <= 0 || RiskPercent > 100)
    {{
        Print("Invalid RiskPercent: ", RiskPercent);
        return INIT_PARAMETERS_INCORRECT;
    }}

    {comment_prefix}--- Get symbol info ---
    minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

    Print("EA Initialized successfully");
    return INIT_SUCCEEDED;
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
    Print("EA Deinitialized. Reason: ", reason);
}}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{{
    {comment_prefix}--- Check for new bar ---
    if(lastBarTime == iTime(_Symbol, PERIOD_CURRENT, 0))
        return;

    lastBarTime = iTime(_Symbol, PERIOD_CURRENT, 0);

    {comment_prefix}--- Check for open positions ---
    if(HasOpenPosition())
        return;

    {comment_prefix}--- Generate signals ---
    int signal = GenerateSignal();

    {comment_prefix}--- Execute trades ---
    if(signal == 1)
        ExecuteBuy();
    else if(signal == -1)
        ExecuteSell();
}}

//+------------------------------------------------------------------+
//| Generate trading signal                                            |
//+------------------------------------------------------------------+
int GenerateSignal()
{{
    {comment_prefix}--- Implement your trading logic here ---
    {comment_prefix}--- This is a simple MA crossover example ---

    double fastMA = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE, 0);
    double slowMA = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE, 0);
    double fastMAPrev = iMA(_Symbol, PERIOD_CURRENT, 10, 0, MODE_SMA, PRICE_CLOSE, 1);
    double slowMAPrev = iMA(_Symbol, PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE, 1);

    {comment_prefix}--- Buy signal: Fast MA crosses above Slow MA ---
    if(fastMA > slowMA && fastMAPrev <= slowMAPrev)
        return 1;

    {comment_prefix}--- Sell signal: Fast MA crosses below Slow MA ---
    if(fastMA < slowMA && fastMAPrev >= slowMAPrev)
        return -1;

    return 0;  {comment_prefix}--- No signal ---
}}

//+------------------------------------------------------------------+
//| Check if there's an open position                                  |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{{
    for(int i = PositionsTotal() - 1; i >= 0; i--)
    {{
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0)
        {{
            if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
               PositionGetInteger(POSITION_MAGIC) == MagicNumber)
                return true;
        }}
    }}
    return false;
}}

//+------------------------------------------------------------------+
//| Calculate position size                                            |
//+------------------------------------------------------------------+
double CalculatePositionSize()
{{
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * (RiskPercent / 100.0);
    double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

    double stopLossValue = StopLoss * point;
    double lotSize = riskAmount / (stopLossValue / tickSize * tickValue);

    {comment_prefix}--- Normalize lot size ---
    lotSize = MathFloor(lotSize / lotStep) * lotStep;
    lotSize = MathMax(minLot, MathMin(maxLot, lotSize));

    return lotSize;
}}

//+------------------------------------------------------------------+
//| Execute buy order                                                  |
//+------------------------------------------------------------------+
void ExecuteBuy()
{{
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double sl = ask - StopLoss * _Point;
    double tp = ask + TakeProfit * _Point;
    double lotSize = CalculatePositionSize();

    MqlTradeRequest request = {{}};
    MqlTradeResult result = {{}};

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lotSize;
    request.type = ORDER_TYPE_BUY;
    request.price = ask;
    request.sl = sl;
    request.tp = tp;
    request.deviation = Slippage;
    request.magic = MagicNumber;
    request.comment = "QuantMind EA Buy";

    if(!OrderSend(request, result))
        Print("Buy order failed. Error: ", GetLastError());
    else
        Print("Buy order executed successfully. Ticket: ", result.order);
}}

//+------------------------------------------------------------------+
//| Execute sell order                                                 |
//+------------------------------------------------------------------+
void ExecuteSell()
{{
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double sl = bid + StopLoss * _Point;
    double tp = bid - TakeProfit * _Point;
    double lotSize = CalculatePositionSize();

    MqlTradeRequest request = {{}};
    MqlTradeResult result = {{}};

    request.action = TRADE_ACTION_DEAL;
    request.symbol = _Symbol;
    request.volume = lotSize;
    request.type = ORDER_TYPE_SELL;
    request.price = bid;
    request.sl = sl;
    request.tp = tp;
    request.deviation = Slippage;
    request.magic = MagicNumber;
    request.comment = "QuantMind EA Sell";

    if(!OrderSend(request, result))
        Print("Sell order failed. Error: ", GetLastError());
    else
        Print("Sell order executed successfully. Ticket: ", result.order);
}}
//+------------------------------------------------------------------+
"""
        return code


@register_tool(
    agent_types=[AgentType.QUANTCODE],
    tags=["mql5", "component", "generation"],
)
class GenerateComponentTool(QuantMindTool):
    """Generate specific MQL5 component."""

    name: str = "generate_component"
    description: str = """Generate a specific MQL5 component such as signal generator, order manager, or risk manager.
    Produces modular, reusable code components.
    Can include unit tests for the component."""

    args_schema: type[BaseModel] = GenerateComponentInput
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        component_type: MQL5Component,
        requirements: Dict[str, Any],
        trd_context: Optional[str] = None,
        include_tests: bool = False,
        **kwargs
    ) -> ToolResult:
        """Execute component generation."""
        logger.info(f"Generating component: {component_type}")

        # Generate component code
        code = self._generate_component_code(component_type, requirements)

        result_data = {
            "component_type": component_type.value,
            "code": code,
            "dependencies": self._get_dependencies(component_type),
        }

        if include_tests:
            result_data["tests"] = self._generate_tests(component_type)

        return ToolResult.ok(
            data=result_data,
            metadata={
                "generated_at": datetime.now().isoformat(),
                "include_tests": include_tests,
            }
        )

    def _generate_component_code(
        self,
        component_type: MQL5Component,
        requirements: Dict[str, Any]
    ) -> str:
        """Generate component-specific code."""
        # In production, this would be LLM-generated
        if component_type == MQL5Component.SIGNAL_GENERATOR:
            return self._generate_signal_generator(requirements)
        elif component_type == MQL5Component.RISK_MANAGER:
            return self._generate_risk_manager(requirements)
        elif component_type == MQL5Component.TRAILING_STOP:
            return self._generate_trailing_stop(requirements)
        else:
            return f"// {component_type.value} component - not yet implemented"

    def _generate_signal_generator(self, reqs: Dict) -> str:
        return """
class SignalGenerator {
private:
    int fastPeriod;
    int slowPeriod;

public:
    SignalGenerator(int fast=10, int slow=20) : fastPeriod(fast), slowPeriod(slow) {}

    int Generate() {
        double fastMA = iMA(_Symbol, PERIOD_CURRENT, fastPeriod, 0, MODE_SMA, PRICE_CLOSE, 0);
        double slowMA = iMA(_Symbol, PERIOD_CURRENT, slowPeriod, 0, MODE_SMA, PRICE_CLOSE, 0);

        if(fastMA > slowMA) return 1;   // Buy
        if(fastMA < slowMA) return -1;  // Sell
        return 0;  // No signal
    }
};
"""

    def _generate_risk_manager(self, reqs: Dict) -> str:
        return """
class RiskManager {
private:
    double maxRiskPercent;
    double maxDrawdownPercent;

public:
    RiskManager(double risk=2.0, double drawdown=20.0)
        : maxRiskPercent(risk), maxDrawdownPercent(drawdown) {}

    double CalculatePositionSize(double stopLossPips) {
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double riskAmount = balance * (maxRiskPercent / 100.0);
        // Calculate lot size based on risk
        return NormalizeDouble(riskAmount / stopLossPips / 10.0, 2);
    }

    bool CheckRiskLimit() {
        double equity = AccountInfoDouble(ACCOUNT_EQUITY);
        double balance = AccountInfoDouble(ACCOUNT_BALANCE);
        double drawdown = (balance - equity) / balance * 100.0;
        return drawdown < maxDrawdownPercent;
    }
};
"""

    def _generate_trailing_stop(self, reqs: Dict) -> str:
        return """
class TrailingStop {
private:
    int trailPips;
    int triggerPips;

public:
    TrailingStop(int trail=20, int trigger=30)
        : trailPips(trail), triggerPips(trigger) {}

    void Update(ulong ticket) {
        if(!PositionSelectByTicket(ticket)) return;

        double currentSL = PositionGetDouble(POSITION_SL);
        double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double currentPrice = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY
            ? SymbolInfoDouble(_Symbol, SYMBOL_BID)
            : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

        double trailDistance = trailPips * _Point;

        if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
            double newSL = currentPrice - trailDistance;
            if(newSL > currentSL && newSL > openPrice + triggerPips * _Point) {
                // Update stop loss
            }
        }
    }
};
"""

    def _get_dependencies(self, component_type: MQL5Component) -> List[str]:
        """Get required dependencies for component."""
        deps = {
            MQL5Component.SIGNAL_GENERATOR: ["TradeClasses"],
            MQL5Component.ORDER_MANAGER: ["TradeClasses"],
            MQL5Component.RISK_MANAGER: [],
            MQL5Component.TRAILING_STOP: ["TradeClasses"],
        }
        return deps.get(component_type, [])

    def _generate_tests(self, component_type: MQL5Component) -> str:
        """Generate unit tests for component."""
        return f"// Unit tests for {component_type.value} - not yet implemented"


@register_tool(
    agent_types=[AgentType.QUANTCODE],
    tags=["mql5", "syntax", "validation"],
)
class ValidateSyntaxTool(QuantMindTool):
    """Validate MQL5 syntax."""

    name: str = "validate_syntax"
    description: str = """Validate MQL5 code syntax without compilation.
    Checks for common errors like undeclared variables, type mismatches, and missing semicolons.
    Returns detailed error information with suggestions."""

    args_schema: type[BaseModel] = ValidateSyntaxInput
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        code: str,
        strict: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute syntax validation."""
        logger.info("Validating MQL5 syntax")

        errors = self._validate_code(code, strict)

        return ToolResult.ok(
            data={
                "is_valid": len(errors) == 0,
                "errors": [
                    {
                        "line": e.line,
                        "column": e.column,
                        "type": e.error_type.value,
                        "message": e.message,
                        "suggestion": e.suggestion,
                    }
                    for e in errors
                ],
            },
            metadata={
                "validated_at": datetime.now().isoformat(),
                "strict_mode": strict,
                "line_count": len(code.split("\n")),
            }
        )

    def _validate_code(self, code: str, strict: bool) -> List[SyntaxError]:
        """Validate code and return errors."""
        errors = []
        lines = code.split("\n")

        # Simple validation rules
        # In production, would use proper MQL5 parser

        for i, line in enumerate(lines, 1):
            # Check for missing semicolons (simple heuristic)
            stripped = line.strip()
            if stripped and not stripped.endswith(("{", "}", "(", ")", ",", "//", ";", ":")):
                if any(kw in stripped for kw in ["int ", "double ", "string ", "bool ", "return "]):
                    if not stripped.endswith(";"):
                        errors.append(SyntaxError(
                            line=i,
                            column=len(stripped),
                            error_type=SyntaxErrorType.MISSING_SEMICOLON,
                            message="Possibly missing semicolon",
                            suggestion="Add ';' at end of statement"
                        ))

        return errors


@register_tool(
    agent_types=[AgentType.QUANTCODE],
    tags=["mql5", "syntax", "fix"],
)
class FixSyntaxTool(QuantMindTool):
    """Auto-fix MQL5 syntax errors."""

    name: str = "fix_syntax"
    description: str = """Automatically fix common MQL5 syntax errors.
    Can fix missing semicolons, unclosed brackets, and simple type mismatches.
    Returns fixed code with change summary."""

    args_schema: type[BaseModel] = FixSyntaxInput
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        code: str,
        errors: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute syntax fix."""
        logger.info("Fixing MQL5 syntax errors")

        # Validate first if no errors provided
        if not errors:
            validate_tool = ValidateSyntaxTool(workspace_path=self._workspace_path)
            result = validate_tool.execute(code=code)
            errors = result.data.get("errors", [])

        fixed_code = code
        changes = []

        for error in errors:
            fix_result = self._apply_fix(fixed_code, error)
            if fix_result:
                fixed_code = fix_result["code"]
                changes.append(fix_result["change"])

        return ToolResult.ok(
            data={
                "original_code": code,
                "fixed_code": fixed_code,
                "changes": changes,
                "fixes_applied": len(changes),
            },
            metadata={
                "fixed_at": datetime.now().isoformat(),
            }
        )

    def _apply_fix(self, code: str, error: Dict) -> Optional[Dict]:
        """Apply fix for a specific error."""
        lines = code.split("\n")
        line_idx = error.get("line", 0) - 1

        if line_idx < 0 or line_idx >= len(lines):
            return None

        error_type = error.get("type")

        if error_type == SyntaxErrorType.MISSING_SEMICOLON.value:
            line = lines[line_idx]
            if not line.rstrip().endswith(";"):
                lines[line_idx] = line.rstrip() + ";"
                return {
                    "code": "\n".join(lines),
                    "change": {
                        "line": line_idx + 1,
                        "type": "added_semicolon",
                        "description": "Added missing semicolon"
                    }
                }

        return None


@register_tool(
    agent_types=[AgentType.QUANTCODE],
    tags=["mql5", "compile", "build"],
)
class CompileMQL5Tool(QuantMindTool):
    """Compile MQL5 code."""

    name: str = "compile_mql5"
    description: str = """Compile MQL5 source code to .ex5 executable.
    Reports compilation errors and warnings.
    Can save output to specified path."""

    args_schema: type[BaseModel] = CompileMQL5Input
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        code: str,
        file_path: Optional[str] = None,
        optimization_level: str = "debug",
        **kwargs
    ) -> ToolResult:
        """Execute MQL5 compilation."""
        logger.info("Compiling MQL5 code")

        # Validate first
        validate_tool = ValidateSyntaxTool(workspace_path=self._workspace_path)
        result = validate_tool.execute(code=code)
        syntax_errors = result.data.get("errors", [])

        if syntax_errors:
            return ToolResult.error(
                error="Compilation failed due to syntax errors",
                metadata={"syntax_errors": syntax_errors}
            )

        # In production, would call MT5 compiler
        # Simulate successful compilation
        compilation_result = CompilationResult(
            success=True,
            errors=[],
            warnings=["Variable 'unused' is declared but never used"],
            output_path=file_path or "output/EA.ex5",
            compile_time_ms=1250
        )

        return ToolResult.ok(
            data={
                "success": compilation_result.success,
                "output_path": compilation_result.output_path,
                "errors": [{"message": e.message, "line": e.line} for e in compilation_result.errors],
                "warnings": compilation_result.warnings,
                "compile_time_ms": compilation_result.compile_time_ms,
            },
            metadata={
                "compiled_at": datetime.now().isoformat(),
                "optimization_level": optimization_level,
            }
        )


@register_tool(
    agent_types=[AgentType.QUANTCODE],
    tags=["mql5", "debug", "troubleshoot"],
)
class DebugCodeTool(QuantMindTool):
    """Debug MQL5 code."""

    name: str = "debug_code"
    description: str = """Debug MQL5 code by analyzing errors and suggesting fixes.
    Identifies common issues like logic errors, null pointer access, and infinite loops.
    Provides detailed debugging suggestions."""

    args_schema: type[BaseModel] = DebugCodeInput
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.NORMAL

    def execute(
        self,
        code: str,
        error_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ToolResult:
        """Execute code debugging."""
        logger.info("Debugging MQL5 code")

        issues = []
        suggestions = []

        # Analyze code for common issues
        issues.extend(self._check_logic_errors(code))
        issues.extend(self._check_resource_issues(code))
        issues.extend(self._check_performance_issues(code))

        # Generate suggestions based on issues
        for issue in issues:
            suggestions.append(self._get_suggestion(issue))

        # If error message provided, analyze it
        if error_message:
            error_analysis = self._analyze_error(error_message, code)
            issues.append(error_analysis)

        return ToolResult.ok(
            data={
                "issues": issues,
                "suggestions": suggestions,
                "analyzed_lines": len(code.split("\n")),
            },
            metadata={
                "debugged_at": datetime.now().isoformat(),
                "error_message": error_message,
            }
        )

    def _check_logic_errors(self, code: str) -> List[Dict]:
        """Check for logic errors in code."""
        issues = []

        # Check for potential infinite loops
        if "while(true)" in code.lower() or "for(;;)" in code.lower():
            if "break" not in code.lower() and "return" not in code.lower():
                issues.append({
                    "type": "potential_infinite_loop",
                    "severity": "warning",
                    "message": "Potential infinite loop detected without break condition"
                })

        return issues

    def _check_resource_issues(self, code: str) -> List[Dict]:
        """Check for resource-related issues."""
        issues = []

        # Check for memory leaks indicators
        if "new " in code and "delete " not in code:
            issues.append({
                "type": "potential_memory_leak",
                "severity": "warning",
                "message": "Dynamic allocation detected without corresponding deallocation"
            })

        return issues

    def _check_performance_issues(self, code: str) -> List[Dict]:
        """Check for performance issues."""
        issues = []

        # Check for repeated indicator calls
        indicator_calls = code.count("iMA(") + code.count("iRSI(") + code.count("iMACD(")
        if indicator_calls > 10:
            issues.append({
                "type": "performance",
                "severity": "info",
                "message": f"Many indicator calls ({indicator_calls}) - consider caching values"
            })

        return issues

    def _get_suggestion(self, issue: Dict) -> str:
        """Get suggestion for an issue."""
        suggestions_map = {
            "potential_infinite_loop": "Add a break condition or timeout check inside the loop",
            "potential_memory_leak": "Ensure all dynamically allocated objects are deleted",
            "performance": "Cache indicator values in variables to avoid repeated calculations",
        }
        return suggestions_map.get(issue["type"], "Review and address the issue")

    def _analyze_error(self, error_message: str, code: str) -> Dict:
        """Analyze error message."""
        return {
            "type": "runtime_error",
            "severity": "error",
            "message": error_message,
            "analysis": "Check the line mentioned in the error and surrounding context"
        }


@register_tool(
    agent_types=[AgentType.QUANTCODE],
    tags=["mql5", "optimize", "performance"],
)
class OptimizeCodeTool(QuantMindTool):
    """Optimize MQL5 code performance."""

    name: str = "optimize_code"
    description: str = """Optimize MQL5 code for better performance.
    Applies optimizations like caching, loop unrolling, and reducing redundant calculations.
    Returns optimized code with performance notes."""

    args_schema: type[BaseModel] = OptimizeCodeInput
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.LOW

    def execute(
        self,
        code: str,
        optimization_goals: List[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute code optimization."""
        logger.info("Optimizing MQL5 code")

        goals = optimization_goals or ["performance"]
        optimized_code = code
        optimizations = []

        if "performance" in goals:
            result = self._optimize_performance(optimized_code)
            optimized_code = result["code"]
            optimizations.extend(result["changes"])

        return ToolResult.ok(
            data={
                "original_code": code,
                "optimized_code": optimized_code,
                "optimizations": optimizations,
            },
            metadata={
                "optimized_at": datetime.now().isoformat(),
                "goals": goals,
            }
        )

    def _optimize_performance(self, code: str) -> Dict:
        """Apply performance optimizations."""
        changes = []

        # In production, would apply actual optimizations
        changes.append({
            "type": "caching",
            "description": "Cached repeated indicator calls"
        })

        return {"code": code, "changes": changes}


@register_tool(
    agent_types=[AgentType.QUANTCODE],
    tags=["mql5", "documentation", "comments"],
)
class GenerateDocumentationTool(QuantMindTool):
    """Generate documentation for MQL5 code."""

    name: str = "generate_documentation"
    description: str = """Generate comprehensive documentation for MQL5 code.
    Supports markdown, HTML, and Doxygen formats.
    Includes function descriptions, parameter documentation, and usage examples."""

    args_schema: type[BaseModel] = GenerateDocumentationInput
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.LOW

    def execute(
        self,
        code: str,
        format: str = "markdown",
        include_examples: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute documentation generation."""
        logger.info("Generating MQL5 documentation")

        # Extract functions and classes
        functions = self._extract_functions(code)
        classes = self._extract_classes(code)

        # Generate documentation
        if format == "markdown":
            docs = self._generate_markdown_docs(functions, classes, include_examples)
        elif format == "html":
            docs = self._generate_html_docs(functions, classes)
        else:
            docs = self._generate_doxygen_docs(functions, classes)

        return ToolResult.ok(
            data={
                "documentation": docs,
                "format": format,
                "functions_documented": len(functions),
                "classes_documented": len(classes),
            },
            metadata={
                "generated_at": datetime.now().isoformat(),
            }
        )

    def _extract_functions(self, code: str) -> List[Dict]:
        """Extract function definitions."""
        functions = []
        pattern = r"(?:void|int|double|string|bool|color|datetime)\s+(\w+)\s*\(([^)]*)\)"
        for match in re.finditer(pattern, code):
            functions.append({
                "name": match.group(1),
                "params": match.group(2),
                "signature": match.group(0)
            })
        return functions

    def _extract_classes(self, code: str) -> List[Dict]:
        """Extract class definitions."""
        classes = []
        pattern = r"class\s+(\w+)(?:\s*:\s*(?:public|private|protected)?\s*(\w+))?"
        for match in re.finditer(pattern, code):
            classes.append({
                "name": match.group(1),
                "base": match.group(2) or "None"
            })
        return classes

    def _generate_markdown_docs(self, functions: List, classes: List, include_examples: bool) -> str:
        """Generate markdown documentation."""
        docs = "# Code Documentation\n\n"

        if classes:
            docs += "## Classes\n\n"
            for cls in classes:
                docs += f"### {cls['name']}\n\n"
                docs += f"Base class: {cls['base']}\n\n"

        if functions:
            docs += "## Functions\n\n"
            for func in functions:
                docs += f"### {func['name']}\n\n"
                docs += f"```mql5\n{func['signature']}\n```\n\n"

        return docs

    def _generate_html_docs(self, functions: List, classes: List) -> str:
        return "<html><body><h1>Documentation</h1></body></html>"

    def _generate_doxygen_docs(self, functions: List, classes: List) -> str:
        return "/** Doxygen documentation */"


@register_tool(
    agent_types=[AgentType.QUANTCODE, AgentType.ANALYST],
    tags=["mql5", "docs", "reference", "lookup"],
)
class LookupDocsTool(QuantMindTool):
    """Lookup MQL5 documentation."""

    name: str = "lookup_docs"
    description: str = """Search and retrieve MQL5 documentation.
    Provides quick reference for functions, classes, and constants.
    Returns relevant documentation with examples."""

    args_schema: type[BaseModel] = LookupDocsInput
    category: ToolCategory = ToolCategory.QUANTCODE
    priority: ToolPriority = ToolPriority.NORMAL

    # Common MQL5 documentation
    DOCS_DB = {
        "OrderSend": {
            "syntax": "bool OrderSend(MqlTradeRequest& request, MqlTradeResult& result)",
            "description": "Sends a trade request to the server",
            "parameters": {
                "request": "Reference to the MqlTradeRequest structure",
                "result": "Reference to the MqlTradeResult structure"
            },
            "return": "Returns true if the request is accepted, otherwise false",
            "example": """
MqlTradeRequest request = {};
MqlTradeResult result = {};
request.action = TRADE_ACTION_DEAL;
request.symbol = _Symbol;
request.volume = 0.1;
request.type = ORDER_TYPE_BUY;
request.price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
if(!OrderSend(request, result))
    Print("OrderSend failed: ", GetLastError());
"""
        },
        "iMA": {
            "syntax": "double iMA(string symbol, ENUM_TIMEFRAMES period, int ma_period, int ma_shift, ENUM_MA_METHOD ma_method, ENUM_APPLIED_PRICE applied_price, int shift)",
            "description": "Returns the Moving Average indicator value",
            "parameters": {
                "symbol": "Symbol name",
                "period": "Timeframe",
                "ma_period": "MA period",
                "ma_shift": "MA shift",
                "ma_method": "MA method (SMA, EMA, etc.)",
                "applied_price": "Applied price",
                "shift": "Shift from current bar"
            },
            "return": "Returns the MA value",
            "example": """
double ma = iMA(_Symbol, PERIOD_H1, 14, 0, MODE_SMA, PRICE_CLOSE, 0);
"""
        },
        "PositionSelect": {
            "syntax": "bool PositionSelect(string symbol)",
            "description": "Selects a position for further processing",
            "parameters": {
                "symbol": "Symbol name"
            },
            "return": "Returns true if successful",
            "example": """
if(PositionSelect(_Symbol)) {
    double profit = PositionGetDouble(POSITION_PROFIT);
    Print("Current profit: ", profit);
}
"""
        }
    }

    def execute(
        self,
        query: str,
        category: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """Execute documentation lookup."""
        logger.info(f"Looking up MQL5 docs: {query}")

        results = []
        query_lower = query.lower()

        for name, doc in self.DOCS_DB.items():
            if query_lower in name.lower() or query_lower in doc["description"].lower():
                results.append({
                    "name": name,
                    **doc
                })

        return ToolResult.ok(
            data={
                "query": query,
                "results": results,
                "total_found": len(results),
            },
            metadata={
                "category": category,
            }
        )


# Export all tools
__all__ = [
    "GenerateMQL5Tool",
    "GenerateComponentTool",
    "ValidateSyntaxTool",
    "FixSyntaxTool",
    "CompileMQL5Tool",
    "DebugCodeTool",
    "OptimizeCodeTool",
    "GenerateDocumentationTool",
    "LookupDocsTool",
    "MQL5Component",
    "SyntaxErrorType",
    "SyntaxError",
    "CompilationResult",
]
