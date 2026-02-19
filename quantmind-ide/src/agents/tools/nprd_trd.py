"""
NPRD/TRD Tools for QuantMind agents.

These tools handle Natural Product Requirements Document (NPRD) and
Technical Requirements Document (TRD) processing:
- parse_nprd: Parse and extract NPRD structure
- validate_nprd: Validate NPRD structure and content
- generate_trd: Generate TRD from NPRD
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

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


class NPRDSection(str, Enum):
    """Standard NPRD sections."""
    OVERVIEW = "overview"
    OBJECTIVES = "objectives"
    TRADING_LOGIC = "trading_logic"
    RISK_MANAGEMENT = "risk_management"
    ENTRY_CONDITIONS = "entry_conditions"
    EXIT_CONDITIONS = "exit_conditions"
    FILTERS = "filters"
    PARAMETERS = "parameters"
    PERFORMANCE_TARGETS = "performance_targets"
    CONSTRAINTS = "constraints"


class TRDSection(str, Enum):
    """Standard TRD sections."""
    ARCHITECTURE = "architecture"
    COMPONENTS = "components"
    INTERFACES = "interfaces"
    DATA_STRUCTURES = "data_structures"
    ALGORITHMS = "algorithms"
    ERROR_HANDLING = "error_handling"
    TESTING_REQUIREMENTS = "testing_requirements"
    DEPLOYMENT_NOTES = "deployment_notes"


@dataclass
class ParsedNPRD:
    """Parsed NPRD structure."""
    title: str
    version: str
    sections: Dict[NPRDSection, Dict[str, Any]]
    raw_content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


@dataclass
class GeneratedTRD:
    """Generated TRD structure."""
    title: str
    version: str
    source_nprd_version: str
    sections: Dict[TRDSection, Dict[str, Any]]
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ParseNPRDInput(BaseModel):
    """Input schema for parse_nprd tool."""
    nprd_content: Optional[str] = Field(
        default=None,
        description="Raw NPRD content as string"
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Path to NPRD file (alternative to nprd_content)"
    )
    format: str = Field(
        default="markdown",
        description="Input format (markdown, json, yaml)"
    )


class ValidateNPRDInput(BaseModel):
    """Input schema for validate_nprd tool."""
    nprd_content: str = Field(
        description="NPRD content to validate"
    )
    strict: bool = Field(
        default=False,
        description="Enable strict validation mode"
    )
    check_completeness: bool = Field(
        default=True,
        description="Check for all required sections"
    )


class GenerateTRDInput(BaseModel):
    """Input schema for generate_trd tool."""
    nprd_content: str = Field(
        description="Parsed NPRD content"
    )
    template: Optional[str] = Field(
        default=None,
        description="TRD template to use"
    )
    include_code_hints: bool = Field(
        default=True,
        description="Include implementation hints in TRD"
    )
    target_framework: str = Field(
        default="mql5",
        description="Target framework (mql5, python)"
    )


@register_tool(
    agent_types=[AgentType.ANALYST],
    tags=["nprd", "trd", "requirements", "parsing"],
)
class ParseNPRDTool(QuantMindTool):
    """Parse Natural Product Requirements Document."""

    name: str = "parse_nprd"
    description: str = """Parse and extract structure from a Natural Product Requirements Document.
    Supports markdown, JSON, and YAML formats.
    Extracts sections, objectives, trading logic, and parameters."""

    args_schema: type[BaseModel] = ParseNPRDInput
    category: ToolCategory = ToolCategory.NPRD_TRD
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        nprd_content: Optional[str] = None,
        file_path: Optional[str] = None,
        format: str = "markdown",
        **kwargs
    ) -> ToolResult:
        """Execute NPRD parsing."""
        # Get content
        if file_path:
            validated_path = self.validate_workspace_path(file_path)
            if not validated_path.exists():
                raise ToolError(
                    f"NPRD file '{file_path}' does not exist",
                    tool_name=self.name,
                    error_code="FILE_NOT_FOUND"
                )
            nprd_content = validated_path.read_text(encoding="utf-8")
        elif not nprd_content:
            raise ToolError(
                "Either nprd_content or file_path must be provided",
                tool_name=self.name,
                error_code="NO_CONTENT"
            )

        # Parse based on format
        if format == "json":
            parsed = self._parse_json(nprd_content)
        elif format == "yaml":
            parsed = self._parse_yaml(nprd_content)
        else:
            parsed = self._parse_markdown(nprd_content)

        return ToolResult.ok(
            data={
                "title": parsed.title,
                "version": parsed.version,
                "sections": {s.value: v for s, v in parsed.sections.items()},
                "metadata": parsed.metadata,
                "issues": parsed.issues,
            },
            metadata={
                "format": format,
                "parsed_at": datetime.now().isoformat(),
                "section_count": len(parsed.sections),
            }
        )

    def _parse_markdown(self, content: str) -> ParsedNPRD:
        """Parse markdown format NPRD."""
        sections: Dict[NPRDSection, Dict[str, Any]] = {}
        issues: List[str] = []

        # Extract title
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Untitled NPRD"

        # Extract version
        version_match = re.search(r"Version:\s*([\d.]+)", content, re.IGNORECASE)
        version = version_match.group(1) if version_match else "1.0.0"

        # Parse sections
        section_pattern = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
        matches = list(section_pattern.finditer(content))

        for i, match in enumerate(matches):
            section_title = match.group(1).lower().replace(" ", "_")
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()

            # Map to NPRD section
            try:
                section_enum = NPRDSection(section_title)
            except ValueError:
                # Try to find closest match
                section_enum = None
                for s in NPRDSection:
                    if s.value.replace("_", " ") in section_title:
                        section_enum = s
                        break

            if section_enum:
                sections[section_enum] = {
                    "content": section_content,
                    "items": self._extract_list_items(section_content),
                }
            else:
                issues.append(f"Unknown section: {section_title}")

        return ParsedNPRD(
            title=title,
            version=version,
            sections=sections,
            raw_content=content,
            issues=issues,
        )

    def _parse_json(self, content: str) -> ParsedNPRD:
        """Parse JSON format NPRD."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ToolError(
                f"Invalid JSON: {e}",
                tool_name=self.name,
                error_code="INVALID_JSON"
            )

        sections = {}
        for key, value in data.get("sections", {}).items():
            try:
                sections[NPRDSection(key)] = value
            except ValueError:
                pass

        return ParsedNPRD(
            title=data.get("title", "Untitled"),
            version=data.get("version", "1.0.0"),
            sections=sections,
            raw_content=content,
            metadata=data.get("metadata", {}),
        )

    def _parse_yaml(self, content: str) -> ParsedNPRD:
        """Parse YAML format NPRD."""
        # Would require PyYAML
        raise ToolError(
            "YAML parsing not yet implemented",
            tool_name=self.name,
            error_code="NOT_IMPLEMENTED"
        )

    def _extract_list_items(self, content: str) -> List[str]:
        """Extract list items from markdown content."""
        items = []
        for match in re.finditer(r"^\s*[-*+]\s+(.+)$", content, re.MULTILINE):
            items.append(match.group(1).strip())
        return items


@register_tool(
    agent_types=[AgentType.ANALYST],
    tags=["nprd", "validation", "requirements"],
)
class ValidateNPRDTool(QuantMindTool):
    """Validate NPRD structure and content."""

    name: str = "validate_nprd"
    description: str = """Validate NPRD structure, completeness, and content quality.
    Checks for required sections, valid parameters, and logical consistency.
    Returns validation result with issues and suggestions."""

    args_schema: type[BaseModel] = ValidateNPRDInput
    category: ToolCategory = ToolCategory.NPRD_TRD
    priority: ToolPriority = ToolPriority.HIGH

    # Required sections for valid NPRD
    REQUIRED_SECTIONS = [
        NPRDSection.OVERVIEW,
        NPRDSection.OBJECTIVES,
        NPRDSection.TRADING_LOGIC,
        NPRDSection.RISK_MANAGEMENT,
    ]

    # Recommended sections
    RECOMMENDED_SECTIONS = [
        NPRDSection.ENTRY_CONDITIONS,
        NPRDSection.EXIT_CONDITIONS,
        NPRDSection.PARAMETERS,
        NPRDSection.PERFORMANCE_TARGETS,
    ]

    def execute(
        self,
        nprd_content: str,
        strict: bool = False,
        check_completeness: bool = True,
        **kwargs
    ) -> ToolResult:
        """Execute NPRD validation."""
        issues: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        suggestions: List[str] = []

        # Parse content first
        parse_tool = ParseNPRDTool(workspace_path=self._workspace_path)
        parse_result = parse_tool.execute(nprd_content=nprd_content)
        parsed_data = parse_result.data

        present_sections = set(NPRDSection(s) for s in parsed_data.get("sections", {}).keys())

        # Check required sections
        if check_completeness:
            for section in self.REQUIRED_SECTIONS:
                if section not in present_sections:
                    issues.append({
                        "type": "missing_required_section",
                        "section": section.value,
                        "severity": "error",
                        "message": f"Required section '{section.value}' is missing",
                    })

            # Check recommended sections
            for section in self.RECOMMENDED_SECTIONS:
                if section not in present_sections:
                    warnings.append({
                        "type": "missing_recommended_section",
                        "section": section.value,
                        "severity": "warning",
                        "message": f"Recommended section '{section.value}' is missing",
                    })

        # Validate content quality
        for section_key, section_data in parsed_data.get("sections", {}).items():
            content = section_data.get("content", "")
            if len(content) < 50:
                warnings.append({
                    "type": "thin_content",
                    "section": section_key,
                    "severity": "warning",
                    "message": f"Section '{section_key}' has very little content",
                })

        # Generate suggestions
        if issues:
            suggestions.append("Add all required sections to complete the NPRD")
        if not any(s in present_sections for s in [NPRDSection.ENTRY_CONDITIONS, NPRDSection.EXIT_CONDITIONS]):
            suggestions.append("Define clear entry and exit conditions")
        if NPRDSection.PARAMETERS not in present_sections:
            suggestions.append("Add a parameters section to make the strategy configurable")

        is_valid = len([i for i in issues if i.get("severity") == "error"]) == 0

        return ToolResult.ok(
            data={
                "is_valid": is_valid,
                "issues": issues,
                "warnings": warnings,
                "suggestions": suggestions,
                "completeness_score": self._calculate_completeness(parsed_data),
            },
            metadata={
                "validated_at": datetime.now().isoformat(),
                "strict_mode": strict,
            }
        )

    def _calculate_completeness(self, parsed_data: Dict) -> float:
        """Calculate NPRD completeness score (0-100)."""
        all_sections = list(NPRDSection)
        present = len(parsed_data.get("sections", {}))
        return round((present / len(all_sections)) * 100, 1)


@register_tool(
    agent_types=[AgentType.ANALYST],
    tags=["trd", "generation", "requirements"],
)
class GenerateTRDTool(QuantMindTool):
    """Generate Technical Requirements Document from NPRD."""

    name: str = "generate_trd"
    description: str = """Generate a Technical Requirements Document from a validated NPRD.
    Transforms natural language requirements into technical specifications.
    Includes architecture design, component breakdown, and implementation hints."""

    args_schema: type[BaseModel] = GenerateTRDInput
    category: ToolCategory = ToolCategory.NPRD_TRD
    priority: ToolPriority = ToolPriority.HIGH

    def execute(
        self,
        nprd_content: str,
        template: Optional[str] = None,
        include_code_hints: bool = True,
        target_framework: str = "mql5",
        **kwargs
    ) -> ToolResult:
        """Execute TRD generation."""
        # Parse NPRD first
        parse_tool = ParseNPRDTool(workspace_path=self._workspace_path)
        parse_result = parse_tool.execute(nprd_content=nprd_content)
        parsed_nprd = parse_result.data

        # Generate TRD sections
        trd_sections = self._generate_trd_sections(
            parsed_nprd,
            include_code_hints,
            target_framework
        )

        # Generate full TRD content
        trd_content = self._generate_trd_content(
            parsed_nprd,
            trd_sections,
            target_framework
        )

        return ToolResult.ok(
            data={
                "title": f"TRD: {parsed_nprd.get('title', 'Untitled')}",
                "version": "1.0.0",
                "source_nprd_version": parsed_nprd.get("version", "1.0.0"),
                "sections": trd_sections,
                "content": trd_content,
            },
            metadata={
                "generated_at": datetime.now().isoformat(),
                "target_framework": target_framework,
                "include_code_hints": include_code_hints,
            }
        )

    def _generate_trd_sections(
        self,
        parsed_nprd: Dict,
        include_code_hints: bool,
        target_framework: str
    ) -> Dict[str, Any]:
        """Generate TRD sections from parsed NPRD."""
        nprd_sections = parsed_nprd.get("sections", {})

        trd_sections = {}

        # Architecture section
        trd_sections[TRDSection.ARCHITECTURE.value] = {
            "pattern": "event_driven",
            "components": ["signal_generator", "order_manager", "risk_manager", "position_tracker"],
            "description": "Event-driven architecture with modular components",
        }

        # Components section
        components = []

        # Signal Generator
        trading_logic = nprd_sections.get(NPRDSection.TRADING_LOGIC.value, {})
        components.append({
            "name": "SignalGenerator",
            "purpose": "Generate entry and exit signals based on trading logic",
            "inputs": ["price_data", "indicators", "parameters"],
            "outputs": ["signal_type", "signal_strength"],
            "dependencies": [],
        })

        # Order Manager
        components.append({
            "name": "OrderManager",
            "purpose": "Manage order submission, modification, and cancellation",
            "inputs": ["signals", "position_info", "risk_limits"],
            "outputs": ["order_requests", "order_status"],
            "dependencies": ["SignalGenerator"],
        })

        # Risk Manager
        risk_mgmt = nprd_sections.get(NPRDSection.RISK_MANAGEMENT.value, {})
        components.append({
            "name": "RiskManager",
            "purpose": "Enforce risk limits and position sizing",
            "inputs": ["account_info", "position_info", "risk_parameters"],
            "outputs": ["position_size", "risk_status", "alerts"],
            "dependencies": [],
        })

        trd_sections[TRDSection.COMPONENTS.value] = components

        # Data Structures
        trd_sections[TRDSection.DATA_STRUCTURES.value] = {
            "Signal": {
                "type": signal_type for signal_type in ["entry_long", "entry_short", "exit_long", "exit_short", "none"],
                "strength": "float (0.0-1.0)",
                "timestamp": "datetime",
                "price": "float",
            },
            "Position": {
                "symbol": "string",
                "direction": "enum (long, short)",
                "size": "float",
                "entry_price": "float",
                "stop_loss": "float",
                "take_profit": "float",
            },
        }

        # Algorithms section
        trd_sections[TRDSection.ALGORITHMS.value] = {
            "signal_generation": "Implement trading logic from NPRD",
            "position_sizing": "Calculate position size based on risk parameters",
            "stop_loss": "Calculate stop loss based on ATR or fixed percentage",
        }

        # Code hints
        if include_code_hints:
            trd_sections["code_hints"] = self._generate_code_hints(
                nprd_sections,
                target_framework
            )

        return trd_sections

    def _generate_code_hints(
        self,
        nprd_sections: Dict,
        target_framework: str
    ) -> Dict[str, str]:
        """Generate code implementation hints."""
        hints = {}

        if target_framework == "mql5":
            hints["entry_signal"] = """
// Example entry signal structure
bool CheckEntrySignal(int signalType) {
    // Get indicator values
    double ma = iMA(_Symbol, PERIOD_CURRENT, 14, 0, MODE_SMA, PRICE_CLOSE);
    double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);

    // Check conditions based on NPRD logic
    if (signalType == SIGNAL_BUY && currentPrice > ma) {
        return true;
    }
    return false;
}
"""
            hints["position_management"] = """
// Example position sizing
double CalculatePositionSize(double riskPercent, double stopLossPips) {
    double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = accountBalance * (riskPercent / 100.0);
    double pipValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double positionSize = riskAmount / (stopLossPips * pipValue);

    return NormalizeDouble(positionSize, 2);
}
"""
        else:  # Python
            hints["entry_signal"] = """
def check_entry_signal(signal_type: str, df: pd.DataFrame) -> bool:
    # Calculate indicators
    df['ma'] = df['close'].rolling(14).mean()

    # Check conditions based on NPRD logic
    if signal_type == 'BUY':
        return df['close'].iloc[-1] > df['ma'].iloc[-1]
    return False
"""

        return hints

    def _generate_trd_content(
        self,
        parsed_nprd: Dict,
        trd_sections: Dict,
        target_framework: str
    ) -> str:
        """Generate full TRD markdown content."""
        content = f"""# Technical Requirements Document

## {parsed_nprd.get('title', 'Untitled')}

**Version**: 1.0.0
**Source NPRD Version**: {parsed_nprd.get('version', '1.0.0')}
**Target Framework**: {target_framework}
**Generated**: {datetime.now().isoformat()}

---

## 1. Architecture

Pattern: {trd_sections.get('architecture', {}).get('pattern', 'event_driven')}
Components: {', '.join(trd_sections.get('architecture', {}).get('components', []))}

{trd_sections.get('architecture', {}).get('description', '')}

## 2. Components

"""
        for comp in trd_sections.get('components', []):
            content += f"""### {comp['name']}
**Purpose**: {comp['purpose']}
**Inputs**: {', '.join(comp['inputs'])}
**Outputs**: {', '.join(comp['outputs'])}
**Dependencies**: {', '.join(comp['dependencies']) or 'None'}

"""

        content += """## 3. Data Structures

"""
        for ds_name, ds_fields in trd_sections.get('data_structures', {}).items():
            content += f"### {ds_name}\n"
            for field_name, field_type in ds_fields.items():
                content += f"- `{field_name}`: {field_type}\n"
            content += "\n"

        if 'code_hints' in trd_sections:
            content += """## 4. Implementation Hints

"""
            for hint_name, hint_code in trd_sections['code_hints'].items():
                content += f"### {hint_name}\n\n```\n{hint_code}\n```\n\n"

        return content


# Export all tools
__all__ = [
    "ParseNPRDTool",
    "ValidateNPRDTool",
    "GenerateTRDTool",
    "ParsedNPRD",
    "GeneratedTRD",
    "NPRDSection",
    "TRDSection",
]
