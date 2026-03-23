"""
MQL5 Compilation Error Parser

Parses MQL5 compiler output to extract structured error information.
"""
import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# Common MQL5 error patterns
ERROR_PATTERNS = {
    "syntax": re.compile(r"(\d+)\((\d+)\):\s*error:\s*(.+)", re.IGNORECASE),
    "deprecated": re.compile(r"(\d+)\((\d+)\):\s*deprecated:\s*(.+)", re.IGNORECASE),
    "warning": re.compile(r"(\d+)\((\d+)\):\s*warning:\s*(.+)", re.IGNORECASE),
    "constant": re.compile(r"(\d+)\((\d+)\):\s*constant\s+(\w+):\s*(.+)", re.IGNORECASE),
    "implicit": re.compile(r"(\d+)\((\d+)\):\s*implicit\s+(\w+):\s*(.+)", re.IGNORECASE),
}

# Common auto-correctable errors
AUTO_CORRECTABLE_ERRORS = {
    "semicolon": ["';' expected", "';' expected end of statement"],
    "parenthesis": ["')' expected", "'(' expected"],
    "bracket": ["'}' expected", "'{' expected"],
    "quote": ["'\"' expected", "string constant must be closed"],
    "identifier": ["invalid identifier", "unknown identifier"],
    "type_mismatch": ["cannot convert", "wrong parameters count"],
}


@dataclass
class CompilationError:
    """Structured MQL5 compilation error."""
    line: int
    column: int
    error_code: str
    message: str
    severity: str  # error, warning, deprecated
    context: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line": self.line,
            "column": self.column,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
        }

    def is_auto_correctable(self) -> bool:
        """Check if this error can be auto-corrected."""
        message_lower = self.message.lower()
        for error_type, patterns in AUTO_CORRECTABLE_ERRORS.items():
            for pattern in patterns:
                if pattern.lower() in message_lower:
                    return True
        return False

    def get_correction_type(self) -> Optional[str]:
        """Get the type of correction needed."""
        message_lower = self.message.lower()
        for error_type, patterns in AUTO_CORRECTABLE_ERRORS.items():
            for pattern in patterns:
                if pattern.lower() in message_lower:
                    return error_type
        return None


class MQL5ErrorParser:
    """
    Parser for MQL5 compilation output.

    Extracts structured error information from raw compiler output.
    """

    def __init__(self):
        self.errors: List[CompilationError] = []
        self.warnings: List[CompilationError] = []
        self.raw_output: str = ""

    def parse(self, output: str) -> Dict[str, List[CompilationError]]:
        """
        Parse compilation output.

        Args:
            output: Raw output from MQL5 compiler

        Returns:
            Dictionary with 'errors' and 'warnings' lists
        """
        self.raw_output = output
        self.errors = []
        self.warnings = []

        lines = output.split("\n")

        for line in lines:
            # Try each pattern
            error = self._parse_line(line)
            if error:
                if error.severity == "error":
                    self.errors.append(error)
                elif error.severity in ("warning", "deprecated"):
                    self.warnings.append(error)

        logger.info(f"Parsed {len(self.errors)} errors and {len(self.warnings)} warnings")

        return {
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def _parse_line(self, line: str) -> Optional[CompilationError]:
        """Parse a single line of compiler output."""
        line = line.strip()
        if not line:
            return None

        # Try syntax error pattern
        match = ERROR_PATTERNS["syntax"].match(line)
        if match:
            return CompilationError(
                line=int(match.group(1)),
                column=int(match.group(2)),
                error_code="MQL5_ERROR",
                message=match.group(3).strip(),
                severity="error",
            )

        # Try warning pattern
        match = ERROR_PATTERNS["warning"].match(line)
        if match:
            return CompilationError(
                line=int(match.group(1)),
                column=int(match.group(2)),
                error_code="MQL5_WARNING",
                message=match.group(3).strip(),
                severity="warning",
            )

        # Try deprecated pattern
        match = ERROR_PATTERNS["deprecated"].match(line)
        if match:
            return CompilationError(
                line=int(match.group(1)),
                column=int(match.group(2)),
                error_code="MQL5_DEPRECATED",
                message=match.group(3).strip(),
                severity="deprecated",
            )

        # Try implicit conversion
        match = ERROR_PATTERNS["implicit"].match(line)
        if match:
            return CompilationError(
                line=int(match.group(1)),
                column=int(match.group(2)),
                error_code="MQL5_IMPLICIT",
                message=f"{match.group(3)}: {match.group(4)}",
                severity="warning",
            )

        # Check for common error patterns in plain text
        if "error" in line.lower() and ":" in line:
            # Try to extract line number
            line_match = re.search(r"line\s*(\d+)", line, re.IGNORECASE)
            line_num = int(line_match.group(1)) if line_match else 0
            # Try to extract column number
            col_match = re.search(r"column\s*(\d+)", line, re.IGNORECASE)
            col_num = int(col_match.group(1)) if col_match else 0
            return CompilationError(
                line=line_num,
                column=col_num,
                error_code="MQL5_ERROR",
                message=line,
                severity="error",
            )

        return None

    def has_blocking_errors(self) -> bool:
        """Check if there are blocking (non-warning) errors."""
        return len(self.errors) > 0

    def has_warnings_only(self) -> bool:
        """Check if compilation has warnings but no errors."""
        return len(self.errors) == 0 and len(self.warnings) > 0

    def get_error_summary(self) -> str:
        """Get a human-readable error summary."""
        if not self.errors and not self.warnings:
            return "No errors or warnings"

        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")

        return ", ".join(parts)

    def get_errors_by_type(self) -> Dict[str, List[CompilationError]]:
        """Group errors by correction type."""
        categorized = {}
        for error in self.errors:
            correction_type = error.get_correction_type()
            if correction_type:
                if correction_type not in categorized:
                    categorized[correction_type] = []
                categorized[correction_type].append(error)
        return categorized


def parse_compilation_output(output: str) -> Dict[str, List[CompilationError]]:
    """
    Convenience function to parse compilation output.

    Args:
        output: Raw output from MQL5 compiler

    Returns:
        Dictionary with 'errors' and 'warnings' lists
    """
    parser = MQL5ErrorParser()
    return parser.parse(output)


def is_error_auto_correctable(error_message: str) -> bool:
    """
    Check if an error message describes an auto-correctable issue.

    Args:
        error_message: The error message text

    Returns:
        True if the error can be auto-corrected
    """
    message_lower = error_message.lower()
    for error_type, patterns in AUTO_CORRECTABLE_ERRORS.items():
        for pattern in patterns:
            if pattern.lower() in message_lower:
                return True
    return False
