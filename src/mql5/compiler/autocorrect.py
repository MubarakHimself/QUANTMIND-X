"""
MQL5 Auto-Correction Logic

Attempts to automatically fix common MQL5 compilation errors.
Maximum 2 auto-correction attempts before escalating.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.mql5.compiler.error_parser import (
    CompilationError,
    MQL5ErrorParser,
    parse_compilation_output,
    is_error_auto_correctable,
)

logger = logging.getLogger(__name__)

# Maximum auto-correction attempts
MAX_AUTO_CORRECT_ATTEMPTS = 2


@dataclass
class CorrectionResult:
    """Result of an auto-correction attempt."""
    success: bool
    corrected_code: Optional[str]
    corrections_applied: List[str]
    remaining_errors: List[str]
    attempt_number: int


class AutoCorrector:
    """
    Auto-correction for MQL5 compilation errors.

    Attempts to fix common errors like:
    - Missing semicolons
    - Typos in identifiers
    - Type mismatches
    - Missing brackets/quotes
    """

    def __init__(self, max_attempts: int = MAX_AUTO_CORRECT_ATTEMPTS):
        self.max_attempts = max_attempts
        self.error_parser = MQL5ErrorParser()

    def correct(
        self,
        source_code: str,
        errors: List[CompilationError],
        attempt_number: int = 1,
    ) -> CorrectionResult:
        """
        Attempt to auto-correct compilation errors.

        Args:
            source_code: Original MQL5 source code
            errors: List of parsed compilation errors
            attempt_number: Current correction attempt (1 or 2)

        Returns:
            CorrectionResult with corrected code if successful
        """
        if attempt_number > self.max_attempts:
            logger.warning("Max auto-correction attempts reached")
            return CorrectionResult(
                success=False,
                corrected_code=None,
                corrections_applied=[],
                remaining_errors=[e.message for e in errors],
                attempt_number=attempt_number,
            )

        corrections_applied = []
        corrected_code = source_code

        # Process errors by line
        errors_by_line: Dict[int, List[CompilationError]] = {}
        for error in errors:
            if error.is_auto_correctable():
                if error.line not in errors_by_line:
                    errors_by_line[error.line] = []
                errors_by_line[error.line].append(error)

        # Apply corrections line by line
        lines = corrected_code.split("\n")

        for line_num in sorted(errors_by_line.keys()):
            if line_num < 1 or line_num > len(lines):
                continue

            line_errors = errors_by_line[line_num]
            original_line = lines[line_num - 1]

            for error in line_errors:
                correction = self._get_correction(original_line, error)
                if correction:
                    lines[line_num - 1] = correction["new_line"]
                    corrections_applied.append(correction["description"])
                    logger.info(f"Line {line_num}: {correction['description']}")

        corrected_code = "\n".join(lines)

        # Verify the correction
        remaining_errors = []
        # Note: We can't re-compile here as we don't have the compiler
        # The verification will happen in the compilation service

        return CorrectionResult(
            success=len(corrections_applied) > 0,
            corrected_code=corrected_code,
            corrections_applied=corrections_applied,
            remaining_errors=[e.message for e in errors],
            attempt_number=attempt_number,
        )

    def _get_correction(
        self,
        line: str,
        error: CompilationError,
    ) -> Optional[Dict[str, str]]:
        """
        Get the correction for a specific error.

        Args:
            line: The original source line
            error: The compilation error

        Returns:
            Dictionary with new_line and description, or None if not correctable
        """
        correction_type = error.get_correction_type()
        if not correction_type:
            return None

        if correction_type == "semicolon":
            return self._fix_missing_semicolon(line, error)
        elif correction_type == "parenthesis":
            return self._fix_matching_parenthesis(line, error)
        elif correction_type == "bracket":
            return self._fix_matching_bracket(line, error)
        elif correction_type == "quote":
            return self._fix_string_quotes(line, error)
        elif correction_type == "identifier":
            return self._fix_identifier_error(line, error)
        elif correction_type == "type_mismatch":
            return self._fix_type_mismatch(line, error)

        return None

    def _fix_missing_semicolon(self, line: str, error: CompilationError) -> Optional[Dict[str, str]]:
        """Fix missing semicolon at end of statement."""
        line = line.strip()

        # Skip empty lines, comments, and control structure lines
        if not line or line.startswith("//") or line.startswith("#"):
            return None

        # Skip lines that already end with semicolon or opening bracket
        if line.endswith(";") or line.endswith("{") or line.endswith("}"):
            return None

        # Add semicolon
        return {
            "new_line": line + ";",
            "description": f"Added missing semicolon at line {error.line}",
        }

    def _fix_matching_parenthesis(self, line: str, error: CompilationError) -> Optional[Dict[str, str]]:
        """Fix mismatched parentheses."""
        open_count = line.count("(")
        close_count = line.count(")")

        if open_count > close_count:
            # Missing closing parenthesis
            return {
                "new_line": line + ")",
                "description": f"Added missing closing parenthesis at line {error.line}",
            }
        elif close_count > open_count:
            # Extra closing parenthesis - hard to fix automatically
            return None

        return None

    def _fix_matching_bracket(self, line: str, error: CompilationError) -> Optional[Dict[str, str]]:
        """Fix mismatched brackets."""
        open_count = line.count("{")
        close_count = line.count("}")

        if open_count > close_count:
            # Missing closing bracket
            return {
                "new_line": line + "}",
                "description": f"Added missing closing bracket at line {error.line}",
            }
        elif close_count > open_count:
            # Extra closing bracket - hard to fix automatically
            return None

        return None

    def _fix_string_quotes(self, line: str, error: CompilationError) -> Optional[Dict[str, str]]:
        """Fix unclosed string literals."""
        # Check for unclosed string
        quote_count = line.count('"')
        if quote_count % 2 == 1:
            # Add missing closing quote
            return {
                "new_line": line + '"',
                "description": f"Added missing closing quote at line {error.line}",
            }

        return None

    def _fix_identifier_error(self, line: str, error: CompilationError) -> Optional[Dict[str, str]]:
        """Fix common identifier errors (typos)."""
        message_lower = error.message.lower()

        # Common MQL5 typos
        common_typos = {
            "ture": "true",
            "flase": "false",
            "nulll": "NULL",
            "stringg": "string",
            "integerr": "integer",
            "doublle": "double",
            "boollean": "bool",
        }

        for typo, correct in common_typos.items():
            if typo in message_lower:
                # Apply to the line
                new_line = line.replace(typo, correct)
                if new_line != line:
                    return {
                        "new_line": new_line,
                        "description": f"Fixed typo '{typo}' -> '{correct}' at line {error.line}",
                    }

        return None

    def _fix_type_mismatch(self, line: str, error: CompilationError) -> Optional[Dict[str, str]]:
        """Attempt to fix type mismatch errors."""
        message_lower = error.message.lower()

        # Try to fix cast issues
        if "cannot convert" in message_lower:
            # Look for common cast patterns
            # e.g., (int)"123" should be (int)123
            cast_pattern = re.search(r'\((\w+)\)"([^"]+)"', line)
            if cast_pattern:
                cast_type = cast_pattern.group(1)
                value = cast_pattern.group(2)
                new_line = line.replace(cast_pattern.group(0), f"({cast_type}){value}")
                return {
                    "new_line": new_line,
                    "description": f"Fixed type cast at line {error.line}",
                }

        return None

    def auto_correct_with_compilation(
        self,
        source_code: str,
        compiler,
        strategy_id: str,
        version: int,
    ) -> Tuple[CorrectionResult, Optional[str]]:
        """
        Run auto-correction loop with compilation verification.

        Args:
            source_code: Original MQL5 source code
            compiler: DockerMQL5Compiler instance
            strategy_id: Strategy identifier
            version: Version number

        Returns:
            Tuple of (CorrectionResult, corrected_code or None)
        """
        current_code = source_code

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"Auto-correction attempt {attempt}/{self.max_attempts}")

            # Compile current code
            result = compiler.compile(
                mq5_file_path="",  # Will be handled differently
                strategy_id=strategy_id,
                version=version,
                attempt_number=attempt,
            )

            if result.success:
                # Compilation succeeded
                return CorrectionResult(
                    success=True,
                    corrected_code=current_code,
                    corrections_applied=[],
                    remaining_errors=[],
                    attempt_number=attempt,
                ), current_code

            # Parse errors
            error_output = "\n".join(result.errors)
            parsed = parse_compilation_output(error_output)

            if not parsed["errors"]:
                break

            # Attempt correction
            correction_result = self.correct(current_code, parsed["errors"], attempt)

            if correction_result.success and correction_result.corrected_code:
                current_code = correction_result.corrected_code
                logger.info(f"Applied {len(correction_result.corrections_applied)} corrections")
            else:
                # Cannot correct further
                break

        # Return final result
        return CorrectionResult(
            success=False,
            corrected_code=None,
            corrections_applied=[],
            remaining_errors=result.errors if 'result' in locals() else [],
            attempt_number=self.max_attempts,
        ), None


def create_corrector(max_attempts: int = MAX_AUTO_CORRECT_ATTEMPTS) -> AutoCorrector:
    """Factory function to create an AutoCorrector."""
    return AutoCorrector(max_attempts=max_attempts)
