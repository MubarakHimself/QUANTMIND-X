"""
MQL5 EA Code Generator

Generates MQL5 EA code from TRD documents.
"""
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from src.trd.schema import TRDDocument
from src.mql5.templates.ea_base_template import EABaseTemplate

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """Exception raised for MQL5 generation errors."""
    pass


class MQL5Generator:
    """
    Generates MQL5 EA code from TRD documents.

    Maps TRD parameters to MQL5 input variables and generates
    complete .mq5 file content.
    """

    def __init__(self):
        self._generated_code: Optional[str] = None
        self._template = EABaseTemplate()

    def generate(self, trd: TRDDocument) -> str:
        """
        Generate MQL5 EA code from TRD document.

        Args:
            trd: Validated TRD document

        Returns:
            Generated MQL5 code as string

        Raises:
            GenerationError: If generation fails
        """
        if not trd.strategy_id or not trd.strategy_name:
            raise GenerationError("TRD must have strategy_id and strategy_name")

        try:
            code = self._template.render(trd)
        except ValueError as e:
            raise GenerationError(str(e))

        self._generated_code = code
        logger.info(f"Generated MQL5 code for {trd.strategy_id}")

        return code

    def _prepare_template_variables(self, trd: TRDDocument) -> Dict[str, Any]:
        """Prepare template variables from TRD document."""
        return self._template._prepare_template_variables(trd)

    def _map_parameters(self, trd: TRDDocument, variables: Dict[str, Any]) -> None:
        """Map TRD parameters to template variables."""
        return self._template._map_parameters(trd, variables)

    def save_to_file(self, code: str, file_path: str) -> None:
        """
        Save generated code to file.

        Args:
            code: MQL5 code to save
            file_path: Full path to save file
        """
        import os

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(code)

        logger.info(f"Saved MQL5 code to {file_path}")

    def get_generated_code(self) -> Optional[str]:
        """Get the last generated code."""
        return self._generated_code

    def validate_mql5_syntax(self, code: str) -> Tuple[bool, str]:
        """
        Basic MQL5 syntax validation.

        Note: This is a basic check only. Full validation requires
        the MQL5 compiler.

        Args:
            code: MQL5 code to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for common issues
        lines = code.split('\n')
        brace_count = 0
        paren_count = 0

        for i, line in enumerate(lines, 1):
            # Count braces
            brace_count += line.count('{') - line.count('}')
            paren_count += line.count('(') - line.count(')')

            # Check for common syntax errors
            if ';;' in line:
                return False, f"Line {i}: Double semicolon"

            if '==' in line and 'if(' not in line and 'while(' not in line and 'for(' not in line:
                # Potential assignment error
                if ' = ' not in line and '!=' not in line:
                    pass  # Could be a warning

        if brace_count != 0:
            return False, f"Unmatched braces: {brace_count} difference"

        if paren_count != 0:
            return False, f"Unmatched parentheses: {paren_count} difference"

        # Check for required functions
        required_functions = ['OnInit', 'OnTick', 'OnDeinit']
        for func in required_functions:
            if func not in code:
                return False, f"Missing required function: {func}"

        return True, ""
