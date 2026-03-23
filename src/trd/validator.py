"""
TRD Document Validator

Validates TRD documents for completeness and parameter validity.
Identifies ambiguous or missing parameters.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple

from src.trd.schema import (
    TRDDocument,
    TRDParameter,
    STANDARD_PARAMETERS,
)

logger = logging.getLogger(__name__)


@dataclass
class Ambiguity:
    """Represents an ambiguous parameter in the TRD."""
    parameter_name: str
    current_value: Any
    issue: str
    suggestion: str
    severity: str = "medium"  # high, medium, low

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "current_value": self.current_value,
            "issue": self.issue,
            "suggestion": self.suggestion,
            "severity": self.severity,
        }


@dataclass
class ValidationError:
    """Represents a validation error in the TRD."""
    parameter_name: str
    error: str
    severity: str = "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "error": self.error,
            "severity": self.severity,
        }


@dataclass
class ValidationResult:
    """Result of TRD validation."""
    is_valid: bool
    is_complete: bool
    errors: List[ValidationError] = field(default_factory=list)
    ambiguities: List[Ambiguity] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_parameters: List[str] = field(default_factory=list)

    def has_blocking_issues(self) -> bool:
        """Check if there are blocking issues that prevent EA generation."""
        return len(self.errors) > 0

    def requires_clarification(self) -> bool:
        """Check if clarification is needed from FloorManager."""
        return len(self.ambiguities) > 0 or len(self.missing_parameters) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "is_complete": self.is_complete,
            "errors": [e.to_dict() for e in self.errors],
            "ambiguities": [a.to_dict() for a in self.ambiguities],
            "warnings": self.warnings,
            "missing_parameters": self.missing_parameters,
        }


class TRDValidator:
    """
    Validator for Trading Strategy Documents.

    Checks for:
    - Required parameter presence
    - Parameter value validity
    - Ambiguous or missing values
    - Trading logic completeness
    """

    # Required parameters for MQL5 EA generation
    REQUIRED_PARAMETERS: List[str] = [
        "symbol",
        "timeframe",
    ]

    # Parameters that should have values for EA generation
    RECOMMENDED_PARAMETERS: List[str] = [
        "session_mask",
        "force_close_hour",
        "daily_loss_cap",
        "spread_filter",
        "max_orders",
        "magic_number",
    ]

    def __init__(self):
        self._validation_result: Optional[ValidationResult] = None

    def validate(self, trd: TRDDocument) -> ValidationResult:
        """
        Validate a TRD document.

        Args:
            trd: TRD document to validate

        Returns:
            ValidationResult with all issues identified
        """
        errors: List[ValidationError] = []
        ambiguities: List[Ambiguity] = []
        warnings: List[str] = []
        missing_parameters: List[str] = []

        # Check required core fields
        if not trd.symbol:
            errors.append(ValidationError(
                parameter_name="symbol",
                error="Symbol is required",
            ))

        if not trd.timeframe:
            errors.append(ValidationError(
                parameter_name="timeframe",
                error="Timeframe is required",
            ))

        # Check entry conditions
        if not trd.entry_conditions or len(trd.entry_conditions) == 0:
            errors.append(ValidationError(
                parameter_name="entry_conditions",
                error="At least one entry condition is required",
            ))

        # Validate each parameter
        for param_name, param_value in trd.parameters.items():
            param_errors, param_ambiguities = self._validate_parameter(param_name, param_value)
            errors.extend(param_errors)
            ambiguities.extend(param_ambiguities)

        # Check for missing recommended parameters
        for param_name in self.RECOMMENDED_PARAMETERS:
            if param_name not in trd.parameters or trd.parameters[param_name] is None:
                # Not an error, but flag as ambiguous/missing
                if param_name not in ["symbol", "timeframe"]:  # Already checked
                    missing_parameters.append(param_name)
                    ambiguities.append(Ambiguity(
                        parameter_name=param_name,
                        current_value=None,
                        issue=f"Parameter '{param_name}' is not specified",
                        suggestion=self._get_parameter_suggestion(param_name),
                        severity="low",
                    ))

        # Check position sizing
        if trd.position_sizing:
            ps_errors, ps_ambiguities = self._validate_position_sizing(trd.position_sizing)
            errors.extend(ps_errors)
            ambiguities.extend(ps_ambiguities)

        # Add warnings for common issues
        if not trd.exit_conditions or len(trd.exit_conditions) == 0:
            warnings.append("No exit conditions specified - EA will need default exit logic")

        if not trd.description:
            warnings.append("No strategy description provided")

        # Determine overall validity
        is_valid = len(errors) == 0
        is_complete = len(missing_parameters) == 0 and len(ambiguities) == 0

        result = ValidationResult(
            is_valid=is_valid,
            is_complete=is_complete,
            errors=errors,
            ambiguities=ambiguities,
            warnings=warnings,
            missing_parameters=missing_parameters,
        )

        self._validation_result = result

        # Log validation summary
        if errors:
            logger.error(f"TRD validation errors for {trd.strategy_id}: {len(errors)} found")
        if ambiguities:
            logger.warning(f"TRD ambiguities for {trd.strategy_id}: {len(ambiguities)} found")

        return result

    def _validate_parameter(
        self,
        param_name: str,
        param_value: Any
    ) -> Tuple[List[ValidationError], List[Ambiguity]]:
        """Validate a single parameter."""
        errors: List[ValidationError] = []
        ambiguities: List[Ambiguity] = []

        # Check if parameter is in standard parameters
        if param_name not in STANDARD_PARAMETERS:
            # Unknown parameter - just log and continue
            return errors, ambiguities

        spec = STANDARD_PARAMETERS[param_name]
        param_type = spec.get("type")
        min_val = spec.get("min_value")
        max_val = spec.get("max_value")
        allowed_values = spec.get("allowed_values")

        # Type validation
        if param_value is None:
            return errors, ambiguities

        if param_type == "integer":
            if not isinstance(param_value, int):
                try:
                    param_value = int(param_value)
                except (ValueError, TypeError):
                    errors.append(ValidationError(
                        parameter_name=param_name,
                        error=f"Expected integer, got {type(param_value).__name__}",
                    ))
                    return errors, ambiguities

            # Range validation
            if min_val is not None and param_value < min_val:
                errors.append(ValidationError(
                    parameter_name=param_name,
                    error=f"Value {param_value} is below minimum {min_val}",
                ))

            if max_val is not None and param_value > max_val:
                errors.append(ValidationError(
                    parameter_name=param_name,
                    error=f"Value {param_value} exceeds maximum {max_val}",
                ))

        elif param_type == "float":
            if not isinstance(param_value, (int, float)):
                try:
                    param_value = float(param_value)
                except (ValueError, TypeError):
                    errors.append(ValidationError(
                        parameter_name=param_name,
                        error=f"Expected float, got {type(param_value).__name__}",
                    ))
                    return errors, ambiguities

            # Range validation
            if min_val is not None and param_value < min_val:
                errors.append(ValidationError(
                    parameter_name=param_name,
                    error=f"Value {param_value} is below minimum {min_val}",
                ))

        elif param_type == "boolean":
            if not isinstance(param_value, bool):
                # Try to parse string boolean
                if isinstance(param_value, str):
                    if param_value.lower() in ("true", "1", "yes"):
                        param_value = True
                    elif param_value.lower() in ("false", "0", "no"):
                        param_value = False
                    else:
                        errors.append(ValidationError(
                            parameter_name=param_name,
                            error=f"Invalid boolean value: {param_value}",
                        ))
                        return errors, ambiguities
                else:
                    errors.append(ValidationError(
                        parameter_name=param_name,
                        error=f"Expected boolean, got {type(param_value).__name__}",
                    ))
                    return errors, ambiguities

        # Allowed values validation
        if allowed_values and param_type == "string":
            if param_value not in allowed_values:
                ambiguities.append(Ambiguity(
                    parameter_name=param_name,
                    current_value=param_value,
                    issue=f"Value '{param_value}' is not in expected values: {allowed_values}",
                    suggestion=f"Use one of: {allowed_values}",
                    severity="medium",
                ))

        return errors, ambiguities

    def _validate_position_sizing(
        self,
        position_sizing
    ) -> Tuple[List[ValidationError], List[Ambiguity]]:
        """Validate position sizing configuration."""
        errors: List[ValidationError] = []
        ambiguities: List[Ambiguity] = []

        # Risk percent validation
        if position_sizing.risk_percent is not None:
            if position_sizing.risk_percent <= 0:
                errors.append(ValidationError(
                    parameter_name="position_sizing.risk_percent",
                    error="Risk percent must be positive",
                ))
            elif position_sizing.risk_percent > 10:
                ambiguities.append(Ambiguity(
                    parameter_name="position_sizing.risk_percent",
                    current_value=position_sizing.risk_percent,
                    issue=f"Risk percent of {position_sizing.risk_percent}% is very high",
                    suggestion="Consider using 1-2% for conservative trading",
                    severity="medium",
                ))

        # Max lots validation
        if position_sizing.max_lots is not None:
            if position_sizing.max_lots <= 0:
                errors.append(ValidationError(
                    parameter_name="position_sizing.max_lots",
                    error="Max lots must be positive",
                ))

        # Fixed lot size validation
        if position_sizing.fixed_lot_size is not None:
            if position_sizing.fixed_lot_size <= 0:
                errors.append(ValidationError(
                    parameter_name="position_sizing.fixed_lot_size",
                    error="Fixed lot size must be positive",
                ))
            elif position_sizing.fixed_lot_size < 0.01:
                warnings = ["Lot size below 0.01 may not be supported by broker"]

        return errors, ambiguities

    def _get_parameter_suggestion(self, param_name: str) -> str:
        """Get suggestion for missing parameter."""
        suggestions = {
            "session_mask": "Specify trading sessions (e.g., 'UK/US', 'Asia')",
            "force_close_hour": "Set hour (0-23) to force close all positions",
            "daily_loss_cap": "Set daily loss limit as percentage (e.g., 2.0 for 2%)",
            "spread_filter": "Set maximum spread in points to filter entries",
            "max_orders": "Set maximum concurrent orders",
            "magic_number": "Set unique magic number for EA trades",
        }
        return suggestions.get(param_name, f"Provide value for {param_name}")

    def get_validation_result(self) -> Optional[ValidationResult]:
        """Get the last validation result."""
        return self._validation_result

    def needs_floor_manager_clarification(self, trd: TRDDocument) -> bool:
        """
        Check if TRD needs clarification from FloorManager.

        Args:
            trd: TRD document to check

        Returns:
            True if clarification is needed
        """
        result = self.validate(trd)
        return result.requires_clarification()

    def get_clarification_request(self, trd: TRDDocument) -> Dict[str, Any]:
        """
        Generate a clarification request for FloorManager.

        Args:
            trd: TRD document

        Returns:
            Dictionary with clarification details
        """
        result = self.validate(trd)

        if not result.requires_clarification():
            return {
                "needs_clarification": False,
                "message": "TRD is complete, no clarification needed",
            }

        return {
            "needs_clarification": True,
            "strategy_id": trd.strategy_id,
            "strategy_name": trd.strategy_name,
            "ambiguous_parameters": [a.to_dict() for a in result.ambiguities],
            "missing_parameters": result.missing_parameters,
            "message": f"TRD has {len(result.ambiguities)} ambiguous and {len(result.missing_parameters)} missing parameters requiring clarification",
        }
