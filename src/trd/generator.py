"""
TRD Generator - Creates Trading Requirements Document from Research Hypothesis

Generates TRD documents from ResearchHead hypothesis output, mapping research
findings to the TRD schema with proper defaults for Islamic compliance parameters.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.trd.schema import (
    TRDDocument,
    PositionSizing,
    PositionSizingMethod,
    StrategyType,
)
from src.trd.validator import TRDValidator
from src.agents.departments.heads.research_head import Hypothesis


class TRDGenerator:
    """
    Generates Trading Requirements Document from Research Hypothesis.

    Maps research hypothesis fields to TRD schema and auto-populates
    default Islamic compliance parameters.
    """

    # Default Islamic compliance parameters (must always be present)
    DEFAULT_ISLAMIC_PARAMS = {
        "force_close_hour": 22,  # 10 PM - end of trading day
        "overnight_hold": False,  # Default: don't hold overnight
        "daily_loss_cap": 2.0,  # 2% daily loss limit
    }

    # Default EA parameters
    DEFAULT_EA_PARAMS = {
        "session_mask": "UK/US",
        "spread_filter": 30.0,  # 30 points max spread
        "max_orders": 3,
        "magic_number": 0,  # Will be set dynamically
        "max_lots": 1.0,
        "slippage": 3,
    }

    # Strategy type mapping from hypothesis keywords
    STRATEGY_TYPE_KEYWORDS = {
        "trend": StrategyType.TREND,
        "breakout": StrategyType.BREAKOUT,
        "mean reversion": StrategyType.MEAN_REVERSION,
        "range": StrategyType.MEAN_REVERSION,
        "scalp": StrategyType.SCALPING,
        "swing": StrategyType.SWING,
    }

    def __init__(self):
        self.validator = TRDValidator()

    def generate_trd(
        self,
        hypothesis: Hypothesis,
        strategy_name: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> TRDDocument:
        """
        Generate TRD from research hypothesis.

        Args:
            hypothesis: ResearchHead hypothesis output
            strategy_name: Optional custom strategy name
            additional_params: Optional additional parameters to include

        Returns:
            TRDDocument ready for validation
        """
        # Generate unique strategy_id
        strategy_id = self._generate_strategy_id(hypothesis.symbol)

        # Determine strategy type from hypothesis
        strategy_type = self._infer_strategy_type(hypothesis.hypothesis)

        # Map hypothesis to TRD fields
        strategy_name = strategy_name or f"{hypothesis.symbol}_{hypothesis.timeframe}_Strategy"

        # Build entry conditions from hypothesis
        entry_conditions = self._extract_entry_conditions(hypothesis)

        # Build exit conditions (default)
        exit_conditions = self._get_default_exit_conditions()

        # Create position sizing with defaults
        position_sizing = PositionSizing(
            method=PositionSizingMethod.FIXED_LOT,
            risk_percent=1.0,
            max_lots=1.0,
            fixed_lot_size=0.01
        )

        # Build parameters - merge defaults with hypothesis-specific
        parameters = self._build_parameters(
            hypothesis=hypothesis,
            additional_params=additional_params
        )

        # Create TRD document
        trd = TRDDocument(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            symbol=hypothesis.symbol,
            timeframe=hypothesis.timeframe,
            strategy_type=strategy_type,
            description=hypothesis.hypothesis[:500],  # Truncate if too long
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            position_sizing=position_sizing,
            parameters=parameters,
            author="Research Department",
            source="research_hypothesis"
        )

        return trd

    def _generate_strategy_id(self, symbol: str) -> str:
        """
        Generate unique strategy_id: {symbol}_{timestamp}_{uuid_short}

        Args:
            symbol: Trading symbol (e.g., EURUSD)

        Returns:
            Unique strategy ID
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        uuid_short = uuid.uuid4().hex[:8]
        return f"{symbol}_{timestamp}_{uuid_short}"

    def _infer_strategy_type(self, hypothesis_text: str) -> StrategyType:
        """
        Infer strategy type from hypothesis text keywords.

        Args:
            hypothesis_text: Hypothesis description

        Returns:
            Inferred StrategyType
        """
        hypothesis_lower = hypothesis_text.lower()

        for keywords, strategy_type in self.STRATEGY_TYPE_KEYWORDS.items():
            if keywords in hypothesis_lower:
                return strategy_type

        # Default to trend
        return StrategyType.TREND

    def _extract_entry_conditions(self, hypothesis: Hypothesis) -> List[str]:
        """
        Extract entry conditions from hypothesis and supporting evidence.

        Args:
            hypothesis: Research hypothesis

        Returns:
            List of entry condition descriptions
        """
        conditions = []

        # Add main hypothesis as primary condition
        if hypothesis.hypothesis:
            conditions.append(f"Primary: {hypothesis.hypothesis[:200]}")

        # Add supporting evidence as conditions
        for i, evidence in enumerate(hypothesis.supporting_evidence[:3], 1):
            if evidence:
                # Extract key points from evidence
                evidence_clean = evidence[:150]
                conditions.append(f"Evidence {i}: {evidence_clean}")

        # Ensure at least one condition
        if not conditions:
            conditions.append("Entry conditions to be defined by Development department")

        return conditions

    def _get_default_exit_conditions(self) -> List[str]:
        """
        Get default exit conditions.

        Returns:
            List of default exit conditions
        """
        return [
            "Stop loss at 2% of account balance",
            "Take profit at 3:1 risk-reward ratio",
            "Close all positions before force_close_hour",
        ]

    def _build_parameters(
        self,
        hypothesis: Hypothesis,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build complete parameter set with defaults and hypothesis-specific values.

        Args:
            hypothesis: Research hypothesis
            additional_params: Optional additional parameters

        Returns:
            Complete parameter dictionary
        """
        # Start with default parameters
        params = {
            **self.DEFAULT_EA_PARAMS,
            **self.DEFAULT_ISLAMIC_PARAMS
        }

        # Add hypothesis-specific parameters
        # (Currently hypothesis doesn't have detailed parameters, but prepared for future)

        # Override with additional params if provided
        if additional_params:
            params.update(additional_params)

        # Generate unique magic number based on strategy
        params["magic_number"] = self._generate_magic_number(hypothesis.symbol)

        return params

    def _generate_magic_number(self, symbol: str) -> int:
        """
        Generate unique magic number for the strategy.

        Args:
            symbol: Trading symbol

        Returns:
            Unique magic number
        """
        # Use hash of symbol + timestamp to create unique magic
        symbol_hash = hash(symbol.upper()) % 1000000
        # Ensure positive
        return abs(symbol_hash)

    def generate_and_validate(
        self,
        hypothesis: Hypothesis,
        strategy_name: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate TRD and validate it in one step.

        Args:
            hypothesis: ResearchHead hypothesis
            strategy_name: Optional strategy name
            additional_params: Optional additional parameters

        Returns:
            Dictionary with TRD and validation results
        """
        # Generate TRD
        trd = self.generate_trd(
            hypothesis=hypothesis,
            strategy_name=strategy_name,
            additional_params=additional_params
        )

        # Validate
        validation_result = self.validator.validate(trd)

        # Check if clarification is needed
        requires_clarification = validation_result.requires_clarification()

        return {
            "trd": trd,
            "validation": validation_result.to_dict(),
            "requires_clarification": requires_clarification,
            "is_valid": validation_result.is_valid,
            "is_complete": validation_result.is_complete,
        }

    def get_clarification_request(
        self,
        trd: TRDDocument
    ) -> Dict[str, Any]:
        """
        Generate clarification request for FloorManager if validation found issues.

        Args:
            trd: Generated TRD document

        Returns:
            Clarification request details
        """
        return self.validator.get_clarification_request(trd)


# Module-level convenience function
def create_trd_from_hypothesis(
    hypothesis: Hypothesis,
    strategy_name: Optional[str] = None,
    additional_params: Optional[Dict[str, Any]] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to create TRD from research hypothesis.

    Args:
        hypothesis: ResearchHead hypothesis output
        strategy_name: Optional strategy name
        additional_params: Optional additional parameters
        validate: Whether to validate the generated TRD

    Returns:
        Dictionary with trd and optional validation results
    """
    generator = TRDGenerator()

    if validate:
        return generator.generate_and_validate(
            hypothesis=hypothesis,
            strategy_name=strategy_name,
            additional_params=additional_params
        )
    else:
        trd = generator.generate_trd(
            hypothesis=hypothesis,
            strategy_name=strategy_name,
            additional_params=additional_params
        )
        return {"trd": trd}