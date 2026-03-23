"""
TRD Document Parser

Parses TRD documents from various formats (JSON, YAML, markdown).
"""
import json
import logging
from typing import Dict, Any, Optional

from src.trd.schema import (
    TRDDocument,
    PositionSizing,
    PositionSizingMethod,
    StrategyType,
    STANDARD_PARAMETERS,
)

logger = logging.getLogger(__name__)


class ParseError(Exception):
    """Exception raised for TRD parsing errors."""
    pass


class TRDParser:
    """
    Parser for Trading Strategy Documents.

    Supports parsing from JSON, YAML, and markdown formats.
    """

    def __init__(self):
        self._parsed_doc: Optional[TRDDocument] = None

    def parse_json(self, json_str: str) -> TRDDocument:
        """
        Parse TRD from JSON string.

        Args:
            json_str: JSON string containing TRD data

        Returns:
            Parsed TRDDocument

        Raises:
            ParseError: If JSON is invalid or missing required fields
        """
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ParseError(f"Invalid JSON: {e}")

        return self._parse_dict(data)

    def parse_dict(self, data: Dict[str, Any]) -> TRDDocument:
        """
        Parse TRD from dictionary.

        Args:
            data: Dictionary containing TRD data

        Returns:
            Parsed TRDDocument
        """
        return self._parse_dict(data)

    def _parse_dict(self, data: Dict[str, Any]) -> TRDDocument:
        """Internal method to parse dictionary to TRDDocument."""
        # Validate required fields
        required_fields = ["strategy_id", "strategy_name"]
        missing = [f for f in required_fields if not data.get(f)]
        if missing:
            raise ParseError(f"Missing required fields: {missing}")

        # Parse position sizing
        position_sizing = None
        if "position_sizing" in data and data["position_sizing"]:
            ps = data["position_sizing"]
            method_str = ps.get("method", "fixed_lot")
            try:
                method = PositionSizingMethod(method_str)
            except ValueError:
                logger.warning(f"Unknown position sizing method: {method_str}, using fixed_lot")
                method = PositionSizingMethod.FIXED_LOT

            position_sizing = PositionSizing(
                method=method,
                risk_percent=ps.get("risk_percent", 1.0),
                max_lots=ps.get("max_lots", 1.0),
                fixed_lot_size=ps.get("fixed_lot_size", 0.01),
            )

        # Parse strategy type
        strategy_type = None
        if "strategy_type" in data and data["strategy_type"]:
            try:
                strategy_type = StrategyType(data["strategy_type"])
            except ValueError:
                logger.warning(f"Unknown strategy type: {data['strategy_type']}, using trend")
                strategy_type = StrategyType.TREND

        # Parse entry/exit conditions
        entry_conditions = data.get("entry_conditions", [])
        if isinstance(entry_conditions, str):
            entry_conditions = [c.strip() for c in entry_conditions.split("\n") if c.strip()]

        exit_conditions = data.get("exit_conditions", [])
        if isinstance(exit_conditions, str):
            exit_conditions = [c.strip() for c in exit_conditions.split("\n") if c.strip()]

        # Parse parameters
        parameters = data.get("parameters", {})

        # Extract known parameters if they're at top level
        known_params = set(STANDARD_PARAMETERS.keys())
        for param_name in list(data.keys()):
            if param_name in known_params and param_name not in parameters:
                parameters[param_name] = data[param_name]

        # Create the document
        doc = TRDDocument(
            strategy_id=data["strategy_id"],
            strategy_name=data["strategy_name"],
            version=data.get("version", 1),
            symbol=data.get("symbol", "EURUSD"),
            timeframe=data.get("timeframe", "H4"),
            strategy_type=strategy_type,
            description=data.get("description", ""),
            entry_conditions=entry_conditions,
            exit_conditions=exit_conditions,
            position_sizing=position_sizing,
            parameters=parameters,
            author=data.get("author", "Research Department"),
            source=data.get("source", "parsed"),
        )

        self._parsed_doc = doc
        logger.info(f"Parsed TRD document: {doc.strategy_id} - {doc.strategy_name}")

        return doc

    def parse_from_research_hypothesis(self, hypothesis_data: Dict[str, Any]) -> TRDDocument:
        """
        Parse TRD from research hypothesis output.

        This creates a basic TRD from a research department hypothesis,
        which can then be enhanced or completed.

        Args:
            hypothesis_data: Dictionary containing hypothesis data

        Returns:
            TRDDocument with basic fields populated
        """
        # Generate strategy_id from symbol and timestamp
        import uuid
        from datetime import datetime

        symbol = hypothesis_data.get("symbol", "EURUSD")
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        strategy_id = f"{symbol.lower()}_{timestamp}_{uuid.uuid4().hex[:8]}"

        # Parse strategy type from hypothesis if available
        strategy_type = StrategyType.TREND
        hypothesis_text = hypothesis_data.get("hypothesis", "").lower()
        if "mean reversion" in hypothesis_text:
            strategy_type = StrategyType.MEAN_REVERSION
        elif "breakout" in hypothesis_text:
            strategy_type = StrategyType.BREAKOUT
        elif "scalp" in hypothesis_text:
            strategy_type = StrategyType.SCALPING

        # Extract entry conditions from hypothesis
        entry_conditions = []
        if "entry" in hypothesis_data:
            entry_conditions = [hypothesis_data["entry"]]

        # Extract supporting evidence as entry conditions
        for i, evidence in enumerate(hypothesis_data.get("supporting_evidence", [])[:3]):
            entry_conditions.append(f"Evidence {i+1}: {evidence[:100]}")

        doc = TRDDocument(
            strategy_id=strategy_id,
            strategy_name=f"{symbol} {strategy_type.value.title()} Strategy",
            symbol=symbol,
            timeframe=hypothesis_data.get("timeframe", "H4"),
            strategy_type=strategy_type,
            description=hypothesis_data.get("hypothesis", ""),
            entry_conditions=entry_conditions,
            exit_conditions=hypothesis_data.get("exit_conditions", []),
            author="Research Department",
            source="research_hypothesis",
            # Default position sizing
            position_sizing=PositionSizing(
                method=PositionSizingMethod.FIXED_LOT,
                risk_percent=1.0,
                max_lots=1.0,
                fixed_lot_size=0.01,
            ),
        )

        self._parsed_doc = doc
        logger.info(f"Created TRD from hypothesis: {doc.strategy_id}")

        return doc

    def get_parsed_document(self) -> Optional[TRDDocument]:
        """Get the last parsed document."""
        return self._parsed_doc
