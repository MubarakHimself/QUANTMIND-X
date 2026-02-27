"""
Strategy Extraction Tools for extracting strategies from videos/PDFs.

WRITE access for Analysis and Research departments.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedStrategy:
    """Strategy extracted from source."""
    name: str
    description: str
    entry_conditions: List[str]
    exit_conditions: List[str]
    indicators: List[str]
    timeframes: List[str]
    symbols: List[str]
    risk_parameters: Dict[str, Any]
    source_type: str  # video, pdf, text
    source_reference: str


@dataclass
class TradingRequirementsDocument:
    """Trading Requirements Document (TRD)."""
    strategy_name: str
    version: str
    created_at: datetime
    strategy_description: str
    entry_rules: List[str]
    exit_rules: List[str]
    risk_management: Dict[str, Any]
    indicators_required: List[Dict[str, Any]]
    timeframes: List[str]
    recommended_symbols: List[str]
    notes: List[str]


class StrategyExtraction:
    """
    Strategy extraction tools.

    WRITE access for Analysis and Research departments.
    """

    def extract_from_video(
        self,
        video_url: str,
        use_transcript: bool = True,
    ) -> ExtractedStrategy:
        """
        Extract strategy from video URL.

        Args:
            video_url: URL of the video
            use_transcript: Whether to use video transcript

        Returns:
            ExtractedStrategy with strategy details
        """
        # In real implementation, this would:
        # 1. Download video or get transcript
        # 2. Use AI to analyze content
        # 3. Extract strategy parameters

        # Simulated extraction
        return ExtractedStrategy(
            name=f"Strategy from {video_url}",
            description="Strategy extracted from video content",
            entry_conditions=[
                "Price crosses above 50 EMA",
                "RSI is below 30 (oversold)",
            ],
            exit_conditions=[
                "Price crosses below 50 EMA",
                "Take profit at 2x risk",
            ],
            indicators=["EMA50", "RSI"],
            timeframes=["H1", "H4"],
            symbols=["EURUSD", "GBPUSD"],
            risk_parameters={
                "stop_loss_percent": 1.0,
                "take_profit_ratio": 2.0,
                "max_positions": 3,
            },
            source_type="video",
            source_reference=video_url,
        )

    def extract_from_pdf(
        self,
        pdf_path: str,
    ) -> ExtractedStrategy:
        """
        Extract strategy from PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractedStrategy with strategy details
        """
        # In real implementation, this would:
        # 1. Parse PDF content
        # 2. Extract strategy rules
        # 3. Identify indicators and parameters

        return ExtractedStrategy(
            name=f"Strategy from {pdf_path}",
            description="Strategy extracted from PDF document",
            entry_conditions=[
                "MACD line crosses above signal line",
                "Price is above 200 EMA",
            ],
            exit_conditions=[
                "MACD line crosses below signal line",
                "Trailing stop at breakeven",
            ],
            indicators=["MACD", "EMA200"],
            timeframes=["H4", "D1"],
            symbols=["XAUUSD", "XAGUSD"],
            risk_parameters={
                "stop_loss_pips": 50,
                "take_profit_pips": 100,
                "risk_per_trade": 2.0,
            },
            source_type="pdf",
            source_reference=pdf_path,
        )

    def extract_from_text(
        self,
        text: str,
    ) -> ExtractedStrategy:
        """
        Extract strategy from text description.

        Args:
            text: Strategy description text

        Returns:
            ExtractedStrategy with strategy details
        """
        # Parse text for strategy components
        lines = text.split("\n")

        entry_conditions = []
        exit_conditions = []
        current_section = None

        for line in lines:
            line_lower = line.lower().strip()
            if "entry" in line_lower:
                current_section = "entry"
            elif "exit" in line_lower:
                current_section = "exit"
            elif current_section == "entry" and line.strip():
                entry_conditions.append(line.strip())
            elif current_section == "exit" and line.strip():
                exit_conditions.append(line.strip())

        return ExtractedStrategy(
            name="Strategy from Text",
            description=text[:200],
            entry_conditions=entry_conditions or ["Entry condition from text"],
            exit_conditions=exit_conditions or ["Exit condition from text"],
            indicators=["SMA", "RSI"],  # Would parse from text
            timeframes=["M15", "H1"],
            symbols=["EURUSD"],
            risk_parameters={"risk_percent": 1.0},
            source_type="text",
            source_reference=text[:100],
        )

    def generate_trd(
        self,
        strategy: ExtractedStrategy,
    ) -> TradingRequirementsDocument:
        """
        Generate Trading Requirements Document from extracted strategy.

        Args:
            strategy: ExtractedStrategy to convert to TRD

        Returns:
            TradingRequirementsDocument
        """
        return TradingRequirementsDocument(
            strategy_name=strategy.name,
            version="1.0",
            created_at=datetime.now(),
            strategy_description=strategy.description,
            entry_rules=strategy.entry_conditions,
            exit_rules=strategy.exit_conditions,
            risk_management=strategy.risk_parameters,
            indicators_required=[
                {"name": ind, "parameters": {}} for ind in strategy.indicators
            ],
            timeframes=strategy.timeframes,
            recommended_symbols=strategy.symbols,
            notes=[
                f"Extracted from {strategy.source_type}",
                f"Source: {strategy.source_reference}",
            ],
        )

    def validate_strategy(
        self,
        strategy: ExtractedStrategy,
    ) -> Dict[str, Any]:
        """
        Validate extracted strategy for completeness.

        Args:
            strategy: Strategy to validate

        Returns:
            Validation result with issues found
        """
        issues = []
        warnings = []

        # Check required fields
        if not strategy.name:
            issues.append("Strategy name is missing")

        if not strategy.entry_conditions:
            issues.append("No entry conditions defined")

        if not strategy.exit_conditions:
            issues.append("No exit conditions defined")

        if not strategy.indicators:
            warnings.append("No indicators specified")

        if not strategy.risk_parameters:
            warnings.append("No risk parameters defined")

        # Validate risk parameters
        if strategy.risk_parameters:
            if "stop_loss" not in strategy.risk_parameters and "stop_loss_percent" not in strategy.risk_parameters:
                warnings.append("Stop loss not specified")

            if "take_profit" not in strategy.risk_parameters and "take_profit_ratio" not in strategy.risk_parameters:
                warnings.append("Take profit not specified")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "completeness_score": max(0, 100 - len(issues) * 20 - len(warnings) * 5),
        }

    def enhance_strategy(
        self,
        strategy: ExtractedStrategy,
        enhancements: Dict[str, Any],
    ) -> ExtractedStrategy:
        """
        Enhance strategy with additional parameters.

        Args:
            strategy: Base strategy to enhance
            enhancements: Additional parameters to add

        Returns:
            Enhanced ExtractedStrategy
        """
        # Update risk parameters
        if "risk_parameters" in enhancements:
            strategy.risk_parameters.update(enhancements["risk_parameters"])

        # Add indicators
        if "additional_indicators" in enhancements:
            strategy.indicators.extend(enhancements["additional_indicators"])

        # Add timeframes
        if "additional_timeframes" in enhancements:
            strategy.timeframes.extend(enhancements["additional_timeframes"])

        return strategy

    def compare_strategies(
        self,
        strategies: List[ExtractedStrategy],
    ) -> Dict[str, Any]:
        """
        Compare multiple extracted strategies.

        Args:
            strategies: List of strategies to compare

        Returns:
            Comparison summary
        """
        comparison = {
            "strategy_names": [s.name for s in strategies],
            "source_types": [s.source_type for s in strategies],
            "indicator_counts": [len(s.indicators) for s in strategies],
            "timeframe_counts": [len(s.timeframes) for s in strategies],
            "entry_condition_counts": [len(s.entry_conditions) for s in strategies],
            "exit_condition_counts": [len(s.exit_conditions) for s in strategies],
        }

        # Find common indicators across all strategies
        if strategies:
            common_indicators = set(strategies[0].indicators)
            for strategy in strategies[1:]:
                common_indicators &= set(strategy.indicators)
            comparison["common_indicators"] = list(common_indicators)

        return comparison

    def export_trd_to_json(
        self,
        trd: TradingRequirementsDocument,
    ) -> str:
        """
        Export TRD to JSON format.

        Args:
            trd: TradingRequirementsDocument to export

        Returns:
            JSON string
        """
        import json

        data = {
            "strategy_name": trd.strategy_name,
            "version": trd.version,
            "created_at": trd.created_at.isoformat(),
            "strategy_description": trd.strategy_description,
            "entry_rules": trd.entry_rules,
            "exit_rules": trd.exit_rules,
            "risk_management": trd.risk_management,
            "indicators_required": trd.indicators_required,
            "timeframes": trd.timeframes,
            "recommended_symbols": trd.recommended_symbols,
            "notes": trd.notes,
        }

        return json.dumps(data, indent=2)

    def export_trd_to_markdown(
        self,
        trd: TradingRequirementsDocument,
    ) -> str:
        """
        Export TRD to Markdown format.

        Args:
            trd: TradingRequirementsDocument to export

        Returns:
            Markdown string
        """
        md = f"""# {trd.strategy_name}

**Version:** {trd.version}
**Created:** {trd.created_at.strftime('%Y-%m-%d %H:%M')}

## Strategy Description

{trd.strategy_description}

## Entry Rules

"""
        for i, rule in enumerate(trd.entry_rules, 1):
            md += f"{i}. {rule}\n"

        md += "\n## Exit Rules\n\n"
        for i, rule in enumerate(trd.exit_rules, 1):
            md += f"{i}. {rule}\n"

        md += f"\n## Risk Management\n\n"
        for key, value in trd.risk_management.items():
            md += f"- **{key}**: {value}\n"

        md += f"\n## Indicators Required\n\n"
        for ind in trd.indicators_required:
            md += f"- {ind['name']}\n"

        md += f"\n## Timeframes\n\n"
        for tf in trd.timeframes:
            md += f"- {tf}\n"

        md += f"\n## Recommended Symbols\n\n"
        for sym in trd.recommended_symbols:
            md += f"- {sym}\n"

        if trd.notes:
            md += "\n## Notes\n\n"
            for note in trd.notes:
                md += f"- {note}\n"

        return md
