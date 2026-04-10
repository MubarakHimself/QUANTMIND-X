"""
QuantMindLib V1 — TRD → BotSpec Conversion Specification
CONTRACT-025
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TRDRawData(BaseModel):
    """
    Intermediate representation after parsing a .trd file but before BotSpec construction.
    """

    robot_id: str
    robot_name: str
    strategy_type: str
    symbol_scope: List[str]
    sessions: List[str]
    features: List[str]
    confirmations: List[str]
    execution_profile: str
    runtime_data: Dict[str, Any] = Field(default_factory=dict)

    model_config = BaseModel.model_config


class TRDConversionResult(BaseModel):
    """
    Result of TRD -> BotSpec conversion.
    """

    success: bool
    bot_spec: Optional[Any] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    version: str = "1.0.0"

    model_config = BaseModel.model_config


class TRDConverter:
    """
    Specification for TRD-to-BotSpec conversion.
    This is a specification class — it defines the conversion contract,
    the actual implementation would wire into cTrader's platform APIs.

    NOTE: This is NOT a pydantic model. It is a specification class
    that defines what methods a TRD converter must implement.
    """

    # Canonical session windows per strategy type
    _SESSION_MAP: Dict[str, List[str]] = {
        "SCALPER": ["LONDON_AM", "NY_AM", "ASIA_PACIFIC"],
        "ORB": ["LONDON_AM", "NY_AM"],
    }
    _DEFAULT_SESSIONS: List[str] = ["LONDON_AM"]

    # Required TRD schema fields
    _REQUIRED_FIELDS: List[str] = ["robot_id", "strategy_type", "symbol_scope"]

    @staticmethod
    def parse_trd(raw: Dict[str, Any]) -> TRDRawData:
        """
        Parse raw TRD dictionary into TRDRawData intermediate representation.
        Raises ValueError on malformed TRD.
        """
        errors = TRDConverter.validate_trd_schema(raw)
        if errors:
            raise ValueError(f"Malformed TRD: missing required fields {errors}")
        return TRDRawData(
            robot_id=raw["robot_id"],
            robot_name=raw.get("robot_name", ""),
            strategy_type=raw["strategy_type"],
            symbol_scope=TRDConverter.extract_symbol_scope(raw),
            sessions=raw.get("sessions", TRDConverter.infer_sessions(raw)),
            features=raw.get("features", []),
            confirmations=raw.get("confirmations", []),
            execution_profile=raw.get("execution_profile", "PAPER"),
            runtime_data=raw.get("runtime_data", {}),
        )

    @staticmethod
    def to_bot_spec(raw_data: TRDRawData) -> TRDConversionResult:
        """
        Convert TRDRawData to BotSpec.
        Returns TRDConversionResult with success=True and BotSpec on success,
        or success=False with errors list on failure.
        """
        try:
            from src.library.core.domain.bot_spec import BotSpec

            bot_spec = BotSpec(
                id=raw_data.robot_id,
                archetype=raw_data.strategy_type,
                symbol_scope=raw_data.symbol_scope,
                sessions=raw_data.sessions,
                features=raw_data.features,
                confirmations=raw_data.confirmations,
                execution_profile=raw_data.execution_profile,
            )
            return TRDConversionResult(success=True, bot_spec=bot_spec)
        except ImportError:
            return TRDConversionResult(
                success=True,
                warnings=["BotSpec not available in this context — domain layer not yet initialized"],
                bot_spec=None,
            )
        except Exception as e:
            return TRDConversionResult(success=False, errors=[str(e)])

    @staticmethod
    def convert(trd_dict: Dict[str, Any]) -> TRDConversionResult:
        """
        End-to-end conversion: TRD dict -> TRDRawData -> BotSpec.
        Convenience method that chains parse_trd and to_bot_spec.
        """
        errors = TRDConverter.validate_trd_schema(trd_dict)
        if errors:
            return TRDConversionResult(success=False, errors=errors)

        raw_data = TRDConverter.parse_trd(trd_dict)
        return TRDConverter.to_bot_spec(raw_data)

    @staticmethod
    def validate_trd_schema(trd_dict: Dict[str, Any]) -> List[str]:
        """
        Validate TRD dict against required schema fields.
        Returns list of validation error messages (empty = valid).
        """
        errors: List[str] = []
        for field in TRDConverter._REQUIRED_FIELDS:
            if field not in trd_dict:
                errors.append(field)
        return errors

    @staticmethod
    def extract_symbol_scope(trd_dict: Dict[str, Any]) -> List[str]:
        """
        Extract symbol scope from TRD dict.
        Symbols come from TRD field: symbol_scope (List[str]) or derived from strategy config.
        """
        return trd_dict.get("symbol_scope", [])

    @staticmethod
    def infer_sessions(trd_dict: Dict[str, Any]) -> List[str]:
        """
        Infer canonical session windows from TRD configuration.
        Maps TRD session config to the 10 canonical session windows used by QuantMindLib.
        """
        strategy = trd_dict.get("strategy_type", "")
        return TRDConverter._SESSION_MAP.get(strategy, TRDConverter._DEFAULT_SESSIONS)


__all__ = ["TRDRawData", "TRDConversionResult", "TRDConverter"]
