"""
TRD Generation API - Alpha Forge Pipeline Integration

Provides endpoints for TRD generation from research hypothesis,
validation, and integration with AlphaForgeFlow.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

from src.trd.generator import TRDGenerator, create_trd_from_hypothesis
from src.trd.storage import save_trd, load_trd as load_trd_from_storage
from src.trd.validator import TRDValidator
from src.trd.schema import TRDDocument

router = APIRouter(prefix="/api/trd/generation", tags=["trd-generation"])


# Request/Response models
class HypothesisInput(BaseModel):
    """Research hypothesis input for TRD generation."""
    symbol: str
    timeframe: str
    hypothesis: str
    supporting_evidence: List[str] = []
    confidence_score: float = 0.5
    recommended_next_steps: List[str] = []


class TRDGenerateRequest(BaseModel):
    """Request to generate TRD from hypothesis."""
    hypothesis: HypothesisInput
    strategy_name: Optional[str] = None
    additional_params: Optional[Dict[str, Any]] = None
    run_validation: bool = True
    auto_save: bool = True


class TRDGenerateResponse(BaseModel):
    """Response from TRD generation."""
    success: bool
    strategy_id: str
    trd: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    requires_clarification: bool = False
    is_valid: bool = False
    is_complete: bool = False
    message: str = ""


class TRDValidateRequest(BaseModel):
    """Request to validate existing TRD."""
    strategy_id: str


class TRDValidateResponse(BaseModel):
    """Response from TRD validation."""
    success: bool
    is_valid: bool
    is_complete: bool
    errors: List[Dict[str, Any]] = []
    ambiguities: List[Dict[str, Any]] = []
    missing_parameters: List[str] = []
    requires_clarification: bool = False


# Endpoint implementations
@router.post("/generate", response_model=TRDGenerateResponse)
async def generate_trd(request: TRDGenerateRequest) -> TRDGenerateResponse:
    """
    Generate TRD from research hypothesis.

    This endpoint:
    1. Takes research hypothesis as input
    2. Generates TRD with proper defaults
    3. Validates the TRD
    4. Optionally saves to storage
    5. Returns validation results
    """
    try:
        # Import hypothesis from research head
        from src.agents.departments.heads.research_head import Hypothesis

        # Convert request to Hypothesis dataclass
        hypothesis = Hypothesis(
            symbol=request.hypothesis.symbol,
            timeframe=request.hypothesis.timeframe,
            hypothesis=request.hypothesis.hypothesis,
            supporting_evidence=request.hypothesis.supporting_evidence,
            confidence_score=request.hypothesis.confidence_score,
            recommended_next_steps=request.hypothesis.recommended_next_steps
        )

        # Generate TRD
        generator = TRDGenerator()
        result = generator.generate_and_validate(
            hypothesis=hypothesis,
            strategy_name=request.strategy_name,
            additional_params=request.additional_params
        )

        trd = result["trd"]
        validation_result = result["validation"]

        # Auto-save if requested
        if request.auto_save:
            save_trd(trd)

        return TRDGenerateResponse(
            success=True,
            strategy_id=trd.strategy_id,
            trd=trd.to_dict(),
            validation=validation_result,
            requires_clarification=result["requires_clarification"],
            is_valid=result["is_valid"],
            is_complete=result["is_complete"],
            message="TRD generated successfully"
        )

    except Exception as e:
        return TRDGenerateResponse(
            success=False,
            strategy_id="",
            message=f"TRD generation failed: {str(e)}"
        )


@router.post("/validate", response_model=TRDValidateResponse)
async def validate_trd(request: TRDValidateRequest) -> TRDValidateResponse:
    """
    Validate an existing TRD document.

    Returns validation results with errors, ambiguities, and missing parameters.
    """
    try:
        # Load TRD from storage
        trd = load_trd_from_storage(request.strategy_id)
        if not trd:
            raise HTTPException(
                status_code=404,
                detail=f"TRD not found: {request.strategy_id}"
            )

        # Validate
        validator = TRDValidator()
        result = validator.validate(trd)

        return TRDValidateResponse(
            success=True,
            is_valid=result.is_valid,
            is_complete=result.is_complete,
            errors=[e.to_dict() for e in result.errors],
            ambiguities=[a.to_dict() for a in result.ambiguities],
            missing_parameters=result.missing_parameters,
            requires_clarification=result.requires_clarification()
        )

    except HTTPException:
        raise
    except Exception as e:
        return TRDValidateResponse(
            success=False,
            is_valid=False,
            is_complete=False,
            message=f"Validation failed: {str(e)}"
        )


@router.get("/clarification/{strategy_id}")
async def get_clarification_request(strategy_id: str) -> Dict[str, Any]:
    """
    Get clarification request for a TRD.

    Returns details about what parameters need clarification from FloorManager.
    """
    try:
        # Load TRD
        trd = load_trd_from_storage(strategy_id)
        if not trd:
            raise HTTPException(
                status_code=404,
                detail=f"TRD not found: {strategy_id}"
            )

        # Get clarification request
        generator = TRDGenerator()
        clarification = generator.get_clarification_request(trd)

        return clarification

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get clarification: {str(e)}"
        )


@router.post("/from-hypothesis-simple")
async def generate_trd_simple(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple TRD generation from dictionary input.

    Convenience endpoint for simpler integration scenarios.
    """
    try:
        from src.agents.departments.heads.research_head import Hypothesis

        # Extract hypothesis data
        hypothesis_data = request.get("hypothesis", {})
        hypothesis = Hypothesis(
            symbol=hypothesis_data.get("symbol", "EURUSD"),
            timeframe=hypothesis_data.get("timeframe", "H4"),
            hypothesis=hypothesis_data.get("hypothesis", ""),
            supporting_evidence=hypothesis_data.get("supporting_evidence", []),
            confidence_score=hypothesis_data.get("confidence_score", 0.5),
            recommended_next_steps=hypothesis_data.get("recommended_next_steps", [])
        )

        # Generate
        generator = TRDGenerator()
        result = generator.generate_and_validate(
            hypothesis=hypothesis,
            strategy_name=request.get("strategy_name"),
            additional_params=request.get("additional_params")
        )

        # Save if valid or clarification not needed
        if request.get("auto_save", True) and result["trd"]:
            save_trd(result["trd"])

        return {
            "success": True,
            "trd": result["trd"].to_dict() if result["trd"] else None,
            "validation": result["validation"],
            "requires_clarification": result["requires_clarification"],
            "is_valid": result["is_valid"],
            "is_complete": result["is_complete"]
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }