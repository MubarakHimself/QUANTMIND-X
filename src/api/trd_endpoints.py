"""
TRD (Technical Requirements Document) API Endpoints

Manages TRD documents including creation, retrieval, update,
and deletion. Supports both Vanilla and Spiced variants.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
import json
from pathlib import Path

router = APIRouter(prefix="/api/trd", tags=["trd"])

# TRD storage path
TRD_DIR = Path("./data/trd")
TRD_DIR.mkdir(parents=True, exist_ok=True)

# Data models
class Condition(BaseModel):
    id: str
    name: str
    description: str
    enabled: bool
    parameters: Dict[str, Any]
    sharedAsset: Optional[str] = None

class ExitConfig(BaseModel):
    type: str  # 'fixed', 'atr', 'rr'
    value: float
    trail: bool = False
    trailActivation: Optional[float] = None
    ratio: Optional[float] = None

class RiskConfig(BaseModel):
    mode: str  # 'fixed', 'kelly'
    lotSize: Optional[float] = None
    kellyConfig: Optional[Dict[str, Any]] = None
    squadLimits: bool = True
    maxPositions: int = 3

class PreferredConditions(BaseModel):
    regime: Dict[str, Any]
    correlation: Dict[str, Any]
    houseMoney: Dict[str, Any]

class BacktestRequirements(BaseModel):
    variants: List[str]
    symbols: List[str]
    period: str
    monteCarloRuns: int
    walkForward: Dict[str, Any]

class SharedAsset(BaseModel):
    name: str
    path: str
    version: str

class TRDDocument(BaseModel):
    id: str
    name: str
    version: str
    status: str  # 'draft', 'review', 'approved', 'deployed'
    created: str
    modified: str
    overview: Dict[str, Any]
    entry: Dict[str, Any]  # vanilla and spiced conditions
    exit: Dict[str, Any]  # vanilla and spiced exit configs
    risk: Dict[str, Any]  # vanilla and spiced risk configs
    preferredConditions: PreferredConditions
    backtest: BacktestRequirements
    sharedAssets: List[SharedAsset]

class TRDListResponse(BaseModel):
    trds: List[Dict[str, Any]]
    total: int

# Helper functions
def get_trd_path(trd_id: str) -> Path:
    """Get the file path for a TRD document."""
    return TRD_DIR / f"{trd_id}.json"

def load_trd(trd_id: str) -> Optional[Dict[str, Any]]:
    """Load a TRD document from file."""
    path = get_trd_path(trd_id)
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

def save_trd(trd: Dict[str, Any]):
    """Save a TRD document to file."""
    path = get_trd_path(trd["id"])
    with open(path, 'w') as f:
        json.dump(trd, f, indent=2)

def list_all_trds() -> List[Dict[str, Any]]:
    """List all TRD documents."""
    trds = []
    for path in TRD_DIR.glob("*.json"):
        try:
            with open(path, 'r') as f:
                trd = json.load(f)
                # Return summary only
                trds.append({
                    "id": trd["id"],
                    "name": trd["name"],
                    "version": trd["version"],
                    "status": trd["status"],
                    "created": trd["created"],
                    "modified": trd["modified"]
                })
        except Exception as e:
            print(f"Error loading TRD from {path}: {e}")
    return trds

# Endpoints

@router.get("/")
async def list_trds() -> TRDListResponse:
    """List all TRD documents."""
    trds = list_all_trds()
    return TRDListResponse(trds=trds, total=len(trds))

@router.get("/{trd_id}")
async def get_trd(trd_id: str) -> Dict[str, Any]:
    """Get a specific TRD document."""
    trd = load_trd(trd_id)
    if not trd:
        raise HTTPException(status_code=404, detail="TRD not found")
    return trd

@router.post("/")
async def create_trd(trd: TRDDocument) -> Dict[str, Any]:
    """Create a new TRD document."""
    # Validate TRD doesn't already exist
    if load_trd(trd.id):
        raise HTTPException(status_code=400, detail="TRD with this ID already exists")

    # Set timestamps
    now = datetime.utcnow().isoformat()
    trd.created = now
    trd.modified = now

    # Save to file
    save_trd(trd.dict())

    return trd.dict()

@router.put("/{trd_id}")
async def update_trd(trd_id: str, trd: TRDDocument) -> Dict[str, Any]:
    """Update an existing TRD document."""
    # Check if TRD exists
    existing = load_trd(trd_id)
    if not existing:
        raise HTTPException(status_code=404, detail="TRD not found")

    # Update timestamps
    trd.modified = datetime.utcnow().isoformat()
    trd.created = existing.get("created", trd.created)

    # Save to file
    save_trd(trd.dict())

    return trd.dict()

@router.delete("/{trd_id}")
async def delete_trd(trd_id: str) -> Dict[str, Any]:
    """Delete a TRD document."""
    path = get_trd_path(trd_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="TRD not found")

    path.unlink()
    return {"success": True, "id": trd_id}

@router.post("/{trd_id}/generate-ea")
async def generate_ea(trd_id: str) -> Dict[str, Any]:
    """Generate MQL5 EA code from TRD document."""
    trd = load_trd(trd_id)
    if not trd:
        raise HTTPException(status_code=404, detail="TRD not found")

    # This would trigger the QuantCode agent to generate EA code
    # For now, return a mock response
    return {
        "success": True,
        "message": "EA generation queued",
        "trd_id": trd_id,
        "job_id": str(uuid.uuid4()),
        "estimated_time": "2-5 minutes"
    }

@router.post("/{trd_id}/run-backtest")
async def run_backtest(trd_id: str, variant: str = "spiced+kelly") -> Dict[str, Any]:
    """Run backtest for TRD document."""
    trd = load_trd(trd_id)
    if not trd:
        raise HTTPException(status_code=404, detail="TRD not found")

    # This would trigger the backtesting system
    return {
        "success": True,
        "message": "Backtest queued",
        "trd_id": trd_id,
        "variant": variant,
        "job_id": str(uuid.uuid4()),
        "estimated_time": "5-15 minutes"
    }
