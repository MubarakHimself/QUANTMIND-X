"""
QuantMind IDE Assets Endpoints

API endpoints for shared assets library.
"""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.ide_handlers import AssetsAPIHandler

router = APIRouter(prefix="/api/assets", tags=["assets"])

assets_handler = AssetsAPIHandler()


class CreateAssetRequest(BaseModel):
    name: str
    category: str
    code: str
    description: str = ""
    dependencies: list[str] = []


class RollbackRequest(BaseModel):
    version: str


@router.get("")
async def list_assets(category: Optional[str] = None):
    """List shared assets in the canonical API shape."""
    return assets_handler.list_assets(category)


@router.get("/shared")
async def list_shared_assets():
    """Compatibility route for the SharedAssetsView legacy contract."""
    return assets_handler.list_assets_legacy()


@router.get("/counts")
async def get_asset_counts():
    """Return canonical shared-asset counts for grid/bootstrap UI surfaces."""
    return assets_handler.get_asset_counts()


@router.post("")
async def create_asset(request: CreateAssetRequest):
    """Create a shared asset."""
    try:
        return assets_handler.create_asset(request.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.post("/upload")
async def upload_asset(
    file: UploadFile = File(...),
    category: str = Form(...),
    description: str = Form(""),
    title: str = Form(""),
    author: str = Form(""),
    url: str = Form(""),
):
    """Upload a file-backed shared asset or knowledge item."""
    try:
        if not file.filename:
            raise ValueError("Filename is required")
        content = await file.read()
        return assets_handler.upload_asset(
            filename=file.filename,
            content=content,
            category=category,
            description=description,
            title=title,
            author=author,
            url=url,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.delete("/{asset_id:path}")
async def delete_asset(asset_id: str):
    """Delete a shared asset."""
    if not assets_handler.delete_asset(asset_id):
        raise HTTPException(404, "Asset not found")
    return {"success": True, "asset_id": asset_id}


@router.get("/{asset_id:path}/history")
async def get_asset_history(asset_id: str):
    """Return available history for an asset."""
    history = assets_handler.get_asset_history(asset_id)
    if history is None:
        raise HTTPException(404, "Asset not found")
    return history


@router.post("/{asset_id:path}/rollback")
async def rollback_asset(asset_id: str, request: RollbackRequest):
    """Rollback an asset to a known version when available."""
    try:
        return assets_handler.rollback_asset(asset_id, request.version)
    except FileNotFoundError:
        raise HTTPException(404, "Asset not found")
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/{asset_id:path}/content")
async def get_asset_content(asset_id: str):
    """Get asset file content."""
    content = assets_handler.get_asset_content(asset_id)
    if content is None:
        raise HTTPException(404, "Asset not found")
    return {"content": content}
