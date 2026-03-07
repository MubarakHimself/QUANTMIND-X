"""
QuantMind IDE Assets Endpoints

API endpoints for shared assets library.
"""

from fastapi import APIRouter, HTTPException
from typing import Optional

from src.api.ide_handlers import AssetsAPIHandler

router = APIRouter(prefix="/api/assets", tags=["assets"])

# Initialize handler
assets_handler = AssetsAPIHandler()


@router.get("")
async def list_assets(category: Optional[str] = None):
    """List shared assets."""
    return assets_handler.list_assets(category)


@router.get("/{asset_id:path}/content")
async def get_asset_content(asset_id: str):
    """Get asset file content."""
    content = assets_handler.get_asset_content(asset_id)
    if content is None:
        raise HTTPException(404, "Asset not found")
    return {"content": content}
