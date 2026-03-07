"""
QuantMind Assets Handler

Business logic for shared assets operations.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.api.ide_models import ASSETS_DIR

logger = logging.getLogger(__name__)


class AssetsAPIHandler:
    """Handler for shared assets operations."""

    def __init__(self):
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    def list_assets(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List shared assets."""
        assets = []

        if not ASSETS_DIR.exists():
            return assets

        categories = [category] if category else ["indicators", "libraries", "templates"]

        for cat in categories:
            cat_path = ASSETS_DIR / cat
            if not cat_path.exists():
                continue

            for item in cat_path.rglob("*"):
                if item.is_file():
                    rel_path = item.relative_to(ASSETS_DIR)
                    assets.append({
                        "id": str(rel_path).replace("\\", "/"),
                        "name": item.name,
                        "type": cat,
                        "path": str(item),
                        "description": f"{cat} file",
                        "used_in": [],
                    })

        return assets

    def get_asset_content(self, asset_id: str) -> Optional[str]:
        """Get asset file content."""
        asset_path = ASSETS_DIR / asset_id
        if not asset_path.exists():
            return None

        try:
            return asset_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading asset {asset_id}: {e}")
            return None
