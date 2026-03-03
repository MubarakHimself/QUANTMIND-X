"""
File Endpoints for QuantMind IDE.

Handles file-related operations:
- Shared assets library
- Knowledge hub
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .models import SharedAsset, KnowledgeItem

logger = logging.getLogger(__name__)


# Configuration
DATA_DIR = Path(os.getenv("QUANTMIND_DATA_DIR", "data"))
ASSETS_DIR = DATA_DIR / "shared_assets"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"


class FileEndpoint:
    """File endpoint handler for QuantMind IDE.

    Manages file-related operations including shared assets
    and knowledge hub.
    """

    def __init__(self):
        """Initialize file endpoint."""
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        (ASSETS_DIR / "indicators").mkdir(exist_ok=True)
        (ASSETS_DIR / "libraries").mkdir(exist_ok=True)
        (ASSETS_DIR / "templates").mkdir(exist_ok=True)
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        self._assets_handler = None
        self._knowledge_handler = None

    def _get_assets_handler(self):
        """Lazy load assets handler."""
        if self._assets_handler is None:
            self._assets_handler = AssetsAPIHandler()
        return self._assets_handler

    def _get_knowledge_handler(self):
        """Lazy load knowledge handler."""
        if self._knowledge_handler is None:
            self._knowledge_handler = KnowledgeAPIHandler()
        return self._knowledge_handler

    def list_assets(self, category: Optional[str] = None) -> List[SharedAsset]:
        """List all shared assets, optionally filtered by category."""
        return self._get_assets_handler().list_assets(category)

    def get_asset(self, asset_id: str) -> Optional[SharedAsset]:
        """Get a specific asset."""
        return self._get_assets_handler().get_asset(asset_id)

    def upload_asset(self, name: str, category: str, content: str) -> Dict[str, Any]:
        """Upload a new asset."""
        return self._get_assets_handler().upload_asset(name, category, content)

    def list_knowledge_items(self, category: Optional[str] = None) -> List[KnowledgeItem]:
        """List knowledge items, optionally filtered by category."""
        return self._get_knowledge_handler().list_knowledge_items(category)

    def get_knowledge_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a specific knowledge item."""
        return self._get_knowledge_handler().get_knowledge_item(item_id)

    def create_knowledge_item(self, title: str, content: str, category: str, tags: List[str]) -> Dict[str, Any]:
        """Create a new knowledge item."""
        return self._get_knowledge_handler().create_knowledge_item(title, content, category, tags)

    def search_knowledge(self, query: str) -> List[KnowledgeItem]:
        """Search knowledge items."""
        return self._get_knowledge_handler().search_knowledge(query)


class AssetsAPIHandler:
    """Handler for shared assets library."""

    def __init__(self):
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        (ASSETS_DIR / "indicators").mkdir(exist_ok=True)
        (ASSETS_DIR / "libraries").mkdir(exist_ok=True)
        (ASSETS_DIR / "templates").mkdir(exist_ok=True)

    def list_assets(self, category: Optional[str] = None) -> List[SharedAsset]:
        """List all shared assets, optionally filtered by category."""
        assets = []

        categories = [category] if category else ["indicators", "libraries", "templates"]

        for cat in categories:
            cat_dir = ASSETS_DIR / cat
            if not cat_dir.exists():
                continue

            for asset_file in cat_dir.iterdir():
                if asset_file.is_file() and asset_file.suffix in ['.mqh', '.mq5', '.py']:
                    assets.append(SharedAsset(
                        id=f"{cat}/{asset_file.name}",
                        name=asset_file.stem,
                        type=cat.rstrip('s'),  # indicators -> indicator
                        category=cat,
                        path=str(asset_file),
                        size=asset_file.stat().st_size,
                        created_at=datetime.fromtimestamp(asset_file.stat().st_ctime).isoformat()
                    ))

        return sorted(assets, key=lambda x: x.created_at, reverse=True)

    def get_asset(self, asset_id: str) -> Optional[SharedAsset]:
        """Get a specific asset."""
        parts = asset_id.split('/')
        if len(parts) != 2:
            return None

        category, filename = parts
        asset_file = ASSETS_DIR / category / filename

        if not asset_file.exists():
            return None

        return SharedAsset(
            id=asset_id,
            name=asset_file.stem,
            type=category.rstrip('s'),
            category=category,
            path=str(asset_file),
            size=asset_file.stat().st_size,
            created_at=datetime.fromtimestamp(asset_file.stat().st_ctime).isoformat()
        )

    def upload_asset(self, name: str, category: str, content: str) -> Dict[str, Any]:
        """Upload a new asset."""
        category_dir = ASSETS_DIR / category
        category_dir.mkdir(exist_ok=True)

        # Determine extension
        ext = ".py"  # default
        if category == "indicators":
            ext = ".mqh"
        elif category == "libraries":
            ext = ".mqh"
        elif category == "templates":
            ext = ".mq5"

        filename = f"{name}{ext}"
        file_path = category_dir / filename

        try:
            with open(file_path, "w") as f:
                f.write(content)

            return {
                "success": True,
                "asset_id": f"{category}/{filename}",
                "path": str(file_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class KnowledgeAPIHandler:
    """Handler for knowledge hub."""

    def __init__(self):
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        (KNOWLEDGE_DIR / "articles").mkdir(exist_ok=True)
        (KNOWLEDGE_DIR / "patterns").mkdir(exist_ok=True)
        (KNOWLEDGE_DIR / "strategies").mkdir(exist_ok=True)

    def list_knowledge_items(self, category: Optional[str] = None) -> List[KnowledgeItem]:
        """List knowledge items, optionally filtered by category."""
        items = []

        categories = [category] if category else ["articles", "patterns", "strategies"]

        for cat in categories:
            cat_dir = KNOWLEDGE_DIR / cat
            if not cat_dir.exists():
                continue

            for item_file in cat_dir.iterdir():
                if item_file.is_file() and item_file.suffix == '.json':
                    try:
                        with open(item_file) as f:
                            data = json.load(f)
                        items.append(KnowledgeItem(
                            id=data.get("id", item_file.stem),
                            title=data.get("title", item_file.stem),
                            content=data.get("content", ""),
                            category=cat,
                            tags=data.get("tags", []),
                            created_at=data.get("created_at", datetime.now().isoformat()),
                            updated_at=data.get("updated_at", datetime.now().isoformat())
                        ))
                    except:
                        pass

        return sorted(items, key=lambda x: x.updated_at, reverse=True)

    def get_knowledge_item(self, item_id: str) -> Optional[KnowledgeItem]:
        """Get a specific knowledge item."""
        # Try each category
        for cat in ["articles", "patterns", "strategies"]:
            item_file = KNOWLEDGE_DIR / cat / f"{item_id}.json"
            if item_file.exists():
                try:
                    with open(item_file) as f:
                        data = json.load(f)
                    return KnowledgeItem(
                        id=data.get("id", item_id),
                        title=data.get("title", item_id),
                        content=data.get("content", ""),
                        category=cat,
                        tags=data.get("tags", []),
                        created_at=data.get("created_at", datetime.now().isoformat()),
                        updated_at=data.get("updated_at", datetime.now().isoformat())
                    )
                except:
                    return None
        return None

    def create_knowledge_item(self, title: str, content: str, category: str, tags: List[str]) -> Dict[str, Any]:
        """Create a new knowledge item."""
        import uuid

        category_dir = KNOWLEDGE_DIR / category
        category_dir.mkdir(exist_ok=True)

        item_id = str(uuid.uuid4())[:8]
        filename = f"{item_id}.json"
        file_path = category_dir / filename

        now = datetime.now().isoformat()
        data = {
            "id": item_id,
            "title": title,
            "content": content,
            "category": category,
            "tags": tags,
            "created_at": now,
            "updated_at": now
        }

        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            return {
                "success": True,
                "item_id": item_id,
                "path": str(file_path)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def search_knowledge(self, query: str) -> List[KnowledgeItem]:
        """Search knowledge items."""
        query_lower = query.lower()
        results = []

        for item in self.list_knowledge_items():
            # Search in title, content, and tags
            if (query_lower in item.title.lower() or
                query_lower in item.content.lower() or
                any(query_lower in tag.lower() for tag in item.tags)):
                results.append(item)

        return results
