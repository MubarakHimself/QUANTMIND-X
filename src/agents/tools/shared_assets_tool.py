"""
Shared Assets Tool - Share reports, screenshots, and trade logs across the system.
Supports: prop firm reports, screenshots, trade logs
"""

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class AssetCategory:
    """Supported asset categories."""
    PROP_FIRM_REPORT = "prop_firm_report"
    SCREENSHOT = "screenshot"
    TRADE_LOG = "trade_log"
    INDICATOR = "indicator"
    STRATEGY = "strategy"
    BACKTEST_RESULT = "backtest_result"


class SharedAssetsTool:
    """Tool for managing shared assets across the system."""

    SUPPORTED_CATEGORIES = [
        AssetCategory.PROP_FIRM_REPORT,
        AssetCategory.SCREENSHOT,
        AssetCategory.TRADE_LOG,
        AssetCategory.INDICATOR,
        AssetCategory.STRATEGY,
        AssetCategory.BACKTEST_RESULT,
    ]

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB max file size

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the SharedAssetsTool.

        Args:
            storage_path: Path to store assets. Defaults to ~/.quantmindx/shared_assets
        """
        self.storage_path = Path(storage_path or self._get_default_storage_path())
        self.metadata_file = self.storage_path / "metadata.json"
        self._ensure_storage_exists()

    def _get_default_storage_path(self) -> str:
        """Get default storage path."""
        home = os.path.expanduser("~")
        return os.path.join(home, ".quantmindx", "shared_assets")

    def _ensure_storage_exists(self) -> None:
        """Ensure storage directory and metadata exist."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if not self.metadata_file.exists():
            self._save_metadata({})

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from file."""
        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _generate_asset_id(self) -> str:
        """Generate a unique asset ID."""
        return f"asset_{uuid4().hex[:12]}"

    def upload(
        self,
        file_path: str,
        name: str,
        category: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Upload an asset to shared storage.

        Args:
            file_path: Path to the file to upload
            name: Name for the asset
            category: Asset category (prop_firm_report, screenshot, trade_log, indicator, strategy, backtest_result)
            description: Optional description
            tags: Optional list of tags

        Returns:
            Asset metadata including ID, path, and checksum
        """
        source_path = Path(file_path)
        if not source_path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        if not source_path.is_file():
            return {"success": False, "error": "Path is not a file"}

        file_size = source_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            return {
                "success": False,
                "error": f"File too large. Max size: {self.MAX_FILE_SIZE / (1024*1024)}MB",
            }

        if category not in self.SUPPORTED_CATEGORIES:
            return {
                "success": False,
                "error": f"Unsupported category. Supported: {self.SUPPORTED_CATEGORIES}",
            }

        # Create category folder
        category_path = self.storage_path / category
        category_path.mkdir(exist_ok=True)

        # Generate unique filename
        asset_id = self._generate_asset_id()
        extension = source_path.suffix
        dest_filename = f"{asset_id}_{name.replace(' ', '_')}{extension}"
        dest_path = category_path / dest_filename

        # Copy file
        shutil.copy2(source_path, dest_path)

        # Calculate checksum
        checksum = self._calculate_checksum(dest_path)

        # Create metadata
        now = datetime.utcnow().isoformat()
        asset_metadata = {
            "id": asset_id,
            "name": name,
            "category": category,
            "description": description,
            "tags": tags or [],
            "file_path": str(dest_path),
            "filename": dest_filename,
            "checksum": checksum,
            "file_size": file_size,
            "mime_type": self._get_mime_type(extension),
            "created_at": now,
            "updated_at": now,
        }

        # Save metadata
        metadata = self._load_metadata()
        metadata[asset_id] = asset_metadata
        self._save_metadata(metadata)

        logger.info(f"Uploaded asset: {name} ({asset_id})")
        return {"success": True, "asset": asset_metadata}

    def download(self, asset_id: str, destination: str) -> Dict[str, Any]:
        """
        Download an asset from shared storage.

        Args:
            asset_id: ID of the asset to download
            destination: Destination path for the downloaded file

        Returns:
            Download result with success status
        """
        metadata = self._load_metadata()
        if asset_id not in metadata:
            return {"success": False, "error": f"Asset not found: {asset_id}"}

        asset = metadata[asset_id]
        source_path = Path(asset["file_path"])
        if not source_path.exists():
            return {"success": False, "error": "Asset file no longer exists"}

        dest_path = Path(destination)
        shutil.copy2(source_path, dest_path)

        # Verify checksum
        new_checksum = self._calculate_checksum(dest_path)
        if new_checksum != asset["checksum"]:
            return {"success": False, "error": "Checksum verification failed"}

        logger.info(f"Downloaded asset: {asset_id} to {destination}")
        return {
            "success": True,
            "asset": asset,
            "destination": str(dest_path),
            "checksum_verified": True,
        }

    def list(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        List shared assets with optional filtering.

        Args:
            category: Filter by category
            tags: Filter by tags (any match)
            search: Search in name and description

        Returns:
            List of matching assets
        """
        metadata = self._load_metadata()
        assets = list(metadata.values())

        # Filter by category
        if category:
            assets = [a for a in assets if a.get("category") == category]

        # Filter by tags
        if tags:
            assets = [
                a for a in assets if any(tag in a.get("tags", []) for tag in tags)
            ]

        # Search in name and description
        if search:
            search_lower = search.lower()
            assets = [
                a
                for a in assets
                if search_lower in a.get("name", "").lower()
                or search_lower in a.get("description", "").lower()
            ]

        # Sort by most recent first
        assets.sort(key=lambda x: x.get("updated_at", ""), reverse=True)

        # Return summary (exclude file_path for security)
        summary = []
        for a in assets:
            summary.append(
                {
                    "id": a["id"],
                    "name": a["name"],
                    "category": a["category"],
                    "description": a.get("description", ""),
                    "tags": a.get("tags", []),
                    "file_size": a.get("file_size", 0),
                    "checksum": a.get("checksum", ""),
                    "created_at": a.get("created_at", ""),
                    "updated_at": a.get("updated_at", ""),
                }
            )

        return {
            "success": True,
            "count": len(summary),
            "assets": summary,
        }

    def get_asset(self, asset_id: str) -> Dict[str, Any]:
        """Get detailed asset information."""
        metadata = self._load_metadata()
        if asset_id not in metadata:
            return {"success": False, "error": f"Asset not found: {asset_id}"}

        asset = metadata[asset_id].copy()
        # Include file_path for download
        return {"success": True, "asset": asset}

    def delete(self, asset_id: str) -> Dict[str, Any]:
        """Delete an asset from shared storage."""
        metadata = self._load_metadata()
        if asset_id not in metadata:
            return {"success": False, "error": f"Asset not found: {asset_id}"}

        asset = metadata[asset_id]

        # Delete file
        file_path = Path(asset["file_path"])
        if file_path.exists():
            file_path.unlink()

        # Remove from metadata
        del metadata[asset_id]
        self._save_metadata(metadata)

        logger.info(f"Deleted asset: {asset_id}")
        return {"success": True, "deleted": asset_id}

    def update(
        self,
        asset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update asset metadata."""
        metadata = self._load_metadata()
        if asset_id not in metadata:
            return {"success": False, "error": f"Asset not found: {asset_id}"}

        asset = metadata[asset_id]
        if name:
            asset["name"] = name
        if description is not None:
            asset["description"] = description
        if tags is not None:
            asset["tags"] = tags

        asset["updated_at"] = datetime.utcnow().isoformat()
        metadata[asset_id] = asset
        self._save_metadata(metadata)

        return {"success": True, "asset": asset}

    def get_categories(self) -> List[str]:
        """Get list of supported categories."""
        return self.SUPPORTED_CATEGORIES.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        metadata = self._load_metadata()
        assets = list(metadata.values())

        # Count by category
        by_category = {}
        total_size = 0
        for asset in assets:
            cat = asset.get("category", "unknown")
            by_category[cat] = by_category.get(cat, 0) + 1
            total_size += asset.get("file_size", 0)

        return {
            "total_assets": len(assets),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "by_category": by_category,
        }

    @staticmethod
    def _get_mime_type(extension: str) -> str:
        """Get MIME type from file extension."""
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".pdf": "application/pdf",
            ".csv": "text/csv",
            ".json": "application/json",
            ".txt": "text/plain",
            ".html": "text/html",
            ".xml": "application/xml",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".zip": "application/zip",
        }
        return mime_types.get(extension.lower(), "application/octet-stream")


# Category enum for schemas
_CAT_ENUM = ["prop_firm_report", "screenshot", "trade_log", "indicator", "strategy", "backtest_result"]

# Tool schemas for API integration
SHARED_ASSETS_TOOL_SCHEMAS = [
    {"name": "shared_assets_upload", "description": "Upload asset (report, screenshot, trade log)",
     "parameters": {"type": "object", "properties": {
         "file_path": {"type": "string"}, "name": {"type": "string"},
         "category": {"type": "string", "enum": _CAT_ENUM},
         "description": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
         "required": ["file_path", "name", "category"]}},
    {"name": "shared_assets_download", "description": "Download asset from shared storage",
     "parameters": {"type": "object", "properties": {
         "asset_id": {"type": "string"}, "destination": {"type": "string"}},
         "required": ["asset_id", "destination"]}},
    {"name": "shared_assets_list", "description": "List assets with filtering",
     "parameters": {"type": "object", "properties": {
         "category": {"type": "string", "enum": _CAT_ENUM},
         "tags": {"type": "array", "items": {"type": "string"}}, "search": {"type": "string"}}}},
    {"name": "shared_assets_get", "description": "Get asset details",
     "parameters": {"type": "object", "properties": {"asset_id": {"type": "string"}}, "required": ["asset_id"]}},
    {"name": "shared_assets_delete", "description": "Delete asset",
     "parameters": {"type": "object", "properties": {"asset_id": {"type": "string"}}, "required": ["asset_id"]}},
    {"name": "shared_assets_update", "description": "Update asset metadata",
     "parameters": {"type": "object", "properties": {
         "asset_id": {"type": "string"}, "name": {"type": "string"},
         "description": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
         "required": ["asset_id"]}},
    {"name": "shared_assets_categories", "description": "Get supported categories",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "shared_assets_stats", "description": "Get storage statistics",
     "parameters": {"type": "object", "properties": {}}},
]


def get_shared_assets_tool_schemas() -> List[Dict]:
    """Get tool schemas for shared assets."""
    return SHARED_ASSETS_TOOL_SCHEMAS


# Default instance
_default_instance: Optional[SharedAssetsTool] = None


def get_default_instance() -> SharedAssetsTool:
    """Get default SharedAssetsTool instance."""
    global _default_instance
    if _default_instance is None:
        _default_instance = SharedAssetsTool()
    return _default_instance
