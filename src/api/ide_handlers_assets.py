"""
QuantMind Assets Handler

Business logic for shared assets operations.
"""

import hashlib
import json
import logging
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.ide_models import ASSETS_DIR, KNOWLEDGE_DIR, SCRAPED_ARTICLES_DIR

logger = logging.getLogger(__name__)


class AssetsAPIHandler:
    """Handler for shared assets operations."""

    def __init__(self):
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        SCRAPED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        self._meta_dir = ASSETS_DIR / ".meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    def _iter_asset_files(self, category: Optional[str] = None):
        categories = [category] if category else ["indicators", "libraries", "templates", "risk", "docs"]

        for cat in categories:
            cat_path = ASSETS_DIR / cat
            if not cat_path.exists():
                continue

            for item in cat_path.rglob("*"):
                if item.is_file() and ".meta" not in item.parts and not item.name.endswith(".meta.json"):
                    yield cat, item

    def _metadata_path(self, rel_path: Path) -> Path:
        safe_name = str(rel_path).replace("/", "__").replace("\\", "__")
        return self._meta_dir / f"{safe_name}.meta.json"

    def _adjacent_metadata_candidates(self, path: Path) -> List[Path]:
        candidates = [path.parent / f"{path.name}.metadata.json"]
        if path.suffix.lower() == ".md":
            candidates.append(path.with_suffix(".json"))
        return candidates

    def _read_any_metadata(self, path: Path, rel_path: Optional[Path] = None) -> Dict[str, Any]:
        candidates: List[Path] = []
        if rel_path is not None:
            candidates.append(self._metadata_path(rel_path))
        candidates.extend(self._adjacent_metadata_candidates(path))

        for meta_path in candidates:
            if not meta_path.exists():
                continue
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to read asset metadata for {path}: {exc}")
        return {}

    def _read_metadata(self, rel_path: Path) -> Dict[str, Any]:
        return self._read_any_metadata(ASSETS_DIR / rel_path, rel_path)

    def _write_metadata(self, rel_path: Path, payload: Dict[str, Any]) -> None:
        meta_path = self._metadata_path(rel_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_adjacent_metadata(self, path: Path, payload: Dict[str, Any]) -> None:
        meta_path = path.parent / f"{path.name}.metadata.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _compute_checksum(self, path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _legacy_category(self, folder: str, item: Path) -> str:
        if folder == "indicators":
            return "Indicator"
        if folder == "docs":
            return "Docs"
        if folder == "risk" or "risk" in item.parts or "risk" in item.stem.lower():
            return "Risk"
        return "Utils"

    def _legacy_asset(
        self,
        asset_id: str,
        category: str,
        item: Path,
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        stat = item.stat()
        metadata = metadata or {}
        checksum = self._compute_checksum(item)
        created_by = metadata.get("created_by", "user")
        if created_by == "QuantCode":
            created_by_value = "QuantCode"
        else:
            created_by_value = "user"

        return {
            "id": asset_id,
            "name": item.name,
            "category": metadata.get("category") or category,
            "version": metadata.get("version") or checksum[:8],
            "filesystem_path": str(item),
            "dependencies": metadata.get("dependencies", []),
            "checksum": checksum,
            "created_by": created_by_value,
            "used_by_count": metadata.get("used_by_count", 0),
            "created_at": metadata.get("created_at") or datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
            "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "description": metadata.get("description", description or f"{category} file"),
        }

    def _legacy_shared_asset(self, folder: str, item: Path) -> Dict[str, Any]:
        rel_path = item.relative_to(ASSETS_DIR)
        metadata = self._read_metadata(rel_path)
        return self._legacy_asset(
            str(rel_path).replace("\\", "/"),
            metadata.get("category") or self._legacy_category(folder, item),
            item,
            metadata=metadata,
            description=f"{folder} file",
        )

    def _iter_knowledge_files(self, category: str):
        root = KNOWLEDGE_DIR / category
        if not root.exists():
            return
        for item in root.rglob("*"):
            if item.is_file() and not item.name.endswith(".metadata.json"):
                yield item

    def _iter_scraped_files(self):
        if not SCRAPED_ARTICLES_DIR.exists():
            return
        for item in SCRAPED_ARTICLES_DIR.rglob("*.md"):
            yield item

    def _legacy_knowledge_asset(self, item: Path, category: str) -> Dict[str, Any]:
        rel_path = item.relative_to(KNOWLEDGE_DIR)
        metadata = self._read_any_metadata(item)
        metadata.setdefault("category", category)
        metadata.setdefault("created_by", "user")
        metadata.setdefault("description", f"{category} file")
        return self._legacy_asset(
            f"knowledge/{str(rel_path).replace('\\', '/')}",
            category,
            item,
            metadata=metadata,
            description=f"{category} file",
        )

    def _legacy_scraped_asset(self, item: Path) -> Dict[str, Any]:
        rel_path = item.relative_to(SCRAPED_ARTICLES_DIR)
        metadata = self._read_any_metadata(item)
        metadata.setdefault("category", "Articles")
        metadata.setdefault("created_by", "QuantCode")
        metadata.setdefault("description", f"Scraped article from {rel_path.parent.name or 'root'}")
        return self._legacy_asset(
            f"scraped/{str(rel_path).replace('\\', '/')}",
            "Articles",
            item,
            metadata=metadata,
            description=f"Scraped article from {rel_path.parent.name or 'root'}",
        )

    def _resolve_asset_path(self, asset_id: str) -> Optional[Path]:
        if asset_id.startswith("knowledge/"):
            return KNOWLEDGE_DIR / asset_id.removeprefix("knowledge/")
        if asset_id.startswith("scraped/"):
            return SCRAPED_ARTICLES_DIR / asset_id.removeprefix("scraped/")
        return ASSETS_DIR / asset_id

    def _delete_metadata_sidecars(self, asset_path: Path, asset_id: str) -> None:
        if asset_id.startswith("knowledge/") or asset_id.startswith("scraped/"):
            for meta_path in self._adjacent_metadata_candidates(asset_path):
                if meta_path.exists():
                    meta_path.unlink()
            return

        rel_path = asset_path.relative_to(ASSETS_DIR)
        meta_path = self._metadata_path(rel_path)
        if meta_path.exists():
            meta_path.unlink()
        for adjacent in self._adjacent_metadata_candidates(asset_path):
            if adjacent.exists():
                adjacent.unlink()

    def list_assets(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List shared assets."""
        assets = []

        for cat, item in self._iter_asset_files(category):
            assets.append({
                "id": str(item.relative_to(ASSETS_DIR)).replace("\\", "/"),
                "name": item.name,
                "type": cat,
                "path": str(item),
                "description": f"{cat} file",
                "used_in": [],
            })

        if category in (None, "books"):
            for item in self._iter_knowledge_files("books") or []:
                assets.append({
                    "id": f"knowledge/{str(item.relative_to(KNOWLEDGE_DIR)).replace('\\', '/')}",
                    "name": item.name,
                    "type": "books",
                    "path": str(item),
                    "description": "books file",
                    "used_in": [],
                })

        if category in (None, "articles"):
            for item in self._iter_knowledge_files("articles") or []:
                assets.append({
                    "id": f"knowledge/{str(item.relative_to(KNOWLEDGE_DIR)).replace('\\', '/')}",
                    "name": item.name,
                    "type": "articles",
                    "path": str(item),
                    "description": "articles file",
                    "used_in": [],
                })
            for item in self._iter_scraped_files() or []:
                assets.append({
                    "id": f"scraped/{str(item.relative_to(SCRAPED_ARTICLES_DIR)).replace('\\', '/')}",
                    "name": item.name,
                    "type": "articles",
                    "path": str(item),
                    "description": "scraped article",
                    "used_in": [],
                })

        return assets

    def list_assets_legacy(self) -> List[Dict[str, Any]]:
        """List shared assets in the legacy UI shape used by SharedAssetsView."""
        assets = [self._legacy_shared_asset(cat, item) for cat, item in self._iter_asset_files()]
        assets.extend(self._legacy_knowledge_asset(item, "Books") for item in self._iter_knowledge_files("books") or [])
        assets.extend(self._legacy_knowledge_asset(item, "Articles") for item in self._iter_knowledge_files("articles") or [])
        assets.extend(self._legacy_scraped_asset(item) for item in self._iter_scraped_files() or [])
        assets.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return assets

    def get_asset_content(self, asset_id: str) -> Optional[str]:
        """Get asset file content."""
        asset_path = self._resolve_asset_path(asset_id)
        if asset_path is None:
            return None
        if not asset_path.exists():
            return None

        try:
            return asset_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            file_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
            return json.dumps(
                {
                    "type": "binary",
                    "mime_type": file_type,
                    "path": str(asset_path),
                    "size_bytes": asset_path.stat().st_size,
                },
                indent=2,
            )
        except Exception as e:
            logger.error(f"Error reading asset {asset_id}: {e}")
            return None

    def create_asset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Create a shared asset and return legacy UI metadata."""
        category = str(payload.get("category", "Utils"))
        code = payload.get("code", "")
        name = str(payload.get("name", "")).strip()
        if not name:
            raise ValueError("Asset name is required")

        folder_map = {
            "Indicator": "indicators",
            "Risk": "risk",
            "Utils": "libraries",
        }
        folder = folder_map.get(category, "libraries")
        target_dir = ASSETS_DIR / folder
        target_dir.mkdir(parents=True, exist_ok=True)

        file_name = name if Path(name).suffix else f"{re.sub(r'[^A-Za-z0-9._-]+', '_', name)}.txt"
        asset_path = target_dir / file_name
        asset_path.write_text(code, encoding="utf-8")

        rel_path = asset_path.relative_to(ASSETS_DIR)
        self._write_metadata(rel_path, {
            "category": category,
            "description": payload.get("description", ""),
            "dependencies": payload.get("dependencies", []),
            "created_by": "user",
        })
        return self._legacy_shared_asset(folder, asset_path)

    def upload_asset(
        self,
        *,
        filename: str,
        content: bytes,
        category: str,
        description: str = "",
        title: str = "",
        author: str = "",
        url: str = "",
    ) -> Dict[str, Any]:
        safe_filename = Path(filename).name
        if not safe_filename:
            raise ValueError("Filename is required")

        upload_targets = {
            "Books": KNOWLEDGE_DIR / "books",
            "Articles": KNOWLEDGE_DIR / "articles",
            "Docs": ASSETS_DIR / "docs",
        }
        target_dir = upload_targets.get(category)
        if target_dir is None:
            raise ValueError(f"Unsupported upload category: {category}")

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_filename
        target_path.write_bytes(content)

        metadata = {
            "category": category,
            "description": description,
            "created_by": "user",
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }
        if title:
            metadata["title"] = title
        if author:
            metadata["author"] = author
        if url:
            metadata["url"] = url

        if target_path.is_relative_to(ASSETS_DIR):
            rel_path = target_path.relative_to(ASSETS_DIR)
            self._write_metadata(rel_path, metadata)
            return self._legacy_shared_asset(rel_path.parts[0], target_path)

        self._write_adjacent_metadata(target_path, metadata)
        return self._legacy_knowledge_asset(target_path, category)

    def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset and its metadata sidecar."""
        asset_path = self._resolve_asset_path(asset_id)
        if asset_path is None:
            return False
        if not asset_path.exists():
            return False

        asset_path.unlink()
        self._delete_metadata_sidecars(asset_path, asset_id)
        return True

    def get_asset_history(self, asset_id: str) -> Optional[List[Dict[str, Any]]]:
        """Return a single-entry history based on the current file version."""
        asset_path = self._resolve_asset_path(asset_id)
        if asset_path is None:
            return None
        if not asset_path.exists():
            return None

        rel_path = asset_path.relative_to(ASSETS_DIR) if asset_path.is_relative_to(ASSETS_DIR) else None
        metadata = self._read_any_metadata(asset_path, rel_path)
        checksum = self._compute_checksum(asset_path)
        stat = asset_path.stat()

        return [{
            "version": metadata.get("version") or checksum[:8],
            "checksum": checksum,
            "created_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "created_by": metadata.get("created_by", "user"),
            "change_description": metadata.get("description") or "Current version",
        }]

    def rollback_asset(self, asset_id: str, version: str) -> Dict[str, Any]:
        """No-op rollback when the requested version matches the current one."""
        history = self.get_asset_history(asset_id)
        if not history:
            raise FileNotFoundError(asset_id)
        current = history[0]
        if version != current["version"]:
            raise ValueError("Requested version is not available for rollback")
        return {"success": True, "asset_id": asset_id, "version": version, "status": "noop"}
