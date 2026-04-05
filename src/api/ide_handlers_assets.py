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
from src.api.knowledge_bootstrap import bootstrap_reference_books
from src.api.wf1_artifacts import WF1_STRATEGIES_ROOT, iter_strategy_roots, strategy_root_has_operator_visible_signal

logger = logging.getLogger(__name__)


class AssetsAPIHandler:
    """Handler for shared assets operations."""

    _COUNT_CATEGORY_ALIASES = {
        "docs": {"docs", "doc", "articles", "article", "books", "book", "libraries", "library", "risk", "utils"},
        "strategy-templates": {"templates", "template", "strategy-templates", "strategy_templates"},
        "indicators": {"indicators", "indicator"},
        "skills": {"skills", "skill"},
        "flow-components": {"flow-components", "flow_components", "flow components"},
        "mcp-configs": {"mcp-configs", "mcp_configs", "mcp configs", "mcp"},
        "strategies": {"strategies", "strategy", "wf1", "workflow-artifacts", "workflow_artifacts"},
    }

    def __init__(self):
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        SCRAPED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
        bootstrap_reference_books(knowledge_dir=KNOWLEDGE_DIR)
        self._meta_dir = ASSETS_DIR / ".meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    def _iter_asset_files(self, category: Optional[str] = None):
        categories = [category] if category else ["indicators", "libraries", "templates", "risk", "docs"]

        for cat in categories:
            if cat == "strategies":
                continue
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

    def _frontmatter_value(self, text: str, key: str) -> Optional[Any]:
        if not text.startswith("---\n"):
            return None
        lines = text.splitlines()
        values: List[str] = []
        in_block = False
        current_key = f"{key}:"
        for line in lines[1:]:
            if line.strip() == "---":
                break
            stripped = line.strip()
            if stripped.startswith(current_key):
                value = stripped[len(current_key):].strip().strip("'\"")
                if value:
                    return value
                in_block = True
                continue
            if in_block:
                if stripped.startswith("- "):
                    values.append(stripped[2:].strip().strip("'\""))
                    continue
                break
        if values:
            return values
        return None

    def _markdown_heading_title(self, text: str) -> Optional[str]:
        if text.startswith("---\n"):
            parts = text.split("\n---\n", 1)
            if len(parts) == 2:
                text = parts[1]
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                if title:
                    return title
        return None

    def _pretty_title(self, name: str) -> str:
        return Path(name).stem.replace("_", " ").replace("-", " ").strip().title()

    def _looks_generated_name(self, name: str) -> bool:
        stem = Path(name).stem.strip().lower()
        if re.fullmatch(r"[0-9a-f]{8,}", stem):
            return True
        if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f-]{27}", stem):
            return True
        return False

    def _asset_display_title(self, item: Path, metadata: Dict[str, Any]) -> str:
        if isinstance(metadata.get("title"), str) and metadata["title"].strip():
            return metadata["title"].strip()

        text = ""
        try:
            if item.suffix.lower() == ".md":
                text = item.read_text(encoding="utf-8")
        except Exception:
            text = ""

        frontmatter_title = self._frontmatter_value(text, "title")
        if isinstance(frontmatter_title, str) and frontmatter_title.strip():
            return frontmatter_title.strip()

        heading_title = self._markdown_heading_title(text)
        if heading_title:
            return heading_title

        return self._pretty_title(item.name)

    def _is_synthetic_scraped_asset(self, rel_path: Path, item: Path, title: str) -> bool:
        if rel_path.parent != Path("."):
            return False

        # Shared Assets should expose the categorized scrape corpus only.
        # Root-level scrape files are legacy scratch/retry artifacts and create
        # the random-number/UUID rows the user flagged in the UI.
        return True

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
            "name": metadata.get("title") or item.name,
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
            metadata={**metadata, "title": self._asset_display_title(item, metadata)},
            description=f"{category} file",
        )

    def _legacy_scraped_asset(self, item: Path) -> Optional[Dict[str, Any]]:
        rel_path = item.relative_to(SCRAPED_ARTICLES_DIR)
        metadata = self._read_any_metadata(item)
        resolved_title = self._asset_display_title(item, metadata)
        if self._is_synthetic_scraped_asset(rel_path, item, resolved_title):
            return None
        if self._looks_generated_name(item.name) and resolved_title == self._pretty_title(item.name):
            return None
        metadata.setdefault("category", "Articles")
        metadata.setdefault("created_by", "QuantCode")
        metadata.setdefault("description", f"Scraped article from {rel_path.parent.name or 'root'}")
        return self._legacy_asset(
            f"scraped/{str(rel_path).replace('\\', '/')}",
            "Articles",
            item,
            metadata={**metadata, "title": resolved_title},
            description=f"Scraped article from {rel_path.parent.name or 'root'}",
        )

    def _resolve_asset_path(self, asset_id: str) -> Optional[Path]:
        if asset_id.startswith("knowledge/"):
            return KNOWLEDGE_DIR / asset_id.removeprefix("knowledge/")
        if asset_id.startswith("scraped/"):
            return SCRAPED_ARTICLES_DIR / asset_id.removeprefix("scraped/")
        if asset_id.startswith("strategies/"):
            return WF1_STRATEGIES_ROOT / asset_id.removeprefix("strategies/")
        return ASSETS_DIR / asset_id

    def _iter_strategy_assets(self):
        for root in iter_strategy_roots() or []:
            if strategy_root_has_operator_visible_signal(root):
                yield root

    def _strategy_asset_metadata(self, root: Path) -> Dict[str, Any]:
        meta_path = root / ".meta.json"
        workflow_path = root / "workflow" / "manifest.json"
        meta: Dict[str, Any] = {}
        workflow_meta: Dict[str, Any] = {}

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to read strategy metadata for {root}: {exc}")
        if workflow_path.exists():
            try:
                workflow_meta = json.loads(workflow_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to read workflow metadata for {root}: {exc}")

        created_at = meta.get("created_at")
        updated_at = meta.get("updated_at") or workflow_meta.get("updated_at")
        stat = root.stat()
        return {
            "name": meta.get("name") or root.name,
            "description": (
                f"{meta.get('strategy_family', 'strategy')} · "
                f"{meta.get('source_bucket', 'workflow')} · "
                f"{meta.get('status', workflow_meta.get('status', 'pending'))}"
            ),
            "version": workflow_meta.get("workflow_id", "")[-8:] or "wf1",
            "created_at": created_at or datetime.fromtimestamp(stat.st_ctime, timezone.utc).isoformat(),
            "updated_at": updated_at or datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "workflow_id": workflow_meta.get("workflow_id"),
            "strategy_id": workflow_meta.get("strategy_id") or root.name,
            "strategy_family": meta.get("strategy_family") or workflow_meta.get("strategy_family"),
            "source_bucket": meta.get("source_bucket") or workflow_meta.get("source_bucket"),
            "status": meta.get("status") or workflow_meta.get("status"),
        }

    def _strategy_tree(self, path: Path, root: Path) -> List[Dict[str, Any]]:
        children: List[Dict[str, Any]] = []
        for item in sorted(path.iterdir(), key=lambda candidate: (candidate.is_file(), candidate.name.lower())):
            rel_path = str(item.relative_to(root)).replace("\\", "/")
            if item.is_dir():
                children.append({
                    "name": item.name,
                    "path": rel_path,
                    "type": "directory",
                    "children": self._strategy_tree(item, root),
                })
            else:
                children.append({
                    "name": item.name,
                    "path": rel_path,
                    "type": "file",
                    "size_bytes": item.stat().st_size,
                    "updated_at": datetime.fromtimestamp(item.stat().st_mtime, timezone.utc).isoformat(),
                })
        return children

    def _legacy_strategy_asset(self, root: Path) -> Dict[str, Any]:
        rel_path = root.relative_to(WF1_STRATEGIES_ROOT)
        metadata = self._strategy_asset_metadata(root)
        return {
            "id": f"strategies/{str(rel_path).replace('\\', '/')}",
            "name": metadata["name"],
            "category": "Strategies",
            "version": metadata["version"],
            "filesystem_path": str(root),
            "dependencies": [],
            "checksum": None,
            "created_by": "QuantCode",
            "used_by_count": 0,
            "created_at": metadata["created_at"],
            "updated_at": metadata["updated_at"],
            "description": metadata["description"],
            "workflow_id": metadata["workflow_id"],
            "strategy_id": metadata["strategy_id"],
            "strategy_family": metadata["strategy_family"],
            "source_bucket": metadata["source_bucket"],
            "status": metadata["status"],
        }

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

        file_categories: Optional[List[str]] = None
        if category == "docs":
            # Shared Assets treats docs as the broader operator-facing knowledge bucket:
            # canonical docs, supporting library files, and risk/runbook utilities all
            # live under the same docs root alongside books/articles.
            file_categories = ["libraries", "risk", "docs"]
        elif category:
            file_categories = [category]

        for cat, item in self._iter_asset_files() if file_categories is None else (
            (cat, item)
            for requested in file_categories
            for cat, item in self._iter_asset_files(requested)
        ):
            rel_path = item.relative_to(ASSETS_DIR)
            metadata = self._read_metadata(rel_path)
            assets.append({
                "id": str(rel_path).replace("\\", "/"),
                "name": self._asset_display_title(item, metadata),
                "type": cat,
                "path": str(item),
                "description": metadata.get("description") or f"{cat} file",
                "used_in": [],
            })

        if category in (None, "books", "docs"):
            for item in self._iter_knowledge_files("books") or []:
                metadata = self._read_any_metadata(item)
                assets.append({
                    "id": f"knowledge/{str(item.relative_to(KNOWLEDGE_DIR)).replace('\\', '/')}",
                    "name": self._asset_display_title(item, metadata),
                    "type": "books",
                    "path": str(item),
                    "description": metadata.get("description") or "books file",
                    "used_in": [],
                })

        if category in (None, "articles", "docs"):
            for item in self._iter_knowledge_files("articles") or []:
                metadata = self._read_any_metadata(item)
                assets.append({
                    "id": f"knowledge/{str(item.relative_to(KNOWLEDGE_DIR)).replace('\\', '/')}",
                    "name": self._asset_display_title(item, metadata),
                    "type": "articles",
                    "path": str(item),
                    "description": metadata.get("description") or "articles file",
                    "used_in": [],
                })
            for item in self._iter_scraped_files() or []:
                rel_path = item.relative_to(SCRAPED_ARTICLES_DIR)
                metadata = self._read_any_metadata(item)
                resolved_title = self._asset_display_title(item, metadata)
                if self._is_synthetic_scraped_asset(rel_path, item, resolved_title):
                    continue
                if self._looks_generated_name(item.name) and resolved_title == self._pretty_title(item.name):
                    continue
                assets.append({
                    "id": f"scraped/{str(rel_path).replace('\\', '/')}",
                    "name": resolved_title,
                    "type": "articles",
                    "path": str(item),
                    "description": metadata.get("description") or "scraped article",
                    "used_in": [],
                })

        if category in (None, "strategies"):
            for root in self._iter_strategy_assets():
                rel_path = root.relative_to(WF1_STRATEGIES_ROOT)
                metadata = self._strategy_asset_metadata(root)
                assets.append({
                    "id": f"strategies/{str(rel_path).replace('\\', '/')}",
                    "name": metadata["name"],
                    "type": "strategies",
                    "path": str(root),
                    "description": metadata["description"],
                    "used_in": [],
                })

        return assets

    def list_assets_legacy(self) -> List[Dict[str, Any]]:
        """List shared assets in the legacy UI shape used by SharedAssetsView."""
        assets = [self._legacy_shared_asset(cat, item) for cat, item in self._iter_asset_files()]
        assets.extend(self._legacy_knowledge_asset(item, "Books") for item in self._iter_knowledge_files("books") or [])
        assets.extend(self._legacy_knowledge_asset(item, "Articles") for item in self._iter_knowledge_files("articles") or [])
        assets.extend(asset for item in self._iter_scraped_files() or [] if (asset := self._legacy_scraped_asset(item)))
        assets.extend(self._legacy_strategy_asset(root) for root in self._iter_strategy_assets())
        assets.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return assets

    def get_asset_counts(self) -> Dict[str, int]:
        """Return canonical shared-asset counts keyed by frontend asset type."""
        counts = {
            "docs": 0,
            "strategy-templates": 0,
            "indicators": 0,
            "skills": 0,
            "flow-components": 0,
            "mcp-configs": 0,
            "strategies": 0,
        }

        for asset in self.list_assets():
            token = str(asset.get("category") or asset.get("type") or "").strip().lower().replace("_", "-")
            for bucket, aliases in self._COUNT_CATEGORY_ALIASES.items():
                if token in aliases:
                    counts[bucket] += 1
                    break

        return counts

    def get_asset_content(self, asset_id: str) -> Optional[str]:
        """Get asset file content."""
        asset_path = self._resolve_asset_path(asset_id)
        if asset_path is None:
            return None
        if not asset_path.exists():
            return None

        if asset_id.startswith("strategies/"):
            metadata = self._strategy_asset_metadata(asset_path)
            strategy_detail = None
            try:
                from src.api.ide_handlers_strategy import StrategyAPIHandler

                strategy_detail = StrategyAPIHandler().get_strategy(asset_id)
            except Exception as exc:
                logger.debug("Strategy detail lookup failed for %s: %s", asset_id, exc)
            payload = {
                "type": "strategy_tree",
                "strategy": metadata,
                "detail": strategy_detail,
                "root": str(asset_path.relative_to(WF1_STRATEGIES_ROOT)).replace("\\", "/"),
                "tree": self._strategy_tree(asset_path, asset_path),
            }
            return json.dumps(payload, indent=2)

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
