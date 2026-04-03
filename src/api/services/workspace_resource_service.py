"""
Workspace Resource Service

Manifest-first workspace resource contract used by chat, tools, and UI.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.api.ide_handlers_assets import AssetsAPIHandler
from src.api.ide_models import ASSETS_DIR, KNOWLEDGE_DIR, SCRAPED_ARTICLES_DIR

logger = logging.getLogger(__name__)


def _normalize_canvas(value: str) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if len(token) >= 2]


def _score_text(haystack: str, tokens: List[str]) -> float:
    if not tokens:
        return 0.0
    score = 0.0
    lowered = haystack.lower()
    for token in tokens:
        if token not in lowered:
            continue
        if lowered.startswith(token):
            score += 3.0
        else:
            score += 1.0
    return score


@dataclass
class WorkspaceResource:
    resource_id: str
    canvas: str
    tab: str
    type: str
    path: str
    label: str
    metadata: Dict[str, Any]
    version: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkspaceResourceService:
    """
    Builds and searches workspace resource descriptors across canvases.
    """

    def __init__(self) -> None:
        self._assets_handler = AssetsAPIHandler()
        self._allowed_roots = [
            ASSETS_DIR.resolve(),
            KNOWLEDGE_DIR.resolve(),
            SCRAPED_ARTICLES_DIR.resolve(),
        ]

    @staticmethod
    def _encode_fs_resource_id(path: Path) -> str:
        encoded = base64.urlsafe_b64encode(str(path).encode("utf-8")).decode("ascii")
        return f"fs:{encoded}"

    @staticmethod
    def _decode_fs_resource_id(resource_id: str) -> Optional[Path]:
        if not resource_id.startswith("fs:"):
            return None
        try:
            payload = resource_id.split(":", 1)[1]
            decoded = base64.urlsafe_b64decode(payload.encode("ascii")).decode("utf-8")
            return Path(decoded)
        except Exception:
            return None

    def _is_path_allowed(self, path: Path) -> bool:
        try:
            resolved = path.resolve()
        except Exception:
            return False
        for root in self._allowed_roots:
            try:
                resolved.relative_to(root)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _canvas_for_category(category: str) -> str:
        normalized = category.strip().lower()
        if normalized in {"articles", "books"}:
            return "research"
        if normalized in {"risk"}:
            return "risk"
        if normalized in {"indicator", "indicators", "signals"}:
            return "trading"
        return "shared-assets"

    @staticmethod
    def _tab_for_category(category: str) -> str:
        normalized = category.strip().lower().replace(" ", "_")
        if normalized.endswith("s"):
            return normalized
        return f"{normalized}s" if normalized else "resources"

    def _iter_asset_resources(self) -> Iterable[WorkspaceResource]:
        assets = self._assets_handler.list_assets_legacy()
        for asset in assets:
            path_raw = str(asset.get("filesystem_path", "")).strip()
            if not path_raw:
                continue
            path = Path(path_raw)
            if not path.exists():
                continue
            if not self._is_path_allowed(path):
                continue

            category = str(asset.get("category", "") or "shared").strip()
            primary_canvas = self._canvas_for_category(category)
            visible_canvases = sorted({primary_canvas, "shared-assets"})
            if primary_canvas == "research":
                visible_canvases = sorted({"research", "shared-assets"})

            source_category = ""
            try:
                if path.is_relative_to(SCRAPED_ARTICLES_DIR.resolve()):
                    rel = path.resolve().relative_to(SCRAPED_ARTICLES_DIR.resolve())
                    source_category = rel.parts[0] if len(rel.parts) > 1 else ""
            except Exception:
                source_category = ""

            resource_id = self._encode_fs_resource_id(path)
            metadata = {
                "asset_id": asset.get("id"),
                "category": category,
                "description": asset.get("description", ""),
                "dependencies": asset.get("dependencies", []),
                "checksum": asset.get("checksum"),
                "visible_canvases": visible_canvases,
                "source_category": source_category,
            }

            yield WorkspaceResource(
                resource_id=resource_id,
                canvas=primary_canvas,
                tab=self._tab_for_category(category),
                type="file",
                path=str(path),
                label=str(asset.get("name", path.name)),
                metadata={k: v for k, v in metadata.items() if v not in ("", None, [])},
                version=str(asset.get("version", "manifest-v1")),
                updated_at=str(asset.get("updated_at", "")),
            )

    def _iter_task_resources(self) -> Iterable[WorkspaceResource]:
        try:
            from src.agents.departments.mail_consumer import get_task_manager
        except Exception:
            return []

        mgr = get_task_manager()
        resources: List[WorkspaceResource] = []
        for department in getattr(mgr, "DEPARTMENTS", []):
            for card in mgr.get_kanban_cards(department):
                payload = card.to_dict()
                card_id = payload.get("id")
                if not card_id:
                    continue
                resources.append(
                    WorkspaceResource(
                        resource_id=f"task:{department}:{card_id}",
                        canvas=_normalize_canvas(department),
                        tab="department-tasks",
                        type="kanban_card",
                        path=f"task://{department}/kanban/{card_id}",
                        label=str(payload.get("title", "Task card")),
                        metadata=payload,
                        version="manifest-v1",
                        updated_at=str(payload.get("updated_at", payload.get("created_at", ""))),
                    )
                )
        return resources

    def list_resources(
        self,
        *,
        canvases: Optional[List[str]] = None,
        tabs: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        canvas_filter = {_normalize_canvas(value) for value in (canvases or []) if value}
        tab_filter = {str(value).strip().lower() for value in (tabs or []) if value}
        type_filter = {str(value).strip().lower() for value in (types or []) if value}

        resources: List[WorkspaceResource] = list(self._iter_asset_resources())
        resources.extend(self._iter_task_resources())

        filtered: List[WorkspaceResource] = []
        for resource in resources:
            visible = {
                _normalize_canvas(resource.canvas),
                *[_normalize_canvas(v) for v in resource.metadata.get("visible_canvases", [])],
            }
            if canvas_filter and visible.isdisjoint(canvas_filter):
                continue
            if tab_filter and resource.tab.lower() not in tab_filter:
                continue
            if type_filter and resource.type.lower() not in type_filter:
                continue
            filtered.append(resource)
            if len(filtered) >= max(1, min(limit, 5000)):
                break

        return [resource.to_dict() for resource in filtered]

    def manifest(
        self,
        *,
        canvases: Optional[List[str]] = None,
        tabs: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        limit: int = 300,
    ) -> Dict[str, Any]:
        resources = self.list_resources(
            canvases=canvases,
            tabs=tabs,
            types=types,
            limit=max(limit, 1),
        )
        by_canvas: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_tab: Dict[str, int] = {}
        for resource in resources:
            by_canvas[resource["canvas"]] = by_canvas.get(resource["canvas"], 0) + 1
            by_type[resource["type"]] = by_type.get(resource["type"], 0) + 1
            by_tab[resource["tab"]] = by_tab.get(resource["tab"], 0) + 1
        return {
            "version": "manifest-v1",
            "strategy": "manifest-first",
            "total_resources": len(resources),
            "by_canvas": by_canvas,
            "by_tab": by_tab,
            "by_type": by_type,
            "resources": resources,
        }

    def search_resources(
        self,
        *,
        query: str,
        canvases: Optional[List[str]] = None,
        tabs: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        tokens = _tokenize(query)
        resources = self.list_resources(canvases=canvases, tabs=tabs, types=types, limit=4000)

        scored: List[tuple[float, Dict[str, Any]]] = []
        for resource in resources:
            score = 0.0
            score += 4.0 * _score_text(str(resource.get("label", "")), tokens)
            score += 3.0 * _score_text(str(resource.get("path", "")), tokens)
            score += 2.0 * _score_text(str(resource.get("tab", "")), tokens)
            score += 2.0 * _score_text(str(resource.get("type", "")), tokens)
            metadata = resource.get("metadata", {})
            if isinstance(metadata, dict):
                score += _score_text(json.dumps(metadata, sort_keys=True), tokens)
            if not tokens:
                score = 1.0
            if score <= 0:
                continue
            enriched = dict(resource)
            enriched["relevance"] = round(min(score / 20.0, 1.0), 4)
            scored.append((score, enriched))

        scored.sort(key=lambda item: item[0], reverse=True)
        deduped: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for _, resource in scored:
            resource_id = str(resource.get("resource_id", ""))
            if not resource_id or resource_id in seen:
                continue
            seen.add(resource_id)
            deduped.append(resource)
            if len(deduped) >= max(1, min(limit, 200)):
                break
        return deduped

    def read_resource(self, resource_id: str, *, max_chars: int = 120_000) -> Dict[str, Any]:
        if resource_id.startswith("task:"):
            parts = resource_id.split(":")
            if len(parts) != 3:
                raise ValueError("Invalid task resource_id")
            _, department, card_id = parts
            try:
                from src.agents.departments.mail_consumer import get_task_manager
            except Exception as exc:
                raise RuntimeError(f"Task manager unavailable: {exc}") from exc

            mgr = get_task_manager()
            card = next(
                (c for c in mgr.get_kanban_cards(department) if c.id == card_id),
                None,
            )
            if card is None:
                raise FileNotFoundError("Task resource not found")
            payload = card.to_dict()
            return {
                "resource_id": resource_id,
                "resource_type": "kanban_card",
                "mime_type": "application/json",
                "text": json.dumps(payload, indent=2),
                "truncated": False,
                "metadata": payload,
            }

        path = self._decode_fs_resource_id(resource_id)
        if path is None:
            raise ValueError("Unsupported resource_id")
        if not path.exists():
            raise FileNotFoundError("Resource does not exist")
        if not self._is_path_allowed(path):
            raise PermissionError("Resource path is outside allowed workspace roots")

        raw = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        truncated = False
        if mime_type.startswith("text/") or mime_type in {"application/json", "application/yaml", "application/xml"}:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
            if len(text) > max_chars:
                text = text[:max_chars]
                truncated = True
            return {
                "resource_id": resource_id,
                "resource_type": "file",
                "mime_type": mime_type,
                "text": text,
                "truncated": truncated,
                "metadata": {
                    "path": str(path),
                    "size_bytes": len(raw),
                },
            }

        # Binary resource: return descriptor only.
        return {
            "resource_id": resource_id,
            "resource_type": "file",
            "mime_type": mime_type,
            "text": "",
            "truncated": False,
            "metadata": {
                "path": str(path),
                "size_bytes": len(raw),
                "binary": True,
            },
        }


_service: Optional[WorkspaceResourceService] = None


def get_workspace_resource_service() -> WorkspaceResourceService:
    global _service
    if _service is None:
        _service = WorkspaceResourceService()
    return _service

