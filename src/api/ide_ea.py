"""
QuantMind IDE EA endpoints.

Legacy compatibility surface for EA inventory. Production mode only:
- scans real WF1/shared-assets strategy trees and compiled outputs
- never returns fabricated EAs
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query

from src.api.pagination import DEFAULT_LIMIT, DEFAULT_OFFSET, MAX_LIMIT, PaginatedResponse
from src.api.wf1_artifacts import iter_strategy_roots

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ide/eas", tags=["ea"])


def _collect_ea_files() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for root in iter_strategy_roots() or []:
        for section_name in ("development", "variants", "compilation"):
            section = root / section_name
            if not section.exists():
                continue
            for pattern in ("*.mq5", "*.ex5"):
                for file_path in section.rglob(pattern):
                    key = (root.name, str(file_path))
                    if key in seen:
                        continue
                    seen.add(key)
                    stat = file_path.stat()
                    rows.append(
                        {
                            "id": file_path.stem,
                            "name": file_path.stem,
                            "symbol": "",
                            "timeframe": "",
                            "status": "compiled" if file_path.suffix.lower() == ".ex5" else "ready",
                            "deployed": False,
                            "profit": None,
                            "trades": None,
                            "win_rate": None,
                            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
                            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                            "file_path": str(file_path),
                            "strategy_id": root.name,
                            "relative_path": str(file_path.relative_to(root)).replace("\\", "/"),
                        }
                    )
    rows.sort(key=lambda item: item["modified_at"], reverse=True)
    return rows


def _find_ea(ea_id: str) -> Dict[str, Any] | None:
    for item in _collect_ea_files():
        if item["id"] == ea_id:
            return item
    return None


def _load_json_sibling(path: Path, name: str) -> Dict[str, Any] | None:
    candidate = path.with_name(name)
    if not candidate.exists():
        return None
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("", response_model=PaginatedResponse[Dict[str, Any]])
async def list_eas(
    limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="Maximum items to return"),
    offset: int = Query(DEFAULT_OFFSET, ge=0, description="Number of items to skip"),
) -> PaginatedResponse[Dict[str, Any]]:
    items = _collect_ea_files()
    total = len(items)
    return PaginatedResponse.create(items=items[offset:offset + limit], total=total, limit=limit, offset=offset)


@router.get("/{ea_id}")
async def get_ea_details(ea_id: str):
    item = _find_ea(ea_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"EA {ea_id} not found")

    file_path = Path(item["file_path"])
    manifest = _load_json_sibling(file_path, f"{file_path.stem}.json") or {}
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    performance = manifest.get("performance") if isinstance(manifest.get("performance"), dict) else {}

    return {
        "id": item["id"],
        "name": item["name"],
        "symbol": item.get("symbol", ""),
        "timeframe": item.get("timeframe", ""),
        "status": item["status"],
        "deployed": False,
        "file_path": item["file_path"],
        "strategy_id": item["strategy_id"],
        "relative_path": item["relative_path"],
        "config": config,
        "performance": performance,
        "recent_trades": [],
    }


@router.post("/{ea_id}/deploy")
async def deploy_ea(ea_id: str):
    item = _find_ea(ea_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"EA {ea_id} not found")
    raise HTTPException(status_code=503, detail="Deployment is handled by the trading/backend node and is not available on this host")


@router.post("/{ea_id}/undeploy")
async def undeploy_ea(ea_id: str):
    item = _find_ea(ea_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"EA {ea_id} not found")
    raise HTTPException(status_code=503, detail="Undeploy is handled by the trading/backend node and is not available on this host")
