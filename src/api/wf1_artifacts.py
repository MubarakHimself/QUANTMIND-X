"""
WF1 / AlgoForge artifact tree helpers.

Shared Assets is the canonical storage surface for workflow artifacts.
This module creates and resolves the filesystem contract used by video ingest,
research, development, compilation, and backtest stages.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from src.api.ide_models import ASSETS_DIR


WF1_STRATEGIES_ROOT = ASSETS_DIR / "strategies"
DEFAULT_STRATEGY_FAMILY = "scalping"


def _slugify(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", (value or "").strip().lower())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def sanitize_strategy_id(strategy_name: str) -> str:
    return _slugify(strategy_name, "video_ingest")


def normalize_strategy_family(strategy_family: Optional[str]) -> str:
    return _slugify(strategy_family or DEFAULT_STRATEGY_FAMILY, DEFAULT_STRATEGY_FAMILY)


def source_bucket_for_playlist(is_playlist: bool) -> str:
    return "playlists" if is_playlist else "single-videos"


def build_workflow_id() -> str:
    return f"wf1_{uuid.uuid4().hex[:16]}"


def build_artifact_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class WF1ArtifactBundle:
    workflow_id: str
    strategy_id: str
    strategy_family: str
    source_bucket: str
    root: Path
    source_dir: Path
    research_dir: Path
    development_dir: Path
    variants_dir: Path
    compilation_dir: Path
    backtests_dir: Path
    reports_dir: Path
    workflow_dir: Path
    meta_path: Path
    request_manifest_path: Path
    workflow_manifest_path: Path
    timeline_dir: Path

    @property
    def relative_root(self) -> str:
        return str(self.root.relative_to(ASSETS_DIR)).replace("\\", "/")

    def as_metadata(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        return {
            "workflow_id": self.workflow_id,
            "strategy_id": self.strategy_id,
            "strategy_family": self.strategy_family,
            "source_bucket": self.source_bucket,
            "root": str(self.root),
            "relative_root": self.relative_root,
            "source_dir": str(self.source_dir),
            "timeline_dir": str(self.timeline_dir),
            "created_at": now,
            "updated_at": now,
        }


def build_bundle(
    strategy_name: str,
    is_playlist: bool,
    workflow_id: Optional[str] = None,
    strategy_family: Optional[str] = None,
) -> WF1ArtifactBundle:
    strategy_id = sanitize_strategy_id(strategy_name)
    family = normalize_strategy_family(strategy_family)
    source_bucket = source_bucket_for_playlist(is_playlist)
    root = WF1_STRATEGIES_ROOT / family / source_bucket / strategy_id
    source_dir = root / "source"
    timeline_dir = source_dir / "timelines"
    workflow_dir = root / "workflow"
    return WF1ArtifactBundle(
        workflow_id=workflow_id or build_workflow_id(),
        strategy_id=strategy_id,
        strategy_family=family,
        source_bucket=source_bucket,
        root=root,
        source_dir=source_dir,
        research_dir=root / "research",
        development_dir=root / "development",
        variants_dir=root / "variants",
        compilation_dir=root / "compilation",
        backtests_dir=root / "backtests",
        reports_dir=root / "reports",
        workflow_dir=workflow_dir,
        meta_path=root / ".meta.json",
        request_manifest_path=source_dir / "request.json",
        workflow_manifest_path=workflow_dir / "manifest.json",
        timeline_dir=timeline_dir,
    )


def ensure_bundle(
    strategy_name: str,
    is_playlist: bool,
    workflow_id: Optional[str] = None,
    strategy_family: Optional[str] = None,
    source_url: Optional[str] = None,
) -> WF1ArtifactBundle:
    bundle = build_bundle(
        strategy_name=strategy_name,
        is_playlist=is_playlist,
        workflow_id=workflow_id,
        strategy_family=strategy_family,
    )

    for path in (
        bundle.source_dir,
        bundle.timeline_dir,
        bundle.research_dir,
        bundle.development_dir,
        bundle.variants_dir,
        bundle.compilation_dir,
        bundle.backtests_dir,
        bundle.reports_dir,
        bundle.workflow_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    meta = {
        "name": strategy_name,
        "status": "pending",
        "created_at": now,
        "updated_at": now,
        "workflow_id": bundle.workflow_id,
        "strategy_id": bundle.strategy_id,
        "strategy_family": bundle.strategy_family,
        "source_bucket": bundle.source_bucket,
        "artifact_root": bundle.relative_root,
        "has_video_ingest": True,
        "has_trd": False,
        "has_ea": False,
        "has_backtest": False,
    }
    bundle.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    request_manifest = {
        "artifact_id": build_artifact_id("source"),
        "artifact_type": "video_ingest_request",
        "workflow_id": bundle.workflow_id,
        "strategy_id": bundle.strategy_id,
        "strategy_family": bundle.strategy_family,
        "source_bucket": bundle.source_bucket,
        "source_url": source_url,
        "created_at": now,
    }
    bundle.request_manifest_path.write_text(json.dumps(request_manifest, indent=2), encoding="utf-8")

    workflow_manifest = {
        "workflow_id": bundle.workflow_id,
        "workflow_type": "wf1_creation",
        "strategy_id": bundle.strategy_id,
        "strategy_family": bundle.strategy_family,
        "source_bucket": bundle.source_bucket,
        "artifact_root": bundle.relative_root,
        "current_stage": "video_ingest",
        "status": "pending",
        "updated_at": now,
    }
    bundle.workflow_manifest_path.write_text(json.dumps(workflow_manifest, indent=2), encoding="utf-8")
    return bundle


def iter_strategy_roots() -> Iterator[Path]:
    if not WF1_STRATEGIES_ROOT.exists():
        return
    for meta_path in WF1_STRATEGIES_ROOT.rglob(".meta.json"):
        yield meta_path.parent


def find_strategy_root(strategy_id: str) -> Optional[Path]:
    target = sanitize_strategy_id(strategy_id)
    for root in iter_strategy_roots() or []:
        if root.name == target:
            return root
    return None

