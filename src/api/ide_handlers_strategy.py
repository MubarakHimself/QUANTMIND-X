"""
QuantMind Strategy Handler

Business logic for strategy folder operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.api.ide_models import StrategyStatus
from src.api import wf1_artifacts
from src.api.wf1_artifacts import (
    ensure_bundle,
    find_strategy_root,
    find_strategy_root_by_workflow_id,
    iter_strategy_roots,
    strategy_root_has_operator_visible_signal,
)

logger = logging.getLogger(__name__)


class StrategyAPIHandler:
    """Handler for strategy folder operations."""

    def __init__(self):
        # Canonical strategy storage is under shared assets / WF1 artifacts.
        pass

    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all strategy folders."""
        strategies = []
        for item in iter_strategy_roots() or []:
            if not strategy_root_has_operator_visible_signal(item):
                continue
            strategy = self._get_strategy_data(item)
            strategies.append(strategy)

        return strategies

    def get_strategy(self, strategy_ref: str) -> Optional[Dict[str, Any]]:
        """Get strategy folder details from a canonical asset id, relative root, or legacy slug."""
        strategy_path = self._resolve_strategy_root(strategy_ref)
        if not strategy_path or not strategy_path.exists():
            return None
        return self._get_strategy_data(strategy_path)

    def _resolve_strategy_root(self, strategy_ref: str) -> Optional[Path]:
        if not strategy_ref:
            return None

        normalized = str(strategy_ref).strip().replace("\\", "/").strip("/")
        if not normalized:
            return None

        if normalized.startswith("strategies/"):
            candidate = wf1_artifacts.WF1_STRATEGIES_ROOT / normalized.removeprefix("strategies/")
            if candidate.exists():
                return candidate

        if "/" in normalized:
            candidate = wf1_artifacts.WF1_STRATEGIES_ROOT / normalized
            if candidate.exists():
                return candidate

        workflow_match = find_strategy_root_by_workflow_id(normalized)
        if workflow_match is not None:
            return workflow_match

        return find_strategy_root(normalized)

    def _get_strategy_data(self, path: Path) -> Dict[str, Any]:
        """Get strategy data from folder."""
        source_dir = path / "source"
        research_dir = path / "research"
        development_dir = path / "development"
        variants_dir = path / "variants"
        compilation_dir = path / "compilation"
        backtests_dir = path / "backtests"
        reports_dir = path / "reports"
        relative_root = str(path.relative_to(wf1_artifacts.WF1_STRATEGIES_ROOT)).replace("\\", "/")

        audio_files = self._relative_files(path, source_dir / "audio")
        caption_files = self._relative_files(path, source_dir / "captions")
        timeline_files = self._relative_files(path, source_dir / "timelines", include_nested=False)
        chunk_manifest_files = self._relative_files(path, source_dir / "chunks")
        chunk_timeline_files = self._relative_files(path, source_dir / "timelines" / "chunks")
        research_files = self._relative_files(path, research_dir)
        development_files = self._relative_files(path, development_dir)
        variant_files = self._relative_files(path, variants_dir)
        compilation_files = self._relative_files(path, compilation_dir)
        backtest_files = self._relative_files(path, backtests_dir)
        report_files = self._relative_files(path, reports_dir)

        # Check for subdirectories to determine what the strategy has
        has_video_ingest = bool(source_dir.exists() and (timeline_files or caption_files or audio_files or chunk_manifest_files))
        has_trd = bool(research_dir.exists() and research_files)
        has_ea = bool(development_dir.exists() and development_files)
        has_backtest = bool(backtests_dir.exists() and backtest_files)
        has_variants = bool(variants_dir.exists() and variant_files)
        has_compilation = bool(compilation_dir.exists() and compilation_files)
        has_reports = bool(reports_dir.exists() and report_files)

        # Try to load metadata
        meta_path = path / ".meta.json"
        status = StrategyStatus.PENDING
        created_at = datetime.now().isoformat()

        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                status = StrategyStatus(meta.get("status", "pending"))
                created_at = meta.get("created_at", created_at)
            except Exception:
                meta = {}
        else:
            meta = {}

        return {
            "id": path.name,
            "asset_id": f"strategies/{relative_root}",
            "relative_root": relative_root,
            "name": path.name,
            "status": status.value,
            "created_at": created_at,
            "workflow_id": meta.get("workflow_id"),
            "strategy_id": meta.get("strategy_id"),
            "strategy_family": meta.get("strategy_family"),
            "source_bucket": meta.get("source_bucket"),
            "blocking_error": meta.get("blocking_error"),
            "blocking_error_detail": meta.get("blocking_error_detail"),
            "has_video_ingest": has_video_ingest,
            "has_trd": has_trd,
            "has_ea": has_ea,
            "has_backtest": has_backtest,
            "has_variants": has_variants,
            "has_compilation": has_compilation,
            "has_reports": has_reports,
            "has_source_audio": bool(audio_files),
            "has_source_captions": bool(caption_files),
            "has_chunk_manifest": bool(chunk_manifest_files),
            "research_files": research_files,
            "development_files": development_files,
            "variant_files": variant_files,
            "compilation_files": compilation_files,
            "report_files": report_files,
            "backtest_files": backtest_files,
            "source_artifacts": {
                "audio_files": audio_files,
                "caption_files": caption_files,
                "timeline_files": timeline_files,
                "chunk_manifest_files": chunk_manifest_files,
                "chunk_timeline_files": chunk_timeline_files,
            },
        }

    def _relative_files(self, root: Path, directory: Path, *, include_nested: bool = True) -> List[str]:
        if not directory.exists():
            return []
        iterator = directory.rglob("*") if include_nested else directory.glob("*")
        return sorted(
            str(item.relative_to(root)).replace("\\", "/")
            for item in iterator
            if item.is_file()
        )

    def create_strategy_folder(self, name: str) -> str:
        """Create a new strategy folder."""
        folder_name = name.lower().replace(" ", "_")
        strategy_path = find_strategy_root(folder_name)
        if strategy_path and strategy_path.exists():
            raise FileExistsError(f"Strategy '{name}' already exists")

        bundle = ensure_bundle(strategy_name=name, is_playlist=False)
        strategy_path = bundle.root

        logger.info(f"Created strategy folder: {folder_name}")
        return folder_name
