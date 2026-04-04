"""
QuantMind Strategy Handler

Business logic for strategy folder operations.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.api.ide_models import STRATEGIES_DIR, StrategyStatus
from src.api.wf1_artifacts import ensure_bundle, find_strategy_root, iter_strategy_roots

logger = logging.getLogger(__name__)


class StrategyAPIHandler:
    """Handler for strategy folder operations."""

    def __init__(self):
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all strategy folders."""
        strategies = []

        seen: set[str] = set()
        for item in iter_strategy_roots() or []:
            strategy = self._get_strategy_data(item)
            strategies.append(strategy)
            seen.add(strategy["id"])

        if STRATEGIES_DIR.exists():
            for item in STRATEGIES_DIR.iterdir():
                if item.is_dir() and item.name not in seen:
                    strategy = self._get_strategy_data(item)
                    strategies.append(strategy)

        return strategies

    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy folder details."""
        strategy_path = find_strategy_root(strategy_id) or (STRATEGIES_DIR / strategy_id)
        if not strategy_path.exists():
            return None
        return self._get_strategy_data(strategy_path)

    def _get_strategy_data(self, path: Path) -> Dict[str, Any]:
        """Get strategy data from folder."""
        source_dir = path / "source"
        research_dir = path / "research"
        development_dir = path / "development"
        backtests_dir = path / "backtests"

        audio_files = self._relative_files(path, source_dir / "audio")
        caption_files = self._relative_files(path, source_dir / "captions")
        timeline_files = self._relative_files(path, source_dir / "timelines", include_nested=False)
        chunk_manifest_files = self._relative_files(path, source_dir / "chunks")
        chunk_timeline_files = self._relative_files(path, source_dir / "timelines" / "chunks")
        research_files = self._relative_files(path, research_dir)
        development_files = self._relative_files(path, development_dir)
        backtest_files = self._relative_files(path, backtests_dir)

        # Check for subdirectories to determine what the strategy has
        has_video_ingest = bool(source_dir.exists() and (timeline_files or caption_files or audio_files or chunk_manifest_files))
        has_trd = bool(research_dir.exists() and research_files)
        has_ea = bool(development_dir.exists() and development_files)
        has_backtest = bool(backtests_dir.exists() and backtest_files)

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
                pass

        return {
            "id": path.name,
            "name": path.name,
            "status": status.value,
            "created_at": created_at,
            "has_video_ingest": has_video_ingest,
            "has_trd": has_trd,
            "has_ea": has_ea,
            "has_backtest": has_backtest,
            "has_source_audio": bool(audio_files),
            "has_source_captions": bool(caption_files),
            "has_chunk_manifest": bool(chunk_manifest_files),
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
