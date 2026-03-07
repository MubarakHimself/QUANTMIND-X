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

logger = logging.getLogger(__name__)


class StrategyAPIHandler:
    """Handler for strategy folder operations."""

    def __init__(self):
        STRATEGIES_DIR.mkdir(parents=True, exist_ok=True)

    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all strategy folders."""
        strategies = []
        if not STRATEGIES_DIR.exists():
            return strategies

        for item in STRATEGIES_DIR.iterdir():
            if item.is_dir():
                strategy = self._get_strategy_data(item)
                strategies.append(strategy)

        return strategies

    def get_strategy(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy folder details."""
        strategy_path = STRATEGIES_DIR / strategy_id
        if not strategy_path.exists():
            return None
        return self._get_strategy_data(strategy_path)

    def _get_strategy_data(self, path: Path) -> Dict[str, Any]:
        """Get strategy data from folder."""
        # Check for subdirectories to determine what the strategy has
        has_video_ingest = (path / "video_ingest").exists()
        has_trd = (path / "trd").exists()
        has_ea = (path / "ea").exists()
        has_backtest = (path / "backtest").exists()

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
        }

    def create_strategy_folder(self, name: str) -> str:
        """Create a new strategy folder."""
        folder_name = name.lower().replace(" ", "_")
        strategy_path = STRATEGIES_DIR / folder_name

        if strategy_path.exists():
            raise FileExistsError(f"Strategy '{name}' already exists")

        # Create directory structure
        strategy_path.mkdir(parents=True)
        (strategy_path / "video_ingest").mkdir(exist_ok=True)
        (strategy_path / "trd").mkdir(exist_ok=True)
        (strategy_path / "ea").mkdir(exist_ok=True)
        (strategy_path / "backtest").mkdir(exist_ok=True)

        # Save metadata
        meta = {
            "name": name,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
        }
        (strategy_path / ".meta.json").write_text(json.dumps(meta))

        logger.info(f"Created strategy folder: {folder_name}")
        return folder_name
