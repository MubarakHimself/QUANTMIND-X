"""Shared configuration helpers for graph memory persistence."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


def resolve_graph_memory_db_path(
    db_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve the canonical graph-memory database path.

    All graph-memory readers and writers should use this helper so the UI,
    API, and agents operate on the same persistent store.
    """
    resolved = Path(db_path) if db_path is not None else Path(
        os.environ.get("GRAPH_MEMORY_DB", "data/graph_memory.db")
    )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
