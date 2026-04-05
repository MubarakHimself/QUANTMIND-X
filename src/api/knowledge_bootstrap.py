"""
Canonical bootstrap helpers for repo-shipped knowledge assets.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from src.api.ide_models import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_BOOKS: List[Dict[str, Any]] = [
    {
        "filename": "mql5.pdf",
        "title": "MQL5 Reference",
        "author": "MetaQuotes",
        "topics": ["mql5", "mt5", "development-reference"],
        "subcategory": "mql5",
        "description": "Core MQL5 reference guide for EA and indicator development.",
    },
    {
        "filename": "mql5book.pdf",
        "title": "MQL5 Book",
        "author": "MetaQuotes",
        "topics": ["mql5", "mt5", "development-guide"],
        "subcategory": "mql5",
        "description": "Long-form MQL5 development guide for MetaTrader 5 automation.",
    },
]


def _bootstrap_enabled() -> bool:
    raw = os.getenv("QMX_BOOTSTRAP_REFERENCE_BOOKS", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _metadata_path(book_path: Path) -> Path:
    return book_path.parent / f"{book_path.name}.metadata.json"


def bootstrap_reference_books(
    *,
    repo_root: Path | None = None,
    knowledge_dir: Path | None = None,
) -> List[Path]:
    """
    Mirror repo-shipped MQL/MT5 PDF guides into canonical knowledge/books storage.
    """
    if not _bootstrap_enabled():
        return []

    repo_root = repo_root or PROJECT_ROOT
    knowledge_dir = knowledge_dir or KNOWLEDGE_DIR
    books_dir = knowledge_dir / "books"
    books_dir.mkdir(parents=True, exist_ok=True)

    synced: List[Path] = []
    for spec in REFERENCE_BOOKS:
        source = repo_root / spec["filename"]
        if not source.exists():
            continue

        target = books_dir / source.name
        target_needs_update = (
            not target.exists()
            or target.stat().st_size != source.stat().st_size
            or int(target.stat().st_mtime) != int(source.stat().st_mtime)
        )
        if target_needs_update:
            shutil.copy2(source, target)

        existing: Dict[str, Any] = {}
        metadata_path = _metadata_path(target)
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed reading existing bootstrap metadata for %s: %s", target, exc)

        metadata = {
            **existing,
            "title": spec["title"],
            "author": spec["author"],
            "topics": spec["topics"],
            "subcategory": spec["subcategory"],
            "description": spec["description"],
            "category": "Books",
            "created_by": "system",
            "source_path": str(source),
            "bootstrap_source": "repo-reference-book",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        synced.append(target)

    return synced
