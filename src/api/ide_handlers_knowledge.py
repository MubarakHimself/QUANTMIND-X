"""
QuantMind Knowledge Handler

Business logic for knowledge hub operations.
"""

import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.api.ide_models import KNOWLEDGE_DIR, SCRAPED_ARTICLES_DIR
from src.api.knowledge_bootstrap import bootstrap_reference_books

logger = logging.getLogger(__name__)


class KnowledgeAPIHandler:
    """Handler for knowledge hub operations."""

    def __init__(self):
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        bootstrap_reference_books(knowledge_dir=KNOWLEDGE_DIR)

    def list_knowledge(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """List knowledge items including scraped articles."""
        items = []

        # First, get items from the main knowledge directory
        if not KNOWLEDGE_DIR.exists():
            KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

        categories = [category] if category else ["books", "articles", "notes"]

        for cat in categories:
            cat_path = KNOWLEDGE_DIR / cat
            if not cat_path.exists():
                continue

            for item in cat_path.rglob("*"):
                if item.is_file() and not item.name.endswith(".metadata.json"):
                    rel_path = item.relative_to(KNOWLEDGE_DIR)
                    try:
                        size = item.stat().st_size
                    except Exception:
                        size = 0

                    metadata = self._read_sidecar_metadata(item)
                    items.append({
                        "id": str(rel_path).replace("\\", "/"),
                        "name": metadata.get("title") or item.stem,
                        "category": cat,
                        "path": str(item),
                        "size_bytes": size,
                        "indexed": False,
                    })

        # Also include scraped articles if they exist.
        if category in (None, "articles"):
            for item in self._iter_scraped_articles():
                rel_path = item.relative_to(SCRAPED_ARTICLES_DIR)
                try:
                    size = item.stat().st_size
                except Exception:
                    size = 0

                metadata = self._read_sidecar_metadata(item)
                items.append({
                    "id": f"scraped/{rel_path}".replace("\\", "/"),
                    "name": metadata.get("title") or item.stem.replace("_", " ").title(),
                    "category": "articles",
                    "path": str(item),
                    "size_bytes": size,
                    "indexed": any(candidate.exists() for candidate in self._sidecar_candidates(item)),
                    "source_category": rel_path.parent.as_posix() if rel_path.parent != Path(".") else "root",
                })

        return items

    def get_content(self, item_id: str) -> Optional[str]:
        """Get knowledge item content."""
        # Check if it's a scraped article
        if item_id.startswith("scraped/"):
            item_path = SCRAPED_ARTICLES_DIR / item_id.replace("scraped/", "")
        else:
            item_path = KNOWLEDGE_DIR / item_id

        if not item_path.exists():
            return None

        try:
            return item_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"Error reading knowledge item {item_id}: {e}")
            return None

    def _sidecar_candidates(self, item: Path) -> List[Path]:
        candidates = [item.parent / f"{item.name}.metadata.json"]
        if item.suffix == ".md":
            candidates.append(item.with_suffix(".json"))
        return candidates

    def _read_sidecar_metadata(self, item: Path) -> Dict[str, Any]:
        for candidate in self._sidecar_candidates(item):
            if not candidate.exists():
                continue
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to read knowledge metadata for {item}: {exc}")
        return {}

    def _iter_scraped_articles(self):
        if not SCRAPED_ARTICLES_DIR.exists():
            return
        for item in SCRAPED_ARTICLES_DIR.rglob("*.md"):
            yield item
