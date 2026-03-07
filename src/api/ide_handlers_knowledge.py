"""
QuantMind Knowledge Handler

Business logic for knowledge hub operations.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.api.ide_models import KNOWLEDGE_DIR, SCRAPED_ARTICLES_DIR

logger = logging.getLogger(__name__)


class KnowledgeAPIHandler:
    """Handler for knowledge hub operations."""

    def __init__(self):
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

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

                    items.append({
                        "id": str(rel_path).replace("\\", "/"),
                        "name": item.stem,
                        "category": cat,
                        "path": str(item),
                        "size_bytes": size,
                        "indexed": False,
                    })

        # Also include scraped articles if they exist
        if SCRAPED_ARTICLES_DIR.exists():
            scraped_categories = [category] if category else ["expert_advisors", "integration", "trading", "trading_systems"]

            for cat in scraped_categories:
                if category and cat != category:
                    continue
                cat_path = SCRAPED_ARTICLES_DIR / cat
                if not cat_path.exists():
                    continue

                for item in cat_path.rglob("*"):
                    if item.is_file() and item.suffix == ".md":
                        rel_path = item.relative_to(SCRAPED_ARTICLES_DIR)
                        try:
                            size = item.stat().st_size
                        except Exception:
                            size = 0

                        # Check if there's a corresponding .json file for metadata
                        json_path = item.with_suffix(".json")
                        has_index = json_path.exists()

                        items.append({
                            "id": f"scraped/{rel_path}".replace("\\", "/"),
                            "name": item.stem.replace("_", " ").title(),
                            "category": cat.replace("_", " "),
                            "path": str(item),
                            "size_bytes": size,
                            "indexed": has_index,
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
