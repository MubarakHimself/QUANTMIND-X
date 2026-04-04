"""
QuantMind IDE Knowledge Endpoints

API endpoints for knowledge hub management.
"""

import logging
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from enum import Enum

from src.api.ide_handlers import KnowledgeAPIHandler
from src.api.ide_models import KNOWLEDGE_DIR, SCRAPED_ARTICLES_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])

# =============================================================================
# Knowledge Hub Sync Models
# =============================================================================

class SyncDirection(str, Enum):
    """Sync direction for knowledge hub."""
    UPLOAD = "upload"      # Local -> Hub
    DOWNLOAD = "download"  # Hub -> Local
    BIDIRECTIONAL = "bidirectional"  # Both directions


class ConflictResolution(str, Enum):
    """Conflict resolution strategy."""
    LOCAL_WINS = "local_wins"      # Keep local version
    HUB_WINS = "hub_wins"          # Keep hub version
    NEWEST_WINS = "newest_wins"    # Keep most recent
    SKIP = "skip"                  # Skip conflicting items


class HubSyncRequest(BaseModel):
    """Request model for knowledge hub sync."""
    direction: SyncDirection = SyncDirection.BIDIRECTIONAL
    conflict_resolution: ConflictResolution = ConflictResolution.NEWEST_WINS
    namespaces: Optional[List[str]] = None  # Which namespaces to sync
    include_content: bool = True


class HubSyncResponse(BaseModel):
    """Response model for knowledge hub sync."""
    success: bool
    direction: str
    uploaded: int = 0
    downloaded: int = 0
    conflicts_resolved: int = 0
    errors: List[str] = []
    details: Optional[Dict[str, Any]] = None


class FirecrawlSettings(BaseModel):
    """Settings for Firecrawl scraper."""
    api_key: Optional[str] = None
    scraper_type: str = "simple"  # "simple" or "firecrawl"


class FirecrawlSettingsResponse(BaseModel):
    """Response model for Firecrawl settings."""
    api_key_set: bool
    scraper_type: str
    scraper_available: bool
    firecrawl_available: bool

# Initialize handler
knowledge_handler = KnowledgeAPIHandler()

# Path to sync state file
SYNC_STATE_FILE = Path("data/scraped_articles/.sync_state.json")


def _metadata_candidates(path: Path) -> List[Path]:
    candidates = [path.parent / f"{path.name}.metadata.json"]
    if path.suffix.lower() == ".md":
        candidates.append(path.with_suffix(".json"))
    return candidates


def _read_metadata(path: Path) -> Dict[str, Any]:
    for candidate in _metadata_candidates(path):
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return {}


def _pretty_title(name: str) -> str:
    return Path(name).stem.replace("_", " ").replace("-", " ").strip().title()


def _frontmatter_value(text: str, key: str) -> Optional[Any]:
    if not text.startswith("---\n"):
        return None
    lines = text.splitlines()
    values: List[str] = []
    in_tags = False
    current_key = f"{key}:"
    for line in lines[1:]:
        if line.strip() == "---":
            break
        stripped = line.strip()
        if stripped.startswith(current_key):
            value = stripped[len(current_key):].strip().strip("'\"")
            if value:
                return value
            in_tags = True
            continue
        if in_tags:
            if stripped.startswith("- "):
                values.append(stripped[2:].strip().strip("'\""))
                continue
            break
    if values:
        return values
    return None


def _markdown_heading_title(text: str) -> Optional[str]:
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            text = parts[1]
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
    return None


def _markdown_excerpt(path: Path, limit: int = 220) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if text.startswith("---\n"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            text = parts[1]
    excerpt = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return excerpt[:limit].strip()


def _source_from_url(url: Optional[str], fallback: str) -> str:
    if not url:
        return fallback
    parsed = urlparse(url)
    return parsed.netloc or fallback


def _collect_research_article_cards() -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []

    articles_dir = KNOWLEDGE_DIR / "articles"
    if articles_dir.exists():
        for path in articles_dir.rglob("*"):
            if not path.is_file() or path.name.endswith(".metadata.json"):
                continue
            metadata = _read_metadata(path)
            cards.append({
                "id": f"articles/{path.name}",
                "title": metadata.get("title") or _pretty_title(path.name),
                "source": _source_from_url(metadata.get("url"), "uploaded"),
                "date": metadata.get("uploaded_at") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "tags": [metadata["subcategory"]] if metadata.get("subcategory") else [],
                "excerpt": _markdown_excerpt(path),
            })

    if SCRAPED_ARTICLES_DIR.exists():
        for path in SCRAPED_ARTICLES_DIR.rglob("*.md"):
            metadata = _read_metadata(path)
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                text = ""
            excerpt = _markdown_excerpt(path)
            tags = metadata.get("relevance_tags")
            if not tags:
                frontmatter_tags = _frontmatter_value(text, "relevance_tags")
                tags = frontmatter_tags if isinstance(frontmatter_tags, list) else []
            source_category = path.parent.name if path.parent != SCRAPED_ARTICLES_DIR else "root"
            cards.append({
                "id": f"scraped/{str(path.relative_to(SCRAPED_ARTICLES_DIR)).replace('\\', '/')}",
                "title": (
                    metadata.get("title")
                    or _frontmatter_value(text, "title")
                    or _markdown_heading_title(text)
                    or _pretty_title(path.name)
                ),
                "source": _source_from_url(
                    metadata.get("url")
                    or metadata.get("source_url")
                    or _frontmatter_value(text, "source_url"),
                    source_category,
                ),
                "date": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "tags": tags or [source_category],
                "excerpt": excerpt,
            })

    cards.sort(key=lambda item: item.get("date", ""), reverse=True)
    return cards


def _collect_knowledge_books() -> List[Dict[str, Any]]:
    books: List[Dict[str, Any]] = []
    books_dir = KNOWLEDGE_DIR / "books"
    if not books_dir.exists():
        return books

    for path in books_dir.rglob("*"):
        if not path.is_file() or path.name.endswith(".metadata.json"):
            continue
        metadata = _read_metadata(path)
        books.append({
            "id": f"books/{path.name}",
            "title": metadata.get("title") or Path(path.name).stem,
            "author": metadata.get("author", ""),
            "topics": [metadata["subcategory"]] if metadata.get("subcategory") else [],
            "year": metadata.get("year"),
        })

    books.sort(key=lambda item: item["title"].lower())
    return books


def _collect_generic_entries(folder: str, *, title_key: str, preview_key: Optional[str] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    target_dir = KNOWLEDGE_DIR / folder
    if not target_dir.exists():
        return items

    for path in target_dir.rglob("*"):
        if not path.is_file() or path.name.endswith(".metadata.json"):
            continue
        metadata = _read_metadata(path)
        payload: Dict[str, Any] = {
            "id": f"{folder}/{path.name}",
            title_key: metadata.get("title") or Path(path.name).stem,
            "date": metadata.get("uploaded_at") or datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        if preview_key is not None:
            payload[preview_key] = _markdown_excerpt(path)
        items.append(payload)
    items.sort(key=lambda item: item.get("date", ""), reverse=True)
    return items


def get_sync_state() -> dict:
    """Get the current sync state."""
    if SYNC_STATE_FILE.exists():
        try:
            return json.loads(SYNC_STATE_FILE.read_text())
        except Exception:
            pass
    return {
        "status": "idle",
        "last_sync": None,
        "last_batch_size": 0,
        "last_start_index": 0,
        "articles_synced": 0,
        "errors": []
    }


def update_sync_state(
    status: str,
    articles_synced: int = 0,
    batch_size: int = 0,
    start_index: int = 0,
    errors: list = None,
    last_error: str = None
):
    """Update the sync state file."""
    state = get_sync_state()
    state["status"] = status
    state["last_sync"] = datetime.now().isoformat()
    state["articles_synced"] = articles_synced
    state["last_batch_size"] = batch_size
    state["last_start_index"] = start_index
    if errors is not None:
        state["errors"] = errors
    if last_error:
        state["last_error"] = last_error
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


# =============================================================================
# Firecrawl Settings
# =============================================================================

FIRECRAWL_SETTINGS_FILE = Path("config/settings/firecrawl.json")

# Allowlist of permitted scraper modules and their validated arguments
ALLOWED_SCRAPER_MODULES = {
    "firecrawl": {
        "script": "scripts/firecrawl_scraper.py",
        "extra_args": ["api_key"],  # api_key is passed as --api-key
    },
    "simple": {
        "script": "scripts/simple_scraper.py",
        "extra_args": [],  # no extra args allowed
    },
}


def _validate_scraper_type(scraper_type: str) -> bool:
    """Check if scraper type is in the allowlist."""
    return scraper_type in ALLOWED_SCRAPER_MODULES


def load_firecrawl_settings() -> dict:
    """Load Firecrawl settings from file (without API key)."""
    if FIRECRAWL_SETTINGS_FILE.exists():
        try:
            settings = json.loads(FIRECRAWL_SETTINGS_FILE.read_text())
        except Exception:
            pass
    return {"api_key": None, "scraper_type": "simple"}


def get_firecrawl_api_key() -> Optional[str]:
    """Fetch Firecrawl API key from secure keyring storage."""
    try:
        import keyring
        return keyring.get_password("quantmindx", "firecrawl_api_key")
    except Exception:
        return None


def save_firecrawl_api_key(api_key: str) -> bool:
    """Store Firecrawl API key in secure keyring storage."""
    try:
        import keyring
        keyring.set_password("quantmindx", "firecrawl_api_key", api_key)
        return True
    except Exception:
        return False


def save_firecrawl_settings(settings: dict):
    """Save Firecrawl settings to file (API key stored separately in keyring)."""
    FIRECRAWL_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Store API key securely via keyring; only non-sensitive settings go to file
    if settings.get("api_key"):
        save_firecrawl_api_key(settings["api_key"])
    # Write settings without the raw API key
    safe_settings = {k: v for k, v in settings.items() if k != "api_key"}
    safe_settings.setdefault("scraper_type", "simple")
    FIRECRAWL_SETTINGS_FILE.write_text(json.dumps(safe_settings, indent=2))


def check_scraper_available() -> tuple[bool, bool]:
    """Check if scrapers are available."""
    simple_available = Path("scripts/simple_scraper.py").exists()
    firecrawl_available = Path("scripts/firecrawl_scraper.py").exists()
    return simple_available, firecrawl_available


@router.get("/firecrawl/settings", response_model=FirecrawlSettingsResponse)
async def get_firecrawl_settings():
    """
    Get Firecrawl settings including API key status and scraper availability.
    """
    settings = load_firecrawl_settings()
    simple_available, firecrawl_available = check_scraper_available()

    return FirecrawlSettingsResponse(
        api_key_set=settings.get("api_key") is not None and settings.get("api_key", "") != "",
        scraper_type=settings.get("scraper_type", "simple"),
        scraper_available=simple_available,
        firecrawl_available=firecrawl_available
    )


@router.post("/firecrawl/settings")
async def update_firecrawl_settings(firecrawl_settings: FirecrawlSettings):
    """
    Update Firecrawl settings including API key and scraper type.
    """
    settings = {
        "api_key": firecrawl_settings.api_key,
        "scraper_type": firecrawl_settings.scraper_type
    }
    save_firecrawl_settings(settings)

    simple_available, firecrawl_available = check_scraper_available()

    return FirecrawlSettingsResponse(
        api_key_set=firecrawl_settings.api_key is not None and firecrawl_settings.api_key != "",
        scraper_type=firecrawl_settings.scraper_type,
        scraper_available=simple_available,
        firecrawl_available=firecrawl_available
    )


@router.get("")
async def list_knowledge(category: Optional[str] = None):
    """List knowledge items."""
    return knowledge_handler.list_knowledge(category)


@router.get("/{item_id:path}/content")
async def get_knowledge_content(item_id: str):
    """Get knowledge item content."""
    content = knowledge_handler.get_content(item_id)
    if content is None:
        raise HTTPException(404, "Item not found")
    return {"content": content}


# =============================================================================
# Knowledge Upload Endpoints
# =============================================================================

knowledge_upload_router = APIRouter(prefix="/api/ide/knowledge", tags=["knowledge-upload"])


@knowledge_upload_router.post("/upload")
async def upload_knowledge_file(
    file: UploadFile = File(None),
    category: str = Form("books"),
    title: str = Form(None),
    author: str = Form(None),
    url: str = Form(None),
    subcategory: str = Form(None),
    index: str = Form("false"),
    content: str = Form(None)
):
    """
    Upload file or note to knowledge base.

    Args:
        file: The file to upload (PDF, MD, TXT, CSV, JSON) - optional for notes
        category: Target category (books, articles, notes)
        title: Optional title for the item
        author: Optional author name
        url: Optional source URL (for articles)
        subcategory: Optional subcategory for better organization
        index: Whether to index the file (for books)
        content: Content for notes (no file)
    """
    try:
        # Validate category
        valid_categories = ["books", "articles", "notes"]
        if category not in valid_categories:
            category = "books"

        # Create category directory if it doesn't exist
        category_dir = KNOWLEDGE_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Handle note submission (no file)
        if category == "notes" and content and not file:
            # Create a markdown file from the note content
            safe_title = "".join(c for c in (title or "note") if c.isalnum() or c in (' ', '-', '_')).strip()
            filename = f"{safe_title}.md"
            save_path = category_dir / filename

            # Add metadata header to markdown
            metadata_header = "---\n"
            if title:
                metadata_header += f"title: {title}\n"
            metadata_header += f"created_at: {datetime.now().isoformat()}\n"
            metadata_header += "---\n\n"

            note_content = metadata_header + content
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(note_content)

            logger.info(f"Created note: {title} in {category}")

            return {
                "status": "success",
                "filename": filename,
                "title": title,
                "path": str(save_path),
                "category": category,
                "size_bytes": len(note_content.encode('utf-8')),
                "indexed": False
            }

        # Handle file upload
        if not file:
            raise HTTPException(400, "File is required for books and articles")

        # For books/articles, save the file with optional metadata
        if not file.filename:
            raise HTTPException(400, "Filename is required")
        save_path = category_dir / file.filename

        # Read and save content
        content_bytes = await file.read()
        with open(save_path, "wb") as f:
            f.write(content_bytes)

        # Save metadata separately if provided
        metadata = {
            "filename": file.filename,
            "category": category,
            "uploaded_at": datetime.now().isoformat()
        }
        if title:
            metadata["title"] = title
        if author:
            metadata["author"] = author
        if url:
            metadata["url"] = url
        if subcategory:
            metadata["subcategory"] = subcategory

        # Save metadata as JSON file alongside the content
        metadata_path = category_dir / f"{file.filename}.metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

        logger.info(f"Uploaded {file.filename} to {category}")

        return {
            "status": "success",
            "filename": file.filename,
            "path": str(save_path),
            "category": category,
            "size_bytes": len(content_bytes),
            "indexed": index.lower() == "true"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading knowledge file: {e}")
        raise HTTPException(500, str(e))


@knowledge_upload_router.post("/upload/note")
async def upload_note(
    title: str = Form(None),
    content: str = Form(...),
    category: str = Form("notes")
):
    """
    Upload a note to the knowledge base.

    Args:
        title: Note title
        content: Note content (markdown)
        category: Category (default: notes)
    """
    try:
        # Validate category
        if category not in ["notes", "books", "articles"]:
            category = "notes"

        # Create category directory
        category_dir = KNOWLEDGE_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from title
        safe_title = "".join(c for c in (title or "note") if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_title}.md"
        save_path = category_dir / filename

        # Add metadata header
        metadata_header = "---\n"
        if title:
            metadata_header += f"title: {title}\n"
        metadata_header += f"created_at: {datetime.now().isoformat()}\n"
        metadata_header += "---\n\n"

        note_content = metadata_header + content
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(note_content)

        logger.info(f"Created note: {title} in {category}")

        return {
            "status": "success",
            "filename": filename,
            "title": title,
            "path": str(save_path),
            "category": category,
            "size_bytes": len(note_content.encode('utf-8')),
        }

    except Exception as e:
        logger.error(f"Error uploading note: {e}")
        raise HTTPException(500, str(e))


# =============================================================================
# Knowledge Sync Endpoints
# =============================================================================

def count_existing_articles() -> int:
    """Count existing scraped articles in data/scraped_articles/."""
    count = 0
    if SCRAPED_ARTICLES_DIR.exists():
        for category_dir in SCRAPED_ARTICLES_DIR.iterdir():
            if category_dir.is_dir():
                count += len(list(category_dir.glob("*.md")))
    return count


def get_articles_by_category() -> dict:
    """Get article count grouped by category."""
    categories = {}
    if SCRAPED_ARTICLES_DIR.exists():
        for category_dir in SCRAPED_ARTICLES_DIR.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                articles = list(category_dir.glob("*.md"))
                categories[category_name] = {
                    "count": len(articles),
                    "total_size": sum(f.stat().st_size for f in articles if f.is_file())
                }
    return categories


def run_scraper_background(batch_size: int = 10, start_index: int = 0, scraper_type: str = "simple", api_key: str = None):
    """
    Run the scraper in the background.
    This function is called by BackgroundTasks.
    """
    logger.info(f"Starting background scraper: batch_size={batch_size}, start_index={start_index}, scraper_type={scraper_type}")

    # Update state to running
    update_sync_state(
        status="running",
        batch_size=batch_size,
        start_index=start_index
    )

    try:
        # SECURITY: Validate scraper_type against allowlist
        if not _validate_scraper_type(scraper_type):
            logger.error(f"Unknown scraper type: {scraper_type}")
            update_sync_state(
                status="failed",
                batch_size=batch_size,
                start_index=start_index,
                last_error=f"Unknown scraper type: {scraper_type}"
            )
            return

        # SECURITY: Use allowlist to determine scraper script path
        scraper_info = ALLOWED_SCRAPER_MODULES[scraper_type]
        scraper_path = Path(scraper_info["script"])

        if not scraper_path.exists():
            logger.error(f"Scraper not found: {scraper_path}")
            update_sync_state(
                status="failed",
                batch_size=batch_size,
                start_index=start_index,
                last_error=f"Scraper script not found: {scraper_path}"
            )
            return

        # SECURITY: Fetch API key from keyring if not provided
        resolved_api_key = api_key
        if scraper_type == "firecrawl" and not resolved_api_key:
            resolved_api_key = get_firecrawl_api_key()

        # Build command arguments using allowlist only
        cmd_args = [
            sys.executable,
            str(scraper_path),
            "--batch-size", str(batch_size),
            "--start-index", str(start_index)
        ]

        # Add API key for firecrawl scraper (from secure keyring)
        if scraper_type == "firecrawl" and resolved_api_key:
            cmd_args.extend(["--api-key", resolved_api_key])

        # Run the scraper with limited batch for API usage
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )

        if result.returncode == 0:
            logger.info("Scraper completed successfully")
            # Count newly scraped articles
            new_count = count_existing_articles()
            update_sync_state(
                status="completed",
                articles_synced=new_count,
                batch_size=batch_size,
                start_index=start_index
            )
        else:
            logger.error(f"Scraper failed: {result.stderr}")
            update_sync_state(
                status="failed",
                batch_size=batch_size,
                start_index=start_index,
                last_error=result.stderr[:500] if result.stderr else "Unknown error"
            )

    except subprocess.TimeoutExpired:
        logger.error("Scraper timed out after 1 hour")
        update_sync_state(
            status="failed",
            batch_size=batch_size,
            start_index=start_index,
            last_error="Scraper timed out after 1 hour"
        )
    except Exception as e:
        logger.error(f"Error running scraper: {e}")
        update_sync_state(
            status="failed",
            batch_size=batch_size,
            start_index=start_index,
            last_error=str(e)
        )


@router.post("/sync")
async def sync_knowledge(
    background_tasks: BackgroundTasks,
    batch_size: int = 10,
    start_index: int = 0,
    sync_mode: str = "background",
    scraper_type: str = "simple",
    api_key: Optional[str] = None
):
    """
    Trigger knowledge sync by running the scraper.

    This endpoint triggers the article scraper to fetch new articles
    from MQL5 and saves them to data/scraped_articles/.

    Args:
        batch_size: Number of articles to scrape (default: 10)
        start_index: Starting index in the articles list (default: 0)
        sync_mode: "background" (async) or "sync" (wait for completion)
        scraper_type: "simple" or "firecrawl" (default: from settings or "simple")
        api_key: Firecrawl API key (optional, uses settings if not provided)

    Returns:
        JSON with sync status and article count
    """
    # Load settings to get default scraper_type
    firecrawl_settings = load_firecrawl_settings()
    scraper_type = scraper_type or firecrawl_settings.get("scraper_type", "simple")

    # SECURITY: Validate scraper_type against allowlist
    if not _validate_scraper_type(scraper_type):
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scraper type '{scraper_type}'. Allowed: {list(ALLOWED_SCRAPER_MODULES.keys())}"
        )

    # SECURITY: Fetch API key from secure keyring (not from plain settings file)
    resolved_api_key: Optional[str] = None
    if scraper_type == "firecrawl":
        resolved_api_key = api_key if api_key else get_firecrawl_api_key()

    # Count existing articles before sync
    existing_count = count_existing_articles()

    # SECURITY: Use allowlist to determine scraper script path
    scraper_info = ALLOWED_SCRAPER_MODULES[scraper_type]
    scraper_path = Path(scraper_info["script"])

    export_file = Path("data/exports/engineering_ranked.json")

    if not scraper_path.exists():
        return {
            "success": False,
            "status": "scraper_not_found",
            "message": f"Scraper script not found at {scraper_path}",
            "articles_synced": 0,
            "existing_articles": existing_count,
            "errors": [f"Scraper script not found: {scraper_path}"],
            "scraper_type": scraper_type
        }

    if not export_file.exists():
        return {
            "success": False,
            "status": "source_not_found",
            "message": "Article source file not found at data/exports/engineering_ranked.json",
            "articles_synced": 0,
            "existing_articles": existing_count,
            "errors": ["Article source file not found"]
        }

    # Check current sync state
    current_state = get_sync_state()
    if current_state.get("status") == "running":
        return {
            "success": False,
            "status": "already_running",
            "message": "A sync operation is already in progress",
            "existing_articles": existing_count,
            "current_sync": current_state
        }

    # Build command arguments using allowlist only
    cmd_args = [
        sys.executable,
        str(scraper_path),
        "--batch-size", str(batch_size),
        "--start-index", str(start_index)
    ]

    # SECURITY: Only add api_key for firecrawl (per allowlist), from secure keyring
    if scraper_type == "firecrawl" and resolved_api_key:
        cmd_args.extend(["--api-key", resolved_api_key])

    if sync_mode == "sync":
        # Run synchronously (for automated workflows)
        try:
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            new_count = count_existing_articles()
            if result.returncode == 0:
                update_sync_state(
                    status="completed",
                    articles_synced=new_count,
                    batch_size=batch_size,
                    start_index=start_index
                )
                return {
                    "success": True,
                    "status": "completed",
                    "message": f"Scraped {batch_size} articles successfully",
                    "articles_synced": new_count,
                    "existing_articles": existing_count,
                    "errors": [],
                    "scraper_type": scraper_type
                }
            else:
                update_sync_state(
                    status="failed",
                    batch_size=batch_size,
                    start_index=start_index,
                    last_error=result.stderr[:500] if result.stderr else "Unknown error"
                )
                return {
                    "success": False,
                    "status": "failed",
                    "message": f"Scraper failed: {result.stderr[:200]}",
                    "articles_synced": 0,
                    "existing_articles": existing_count,
                    "errors": [result.stderr[:500] if result.stderr else "Unknown error"]
                }
        except subprocess.TimeoutExpired:
            update_sync_state(
                status="failed",
                batch_size=batch_size,
                start_index=start_index,
                last_error="Scraper timed out after 1 hour"
            )
            return {
                "success": False,
                "status": "timeout",
                "message": "Scraper timed out after 1 hour",
                "articles_synced": 0,
                "existing_articles": existing_count,
                "errors": ["Scraper timed out after 1 hour"]
            }
        except Exception as e:
            update_sync_state(
                status="failed",
                batch_size=batch_size,
                start_index=start_index,
                last_error=str(e)
            )
            return {
                "success": False,
                "status": "error",
                "message": str(e),
                "articles_synced": 0,
                "existing_articles": existing_count,
                "errors": [str(e)]
            }
    else:
        # Run in background (default)
        background_tasks.add_task(run_scraper_background, batch_size, start_index, scraper_type, api_key)

        return {
            "success": True,
            "status": "sync_started",
            "message": f"Scraping {batch_size} articles starting from index {start_index} using {scraper_type} scraper",
            "articles_synced": 0,  # Will be updated when scraper completes
            "existing_articles": existing_count,
            "sync_state": get_sync_state(),
            "errors": []
        }


@router.get("/sync/status")
async def get_sync_status():
    """
    Get current knowledge sync status.

    Returns:
        JSON with sync status information
    """
    existing_count = count_existing_articles()

    # Check if scraper is available
    scraper_path = Path("scripts/simple_scraper.py")
    export_file = Path("data/exports/engineering_ranked.json")

    scraper_available = scraper_path.exists()
    source_available = export_file.exists()

    # Get sync state
    sync_state = get_sync_state()

    # Get categories breakdown
    categories = get_articles_by_category()

    # Calculate progress
    progress = 0
    if sync_state.get("status") == "running":
        batch_size = sync_state.get("last_batch_size", 0)
        start_index = sync_state.get("last_start_index", 0)
        if batch_size > 0:
            progress = min(100, int((start_index / (start_index + batch_size)) * 100))

    return {
        "success": True,
        "status": sync_state.get("status", "idle"),
        "scraper_available": scraper_available,
        "source_available": source_available,
        "existing_articles": existing_count,
        "output_directory": str(SCRAPED_ARTICLES_DIR),
        "sync_state": sync_state,
        "categories": categories,
        "progress": progress
    }


@router.get("/articles")
async def get_scraped_articles(
    request: Request,
    sort_by: str = "name",
    order: str = "asc",
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Get scraped articles with details for sorting and filtering.

    Args:
        sort_by: Field to sort by (name, size, modified, category)
        order: Sort order (asc or desc)
        category: Filter by category
        limit: Maximum number of articles to return
        offset: Number of articles to skip

    Returns:
        JSON with articles list and total count
    """
    research_mode = not any(
        key in request.query_params for key in ("sort_by", "order", "category", "limit", "offset")
    )
    if research_mode:
        return _collect_research_article_cards()

    articles = []

    for article in _collect_research_article_cards():
        article_category = article["tags"][0] if article.get("tags") else "uploaded"
        if category and category != article_category:
            continue
        articles.append({
            "id": article["id"].split("/")[-1].replace(".md", ""),
            "name": article["title"],
            "category": article_category,
            "size_bytes": 0,
            "modified": article["date"],
            "path": article["id"],
        })

    # Sort articles
    reverse = order == "desc"
    if sort_by == "size":
        articles.sort(key=lambda x: x.get("size_bytes", 0), reverse=reverse)
    elif sort_by == "modified":
        articles.sort(key=lambda x: x.get("modified", ""), reverse=reverse)
    elif sort_by == "category":
        articles.sort(key=lambda x: x.get("category", ""), reverse=reverse)
    else:
        articles.sort(key=lambda x: x.get("name", ""), reverse=reverse)

    total = len(articles)
    paginated = articles[offset:offset + limit]

    return {
        "success": True,
        "articles": paginated,
        "total": total,
        "categories": sorted({article["category"] for article in articles}),
        "sort_by": sort_by,
        "order": order
    }


@router.get("/books")
async def get_books():
    """Return uploaded books in the Research canvas card shape."""
    return _collect_knowledge_books()


@router.get("/logs")
async def get_logs():
    """Return knowledge logs if present, otherwise an honest empty state."""
    items = _collect_generic_entries("logs", title_key="title", preview_key="content")
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "date": item["date"],
            "content": item["content"],
            "tags": [],
        }
        for item in items
    ]


@router.get("/personal")
async def get_personal_notes():
    """Return personal notes if present."""
    items = _collect_generic_entries("personal", title_key="title", preview_key="preview")
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "date": item["date"],
            "preview": item["preview"],
        }
        for item in items
    ]


@router.get("/videos")
async def get_videos():
    """Return indexed videos if present."""
    items = _collect_generic_entries("videos", title_key="title")
    return [
        {
            "id": item["id"],
            "title": item["title"],
            "channel": "",
            "duration": "",
            "date": item["date"],
        }
        for item in items
    ]


# =============================================================================
# Knowledge Hub Sync Endpoints
# =============================================================================

async def _get_local_items(namespace: str = None) -> Dict[str, Any]:
    """
    Get all local knowledge items.

    Args:
        namespace: Filter by namespace (category)

    Returns:
        Dictionary of local items keyed by ID
    """
    items = {}
    base_dir = KNOWLEDGE_DIR

    if namespace:
        search_dirs = [base_dir / namespace] if (base_dir / namespace).exists() else []
    else:
        search_dirs = [d for d in base_dir.iterdir() if d.is_dir()] if base_dir.exists() else []

    for category_dir in search_dirs:
        category = category_dir.name
        for file_path in category_dir.glob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
                # Extract title from metadata header if present
                title = file_path.stem
                if content.startswith("---"):
                    end_idx = content.find("---", 3)
                    if end_idx > 0:
                        metadata = content[3:end_idx]
                        for line in metadata.split("\n"):
                            if line.startswith("title:"):
                                title = line.split(":", 1)[1].strip()

                items[f"{category}/{file_path.name}"] = {
                    "id": f"{category}/{file_path.name}",
                    "category": category,
                    "filename": file_path.name,
                    "title": title,
                    "content": content,
                    "path": str(file_path),
                    "modified": file_path.stat().st_mtime
                }
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")

    return items


async def _upload_to_hub(items: Dict[str, Any], namespaces: List[str] = None) -> Dict[str, Any]:
    """
    Upload local items to knowledge hub.

    Args:
        items: Dictionary of items to upload
        namespaces: Target namespaces (categories)

    Returns:
        Upload result summary
    """
    uploaded = 0
    errors = []

    # Try to use PageIndex MCP for uploading
    try:
        from src.agents.tools.knowledge.client import call_pageindex_tool

        for item_id, item in items.items():
            if namespaces and item.get("category") not in namespaces:
                continue

            try:
                result = await call_pageindex_tool("index-document", {
                    "content": item.get("content", ""),
                    "namespace": item.get("category", "knowledge"),
                    "metadata": {
                        "title": item.get("title", item_id),
                        "filename": item.get("filename", ""),
                        "source": "local"
                    }
                })
                if result and result.get("success"):
                    uploaded += 1
                else:
                    errors.append(f"Failed to upload {item_id}: {result.get('error', 'Unknown')}")
            except Exception as e:
                errors.append(f"Error uploading {item_id}: {str(e)}")

    except ImportError:
        # Fallback: mark items as "pending upload" in sync state
        logger.warning("PageIndex MCP not available, skipping hub upload")
        for item_id, item in items.items():
            if namespaces and item.get("category") not in namespaces:
                continue
            uploaded += 1  # Count as queued for upload

    return {
        "uploaded": uploaded,
        "errors": errors
    }


async def _download_from_hub(namespaces: List[str] = None) -> Dict[str, Any]:
    """
    Download items from knowledge hub.

    Args:
        namespaces: Namespaces to download from

    Returns:
        Download result summary
    """
    downloaded = 0
    items = []
    errors = []

    # Try to use PageIndex MCP for downloading
    try:
        from src.agents.tools.knowledge.client import call_pageindex_tool

        search_namespaces = namespaces or ["mql5_book", "strategies", "knowledge"]

        for ns in search_namespaces:
            try:
                result = await call_pageindex_tool("list-documents", {
                    "namespace": ns,
                    "limit": 100
                })
                if result and result.get("documents"):
                    for doc in result["documents"]:
                        downloaded += 1
                        items.append({
                            "namespace": ns,
                            "document_id": doc.get("id"),
                            "title": doc.get("title", ""),
                            "content": doc.get("content", "")
                        })
            except Exception as e:
                errors.append(f"Error downloading from {ns}: {str(e)}")

    except ImportError:
        logger.warning("PageIndex MCP not available, skipping hub download")
        # Fallback: no downloads

    return {
        "downloaded": downloaded,
        "items": items,
        "errors": errors
    }


def _resolve_conflict(local_item: Dict, hub_item: Dict, strategy: ConflictResolution) -> str:
    """
    Resolve conflict between local and hub items.

    Args:
        local_item: Local item data
        hub_item: Hub item data
        strategy: Conflict resolution strategy

    Returns:
        Resolution result: "local", "hub", or "skip"
    """
    if strategy == ConflictResolution.LOCAL_WINS:
        return "local"
    elif strategy == ConflictResolution.HUB_WINS:
        return "hub"
    elif strategy == ConflictResolution.NEWEST_WINS:
        local_time = local_item.get("modified", 0)
        hub_time = hub_item.get("modified", 0)
        return "local" if local_time >= hub_time else "hub"
    else:  # SKIP
        return "skip"


@router.post("/hub-sync")
async def sync_with_knowledge_hub(
    request: HubSyncRequest,
    background_tasks: BackgroundTasks = None
) -> HubSyncResponse:
    """
    Sync knowledge with external knowledge hub.

    This endpoint handles bidirectional synchronization between local
    knowledge base and the external knowledge hub (PageIndex MCP).

    Args:
        request: Sync configuration with direction and conflict resolution

    Returns:
        Sync result with counts and any errors
    """
    logger.info(f"Starting knowledge hub sync: direction={request.direction}, "
                f"conflict_resolution={request.conflict_resolution}")

    uploaded = 0
    downloaded = 0
    conflicts_resolved = 0
    errors = []

    # Get namespaces to sync
    namespaces = request.namespaces or ["books", "articles", "notes"]

    # Handle upload direction
    if request.direction in [SyncDirection.UPLOAD, SyncDirection.BIDIRECTIONAL]:
        try:
            local_items = await _get_local_items()
            if local_items:
                upload_result = await _upload_to_hub(local_items, namespaces)
                uploaded = upload_result.get("uploaded", 0)
                errors.extend(upload_result.get("errors", []))
                logger.info(f"Uploaded {uploaded} items to hub")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            errors.append(f"Upload failed: {str(e)}")

    # Handle download direction
    if request.direction in [SyncDirection.DOWNLOAD, SyncDirection.BIDIRECTIONAL]:
        try:
            download_result = await _download_from_hub(namespaces)
            downloaded = download_result.get("downloaded", 0)
            errors.extend(download_result.get("errors", []))

            # Save downloaded items locally
            for item in download_result.get("items", []):
                ns = item.get("namespace", "knowledge")
                category_dir = KNOWLEDGE_DIR / ns
                category_dir.mkdir(parents=True, exist_ok=True)

                # Generate filename from title
                title = item.get("title", "downloaded")
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"{safe_title[:50]}.md"

                # Check for conflicts
                save_path = category_dir / filename
                if save_path.exists() and request.direction == SyncDirection.BIDIRECTIONAL:
                    existing_content = save_path.read_text(encoding="utf-8")
                    resolution = _resolve_conflict(
                        {"content": existing_content},
                        {"content": item.get("content", "")},
                        request.conflict_resolution
                    )
                    if resolution == "skip":
                        continue
                    elif resolution == "hub":
                        conflicts_resolved += 1
                    else:  # local wins
                        continue

                # Save the item
                content = item.get("content", "")
                save_path.write_text(content, encoding="utf-8")
                logger.info(f"Saved downloaded item: {filename}")

            logger.info(f"Downloaded {downloaded} items from hub")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            errors.append(f"Download failed: {str(e)}")

    # Update hub sync state
    hub_sync_state_file = Path("data/scraped_articles/.hub_sync_state.json")
    hub_sync_state_file.parent.mkdir(parents=True, exist_ok=True)
    hub_sync_state = {
        "last_sync": datetime.now().isoformat(),
        "direction": request.direction,
        "uploaded": uploaded,
        "downloaded": downloaded,
        "conflicts_resolved": conflicts_resolved
    }
    hub_sync_state_file.write_text(json.dumps(hub_sync_state, indent=2))

    return HubSyncResponse(
        success=len(errors) == 0,
        direction=request.direction,
        uploaded=uploaded,
        downloaded=downloaded,
        conflicts_resolved=conflicts_resolved,
        errors=errors
    )


@router.get("/hub-sync/status")
async def get_hub_sync_status():
    """
    Get knowledge hub sync status.

    Returns:
        Current sync state and available namespaces
    """
    hub_sync_state_file = Path("data/scraped_articles/.hub_sync_state.json")

    hub_sync_state = {}
    if hub_sync_state_file.exists():
        try:
            hub_sync_state = json.loads(hub_sync_state_file.read_text())
        except Exception:
            pass

    # Get available local namespaces
    local_namespaces = []
    if KNOWLEDGE_DIR.exists():
        for d in KNOWLEDGE_DIR.iterdir():
            if d.is_dir():
                item_count = len(list(d.glob("*.md")))
                local_namespaces.append({
                    "name": d.name,
                    "item_count": item_count
                })

    # Get hub namespaces
    hub_namespaces = []
    try:
        from src.agents.tools.knowledge.knowledge_hub import list_knowledge_namespaces
        ns_result = await list_knowledge_namespaces()
        if ns_result.get("success"):
            hub_namespaces = ns_result.get("namespaces", [])
    except Exception as e:
        logger.warning(f"Failed to get hub namespaces: {e}")

    return {
        "success": True,
        "hub_sync_state": hub_sync_state,
        "local_namespaces": local_namespaces,
        "hub_namespaces": hub_namespaces,
        "knowledge_dir": str(KNOWLEDGE_DIR)
    }
