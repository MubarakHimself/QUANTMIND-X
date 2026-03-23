"""
QuantMind Knowledge Ingest API Endpoints

Provides web scraping (via Firecrawl) and personal knowledge ingestion
endpoints. Scraped and personal content is written to the PageIndex
volume-mount directories so the running PageIndex Docker instances
auto-index the files.

Story 6.2: Web Scraping, Article Ingestion & Personal Knowledge API
"""

import asyncio
import logging
import os
import re
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import AnyHttpUrl, BaseModel

try:
    from firecrawl import FirecrawlApp  # type: ignore[import]
except ImportError:  # pragma: no cover
    FirecrawlApp = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge-ingest"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRAPED_ARTICLES_DIR = Path("data/scraped_articles")
PERSONAL_NOTES_DIR = Path("data/knowledge_base/personal")

ALLOWED_PERSONAL_FILE_TYPES = {".pdf", ".md", ".txt"}

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IngestUrlRequest(BaseModel):
    url: AnyHttpUrl
    relevance_tags: List[str] = []


class IngestJobResponse(BaseModel):
    job_id: str
    status: str  # "success" | "failed"
    source_url: Optional[str] = None
    scraped_at_utc: Optional[str] = None
    type: Optional[str] = None
    created_at_utc: Optional[str] = None
    source_description: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Retry helper (no external library — manual exponential backoff)
# ---------------------------------------------------------------------------


async def _retry_with_backoff(
    func, *args, max_retries: int = 3, base_delay: float = 1.0, **kwargs
):
    """Execute a synchronous function in an executor with exponential backoff retry.

    Delays between attempts: 1s, 2s, 4s (base_delay * 2 ** attempt).
    Raises the last exception after max_retries are exhausted.
    """
    loop = asyncio.get_running_loop()
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s. Retrying in %.1fs…",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

    logger.error(
        "All %d attempts failed. Last error: %s\n%s",
        max_retries,
        last_exc,
        traceback.format_exc(),
    )
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_filename(text: str, max_len: int = 80) -> str:
    """Convert arbitrary text to a safe filesystem filename."""
    # Replace non-alphanumeric characters with underscore
    safe = re.sub(r"[^\w\-]", "_", text)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:max_len] or "untitled"


def _yaml_safe_str(value: str) -> str:
    """Sanitize a string value for safe embedding in YAML front-matter.

    Strips newlines and characters that could break or inject into the YAML
    structure (e.g. a URL containing '\\n' followed by a YAML key would inject
    arbitrary front-matter fields). The value is also single-quoted so that
    colons and other YAML metacharacters are treated as literals.

    Colons preceded by a newline (field patterns like "key:\\n") have the
    colon escaped to prevent YAML injection when content is written after
    the front-matter separator.
    """
    s = re.sub(r"(\n)(\s*\w+):", r"\1\2\\:", str(value))
    # Replace remaining problematic characters
    s = re.sub(r"[\r\n\x00]", " ", s)
    # Escape single quotes by doubling them
    s = s.replace("'", "''")
    return f"'{s}'"


def _extract_markdown(result) -> Optional[str]:
    """Extract markdown content from a Firecrawl SDK response.

    The SDK may return either a dict or an object — handle both forms,
    mirroring the pattern in scripts/firecrawl_scraper.py:139-157.
    """
    if isinstance(result, dict):
        return (
            result.get("markdown")
            or result.get("data", {}).get("markdown")
            or result.get("content")
        )
    if hasattr(result, "markdown"):
        return result.markdown
    if hasattr(result, "data") and hasattr(result.data, "markdown"):
        return result.data.markdown
    return None


def _write_article(
    content: str,
    source_url: str,
    scraped_at_utc: str,
    relevance_tags: List[str],
    job_id: str,
) -> Path:
    """Write a scraped article as a markdown file to the articles volume-mount dir."""
    SCRAPED_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{job_id}.md"
    dest = SCRAPED_ARTICLES_DIR / filename

    front_matter = (
        f"---\n"
        f"job_id: {job_id}\n"
        f"source_url: {_yaml_safe_str(source_url)}\n"
        f"scraped_at_utc: {scraped_at_utc}\n"
        f"relevance_tags: {[_yaml_safe_str(t) for t in relevance_tags]}\n"
        f"---\n\n"
    )
    dest.write_text(front_matter + content, encoding="utf-8")
    logger.info("Wrote scraped article to %s (%d chars)", dest, len(content))
    return dest


def _write_personal_note(
    content: str,
    source_description: str,
    tags: List[str],
    job_id: str,
    created_at_utc: str,
) -> Path:
    """Write a personal note as a markdown file to the personal knowledge volume-mount dir."""
    PERSONAL_NOTES_DIR.mkdir(parents=True, exist_ok=True)
    safe_desc = _safe_filename(source_description or "note")
    filename = f"{job_id}_{safe_desc}.md"
    dest = PERSONAL_NOTES_DIR / filename

    all_tags = ["personal"] + [_yaml_safe_str(t) for t in tags]
    front_matter = (
        f"---\n"
        f"job_id: {job_id}\n"
        f"type: personal\n"
        f"source_description: {_yaml_safe_str(source_description)}\n"
        f"created_at_utc: {created_at_utc}\n"
        f"tags: {all_tags}\n"
        f"---\n\n"
    )
    dest.write_text(front_matter + _yaml_safe_str(content) + "\n", encoding="utf-8")
    logger.info("Wrote personal note to %s (%d chars)", dest, len(content))
    return dest


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestJobResponse)
async def ingest_url(request: IngestUrlRequest) -> IngestJobResponse:
    """POST /api/knowledge/ingest

    Scrape the given URL via Firecrawl, then write the markdown to the
    PageIndex articles volume-mount directory so it is auto-indexed.

    Returns:
        IngestJobResponse with job_id, status, source_url, scraped_at_utc.
    """
    # Validate URL length BEFORE API key check (so we return 422 not 503)
    url_str = str(request.url)
    if len(url_str) > 2000:
        raise HTTPException(
            status_code=422,
            detail={"error": f"URL exceeds maximum length of 2000 characters (got {len(url_str)})"},
        )

    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail={"error": "Firecrawl API key not configured"},
        )

    url_str = str(request.url)
    job_id = str(uuid.uuid4())
    scraped_at_utc = datetime.now(timezone.utc).isoformat()

    if FirecrawlApp is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "firecrawl-py package is not installed"},
        )

    try:
        app_instance = FirecrawlApp(api_key=api_key)

        result = await _retry_with_backoff(
            app_instance.scrape,
            url_str,
            formats=["markdown"],
        )
    except Exception as e:
        logger.error(
            "Firecrawl ingest failed for %s: %s\n%s",
            url_str,
            e,
            traceback.format_exc(),
        )
        return IngestJobResponse(
            job_id=job_id,
            status="failed",
            source_url=url_str,
            scraped_at_utc=scraped_at_utc,
            error=str(e),
        )

    markdown_content = _extract_markdown(result)

    if not markdown_content:
        logger.warning(
            "Firecrawl returned empty markdown for %s (possibly paywalled). job_id=%s",
            url_str,
            job_id,
        )
        return IngestJobResponse(
            job_id=job_id,
            status="failed",
            source_url=url_str,
            scraped_at_utc=scraped_at_utc,
            error="Empty content returned — page may be paywalled or inaccessible",
        )

    try:
        _write_article(
            content=markdown_content,
            source_url=url_str,
            scraped_at_utc=scraped_at_utc,
            relevance_tags=request.relevance_tags,
            job_id=job_id,
        )
    except OSError as e:
        logger.error(
            "Failed to write article file for job %s: %s\n%s",
            job_id,
            e,
            traceback.format_exc(),
        )
        return IngestJobResponse(
            job_id=job_id,
            status="failed",
            source_url=url_str,
            scraped_at_utc=scraped_at_utc,
            error=f"File write error: {e}",
        )

    return IngestJobResponse(
        job_id=job_id,
        status="success",
        source_url=url_str,
        scraped_at_utc=scraped_at_utc,
    )


@router.post("/personal", response_model=IngestJobResponse)
async def ingest_personal(
    content: Optional[str] = Form(default=None),
    source_description: str = Form(default=""),
    tags: str = Form(default=""),
    file: Optional[UploadFile] = File(default=None),
) -> IngestJobResponse:
    """POST /api/knowledge/personal

    Accept a personal note (text/markdown via form field) or a file upload
    (PDF, MD, TXT) and write it to the personal knowledge volume-mount
    directory so PageIndex auto-indexes it.

    Form fields:
        content            (str)  — Markdown or plain text body
        source_description (str)  — e.g. "Trading journal 2026-03-19"
        tags               (str)  — Comma-separated list of tags
        file               (file) — Optional PDF, MD, or TXT file upload

    Returns:
        IngestJobResponse with job_id, status, type="personal", created_at_utc.
    """
    job_id = str(uuid.uuid4())
    created_at_utc = datetime.now(timezone.utc).isoformat()
    parsed_tags = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    # Determine content source
    if file is not None:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_PERSONAL_FILE_TYPES:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_PERSONAL_FILE_TYPES)}",
            )

        raw_bytes = await file.read()

        if suffix == ".pdf":
            # Store raw PDF bytes directly — PageIndex will handle PDF parsing
            PERSONAL_NOTES_DIR.mkdir(parents=True, exist_ok=True)
            safe_desc = _safe_filename(source_description or file.filename or "pdf")
            pdf_path = PERSONAL_NOTES_DIR / f"{job_id}_{safe_desc}.pdf"
            pdf_path.write_bytes(raw_bytes)
            logger.info(
                "Wrote personal PDF to %s (%d bytes)", pdf_path, len(raw_bytes)
            )
        else:
            # MD / TXT — decode and write as markdown
            try:
                note_content = raw_bytes.decode("utf-8", errors="replace")
            except Exception as e:
                logger.error(
                    "Failed to decode personal file upload (job=%s): %s\n%s",
                    job_id,
                    e,
                    traceback.format_exc(),
                )
                return IngestJobResponse(
                    job_id=job_id,
                    status="failed",
                    type="personal",
                    created_at_utc=created_at_utc,
                    source_description=source_description,
                    error=f"Failed to decode file: {e}",
                )

            desc = source_description or file.filename or "uploaded-note"
            try:
                _write_personal_note(
                    content=note_content,
                    source_description=desc,
                    tags=parsed_tags,
                    job_id=job_id,
                    created_at_utc=created_at_utc,
                )
            except OSError as e:
                logger.error(
                    "Failed to write personal note file for job %s: %s\n%s",
                    job_id,
                    e,
                    traceback.format_exc(),
                )
                return IngestJobResponse(
                    job_id=job_id,
                    status="failed",
                    type="personal",
                    created_at_utc=created_at_utc,
                    source_description=desc,
                    error=f"File write error: {e}",
                )

    elif content:
        try:
            _write_personal_note(
                content=content,
                source_description=source_description,
                tags=parsed_tags,
                job_id=job_id,
                created_at_utc=created_at_utc,
            )
        except OSError as e:
            logger.error(
                "Failed to write personal note for job %s: %s\n%s",
                job_id,
                e,
                traceback.format_exc(),
            )
            return IngestJobResponse(
                job_id=job_id,
                status="failed",
                type="personal",
                created_at_utc=created_at_utc,
                source_description=source_description,
                error=f"File write error: {e}",
            )
    else:
        raise HTTPException(
            status_code=422,
            detail="Either 'content' field or file upload is required",
        )

    return IngestJobResponse(
        job_id=job_id,
        status="success",
        type="personal",
        created_at_utc=created_at_utc,
        source_description=source_description,
    )
