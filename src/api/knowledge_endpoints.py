"""
QuantMind Unified Knowledge API Endpoints

Provides unified search and source-status endpoints that fan out
queries to all 3 PageIndex instances (articles, books, logs) in
parallel and merge results by relevance score.

Story 6.1: PageIndex Integration & Knowledge API
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

from src.agents.knowledge.router import PageIndexClient, kb_router as _kb_router

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["knowledge-unified"])

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

KNOWN_COLLECTIONS: List[str] = ["articles", "books", "logs", "ea_records"]


class KnowledgeSourceStatus(BaseModel):
    id: str              # "articles" | "books" | "logs"
    type: str            # same as id for now
    status: str          # "online" | "offline"
    document_count: int  # from PageIndex /stats


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query text")
    sources: Optional[List[str]] = Field(
        default=None,
        description="Subset of ['articles','books','logs']; null = all 3",
    )
    limit: int = Field(default=5, ge=1, le=100, description="Max results per source")


class KnowledgeSearchResult(BaseModel):
    source_type: str
    title: str
    excerpt: str
    relevance_score: float
    provenance: Dict[str, Any]  # { source_url, source_type, indexed_at_utc }


class KnowledgeSearchResponse(BaseModel):
    results: List[KnowledgeSearchResult]
    total: int
    query: str
    warnings: List[str]  # offline instances reported here


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/sources", response_model=List[KnowledgeSourceStatus])
async def get_knowledge_sources() -> List[KnowledgeSourceStatus]:
    """
    GET /api/knowledge/sources

    Query all 3 PageIndex instances for their health and document count.
    Offline instances are returned with status="offline" and document_count=0
    rather than raising an error.

    Returns:
        List of KnowledgeSourceStatus for articles, books, and logs.
    """
    # Reuse the module-level singleton client to avoid creating new connection pools per request
    client: PageIndexClient = _kb_router.client if _kb_router.client else PageIndexClient()

    async def _check_source(collection: str) -> KnowledgeSourceStatus:
        try:
            health_task = client.health_check_async(collection)
            stats_task = client.get_stats_async(collection)
            health, stats = await asyncio.gather(health_task, stats_task)

            is_healthy = (
                isinstance(health, dict)
                and health.get("status") not in ("unhealthy", "error")
            )
            document_count = int(stats.get("document_count", stats.get("count", 0)))
            return KnowledgeSourceStatus(
                id=collection,
                type=collection,
                status="online" if is_healthy else "offline",
                document_count=document_count,
            )
        except Exception as e:
            logger.warning(f"Source check failed for {collection}: {e}")
            return KnowledgeSourceStatus(
                id=collection,
                type=collection,
                status="offline",
                document_count=0,
            )

    tasks = [_check_source(c) for c in KNOWN_COLLECTIONS]
    results = await asyncio.gather(*tasks)
    return list(results)


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(request: KnowledgeSearchRequest) -> KnowledgeSearchResponse:
    """
    POST /api/knowledge/search

    Fan out the search query to all requested PageIndex instances in
    parallel using asyncio.gather(return_exceptions=True).  Offline
    instances are skipped and reported in the warnings list.  Results
    from online instances are merged and sorted by relevance_score
    descending.

    Request body:
        query   (str)           - Search query text
        sources (List[str]|null) - Subset of ["articles","books","logs"]; null = all
        limit   (int)           - Max results per source (default 5)

    Returns:
        KnowledgeSearchResponse with merged results and any warnings.
    """
    requested = request.sources if request.sources else KNOWN_COLLECTIONS
    # Guard: filter to valid collections only; warn on unknown source names
    unknown_sources = [s for s in requested if s not in KNOWN_COLLECTIONS]
    sources = [s for s in requested if s in KNOWN_COLLECTIONS]

    # Reuse the module-level singleton client to avoid creating new connection pools per request
    client: PageIndexClient = _kb_router.client if _kb_router.client else PageIndexClient()

    tasks = [
        client.search_async(request.query, src, request.limit)
        for src in sources
    ]
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    merged: List[KnowledgeSearchResult] = []
    # Pre-populate warnings for any unknown sources requested by the caller
    warnings: List[str] = [
        f"Unknown source '{s}' ignored — valid sources are {KNOWN_COLLECTIONS}"
        for s in unknown_sources
    ]

    for source, result in zip(sources, raw_results):
        if isinstance(result, Exception):
            warning_msg = f"{source} instance offline: {str(result)[:100]}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)
            continue

        for item in result:
            raw_source = item.get("source", "unknown")
            # Derive a human-readable title from the file path / URL
            # e.g. "path/to/my_article.pdf" → "my_article.pdf"
            title = raw_source.split("/")[-1] if raw_source and "/" in raw_source else raw_source
            merged.append(
                KnowledgeSearchResult(
                    source_type=source,
                    title=title,
                    excerpt=item.get("content", "")[:300],
                    relevance_score=float(item.get("score", 0.0)),
                    provenance={
                        "source_url": raw_source,
                        "source_type": source,
                        "indexed_at_utc": None,
                    },
                )
            )

    merged.sort(key=lambda r: r.relevance_score, reverse=True)

    return KnowledgeSearchResponse(
        results=merged,
        total=len(merged),
        query=request.query,
        warnings=warnings,
    )


@router.get("/uploads")
async def list_uploaded_files():
    """
    GET /api/knowledge/uploads

    Returns all user-uploaded files from the personal knowledge base directory.
    """
    personal_dir = Path("data/knowledge_base/personal")
    files = []
    if personal_dir.exists():
        for f in personal_dir.iterdir():
            if f.is_file():
                files.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_kb": f.stat().st_size // 1024,
                    "modified": f.stat().st_mtime,
                    "type": f.suffix.lower()
                })
    return {"files": files, "count": len(files)}
