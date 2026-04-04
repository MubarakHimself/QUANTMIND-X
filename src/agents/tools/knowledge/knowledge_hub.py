# src/agents/tools/knowledge/knowledge_hub.py
"""Knowledge Hub Tools."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from src.agents.tools.knowledge.client import call_pageindex_tool

logger = logging.getLogger(__name__)

EA_NAMESPACE = "ea_records"
BACKTEST_REPORTS_NAMESPACE = "backtest_reports"
DEFAULT_NAMESPACES = ["mql5_book", "strategies", "knowledge", "ea_records", "backtest_reports"]

# File-based backtest reports storage (used when MCP store is unavailable)
_BACKTEST_REPORTS_DIR = Path("data/backtest_reports")
_BACKTEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)


async def search_knowledge_hub(
    query: str,
    namespaces: Optional[List[str]] = None,
    max_results: int = 10,
    include_content: bool = True
) -> Dict[str, Any]:
    """
    Search all indexed PDFs in the knowledge hub using PageIndex MCP.

    This tool searches across all indexed documents including:
    - mql5_book: MQL5 programming book
    - strategies: Trading strategies documentation
    - knowledge: General knowledge base

    Args:
        query: Search query
        namespaces: List of namespaces to search (default: all)
        max_results: Maximum results per namespace (default: 10)
        include_content: Whether to include content snippets (default: True)

    Returns:
        Dictionary containing:
        - results: List of matching documents grouped by namespace
        - total: Total number of matches
        - query: Original query
    """
    logger.info(f"Searching knowledge hub: {query}")

    if namespaces is None:
        namespaces = ["mql5_book", "strategies", "knowledge", "ea_records"]

    try:
        # Call PageIndex MCP for multi-namespace search
        result = await call_pageindex_tool("search-all", {
            "query": query,
            "namespaces": namespaces,
            "max_results": max_results,
            "include_content": include_content
        })

        results = []
        if isinstance(result, dict):
            # Group results by namespace
            namespace_results = {}
            for item in result.get("results", []):
                ns = item.get("namespace", "unknown")
                if ns not in namespace_results:
                    namespace_results[ns] = []
                namespace_results[ns].append({
                    "document": item.get("filename", item.get("document", "")),
                    "page": item.get("page", 1),
                    "content": item.get("content") if include_content else None,
                    "relevance": item.get("relevance", 0.0)
                })

            for ns, matches in namespace_results.items():
                results.append({
                    "namespace": ns,
                    "matches": matches
                })

        return {
            "success": True,
            "results": results,
            "total": sum(len(r["matches"]) for r in results),
            "query": query,
            "namespaces_searched": namespaces
        }

    except Exception as e:
        logger.error(f"Knowledge hub search failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "total": 0,
            "query": query,
            "namespaces_searched": namespaces
        }


async def get_article_content(
    article_id: str,
    namespace: str
) -> Dict[str, Any]:
    """
    Retrieve full content of an indexed article/document using PageIndex MCP.

    Args:
        article_id: Article or document identifier
        namespace: Namespace the article belongs to

    Returns:
        Dictionary containing the full article content
    """
    logger.info(f"Getting article content: {article_id} from {namespace}")

    try:
        # Call PageIndex MCP to get document content
        result = await call_pageindex_tool("get-document", {
            "document_id": article_id,
            "namespace": namespace
        })

        if isinstance(result, dict):
            return {
                "success": True,
                "article_id": article_id,
                "namespace": namespace,
                "title": result.get("title", f"Article {article_id}"),
                "content": result.get("content", ""),
                "metadata": {
                    "indexed_at": result.get("indexed_at", ""),
                    "pages": result.get("pages", 0)
                }
            }

        return {
            "success": False,
            "article_id": article_id,
            "namespace": namespace,
            "title": "",
            "content": "",
            "error": "Article not found"
        }

    except Exception as e:
        logger.error(f"Failed to get article content: {e}")
        return {
            "success": False,
            "article_id": article_id,
            "namespace": namespace,
            "title": "",
            "content": "",
            "error": str(e)
        }


async def list_knowledge_namespaces() -> Dict[str, Any]:
    """
    List all available knowledge namespaces using PageIndex MCP.

    Returns:
        Dictionary containing list of namespaces with metadata
    """
    logger.info("Listing knowledge namespaces")

    try:
        # Call PageIndex MCP to list namespaces
        result = await call_pageindex_tool("list-namespaces", {})

        namespaces = []
        if isinstance(result, dict):
            for ns in result.get("namespaces", []):
                namespaces.append({
                    "name": ns.get("name", ""),
                    "description": ns.get("description", ""),
                    "document_count": ns.get("document_count", 0),
                    "total_pages": ns.get("total_pages", 0),
                    "indexed_at": ns.get("indexed_at")
                })

        # Add default namespaces if not present
        default_namespaces = {
            "mql5_book": "MQL5 Programming Book",
            "strategies": "Trading strategies documentation",
            "knowledge": "General knowledge base"
        }

        existing_names = [ns["name"] for ns in namespaces]
        for name, desc in default_namespaces.items():
            if name not in existing_names:
                namespaces.append({
                    "name": name,
                    "description": desc,
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                })

        return {
            "success": True,
            "namespaces": sorted(namespaces, key=lambda x: x["name"]),
            "total": len(namespaces)
        }

    except Exception as e:
        logger.error(f"Failed to list namespaces: {e}")
        return {
            "success": False,
            "error": str(e),
            "namespaces": [
                {
                    "name": "mql5_book",
                    "description": "MQL5 Programming Book",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                },
                {
                    "name": "strategies",
                    "description": "Trading strategies documentation",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                },
                {
                    "name": "knowledge",
                    "description": "General knowledge base",
                    "document_count": 0,
                    "total_pages": 0,
                    "indexed_at": None
                }
            ],
            "total": 3
        }


async def resurrect_strategy_from_retired(strategy_id: str, session_mask: str = None) -> dict:
    """
    Moves a strategy from RETIRED state back to PAPER state.

    This function:
    1. Looks up the EA record file for the strategy
    2. Validates the strategy is in RETIRED state (not ARCHIVED)
    3. Updates the BotRegistry to reflect the state transition
    4. Updates the SSL state in the database (BotCircuitBreaker)
    5. Updates the EA record file with the new state
    6. Dispatches a mail notification to Portfolio Head

    Args:
        strategy_id: The strategy/bot ID to resurrect
        session_mask: Optional session mask for context

    Returns:
        Dict with success status and details
    """
    from pathlib import Path
    import re

    from src.agents.departments.department_mail import (
        get_mail_service,
        MessageType,
        Priority,
    )
    from src.router.bot_manifest import BotRegistry
    from src.risk.ssl.state import SSLCircuitBreakerState, SSLState

    logger.info(f"Resurrecting strategy {strategy_id} from RETIRED to PAPER")

    # 1. Look up the EA record file
    ea_record_path = Path("data/ea_records") / f"{strategy_id}.md"
    if not ea_record_path.exists():
        return {
            "success": False,
            "error": f"EA record not found for strategy {strategy_id}",
            "status": "not_found",
        }

    # Read and parse the EA record to validate current state
    try:
        ea_content = ea_record_path.read_text()
    except Exception as e:
        logger.error(f"Failed to read EA record for {strategy_id}: {e}")
        return {
            "success": False,
            "error": f"Failed to read EA record: {str(e)}",
            "status": "read_error",
        }

    # 2. Validate it is RETIRED (not ARCHIVED)
    # Look for "Status: RETIRED" or "Status: ARCHIVED" in the EA record
    status_match = re.search(r"\*\*Status:\*\*\s*(\w+)", ea_content)
    if not status_match:
        return {
            "success": False,
            "error": "Could not parse status from EA record",
            "status": "parse_error",
        }

    current_status = status_match.group(1).upper()
    if current_status == "ARCHIVED":
        return {
            "success": False,
            "error": f"Cannot resurrect ARCHIVED strategy {strategy_id}. Archives are permanent.",
            "status": "archived",
        }
    if current_status != "RETIRED":
        return {
            "success": False,
            "error": f"Strategy {strategy_id} is in {current_status} state, expected RETIRED",
            "status": "invalid_state",
        }

    # 3. Update BotRegistry
    try:
        registry = BotRegistry()
        manifest = registry.get(strategy_id)
        if manifest is None:
            logger.warning(f"Strategy {strategy_id} not found in BotRegistry, will update EA record only")
        else:
            # Update the manifest's trading mode back to PAPER
            # Remove @dead tag if present, ensure @paper or similar tag is set
            if hasattr(manifest, 'tags'):
                tags = manifest.tags
                if '@dead' in tags:
                    tags.remove('@dead')
                if '@paper' not in tags:
                    tags.append('@paper')
                manifest.tags = tags
            registry._save()
            logger.info(f"Updated BotRegistry for {strategy_id}")
    except Exception as e:
        logger.error(f"Failed to update BotRegistry for {strategy_id}: {e}")
        # Continue with other updates even if this fails

    # 4. Update SSL state in database
    try:
        state_manager = SSLCircuitBreakerState()
        current_ssl_state = state_manager.get_state(strategy_id)
        if current_ssl_state == SSLState.RETIRED:
            # Force transition from RETIRED to PAPER (bypass normal validation for manual resurrection)
            success = state_manager.update_state(
                strategy_id,
                new_state=SSLState.PAPER,
                consecutive_losses=0,
                recovery_win_count=0,
                force=True,
            )
            if success:
                logger.info(f"Updated SSL state for {strategy_id} from RETIRED to PAPER")
            else:
                logger.warning(f"Failed to update SSL state for {strategy_id}")
        else:
            logger.info(f"SSL state for {strategy_id} is {current_ssl_state.value}, no DB update needed")
    except Exception as e:
        logger.error(f"Failed to update SSL state for {strategy_id}: {e}")
        # Continue with other updates even if this fails

    # 5. Update EA record file with new state
    try:
        new_content = re.sub(
            r"(\*\*Status:\*\*\s*)(RETIRED)",
            r"\1PAPER",
            ea_content
        )
        ea_record_path.write_text(new_content)
        logger.info(f"Updated EA record for {strategy_id} to PAPER state")
    except Exception as e:
        logger.error(f"Failed to update EA record for {strategy_id}: {e}")
        return {
            "success": False,
            "error": f"Failed to update EA record: {str(e)}",
            "status": "write_error",
        }

    # 6. Dispatch mail notification to Portfolio Head
    try:
        mail_service = get_mail_service()
        message = mail_service.send(
            from_dept="knowledge_hub",
            to_dept="portfolio",
            type=MessageType.STATUS,
            subject=f"Strategy Resurrected: {strategy_id}",
            body=f"""Strategy {strategy_id} has been resurrected from RETIRED state back to PAPER trading.

Session Mask: {session_mask or 'N/A'}

This strategy is now eligible for paper trading evaluation.

EA Record: data/ea_records/{strategy_id}.md
""",
            priority=Priority.NORMAL,
        )
        logger.info(f"Dispatched resurrection notification for {strategy_id}: message_id={message.id}")
    except Exception as e:
        logger.error(f"Failed to dispatch mail notification for {strategy_id}: {e}")
        # Mail failure is non-fatal, continue

    return {
        "success": True,
        "strategy_id": strategy_id,
        "previous_state": "RETIRED",
        "new_state": "PAPER",
        "status": "resurrected",
        "session_mask": session_mask,
    }


def store_backtest_report(
    strategy_id: str,
    report: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Store a backtest report in the knowledge hub's backtest_reports namespace.

    Stores the report as a JSON file keyed by strategy_id and timestamp,
    enabling retrieval by BotAnalystSubAgent and other agents.

    Args:
        strategy_id: Strategy identifier
        report: Markdown report string
        metadata: Optional metadata (trd_data, sit_result, backtest_summary)

    Returns:
        Dict with success status and report_id
    """
    import hashlib

    timestamp = datetime.utcnow().isoformat()
    report_id = hashlib.sha256(f"{strategy_id}_{timestamp}".encode()).hexdigest()[:16]

    record = {
        "report_id": report_id,
        "strategy_id": strategy_id,
        "timestamp": timestamp,
        "report": report,
        "metadata": metadata or {},
    }

    filepath = _BACKTEST_REPORTS_DIR / f"{strategy_id}_{report_id}.json"
    try:
        filepath.write_text(json.dumps(record, indent=2))
        logger.info(f"Stored backtest report for {strategy_id} as {report_id}")
        return {
            "success": True,
            "report_id": report_id,
            "strategy_id": strategy_id,
            "stored_at": timestamp,
            "namespace": BACKTEST_REPORTS_NAMESPACE,
        }
    except Exception as e:
        logger.error(f"Failed to store backtest report for {strategy_id}: {e}")
        return {
            "success": False,
            "strategy_id": strategy_id,
            "error": str(e),
        }


def retrieve_backtest_report(strategy_id: str, limit: int = 5) -> Dict[str, Any]:
    """
    Retrieve the most recent backtest reports for a strategy from the knowledge hub.

    Args:
        strategy_id: Strategy identifier
        limit: Maximum number of reports to return (default 5)

    Returns:
        Dict with success status and list of report records
    """
    try:
        pattern = f"{strategy_id}_*.json"
        files = sorted(
            _BACKTEST_REPORTS_DIR.glob(pattern),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )[:limit]

        reports = []
        for f in files:
            try:
                record = json.loads(f.read_text())
                reports.append({
                    "report_id": record.get("report_id"),
                    "strategy_id": record.get("strategy_id"),
                    "timestamp": record.get("timestamp"),
                    "report": record.get("report"),
                    "metadata": record.get("metadata", {}),
                })
            except Exception:
                continue

        return {
            "success": True,
            "strategy_id": strategy_id,
            "reports": reports,
            "count": len(reports),
            "namespace": BACKTEST_REPORTS_NAMESPACE,
        }
    except Exception as e:
        logger.error(f"Failed to retrieve backtest reports for {strategy_id}: {e}")
        return {
            "success": False,
            "strategy_id": strategy_id,
            "reports": [],
            "error": str(e),
        }
