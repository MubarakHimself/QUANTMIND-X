"""
Knowledge Uploads Tool

Provides tools for agents to list and read user-uploaded files
from the personal knowledge base directory.
"""

import os

import httpx
from pathlib import Path
from typing import Dict, Any, List

from src.config import get_internal_api_base_url


async def list_available_uploads() -> Dict[str, Any]:
    """
    Returns all user-uploaded files accessible to this agent.

    Returns:
        Dictionary with 'files' list and 'count' of files
    """
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{get_internal_api_base_url()}/api/knowledge/uploads")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"files": [], "count": 0, "error": str(e)}


async def read_upload(filename: str) -> Dict[str, Any]:
    """
    Returns text content of an uploaded file.

    Args:
        filename: Name of the file to read

    Returns:
        Dictionary with 'content' and 'filename', or 'error'

    Note:
        This reads from local disk assuming the agent runs on the same node
        as the knowledge base storage. For distributed setups, this would
        need to be replaced with an HTTP call to the appropriate node.
    """
    path = Path("data/knowledge_base/personal") / filename
    if not path.exists():
        return {"error": f"File {filename} not found"}
    try:
        content = path.read_text(errors='replace')
        return {"content": content, "filename": filename}
    except Exception as e:
        return {"error": f"Failed to read {filename}: {e}"}


def list_uploads_sync() -> List[Dict[str, Any]]:
    """
    Synchronous version of list_available_uploads for non-async contexts.

    Returns:
        List of file dictionaries
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
    return files
