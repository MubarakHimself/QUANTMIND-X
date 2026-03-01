"""
NotebookLM MCP Server Integration

Provides MCP tools for NotebookLM operations:
- List notebooks
- Create notebooks
- Add sources
- Query notebooks
- Create audio/video

Note: Requires browser cookie authentication via:
  nlm setup add (opens browser for auth)
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# NotebookLM MCP client instance
_notebooklm_client = None


def get_notebooklm_client():
    """Get or create NotebookLM MCP client."""
    global _notebooklm_client
    if _notebooklm_client is None:
        _notebooklm_client = NotebookLMClient()
    return _notebooklm_client


class NotebookLMClient:
    """Client for NotebookLM MCP operations."""

    def __init__(self):
        self.cookie_path = os.path.expanduser("~/.config/notebooklm/cookies.json")
        self._check_auth()

    def _check_auth(self):
        """Check if NotebookLM is authenticated."""
        if not os.path.exists(self.cookie_path):
            logger.warning(
                "NotebookLM not authenticated. Run 'nlm setup add' to authenticate."
            )

    async def list_notebooks(self) -> Dict[str, Any]:
        """List all notebooks."""
        proc = await asyncio.create_subprocess_exec(
            "nlm", "notebook", "list",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"NotebookLM error: {stderr.decode()}")
        return {"notebooks": stdout.decode().splitlines()}

    async def create_notebook(self, name: str) -> Dict[str, Any]:
        """Create a new notebook."""
        proc = await asyncio.create_subprocess_exec(
            "nlm", "notebook", "create", name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"NotebookLM error: {stderr.decode()}")
        return {"notebook_id": stdout.decode().strip()}

    async def add_source(self, source: str, notebook_id: str) -> Dict[str, Any]:
        """Add source to notebook (URL, file, or text)."""
        proc = await asyncio.create_subprocess_exec(
            "nlm", "source", "add", source, "--notebook", notebook_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"NotebookLM error: {stderr.decode()}")
        return {"source_added": stdout.decode().strip()}

    async def query_notebook(self, notebook_id: str, query: str) -> Dict[str, Any]:
        """Query a notebook with AI."""
        proc = await asyncio.create_subprocess_exec(
            "nlm", "notebook", "query", "--notebook", notebook_id, query,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"NotebookLM error: {stderr.decode()}")
        return {"response": stdout.decode()}

    async def create_audio(self, notebook_id: str) -> Dict[str, Any]:
        """Create audio overview from notebook."""
        proc = await asyncio.create_subprocess_exec(
            "nlm", "studio", "create", "--notebook", notebook_id, "--type", "audio",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"NotebookLM error: {stderr.decode()}")
        return {"audio_created": stdout.decode().strip()}


# MCP tool definitions for Claude Agent SDK
NOTEBOOKLM_TOOLS = [
    {
        "name": "notebook_list",
        "description": "List all notebooks in NotebookLM",
    },
    {
        "name": "notebook_create",
        "description": "Create a new notebook",
        "parameters": {
            "name": {"type": "string", "description": "Notebook name"}
        }
    },
    {
        "name": "notebook_query",
        "description": "Query a notebook with AI",
        "parameters": {
            "notebook_id": {"type": "string"},
            "query": {"type": "string"}
        }
    },
    {
        "name": "notebook_add_source",
        "description": "Add a source to a notebook",
        "parameters": {
            "notebook_id": {"type": "string"},
            "source": {"type": "string", "description": "URL, file path, or text"}
        }
    },
]
