import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.agents.tools.knowledge.knowledge_uploads_tool import list_available_uploads


def test_list_available_uploads_uses_internal_api_base_url(monkeypatch):
    monkeypatch.setenv("INTERNAL_API_BASE_URL", "https://internal.quantmindx.local")

    response = MagicMock()
    response.json.return_value = {"files": [], "count": 0}
    response.raise_for_status.return_value = None

    client = AsyncMock()
    client.get.return_value = response

    async_client = AsyncMock()
    async_client.__aenter__.return_value = client
    async_client.__aexit__.return_value = False

    monkeypatch.setattr(
        "src.agents.tools.knowledge.knowledge_uploads_tool.httpx.AsyncClient",
        lambda: async_client,
    )

    result = asyncio.run(list_available_uploads())

    assert result == {"files": [], "count": 0}
    client.get.assert_awaited_once_with("https://internal.quantmindx.local/api/knowledge/uploads")
