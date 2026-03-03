# tests/agents/tools/test_knowledge.py
"""Tests for knowledge tools modular structure.

This test suite verifies the refactored knowledge_tools.py into a modular structure.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestKnowledgeModuleImports:
    """Test that all modules can be imported."""

    def test_import_knowledge_client(self):
        """Should import knowledge client module."""
        from src.agents.tools.knowledge.client import PageIndexClient
        assert PageIndexClient is not None

    def test_import_mql5_book_tools(self):
        """Should import MQL5 book tools."""
        from src.agents.tools.knowledge.mql5_book import (
            search_mql5_book,
            get_mql5_book_section,
        )
        assert search_mql5_book is not None
        assert get_mql5_book_section is not None

    def test_import_knowledge_hub_tools(self):
        """Should import knowledge hub tools."""
        from src.agents.tools.knowledge.knowledge_hub import (
            search_knowledge_hub,
            get_article_content,
            list_knowledge_namespaces,
        )
        assert search_knowledge_hub is not None
        assert get_article_content is not None
        assert list_knowledge_namespaces is not None

    def test_import_pdf_indexing_tools(self):
        """Should import PDF indexing tools."""
        from src.agents.tools.knowledge.pdf_indexing import (
            index_pdf_document,
            get_indexing_status,
            list_indexed_documents,
            remove_indexed_document,
        )
        assert index_pdf_document is not None
        assert get_indexing_status is not None
        assert list_indexed_documents is not None
        assert remove_indexed_document is not None

    def test_import_strategy_tools(self):
        """Should import strategy-specific tools."""
        from src.agents.tools.knowledge.strategies import (
            search_strategy_patterns,
            get_indicator_template,
        )
        assert search_strategy_patterns is not None
        assert get_indicator_template is not None

    def test_import_registry(self):
        """Should import tool registry."""
        from src.agents.tools.knowledge.registry import (
            KNOWLEDGE_TOOLS,
            get_knowledge_tool,
            list_knowledge_tools,
            invoke_knowledge_tool,
        )
        assert KNOWLEDGE_TOOLS is not None
        assert get_knowledge_tool is not None
        assert list_knowledge_tools is not None
        assert invoke_knowledge_tool is not None


class TestBackwardCompatibility:
    """Test that original knowledge_tools.py still works."""

    def test_import_original_module(self):
        """Should import original module for backward compatibility."""
        from src.agents.tools import knowledge_tools
        assert knowledge_tools is not None

    def test_original_functions_available(self):
        """Should have all original functions available."""
        from src.agents.tools.knowledge_tools import (
            search_mql5_book,
            get_mql5_book_section,
            search_knowledge_hub,
            get_article_content,
            list_knowledge_namespaces,
            index_pdf_document,
            get_indexing_status,
            list_indexed_documents,
            remove_indexed_document,
            search_strategy_patterns,
            get_indicator_template,
            get_knowledge_tool,
            list_knowledge_tools,
            invoke_knowledge_tool,
        )
        # All functions should be importable
        assert callable(search_mql5_book)
        assert callable(get_mql5_book_section)

    def test_tool_registry_compatible(self):
        """Tool registry should have expected structure."""
        from src.agents.tools.knowledge_tools import KNOWLEDGE_TOOLS
        assert isinstance(KNOWLEDGE_TOOLS, dict)
        assert "search_mql5_book" in KNOWLEDGE_TOOLS
        assert "search_knowledge_hub" in KNOWLEDGE_TOOLS
        assert "index_pdf_document" in KNOWLEDGE_TOOLS


class TestPageIndexClient:
    """Test PageIndexClient class."""

    def test_client_initialization(self):
        """Client should initialize with correct defaults."""
        from src.agents.tools.knowledge.client import PageIndexClient
        client = PageIndexClient()
        assert client is not None

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """Should call PageIndex tool with proper error handling."""
        from src.agents.tools.knowledge.client import PageIndexClient

        with patch('src.agents.tools.knowledge.client.get_pageindex_manager') as mock_get_manager:
            mock_manager = AsyncMock()
            mock_manager.call_tool = AsyncMock(return_value={"results": []})
            mock_get_manager.return_value = mock_manager

            client = PageIndexClient()
            result = await client.call_tool("search", {"query": "test"})

            assert isinstance(result, dict)


class TestMQL5BookTools:
    """Test MQL5 book search functions."""

    @pytest.mark.asyncio
    async def test_search_mql5_book_returns_structure(self):
        """search_mql5_book should return expected structure."""
        from src.agents.tools.knowledge.mql5_book import search_mql5_book

        with patch('src.agents.tools.knowledge.mql5_book.call_pageindex_tool') as mock_call:
            mock_call.return_value = {
                "results": [
                    {"page": 10, "chapter": "Chapter 1", "content": "Test content", "relevance": 0.9}
                ],
                "total": 1
            }

            result = await search_mql5_book("test query")

            assert result["success"] is True
            assert "results" in result
            assert result["query"] == "test query"
            assert result["namespace"] == "mql5_book"

    @pytest.mark.asyncio
    async def test_search_mql5_book_handles_error(self):
        """search_mql5_book should handle errors gracefully."""
        from src.agents.tools.knowledge.mql5_book import search_mql5_book

        with patch('src.agents.tools.knowledge.mql5_book.call_pageindex_tool') as mock_call:
            mock_call.side_effect = Exception("MCP error")

            result = await search_mql5_book("test query")

            assert result["success"] is False
            assert "error" in result


class TestKnowledgeHubTools:
    """Test knowledge hub functions."""

    @pytest.mark.asyncio
    async def test_search_knowledge_hub_returns_structure(self):
        """search_knowledge_hub should return grouped results."""
        from src.agents.tools.knowledge.knowledge_hub import search_knowledge_hub

        with patch('src.agents.tools.knowledge.knowledge_hub.call_pageindex_tool') as mock_call:
            mock_call.return_value = {
                "results": [
                    {"namespace": "mql5_book", "page": 1, "content": "test", "relevance": 0.9}
                ]
            }

            result = await search_knowledge_hub("test query")

            assert result["success"] is True
            assert "results" in result
            assert "namespaces_searched" in result

    @pytest.mark.asyncio
    async def test_list_knowledge_namespaces_returns_defaults(self):
        """list_knowledge_namespaces should include default namespaces."""
        from src.agents.tools.knowledge.knowledge_hub import list_knowledge_namespaces

        with patch('src.agents.tools.knowledge.knowledge_hub.call_pageindex_tool') as mock_call:
            mock_call.return_value = {"namespaces": []}

            result = await list_knowledge_namespaces()

            assert result["success"] is True
            assert "namespaces" in result
            names = [ns["name"] for ns in result["namespaces"]]
            assert "mql5_book" in names
            assert "strategies" in names
            assert "knowledge" in names


class TestPDFIndexingTools:
    """Test PDF indexing functions."""

    @pytest.mark.asyncio
    async def test_index_pdf_validates_path(self):
        """index_pdf_document should validate file path."""
        from src.agents.tools.knowledge.pdf_indexing import index_pdf_document

        result = await index_pdf_document("/nonexistent/file.pdf", "test_ns")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_index_pdf_validates_pdf_extension(self):
        """index_pdf_document should validate PDF extension."""
        from src.agents.tools.knowledge.pdf_indexing import index_pdf_document
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            result = await index_pdf_document(temp_path, "test_ns")
            assert result["success"] is False
            assert "pdf" in result["error"].lower()
        finally:
            os.unlink(temp_path)


class TestStrategyTools:
    """Test strategy-specific tools."""

    @pytest.mark.asyncio
    async def test_search_strategy_patterns_returns_defaults_on_error(self):
        """search_strategy_patterns should return defaults on error."""
        from src.agents.tools.knowledge.strategies import search_strategy_patterns

        with patch('src.agents.tools.knowledge.strategies.call_pageindex_tool') as mock_call:
            mock_call.side_effect = Exception("Error")

            result = await search_strategy_patterns("entry")

            assert result["success"] is True
            assert "patterns" in result
            assert len(result["patterns"]) > 0

    @pytest.mark.asyncio
    async def test_get_indicator_template_builtin(self):
        """get_indicator_template should return built-in templates."""
        from src.agents.tools.knowledge.strategies import get_indicator_template

        result = await get_indicator_template("RSI")

        assert result["success"] is True
        assert result["name"] == "Relative Strength Index"
        assert "code" in result
        assert result["source"] == "builtin"

    @pytest.mark.asyncio
    async def test_get_indicator_template_not_found(self):
        """get_indicator_template should handle unknown indicators."""
        from src.agents.tools.knowledge.strategies import get_indicator_template

        result = await get_indicator_template("UnknownIndicatorXYZ")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestToolRegistry:
    """Test tool registry functions."""

    def test_get_knowledge_tool_returns_tool(self):
        """get_knowledge_tool should return tool definition."""
        from src.agents.tools.knowledge.registry import get_knowledge_tool

        tool = get_knowledge_tool("search_mql5_book")

        assert tool is not None
        assert "function" in tool
        assert "description" in tool
        assert "parameters" in tool

    def test_get_knowledge_tool_returns_none_for_unknown(self):
        """get_knowledge_tool should return None for unknown tool."""
        from src.agents.tools.knowledge.registry import get_knowledge_tool

        tool = get_knowledge_tool("nonexistent_tool")

        assert tool is None

    def test_list_knowledge_tools_returns_all_tools(self):
        """list_knowledge_tools should return all tool names."""
        from src.agents.tools.knowledge.registry import list_knowledge_tools

        tools = list_knowledge_tools()

        assert isinstance(tools, list)
        assert len(tools) >= 10
        assert "search_mql5_book" in tools
        assert "index_pdf_document" in tools

    @pytest.mark.asyncio
    async def test_invoke_knowledge_tool(self):
        """invoke_knowledge_tool should invoke correct function."""
        from src.agents.tools.knowledge.registry import invoke_knowledge_tool

        with patch('src.agents.tools.knowledge.mql5_book.call_pageindex_tool') as mock_call:
            mock_call.return_value = {"results": [], "total": 0}

            result = await invoke_knowledge_tool("search_mql5_book", query="test")

            assert result["success"] is True
            mock_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_knowledge_tool_raises_for_unknown(self):
        """invoke_knowledge_tool should raise for unknown tool."""
        from src.agents.tools.knowledge.registry import invoke_knowledge_tool

        with pytest.raises(ValueError, match="Unknown knowledge tool"):
            await invoke_knowledge_tool("nonexistent_tool")
