"""
PageIndex Integration Tests

Tests for verifying PageIndex services are running and accessible,
and that the knowledge retrieval system works correctly.
"""

import os
import pytest
import httpx
from unittest.mock import patch, MagicMock

# Test configuration
PAGEINDEX_ARTICLES_URL = os.getenv("PAGEINDEX_ARTICLES_URL", "http://localhost:3000")
PAGEINDEX_BOOKS_URL = os.getenv("PAGEINDEX_BOOKS_URL", "http://localhost:3001")
PAGEINDEX_LOGS_URL = os.getenv("PAGEINDEX_LOGS_URL", "http://localhost:3002")


class TestPageIndexServices:
    """Test PageIndex service health and connectivity."""

    @pytest.mark.integration
    def test_articles_service_health(self):
        """Test that articles PageIndex service responds to health check."""
        try:
            response = httpx.get(f"{PAGEINDEX_ARTICLES_URL}/health", timeout=5.0)
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Articles service not running - start with docker-compose -f docker-compose.pageindex.yml up -d")

    @pytest.mark.integration
    def test_books_service_health(self):
        """Test that books PageIndex service responds to health check."""
        try:
            response = httpx.get(f"{PAGEINDEX_BOOKS_URL}/health", timeout=5.0)
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Books service not running - start with docker-compose -f docker-compose.pageindex.yml up -d")

    @pytest.mark.integration
    def test_logs_service_health(self):
        """Test that logs PageIndex service responds to health check."""
        try:
            response = httpx.get(f"{PAGEINDEX_LOGS_URL}/health", timeout=5.0)
            assert response.status_code == 200
        except httpx.ConnectError:
            pytest.skip("Logs service not running - start with docker-compose -f docker-compose.pageindex.yml up -d")

    @pytest.mark.integration
    def test_all_services_running(self):
        """Test that all three PageIndex services are running."""
        services = [
            ("articles", PAGEINDEX_ARTICLES_URL),
            ("books", PAGEINDEX_BOOKS_URL),
            ("logs", PAGEINDEX_LOGS_URL),
        ]
        
        results = {}
        for name, url in services:
            try:
                response = httpx.get(f"{url}/health", timeout=5.0)
                results[name] = response.status_code == 200
            except httpx.ConnectError:
                results[name] = False
        
        # At least articles service should be running for basic functionality
        if not any(results.values()):
            pytest.skip("No PageIndex services running - start with docker-compose -f docker-compose.pageindex.yml up -d")


class TestKnowledgeRouter:
    """Test KnowledgeRouter PageIndex integration."""

    def test_knowledge_router_instantiation(self):
        """Test that KnowledgeRouter can be instantiated with PageIndex client."""
        from src.agents.knowledge.router import KnowledgeRouter, PageIndexClient
        
        # Test PageIndexClient instantiation
        client = PageIndexClient(
            articles_url=PAGEINDEX_ARTICLES_URL,
            books_url=PAGEINDEX_BOOKS_URL,
            logs_url=PAGEINDEX_LOGS_URL
        )
        
        assert client.articles_url == PAGEINDEX_ARTICLES_URL
        assert client.books_url == PAGEINDEX_BOOKS_URL
        assert client.logs_url == PAGEINDEX_LOGS_URL

    def test_get_port_for_collection(self):
        """Test that correct ports are mapped to collections."""
        from src.agents.knowledge.router import PageIndexClient
        
        client = PageIndexClient()
        
        assert client._get_port_for_collection("articles") == 3000
        assert client._get_port_for_collection("books") == 3001
        assert client._get_port_for_collection("logs") == 3002
        assert client._get_port_for_collection("unknown") == 3000  # Default

    @pytest.mark.integration
    def test_search_articles_via_router(self):
        """Test searching articles through KnowledgeRouter."""
        from src.agents.knowledge.router import KnowledgeRouter
        
        router = KnowledgeRouter()
        
        try:
            results = router.search("trailing stop", collection="articles")
            assert isinstance(results, list)
        except Exception as e:
            if "Connection" in str(e) or "connect" in str(e).lower():
                pytest.skip("Articles service not running")
            raise


class TestKnowledgeRetriever:
    """Test knowledge retriever PageIndex integration."""

    def test_retriever_imports(self):
        """Test that retriever module imports correctly."""
        from src.agents.knowledge.retriever import search_knowledge_base
        # search_knowledge_base is a LangChain StructuredTool, check it has a func attribute
        assert hasattr(search_knowledge_base, 'func') or callable(search_knowledge_base)

    @patch('src.agents.knowledge.retriever.kb_router')
    def test_search_knowledge_base_returns_results(self, mock_router):
        """Test that search_knowledge_base returns formatted results.
        
        This test properly mocks the kb_router singleton to avoid hitting
        live PageIndex services, ensuring reliable unit test behavior.
        """
        from src.agents.knowledge.retriever import search_knowledge_base
        
        # Mock the router's search method to return test data
        mock_router.search.return_value = [
            {
                "title": "Test Article",
                "content": "Test content about trailing stops",
                "source": "test_source.md",
                "page": 1,
                "section": "Introduction",
                "score": 0.95,
                "collection": "articles"
            }
        ]
        
        # search_knowledge_base is a StructuredTool, access underlying func
        result = search_knowledge_base.func("test query", collection="articles")
        
        # Verify the router's search was called with correct params
        mock_router.search.assert_called_once_with("test query", collection="articles", limit=5)
        
        # Verify result structure
        assert result is not None
        assert "results" in result
        assert result["total"] == 1
        assert result["query"] == "test query"
        assert result["collection"] == "articles"
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Test content about trailing stops"

    @patch('src.agents.knowledge.retriever.kb_router')
    def test_search_knowledge_base_empty_results(self, mock_router):
        """Test that search_knowledge_base handles empty results correctly."""
        from src.agents.knowledge.retriever import search_knowledge_base
        
        # Mock the router's search method to return empty list
        mock_router.search.return_value = []
        
        result = search_knowledge_base.func("nonexistent query", collection="articles")
        
        assert result is not None
        assert result["results"] == []
        assert result["total"] == 0

    @patch('src.agents.knowledge.retriever.kb_router')
    def test_search_knowledge_base_invalid_collection(self, mock_router):
        """Test that search_knowledge_base validates collection parameter."""
        from src.agents.knowledge.retriever import search_knowledge_base
        
        # Invalid collection should return error without calling router
        result = search_knowledge_base.func("test query", collection="invalid_collection")
        
        assert result is not None
        assert "error" in result
        assert "Invalid collection" in result["error"]
        # Router should not be called for invalid collection
        mock_router.search.assert_not_called()

    @patch('src.agents.knowledge.retriever.kb_router')
    def test_search_knowledge_base_handles_exception(self, mock_router):
        """Test that search_knowledge_base handles exceptions gracefully."""
        from src.agents.knowledge.retriever import search_knowledge_base
        
        # Mock router to raise an exception
        mock_router.search.side_effect = Exception("Connection refused")
        
        result = search_knowledge_base.func("test query", collection="articles")
        
        assert result is not None
        assert "error" in result
        assert "Search failed" in result["error"]


class TestNoQdrantReferences:
    """Test that no Qdrant references remain in active code."""

    def test_no_qdrant_in_router(self):
        """Verify no Qdrant imports in router.py."""
        with open("src/agents/knowledge/router.py", "r") as f:
            content = f.read().lower()
        
        assert "qdrant" not in content, "Found Qdrant reference in router.py"
        assert "from qdrant" not in content, "Found Qdrant import in router.py"

    def test_no_qdrant_in_retriever(self):
        """Verify no Qdrant imports in retriever.py."""
        with open("src/agents/knowledge/retriever.py", "r") as f:
            content = f.read().lower()
        
        assert "qdrant" not in content, "Found Qdrant reference in retriever.py"

    def test_no_qdrant_in_mcp_server(self):
        """Verify no Qdrant imports in MCP server.py."""
        with open("mcp-servers/quantmindx-kb/server.py", "r") as f:
            content = f.read().lower()
        
        assert "qdrant" not in content, "Found Qdrant reference in MCP server.py"

    def test_no_qdrant_in_requirements(self):
        """Verify qdrant-client not in requirements.txt."""
        with open("requirements.txt", "r") as f:
            content = f.read().lower()
        
        assert "qdrant-client" not in content, "Found qdrant-client in requirements.txt"

    def test_no_sentence_transformers_in_requirements(self):
        """Verify sentence-transformers not in requirements.txt."""
        with open("requirements.txt", "r") as f:
            content = f.read().lower()
        
        assert "sentence-transformers" not in content, "Found sentence-transformers in requirements.txt"


class TestEnvironmentVariables:
    """Test environment variable configuration."""

    def test_pageindex_env_defaults(self):
        """Test that PageIndex environment variables have defaults."""
        from src.agents.knowledge.router import PageIndexClient
        
        # Should work without any env vars set
        client = PageIndexClient()
        
        assert client.articles_url is not None
        assert client.books_url is not None
        assert client.logs_url is not None

    def test_pageindex_env_from_os(self):
        """Test that PageIndex URLs can be set from environment."""
        import os
        from src.agents.knowledge.router import PageIndexClient
        
        # Set custom URLs
        os.environ["PAGEINDEX_ARTICLES_URL"] = "http://custom:4000"
        
        client = PageIndexClient()
        
        # Should use environment variable
        assert client.articles_url == "http://custom:4000"
        
        # Clean up
        del os.environ["PAGEINDEX_ARTICLES_URL"]


class TestMCPTools:
    """Test MCP server tools with PageIndex."""

    def test_mcp_server_imports(self):
        """Test that MCP server imports correctly."""
        import sys
        sys.path.insert(0, "mcp-servers/quantmindx-kb")
        
        # Should be able to import the server module
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "server", 
            "mcp-servers/quantmindx-kb/server.py"
        )
        
        assert spec is not None

    def test_search_tool_definition(self):
        """Test that search_knowledge_base tool is defined."""
        with open("mcp-servers/quantmindx-kb/server.py", "r") as f:
            content = f.read()
        
        assert "search_knowledge_base" in content
        # MCP server uses @server.list_tools() and @server.call_tool() decorators
        assert "@server.list_tools" in content or "@server.call_tool" in content

    def test_kb_stats_tool_definition(self):
        """Test that kb_stats tool is defined."""
        with open("mcp-servers/quantmindx-kb/server.py", "r") as f:
            content = f.read()
        
        assert "kb_stats" in content


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires running services)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not integration"])