"""
Tests for MCP Tools Module Structure.

This test validates that the modular MCP tools structure:
1. Maintains backward compatibility with the original mcp_tools.py
2. Provides proper imports from the mcp/ subpackage
3. Exposes all expected tool functions
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from typing import Dict, Any


class TestMCPModuleStructure:
    """Tests for the modular MCP tools structure."""

    def test_import_mcp_package(self):
        """Test that the mcp subpackage can be imported."""
        from src.agents.tools.mcp import (
            MCPClientManager,
            get_mql5_documentation,
            get_mql5_examples,
            sequential_thinking,
            analyze_errors,
            index_pdf,
            search_pdf,
            get_indexed_documents,
            run_backtest,
            get_backtest_status,
            get_backtest_results,
            compare_backtests,
            compile_mql5_code,
            validate_mql5_syntax,
            get_compilation_errors,
        )
        # All imports should succeed
        assert MCPClientManager is not None

    def test_import_from_main_module(self):
        """Test backward compatibility - imports from original module path."""
        from src.agents.tools.mcp_tools import (
            MCPClientManager,
            get_mql5_documentation,
            get_mql5_examples,
            sequential_thinking,
            analyze_errors,
            index_pdf,
            search_pdf,
            get_indexed_documents,
            run_backtest,
            get_backtest_status,
            get_backtest_results,
            compare_backtests,
            compile_mql5_code,
            validate_mql5_syntax,
            get_compilation_errors,
            get_mcp_tool,
            list_mcp_tools,
            invoke_mcp_tool,
        )
        # All imports should succeed
        assert MCPClientManager is not None


class TestMCPManager:
    """Tests for MCPClientManager class."""

    def test_manager_initialization(self):
        """Test MCPClientManager can be initialized."""
        from src.agents.tools.mcp.manager import MCPClientManager

        manager = MCPClientManager()
        assert manager is not None
        assert hasattr(manager, 'initialize')
        assert hasattr(manager, 'connect_server')
        assert hasattr(manager, 'call_tool')
        assert hasattr(manager, 'list_tools')
        assert hasattr(manager, 'get_status')
        assert hasattr(manager, 'close')

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.manager.MCP_CLIENT_AVAILABLE', False)
    async def test_manager_initialize(self):
        """Test MCPClientManager.initialize() method."""
        from src.agents.tools.mcp.manager import MCPClientManager

        manager = MCPClientManager(configs={})
        await manager.initialize()

        assert manager._initialized is True


class TestContext7Tools:
    """Tests for Context7 (MQL5 documentation) tools."""

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.context7.get_mcp_manager')
    async def test_get_mql5_documentation(self, mock_get_manager):
        """Test get_mql5_documentation function."""
        from src.agents.tools.mcp.context7 import get_mql5_documentation

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "results": [{"text": "MQL5 documentation content"}],
            "total": 1
        })
        mock_get_manager.return_value = mock_manager

        result = await get_mql5_documentation("iMA")

        assert "results" in result
        mock_manager.call_tool.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.context7.get_mcp_manager')
    async def test_get_mql5_examples(self, mock_get_manager):
        """Test get_mql5_examples function."""
        from src.agents.tools.mcp.context7 import get_mql5_examples

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "examples": [{"code": "Example code"}]
        })
        mock_get_manager.return_value = mock_manager

        result = await get_mql5_examples("Moving Average")

        assert "examples" in result
        mock_manager.call_tool.assert_called_once()


class TestSequentialThinkingTools:
    """Tests for Sequential Thinking tools."""

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.sequential_thinking.get_mcp_manager')
    async def test_sequential_thinking(self, mock_get_manager):
        """Test sequential_thinking function."""
        from src.agents.tools.mcp.sequential_thinking import sequential_thinking

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "steps": ["Step 1", "Step 2"],
            "reasoning": "Thinking..."
        })
        mock_get_manager.return_value = mock_manager

        result = await sequential_thinking("How to implement a trading bot?")

        assert "steps" in result
        mock_manager.call_tool.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.sequential_thinking.get_mcp_manager')
    async def test_analyze_errors(self, mock_get_manager):
        """Test analyze_errors function."""
        from src.agents.tools.mcp.sequential_thinking import analyze_errors

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "analysis": ["Fix 1", "Fix 2"]
        })
        mock_get_manager.return_value = mock_manager

        result = await analyze_errors(["Error code 4756"])

        assert "analysis" in result


class TestPageIndexTools:
    """Tests for PageIndex (PDF) tools."""

    @pytest.mark.asyncio
    @patch('pathlib.Path.exists', return_value=True)
    @patch('src.agents.tools.mcp.page_index.get_mcp_manager')
    async def test_index_pdf(self, mock_get_manager, mock_exists):
        """Test index_pdf function."""
        from src.agents.tools.mcp.page_index import index_pdf

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "job_id": "doc_123",
            "status": "completed"
        })
        mock_get_manager.return_value = mock_manager

        result = await index_pdf("/path/to/document.pdf", "my_namespace")

        assert "job_id" in result
        mock_manager.call_tool.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.page_index.get_mcp_manager')
    async def test_search_pdf(self, mock_get_manager):
        """Test search_pdf function."""
        from src.agents.tools.mcp.page_index import search_pdf

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "results": [{"page": 1, "text": "Found content"}],
            "total": 1
        })
        mock_get_manager.return_value = mock_manager

        result = await search_pdf("trading strategy", "my_namespace")

        assert "results" in result


class TestBacktestTools:
    """Tests for Backtest tools."""

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.backtest.get_mcp_manager')
    async def test_run_backtest(self, mock_get_manager):
        """Test run_backtest function."""
        from src.agents.tools.mcp.backtest import run_backtest

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "backtest_id": "bt_123",
            "status": "queued"
        })
        mock_get_manager.return_value = mock_manager

        result = await run_backtest("MyStrategy", {})

        assert "backtest_id" in result
        mock_manager.call_tool.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.backtest.get_mcp_manager')
    async def test_get_backtest_status(self, mock_get_manager):
        """Test get_backtest_status function."""
        from src.agents.tools.mcp.backtest import get_backtest_status

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "status": "completed",
            "progress": 100
        })
        mock_get_manager.return_value = mock_manager

        result = await get_backtest_status("bt_123")

        assert "status" in result


class TestMT5CompilerTools:
    """Tests for MT5 Compiler tools."""

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.mt5_compiler.get_mcp_manager')
    async def test_compile_mql5_code(self, mock_get_manager):
        """Test compile_mql5_code function."""
        from src.agents.tools.mcp.mt5_compiler import compile_mql5_code

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "success": True,
            "errors": []
        })
        mock_get_manager.return_value = mock_manager

        result = await compile_mql5_code("//+MQL5 code", "test_expert")

        assert "success" in result
        mock_manager.call_tool.assert_called_once()

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.mt5_compiler.get_mcp_manager')
    async def test_validate_mql5_syntax(self, mock_get_manager):
        """Test validate_mql5_syntax function."""
        from src.agents.tools.mcp.mt5_compiler import validate_mql5_syntax

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "valid": True,
            "errors": []
        })
        mock_get_manager.return_value = mock_manager

        result = await validate_mql5_syntax("//+MQL5 code")

        assert "valid" in result


class TestToolDiscovery:
    """Tests for tool discovery functions."""

    def test_list_mcp_tools(self):
        """Test list_mcp_tools returns expected tool names."""
        from src.agents.tools.mcp_tools import list_mcp_tools

        tools = list_mcp_tools()

        # Should return a list
        assert isinstance(tools, list)

    def test_get_mcp_tool(self):
        """Test get_mcp_tool returns tool definition."""
        from src.agents.tools.mcp_tools import get_mcp_tool

        # Should return None or a dict
        result = get_mcp_tool("nonexistent_tool")
        assert result is None or isinstance(result, dict)

    @pytest.mark.asyncio
    @patch('src.agents.tools.mcp.context7.get_mcp_manager')
    async def test_invoke_mcp_tool(self, mock_get_manager):
        """Test invoke_mcp_tool function."""
        from src.agents.tools.mcp_tools import invoke_mcp_tool

        mock_manager = MagicMock()
        mock_manager.call_tool = AsyncMock(return_value={
            "results": [{"text": "doc content"}],
            "total": 1
        })
        mock_get_manager.return_value = mock_manager

        result = await invoke_mcp_tool("get_mql5_documentation", query="iMA")

        # Should return the results from the tool
        assert "results" in result
