#!/usr/bin/env python3
"""
Task Group 8: Agent Integration and Deployment Trigger Tests

Tests for:
- MCP client connection management in BaseAgent
- Git storage of generated code with strategy metadata
- Paper trading deployment trigger for @primal agents
- Agent state metadata tracking
- Integration between QuantCodeAgent and MT5 MCP server

Run with: pytest tests/agents/test_task_group_8_integration.py -v
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from src.agents.core.base_agent import BaseAgent
from src.agents.implementations.quant_code import (
    QuantCodeAgent,
    CodeState,
    create_quant_code_agent
)
from src.agents.integrations.git_client import GitClient


class TestMCPClientConnection:
    """Test suite for MCP client connection management in BaseAgent."""

    @pytest.fixture
    def base_agent(self):
        """Create a BaseAgent instance for testing."""
        return BaseAgent(
            name="TestAgent",
            role="Test Role",
            model_name="gpt-4-turbo-preview",
            mcp_servers=[
                {
                    "name": "test-mcp-server",
                    "command": "echo",
                    "args": ["test"]
                }
            ]
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_connect_to_mcp_success(self, base_agent):
        """Test successful MCP server connection."""
        with patch('src.agents.core.base_agent.stdio_client') as mock_stdio:
            # Mock stdio client
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            mock_stdio.return_value = (None, None)

            with patch('src.agents.core.base_agent.ClientSession', return_value=mock_session):
                result = await base_agent.connect_to_mcp(
                    name="test-server",
                    command="echo",
                    args=["test"]
                )

                assert result is not None
                assert "test-server" in base_agent.mcp_clients
                assert base_agent.state_metadata["mcp_connections"]["test-server"]["status"] == "connected"
                mock_session.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_to_mcp_retry_logic(self, base_agent):
        """Test MCP connection retry logic with exponential backoff."""
        with patch('src.agents.core.base_agent.stdio_client') as mock_stdio:
            # Mock stdio client
            mock_session = AsyncMock()
            mock_session.initialize = AsyncMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()

            # First call fails, second succeeds
            call_count = [0]

            def side_effect(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] < 2:
                    raise Exception("Connection failed")
                return (None, None)

            mock_stdio.side_effect = side_effect

            with patch('src.agents.core.base_agent.ClientSession', return_value=mock_session):
                with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
                    result = await base_agent.connect_to_mcp(
                        name="test-server",
                        command="echo",
                        args=["test"],
                        max_retries=3,
                        initial_backoff=0.1
                    )

                    assert result is not None
                    assert call_count[0] == 2  # Failed once, succeeded on retry
                    assert mock_sleep.call_count == 1  # Slept once before retry

    @pytest.mark.asyncio
    async def test_call_mcp_tool_success(self, base_agent):
        """Test successful MCP tool call."""
        # Mock session
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(return_value={"result": "success"})
        base_agent.mcp_clients["test-server"] = mock_session

        result = await base_agent.call_mcp_tool(
            server_name="test-server",
            tool_name="test_tool",
            arguments={"param": "value"}
        )

        assert result is not None
        assert result["result"] == "success"
        mock_session.call_tool.assert_called_once_with("test_tool", {"param": "value"})

    @pytest.mark.asyncio
    async def test_call_mcp_tool_timeout(self, base_agent):
        """Test MCP tool call with timeout."""
        # Mock session that hangs
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=asyncio.TimeoutError())
        base_agent.mcp_clients["test-server"] = mock_session

        result = await base_agent.call_mcp_tool(
            server_name="test-server",
            tool_name="test_tool",
            arguments={},
            timeout=0.1
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_disconnect_mcp(self, base_agent):
        """Test MCP server disconnection."""
        mock_session = AsyncMock()
        mock_session.__aexit__ = AsyncMock()
        base_agent.mcp_clients["test-server"] = mock_session

        result = await base_agent.disconnect_mcp("test-server")

        assert result is True
        assert "test-server" not in base_agent.mcp_clients
        assert base_agent.state_metadata["mcp_connections"]["test-server"]["status"] == "disconnected"
        mock_session.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_mcp_tools(self, base_agent):
        """Test listing available MCP tools."""
        # Mock session with tools
        mock_session = AsyncMock()
        mock_tool = Mock()
        mock_tool.name = "test_tool"
        mock_tool.description = "A test tool"
        mock_tool.inputSchema = {"type": "object"}

        mock_response = Mock()
        mock_response.tools = [mock_tool]

        mock_session.list_tools = AsyncMock(return_value=mock_response)
        base_agent.mcp_clients["test-server"] = mock_session

        tools = await base_agent.get_mcp_tools("test-server")

        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["description"] == "A test tool"


class TestGitCodeStorage:
    """Test suite for Git storage of generated code."""

    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary Git repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)

        # Initialize Git repo
        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True)

        yield repo_path

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def quant_agent(self, temp_git_repo):
        """Create a QuantCodeAgent with temporary Git repo."""
        agent = QuantCodeAgent(
            name="TestQuantCode",
            role="Test Developer",
            git_repo_path=str(temp_git_repo),
            model_name="gpt-4-turbo-preview"
        )
        return agent

    @pytest.mark.asyncio
    async def test_store_code_in_git_success(self, quant_agent, temp_git_repo):
        """Test successful code storage in Git."""
        code = """
//+------------------------------------------------------------------+
//|                                          TestStrategy.mq5        |
//|                                    Generated by QuantMindX      |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property version   "1.00"

double backtrader = 1.0;
"""
        strategy_name = "RSI Mean Reversion"
        backtest_results = {
            "sharpe": 2.1,
            "drawdown": 8.5,
            "profit_factor": 1.9
        }

        result = await quant_agent.store_code_in_git(
            code=code,
            strategy_name=strategy_name,
            backtest_results=backtest_results
        )

        assert result is True

        # Verify file was created
        generated_bots_path = temp_git_repo / "templates" / "generated_bots"
        assert generated_bots_path.exists()

        files = list(generated_bots_path.glob("*.mq5"))
        assert len(files) > 0

        # Verify file content
        file_content = files[0].read_text()
        assert "backtrader" in file_content

        # Verify commit was made
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True
        )
        assert "RSI Mean Reversion" in result.stdout

    @pytest.mark.asyncio
    async def test_store_code_updates_deployment_history(self, quant_agent):
        """Test that storing code updates deployment history."""
        code = "double backtrader = 1.0;"
        strategy_name = "Test Strategy"
        backtest_results = {"sharpe": 1.5}

        await quant_agent.store_code_in_git(code, strategy_name, backtest_results)

        history = quant_agent.get_deployment_history()
        assert len(history) > 0
        assert history[0]["strategy_name"] == strategy_name
        assert "backtest_results" in history[0]

    @pytest.mark.asyncio
    async def test_store_code_sanitizes_filename(self, quant_agent, temp_git_repo):
        """Test that strategy names are sanitized for filenames."""
        code = "double backtrader = 1.0;"
        strategy_name = "Test/Special @#$% Strategy"  # Characters that need sanitizing
        backtest_results = {"sharpe": 1.5}

        await quant_agent.store_code_in_git(code, strategy_name, backtest_results)

        generated_bots_path = temp_git_repo / "templates" / "generated_bots"
        files = list(generated_bots_path.glob("*.mq5"))

        assert len(files) == 1
        # Filename should not contain special characters
        assert "/" not in files[0].name
        assert "@" not in files[0].name
        assert "#" not in files[0].name


class TestPaperTradingDeployment:
    """Test suite for paper trading deployment trigger."""

    @pytest.fixture
    def temp_git_repo(self):
        """Create a temporary Git repository."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)

        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True)

        yield repo_path

        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def quant_agent(self, temp_git_repo):
        """Create QuantCodeAgent for testing."""
        agent = QuantCodeAgent(
            name="TestQuantCode",
            role="Test Developer",
            git_repo_path=str(temp_git_repo)
        )
        return agent

    @pytest.mark.asyncio
    async def test_trigger_paper_trading_deployment_primal(self, quant_agent):
        """Test paper trading deployment trigger for @primal strategy."""
        strategy_name = "Alpha Trend Following"
        code = "double backtrader = 1.0;"
        backtest_results = {"sharpe": 2.5, "drawdown": 5.0}

        result = await quant_agent.trigger_paper_trading_deployment(
            strategy_name=strategy_name,
            code=code,
            backtest_results=backtest_results
        )

        assert result is True

        # Verify deployment was recorded
        history = quant_agent.get_deployment_history()
        assert len(history) > 0
        assert history[-1]["strategy_name"] == strategy_name
        assert history[-1]["status"] == "triggered"
        assert "deployment_config" in history[-1]

    @pytest.mark.asyncio
    async def test_trigger_deployment_includes_config(self, quant_agent):
        """Test that deployment includes custom configuration."""
        strategy_name = "Custom Strategy"
        code = "double backtrader = 1.0;"
        backtest_results = {"sharpe": 1.8}
        custom_config = {
            "risk_per_trade": 0.02,
            "max_positions": 5,
            "symbol": "EURUSD"
        }

        await quant_agent.trigger_paper_trading_deployment(
            strategy_name=strategy_name,
            code=code,
            backtest_results=backtest_results,
            deployment_config=custom_config
        )

        history = quant_agent.get_deployment_history()
        deployment_entry = history[-1]

        assert deployment_entry["deployment_config"]["risk_per_trade"] == 0.02
        assert deployment_entry["deployment_config"]["max_positions"] == 5
        assert deployment_entry["deployment_config"]["symbol"] == "EURUSD"

    @pytest.mark.asyncio
    async def test_deployment_failure_when_git_fails(self, quant_agent):
        """Test that deployment fails if Git storage fails."""
        # Patch Git client to fail
        with patch.object(quant_agent.git_client, 'write_template', return_value=False):
            result = await quant_agent.trigger_paper_trading_deployment(
                strategy_name="Test",
                code="code",
                backtest_results={}
            )

            assert result is False


class TestAgentStateMetadata:
    """Test suite for agent state metadata tracking."""

    @pytest.fixture
    def base_agent(self):
        """Create BaseAgent for testing."""
        return BaseAgent(
            name="TestAgent",
            role="Test Role",
            model_name="gpt-4-turbo-preview"
        )

    def test_initial_state_metadata(self, base_agent):
        """Test that agent initializes with proper metadata."""
        metadata = base_agent.get_state_metadata()

        assert "created_at" in metadata
        assert "last_activity" in metadata
        assert "invocation_count" in metadata
        assert "mcp_connections" in metadata
        assert "deployments_triggered" in metadata

    def test_update_state_metadata(self, base_agent):
        """Test updating agent state metadata."""
        base_agent.update_state_metadata("custom_key", "custom_value")
        metadata = base_agent.get_state_metadata()

        assert metadata["custom_key"] == "custom_value"

    def test_invocation_count_increments(self, base_agent):
        """Test that invocation count is tracked."""
        initial_count = base_agent.state_metadata["invocation_count"]

        # Simulate invocation
        base_agent.state_metadata["invocation_count"] += 1
        base_agent.state_metadata["last_activity"] = datetime.now().timestamp()

        metadata = base_agent.get_state_metadata()
        assert metadata["invocation_count"] == initial_count + 1


class TestQuantCodeAgentIntegration:
    """Integration tests for QuantCodeAgent with MCP and Git."""

    @pytest.fixture
    def temp_git_repo(self):
        """Create temporary Git repository."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)

        import subprocess
        subprocess.run(["git", "init"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, capture_output=True)

        yield repo_path

        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_create_quant_code_agent_factory(self, temp_git_repo):
        """Test QuantCodeAgent factory function."""
        agent = create_quant_code_agent(
            git_repo_path=str(temp_git_repo),
            model_name="gpt-4-turbo-preview",
            enable_mcp=False
        )

        assert isinstance(agent, QuantCodeAgent)
        assert agent.name == "QuantCode"
        assert agent.git_client.repo_path == temp_git_repo

    @pytest.mark.asyncio
    async def test_validation_node_triggers_deployment_for_primal(self, temp_git_repo):
        """Test that validation node triggers deployment for is_primal=True."""
        agent = QuantCodeAgent(
            name="TestQuantCode",
            role="Test",
            git_repo_path=str(temp_git_repo)
        )

        state = CodeState(
            messages=[],
            trd_content="Test TRD",
            current_code="import backtrader as bt\nclass MyStrategy(bt.Strategy):\n    pass",
            backtest_results=None,
            compilation_error="",
            retry_count=0,
            is_primal=True,
            strategy_name="Primal Strategy",
            deployment_config=None
        )

        result = await agent.validation_node(state)

        assert result["compilation_error"] is None
        assert "backtest_results" in result

        # Deployment should be triggered
        history = agent.get_deployment_history()
        assert len(history) > 0
        assert history[-1]["strategy_name"] == "Primal Strategy"

    @pytest.mark.asyncio
    async def test_validation_node_skips_deployment_for_non_primal(self, temp_git_repo):
        """Test that validation node skips deployment for is_primal=False."""
        agent = QuantCodeAgent(
            name="TestQuantCode",
            role="Test",
            git_repo_path=str(temp_git_repo)
        )

        state = CodeState(
            messages=[],
            trd_content="Test TRD",
            current_code="import backtrader as bt\nclass MyStrategy(bt.Strategy):\n    pass",
            backtest_results=None,
            compilation_error="",
            retry_count=0,
            is_primal=False,  # Not primal
            strategy_name="Regular Strategy",
            deployment_config=None
        )

        result = await agent.validation_node(state)

        assert result["compilation_error"] is None

        # Deployment should NOT be triggered
        history = agent.get_deployment_history()
        assert len(history) == 0


class TestMCPMT5Integration:
    """Test MCP server configuration for MT5 integration."""

    def test_mt5_mcp_config_in_factory(self):
        """Test that MT5 MCP server is configured in factory."""
        agent = create_quant_code_agent(enable_mcp=True)

        assert len(agent.mcp_servers_config) > 0

        # Check for MT5 server config
        mt5_config = next((c for c in agent.mcp_servers_config if c.get("name") == "mt5-server"), None)
        assert mt5_config is not None
        assert mt5_config["command"] == "python"

        # Check for backtest server config
        backtest_config = next((c for c in agent.mcp_servers_config if c.get("name") == "backtest-server"), None)
        assert backtest_config is not None

    def test_agent_has_mcp_connection_methods(self):
        """Test that agent has MCP connection methods."""
        agent = create_quant_code_agent()

        assert hasattr(agent, 'connect_to_mcp')
        assert hasattr(agent, 'call_mcp_tool')
        assert hasattr(agent, 'disconnect_mcp')
        assert hasattr(agent, 'get_mcp_tools')
        assert hasattr(agent, 'connect_all_mcp_servers')
