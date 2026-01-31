"""
End-to-End Integration Test Configuration

Task Group 15: E2E Integration Testing for QuantMindX Architecture

This conftest.py provides fixtures and helpers for comprehensive E2E tests
covering the full workflow: TRD -> Generate -> Backtest -> Deploy -> Monitor
"""

import os
import sys
import json
import asyncio
import pytest
import logging
from pathlib import Path
from typing import Dict, Any, Generator, Optional
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class E2ETestHelper:
    """Helper class for E2E test utilities."""

    @staticmethod
    def create_sample_trd() -> str:
        """Create a sample Technical Requirements Document (TRD)."""
        return """
# Technical Requirements Document: Moving Average Crossover Strategy

## Strategy Overview
Implement a moving average crossover trading strategy that:
- Uses Fast MA (10-period) and Slow MA (20-period)
- Executes buy when Fast MA crosses above Slow MA
- Executes sell when Fast MA crosses below Slow MA
- Includes stop loss at 50 pips and take profit at 100 pips

## Technical Specifications
- Language: Python with Backtrader
- Indicators: Simple Moving Average (SMA)
- Risk Management: Fixed lot size 0.1, 2% risk per trade
- Trading Hours: 08:00-17:00 GMT
- Filter: Avoid trading during high volatility news events

## Performance Requirements
- Target Sharpe Ratio: > 1.5
- Maximum Drawdown: < 20%
- Win Rate: > 45%
- Profit Factor: > 1.5

## Input Parameters
- Symbol: EURUSD
- Timeframe: H1 (60 minutes)
- Backtest Period: 2024-01-01 to 2024-12-31
- Initial Capital: $10,000
"""

    @staticmethod
    def create_sample_strategy_code() -> str:
        """Create sample Python strategy code for testing."""
        return """
import backtrader as bt

class MovingAverageCross(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 20),
        ('stop_loss_pips', 50),
        ('take_profit_pips', 100),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy(size=0.1)
        else:
            if self.crossover < 0:
                self.sell(size=self.position.size)
"""

    @staticmethod
    def create_mock_backtest_result(sharpe_ratio: float = 1.8,
                                    max_drawdown: float = 15.0,
                                    win_rate: float = 52.0) -> Dict[str, Any]:
        """Create mock backtest result with configurable metrics."""
        return {
            "backtest_id": "test-bt-001",
            "status": "success",
            "metrics": {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_return": 25.3,
                "win_rate": win_rate,
                "profit_factor": 1.8
            },
            "equity_curve": [
                {"timestamp": "2024-01-01T00:00:00Z", "value": 10000.0},
                {"timestamp": "2024-01-02T00:00:00Z", "value": 10050.0},
                {"timestamp": "2024-01-03T00:00:00Z", "value": 10120.0},
            ],
            "trade_log": [
                {
                    "entry": "2024-01-02T10:00:00Z",
                    "exit": "2024-01-03T14:00:00Z",
                    "pnl": 120.0
                }
            ],
            "logs": "Backtest completed successfully",
            "execution_time_seconds": 45.2
        }

    @staticmethod
    def create_deployment_config() -> Dict[str, Any]:
        """Create sample paper trading deployment configuration."""
        return {
            "strategy_name": "MA_Crossover_Test",
            "symbol": "EURUSD",
            "timeframe": "H1",
            "initial_capital": 10000.0,
            "mt5_credentials": {
                "account": "12345678",
                "password": "demo_password",
                "server": "MetaQuotes-Demo"
            },
            "container_config": {
                "image": "quantmindx:strategy-agent",
                "cpu_limit": 0.5,
                "memory_limit": "512MB"
            }
        }


@pytest.fixture
def e2e_helper():
    """Provide E2E test helper instance."""
    return E2ETestHelper()


@pytest.fixture
def sample_trd(e2e_helper):
    """Provide sample TRD content."""
    return e2e_helper.create_sample_trd()


@pytest.fixture
def sample_strategy_code(e2e_helper):
    """Provide sample strategy code."""
    return e2e_helper.create_sample_strategy_code()


@pytest.fixture
def mock_backtest_result(e2e_helper):
    """Provide mock backtest result."""
    return e2e_helper.create_mock_backtest_result()


@pytest.fixture
def deployment_config(e2e_helper):
    """Provide deployment configuration."""
    return e2e_helper.create_deployment_config()


@pytest.fixture
def temp_git_repo():
    """Create temporary Git repository for testing."""
    temp_dir = tempfile.mkdtemp(prefix="e2e_git_repo_")
    repo_path = Path(temp_dir)

    # Initialize Git repository
    import subprocess
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "E2E Test"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "e2e@test.com"], cwd=repo_path, check=True, capture_output=True)

    # Create directory structure
    (repo_path / "generated_bots").mkdir(parents=True, exist_ok=True)
    (repo_path / "templates").mkdir(parents=True, exist_ok=True)
    (repo_path / "skills").mkdir(parents=True, exist_ok=True)

    yield repo_path

    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for Pub/Sub testing."""
    client = MagicMock()
    client.publish = AsyncMock(return_value=1)
    client.subscribe = MagicMock()
    client.unsubscribe = MagicMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_docker_client():
    """Mock Docker client for container testing."""
    client = MagicMock()
    client.containers = MagicMock()
    client.images = MagicMock()

    # Mock container creation
    mock_container = MagicMock()
    mock_container.id = "container-123"
    mock_container.name = "agent-ma-crossover-test"
    mock_container.status = "running"
    mock_container.logs = MagicMock(return_value=b"Container started\nHeartbeat published\n")
    mock_container.stop = MagicMock()
    mock_container.remove = MagicMock()
    client.containers.create = MagicMock(return_value=mock_container)
    client.containers.get = MagicMock(return_value=mock_container)
    client.containers.list = MagicMock(return_value=[mock_container])

    return client


@pytest.fixture
def mock_mcp_servers():
    """Mock MCP server connections for testing."""
    mock_servers = {
        "quantmindx-kb": MagicMock(),
        "backtest-server": MagicMock(),
        "mt5-server": MagicMock()
    }

    # Mock quantmindx-kb tools
    async def mock_get_coding_standards(**kwargs):
        return {
            "project_name": "QuantMindX",
            "version": "1.0",
            "conventions": ["Python 3.10+", "Type Hints", "Docstrings"],
            "file_structure": "src/, tests/, docs/"
        }

    async def mock_get_bad_patterns(**kwargs):
        return {
            "patterns": [
                {"pattern": "Global mutable state", "severity": "high", "solution": "Use dependency injection"},
                {"pattern": "Missing error handling", "severity": "medium", "solution": "Add try/except blocks"}
            ]
        }

    async def mock_get_algorithm_template(**kwargs):
        return {
            "name": "Moving Average Crossover",
            "category": "trading",
            "description": "Classic MA crossover strategy",
            "code": "class MAStrategy: ..."
        }

    async def call_quantmindx_kb_tool(tool, args):
        if tool == "get_coding_standards":
            return await mock_get_coding_standards(**args)
        elif tool == "get_bad_patterns":
            return await mock_get_bad_patterns(**args)
        elif tool == "get_algorithm_template":
            return await mock_get_algorithm_template(**args)
        return None

    mock_servers["quantmindx-kb"].call_tool = AsyncMock(side_effect=call_quantmindx_kb_tool)

    # Mock backtest-server tools
    async def mock_run_backtest(**kwargs):
        return json.dumps({"backtest_id": "bt-001", "status": "queued"})

    async def mock_get_backtest_status(**kwargs):
        return json.dumps({
            "backtest_id": "bt-001",
            "status": "completed",
            "progress_percent": 100.0,
            "result": E2ETestHelper.create_mock_backtest_result()
        })

    async def call_backtest_tool(tool, args):
        if tool == "run_backtest":
            return await mock_run_backtest(**args)
        elif tool == "get_backtest_status":
            return await mock_get_backtest_status(**args)
        return None

    mock_servers["backtest-server"].call_tool = AsyncMock(side_effect=call_backtest_tool)

    # Mock mt5-server tools
    async def mock_deploy_paper_agent(**kwargs):
        return json.dumps({
            "agent_id": "agent-001",
            "container_id": "container-123",
            "status": "running",
            "redis_channel": "agent:heartbeat:agent-001",
            "logs_url": "/logs/agent-001"
        })

    async def mock_list_paper_agents(**kwargs):
        return json.dumps([{
            "agent_id": "agent-001",
            "strategy_name": "MA_Crossover",
            "status": "running",
            "uptime": 3600
        }])

    async def call_mt5_tool(tool, args):
        if tool == "deploy_paper_agent":
            return await mock_deploy_paper_agent(**args)
        elif tool == "list_paper_agents":
            return await mock_list_paper_agents(**args)
        return None

    mock_servers["mt5-server"].call_tool = AsyncMock(side_effect=call_mt5_tool)

    return mock_servers


@pytest.fixture
def mock_agent_state():
    """Provide initial agent state for testing."""
    return {
        "messages": [],
        "trd_content": "",
        "current_code": "",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "",
        "deployment_config": None
    }


@pytest.fixture
def quant_code_agent(temp_git_repo, mock_mcp_servers):
    """Create QuantCode agent instance for E2E testing."""
    from src.agents.implementations.quant_code import QuantCodeAgent
    from langchain_core.messages import AIMessage

    # Mock LLM response
    mock_llm_response = AIMessage(content="""
Based on the TRD for a Moving Average Crossover strategy, here is the implementation plan:

## Implementation Plan

### File Structure:
- strategy_ma_crossover.py - Main strategy class
- indicators.py - SMA indicator implementation
- risk_management.py - Stop loss and take profit logic

### Key Components:
1. **Strategy Class**: MovingAverageCross
   - Parameters: fast_period=10, slow_period=20
   - Signals: Crossover detection
   - Entry/Exit logic

2. **Indicators**:
   - Simple Moving Average (SMA)
   - Crossover detection

3. **Risk Management**:
   - Stop loss: 50 pips
   - Take profit: 100 pips
   - Fixed lot size: 0.1

### Code Generation:
Following Python 3.10+ standards with type hints.
Avoiding global mutable state patterns.
Comprehensive error handling with try/except blocks.

## Generated Code

```python
import backtrader as bt

class MovingAverageCross(bt.Strategy):
    params = (
        ('fast_period', 10),
        ('slow_period', 20),
        ('stop_loss_pips', 50),
        ('take_profit_pips', 100),
    )

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy(size=0.1)
        else:
            if self.crossover < 0:
                self.sell(size=self.position.size)
```
""")

    # Patch MCP connections
    with patch('src.agents.core.base_agent.BaseAgent.connect_to_mcp'):

        # Create agent
        agent = QuantCodeAgent(
            git_repo_path=str(temp_git_repo),
            model_name="gpt-4",
            mcp_servers=[]
        )

        # Mock the LLM
        agent.llm = MagicMock()
        agent.llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        # Inject mock MCP servers
        agent.mcp_clients = mock_mcp_servers

        # Override call_mcp_tool to properly call mock servers
        async def mock_call_mcp_tool(server_name, tool_name, args):
            if server_name in mock_mcp_servers:
                server = mock_mcp_servers[server_name]
                if hasattr(server, 'call_tool'):
                    result = await server.call_tool(tool_name, args)
                    return result
            return None

        agent.call_mcp_tool = mock_call_mcp_tool

        yield agent


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def set_test_environment(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-mock-key-for-testing")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-mock-key-for-testing")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-mock-key-for-testing")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
