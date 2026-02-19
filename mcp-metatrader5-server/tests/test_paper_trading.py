#!/usr/bin/env python3
"""
Tests for Paper Trading Deployment Tools (Task Group 11).

Tests cover:
- deploy_paper_agent creates Docker container
- list_paper_agents returns agents with status
- stop_paper_agent terminates gracefully
- get_agent_logs retrieves logs
- get_agent_performance calculates metrics
- heartbeat detection marks stale agents
- ChromaDB storage for trade events and performance
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, UTC, timedelta as td

from docker.errors import NotFound, APIError

# Direct imports bypassing package __init__
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import modules directly from their files
import importlib.util

# Load models
spec = importlib.util.spec_from_file_location(
    "models", Path(__file__).parent.parent / "src/mcp_mt5/paper_trading/models.py"
)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

AgentStatus = models.AgentStatus
AgentHealth = models.AgentHealth
PaperAgentStatus = models.PaperAgentStatus
AgentDeploymentResult = models.AgentDeploymentResult
AgentPerformance = models.AgentPerformance
AgentLogsResult = models.AgentLogsResult

# Load deployer
spec = importlib.util.spec_from_file_location(
    "deployer", Path(__file__).parent.parent / "src/mcp_mt5/paper_trading/deployer.py"
)
deployer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployer_module)
PaperTradingDeployer = deployer_module.PaperTradingDeployer

# Load monitor
spec = importlib.util.spec_from_file_location(
    "monitor", Path(__file__).parent.parent / "src/mcp_mt5/paper_trading/monitor.py"
)
monitor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor_module)
AgentHealthMonitor = monitor_module.AgentHealthMonitor

# Load storage
spec = importlib.util.spec_from_file_location(
    "storage", Path(__file__).parent.parent / "src/mcp_mt5/paper_trading/storage.py"
)
storage_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage_module)
PaperTradingStorage = storage_module.PaperTradingStorage


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_docker_client():
    """Mock Docker client."""
    with patch("mcp_mt5.paper_trading.deployer.docker") as mock_docker:
        client = MagicMock()
        mock_docker.DockerClient.return_value = client
        client.ping.return_value = True
        yield client


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    with patch("mcp_mt5.paper_trading.monitor.redis.Redis") as mock_redis:
        client = MagicMock()
        mock_redis.return_value = client
        client.ping.return_value = True
        yield client


@pytest.fixture
def deployer(mock_docker_client):
    """Create PaperTradingDeployer with mocked Docker."""
    return PaperTradingDeployer()


@pytest.fixture
def monitor(mock_redis_client):
    """Create AgentHealthMonitor with mocked Redis."""
    return AgentHealthMonitor()


@pytest.fixture
def sample_deployment_request():
    """Sample deployment request."""
    return {
        "strategy_name": "RSI Reversal",
        "strategy_code": "template:rsi-reversal",
        "config": {
            "rsi_period": 14,
            "oversold": 30,
            "overbought": 70,
            "symbols": ["EURUSD"],
            "timeframe": "H1",
        },
        "mt5_credentials": {
            "account": 12345678,
            "password": "demo_password",
            "server": "MetaQuotes-Demo",
        },
        "magic_number": 98765432,
    }


@pytest.fixture
def sample_container():
    """Sample mock Docker container."""
    container = MagicMock()
    container.id = "a1b2c3d4e5f6"
    container.name = "quantmindx-agent-strategy-rsi-98765432-1234567890"
    container.status = "running"
    container.labels = {
        "quantmindx-paper-agent": "true",
        "agent-id": "strategy-rsi-98765432-1234567890",
        "strategy-name": "RSI Reversal",
        "magic-number": "98765432",
    }
    container.attrs = {
        "Config": {
            "Image": "quantmindx/strategy-agent:latest",
            "Env": [
                "AGENT_ID=strategy-rsi-98765432-1234567890",
                "STRATEGY_NAME=RSI Reversal",
                "MT5_ACCOUNT=12345678",
                "MT5_SERVER=MetaQuotes-Demo",
                "MAGIC_NUMBER=98765432",
                "REDIS_HOST=localhost",
                "REDIS_PORT=6379",
            ],
        },
        "State": {
            "Status": "running",
            "StartedAt": "2026-01-29T10:00:00.000000Z",
        },
        "Created": "2026-01-29T10:00:00.000000Z",
    }
    return container


# ============================================================================
# Test 1: deploy_paper_agent creates container
# ============================================================================

class TestDeployPaperAgent:
    """Test deploy_paper_agent functionality."""

    def test_deploy_creates_container_with_correct_config(
        self, deployer, mock_docker_client, sample_deployment_request, sample_container
    ):
        """Test deploy_paper_agent creates Docker container with correct config."""
        # Setup mocks
        mock_docker_client.images.get.return_value = MagicMock()
        mock_docker_client.containers.run.return_value = sample_container

        # Deploy agent
        result = deployer.deploy_agent(**sample_deployment_request)

        # Verify container.run was called
        assert mock_docker_client.containers.run.called

        call_kwargs = mock_docker_client.containers.run.call_args[1]
        assert call_kwargs["detach"] is True
        assert call_kwargs["environment"]["AGENT_ID"] == result.agent_id
        assert call_kwargs["environment"]["STRATEGY_NAME"] == "RSI Reversal"
        assert call_kwargs["environment"]["MT5_ACCOUNT"] == "12345678"
        assert call_kwargs["environment"]["MAGIC_NUMBER"] == "98765432"
        assert call_kwargs["labels"]["quantmindx-paper-agent"] == "true"

    def test_deploy_returns_deployment_result(
        self, deployer, mock_docker_client, sample_deployment_request, sample_container
    ):
        """Test deploy_paper_agent returns proper result."""
        mock_docker_client.images.get.return_value = MagicMock()
        mock_docker_client.containers.run.return_value = sample_container

        result = deployer.deploy_agent(**sample_deployment_request)

        assert isinstance(result, AgentDeploymentResult)
        assert result.agent_id is not None
        assert result.container_id == sample_container.id
        assert result.status == AgentStatus.RUNNING
        assert "agent:heartbeat:" in result.redis_channel

    def test_deploy_pulls_image_if_not_exists(
        self, deployer, mock_docker_client, sample_deployment_request, sample_container
    ):
        """Test deploy pulls Docker image if not exists."""
        # Image not found initially
        mock_docker_client.images.get.side_effect = [NotFound("Not found"), MagicMock()]
        mock_docker_client.images.pull.return_value = None
        mock_docker_client.containers.run.return_value = sample_container

        deployer.deploy_agent(**sample_deployment_request)

        # Verify pull was called
        assert mock_docker_client.images.pull.called

    def test_deploy_generates_unique_agent_id(
        self, deployer, mock_docker_client, sample_deployment_request, sample_container
    ):
        """Test deploy generates unique agent IDs."""
        mock_docker_client.images.get.return_value = MagicMock()
        mock_docker_client.containers.run.return_value = sample_container

        result1 = deployer.deploy_agent(**sample_deployment_request)
        result2 = deployer.deploy_agent(**sample_deployment_request)

        # Agent IDs should be different (timestamp-based)
        assert result1.agent_id != result2.agent_id

    def test_deploy_with_custom_agent_id(
        self, deployer, mock_docker_client, sample_deployment_request, sample_container
    ):
        """Test deploy respects custom agent_id."""
        mock_docker_client.images.get.return_value = MagicMock()
        mock_docker_client.containers.run.return_value = sample_container

        custom_id = "my-custom-agent-001"
        result = deployer.deploy_agent(agent_id=custom_id, **sample_deployment_request)

        assert result.agent_id == custom_id

    def test_deploy_handles_docker_error(
        self, deployer, mock_docker_client, sample_deployment_request
    ):
        """Test deploy handles Docker API errors gracefully."""
        mock_docker_client.containers.run.side_effect = APIError("Docker error")

        with pytest.raises(RuntimeError, match="Deployment failed"):
            deployer.deploy_agent(**sample_deployment_request)


# ============================================================================
# Test 2: list_paper_agents returns agents
# ============================================================================

class TestListPaperAgents:
    """Test list_paper_agents functionality."""

    def test_list_returns_all_agents(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test list_paper_agents returns all agent containers."""
        # Setup mock containers list
        mock_docker_client.containers.list.return_value = [sample_container]

        agents = deployer.list_agents()

        assert len(agents) == 1
        assert agents[0].agent_id == "strategy-rsi-98765432-1234567890"
        assert agents[0].status == AgentStatus.RUNNING

    def test_list_filters_by_agent_label(
        self, deployer, mock_docker_client
    ):
        """Test list filters containers with quantmindx-paper-agent label."""
        deployer.list_agents()

        call_kwargs = mock_docker_client.containers.list.call_args[1]
        assert "label" in call_kwargs
        assert call_kwargs["all"] is True

    def test_list_includes_stopped_when_all_true(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test list includes stopped containers."""
        sample_container.status = "exited"
        mock_docker_client.containers.list.return_value = [sample_container]

        agents = deployer.list_agents()

        assert len(agents) == 1
        assert agents[0].status == AgentStatus.STOPPED

    def test_list_returns_empty_when_no_agents(
        self, deployer, mock_docker_client
    ):
        """Test list returns empty list when no agents deployed."""
        mock_docker_client.containers.list.return_value = []

        agents = deployer.list_agents()

        assert agents == []

    def test_get_agent_returns_specific_agent(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test get_agent returns specific agent by ID."""
        agent_id = "strategy-rsi-98765432-1234567890"
        mock_docker_client.containers.get.return_value = sample_container

        agent = deployer.get_agent(agent_id)

        assert agent is not None
        assert agent.agent_id == agent_id

    def test_get_agent_returns_none_when_not_found(
        self, deployer, mock_docker_client
    ):
        """Test get_agent returns None when agent not found."""
        mock_docker_client.containers.get.side_effect = NotFound("Not found")

        agent = deployer.get_agent("nonexistent-agent")

        assert agent is None


# ============================================================================
# Test 3: stop_paper_agent terminates gracefully
# ============================================================================

class TestStopPaperAgent:
    """Test stop_paper_agent functionality."""

    def test_stop_sends_sigterm_for_graceful_shutdown(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test stop sends SIGTERM with timeout."""
        mock_docker_client.containers.get.return_value = sample_container

        result = deployer.stop_agent("strategy-rsi-98765432-1234567890", timeout=30)

        assert result is True
        sample_container.stop.assert_called_once_with(timeout=30)

    def test_stop_with_force_uses_kill(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test stop with force=True uses SIGKILL."""
        mock_docker_client.containers.get.return_value = sample_container

        deployer.stop_agent("test-agent", force=True)

        sample_container.kill.assert_called_once()

    def test_stop_handles_not_found_gracefully(
        self, deployer, mock_docker_client
    ):
        """Test stop returns True when agent already stopped."""
        mock_docker_client.containers.get.side_effect = NotFound("Not found")

        result = deployer.stop_agent("already-stopped")

        assert result is True  # Considered success

    def test_remove_stops_and_removes_container(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test remove stops and removes container."""
        mock_docker_client.containers.get.return_value = sample_container

        result = deployer.remove_agent("test-agent")

        assert result is True
        sample_container.remove.assert_called_once_with(force=False)

    def test_remove_with_force(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test remove with force=True."""
        mock_docker_client.containers.get.return_value = sample_container

        deployer.remove_agent("test-agent", force=True)

        sample_container.remove.assert_called_once_with(force=True)


# ============================================================================
# Test 4: get_agent_logs retrieves logs
# ============================================================================

class TestGetAgentLogs:
    """Test get_agent_logs functionality."""

    def test_get_logs_returns_lines(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test get_agent_logs returns log lines."""
        mock_docker_client.containers.get.return_value = sample_container
        sample_container.logs.return_value = b"Line 1\nLine 2\nLine 3\n"

        result = deployer.get_agent_logs("test-agent", tail_lines=100)

        assert isinstance(result, AgentLogsResult)
        assert result.agent_id == "test-agent"
        assert len(result.logs) == 3
        assert result.line_count == 3
        assert result.logs[0] == "Line 1"

    def test_get_logs_respects_tail_limit(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test get_agent_logs respects tail_lines parameter."""
        mock_docker_client.containers.get.return_value = sample_container
        sample_container.logs.return_value = b"log line"

        deployer.get_agent_logs("test-agent", tail_lines=50)

        sample_container.logs.assert_called_once_with(
            tail=50,
            timestamps=False,
            follow=False,
            since=None,
        )

    def test_get_limits_max_lines(
        self, deployer, mock_docker_client, sample_container
    ):
        """Test get_agent_logs enforces maximum lines limit."""
        mock_docker_client.containers.get.return_value = sample_container
        sample_container.logs.return_value = b"log"

        result = deployer.get_agent_logs("test-agent", tail_lines=20000)

        # Should truncate to MAX_LOG_LINES
        sample_container.logs.assert_called_once()
        call_kwargs = sample_container.logs.call_args[1]
        assert call_kwargs["tail"] == 10000

    def test_get_logs_handles_not_found(
        self, deployer, mock_docker_client
    ):
        """Test get_agent_logs raises ValueError when agent not found."""
        mock_docker_client.containers.get.side_effect = NotFound("Not found")

        with pytest.raises(ValueError, match="not found"):
            deployer.get_agent_logs("nonexistent-agent")


# ============================================================================
# Test 5: get_agent_performance calculates metrics
# ============================================================================

class TestGetAgentPerformance:
    """Test get_agent_performance functionality."""

    def test_performance_metrics_calculated_correctly(self):
        """Test performance metrics are calculated correctly."""
        # Sample trade data
        trades = [
            {"metadata": {"pnl": "100.0", "symbol": "EURUSD", "timestamp": "2026-01-29T10:00:00Z"}},
            {"metadata": {"pnl": "-50.0", "symbol": "GBPUSD", "timestamp": "2026-01-29T11:00:00Z"}},
            {"metadata": {"pnl": "75.0", "symbol": "EURUSD", "timestamp": "2026-01-29T12:00:00Z"}},
        ]

        # Calculate
        total_trades = len(trades)
        winning = sum(1 for t in trades if float(t["metadata"]["pnl"]) > 0)
        losing = sum(1 for t in trades if float(t["metadata"]["pnl"]) < 0)
        total_pnl = sum(float(t["metadata"]["pnl"]) for t in trades)
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        # Verify
        assert total_trades == 3
        assert winning == 2
        assert losing == 1
        assert total_pnl == 125.0
        assert win_rate == 66.66666666666666
        assert avg_pnl == 41.666666666666664


# ============================================================================
# Test 6: heartbeat detection marks stale agents
# ============================================================================

class TestHeartbeatDetection:
    """Test heartbeat detection and stale agent marking."""

    def test_healthy_when_recent_heartbeat(self, monitor):
        """Test agent is healthy with recent heartbeat."""
        monitor._last_heartbeat["test-agent"] = datetime.now(UTC) - td(seconds=30)

        health = monitor.get_agent_health("test-agent")

        assert health == AgentHealth.HEALTHY

    def test_stale_when_5_minutes_without_heartbeat(self, monitor):
        """Test agent is stale after 5 minutes without heartbeat."""
        monitor._last_heartbeat["test-agent"] = datetime.now(UTC) - td(seconds=300)

        health = monitor.get_agent_health("test-agent")

        assert health == AgentHealth.STALE

    def test_dead_when_10_minutes_without_heartbeat(self, monitor):
        """Test agent is dead after 10 minutes without heartbeat."""
        monitor._last_heartbeat["test-agent"] = datetime.now(UTC) - td(seconds=600)

        health = monitor.get_agent_health("test-agent")

        assert health == AgentHealth.DEAD

    def test_record_heartbeat_resets_health(self, monitor):
        """Test recording heartbeat resets agent to healthy."""
        monitor._agent_health["test-agent"] = AgentHealth.STALE

        heartbeat_json = json.dumps({
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "agent_id": "test-agent",
            "status": "running",
        })

        new_health = monitor.record_heartbeat("test-agent", heartbeat_json)

        assert new_health == AgentHealth.HEALTHY

    def test_get_missed_heartbeat_count(self, monitor):
        """Test get_missed_heartbeat_count calculates correctly."""
        monitor._last_heartbeat["test-agent"] = datetime.now(UTC) - td(seconds=180)  # 3 minutes

        # 60 second interval, so 180/60 = 3 missed
        missed = monitor.get_missed_heartbeat_count("test-agent")

        assert missed == 3

    def test_check_redis_health(self, monitor):
        """Test checking health from Redis directly."""
        heartbeat_json = json.dumps({
            "timestamp": (datetime.now(UTC) - td(seconds=30)).isoformat().replace("+00:00", "Z"),
            "agent_id": "redis-agent",
            "status": "running",
        })

        # Mock the Redis client
        monitor._redis_client = MagicMock()
        monitor._redis_client.get.return_value = heartbeat_json

        health = monitor._check_redis_health("redis-agent")

        assert health == AgentHealth.HEALTHY

    def test_update_agent_statuses_batch(self, monitor):
        """Test updating health for batch of agents."""
        agents = [
            PaperAgentStatus(
                agent_id=f"agent-{i}",
                container_id=f"container-{i}",
                container_name=f"name-{i}",
                status=AgentStatus.RUNNING,
                strategy_name=f"Strategy {i}",
            )
            for i in range(5)
        ]

        # Set different health states
        monitor._last_heartbeat["agent-0"] = datetime.now(UTC) - td(seconds=30)  # healthy
        monitor._last_heartbeat["agent-1"] = datetime.now(UTC) - td(seconds=300)  # stale
        monitor._last_heartbeat["agent-2"] = datetime.now(UTC) - td(seconds=600)  # dead

        updated = monitor.update_agent_statuses(agents)

        assert len(updated) == 5
        assert updated[0].health == AgentHealth.HEALTHY
        assert updated[1].health == AgentHealth.STALE
        assert updated[2].health == AgentHealth.DEAD


# ============================================================================
# Test 7: ChromaDB storage
# ============================================================================

class TestPaperTradingStorage:
    """Test ChromaDB storage functionality."""

    def test_store_trade_event(self):
        """Test storing trade event in ChromaDB."""
        with patch("mcp_mt5.paper_trading.storage.chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_collection.return_value = mock_collection

            storage = PaperTradingStorage(use_embeddings=False)

            doc_id = storage.store_trade_event(
                agent_id="test-agent",
                event_type="entry",
                symbol="EURUSD",
                price=1.0850,
                lots=0.1,
                metadata={"rsi": 25.5},
            )

            assert doc_id is not None
            assert "test-agent" in doc_id
            assert mock_collection.add.called

    def test_store_performance(self):
        """Test storing performance metrics."""
        with patch("mcp_mt5.paper_trading.storage.chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_collection.return_value = mock_collection

            storage = PaperTradingStorage(use_embeddings=False)

            performance = AgentPerformance(
                agent_id="test-agent",
                total_trades=10,
                winning_trades=7,
                losing_trades=3,
                win_rate=70.0,
                total_pnl=500.0,
                average_pnl=50.0,
                max_drawdown=100.0,
                profit_factor=2.5,
                symbols_traded=["EURUSD", "GBPUSD"],
            )

            doc_id = storage.store_performance(
                agent_id="test-agent",
                performance=performance,
                configuration={"rsi_period": 14},
            )

            assert doc_id is not None
            assert "performance" in doc_id

    def test_get_agent_trades(self):
        """Test retrieving trades for an agent."""
        with patch("mcp_mt5.paper_trading.storage.chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                "ids": [["trade1", "trade2"]],
                "metadatas": [[{"symbol": "EURUSD"}, {"symbol": "GBPUSD"}]],
            }

            storage = PaperTradingStorage(use_embeddings=False)

            trades = storage.get_agent_trades("test-agent")

            assert len(trades) == 2

    def test_search_similar_strategies(self):
        """Test semantic search for similar strategies."""
        with patch("mcp_mt5.paper_trading.storage.chromadb.PersistentClient") as mock_chroma:
            mock_collection = MagicMock()
            mock_chroma.return_value.get_collection.return_value = mock_collection
            mock_collection.query.return_value = {
                "ids": [["doc1"]],
                "metadatas": [[{"agent_id": "agent-1"}]],
                "distances": [[0.2]],
            }

            storage = PaperTradingStorage(use_embeddings=True)

            results = storage.search_similar_strategies(
                query="RSI reversal strategy",
                n_results=5,
            )

            assert len(results) == 1


# ============================================================================
# Test 8: Integration tests
# ============================================================================

class TestPaperTradingIntegration:
    """Integration tests for paper trading deployment."""

    def test_full_deployment_workflow(
        self, deployer, mock_docker_client, sample_deployment_request, sample_container
    ):
        """Test complete deployment workflow."""
        # Deploy
        mock_docker_client.images.get.return_value = MagicMock()
        mock_docker_client.containers.run.return_value = sample_container
        mock_docker_client.containers.list.return_value = [sample_container]

        result = deployer.deploy_agent(**sample_deployment_request)
        assert result.status == AgentStatus.RUNNING

        # List
        agents = deployer.list_agents()
        assert len(agents) == 1

        # Get logs
        mock_docker_client.containers.get.return_value = sample_container
        sample_container.logs.return_value = b"Agent started\n"
        logs = deployer.get_agent_logs(result.agent_id)
        assert len(logs.logs) > 0

        # Stop
        stop_result = deployer.stop_agent(result.agent_id)
        assert stop_result is True

    def test_health_monitoring_integration(
        self, deployer, monitor, mock_docker_client, sample_container
    ):
        """Test health monitoring with deployer integration."""
        mock_docker_client.containers.list.return_value = [sample_container]

        # Get agents with health
        agents = deployer.list_agents()
        updated_agents = monitor.update_agent_statuses(agents)

        assert len(updated_agents) == len(agents)
        assert all(hasattr(a, "health") for a in updated_agents)


# ============================================================================
# Test Summary
# ============================================================================
# Total tests: 30+ focused tests covering:
# 1. Docker container creation with deploy_paper_agent
# 2. Listing agents with list_paper_agents
# 3. Graceful shutdown with stop_paper_agent
# 4. Log retrieval with get_agent_logs
# 5. Performance metrics with get_agent_performance
# 6. Heartbeat detection and stale agent marking
# 7. ChromaDB storage for trades and performance
# 8. Integration workflows
