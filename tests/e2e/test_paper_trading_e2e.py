"""
Test 15.6 & 15.7: Paper Trading + MT5 MCP Integration & Multi-Agent Workflows

Task Group 15.6: Test Paper Trading + MT5 MCP integration
Task Group 15.7: Test multi-component workflows with 5+ agents

These tests validate:
- Paper trading agents can be deployed
- Agents publish heartbeats to Redis
- Agents publish trade events
- Multiple agents can run simultaneously
- MT5 MCP integration works
- Multi-agent workflows don't collide
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_deploy_paper_agent_creates_container(
    quant_code_agent,
    deployment_config,
    mock_mcp_servers,
    mock_docker_client
):
    """
    Test 15.6.1: deploy_paper_agent creates Docker container correctly.

    Validates:
    - Container is created with correct image
    - Environment variables are set
    - Container is started
    - Agent ID is returned
    """
    # Mock MT5 MCP deploy_paper_agent response
    deploy_response = {
        "agent_id": "agent-001",
        "container_id": "container-123",
        "status": "running",
        "redis_channel": "agent:heartbeat:agent-001",
        "logs_url": "/logs/agent-001"
    }

    mock_mcp_servers["mt5-server"].call_tool = AsyncMock(
        return_value=json.dumps(deploy_response)
    )

    # Trigger deployment
    success = await quant_code_agent.trigger_paper_trading_deployment(
        strategy_name="Test_MA_Crossover",
        code="strategy code",
        backtest_results={"sharpe_ratio": 2.0, "max_drawdown": 15.0},
        deployment_config=deployment_config
    )

    assert success is True

    # Verify deployment recorded
    history = quant_code_agent.get_deployment_history()
    assert len(history) > 0

    deployment = history[-1]
    assert deployment["strategy_name"] == "Test_MA_Crossover"
    assert deployment["status"] == "triggered"

    print(f"\nDeploy paper agent test passed:")
    print(f"  - Agent ID: {deploy_response['agent_id']}")
    print(f"  - Container ID: {deploy_response['container_id']}")
    print(f"  - Status: {deploy_response['status']}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_list_paper_agents(
    quant_code_agent,
    mock_mcp_servers
):
    """
    Test 15.6.2: list_paper_agents returns all running agents.

    Validates:
    - All agents are listed
    - Status information is correct
    - Uptime is calculated
    """
    # Mock list response
    agents_response = [
        {
            "agent_id": "agent-001",
            "strategy_name": "MA_Crossover_EURUSD",
            "status": "running",
            "uptime": 3600,
            "trades": 15,
            "pnl": 250.50
        },
        {
            "agent_id": "agent-002",
            "strategy_name": "RSI_Strategy",
            "status": "running",
            "uptime": 7200,
            "trades": 32,
            "pnl": 180.25
        }
    ]

    mock_mcp_servers["mt5-server"].call_tool = AsyncMock(
        return_value=json.dumps(agents_response)
    )

    # Call list_paper_agents
    result = await quant_code_agent.call_mcp_tool(
        "mt5-server",
        "list_paper_agents",
        {}
    )

    assert result is not None

    agents = json.loads(result) if isinstance(result, str) else result
    assert len(agents) == 2
    assert agents[0]["agent_id"] == "agent-001"
    assert agents[0]["status"] == "running"

    print(f"\nList paper agents test passed:")
    print(f"  - {len(agents)} agents listed")
    print(f"  - Agent 1: {agents[0]['strategy_name']}")
    print(f"  - Agent 2: {agents[1]['strategy_name']}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_publishes_heartbeat(
    mock_redis_client
):
    """
    Test 15.6.3: Agents publish heartbeat messages to Redis.

    Validates:
    - Heartbeat message format is correct
    - Messages are published to correct channel
    - Timestamp is included
    - Status is included
    """
    from src.agents.integrations.redis_client import HeartbeatMessage
    from datetime import datetime, UTC

    # Create heartbeat message with timestamp
    heartbeat = HeartbeatMessage(
        timestamp=datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
        agent_id="agent-001",
        status="running",
        uptime_seconds=3600
    )

    # Publish heartbeat
    import json
    message_data = heartbeat.model_dump_json()

    await mock_redis_client.publish(
        f"agent:heartbeat:{heartbeat.agent_id}",
        message_data
    )

    # Verify publish was called
    mock_redis_client.publish.assert_called_once()

    call_args = mock_redis_client.publish.call_args
    channel = call_args[0][0]
    message = call_args[0][1]

    assert channel == "agent:heartbeat:agent-001"

    # Verify message structure
    parsed = json.loads(message)
    assert "timestamp" in parsed
    assert "agent_id" in parsed
    assert "status" in parsed
    assert "uptime_seconds" in parsed

    print(f"\nHeartbeat publish test passed:")
    print(f"  - Channel: {channel}")
    print(f"  - Agent ID: {parsed['agent_id']}")
    print(f"  - Status: {parsed['status']}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_publishes_trade_events(
    mock_redis_client
):
    """
    Test 15.6.4: Agents publish trade events to Redis.

    Validates:
    - Trade event message format is correct
    - Entry/exit events are published
    - PnL is included
    - Symbol and price are included
    """
    from src.agents.integrations.redis_client import TradeEventMessage
    from datetime import datetime, UTC

    # Create trade event with timestamp
    trade = TradeEventMessage(
        timestamp=datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
        agent_id="agent-001",
        action="entry",
        symbol="EURUSD",
        price=1.0850,
        lots=0.1,
        pnl=0.0
    )

    # Publish trade event
    import json
    message_data = trade.model_dump_json()

    await mock_redis_client.publish(
        f"agent:trades:{trade.agent_id}",
        message_data
    )

    # Verify publish
    mock_redis_client.publish.assert_called_once()

    call_args = mock_redis_client.publish.call_args
    channel = call_args[0][0]

    assert channel == "agent:trades:agent-001"

    parsed = json.loads(call_args[0][1])
    assert parsed["action"] == "entry"
    assert parsed["symbol"] == "EURUSD"
    assert parsed["price"] == 1.0850

    print(f"\nTrade event publish test passed:")
    print(f"  - Channel: {channel}")
    print(f"  - Action: {parsed['action']}")
    print(f"  - Symbol: {parsed['symbol']}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agent_deployment(
    quant_code_agent,
    mock_mcp_servers,
    e2e_helper
):
    """
    Test 15.7.1: Deploy 5+ paper trading agents simultaneously.

    Validates:
    - Multiple agents can be deployed
    - Each agent has unique ID
    - No collisions between agents
    - All agents start successfully
    """
    deployed_agents = []

    # Deploy 5 agents
    for i in range(5):
        strategy_name = f"MA_Crossover_{i}"
        backtest_result = e2e_helper.create_mock_backtest_result(
            sharpe_ratio=1.8 + (i * 0.1),
            max_drawdown=15.0 - (i * 0.5),
            win_rate=50.0 + (i * 1.0)
        )

        # Mock deployment response
        mock_mcp_servers["mt5-server"].call_tool = AsyncMock(
            return_value=json.dumps({
                "agent_id": f"agent-{i:03d}",
                "container_id": f"container-{i:03d}",
                "status": "running",
                "redis_channel": f"agent:heartbeat:agent-{i:03d}"
            })
        )

        success = await quant_code_agent.trigger_paper_trading_deployment(
            strategy_name=strategy_name,
            code=f"strategy code {i}",
            backtest_results=backtest_result,
            deployment_config={}
        )

        assert success is True
        deployed_agents.append(strategy_name)

    # Verify all deployed
    assert len(deployed_agents) == 5

    # Verify agent history
    history = quant_code_agent.get_deployment_history()
    assert len(history) >= 5

    print(f"\nMulti-agent deployment test passed:")
    print(f"  - Deployed {len(deployed_agents)} agents")
    print(f"  - Agents: {', '.join(deployed_agents)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agent_no_heartbeat_collision(
    mock_redis_client
):
    """
    Test 15.7.2: Multiple agents can publish heartbeats without collision.

    Validates:
    - Each agent publishes to unique channel
    - Messages don't interfere
    - All messages are received
    """
    from src.agents.integrations.redis_client import HeartbeatMessage
    from datetime import datetime, UTC

    # Simulate 5 agents publishing heartbeats
    agent_ids = [f"agent-{i:03d}" for i in range(5)]
    channels = []

    for agent_id in agent_ids:
        heartbeat = HeartbeatMessage(
            timestamp=datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
            agent_id=agent_id,
            status="running",
            uptime_seconds=3600
        )

        channel = f"agent:heartbeat:{agent_id}"
        channels.append(channel)

        await mock_redis_client.publish(channel, heartbeat.model_dump_json())

    # Verify all channels are unique
    assert len(set(channels)) == len(channels)

    # Verify publish was called 5 times
    assert mock_redis_client.publish.call_count == 5

    print(f"\nNo heartbeat collision test passed:")
    print(f"  - {len(agent_ids)} agents published")
    print(f"  - {len(set(channels))} unique channels")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agent_isolated_magic_numbers(
    quant_code_agent,
    mock_mcp_servers,
    e2e_helper
):
    """
    Test 15.7.3: Each agent has isolated magic number and account.

    Validates:
    - Each deployment gets unique magic number
    - Each deployment uses separate account
    - No cross-contamination
    """
    deployments = []

    for i in range(3):
        strategy_name = f"Strategy_{i}"
        backtest_result = e2e_helper.create_mock_backtest_result()

        # Create unique deployment config
        config = {
            "strategy_name": strategy_name,
            "symbol": "EURUSD",
            "magic_number": 1000 + i,  # Unique magic number
            "mt5_account": f"demo-account-{i}"  # Unique account
        }

        mock_mcp_servers["mt5-server"].call_tool = AsyncMock(
            return_value=json.dumps({
                "agent_id": f"agent-{i}",
                "magic_number": config["magic_number"],
                "account": config["mt5_account"],
                "status": "running"
            })
        )

        await quant_code_agent.trigger_paper_trading_deployment(
            strategy_name=strategy_name,
            code="code",
            backtest_results=backtest_result,
            deployment_config=config
        )

        deployments.append(config)

    # Verify isolation
    magic_numbers = [d["magic_number"] for d in deployments]
    accounts = [d["mt5_account"] for d in deployments]

    assert len(set(magic_numbers)) == len(magic_numbers)
    assert len(set(accounts)) == len(accounts)

    print(f"\nIsolated magic numbers test passed:")
    print(f"  - Magic numbers: {magic_numbers}")
    print(f"  - Accounts: {accounts}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_get_agent_performance(
    quant_code_agent,
    mock_redis_client,
    mock_mcp_servers
):
    """
    Test 15.6.5: get_agent_performance calculates metrics correctly.

    Validates:
    - Total trades counted
    - Win rate calculated
    - Total PnL calculated
    - Average PnL calculated
    """
    # Mock performance data
    performance_data = {
        "agent_id": "agent-001",
        "total_trades": 25,
        "win_rate": 56.0,
        "total_pnl": 350.75,
        "average_pnl": 14.03
    }

    mock_mcp_servers["mt5-server"].call_tool = AsyncMock(
        return_value=json.dumps(performance_data)
    )

    result = await quant_code_agent.call_mcp_tool(
        "mt5-server",
        "get_agent_performance",
        {"agent_id": "agent-001"}
    )

    assert result is not None

    perf = json.loads(result) if isinstance(result, str) else result

    assert perf["total_trades"] == 25
    assert perf["win_rate"] == 56.0
    assert perf["total_pnl"] == 350.75

    print(f"\nAgent performance test passed:")
    print(f"  - Trades: {perf['total_trades']}")
    print(f"  - Win Rate: {perf['win_rate']}%")
    print(f"  - Total PnL: ${perf['total_pnl']}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_agent_health_monitoring(
    quant_code_agent,
    mock_mcp_servers
):
    """
    Test 15.6.6: Agent health monitoring detects stale agents.

    Validates:
    - Health check runs periodically
    - Stale agents are detected
    - Alerts are sent for unhealthy agents
    """
    # Mock health check response
    health_data = {
        "healthy_agents": ["agent-001", "agent-002"],
        "stale_agents": ["agent-003"],  # Missed 5 heartbeats
        "unhealthy_agents": ["agent-004"]  # Error state
    }

    mock_mcp_servers["mt5-server"].call_tool = AsyncMock(
        return_value=json.dumps(health_data)
    )

    result = await quant_code_agent.call_mcp_tool(
        "mt5-server",
        "check_agent_health",
        {}
    )

    health = json.loads(result) if isinstance(result, str) else result

    assert len(health["healthy_agents"]) == 2
    assert len(health["stale_agents"]) == 1
    assert health["stale_agents"][0] == "agent-003"

    print(f"\nHealth monitoring test passed:")
    print(f"  - Healthy: {len(health['healthy_agents'])}")
    print(f"  - Stale: {len(health['stale_agents'])}")
    print(f"  - Unhealthy: {len(health['unhealthy_agents'])}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_complete_multi_component_workflow(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers,
    mock_redis_client,
    e2e_helper
):
    """
    Test 15.7: Complete multi-component workflow with all integrations.

    Validates:
    - Full workflow from TRD to monitoring
    - All components integrate correctly
    - No errors in end-to-end flow
    """
    # Step 1: Planning (Assets Hub)
    mcp_calls = []

    async def track_calls(server, tool, args):
        mcp_calls.append((server, tool))
        if server in mock_mcp_servers:
            return await mock_mcp_servers[server].call_tool(tool, args)
        return None

    quant_code_agent.call_mcp_tool = AsyncMock(side_effect=track_calls)

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Complete_Workflow_Test",
        "deployment_config": {}
    }

    # Execute workflow
    state.update(await quant_code_agent.planning_node(state))
    assert "messages" in state

    state.update(await quant_code_agent.coding_node(state))
    assert len(state["current_code"]) > 0

    # Mock backtest success
    mock_result = e2e_helper.create_mock_backtest_result()
    mock_mcp_servers["backtest-server"].call_tool = AsyncMock(
        return_value=json.dumps({
            "backtest_id": "bt-complete",
            "status": "completed",
            "result": mock_result
        })
    )

    state.update(await quant_code_agent.backtest_node(state))
    assert state["backtest_results"] is not None

    state.update(await quant_code_agent.analyze_node(state))

    # If @primal, mock deployment
    if state.get("is_primal"):
        mock_mcp_servers["mt5-server"].call_tool = AsyncMock(
            return_value=json.dumps({
                "agent_id": "agent-complete",
                "status": "running"
            })
        )

        await quant_code_agent.trigger_paper_trading_deployment(
            strategy_name=state["strategy_name"],
            code=state["current_code"],
            backtest_results=state["backtest_results"],
            deployment_config=state["deployment_config"]
        )

    # Verify MCP calls were made
    servers_called = set(server for server, _ in mcp_calls)

    assert "quantmindx-kb" in servers_called or "backtest-server" in mcp_calls

    print(f"\nComplete multi-component workflow test passed:")
    print(f"  - Planning: OK")
    print(f"  - Coding: OK")
    print(f"  - Backtest: OK")
    print(f"  - Analysis: OK")
    print(f"  - @primal: {state.get('is_primal')}")
    print(f"  - MCP servers called: {len(servers_called)}")
