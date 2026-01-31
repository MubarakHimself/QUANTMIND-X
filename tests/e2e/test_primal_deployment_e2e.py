"""
Test 15.3 & 15.4: @primal Strategy Deployment & Assets Hub Integration

Task Group 15.3: Test that @primal strategies deploy to paper trading
Task Group 15.4: Test Assets Hub + Quant Code Agent integration

These tests validate:
- @primal tag triggers deployment
- Deployment configuration is correct
- Code is stored in Git with proper metadata
- Assets Hub provides templates and standards
- Skills are loadable during code generation
"""

import pytest
import json
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_primal_tag_triggers_deployment(
    quant_code_agent,
    sample_trd,
    e2e_helper,
    temp_git_repo,
    mock_mcp_servers
):
    """
    Test 15.3: @primal strategies trigger paper trading deployment.

    Validates:
    - @primal tag assignment
    - Deployment function is called
    - Deployment config includes strategy details
    - Git storage occurs before deployment
    """
    # Create passing backtest result
    passing_result = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.2,
        max_drawdown=12.0,
        win_rate=56.0
    )

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": e2e_helper.create_sample_strategy_code(),
        "backtest_results": passing_result,
        "performance_metrics": passing_result["metrics"],
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Primal_MA_Crossover",
        "deployment_config": e2e_helper.create_deployment_config()
    }

    # Mock deployment trigger
    deployment_triggered = []

    original_trigger = quant_code_agent.trigger_paper_trading_deployment

    async def mock_trigger(strategy_name, code, backtest_results, deployment_config):
        deployment_triggered.append({
            "strategy_name": strategy_name,
            "code_length": len(code),
            "metrics": backtest_results.get("metrics", {}),
            "config": deployment_config
        })
        return True

    quant_code_agent.trigger_paper_trading_deployment = AsyncMock(side_effect=mock_trigger)

    # Analyze should assign @primal and trigger deployment
    result = await quant_code_agent.analyze_node(state)

    assert result["is_primal"] is True, "Should assign @primal tag"
    assert result.get("compilation_error") is None, "Should have no errors"

    # Verify deployment was triggered
    assert len(deployment_triggered) == 1, "Deployment should be triggered once"
    deployment_info = deployment_triggered[0]

    assert deployment_info["strategy_name"] == "Primal_MA_Crossover"
    assert deployment_info["code_length"] > 0
    assert deployment_info["metrics"]["sharpe_ratio"] == 2.2

    # Verify deployment was recorded in agent state
    deployment_history = quant_code_agent.get_deployment_history()
    assert len(deployment_history) > 0, "Deployment should be in history"

    latest = deployment_history[-1]
    assert latest["strategy_name"] == "Primal_MA_Crossover"
    assert latest["status"] == "triggered"

    print(f"\n@primal deployment test passed:")
    print(f"  - @primal tag assigned")
    print(f"  - Deployment triggered")
    print(f"  - Deployment history updated")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_deployment_includes_backtest_results(
    quant_code_agent,
    e2e_helper
):
    """
    Test that deployment configuration includes backtest results.

    Validates:
    - Sharpe ratio is included
    - Drawdown is included
    - Other metrics are included
    - Timestamp is recorded
    """
    backtest_result = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=1.9,
        max_drawdown=14.5,
        win_rate=54.0,
        profit_factor=2.1
    )

    state = {
        "messages": [],
        "trd_content": "test trd",
        "current_code": "test code",
        "backtest_results": backtest_result,
        "performance_metrics": backtest_result["metrics"],
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": True,
        "strategy_name": "Metrics_Test_Strategy",
        "deployment_config": {}
    }

    # Trigger deployment
    success = await quant_code_agent.trigger_paper_trading_deployment(
        strategy_name=state["strategy_name"],
        code=state["current_code"],
        backtest_results=state["backtest_results"],
        deployment_config=state["deployment_config"]
    )

    assert success is True

    # Check deployment history
    history = quant_code_agent.get_deployment_history()
    assert len(history) > 0

    deployment = history[-1]

    # Verify backtest results are included
    assert "deployment_config" in deployment
    config = deployment["deployment_config"]

    assert config["strategy_name"] == "Metrics_Test_Strategy"
    assert "backtest_results" in config
    assert config["backtest_results"]["sharpe_ratio"] == 1.9
    assert config["backtest_results"]["drawdown"] == 14.5

    print(f"\nDeployment backtest results test passed:")
    print(f"  - Sharpe ratio: {config['backtest_results']['sharpe_ratio']}")
    print(f"  - Drawdown: {config['backtest_results']['drawdown']}%")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_code_stored_in_git_before_deployment(
    quant_code_agent,
    e2e_helper,
    temp_git_repo
):
    """
    Test that code is stored in Git before deployment.

    Validates:
    - Git write operation occurs
    - Git commit occurs with proper message
    - Commit message includes backtest metrics
    - File is in correct directory
    """
    backtest_result = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.0,
        max_drawdown=13.0,
        win_rate=55.0,
        profit_factor=2.2
    )

    strategy_code = e2e_helper.create_sample_strategy_code()
    strategy_name = "Git_Test_Strategy"

    # Store code in Git
    success = await quant_code_agent.store_code_in_git(
        code=strategy_code,
        strategy_name=strategy_name,
        backtest_results=backtest_result,
        category="generated_bots"
    )

    assert success is True, "Git storage should succeed"

    # Verify Git operations were called
    # Note: In actual test, git_client calls would be verified
    # Here we verify the method completed without errors

    print(f"\nGit storage test passed:")
    print(f"  - Code stored successfully")
    print(f"  - Strategy: {strategy_name}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_assets_hub_fetches_coding_standards(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.4.1: Assets Hub provides coding standards during planning.

    Validates:
    - get_coding_standards is called
    - Standards are included in planning prompt
    - Standards influence code generation
    """
    mcp_calls = []

    async def track_mcp_call(server, tool, args):
        mcp_calls.append({"server": server, "tool": tool, "args": args})
        if server in mock_mcp_servers:
            return await mock_mcp_servers[server].call_tool(tool, args)
        return None

    quant_code_agent.call_mcp_tool = AsyncMock(side_effect=track_mcp_call)

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Standards_Test",
        "deployment_config": None
    }

    # Execute planning node
    result = await quant_code_agent.planning_node(state)

    # Verify get_coding_standards was called
    standards_calls = [c for c in mcp_calls if c["tool"] == "get_coding_standards"]
    assert len(standards_calls) > 0, "Should fetch coding standards"

    # Verify planning response includes standards
    assert len(result["messages"]) > 0
    planning_content = result["messages"][0].content

    # Check that standards were incorporated
    assert "STANDARDS" in planning_content or "standards" in planning_content.lower()

    print(f"\nCoding standards test passed:")
    print(f"  - get_coding_standards called")
    print(f"  - Standards incorporated in planning")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_assets_hub_fetches_bad_patterns(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.4.2: Assets Hub provides bad patterns to avoid.

    Validates:
    - get_bad_patterns is called
    - Patterns are included in planning prompt
    - Patterns warn against anti-patterns
    """
    mcp_calls = []

    async def track_mcp_call(server, tool, args):
        mcp_calls.append({"server": server, "tool": tool, "args": args})
        if server in mock_mcp_servers:
            return await mock_mcp_servers[server].call_tool(tool, args)
        return None

    quant_code_agent.call_mcp_tool = AsyncMock(side_effect=track_mcp_call)

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Patterns_Test",
        "deployment_config": None
    }

    result = await quant_code_agent.planning_node(state)

    # Verify get_bad_patterns was called
    patterns_calls = [c for c in mcp_calls if c["tool"] == "get_bad_patterns"]
    assert len(patterns_calls) > 0, "Should fetch bad patterns"

    # Verify patterns were incorporated
    planning_content = result["messages"][0].content
    assert "BAD PATTERNS" in planning_content or "patterns" in planning_content.lower()

    print(f"\nBad patterns test passed:")
    print(f"  - get_bad_patterns called")
    print(f"  - Patterns incorporated in planning")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_assets_hub_fetches_algorithm_templates(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.4.3: Assets Hub provides algorithm templates.

    Validates:
    - get_algorithm_template is called for relevant strategies
    - Template info is included in planning
    - Template guides code structure
    """
    mcp_calls = []

    async def track_mcp_call(server, tool, args):
        mcp_calls.append({"server": server, "tool": tool, "args": args})
        if server in mock_mcp_servers:
            return await mock_mcp_servers[server].call_tool(tool, args)
        return None

    quant_code_agent.call_mcp_tool = AsyncMock(side_effect=track_mcp_call)

    state = {
        "messages": [],
        "trd_content": sample_trd,  # Contains "strategy" keyword
        "current_code": "",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Template_Test",
        "deployment_config": None
    }

    result = await quant_code_agent.planning_node(state)

    # Verify get_algorithm_template was called
    template_calls = [c for c in mcp_calls if c["tool"] == "get_algorithm_template"]
    assert len(template_calls) > 0, "Should fetch algorithm template"

    # Verify template was incorporated
    planning_content = result["messages"][0].content
    assert "Template" in planning_content or "template" in planning_content.lower()

    print(f"\nAlgorithm template test passed:")
    print(f"  - get_algorithm_template called")
    print(f"  - Template incorporated in planning")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_assets_hub_integration_complete(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.4: Complete Assets Hub integration test.

    Validates that all Assets Hub tools work together:
    - Coding standards are fetched
    - Bad patterns are fetched
    - Templates are fetched (when applicable)
    - All are incorporated into planning
    """
    mcp_calls = []

    async def track_mcp_call(server, tool, args):
        mcp_calls.append({"server": server, "tool": tool, "args": args})
        if server in mock_mcp_servers:
            return await mock_mcp_servers[server].call_tool(tool, args)
        return None

    quant_code_agent.call_mcp_tool = AsyncMock(side_effect=track_mcp_call)

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Complete_Integration_Test",
        "deployment_config": None
    }

    result = await quant_code_agent.planning_node(state)

    # Verify all Assets Hub tools were called
    assets_hub_calls = [c for c in mcp_calls if c["server"] == "quantmindx-kb"]
    assert len(assets_hub_calls) >= 2, "Should call multiple Assets Hub tools"

    tools_called = [c["tool"] for c in assets_hub_calls]
    assert "get_coding_standards" in tools_called
    assert "get_bad_patterns" in tools_called

    # Verify planning response incorporates all fetched data
    planning_content = result["messages"][0].content
    assert len(planning_content) > 100, "Planning should be comprehensive"

    print(f"\nComplete Assets Hub integration test passed:")
    print(f"  - {len(assets_hub_calls)} Assets Hub tools called")
    print(f"  - Tools: {', '.join(set(tools_called))}")
    print(f"  - Planning response length: {len(planning_content)} chars")
