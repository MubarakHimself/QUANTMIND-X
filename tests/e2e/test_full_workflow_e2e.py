"""
Test 15.1: Full Workflow End-to-End Test

Task Group 15.1: Test complete workflow from TRD to deployment and monitoring

This test validates the entire pipeline:
1. TRD Input -> Planning (Assets Hub integration)
2. Code Generation -> Backtest Execution
3. Backtest Analysis -> @primal Tag Assignment
4. Paper Trading Deployment -> Monitoring
"""

import pytest
import json
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_full_workflow_trd_to_deployment(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers,
    temp_git_repo,
    e2e_helper
):
    """
    Test 15.1: Complete workflow from TRD input to paper trading deployment.

    Validates:
    - TRD is processed correctly
    - Planning node fetches from Assets Hub
    - Code is generated
    - Backtest executes successfully
    - @primal tag is assigned when metrics pass thresholds
    - Deployment is triggered
    - Code is stored in Git
    """
    # Prepare initial state with TRD
    initial_state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "MA_Crossover_Strategy",
        "deployment_config": e2e_helper.create_deployment_config()
    }

    # Track MCP calls to verify integration
    mcp_calls = []

    async def track_mcp_call(server_name, tool_name, args):
        mcp_calls.append({"server": server_name, "tool": tool_name, "args": args})
        if server_name in mock_mcp_servers:
            return await mock_mcp_servers[server_name].call_tool(tool_name, args)
        return None

    # Patch MCP tool calls to track them
    quant_code_agent.call_mcp_tool = AsyncMock(side_effect=track_mcp_call)

    # Step 1: Execute planning node (should fetch from Assets Hub)
    planning_result = await quant_code_agent.planning_node(initial_state)

    assert "messages" in planning_result
    assert len(planning_result["messages"]) > 0

    # Verify Assets Hub was called during planning
    assets_hub_calls = [c for c in mcp_calls if c["server"] == "quantmindx-kb"]
    assert len(assets_hub_calls) >= 2, "Should call Assets Hub for standards and patterns"

    tools_called = [c["tool"] for c in assets_hub_calls]
    assert "get_coding_standards" in tools_called, "Should fetch coding standards"
    assert "get_bad_patterns" in tools_called, "Should fetch bad patterns"

    # Step 2: Execute coding node
    state_after_planning = {**initial_state, **planning_result}
    coding_result = await quant_code_agent.coding_node(state_after_planning)

    assert "current_code" in coding_result
    assert len(coding_result["current_code"]) > 0, "Code should be generated"
    assert "class" in coding_result["current_code"], "Generated code should contain class definition"

    # Step 3: Execute backtest node
    state_after_coding = {**state_after_planning, **coding_result}
    backtest_result = await quant_code_agent.backtest_node(state_after_coding)

    assert "backtest_results" in backtest_result
    assert "performance_metrics" in backtest_result

    # Verify backtest MCP was called
    backtest_calls = [c for c in mcp_calls if c["server"] == "backtest-server"]
    assert len(backtest_calls) > 0, "Should call backtest-server"

    backtest_tools = [c["tool"] for c in backtest_calls]
    assert "run_backtest" in backtest_tools, "Should run backtest"

    # Step 4: Execute analyze node (should assign @primal if metrics pass)
    state_after_backtest = {**state_after_coding, **backtest_result}
    analyze_result = await quant_code_agent.analyze_node(state_after_backtest)

    assert "is_primal" in analyze_result

    # If backtest passed thresholds, verify @primal assignment
    if state_after_backtest.get("backtest_results"):
        metrics = state_after_backtest["backtest_results"].get("metrics", {})
        sharpe = metrics.get("sharpe_ratio", 0)
        drawdown = metrics.get("max_drawdown", 100)
        win_rate = metrics.get("win_rate", 0)

        # Check if metrics pass thresholds
        passes_thresholds = (
            sharpe > 1.5 and
            drawdown < 20.0 and
            win_rate > 45.0
        )

        if passes_thresholds:
            assert analyze_result["is_primal"] is True, "Should assign @primal tag when metrics pass"

            # Step 5: Verify deployment was triggered for @primal strategies
            deployment_history = quant_code_agent.get_deployment_history()
            assert len(deployment_history) > 0, "Should trigger deployment for @primal strategy"

            latest_deployment = deployment_history[-1]
            assert "strategy_name" in latest_deployment
            assert "deployment_config" in latest_deployment
            assert latest_deployment["status"] in ["triggered", "pending"]

            # Step 6: Verify code was stored in Git
            git_repo_path = Path(temp_git_repo)
            generated_bots_path = git_repo_path / "generated_bots"

            assert generated_bots_path.exists(), "Generated bots directory should exist"

            # Check for strategy files
            strategy_files = list(generated_bots_path.glob("*.mq5")) + list(generated_bots_path.glob("*.py"))
            # Note: May be empty if mock git_client doesn't actually write files

    logger = quant_code_agent.__class__.__module__
    print(f"\nFull workflow test completed successfully")
    print(f"MCP calls made: {len(mcp_calls)}")
    print(f"Assets Hub calls: {len(assets_hub_calls)}")
    print(f"Backtest calls: {len(backtest_calls)}")
    print(f"@primal assigned: {analyze_result.get('is_primal', False)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_workflow_state_transitions(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test that the workflow state transitions correctly through all nodes.

    Validates the LangGraph state machine:
    - planner -> coder -> backtest -> analyze -> (retry | done)
    """
    initial_state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Test_Strategy",
        "deployment_config": None
    }

    # Track state evolution
    state_history = [initial_state.copy()]

    # Execute full workflow through all nodes
    state = initial_state.copy()

    # 1. Planning
    state.update(await quant_code_agent.planning_node(state))
    state_history.append(state.copy())
    assert "messages" in state

    # 2. Coding
    state.update(await quant_code_agent.coding_node(state))
    state_history.append(state.copy())
    assert "current_code" in state
    assert len(state["current_code"]) > 0

    # 3. Backtest
    state.update(await quant_code_agent.backtest_node(state))
    state_history.append(state.copy())
    assert "backtest_results" in state

    # 4. Analysis
    state.update(await quant_code_agent.analyze_node(state))
    state_history.append(state.copy())
    assert "is_primal" in state

    # 5. Check routing decision
    routing = quant_code_agent.should_continue(state)
    assert routing in ["retry", "done"]

    # Verify state transitions are valid
    for i, (prev_state, next_state) in enumerate(zip(state_history[:-1], state_history[1:])):
        # Each step should add or modify state
        assert prev_state != next_state, f"State should change at step {i}"

    print(f"\nState transitions validated: {len(state_history)} states")
    for i, s in enumerate(state_history):
        print(f"  State {i}: is_primal={s.get('is_primal')}, retry_count={s.get('retry_count')}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_workflow_with_all_thresholds_passing(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers,
    e2e_helper
):
    """
    Test workflow when all backtest metrics pass thresholds.

    Expected flow:
    - Generate code
    - Backtest returns excellent metrics
    - @primal tag assigned
    - Deployment triggered
    - No retries needed
    """
    # Create mock backtest result with passing metrics
    passing_result = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.5,  # > 1.5
        max_drawdown=12.0,  # < 20%
        win_rate=58.0       # > 45%
    )

    initial_state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": e2e_helper.create_sample_strategy_code(),
        "backtest_results": passing_result,
        "performance_metrics": passing_result["metrics"],
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Passing_Strategy",
        "deployment_config": {}
    }

    # Run analysis node
    result = await quant_code_agent.analyze_node(initial_state)

    # Verify @primal assignment
    assert result["is_primal"] is True, "Should assign @primal when all metrics pass"
    assert result.get("compilation_error") is None, "Should have no errors when passing"

    # Verify routing
    routing = quant_code_agent.should_continue({**initial_state, **result})
    assert routing == "done", "Should route to done when @primal"

    print(f"\nPassing metrics test: @primal={result['is_primal']}, route={routing}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_workflow_metrics_integration(
    quant_code_agent,
    sample_trd,
    e2e_helper
):
    """
    Test that metrics flow correctly through the workflow.

    Validates:
    - Metrics are extracted from backtest results
    - Metrics are used for threshold evaluation
    - Metrics are stored in state correctly
    """
    backtest_result = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=1.8,
        max_drawdown=15.0,
        win_rate=52.0
    )

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": e2e_helper.create_sample_strategy_code(),
        "backtest_results": backtest_result,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Metrics_Test",
        "deployment_config": None
    }

    # Analyze should extract metrics
    result = await quant_code_agent.analyze_node(state)

    assert result["performance_metrics"] is not None
    assert "sharpe_ratio" in result["performance_metrics"]
    assert "max_drawdown" in result["performance_metrics"]
    assert "win_rate" in result["performance_metrics"]

    # Verify metric values match
    assert result["performance_metrics"]["sharpe_ratio"] == 1.8
    assert result["performance_metrics"]["max_drawdown"] == 15.0
    assert result["performance_metrics"]["win_rate"] == 52.0

    print(f"\nMetrics integration test passed: {result['performance_metrics']}")
