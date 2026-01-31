"""
Test 15.5: Backtest Engine + Quant Code Agent Integration

Task Group 15.5: Test integration between Backtest Engine and Quant Code Agent

This test validates:
- backtest_node calls Backtest MCP correctly
- Backtest results are parsed accurately
- Error handling for syntax/data/runtime/timeout errors
- Metrics are calculated correctly
- Status polling works
"""

import pytest
import json
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_backtest_node_calls_mcp_correctly(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.5.1: backtest_node calls Backtest MCP with correct parameters.

    Validates:
    - run_backtest is called with code content
    - Config is passed correctly
    - Language parameter is correct
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
        "current_code": """
class SimpleStrategy:
    def __init__(self):
        self.name = "Test"

    def execute(self):
        return "BUY"
""",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Backtest_Call_Test",
        "deployment_config": None
    }

    # Execute backtest node
    result = await quant_code_agent.backtest_node(state)

    # Verify backtest MCP was called
    backtest_calls = [c for c in mcp_calls if c["server"] == "backtest-server"]
    assert len(backtest_calls) > 0, "Should call backtest-server"

    # Check run_backtest was called
    run_calls = [c for c in backtest_calls if c["tool"] == "run_backtest"]
    assert len(run_calls) > 0, "Should call run_backtest"

    # Verify parameters
    call_args = run_calls[0]["args"]
    assert "code_content" in call_args
    assert "language" in call_args
    assert "config" in call_args

    assert call_args["language"] == "python"
    assert call_args["config"]["symbol"] == "EURUSD"
    assert call_args["config"]["timeframe"] == 60

    print(f"\nBacktest MCP call test passed:")
    print(f"  - run_backtest called with correct parameters")
    print(f"  - Symbol: {call_args['config']['symbol']}")
    print(f"  - Timeframe: {call_args['config']['timeframe']}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_backtest_results_parsed_accurately(
    quant_code_agent,
    sample_trd,
    e2e_helper,
    mock_mcp_servers
):
    """
    Test 15.5.2: Backtest results are parsed and stored correctly.

    Validates:
    - BacktestResult structure is correct
    - Metrics are extracted
    - Equity curve is captured
    - Trade log is captured
    """
    # Mock successful backtest response
    mock_result = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.1,
        max_drawdown=11.5,
        win_rate=57.0,
        profit_factor=2.3
    )

    mock_mcp_servers["backtest-server"].call_tool = AsyncMock(
        side_effect=lambda tool, args: json.dumps({
            "backtest_id": "bt-parse-test-001",
            "status": "completed",
            "progress_percent": 100.0,
            "result": mock_result
        })
    )

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "strategy code",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Parse_Test",
        "deployment_config": None
    }

    result = await quant_code_agent.backtest_node(state)

    # Verify results were parsed
    assert result["backtest_results"] is not None
    assert result["performance_metrics"] is not None

    metrics = result["performance_metrics"]
    assert metrics["sharpe_ratio"] == 2.1
    assert metrics["max_drawdown"] == 11.5
    assert metrics["win_rate"] == 57.0
    assert metrics["profit_factor"] == 2.3

    # Verify no errors
    assert result.get("compilation_error") is None or result["compilation_error"] == ""

    print(f"\nBacktest parsing test passed:")
    print(f"  - Sharpe: {metrics['sharpe_ratio']}")
    print(f"  - Drawdown: {metrics['max_drawdown']}%")
    print(f"  - Win Rate: {metrics['win_rate']}%")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_backtest_syntax_error_handling(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.5.3: Syntax errors are handled correctly.

    Validates:
    - Syntax error is detected
    - Error message is clear
    - Retry is triggered
    """
    # Mock syntax error response
    mock_mcp_servers["backtest-server"].call_tool = AsyncMock(
        side_effect=lambda tool, args: json.dumps({
            "backtest_id": "bt-syntax-error",
            "status": "failed",
            "error_type": "syntax_error",
            "error_message": "IndentationError: unexpected indent"
        })
    )

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "def broken(\n    return 1",  # Syntax error
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Syntax_Error_Test",
        "deployment_config": None
    }

    result = await quant_code_agent.backtest_node(state)

    # Verify error was captured
    assert result.get("compilation_error") is not None
    error_msg = result["compilation_error"].lower()

    # Should mention syntax error
    assert "syntax" in error_msg or "error" in error_msg

    # Verify retry count incremented
    assert result["retry_count"] == 1

    print(f"\nSyntax error handling test passed:")
    print(f"  - Error detected: {result['compilation_error'][:50]}...")
    print(f"  - Retry count: {result['retry_count']}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_backtest_data_error_handling(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.5.4: Data errors are handled correctly.

    Validates:
    - Insufficient data is detected
    - Invalid symbol is detected
    - Clear error message provided
    """
    # Mock data error response
    mock_mcp_servers["backtest-server"].call_tool = AsyncMock(
        side_effect=lambda tool, args: json.dumps({
            "backtest_id": "bt-data-error",
            "status": "failed",
            "error_type": "data_error",
            "error_message": "Insufficient data: Only 25 data points available, minimum 50 required"
        })
    )

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "strategy code",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Data_Error_Test",
        "deployment_config": None
    }

    result = await quant_code_agent.backtest_node(state)

    # Verify data error captured
    assert result.get("compilation_error") is not None
    error_msg = result["compilation_error"].lower()

    assert "data" in error_msg

    print(f"\nData error handling test passed:")
    print(f"  - Error: {result['compilation_error'][:60]}...")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_backtest_timeout_handling(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.5.5: Timeout errors are handled correctly.

    Validates:
    - Timeout is detected
    - Timeout error message provided
    - No infinite waiting
    """
    # Mock timeout error
    mock_mcp_servers["backtest-server"].call_tool = AsyncMock(
        side_effect=lambda tool, args: json.dumps({
            "backtest_id": "bt-timeout",
            "status": "failed",
            "error_type": "timeout_error",
            "error_message": "Backtest exceeded 300 second timeout limit"
        })
    )

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "strategy code",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Timeout_Test",
        "deployment_config": None
    }

    result = await quant_code_agent.backtest_node(state)

    # Verify timeout captured
    assert result.get("compilation_error") is not None
    error_msg = result["compilation_error"].lower()

    assert "timeout" in error_msg

    print(f"\nTimeout handling test passed:")
    print(f"  - Timeout detected")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_analyze_node_evaluates_metrics_accurately(
    quant_code_agent,
    e2e_helper
):
    """
    Test 15.5.6: analyze_node evaluates metrics against thresholds.

    Validates:
    - Sharpe ratio > 1.5 threshold
    - Max drawdown < 20% threshold
    - Win rate > 45% threshold
    - @primal assigned only when all pass
    """
    # Test case 1: All pass
    result1 = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.5, max_drawdown=12.0, win_rate=58.0
    )
    state1 = {
        "trd_content": "test", "current_code": "code",
        "backtest_results": result1, "retry_count": 0, "is_primal": False
    }
    analysis1 = await quant_code_agent.analyze_node(state1)
    assert analysis1["is_primal"] is True

    # Test case 2: Sharpe fails
    result2 = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=1.2, max_drawdown=12.0, win_rate=58.0
    )
    state2 = {
        "trd_content": "test", "current_code": "code",
        "backtest_results": result2, "retry_count": 0, "is_primal": False
    }
    analysis2 = await quant_code_agent.analyze_node(state2)
    assert analysis2["is_primal"] is False
    assert "Sharpe" in analysis2["compilation_error"]

    # Test case 3: Drawdown fails
    result3 = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.5, max_drawdown=25.0, win_rate=58.0
    )
    state3 = {
        "trd_content": "test", "current_code": "code",
        "backtest_results": result3, "retry_count": 0, "is_primal": False
    }
    analysis3 = await quant_code_agent.analyze_node(state3)
    assert analysis3["is_primal"] is False
    assert "drawdown" in analysis3["compilation_error"].lower()

    # Test case 4: Win rate fails
    result4 = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.5, max_drawdown=12.0, win_rate=40.0
    )
    state4 = {
        "trd_content": "test", "current_code": "code",
        "backtest_results": result4, "retry_count": 0, "is_primal": False
    }
    analysis4 = await quant_code_agent.analyze_node(state4)
    assert analysis4["is_primal"] is False
    assert ("Win rate" in analysis4["compilation_error"] or
            "win_rate" in analysis4["compilation_error"])

    print(f"\nMetrics evaluation test passed:")
    print(f"  - All thresholds validated correctly")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_backtest_status_polling(
    quant_code_agent,
    sample_trd,
    mock_mcp_servers
):
    """
    Test 15.5.7: Backtest status polling works correctly.

    Validates:
    - Status is checked after submission
    - Queued/running backtests are polled
    - Completed results are retrieved
    """
    poll_count = [0]

    async def mock_poll(tool, args):
        poll_count[0] += 1
        if tool == "run_backtest":
            return json.dumps({"backtest_id": "bt-poll-001", "status": "queued"})
        elif tool == "get_backtest_status":
            if poll_count[0] < 3:
                return json.dumps({
                    "backtest_id": "bt-poll-001",
                    "status": "running",
                    "progress_percent": 50.0
                })
            else:
                return json.dumps({
                    "backtest_id": "bt-poll-001",
                    "status": "completed",
                    "progress_percent": 100.0,
                    "result": e2e_helper.create_mock_backtest_result()
                })

    mock_mcp_servers["backtest-server"].call_tool = AsyncMock(side_effect=mock_poll)

    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "strategy code",
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Polling_Test",
        "deployment_config": None
    }

    result = await quant_code_agent.backtest_node(state)

    # Verify polling occurred
    assert poll_count[0] > 1, "Should poll for status"

    # Verify final result retrieved
    assert result["backtest_results"] is not None

    print(f"\nStatus polling test passed:")
    print(f"  - Polled {poll_count[0]} times")
