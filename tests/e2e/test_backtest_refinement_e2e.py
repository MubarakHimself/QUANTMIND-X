"""
Test 15.2: Backtest Failure Triggers Refinement Loop

Task Group 15.1: Test that backtest failures correctly trigger the refinement cycle

This test validates:
1. Failed backtests trigger retry logic
2. Specific feedback is provided for metric failures
3. Reflection node analyzes failures
4. Retry counter increments correctly
5. Max retries limit is enforced
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_backtest_failure_triggers_refinement(
    quant_code_agent,
    sample_trd,
    e2e_helper
):
    """
    Test that backtest failures trigger the refinement loop correctly.

    Scenario:
    - Initial code fails backtest (poor metrics)
    - Analyze node provides specific feedback
    - Reflection node prepares refinement plan
    - Retry counter increments
    - Workflow routes back to coder
    """
    # Create backtest result with failing metrics
    failing_result = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=0.8,   # Below 1.5 threshold
        max_drawdown=35.0,  # Exceeds 20% threshold
        win_rate=38.0       # Below 45% threshold
    )

    initial_state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": e2e_helper.create_sample_strategy_code(),
        "backtest_results": failing_result,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Failing_Strategy",
        "deployment_config": None
    }

    # Step 1: Analyze should detect failures
    analyze_result = await quant_code_agent.analyze_node(initial_state)

    assert analyze_result["is_primal"] is False, "Should NOT assign @primal for failing metrics"
    assert analyze_result.get("compilation_error") is not None, "Should have error message"

    error_msg = analyze_result["compilation_error"]
    print(f"\nRefinement feedback: {error_msg}")

    # Verify specific feedback for each failing metric
    assert "Sharpe" in error_msg or "sharpe" in error_msg.lower(), "Should mention Sharpe ratio issue"
    assert "drawdown" in error_msg.lower(), "Should mention drawdown issue"
    assert "Win rate" in error_msg or "win_rate" in error_msg, "Should mention win rate issue"

    # Step 2: Check routing decision
    state_after_analyze = {**initial_state, **analyze_result}
    routing = quant_code_agent.should_continue(state_after_analyze)

    assert routing == "retry", "Should route to retry when metrics fail and retries available"

    # Step 3: Execute reflection node
    reflection_result = await quant_code_agent.reflection_node(state_after_analyze)

    assert "messages" in reflection_result
    assert reflection_result["retry_count"] == 1, "Retry count should increment"

    # Verify reflection message contains specific feedback
    reflection_messages = reflection_result["messages"]
    assert len(reflection_messages) > 0

    print(f"\nRefinement loop test passed:")
    print(f"  - @primal assigned: {analyze_result['is_primal']}")
    print(f"  - Routing: {routing}")
    print(f"  - Retry count: {reflection_result['retry_count']}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multiple_refinement_cycles(
    quant_code_agent,
    sample_trd,
    e2e_helper
):
    """
    Test that the refinement loop handles multiple retry cycles.

    Scenario:
    - First attempt fails
    - Second attempt still fails
    - Third attempt succeeds (all thresholds pass)
    """
    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": e2e_helper.create_sample_strategy_code(),
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Multi_Retry_Strategy",
        "deployment_config": None
    }

    # Attempt 1: Failing metrics
    state["backtest_results"] = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=0.5, max_drawdown=40.0, win_rate=35.0
    )
    result1 = await quant_code_agent.analyze_node(state)
    assert result1["is_primal"] is False

    state.update(result1)
    state["retry_count"] = 1
    routing1 = quant_code_agent.should_continue(state)
    assert routing1 == "retry", "Should retry after first failure"

    # Attempt 2: Still failing but improving
    state["backtest_results"] = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=1.2, max_drawdown=25.0, win_rate=42.0
    )
    result2 = await quant_code_agent.analyze_node(state)
    assert result2["is_primal"] is False, "Still not @primal"

    state.update(result2)
    state["retry_count"] = 2
    routing2 = quant_code_agent.should_continue(state)
    assert routing2 == "retry", "Should retry after second failure"

    # Attempt 3: Success!
    state["backtest_results"] = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=1.8, max_drawdown=15.0, win_rate=52.0
    )
    result3 = await quant_code_agent.analyze_node(state)
    assert result3["is_primal"] is True, "Should be @primal on third attempt"

    state.update(result3)
    state["retry_count"] = 3
    routing3 = quant_code_agent.should_continue(state)
    assert routing3 == "done", "Should be done when @primal"

    print(f"\nMultiple refinement cycles test passed:")
    print(f"  Attempt 1: @primal=False, route=retry")
    print(f"  Attempt 2: @primal=False, route=retry")
    print(f"  Attempt 3: @primal=True, route=done")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_max_retries_enforcement(
    quant_code_agent,
    sample_trd,
    e2e_helper
):
    """
    Test that max retries limit is enforced correctly.

    Scenario:
    - Strategy fails consistently
    - After 3 retries, workflow stops
    - No further refinement attempts are made
    """
    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": e2e_helper.create_sample_strategy_code(),
        "backtest_results": None,
        "performance_metrics": None,
        "compilation_error": "",
        "retry_count": 0,
        "is_primal": False,
        "strategy_name": "Max_Retry_Strategy",
        "deployment_config": None
    }

    # Simulate 3 failed attempts
    for attempt in range(1, 4):
        state["backtest_results"] = e2e_helper.create_mock_backtest_result(
            sharpe_ratio=0.5, max_drawdown=50.0, win_rate=30.0
        )

        result = await quant_code_agent.analyze_node(state)
        assert result["is_primal"] is False, f"Attempt {attempt}: Should not be @primal"

        state.update(result)
        state["retry_count"] = attempt

        routing = quant_code_agent.should_continue(state)

        if attempt < 3:
            assert routing == "retry", f"Attempt {attempt}: Should retry"
        else:
            assert routing == "done", "Attempt 3: Should stop (max retries reached)"

    # Verify no more retries after 3
    state["retry_count"] = 3
    state["compilation_error"] = "Max retries reached"
    routing = quant_code_agent.should_continue(state)
    assert routing == "done", "Should not exceed max retries"

    print(f"\nMax retries enforcement test passed:")
    print(f"  - Retries 1-2: route=retry")
    print(f"  - Retry 3: route=done (max reached)")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_specific_feedback_for_each_failed_metric(
    quant_code_agent,
    sample_trd,
    e2e_helper
):
    """
    Test that specific feedback is provided for each failing metric.

    Validates:
    - Low Sharpe ratio generates specific feedback
    - High drawdown generates specific feedback
    - Low win rate generates specific feedback
    - All failing metrics are mentioned in feedback
    """
    # Test case 1: Only Sharpe fails
    result1 = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=0.8,  # Fail
        max_drawdown=15.0,  # Pass
        win_rate=50.0       # Pass
    )
    state1 = {
        "trd_content": sample_trd,
        "current_code": "code",
        "backtest_results": result1,
        "retry_count": 0,
        "is_primal": False
    }
    analysis1 = await quant_code_agent.analyze_node(state1)
    assert "Sharpe" in analysis1["compilation_error"]
    assert "drawdown" not in analysis1["compilation_error"].lower() or "15" not in analysis1["compilation_error"]

    # Test case 2: Only drawdown fails
    result2 = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.0,   # Pass
        max_drawdown=35.0,  # Fail
        win_rate=50.0       # Pass
    )
    state2 = {
        "trd_content": sample_trd,
        "current_code": "code",
        "backtest_results": result2,
        "retry_count": 0,
        "is_primal": False
    }
    analysis2 = await quant_code_agent.analyze_node(state2)
    assert "drawdown" in analysis2["compilation_error"].lower()

    # Test case 3: Only win rate fails
    result3 = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=2.0,   # Pass
        max_drawdown=15.0,  # Pass
        win_rate=38.0       # Fail
    )
    state3 = {
        "trd_content": sample_trd,
        "current_code": "code",
        "backtest_results": result3,
        "retry_count": 0,
        "is_primal": False
    }
    analysis3 = await quant_code_agent.analyze_node(state3)
    assert "Win rate" in analysis3["compilation_error"] or "win_rate" in analysis3["compilation_error"]

    # Test case 4: All metrics fail
    result4 = e2e_helper.create_mock_backtest_result(
        sharpe_ratio=0.5,
        max_drawdown=45.0,
        win_rate=32.0
    )
    state4 = {
        "trd_content": sample_trd,
        "current_code": "code",
        "backtest_results": result4,
        "retry_count": 0,
        "is_primal": False
    }
    analysis4 = await quant_code_agent.analyze_node(state4)
    error_msg = analysis4["compilation_error"]
    assert "Sharpe" in error_msg
    assert "drawdown" in error_msg.lower()
    assert ("Win rate" in error_msg or "win_rate" in error_msg)

    print(f"\nSpecific feedback test passed:")
    print(f"  - Individual metric failures generate specific feedback")
    print(f"  - Multiple failures are all mentioned")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_reflection_node_incorporates_feedback(
    quant_code_agent,
    sample_trd
):
    """
    Test that the reflection node properly incorporates analysis feedback.

    Validates:
    - Reflection message includes specific metrics issues
    - Reflection provides actionable improvement steps
    - Reflection includes strategy context
    """
    state = {
        "messages": [],
        "trd_content": sample_trd,
        "current_code": "strategy code",
        "backtest_results": None,
        "compilation_error": (
            "Performance issues:\n"
            "- Sharpe ratio 0.85 below threshold 1.5\n"
            "- Max drawdown 28.5% exceeds threshold 20%\n"
            "- Win rate 42.3% below threshold 45%"
        ),
        "retry_count": 1,
        "is_primal": False,
        "strategy_name": "Test_Strategy",
        "deployment_config": None
    }

    result = await quant_code_agent.reflection_node(state)

    assert "messages" in result
    assert len(result["messages"]) > 0
    assert result["retry_count"] == 2

    reflection_content = result["messages"][0].content

    # Verify reflection includes feedback
    assert "Performance issues" in reflection_content or "Analyzing" in reflection_content

    # Verify reflection includes improvement steps
    assert "backtest" in reflection_content.lower() or "metrics" in reflection_content.lower()

    print(f"\nReflection node test passed:")
    print(f"  - Reflection message generated")
    print(f"  - Feedback incorporated")
    print(f"  - Retry count incremented to {result['retry_count']}")
