"""
LangGraph Workflow Tests for Quant Code Agent

Task Group 7.1: Write 2-8 focused tests for LangGraph nodes and state transitions.

Tests cover:
- planning_node fetches coding standards and bad patterns
- backtest_node calls Backtest MCP correctly
- analyze_node evaluates metrics against thresholds
- retry logic triggers when metrics fail
- @primal tag is assigned when all thresholds pass
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from langchain_core.messages import AIMessage

from src.agents.implementations.quant_code import QuantCodeAgent


class TestLangGraphWorkflow:
    """Test LangGraph workflow nodes and state transitions."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = AsyncMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test response"))
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        """Create a QuantCodeAgent instance for testing."""
        # Set fake API key for testing
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_BASE_URL"] = "https://fake.openrouter.ai/api/v1"

        agent = QuantCodeAgent(git_repo_path="/tmp/test_git_assets")
        # Replace the LLM with mock
        agent.llm = mock_llm
        return agent

    @pytest.fixture
    def sample_state(self):
        """Create a sample CodeState for testing."""
        return {
            "messages": [],
            "trd_content": "Create a moving average crossover strategy",
            "current_code": "",
            "backtest_results": None,
            "compilation_error": "",
            "retry_count": 0,
            "is_primal": False,
            "strategy_name": "ma_crossover",
            "deployment_config": None
        }

    @pytest.mark.asyncio
    async def test_planning_node_fetches_standards(self, agent, sample_state):
        """
        Test that planning_node fetches coding standards and bad patterns from Assets Hub.

        Task Group 7.5: planning_node should call Assets Hub tools:
        - get_coding_standards
        - get_bad_patterns
        - get_algorithm_template (if applicable)
        """
        # Mock MCP tool calls for Assets Hub
        mock_standards = {
            "project_name": "QuantMindX",
            "version": "1.0",
            "conventions": ["Use Python 3.10+", "Add type hints"]
        }

        mock_bad_patterns = [
            {
                "pattern": "Hardcoded values",
                "severity": "high",
                "solution": "Use configuration constants"
            }
        ]

        # Patch the call_mcp_tool method
        with patch.object(agent, 'call_mcp_tool', new_callable=AsyncMock) as mock_mcp:
            # Setup mock returns
            mock_mcp.side_effect = [
                mock_standards,  # First call: get_coding_standards
                mock_bad_patterns  # Second call: get_bad_patterns
            ]

            # Execute planning_node
            result = await agent.planning_node(sample_state)

            # Verify MCP tools were called
            assert agent.call_mcp_tool.call_count >= 1

            # Verify result contains messages
            assert "messages" in result
            assert len(result["messages"]) > 0

    @pytest.mark.asyncio
    async def test_backtest_node_calls_mcp(self, agent, sample_state):
        """
        Test that backtest_node calls Backtest MCP with current_code.

        Task Group 7.3: backtest_node should:
        - Call Backtest MCP run_backtest with current_code
        - Parse BacktestResult and store in state
        - Handle errors: syntax_error, data_error, runtime_error, timeout_error
        """
        # Add code to state
        sample_state["current_code"] = """
import backtrader as bt

class MAStrategy(bt.Strategy):
    def __init__(self):
        self.sma_short = bt.indicators.SMA(period=10)
        self.sma_long = bt.indicators.SMA(period=20)
"""

        # Mock Backtest MCP response
        mock_backtest_result = {
            "backtest_id": "test-123",
            "status": "completed",
            "result": {
                "backtest_id": "test-123",
                "status": "success",
                "metrics": {
                    "sharpe_ratio": 1.8,
                    "max_drawdown": 15.0,
                    "total_return": 25.0,
                    "win_rate": 55.0,
                    "profit_factor": 1.9
                },
                "equity_curve": [{"timestamp": "2024-01-01", "value": 10000}],
                "trade_log": [{"entry": "2024-01-01", "exit": "2024-01-02", "pnl": 100}],
                "logs": "Backtest completed successfully",
                "execution_time_seconds": 45.0
            }
        }

        # Patch the call_mcp_tool method
        with patch.object(agent, 'call_mcp_tool', new_callable=AsyncMock) as mock_mcp:
            mock_mcp.return_value = mock_backtest_result

            # Execute backtest_node
            result = await agent.backtest_node(sample_state)

            # Verify Backtest MCP was called
            mock_mcp.assert_called_once()
            call_args = mock_mcp.call_args
            assert call_args[0][0] == "backtest-server"  # server_name
            assert call_args[0][1] == "run_backtest"  # tool_name

            # Verify result contains backtest_results
            assert "backtest_results" in result
            assert result["backtest_results"]["metrics"]["sharpe_ratio"] == 1.8

    @pytest.mark.asyncio
    async def test_backtest_node_handles_syntax_error(self, agent, sample_state):
        """
        Test that backtest_node handles syntax errors correctly.

        Task Group 7.3: Handle syntax_error from Backtest MCP.
        """
        sample_state["current_code"] = "def invalid_syntax("

        # Mock syntax error response
        with patch.object(agent, 'call_mcp_tool', new_callable=AsyncMock) as mock_mcp:
            mock_mcp.side_effect = ValueError("[SYN_001] Syntax error at line 1: Missing colon")

            # Execute backtest_node
            result = await agent.backtest_node(sample_state)

            # Verify error is handled
            assert "compilation_error" in result
            assert "syntax" in result["compilation_error"].lower()

    @pytest.mark.asyncio
    async def test_analyze_node_evaluates_metrics(self, agent, sample_state):
        """
        Test that analyze_node evaluates metrics against thresholds.

        Task Group 7.4: analyze_node should:
        - Evaluate metrics vs thresholds:
          - sharpe_ratio > 1.5
          - max_drawdown < 20%
          - win_rate > 45%
        - Set is_primal = True if all thresholds pass
        - Generate specific feedback for failed metrics
        """
        # Add backtest results
        sample_state["backtest_results"] = {
            "metrics": {
                "sharpe_ratio": 1.8,
                "max_drawdown": 15.0,
                "total_return": 25.0,
                "win_rate": 55.0,
                "profit_factor": 1.9
            }
        }

        # Execute analyze_node
        result = await agent.analyze_node(sample_state)

        # Verify is_primal is True (all thresholds pass)
        assert result["is_primal"] == True

    @pytest.mark.asyncio
    async def test_analyze_node_generates_feedback_for_failed_metrics(self, agent, sample_state):
        """
        Test that analyze_node generates specific feedback for failed metrics.

        Task Group 7.4: Generate specific feedback for failed metrics
        (e.g., "Sharpe 0.8 below threshold 1.5").
        """
        # Add backtest results with failing metrics
        sample_state["backtest_results"] = {
            "metrics": {
                "sharpe_ratio": 0.8,  # Below threshold 1.5
                "max_drawdown": 25.0,  # Above threshold 20%
                "win_rate": 40.0,  # Below threshold 45%
                "profit_factor": 1.2
            }
        }

        # Execute analyze_node
        result = await agent.analyze_node(sample_state)

        # Verify is_primal is False
        assert result["is_primal"] == False

        # Verify compilation_error contains specific feedback
        assert "compilation_error" in result
        error_msg = result["compilation_error"].lower()
        assert "sharpe" in error_msg
        assert "0.8" in error_msg or "below" in error_msg

    @pytest.mark.asyncio
    async def test_retry_logic_with_primal_false(self, agent, sample_state):
        """
        Test that retry logic triggers when is_primal = False and retry_count < 3.

        Task Group 7.7: Update should_continue to:
        - Check is_primal flag
        - Route to reflection_node if is_primal = False and retry_count < 3
        """
        sample_state["is_primal"] = False
        sample_state["retry_count"] = 1
        sample_state["compilation_error"] = "Sharpe ratio too low"

        # Execute should_continue
        result = agent.should_continue(sample_state)

        # Verify route is "retry"
        assert result == "retry"

    @pytest.mark.asyncio
    async def test_retry_logic_exhausted(self, agent, sample_state):
        """
        Test that retry logic stops when retry_count >= 3.

        Task Group 7.7: Stop retrying when max retries reached.
        """
        sample_state["is_primal"] = False
        sample_state["retry_count"] = 3  # Max retries reached
        sample_state["compilation_error"] = "Sharpe ratio too low"

        # Execute should_continue
        result = agent.should_continue(sample_state)

        # Verify route is "done" (no more retries)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_primal_tag_assignment_on_success(self, agent, sample_state):
        """
        Test that @primal tag is assigned when all thresholds pass.

        Task Group 7.4: Set is_primal = True if all thresholds pass.
        """
        # Add backtest results with passing metrics
        sample_state["backtest_results"] = {
            "metrics": {
                "sharpe_ratio": 2.5,  # Above 1.5
                "max_drawdown": 10.0,  # Below 20%
                "win_rate": 60.0,  # Above 45%
                "profit_factor": 2.2
            }
        }

        # Execute analyze_node
        result = await agent.analyze_node(sample_state)

        # Verify is_primal is True
        assert result["is_primal"] == True

        # Verify no compilation_error
        assert result.get("compilation_error") is None or result.get("compilation_error") == ""

    @pytest.mark.asyncio
    async def test_reflection_node_provides_feedback(self, agent, sample_state):
        """
        Test that reflection_node provides specific feedback for refinement.

        Task Group 7.7: Pass specific feedback from analyze_node to reflection_node.
        """
        sample_state["compilation_error"] = "Sharpe ratio 0.8 below threshold 1.5. Max drawdown 25% exceeds 20%."

        # Execute reflection_node
        result = await agent.reflection_node(sample_state)

        # Verify reflection contains error feedback
        assert "messages" in result
        assert len(result["messages"]) > 0

        # Verify feedback message mentions the error
        message_content = result["messages"][0].content
        assert "sharpe" in message_content.lower() or "drawdown" in message_content.lower()


class TestLangGraphIntegration:
    """Integration tests for LangGraph workflow."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = AsyncMock()
        # Return different content for different prompts
        async def mock_ainvoke(messages):
            content = str(messages)
            if "Generate the code" in content:
                return AIMessage(content="import backtrader as bt\n\nclass TestStrategy(bt.Strategy):\n    pass")
            return AIMessage(content="Test response")
        llm.ainvoke = mock_ainvoke
        return llm

    @pytest.fixture
    def agent(self, mock_llm):
        """Create a QuantCodeAgent instance for testing."""
        os.environ["OPENROUTER_API_KEY"] = "test-key"
        os.environ["OPENROUTER_BASE_URL"] = "https://fake.openrouter.ai/api/v1"

        agent = QuantCodeAgent(git_repo_path="/tmp/test_git_assets")
        # Replace the LLM with mock
        agent.llm = mock_llm
        return agent

    @pytest.mark.asyncio
    async def test_workflow_planning_to_coding(self, agent):
        """
        Test workflow transition: planning_node -> coding_node.

        Task Group 7.8: Verify state transitions follow the workflow.
        """
        initial_state = {
            "messages": [],
            "trd_content": "Create a moving average crossover strategy",
            "current_code": "",
            "backtest_results": None,
            "compilation_error": "",
            "retry_count": 0,
            "is_primal": False,
            "strategy_name": "ma_crossover",
            "deployment_config": None
        }

        # Execute planning_node
        planning_result = await agent.planning_node(initial_state)
        assert "messages" in planning_result

        # Update state and execute coding_node
        updated_state = {**initial_state, **planning_result}
        coding_result = await agent.coding_node(updated_state)

        # Verify coding node produces code
        assert "current_code" in coding_result
        assert len(coding_result["current_code"]) > 0

    @pytest.mark.asyncio
    async def test_workflow_complete_cycle(self, agent):
        """
        Test complete workflow cycle: planning -> coding -> backtest -> analyze.

        Task Group 7.8: Verify full workflow executes correctly.
        """
        initial_state = {
            "messages": [],
            "trd_content": "Create a simple RSI strategy",
            "current_code": "",
            "backtest_results": None,
            "performance_metrics": None,
            "compilation_error": "",
            "retry_count": 0,
            "is_primal": False,
            "strategy_name": "rsi_strategy",
            "deployment_config": None
        }

        # Mock all MCP calls and deployment
        with patch.object(agent, 'call_mcp_tool', new_callable=AsyncMock) as mock_mcp, \
             patch.object(agent, 'trigger_paper_trading_deployment', new_callable=AsyncMock) as mock_deploy:

            # Mock successful backtest result - return as JSON string
            import json
            mock_mcp.return_value = json.dumps({
                "backtest_id": "test-complete",
                "status": "completed",
                "result": {
                    "backtest_id": "test-complete",
                    "status": "success",
                    "metrics": {
                        "sharpe_ratio": 2.0,
                        "max_drawdown": 12.0,
                        "win_rate": 52.0,
                        "profit_factor": 1.8
                    }
                }
            })

            # Mock deployment success
            mock_deploy.return_value = True

            # Execute planning - update state with new values
            planning_result = await agent.planning_node(initial_state)
            state = {**initial_state, **planning_result}

            # Execute coding - update state with new values
            coding_result = await agent.coding_node(state)
            state = {**state, **coding_result}

            # Execute backtest - update state with new values
            backtest_result = await agent.backtest_node(state)
            state = {**state, **backtest_result}

            # Debug: Check what we have before analyze
            print(f"Before analyze - backtest_results: {state.get('backtest_results')}")
            print(f"Before analyze - compilation_error: {state.get('compilation_error')}")

            # Execute analyze - update state with new values
            analyze_result = await agent.analyze_node(state)
            state = {**state, **analyze_result}

            # Debug: Check what we have after analyze
            print(f"After analyze - is_primal: {state.get('is_primal')}")
            print(f"After analyze - compilation_error: {state.get('compilation_error')}")

            # Verify final state
            assert state["is_primal"] == True, f"Expected is_primal=True, got {state.get('is_primal')}. Error: {state.get('compilation_error')}"
            assert state["backtest_results"] is not None
            assert state["compilation_error"] is None or state["compilation_error"] == ""
