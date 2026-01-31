"""
Unit Tests for LangGraph Agent Workflows

Tests agent state transitions and workflow execution.

**Validates: Requirements 8.10, 8.11**
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.state import AgentState, AnalystState, QuantCodeState, ExecutorState, RouterState
from src.agents.analyst import (
    research_node, extraction_node, synthesis_node, validation_node,
    create_analyst_graph, compile_analyst_graph
)
from src.agents.quantcode import (
    planning_node, coding_node, backtesting_node, analysis_node, reflection_node,
    create_quantcode_graph, compile_quantcode_graph
)
from src.agents.executor import (
    deployment_node, compilation_node, validation_node as executor_validation_node, monitoring_node,
    create_executor_graph, compile_executor_graph
)
from src.agents.router import (
    classify_task_node, delegate_node,
    create_router_graph, compile_router_graph
)


class TestAgentState:
    """Tests for agent state definitions."""
    
    def test_agent_state_structure(self):
        """Test AgentState has required fields."""
        state = AgentState(
            messages=[HumanMessage(content="test")],
            current_task="test_task",
            workspace_path="workspaces/test",
            context={},
            memory_namespace=("memories", "test")
        )
        
        assert "messages" in state
        assert "current_task" in state
        assert "workspace_path" in state
        assert "context" in state
        assert "memory_namespace" in state
    
    def test_analyst_state_structure(self):
        """Test AnalystState has required fields."""
        state = AnalystState(
            messages=[],
            current_task=None,
            workspace_path="workspaces/analyst",
            context={},
            memory_namespace=("memories", "analyst"),
            research_query="test query",
            extracted_data=None,
            synthesis_result=None,
            validation_status=None
        )
        
        assert "research_query" in state
        assert "extracted_data" in state
        assert "synthesis_result" in state
        assert "validation_status" in state


class TestAnalystAgent:
    """Tests for Analyst agent workflow."""
    
    def test_research_node(self):
        """Test research node execution."""
        state = AnalystState(
            messages=[],
            current_task="research",
            workspace_path="workspaces/analyst",
            context={},
            memory_namespace=("memories", "analyst"),
            research_query="market analysis",
            extracted_data=None,
            synthesis_result=None,
            validation_status=None
        )
        
        result = research_node(state)
        
        assert "messages" in result
        assert len(result["messages"]) > 0
        assert "context" in result
        assert "research_results" in result["context"]
    
    def test_extraction_node(self):
        """Test extraction node execution."""
        state = AnalystState(
            messages=[],
            current_task="extraction",
            workspace_path="workspaces/analyst",
            context={"research_results": {"query": "test"}},
            memory_namespace=("memories", "analyst"),
            research_query="test",
            extracted_data=None,
            synthesis_result=None,
            validation_status=None
        )
        
        result = extraction_node(state)
        
        assert "extracted_data" in result
        assert result["extracted_data"] is not None
    
    def test_analyst_graph_creation(self):
        """Test Analyst graph can be created."""
        graph = create_analyst_graph()
        
        assert graph is not None
        assert hasattr(graph, 'nodes')
    
    def test_analyst_graph_compilation(self):
        """Test Analyst graph can be compiled."""
        compiled_graph = compile_analyst_graph()
        
        assert compiled_graph is not None


class TestQuantCodeAgent:
    """Tests for QuantCode agent workflow."""
    
    def test_planning_node(self):
        """Test planning node execution."""
        state = QuantCodeState(
            messages=[],
            current_task="planning",
            workspace_path="workspaces/quant",
            context={},
            memory_namespace=("memories", "quantcode"),
            strategy_plan=None,
            code_implementation=None,
            backtest_results=None,
            analysis_report=None,
            reflection_notes=None
        )
        
        result = planning_node(state)
        
        assert "strategy_plan" in result
        assert result["strategy_plan"] is not None
    
    def test_backtesting_node(self):
        """Test backtesting node execution."""
        state = QuantCodeState(
            messages=[],
            current_task="backtesting",
            workspace_path="workspaces/quant",
            context={},
            memory_namespace=("memories", "quantcode"),
            strategy_plan="test plan",
            code_implementation="test code",
            backtest_results=None,
            analysis_report=None,
            reflection_notes=None
        )
        
        result = backtesting_node(state)
        
        assert "backtest_results" in result
        assert result["backtest_results"] is not None
        assert "total_trades" in result["backtest_results"]
    
    def test_quantcode_graph_creation(self):
        """Test QuantCode graph can be created."""
        graph = create_quantcode_graph()
        
        assert graph is not None
    
    def test_quantcode_graph_compilation(self):
        """Test QuantCode graph can be compiled."""
        compiled_graph = compile_quantcode_graph()
        
        assert compiled_graph is not None


class TestExecutorAgent:
    """Tests for Executor agent workflow."""
    
    def test_deployment_node(self):
        """Test deployment node execution."""
        state = ExecutorState(
            messages=[],
            current_task="deployment",
            workspace_path="workspaces/executor",
            context={},
            memory_namespace=("memories", "executor"),
            deployment_manifest=None,
            compilation_status=None,
            validation_results=None,
            monitoring_data=None
        )
        
        result = deployment_node(state)
        
        assert "deployment_manifest" in result
        assert result["deployment_manifest"] is not None
    
    def test_monitoring_node(self):
        """Test monitoring node execution."""
        state = ExecutorState(
            messages=[],
            current_task="monitoring",
            workspace_path="workspaces/executor",
            context={},
            memory_namespace=("memories", "executor"),
            deployment_manifest={"ea_name": "test"},
            compilation_status="SUCCESS",
            validation_results={"status": "PASSED"},
            monitoring_data=None
        )
        
        result = monitoring_node(state)
        
        assert "monitoring_data" in result
        assert result["monitoring_data"] is not None
    
    def test_executor_graph_creation(self):
        """Test Executor graph can be created."""
        graph = create_executor_graph()
        
        assert graph is not None
    
    def test_executor_graph_compilation(self):
        """Test Executor graph can be compiled."""
        compiled_graph = compile_executor_graph()
        
        assert compiled_graph is not None


class TestRouterAgent:
    """Tests for Router agent workflow."""
    
    def test_classify_task_node_research(self):
        """Test task classification for research tasks."""
        state = RouterState(
            messages=[HumanMessage(content="Analyze market trends")],
            current_task="routing",
            workspace_path="workspaces",
            context={},
            memory_namespace=("memories", "router"),
            task_type=None,
            target_agent=None,
            delegation_history=[]
        )
        
        result = classify_task_node(state)
        
        assert result["task_type"] == "research"
        assert result["target_agent"] == "analyst"
    
    def test_classify_task_node_strategy(self):
        """Test task classification for strategy tasks."""
        state = RouterState(
            messages=[HumanMessage(content="Create a new trading strategy")],
            current_task="routing",
            workspace_path="workspaces",
            context={},
            memory_namespace=("memories", "router"),
            task_type=None,
            target_agent=None,
            delegation_history=[]
        )
        
        result = classify_task_node(state)
        
        assert result["task_type"] == "strategy_development"
        assert result["target_agent"] == "quantcode"
    
    def test_classify_task_node_deployment(self):
        """Test task classification for deployment tasks."""
        state = RouterState(
            messages=[HumanMessage(content="Deploy the EA to production")],
            current_task="routing",
            workspace_path="workspaces",
            context={},
            memory_namespace=("memories", "router"),
            task_type=None,
            target_agent=None,
            delegation_history=[]
        )
        
        result = classify_task_node(state)
        
        assert result["task_type"] == "deployment"
        assert result["target_agent"] == "executor"
    
    def test_router_graph_creation(self):
        """Test Router graph can be created."""
        graph = create_router_graph()
        
        assert graph is not None
    
    def test_router_graph_compilation(self):
        """Test Router graph can be compiled."""
        compiled_graph = compile_router_graph()
        
        assert compiled_graph is not None


class TestAgentIntegration:
    """Integration tests for complete agent workflows."""
    
    def test_analyst_workflow_execution(self):
        """Test complete Analyst workflow execution."""
        from src.agents.analyst import run_analyst_workflow
        
        result = run_analyst_workflow(
            research_query="Test market analysis",
            workspace_path="workspaces/analyst",
            memory_namespace=("memories", "analyst", "test")
        )
        
        assert result is not None
        assert "validation_status" in result
        assert result["validation_status"] in ["PASSED", "FAILED"]
    
    def test_quantcode_workflow_execution(self):
        """Test complete QuantCode workflow execution."""
        from src.agents.quantcode import run_quantcode_workflow
        
        result = run_quantcode_workflow(
            strategy_request="Create momentum strategy",
            workspace_path="workspaces/quant",
            memory_namespace=("memories", "quantcode", "test")
        )
        
        assert result is not None
        assert "backtest_results" in result
    
    def test_executor_workflow_execution(self):
        """Test complete Executor workflow execution."""
        from src.agents.executor import run_executor_workflow
        
        result = run_executor_workflow(
            deployment_request="Deploy EA",
            workspace_path="workspaces/executor",
            memory_namespace=("memories", "executor", "test")
        )
        
        assert result is not None
        assert "monitoring_data" in result
    
    def test_router_workflow_execution(self):
        """Test complete Router workflow execution."""
        from src.agents.router import run_router_workflow
        
        result = run_router_workflow(
            task_request="Analyze market conditions",
            workspace_path="workspaces",
            memory_namespace=("memories", "router", "test")
        )
        
        assert result is not None
        assert "target_agent" in result
        assert result["target_agent"] in ["analyst", "quantcode", "executor"]
