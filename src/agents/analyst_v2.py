"""
Analyst Agent Workflow with ToolNode Integration

Implements the Analyst agent using LangGraph with dynamic tool calling for
market research, NPRD parsing, and TRD generation.

**Validates: Requirements 8.2**
"""

import logging
from typing import Dict, Any, List, Optional, Union, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from src.agents.state import AnalystState
from src.agents.llm_provider import get_analyst_llm

logger = logging.getLogger(__name__)


# ============================================================================
# Tool Definitions for Analyst Agent
# ============================================================================

@tool
def research_market_data(
    query: str,
    symbols: List[str] = None,
    timeframe: str = "H1"
) -> Dict[str, Any]:
    """
    Research market data and gather information.

    Args:
        query: Research query
        symbols: List of symbols to analyze
        timeframe: Data timeframe

    Returns:
        Research results with market data
    """
    symbols = symbols or ["EURUSD", "GBPUSD"]
    return {
        "query": query,
        "sources": ["market_data", "news_feeds", "technical_indicators"],
        "symbols": symbols,
        "data_points": 500,
        "timestamp": "2024-01-31T00:00:00Z"
    }


@tool
def extract_insights(
    research_data: Dict[str, Any],
    focus_areas: List[str] = None
) -> Dict[str, Any]:
    """
    Extract key insights from research data.

    Args:
        research_data: Research data to analyze
        focus_areas: Areas to focus extraction on

    Returns:
        Extracted insights and metrics
    """
    return {
        "key_insights": ["Market trend identified", "Volatility analysis complete"],
        "metrics": {"volatility": 0.15, "trend_strength": 0.75},
        "recommendations": ["Consider long positions", "Monitor support levels"],
        "confidence": 0.85
    }


@tool
def parse_nprd(
    nprd_content: str,
    format: str = "markdown"
) -> Dict[str, Any]:
    """
    Parse a Natural Product Requirements Document.

    Args:
        nprd_content: Raw NPRD content
        format: Input format (markdown, json, yaml)

    Returns:
        Parsed NPRD structure
    """
    # Simplified parsing - would use actual parser in production
    return {
        "title": "Parsed Strategy",
        "version": "1.0.0",
        "sections": {
            "overview": {"content": "Strategy overview..."},
            "objectives": {"items": ["Profitable trading", "Risk management"]},
            "trading_logic": {"content": "Entry and exit rules..."},
            "risk_management": {"content": "Risk parameters..."}
        },
        "is_valid": True
    }


@tool
def validate_nprd(
    parsed_nprd: Dict[str, Any],
    strict: bool = False
) -> Dict[str, Any]:
    """
    Validate an NPRD structure and content.

    Args:
        parsed_nprd: Parsed NPRD to validate
        strict: Enable strict validation mode

    Returns:
        Validation result with issues and suggestions
    """
    required_sections = ["overview", "objectives", "trading_logic", "risk_management"]
    issues = []

    for section in required_sections:
        if section not in parsed_nprd.get("sections", {}):
            issues.append(f"Missing required section: {section}")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "completeness_score": 100 - (len(issues) * 25),
        "suggestions": ["Add more detail to entry conditions"] if len(issues) == 0 else []
    }


@tool
def generate_trd(
    nprd_content: str,
    target_framework: str = "mql5",
    include_code_hints: bool = True
) -> Dict[str, Any]:
    """
    Generate a Technical Requirements Document from NPRD.

    Args:
        nprd_content: NPRD content
        target_framework: Target framework (mql5, python)
        include_code_hints: Include implementation hints

    Returns:
        Generated TRD
    """
    return {
        "title": "TRD: Strategy Implementation",
        "version": "1.0.0",
        "target_framework": target_framework,
        "sections": {
            "architecture": {
                "pattern": "event_driven",
                "components": ["SignalGenerator", "OrderManager", "RiskManager"]
            },
            "components": [
                {"name": "SignalGenerator", "purpose": "Generate trading signals"},
                {"name": "OrderManager", "purpose": "Manage order execution"},
                {"name": "RiskManager", "purpose": "Enforce risk limits"}
            ],
            "code_hints": {
                "signal_generation": "// Implement signal logic here"
            }
        },
        "content": "# Technical Requirements Document\n\n..."
    }


@tool
def analyze_backtest(
    backtest_id: str,
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze backtest results and calculate metrics.

    Args:
        backtest_id: ID of backtest to analyze
        metrics: Specific metrics to calculate

    Returns:
        Performance analysis
    """
    return {
        "backtest_id": backtest_id,
        "performance": {
            "total_trades": 245,
            "win_rate": 60.4,
            "profit_factor": 1.76,
            "sharpe_ratio": 1.45,
            "max_drawdown": 12.3,
            "cagr": 18.5
        },
        "recommendations": ["Optimize entry timing", "Consider additional filters"]
    }


@tool
def compare_strategies(
    backtest_ids: List[str],
    primary_metric: str = "sharpe_ratio"
) -> Dict[str, Any]:
    """
    Compare multiple trading strategies.

    Args:
        backtest_ids: List of backtest IDs to compare
        primary_metric: Primary metric for ranking

    Returns:
        Comparison results
    """
    strategies = [
        {"backtest_id": bid, "sharpe_ratio": 1.2 + i * 0.1, "rank": i + 1}
        for i, bid in enumerate(backtest_ids)
    ]
    strategies.sort(key=lambda x: x.get(primary_metric, 0), reverse=True)

    return {
        "strategies": strategies,
        "winner": strategies[0]["backtest_id"] if strategies else None,
        "comparison_metric": primary_metric
    }


@tool
def generate_optimization_report(
    backtest_id: str,
    format: str = "markdown"
) -> Dict[str, Any]:
    """
    Generate an optimization report for a backtest.

    Args:
        backtest_id: Backtest ID
        format: Report format (markdown, html, json)

    Returns:
        Generated report
    """
    return {
        "backtest_id": backtest_id,
        "format": format,
        "report": f"# Optimization Report\n\n## {backtest_id}\n\n...",
        "recommendations": [
            "Tighten stop losses during low volatility",
            "Implement dynamic position sizing"
        ]
    }


# List of all Analyst tools
ANALYST_TOOLS = [
    research_market_data,
    extract_insights,
    parse_nprd,
    validate_nprd,
    generate_trd,
    analyze_backtest,
    compare_strategies,
    generate_optimization_report,
]


# ============================================================================
# Agent Node Functions
# ============================================================================

def agent_node(state: MessagesState) -> Dict[str, Any]:
    """
    Main agent node that uses LLM to decide which tools to call.
    Uses the project's provider config (OpenRouter/Zhipu) with fallbacks.
    """
    llm_with_tools = get_analyst_llm(tools=ANALYST_TOOLS)

    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


# ============================================================================
# Graph Construction with ToolNode
# ============================================================================

def create_analyst_graph_with_tools() -> StateGraph:
    """
    Create the Analyst agent workflow graph with ToolNode.

    Workflow:
    START -> agent -> (tool_condition) -> tools -> agent -> ...
                    -> (no tools) -> END
    """
    tool_node = ToolNode(ANALYST_TOOLS)

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    workflow.add_edge("tools", "agent")

    return workflow


def create_analyst_graph_legacy() -> StateGraph:
    """
    Create the legacy Analyst agent workflow graph.

    Kept for backward compatibility.
    """
    workflow = StateGraph(AnalystState)

    # Legacy hardcoded nodes
    def research_node(state: AnalystState) -> Dict[str, Any]:
        logger.info(f"Research node processing: {state.get('research_query')}")
        return {
            "messages": [AIMessage(content=f"Research completed")],
            "context": {"research_results": {"query": state.get('research_query')}}
        }

    def extraction_node(state: AnalystState) -> Dict[str, Any]:
        logger.info("Extraction node processing")
        return {
            "messages": [AIMessage(content="Extracted insights")],
            "extracted_data": {"key_insights": ["Trend identified"]},
            "context": {**state.get('context', {}), "extracted_data": {}}
        }

    def synthesis_node(state: AnalystState) -> Dict[str, Any]:
        logger.info("Synthesis node creating report")
        return {
            "messages": [AIMessage(content="Analysis report generated")],
            "synthesis_result": "Market Analysis Report...",
            "context": {**state.get('context', {}), "synthesis_result": ""}
        }

    def validation_node(state: AnalystState) -> Dict[str, Any]:
        logger.info("Validation node checking quality")
        return {
            "messages": [AIMessage(content="Validation PASSED")],
            "validation_status": "PASSED",
            "context": {**state.get('context', {}), "validation_status": "PASSED"}
        }

    workflow.add_node("research", research_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("validation", validation_node)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "extraction")
    workflow.add_edge("extraction", "synthesis")
    workflow.add_edge("synthesis", "validation")
    workflow.add_edge("validation", END)

    return workflow


def compile_analyst_graph(
    checkpointer: MemorySaver = None,
    use_tool_node: bool = True
) -> Any:
    """
    Compile the Analyst agent graph.

    Args:
        checkpointer: Optional memory checkpointer
        use_tool_node: Use ToolNode architecture if True

    Returns:
        Compiled graph
    """
    if use_tool_node:
        workflow = create_analyst_graph_with_tools()
    else:
        workflow = create_analyst_graph_legacy()

    if checkpointer is None:
        checkpointer = MemorySaver()

    compiled_graph = workflow.compile(checkpointer=checkpointer)
    logger.info(f"Analyst agent graph compiled (ToolNode={use_tool_node})")

    return compiled_graph


def run_analyst_workflow(
    research_query: str,
    workspace_path: str = "workspaces/analyst",
    memory_namespace: tuple = ("memories", "analyst", "default"),
    use_tool_node: bool = True
) -> Dict[str, Any]:
    """
    Execute the Analyst agent workflow.

    Args:
        research_query: The research query
        workspace_path: Path to workspace
        memory_namespace: Memory namespace tuple
        use_tool_node: Use ToolNode architecture if True

    Returns:
        Final state after workflow completion
    """
    if use_tool_node:
        initial_state = {
            "messages": [HumanMessage(content=research_query)]
        }
    else:
        initial_state = AnalystState(
            messages=[HumanMessage(content=research_query)],
            current_task="market_analysis",
            workspace_path=workspace_path,
            context={},
            memory_namespace=memory_namespace,
            research_query=research_query,
            extracted_data=None,
            synthesis_result=None,
            validation_status=None
        )

    graph = compile_analyst_graph(use_tool_node=use_tool_node)
    config = {"configurable": {"thread_id": "analyst_001"}}
    final_state = graph.invoke(initial_state, config)

    logger.info(f"Analyst workflow completed")

    return final_state


# =============================================================================
# Factory-Based Agent Creation (Phase 4)
# =============================================================================

def create_analyst_from_config(config: AgentConfig) -> CompiledAgent:
    """
    Create an Analyst agent from configuration using the factory pattern.
    
    This function integrates the existing agent with the new factory-based
    system while maintaining backward compatibility.
    
    Args:
        config: AgentConfig instance with analyst configuration
        
    Returns:
        CompiledAgent instance
        
    Raises:
        ValueError: If config is not for an analyst agent
    """
    import warnings
    
    # Validate agent type
    if config.agent_type != "analyst":
        raise ValueError(f"Expected agent_type='analyst', got '{config.agent_type}'")
    
    # Issue deprecation warning for legacy approach
    warnings.warn(
        "create_analyst_from_config() uses the factory pattern. "
        "Consider using AgentFactory directly for new code.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Import factory components
    from src.agents.factory import get_factory
    from src.agents.di_container import get_container
    from src.agents.observers.logging_observer import LoggingObserver
    from src.agents.observers.prometheus_observer import PrometheusObserver
    
    # Get or create container and factory
    container = get_container()
    factory = get_factory(container)
    
    # Add default observers if none exist
    if not container.get_observers():
        container.add_observer(LoggingObserver())
        container.add_observer(PrometheusObserver())
    
    # Use factory to create agent
    agent = factory.create(config)
    
    logger.info(f"Created analyst agent from config: {config.agent_id}")
    
    return agent


def create_analyst_agent(
    agent_id: str = "analyst_001",
    name: str = "Analyst Agent",
    llm_model: str = "anthropic/claude-sonnet-4",
    temperature: float = 0.0,
    **kwargs
) -> CompiledAgent:
    """
    Convenience function to create an Analyst agent with custom settings.
    
    Args:
        agent_id: Unique agent identifier
        name: Agent display name
        llm_model: LLM model to use
        temperature: LLM temperature
        **kwargs: Additional configuration options
        
    Returns:
        CompiledAgent instance
    """
    from src.agents.config import AgentConfig
    
    config = AgentConfig(
        agent_id=agent_id,
        agent_type="analyst",
        name=name,
        llm_model=llm_model,
        temperature=temperature,
        tools=[
            "research_market_data",
            "extract_insights",
            "parse_nprd",
            "validate_nprd",
            "generate_trd",
            "analyze_backtest",
            "compare_strategies",
            "generate_optimization_report",
        ],
        **kwargs
    )
    
    return create_analyst_from_config(config)
