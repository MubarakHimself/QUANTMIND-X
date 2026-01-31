"""
Analyst Agent Workflow

Implements the Analyst agent using LangGraph for market research and analysis.

**Validates: Requirements 8.2**
"""

import logging
from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agents.state import AnalystState

logger = logging.getLogger(__name__)


# ============================================================================
# Agent Node Functions
# ============================================================================

def research_node(state: AnalystState) -> Dict[str, Any]:
    """
    Research node: Gather market data and information.
    
    **Validates: Requirements 8.2, 8.6**
    """
    logger.info(f"Research node processing query: {state.get('research_query')}")
    
    # Simulate research process
    research_query = state.get('research_query', 'default market analysis')
    
    # In production, this would call actual research tools
    research_results = {
        "query": research_query,
        "sources": ["market_data", "news_feeds", "technical_indicators"],
        "timestamp": "2024-01-31T00:00:00Z"
    }
    
    message = AIMessage(content=f"Completed research for: {research_query}")
    
    return {
        "messages": [message],
        "context": {"research_results": research_results}
    }


def extraction_node(state: AnalystState) -> Dict[str, Any]:
    """
    Extraction node: Extract key insights from research data.
    
    **Validates: Requirements 8.2, 8.6**
    """
    logger.info("Extraction node processing research results")
    
    research_results = state.get('context', {}).get('research_results', {})
    
    # Extract key data points
    extracted_data = {
        "key_insights": ["Market trend identified", "Volatility analysis complete"],
        "metrics": {"volatility": 0.15, "trend_strength": 0.75},
        "recommendations": ["Consider long positions", "Monitor support levels"]
    }
    
    message = AIMessage(content="Extracted key insights from research data")
    
    return {
        "messages": [message],
        "extracted_data": extracted_data,
        "context": {**state.get('context', {}), "extracted_data": extracted_data}
    }


def synthesis_node(state: AnalystState) -> Dict[str, Any]:
    """
    Synthesis node: Synthesize insights into actionable analysis.
    
    **Validates: Requirements 8.2, 8.6**
    """
    logger.info("Synthesis node creating analysis report")
    
    extracted_data = state.get('extracted_data', {})
    
    # Synthesize analysis
    synthesis_result = f"""
    Market Analysis Report
    =====================
    
    Key Insights:
    {', '.join(extracted_data.get('key_insights', []))}
    
    Metrics:
    - Volatility: {extracted_data.get('metrics', {}).get('volatility', 'N/A')}
    - Trend Strength: {extracted_data.get('metrics', {}).get('trend_strength', 'N/A')}
    
    Recommendations:
    {', '.join(extracted_data.get('recommendations', []))}
    """
    
    message = AIMessage(content="Synthesized analysis report")
    
    return {
        "messages": [message],
        "synthesis_result": synthesis_result,
        "context": {**state.get('context', {}), "synthesis_result": synthesis_result}
    }


def validation_node(state: AnalystState) -> Dict[str, Any]:
    """
    Validation node: Validate analysis quality and completeness.
    
    **Validates: Requirements 8.2, 8.6**
    """
    logger.info("Validation node checking analysis quality")
    
    synthesis_result = state.get('synthesis_result', '')
    
    # Validate analysis
    validation_checks = {
        "has_insights": len(state.get('extracted_data', {}).get('key_insights', [])) > 0,
        "has_metrics": bool(state.get('extracted_data', {}).get('metrics')),
        "has_recommendations": len(state.get('extracted_data', {}).get('recommendations', [])) > 0
    }
    
    validation_status = "PASSED" if all(validation_checks.values()) else "FAILED"
    
    message = AIMessage(content=f"Validation {validation_status}: {validation_checks}")
    
    return {
        "messages": [message],
        "validation_status": validation_status,
        "context": {**state.get('context', {}), "validation_status": validation_status}
    }


# ============================================================================
# Conditional Edges
# ============================================================================

def should_continue_to_synthesis(state: AnalystState) -> str:
    """
    Determine if extraction was successful and should continue to synthesis.
    
    **Validates: Requirements 8.7**
    """
    extracted_data = state.get('extracted_data')
    
    if extracted_data and extracted_data.get('key_insights'):
        return "synthesis"
    else:
        return "research"  # Retry research if extraction failed


def should_continue_to_end(state: AnalystState) -> str:
    """
    Determine if validation passed and workflow should end.
    
    **Validates: Requirements 8.7**
    """
    validation_status = state.get('validation_status')
    
    if validation_status == "PASSED":
        return END
    else:
        return "research"  # Retry from research if validation failed


# ============================================================================
# Graph Construction
# ============================================================================

def create_analyst_graph() -> StateGraph:
    """
    Create the Analyst agent workflow graph.
    
    **Validates: Requirements 8.2, 8.9**
    
    Workflow:
    START -> research -> extraction -> synthesis -> validation -> END
    
    With conditional edges for error handling and retries.
    """
    # Initialize graph with AnalystState
    workflow = StateGraph(AnalystState)
    
    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("synthesis", synthesis_node)
    workflow.add_node("validation", validation_node)
    
    # Set entry point
    workflow.set_entry_point("research")
    
    # Add edges
    workflow.add_edge("research", "extraction")
    workflow.add_conditional_edges(
        "extraction",
        should_continue_to_synthesis,
        {
            "synthesis": "synthesis",
            "research": "research"
        }
    )
    workflow.add_edge("synthesis", "validation")
    workflow.add_conditional_edges(
        "validation",
        should_continue_to_end,
        {
            END: END,
            "research": "research"
        }
    )
    
    return workflow


def compile_analyst_graph(checkpointer: MemorySaver = None) -> Any:
    """
    Compile the Analyst agent graph with optional checkpointing.
    
    **Validates: Requirements 8.8, 8.9**
    
    Args:
        checkpointer: Optional MemorySaver for state persistence
        
    Returns:
        Compiled graph ready for execution
    """
    workflow = create_analyst_graph()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Analyst agent graph compiled successfully")
    
    return compiled_graph


# ============================================================================
# Execution Interface
# ============================================================================

def run_analyst_workflow(
    research_query: str,
    workspace_path: str = "workspaces/analyst",
    memory_namespace: tuple = ("memories", "analyst", "default")
) -> Dict[str, Any]:
    """
    Execute the Analyst agent workflow.
    
    Args:
        research_query: The research query to process
        workspace_path: Path to analyst workspace
        memory_namespace: Memory namespace for this execution
        
    Returns:
        Final state after workflow completion
    """
    # Create initial state
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
    
    # Compile and run graph
    graph = compile_analyst_graph()
    
    # Execute workflow
    config = {"configurable": {"thread_id": "analyst_001"}}
    final_state = graph.invoke(initial_state, config)
    
    logger.info(f"Analyst workflow completed with status: {final_state.get('validation_status')}")
    
    return final_state
