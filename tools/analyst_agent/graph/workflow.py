"""LangGraph workflow for Analyst Agent.

This module implements the state machine workflow for converting
NPRD video outputs or strategy documents into structured TRD files.

The workflow uses LangGraph StateGraph with the following flow:
    extract -> search -> check_gaps -> [human_input | generate] -> review
"""

import logging
from typing import Dict, Any, Optional, Literal
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AnalystState, create_initial_state, validate_state
from .nodes import (
    extract_node,
    search_node,
    check_missing_node,
    human_input_node,
    generate_node,
    review_node,
    has_missing_info,
    should_continue,
)

logger = logging.getLogger(__name__)


def create_analyst_graph(
    auto_mode: bool = False,
    checkpoint: bool = True
) -> StateGraph:
    """Create the LangGraph workflow for Analyst Agent.

    The graph structure:
        extract (extract concepts from NPRD)
        -> search (query KB for relevant articles)
        -> check_gaps (identify gaps)
        -> [human_input (if gaps and !auto) OR generate (if complete or auto)]
        -> review (validate output)
        -> END

    Args:
        auto_mode: If True, skip human_input node and use defaults
        checkpoint: If True, enable checkpointing for state persistence

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    logger.info(f"Creating analyst graph (auto_mode={auto_mode}, checkpoint={checkpoint})")

    # Create state graph
    workflow = StateGraph(AnalystState)

    # Add all nodes
    workflow.add_node("extract", extract_node)
    workflow.add_node("search", search_node)
    workflow.add_node("check_gaps", check_missing_node)
    workflow.add_node("human_input", human_input_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("review", review_node)

    # Define edges (workflow sequence)
    workflow.set_entry_point("extract")

    # Sequential flow: extract -> search -> check_gaps
    workflow.add_edge("extract", "search")
    workflow.add_edge("search", "check_gaps")

    # Conditional edge: check_gaps -> human_input OR generate
    workflow.add_conditional_edges(
        "check_gaps",
        has_missing_info,
        {
            "human_input": "human_input",
            "generate": "generate"
        }
    )

    # After human_input (if used), go to generate
    workflow.add_edge("human_input", "generate")

    # After generate, go to review
    workflow.add_edge("generate", "review")

    # After review, end workflow
    workflow.add_edge("review", END)

    # Add optional checkpointing for persistence
    if checkpoint:
        checkpointer = MemorySaver()
        compiled = workflow.compile(
            checkpointer=checkpointer,
            interrupt_before=["human_input"] if not auto_mode else None
        )
    else:
        interrupt_before = ["human_input"] if not auto_mode else None
        compiled = workflow.compile(
            interrupt_before=interrupt_before
        )

    logger.info("Analyst graph created successfully")
    return compiled


async def run_analyst_workflow(
    nprd_data: Dict[str, Any],
    input_type: str = "nprd",
    auto_mode: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run the analyst workflow end-to-end.

    Args:
        nprd_data: Raw input data from NPRD or strategy document
        input_type: Type of input (nprd or strategy_doc)
        auto_mode: If True, skip human input and use defaults
        config: Optional configuration for workflow execution

    Returns:
        Dictionary containing:
        - success: bool - Whether workflow completed successfully
        - trd_output: str - Generated TRD content
        - trd_path: str - Path where TRD was saved
        - state: AnalystState - Final workflow state
        - errors: List[str] - Any errors that occurred
    """
    logger.info(f"Starting analyst workflow (input_type={input_type}, auto_mode={auto_mode})")

    # Initialize state
    state = create_initial_state(nprd_data, input_type)

    # Add auto_mode to metadata
    state["metadata"]["auto_mode"] = auto_mode

    # Validate initial state
    is_valid, errors = validate_state(state)
    if not is_valid:
        logger.error(f"Invalid initial state: {errors}")
        return {
            "success": False,
            "trd_output": "",
            "trd_path": "",
            "state": state,
            "errors": errors
        }

    try:
        # Create graph
        graph = create_analyst_graph(auto_mode=auto_mode)

        # Prepare config for execution
        thread_config = {
            "configurable": {
                "thread_id": f"analyst_{datetime.utcnow().timestamp()}"
            }
        }
        if config:
            thread_config.update(config)

        # Execute workflow
        logger.info("Executing workflow...")
        result = None
        async for event in graph.astream(state, thread_config):
            logger.debug(f"Workflow event: {event}")
            result = event

        # Extract final state from result
        final_state = None
        if result:
            # Get the last state update
            for state_update in result.values():
                final_state = state_update

        if not final_state:
            # Fallback to initial state with updated status
            final_state = state

        # Determine success
        success = final_state.get("status") == "complete"
        errors = final_state.get("errors", [])

        logger.info(f"Workflow completed: success={success}, status={final_state.get('status')}")

        return {
            "success": success,
            "trd_output": final_state.get("trd_output", ""),
            "trd_path": final_state.get("metadata", {}).get("trd_path", ""),
            "state": final_state,
            "errors": errors
        }

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "trd_output": "",
            "trd_path": "",
            "state": state,
            "errors": [f"Workflow execution error: {str(e)}"]
        }


async def run_analyst_workflow_interactive(
    nprd_data: Dict[str, Any],
    input_type: str = "nprd",
    user_input_callback: Optional[callable] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run analyst workflow in interactive mode with user input handling.

    This version handles the HITL (Human-in-the-Loop) interaction by
    pausing at the human_input node and collecting user responses
    via the provided callback.

    Args:
        nprd_data: Raw input data from NPRD or strategy document
        input_type: Type of input (nprd or strategy_doc)
        user_input_callback: Async function called to collect user input.
            Receives (missing_info: List[Dict]) -> Dict[str, str]
        config: Optional configuration for workflow execution

    Returns:
        Dictionary containing workflow results
    """
    logger.info("Starting interactive analyst workflow")

    # Initialize state
    state = create_initial_state(nprd_data, input_type)
    state["metadata"]["auto_mode"] = False

    try:
        # Create graph with interrupts
        graph = create_analyst_graph(auto_mode=False)

        # Thread config for checkpointing
        thread_config = {
            "configurable": {
                "thread_id": f"analyst_interactive_{datetime.utcnow().timestamp()}"
            }
        }
        if config:
            thread_config.update(config)

        # Execute workflow until first interrupt
        logger.info("Running workflow to human_input checkpoint...")
        result = await graph.ainvoke(state, thread_config)

        # Check if we're waiting for human input
        while result.get("status") == "prompting":
            missing_info = result.get("missing_info", [])

            if not missing_info:
                # No missing info, continue
                break

            logger.info(f"Collecting user input for {len(missing_info)} fields")

            # Collect user input via callback
            if user_input_callback:
                user_answers = await user_input_callback(missing_info)
                result["user_answers"].update(user_answers)
            else:
                # Default: skip all
                logger.warning("No user_input_callback provided, skipping all fields")
                result["user_answers"] = {
                    field["field_name"]: "SKIPPED"
                    for field in missing_info
                }

            # Continue workflow
            logger.info("Resuming workflow with user input...")
            result = await graph.ainvoke(result, thread_config)

        # Workflow complete
        success = result.get("status") == "complete"
        logger.info(f"Interactive workflow completed: success={success}")

        return {
            "success": success,
            "trd_output": result.get("trd_output", ""),
            "trd_path": result.get("metadata", {}).get("trd_path", ""),
            "state": result,
            "errors": result.get("errors", [])
        }

    except Exception as e:
        logger.error(f"Interactive workflow failed: {e}", exc_info=True)
        return {
            "success": False,
            "trd_output": "",
            "trd_path": "",
            "state": state,
            "errors": [f"Interactive workflow error: {str(e)}"]
        }


def get_workflow_graphviz() -> str:
    """Generate Graphviz DOT representation of the workflow.

    Returns:
        DOT format string for visualization
    """
    return '''
digraph AnalystWorkflow {
    rankdir=LR;
    node [shape=box, style=rounded];

    extract [label="Extract Concepts\\nExtract trading concepts\\nfrom NPRD data"];
    search [label="Search KB\\nQuery ChromaDB for\\nrelevant articles"];
    check_gaps [label="Check Gaps\\nIdentify gaps in\\nrequired fields"];
    human_input [label="Human Input\\nPrompt user for\\nmissing data", style="dashed"];
    generate [label="Generate TRD\\nCreate structured\\nmarkdown document"];
    review [label="Review\\nValidate output\\nand finalize"];

    extract -> search;
    search -> check_gaps;
    check_gaps -> human_input [label="if gaps & !auto"];
    check_gaps -> generate [label="if complete or auto"];
    human_input -> generate;
    generate -> review;
}
'''


# Export main components
__all__ = [
    "create_analyst_graph",
    "run_analyst_workflow",
    "run_analyst_workflow_interactive",
    "get_workflow_graphviz",
]
