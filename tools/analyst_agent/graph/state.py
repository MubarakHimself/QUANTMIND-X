"""State schema for Analyst Agent LangGraph workflow."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from typing_extensions import Required
import operator


class AnalystState(TypedDict):
    """State for analyst workflow graph.

    This state flows through all nodes in the LangGraph workflow,
    accumulating information as the analysis progresses.
    """

    # Input data
    nprd_data: Required[Dict[str, Any]]
    """Raw input data from NPRD JSON or strategy document."""

    # Extracted concepts
    concepts: Required[Dict[str, Any]]
    """Extracted trading concepts including:
    - strategy_name: str
    - entry_conditions: List[str]
    - exit_conditions: List[str]
    - filters: List[str]
    - time_filters: List[str]
    - indicators: List[Dict[str, str]]
    - trading_concepts: List[str]
    """

    # Knowledge base search results
    kb_results: Required[List[Dict[str, Any]]]
    """Search results from ChromaDB containing:
    - title: str
    - url: str
    - category: str
    - relevance_score: float
    - preview: str
    """

    # Missing information tracking
    missing_info: Required[List[Dict[str, Any]]]
    """List of missing fields with:
    - field_name: str
    - category: str (critical|important|optional)
    - description: str
    - example: str
    - user_provided: Optional[str]
    """

    # TRD output
    trd_output: Required[str]
    """Generated TRD markdown content."""

    # Workflow control
    status: Required[str]
    """Current status: parsing|extracting|searching|checking|prompting|generating|reviewing|complete|error"""

    # Additional tracking fields
    user_answers: Annotated[Dict[str, str], operator.add]
    """User-provided answers for missing information (merged across nodes)."""

    errors: List[str]
    """Accumulated errors during workflow execution."""

    metadata: Dict[str, Any]
    """Additional metadata like timestamps, sources, etc."""


def create_initial_state(
    nprd_data: Dict[str, Any],
    input_type: str = "nprd"
) -> AnalystState:
    """Create initial state for workflow execution.

    Args:
        nprd_data: Raw input data from NPRD or strategy document
        input_type: Type of input (nprd or strategy_doc)

    Returns:
        Initial AnalystState dictionary
    """
    from datetime import datetime

    return {
        "nprd_data": nprd_data,
        "concepts": {},
        "kb_results": [],
        "missing_info": [],
        "trd_output": "",
        "status": "parsing",
        "user_answers": {},
        "errors": [],
        "metadata": {
            "input_type": input_type,
            "created_at": datetime.utcnow().isoformat(),
            "source": nprd_data.get("source", "unknown")
        }
    }


def validate_state(state: AnalystState) -> tuple[bool, List[str]]:
    """Validate state consistency.

    Args:
        state: AnalystState to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    # Check required fields
    if not state.get("nprd_data"):
        errors.append("nprd_data is required")
    if "status" not in state:
        errors.append("status is required")
    if "concepts" not in state:
        errors.append("concepts is required")

    # Validate status value
    valid_statuses = {
        "parsing", "extracting", "searching", "checking",
        "prompting", "generating", "reviewing", "complete", "error"
    }
    if state.get("status") not in valid_statuses:
        errors.append(f"Invalid status: {state.get('status')}")

    return (len(errors) == 0, errors)
