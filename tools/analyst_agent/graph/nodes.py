"""LangGraph nodes for Analyst Agent workflow."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .state import AnalystState

logger = logging.getLogger(__name__)


# =============================================================================
# Node: Extract Concepts
# =============================================================================

async def extract_node(state: AnalystState) -> AnalystState:
    """Extract trading concepts from NPRD data.

    This node analyzes the input transcript/content and extracts:
    - Strategy name
    - Entry conditions
    - Exit conditions
    - Time filters
    - Volatility filters
    - Indicators with settings
    - Trading concepts

    Args:
        state: Current workflow state

    Returns:
        Updated state with extracted concepts
    """
    logger.info("Starting concept extraction...")

    try:
        # Import extraction chain (will be implemented in chains module)
        from ..chains.extraction import extract_concepts

        # Get transcript from NPRD data
        transcript = ""
        if state["nprd_data"].get("type") == "nprd":
            # Concatenate all chunk transcripts
            chunks = state["nprd_data"].get("chunks", [])
            transcript = " ".join([
                chunk.get("transcript", "")
                for chunk in chunks
            ])
            # Add OCR text if available
            ocr_texts = [
                chunk.get("ocr_text", "")
                for chunk in chunks
                if chunk.get("ocr_text")
            ]
            if ocr_texts:
                transcript += " " + " ".join(ocr_texts)
        elif state["nprd_data"].get("type") == "strategy_doc":
            transcript = state["nprd_data"].get("content", "")

        if not transcript:
            raise ValueError("No transcript content found in input data")

        # Extract concepts using LLM
        concepts = await extract_concepts(transcript)

        # Update state
        state["concepts"] = concepts
        state["status"] = "extracting"

        logger.info(f"Extracted concepts: strategy_name={concepts.get('strategy_name', 'unknown')}")

    except Exception as e:
        logger.error(f"Concept extraction failed: {e}")
        state["errors"].append(f"Extraction error: {str(e)}")
        state["status"] = "error"

    return state


# =============================================================================
# Node: Search Knowledge Base
# =============================================================================

async def search_node(state: AnalystState) -> AnalystState:
    """Search knowledge base for relevant articles.

    Generates semantic queries from extracted concepts and searches
    the ChromaDB analyst_kb collection for relevant MQL5 articles.

    Args:
        state: Current workflow state

    Returns:
        Updated state with KB search results
    """
    logger.info("Searching knowledge base...")

    try:
        from ..kb.client import AnalystKBClient

        # Initialize KB client
        kb_client = AnalystKBClient()

        # Generate search queries from concepts
        queries = _generate_search_queries(state["concepts"])

        # Search KB and deduplicate results
        all_results = []
        seen_titles = set()

        for query in queries:
            results = kb_client.search(query, n_results=5)
            for result in results:
                title = result.get("title", "")
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    all_results.append(result)

        # Sort by relevance and limit to top 10
        all_results.sort(
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )
        state["kb_results"] = all_results[:10]
        state["status"] = "searching"

        logger.info(f"Found {len(state['kb_results'])} relevant articles")

    except Exception as e:
        logger.warning(f"KB search failed (continuing without KB): {e}")
        state["errors"].append(f"KB search warning: {str(e)}")
        state["kb_results"] = []
        state["status"] = "searching"

    return state


def _generate_search_queries(concepts: Dict[str, Any]) -> List[str]:
    """Generate semantic search queries from extracted concepts.

    Args:
        concepts: Extracted trading concepts

    Returns:
        List of search query strings
    """
    queries = []

    # Query from strategy name
    if concepts.get("strategy_name"):
        queries.append(f"{concepts['strategy_name']} trading strategy implementation")

    # Query from main trading concepts
    if concepts.get("trading_concepts"):
        top_concepts = concepts["trading_concepts"][:3]
        queries.append(f"MQL5 {' '.join(top_concepts)} expert advisor")

    # Query from indicators
    if concepts.get("indicators"):
        indicators = [ind.get("name", "") for ind in concepts["indicators"]]
        if indicators:
            queries.append(f"MQL5 {' '.join(indicators[:3])} indicator")

    # Query from entry/exit patterns
    entry_patterns = " ".join(concepts.get("entry_conditions", [])[:2])
    if entry_patterns:
        queries.append(f"MQL5 {entry_patterns} entry signal")

    return queries[:5]  # Limit to 5 queries


# =============================================================================
# Node: Check Missing Information
# =============================================================================

async def check_missing_node(state: AnalystState) -> AnalystState:
    """Check if required information is missing.

    Validates extracted concepts against required TRD fields and
    identifies gaps that need user input.

    Args:
        state: Current workflow state

    Returns:
        Updated state with missing_info populated
    """
    logger.info("Checking for missing information...")

    try:
        concepts = state["concepts"]
        missing = []

        # Define required fields with validation
        required_fields = {
            "entry_logic": {
                "fields": ["entry_trigger", "entry_confirmation"],
                "category": "critical",
                "examples": ["Price breaks above resistance", "RSI < 30 and price touches support"]
            },
            "exit_logic": {
                "fields": ["take_profit", "stop_loss"],
                "category": "critical",
                "examples": ["Take profit at 1.5x risk", "Stop loss at 15 pips"]
            },
            "risk_management": {
                "fields": ["position_sizing", "max_risk_per_trade"],
                "category": "critical",
                "examples": ["0.5% of account balance", "Risk-based on ATR"]
            },
            "filters": {
                "fields": ["time_filters"],
                "category": "important",
                "examples": ["London session only", "Avoid news hours"]
            },
            "indicators": {
                "fields": ["indicator_settings"],
                "category": "important",
                "examples": ["RSI period 14", "EMA 20 close"]
            }
        }

        # Check each category
        for category, config in required_fields.items():
            for field in config["fields"]:
                if not _has_field_value(concepts, field):
                    missing.append({
                        "field_name": field,
                        "category": config["category"],
                        "description": f"{field.replace('_', ' ').title()} for {category.replace('_', ' ')}",
                        "example": config["examples"][0],
                        "user_provided": None
                    })

        state["missing_info"] = missing
        state["status"] = "checking"

        logger.info(f"Found {len(missing)} missing fields")

    except Exception as e:
        logger.error(f"Gap check failed: {e}")
        state["errors"].append(f"Gap check error: {str(e)}")
        state["status"] = "error"

    return state


def _has_field_value(concepts: Dict[str, Any], field: str) -> bool:
    """Check if a field has a valid value in concepts.

    Args:
        concepts: Extracted concepts dictionary
        field: Field name to check

    Returns:
        True if field has valid value
    """
    # Check direct field
    if concepts.get(field):
        return True

    # Check nested fields
    field_parts = field.split("_")
    for part in field_parts:
        if any(part in str(v).lower() for v in concepts.values()):
            return True

    return False


# =============================================================================
# Node: Human Input
# =============================================================================

async def human_input_node(state: AnalystState) -> AnalystState:
    """Prompt user for missing information.

    This node implements HITL (Human-in-the-Loop) by interrupting
    the workflow and prompting for each missing field.

    Args:
        state: Current workflow state

    Returns:
        Updated state with user answers populated
    """
    logger.info("Prompting for user input...")

    try:
        # This is where LangGraph interrupt happens
        # The actual prompting will be handled by the CLI layer
        # This node just marks that we need input

        state["status"] = "prompting"

        # Note: In actual execution, LangGraph will interrupt here
        # and the CLI will collect input before resuming

        logger.info(f"Waiting for input on {len(state['missing_info'])} fields")

    except Exception as e:
        logger.error(f"Human input node failed: {e}")
        state["errors"].append(f"Input prompt error: {str(e)}")
        state["status"] = "error"

    return state


# =============================================================================
# Node: Generate TRD
# =============================================================================

async def generate_node(state: AnalystState) -> AnalystState:
    """Generate final TRD document.

    Combines all collected information into a structured TRD
    markdown document with proper formatting.

    Args:
        state: Current workflow state

    Returns:
        Updated state with trd_output populated
    """
    logger.info("Generating TRD document...")

    try:
        from ..chains.generation import generate_trd

        # Prepare generation context
        context = {
            "concepts": state["concepts"],
            "kb_results": state["kb_results"],
            "user_answers": state.get("user_answers", {}),
            "metadata": state.get("metadata", {})
        }

        # Generate TRD content
        trd_content = await generate_trd(context)

        state["trd_output"] = trd_content
        state["status"] = "generating"

        logger.info("TRD generated successfully")

    except Exception as e:
        logger.error(f"TRD generation failed: {e}")
        state["errors"].append(f"Generation error: {str(e)}")
        state["status"] = "error"

    return state


# =============================================================================
# Node: Review
# =============================================================================

async def review_node(state: AnalystState) -> AnalystState:
    """Review and validate generated TRD.

    Performs validation checks on the generated TRD to ensure
    all required sections are present and properly formatted.

    Args:
        state: Current workflow state

    Returns:
        Updated state with final status
    """
    logger.info("Reviewing generated TRD...")

    try:
        trd = state["trd_output"]
        issues = []

        # Validate required sections
        required_sections = [
            "## Overview",
            "## Entry Logic",
            "## Exit Logic",
            "## Filters",
            "## Indicators",
            "## Position Sizing & Risk Management"
        ]

        for section in required_sections:
            if section not in trd:
                issues.append(f"Missing section: {section}")

        # Validate YAML frontmatter
        if not trd.startswith("---"):
            issues.append("Missing YAML frontmatter")

        # Validate status based on issues
        if issues:
            state["status"] = "draft"
            state["errors"].extend([f"Review issue: {i}" for i in issues])
            logger.warning(f"TRD review found {len(issues)} issues")
        else:
            state["status"] = "complete"
            logger.info("TRD review passed")

    except Exception as e:
        logger.error(f"Review failed: {e}")
        state["errors"].append(f"Review error: {str(e)}")
        state["status"] = "error"

    return state


# =============================================================================
# Conditional Edge Functions
# =============================================================================

def has_missing_info(state: AnalystState) -> str:
    """Determine if workflow should prompt for missing info.

    Args:
        state: Current workflow state

    Returns:
        "human_input" if missing info exists, "generate" otherwise
    """
    missing_count = len(state.get("missing_info", []))

    if missing_count > 0 and not state.get("metadata", {}).get("auto_mode"):
        logger.info(f"Workflow branching to human_input ({missing_count} fields)")
        return "human_input"

    logger.info("Workflow branching directly to generate (no missing info or auto mode)")
    return "generate"


def should_continue(state: AnalystState) -> bool:
    """Determine if workflow should continue.

    Args:
        state: Current workflow state

    Returns:
        True if workflow should continue, False if complete or error
    """
    return state.get("status") not in {"complete", "error"}
