"""
TRD Generation Chain for Analyst Agent.

This module implements the LangChain chain for generating Technical Requirements Documents (TRD)
from extracted trading concepts, knowledge base articles, and user input.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

from ..prompts.templates import TRD_GENERATION_PROMPT


async def generation_chain(
    extracted_concepts: dict,
    kb_articles: list[dict],
    user_answers: dict = None,
    source: str = "",
    llm: BaseLanguageModel = None
) -> str:
    """
    Generate a Technical Requirements Document (TRD) in markdown format.

    Args:
        extracted_concepts: Dictionary containing extracted trading concepts
        kb_articles: List of knowledge base articles with metadata and content
        user_answers: Dictionary of user-provided answers to fill gaps (optional)
        source: Source identifier for the strategy
        llm: Language model to use for generation

    Returns:
        str: Complete markdown-formatted TRD document
    """
    if llm is None:
        raise ValueError("LLM must be provided for TRD generation")

    # Prepare inputs for the prompt
    current_time = datetime.now().isoformat()

    # Format KB articles for the prompt
    kb_articles_formatted = []
    for article in kb_articles:
        kb_articles_formatted.append(
            f"- **[{article.get('title', 'Untitled')}]({article.get('url', '#')})**\n"
            f"  - **Relevance:** {article.get('relevance', 'N/A')}\n"
            f"  - **Key Insight:** {article.get('key_insight', 'N/A')}\n"
            f"  - **Category:** {article.get('category', 'N/A')}"
        )

    kb_articles_text = "\n".join(kb_articles_formatted) if kb_articles_formatted else "No relevant articles found"

    # Format user answers if provided
    user_answers_text = json.dumps(user_answers, indent=2) if user_answers else "No user answers provided"

    # Prepare the prompt with all inputs
    prompt = TRD_GENERATION_PROMPT.format(
        extracted_concepts=json.dumps(extracted_concepts, indent=2),
        user_answers=user_answers_text,
        kb_articles=kb_articles_text,
        source=source,
        timestamp=current_time
    )

    # Generate the TRD using the LLM
    response = await llm.ainvoke(prompt)

    # Extract the markdown content from the response
    trd_content = response.content

    # Ensure the response is properly formatted markdown
    if not trd_content.startswith("---"):
        # If the response doesn't start with YAML frontmatter, add it
        strategy_name = extracted_concepts.get("strategy_name", "Unnamed Strategy")
        trd_content = f"""---
strategy_name: "{strategy_name}"
source: "{source}"
generated_at: "{current_time}"
status: "draft"
version: "1.0"
analyst_version: "1.0"
kb_collection: "analyst_kb"
kb_articles_count: {len(kb_articles)}
---

{trd_content}"""

    return trd_content


# For backward compatibility and testing
def generate_trd_sync(
    extracted_concepts: dict,
    kb_articles: list[dict],
    user_answers: dict = None,
    source: str = "",
    llm=None
) -> str:
    """
    Synchronous wrapper for TRD generation (for testing and compatibility).

    Note: This is a synchronous wrapper around the async function.
    In production, use the async version with proper async/await patterns.
    """
    import asyncio

    # Run the async function synchronously
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(
        generation_chain(extracted_concepts, kb_articles, user_answers, source, llm)
    )