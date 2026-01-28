"""
LangChain KB Search Chain for QuantMindX Analyst Agent.

Provides a chain that generates search queries from extracted concepts
and retrieves relevant knowledge base articles using ChromaDB.
"""

from typing import List, Dict, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import JsonOutputParser

from ..prompts.templates import SEARCH_QUERY_PROMPT
from ..kb.client import ChromaKBClient, SearchResult


async def search_chain(
    extracted_concepts: dict,
    kb_client: ChromaKBClient,
    llm: BaseLanguageModel = None,
    max_results_per_query: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate search queries from extracted concepts and search the knowledge base.

    Args:
        extracted_concepts: Dictionary of extracted trading concepts
        kb_client: ChromaKBClient instance for KB operations
        llm: BaseLanguageModel for generating search queries (optional)
        max_results_per_query: Maximum results per search query

    Returns:
        List of aggregated KB articles with title, file_path, categories, score, preview

    The function:
    1. Generates 3-5 search queries using SEARCH_QUERY_PROMPT + LLM
    2. Searches the KB for each query
    3. Aggregates and deduplicates results
    4. Returns sorted by relevance score
    """
    if not llm:
        raise ValueError("LLM is required for search query generation")

    if not extracted_concepts:
        return []

    # Generate search queries using the prompt
    query_generator = SEARCH_QUERY_PROMPT | llm | JsonOutputParser()

    # Prepare input for query generation
    query_input = {
        "extracted_concepts": extracted_concepts,
        "keywords": extracted_concepts.get("mentioned_concepts", []) if "mentioned_concepts" in extracted_concepts else []
    }

    # Generate queries
    query_results = await query_generator.ainvoke(query_input)
    queries = query_results.get("queries", [])

    if not queries:
        return []

    # Search KB for each query
    all_results = []

    for query in queries:
        try:
            # Search with the current query
            search_results = kb_client.search(
                query=query,
                n=max_results_per_query
            )
            all_results.extend(search_results)
        except Exception as e:
            # Log error but continue with other queries
            import logging
            logging.getLogger(__name__).warning(f"Search failed for query '{query}': {e}")
            continue

    # Deduplicate results by title
    unique_results = {}
    for result in all_results:
        title = result.get("title", "")
        if title not in unique_results:
            unique_results[title] = result

    # Sort by relevance score (descending)
    sorted_results = sorted(
        unique_results.values(),
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    return sorted_results


# For backward compatibility and easier usage
def create_search_chain(llm: BaseLanguageModel) -> RunnableSequence:
    """
    Create a runnable search chain for LangChain integration.

    Args:
        llm: BaseLanguageModel for query generation

    Returns:
        RunnableSequence that can be invoked with extracted_concepts
    """
    from langchain_core.runnables import RunnableLambda

    async def chain_invoke(extracted_concepts: dict) -> List[Dict[str, Any]]:
        return await search_chain(extracted_concepts, ChromaKBClient(), llm)

    return RunnableLambda(chain_invoke)