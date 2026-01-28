"""
QuantMindX Agent Framework - RAG Usage Examples

Demonstrates RAG (Retrieval-Augmented Generation) with agents.
"""

import os
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from kb.client import ChromaKBClient
from agent import (
    create_analyst_agent,
    create_analyst_agent_with_rag,
    enable_rag_for_agent,
    ChromaDBRetriever,
    create_retriever_tool,
    create_rag_chain,
    RAGEnabledAgent
)


def example_1_basic_agent_with_rag():
    """
    Example 1: Create agent with RAG enabled from scratch.
    """
    print("=" * 60)
    print("Example 1: Basic Agent with RAG")
    print("=" * 60)

    # Initialize KB
    kb_client = ChromaKBClient()

    # Create RAG-enabled agent
    agent = create_analyst_agent_with_rag(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        kb_client=kb_client,
        collection="analyst_kb"
    )

    # Ask with RAG
    response = agent.ask_with_rag("What is Kelly criterion?", use_retrieval=True, k=3)
    print(f"Response: {response[:200]}...")


def example_2_enable_rag_existing_agent():
    """
    Example 2: Enable RAG on existing agent.
    """
    print("\n" + "=" * 60)
    print("Example 2: Enable RAG on Existing Agent")
    print("=" * 60)

    # Initialize KB
    kb_client = ChromaKBClient()

    # Create regular agent
    agent = create_analyst_agent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        kb_client=kb_client
    )

    # Enable RAG
    rag_agent = enable_rag_for_agent(agent, kb_client, collection="analyst_kb")

    # Now can use ask_with_rag
    response = rag_agent.ask_with_rag("Explain ORB strategy", use_retrieval=True)
    print(f"Response: {response[:200]}...")


def example_3_retriever_as_tool():
    """
    Example 3: Use retriever as a tool for LangGraph agents.
    """
    print("\n" + "=" * 60)
    print("Example 3: Retriever as Tool")
    print("=" * 60)

    from agent import create_analyst_agent
    from rich.console import Console
    console = Console()

    # Initialize
    kb_client = ChromaKBClient()
    agent = create_analyst_agent(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        kb_client=kb_client
    )

    # Create retrieval tool
    retrieval_tool = create_retriever_tool(kb_client, collection="analyst_kb")

    # Add to agent
    agent.add_tool(retrieval_tool)

    console.print("[green]✓ Added retrieval tool to agent[/green]")

    # Agent can now use the tool when needed
    # (The agent will decide when to call it based on user query)


def example_4_rag_chain():
    """
    Example 4: Create standalone RAG chain.
    """
    print("\n" + "=" * 60)
    print("Example 4: Standalone RAG Chain")
    print("=" * 60)

    from langchain_openai import ChatOpenAI

    # Initialize
    kb_client = ChromaKBClient()

    # Create retriever
    retriever = ChromaDBRetriever(
        kb_client=kb_client,
        collection="analyst_kb",
        search_kwargs={"k": 3}
    )

    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create RAG chain
    rag_chain = create_rag_chain(
        retriever=retriever,
        llm=llm,
        system_prompt="You are a trading strategy expert. Use context to answer."
    )

    # Invoke chain
    response = rag_chain.invoke("What is the optimal position sizing formula?")
    print(f"Response: {response[:200]}...")


def example_5_langgraph_agent_with_retrieval():
    """
    Example 5: Create LangGraph ReAct agent with retrieval.
    """
    print("\n" + "=" * 60)
    print("Example 5: LangGraph Agent with Retrieval")
    print("=" * 60)

    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool

    # Initialize
    kb_client = ChromaKBClient()

    # Create retriever
    retriever = ChromaDBRetriever(kb_client, collection="analyst_kb")

    # Create LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # Define additional tools
    @tool
    def calculate_risk(capital: float, risk_percent: float) -> str:
        """Calculate risk amount."""
        risk = capital * (risk_percent / 100)
        return f"Risk amount: ${risk:.2f}"

    # Create agent with retrieval
    agent = create_agent_with_retrieval(
        llm=llm,
        tools=[calculate_risk],
        retriever=retriever,
        system_prompt="You are a trading strategy analyst."
    )

    # Invoke agent (it will use retrieval tool when needed)
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What does Kelly criterion say about position sizing?"}]
    })

    print(f"Response: {response}")


def example_6_custom_retrieval():
    """
    Example 6: Custom retrieval with filtering.
    """
    print("\n" + "=" * 60)
    print("Example 6: Custom Retrieval")
    print("=" * 60)

    from kb.client import ChromaKBClient

    # Initialize
    kb_client = ChromaKBClient()

    # Create agent with RAG
    agent = create_analyst_agent_with_rag(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        kb_client=kb_client
    )

    # Direct retriever access
    retriever = agent.retriever

    # Get documents for a query
    docs = retriever.get_relevant_documents("risk management")

    print(f"Retrieved {len(docs)} documents:")
    for doc in docs[:3]:
        print(f"  - {doc.metadata.get('title', 'Untitled')}")
        print(f"    Score: {doc.metadata.get('score', 0):.2f}")


def example_7_rag_with_chat_mode():
    """
    Example 7: RAG in chat mode with conversation history.
    """
    print("\n" + "=" * 60)
    print("Example 7: RAG Chat Mode")
    print("=" * 60)

    from kb.client import ChromaKBClient

    # Initialize
    kb_client = ChromaKBClient()

    # Create RAG agent
    agent = create_analyst_agent_with_rag(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        kb_client=kb_client
    )

    # Multi-turn conversation
    questions = [
        "What is ORB strategy?",
        "How do I manage risk with it?",
        "What indicators should I use?"
    ]

    for q in questions:
        print(f"\nUser: {q}")
        response = agent.ask_with_rag(q, use_retrieval=True, k=2)
        print(f"Agent: {response[:150]}...")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("QuantMindX RAG Examples")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        print("\n[yellow]Warning: OPENROUTER_API_KEY not set[/yellow]")
        print("Set it with: export OPENROUTER_API_KEY='your-key'")
        return

    # Run examples (comment out as needed)
    # example_1_basic_agent_with_rag()
    # example_2_enable_rag_existing_agent()
    # example_3_retriever_as_tool()
    # example_4_rag_chain()
    # example_5_langgraph_agent_with_retrieval()
    # example_6_custom_retrieval()
    # example_7_rag_with_chat_mode()

    print("\n[green]✓ All RAG examples available[/green]")
    print("[dim]Uncomment examples in main() to run them[/dim]")


if __name__ == "__main__":
    main()
