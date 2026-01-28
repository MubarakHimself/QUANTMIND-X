"""
RAG (Retrieval-Augmented Generation) integration for QuantMindX agents.

Provides LangChain-compatible retrieval tools for agent use.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document


class ChromaDBRetriever(BaseRetriever):
    """
    LangChain-compatible retriever for ChromaDB knowledge base.

    Integrates with ChromaKBClient for semantic search.
    """

    def __init__(
        self,
        kb_client,
        collection: str = "analyst_kb",
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize retriever.

        Args:
            kb_client: ChromaKBClient instance
            collection: Collection name to search
            search_kwargs: Additional search parameters (k, score_threshold, etc.)
        """
        super().__init__()
        self.kb_client = kb_client
        self.collection = collection
        self.search_kwargs = search_kwargs or {"k": 3}

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for query.

        Args:
            query: Search query

        Returns:
            List of Document objects
        """
        try:
            results = self.kb_client.search(
                query,
                collection=self.collection,
                n=self.search_kwargs.get("k", 3)
            )

            documents = []
            for r in results:
                # Combine title and preview for document content
                content = f"**{r.get('title', 'Untitled')}**\n\n{r.get('preview', '')}"

                # Create Document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": r.get("title", ""),
                        "score": r.get("score", 0),
                        "categories": r.get("categories", ""),
                        "collection": self.collection
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            return []


def create_retriever_tool(kb_client, collection: str = "analyst_kb"):
    """
    Create a LangChain tool for KB retrieval.

    This tool can be added to agent's tool list for RAG.

    Args:
        kb_client: ChromaKBClient instance
        collection: Collection name

    Returns:
        LangCNn tool function
    """
    @tool
    def search_knowledge_base(query: str, k: int = 3) -> str:
        """
        Search the QuantMindX knowledge base for relevant trading articles.

        Args:
            query: Search query (keywords, concepts, questions)
            k: Number of results to return (default: 3)

        Returns:
            Formatted search results with titles and previews
        """
        try:
            results = kb_client.search(query, collection=collection, n=k)

            if not results:
                return f"No results found for query: {query}"

            output = f"Found {len(results)} relevant articles:\n\n"
            for i, r in enumerate(results, 1):
                output += f"{i}. **{r.get('title', 'Untitled')}**\n"
                output += f"   Relevance: {r.get('score', 0):.2f}\n"
                output += f"   {r.get('preview', '')[:300]}...\n\n"

            return output

        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"

    return search_knowledge_base


def create_rag_chain(retriever: BaseRetriever, llm, system_prompt: str = None):
    """
    Create a RAG chain for retrieval-augmented generation.

    This combines retrieval with LLM generation for context-aware responses.

    Args:
        retriever: LangChain retriever
        llm: LangChain LLM
        system_prompt: Optional system prompt

    Returns:
        RAG chain function
    """
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    # Default RAG prompt
    default_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt or """You are QuantMindX Analyst, a trading strategy expert.
Use the following retrieved context to answer questions. If you don't know the answer
based on the context, say so. Always cite your sources.

Context:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        """Format documents for prompt."""
        return "\n\n".join([
            f"**{d.metadata.get('title', 'Document')}**\n{d.page_content}"
            for d in docs
        ])

    # Create RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | default_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def create_agent_with_retrieval(
    llm,
    tools: List,
    retriever: BaseRetriever,
    system_prompt: str = None
):
    """
    Create an agent with retrieval capability using LangGraph.

    This follows the LangChain pattern for agents with RAG.

    Args:
        llm: LangChain LLM
        tools: List of tools
        retriever: LangChain retriever
        system_prompt: System prompt

    Returns:
        Agent with retrieval
    """
    from langgraph.prebuilt import create_react_agent

    # Create retrieval tool
    @tool
    def retrieve_context(query: str, k: int = 3) -> str:
        """Retrieve relevant context from knowledge base."""
        docs = retriever.get_relevant_documents(query)
        return "\n\n".join([d.page_content for d in docs[:k]])

    # Add retrieval to tools
    all_tools = tools + [retrieve_context]

    # Create ReAct agent
    agent = create_react_agent(
        llm,
        all_tools,
        state_modifier=system_prompt or "You are a helpful trading strategy analyst."
    )

    return agent


class RAGEnabledAgent:
    """
    Agent with built-in RAG capabilities.

    Extends standard agent with retrieval tools and chains.
    """

    def __init__(
        self,
        base_agent,
        kb_client,
        collection: str = "analyst_kb"
    ):
        """
        Initialize RAG-enabled agent.

        Args:
            base_agent: Base agent instance
            kb_client: ChromaKBClient
            collection: KB collection name
        """
        self.agent = base_agent
        self.kb_client = kb_client
        self.collection = collection

        # Create retriever
        self.retriever = ChromaDBRetriever(
            kb_client=kb_client,
            collection=collection
        )

        # Create retrieval tool
        self.retrieval_tool = create_retriever_tool(kb_client, collection)

        # Add to agent's tools
        if hasattr(self.agent, 'add_tool'):
            self.agent.add_tool(self.retrieval_tool)

    def ask_with_rag(
        self,
        question: str,
        use_retrieval: bool = True,
        k: int = 3
    ) -> str:
        """
        Ask a question with RAG retrieval.

        Args:
            question: User's question
            use_retrieval: Whether to use KB retrieval
            k: Number of documents to retrieve

        Returns:
            Agent response with retrieved context
        """
        from langchain_core.messages import HumanMessage

        if use_retrieval:
            # Retrieve context
            docs = self.retriever.get_relevant_documents(question)
            context_docs = docs[:k]

            if context_docs:
                # Format context
                context = "\n\n".join([
                    f"**{d.metadata.get('title', 'Document')}**\n{d.page_content}"
                    for d in context_docs
                ])

                # Add context to message
                message = f"""Context from knowledge base:

{context}

Question: {question}"""
            else:
                message = question
        else:
            message = question

        # Invoke agent
        if hasattr(self.agent, 'invoke'):
            return self.agent.invoke([HumanMessage(content=message)])
        elif hasattr(self.agent, 'chat'):
            return self.agent.chat(message)
        else:
            raise AttributeError("Agent has no invoke or chat method")

    def create_rag_chain(self, system_prompt: str = None):
        """Create a RAG chain with this agent's LLM."""
        if not hasattr(self.agent, '_llm'):
            raise AttributeError("Agent has no _llm attribute")

        return create_rag_chain(
            retriever=self.retriever,
            llm=self.agent._llm,
            system_prompt=system_prompt
        )


def enable_rag_for_agent(base_agent, kb_client, collection: str = "analyst_kb") -> RAGEnabledAgent:
    """
    Enable RAG for an existing agent.

    Args:
        base_agent: Existing agent instance
        kb_client: ChromaKBClient
        collection: KB collection name

    Returns:
        RAGEnabledAgent wrapping the base agent
    """
    return RAGEnabledAgent(
        base_agent=base_agent,
        kb_client=kb_client,
        collection=collection
    )


# Convenience function for AnalystAgent

def create_analyst_agent_with_rag(
    api_key: str = None,
    model: str = "qwen/qwen3-vl-30b-a3b-thinking",
    kb_client=None,
    collection: str = "analyst_kb"
):
    """
    Create AnalystAgent with RAG enabled.

    Args:
        api_key: OpenRouter API key
        model: Model name
        kb_client: ChromaKBClient
        collection: KB collection

    Returns:
        RAGEnabledAnalystAgent
    """
    from .analyst_agent import create_analyst_agent

    # Create base agent
    agent = create_analyst_agent(
        api_key=api_key,
        model=model,
        kb_client=kb_client
    )

    # Enable RAG
    return enable_rag_for_agent(agent, kb_client, collection)
