"""
Base Agent class for all QuantMindX agents.

Provides unified foundation with:
- LangChain + OpenRouter integration
- Skills and tools management
- MCP access
- Memory with LangMem
- Knowledge base integration
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime

from rich.console import Console

console = Console()


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    name: str
    description: str = ""

    # LLM Settings
    provider: str = "openrouter"
    model: str = "qwen/qwen3-vl-30b-a3b-thinking"  # High-performance reasoning model
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: Optional[str] = None
    base_url: str = "https://openrouter.ai/api/v1"

    # Memory Settings
    use_memory: bool = True
    memory_backend: str = "chroma"  # chroma-based memory for retrieval
    streaming: bool = True  # Enable streaming responses

    # KB Settings
    use_kb: bool = True
    kb_collection: str = "analyst_kb"

    # MCP Settings
    use_mcp: bool = True
    mcp_servers: List[str] = field(default_factory=list)

    # Streaming
    streaming: bool = False

    # Repository info for OpenRouter
    git_remote: str = ""

    def __post_init__(self):
        if not self.git_remote:
            try:
                import subprocess
                self.git_remote = subprocess.check_output(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=Path(__file__).parent.parent.parent.parent,
                    stderr=subprocess.DEVNULL
                ).decode().strip()
            except Exception:
                self.git_remote = "https://github.com/user/quantmindx"


@dataclass
class AgentCapabilities:
    """Agent capabilities definition."""

    skills: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    mcp_tools: List[str] = field(default_factory=list)

    def can_use(self, capability: str) -> bool:
        """Check if agent has a capability."""
        return (
            capability in self.skills or
            capability in self.tools or
            capability in self.mcp_tools
        )


class BaseAgent(ABC):
    """
    Base class for all QuantMindX agents.

    Provides:
    - LangChain LLM integration
    - Skills registry
    - Tools registry
    - MCP client
    - Memory (LangMem)
    - Knowledge base access
    """

    def __init__(
        self,
        config: AgentConfig,
        kb_client=None,
        mcp_client=None
    ):
        self.config = config
        self.kb_client = kb_client
        self.mcp_client = mcp_client
        self.capabilities = AgentCapabilities()

        # Import registries
        from .skills import SkillRegistry
        from .tools import ToolRegistry

        self.skills = SkillRegistry()
        self.tools = ToolRegistry()

        # Initialize LLM
        self._llm = None
        self._init_llm()

        # Initialize memory
        self._memory = None
        if config.use_memory:
            self._init_memory()

        # Register agent-specific capabilities
        self._register_capabilities()

    def _init_llm(self):
        """Initialize LangChain LLM with streaming support."""
        try:
            from langchain_openai import ChatOpenAI

            api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                console.print("[yellow]Warning: No API key found[/yellow]")
                return

            self._llm = ChatOpenAI(
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming,
                api_key=api_key,
                base_url=self.config.base_url,
                default_headers={
                    "HTTP-Referer": self.config.git_remote,
                    "X-Title": f"QuantMindX {self.config.name}"
                }
            )
        except ImportError:
            console.print("[yellow]LangChain not installed. Run: pip install langchain-openai[/yellow]")

    def _init_memory(self):
        """Initialize memory using ChromaDB for conversation history."""
        if self.config.memory_backend == "langmem":
            try:
                from langmem import Memory

                self._memory = Memory(
                    backend="chroma",
                    namespace=f"agent_{self.config.name}"
                )
            except ImportError:
                console.print("[yellow]LangMem not installed. Run: pip install langmem[/yellow]")
                self._memory = None
        elif self.config.memory_backend == "chroma":
            # Use ChromaDB directly for memory storage
            try:
                import chromadb

                # Create persistent client
                client = chromadb.PersistentClient(
                    path=str(Path(__file__).parent.parent.parent.parent / "data" / "chromadb")
                )

                # Get or create memory collection
                self._memory_collection = client.get_or_create_collection(
                    name=f"agent_{self.config.name}_memory",
                    metadata={"hnsw:space": "cosine"}
                )

                # Simple memory wrapper
                class ChromaMemory:
                    def __init__(self, collection):
                        self.collection = collection
                        self.history = []

                    def add_message(self, role: str, content: str, metadata: dict = None):
                        """Add message to memory."""
                        import uuid
                        msg_id = str(uuid.uuid4())

                        self.history.append({
                            "role": role,
                            "content": content,
                            "id": msg_id,
                            "metadata": metadata or {}
                        })

                        # Store in ChromaDB for retrieval
                        try:
                            self.collection.add(
                                documents=[content],
                                metadatas=[{
                                    "role": role,
                                    "msg_id": msg_id,
                                    "timestamp": str(datetime.now().isoformat()),
                                    **(metadata or {})
                                }],
                                ids=[msg_id]
                            )
                        except Exception as e:
                            console.print(f"[dim]Memory add error: {e}[/dim]")

                    def get_recent(self, n: int = 10) -> list:
                        """Get recent messages."""
                        return self.history[-n:]

                    def search(self, query: str, n: int = 5) -> list:
                        """Search memory by semantic similarity."""
                        try:
                            results = self.collection.query(
                                query_texts=[query],
                                n_results=min(n, len(self.history))
                            )

                            if not results or not results['ids'][0]:
                                return []

                            # Reconstruct messages from search results
                            messages = []
                            for i, msg_id in enumerate(results['ids'][0]):
                                # Find in history
                                for msg in self.history:
                                    if msg['id'] == msg_id:
                                        messages.append(msg)
                                        break

                            return messages
                        except Exception as e:
                            console.print(f"[dim]Memory search error: {e}[/dim]")
                            return []

                self._memory = ChromaMemory(self._memory_collection)

            except ImportError:
                console.print("[yellow]ChromaDB not installed. Run: pip install chromadb[/yellow]")
                self._memory = None
            except Exception as e:
                console.print(f"[yellow]Memory initialization error: {e}[/yellow]")
                self._memory = None
        else:
            self._memory = None

    @abstractmethod
    def _register_capabilities(self):
        """Register agent-specific skills and tools."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get agent's system prompt."""
        pass

    def add_skill(self, skill: "Skill"):
        """Add a skill to the agent."""
        self.skills.register(skill)
        self.capabilities.skills.append(skill.name)

    def add_tool(self, tool: "Tool"):
        """Add a tool to the agent."""
        self.tools.register(tool)
        self.capabilities.tools.append(tool.name)

    def get_skill(self, name: str) -> Optional["Skill"]:
        """Get a skill by name."""
        return self.skills.get(name)

    def get_tool(self, name: str) -> Optional["Tool"]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_skills(self) -> List[str]:
        """List all available skills."""
        return list(self.skills.skills.keys())

    def list_tools(self) -> List[str]:
        """List all available tools."""
        return list(self.tools.tools.keys())

    def invoke(
        self,
        messages: List[Any],
        use_kb: bool = True,
        use_memory: bool = True
    ) -> str:
        """
        Invoke the agent with messages.

        Args:
            messages: List of LangChain messages
            use_kb: Whether to include KB context
            use_memory: Whether to use memory

        Returns:
            Agent response
        """
        if not self._llm:
            return "Error: LLM not initialized"

        from langchain_core.messages import SystemMessage

        # Add system prompt
        system_prompt = self.get_system_prompt()

        # Add KB context if enabled
        if use_kb and self.kb_client:
            kb_context = self._get_kb_context(messages)
            if kb_context:
                system_prompt += "\n\n" + kb_context

        messages = [SystemMessage(content=system_prompt)] + messages

        # Store in memory if enabled
        if use_memory and self._memory:
            self._memory.store(messages)

        # Invoke LLM
        try:
            response = self._llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error invoking LLM: {str(e)}"

    def _get_kb_context(self, messages: List[Any]) -> str:
        """Get relevant KB context from messages."""
        if not self.kb_client:
            return ""

        # Extract query from last user message
        from langchain_core.messages import HumanMessage

        query = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        if not query:
            return ""

        try:
            results = self.kb_client.search(
                query,
                collection=self.config.kb_collection,
                n=3
            )

            if not results:
                return ""

            context = "\n\n**Relevant Knowledge Base Articles:**\n\n"
            for i, article in enumerate(results[:3], 1):
                context += f"{i}. **{article.get('title', 'Untitled')}**\n"
                context += f"   Relevance: {article.get('score', 0):.2f}\n"
                preview = article.get('preview', '')[:200]
                context += f"   Preview: {preview}...\n\n"

            return context
        except Exception as e:
            console.print(f"[dim]KB search error: {e}[/dim]")
            return ""

    def chat(self, user_message: str, use_kb: bool = True) -> str:
        """
        Simple chat interface.

        Args:
            user_message: User's message
            use_kb: Whether to search KB

        Returns:
            Agent response
        """
        from langchain_core.messages import HumanMessage

        messages = [HumanMessage(content=user_message)]
        return self.invoke(messages, use_kb=use_kb)

    def stream(self, user_message: str, use_kb: bool = True):
        """
        Stream response from agent.

        Args:
            user_message: User's message
            use_kb: Whether to search KB

        Yields:
            Response chunks
        """
        if not self._llm:
            yield "Error: LLM not initialized"
            return

        from langchain_core.messages import HumanMessage, SystemMessage

        system_prompt = self.get_system_prompt()

        if use_kb and self.kb_client:
            kb_context = self._get_kb_context([HumanMessage(content=user_message)])
            if kb_context:
                system_prompt += "\n\n" + kb_context

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        for chunk in self._llm.stream(messages):
            yield chunk.content

    def get_capabilities(self) -> AgentCapabilities:
        """Get agent capabilities."""
        return self.capabilities

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.config.name}', model='{self.config.model}')"
