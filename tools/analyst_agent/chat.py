"""
Chat interface for Analyst Agent.
Conversational AI with knowledge base integration.

Uses LangChain + OpenRouter for proper LLM integration.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich import box

console = Console()

# Try to import LangChain
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    console.print("[yellow]LangChain not installed. Chat will use fallback mode.[/yellow]")


def create_openrouter_llm(
    api_key: str,
    model: str = "qwen/qwen3-vl-30b-a3b-thinking",
    temperature: float = 0.7
):
    """
    Create a LangChain ChatOpenAI instance for OpenRouter.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available. Install with: pip install langchain-openai")

    # Get repository info for headers
    try:
        import subprocess
        git_remote = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=Path(__file__).parent.parent.parent.parent,
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_remote = "https://github.com/user/quantmindx"

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": git_remote,
            "X-Title": "QuantMindX Analyst Agent Chat"
        }
    )

    return llm


@dataclass
class Message:
    """A chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    sources: list = field(default_factory=list)


@dataclass
class ChatSession:
    """A chat session with conversation history."""
    messages: List[Message] = field(default_factory=list)
    max_history: int = 10

    def add_message(self, role: str, content: str, sources: list = None):
        """Add a message to the session."""
        self.messages.append(Message(role=role, content=content, sources=sources or []))

        # Keep only recent messages to manage token count
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_history_for_api(self) -> List[Dict]:
        """Get formatted history for API call."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
            if m.role in ["user", "assistant"]
        ]


class ChatAgent:
    """
    Conversational AI agent with knowledge base integration.

    Uses OpenRouter API with configurable model.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-vl-30b-a3b-thinking",
        kb_client=None,
        system_prompt: Optional[str] = None
    ):
        self.api_key = api_key
        self.model = model
        self.kb_client = kb_client
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.session = ChatSession()

        # Default system prompt
        self.system_prompt = system_prompt or """You are QuantMindX Analyst, an expert trading strategy AI assistant with deep knowledge of:

1. **MQL5/MetaTrader Development** - Expert Advisors, indicators, scripts
2. **Trading Strategies** - ORB, breakout, reversal, scalping, swing trading
3. **Risk Management** - Kelly criterion, position sizing, drawdown control
4. **Technical Analysis** - Indicators, patterns, market structure
5. **Algorithmic Trading** - Backtesting, optimization, automation

**Your Capabilities:**
- Explain trading concepts clearly
- Generate Technical Requirements Documents (TRDs)
- Compare strategies and approaches
- Suggest implementations based on MQL5 best practices
- Answer questions using your knowledge base

**Knowledge Base Access:**
You have access to a curated knowledge base of 1,805+ MQL5 articles. When answering questions, reference relevant concepts from these articles.

**Communication Style:**
- Be direct and practical
- Use trading terminology correctly
- Provide specific examples when possible
- Flag when information might be missing or needs verification

**When generating TRDs:**
- Use the exact markdown structure specified
- Include all required sections
- Be thorough and specific
- Reference KB articles when relevant"""

    def _search_kb(self, query: str, n: int = 3) -> List[Dict]:
        """Search knowledge base for relevant articles."""
        if not self.kb_client:
            return []

        try:
            # Try analyst_kb first, fallback to mql5_knowledge
            collections = self.kb_client.list_collections()
            collection = "analyst_kb" if "analyst_kb" in collections else "mql5_knowledge"

            results = self.kb_client.search(query, collection=collection, n=n)

            # Format results
            formatted = []
            for r in results:
                formatted.append({
                    "title": r.get("title", "Untitled"),
                    "score": r.get("score", 0),
                    "preview": r.get("preview", "")[:300],
                    "categories": r.get("categories", ""),
                })
            return formatted
        except Exception as e:
            console.print(f"[dim]KB search error: {e}[/dim]")
            return []

    def _format_kb_context(self, kb_results: List[Dict]) -> str:
        """Format KB results as context for the prompt."""
        if not kb_results:
            return ""

        context = "\n\n**Relevant Knowledge Base Articles:**\n\n"
        for i, article in enumerate(kb_results, 1):
            context += f"{i}. **{article['title']}** (relevance: {article['score']:.2f})\n"
            context += f"   {article['preview'][:200]}...\n\n"

        return context

    def chat(self, user_message: str, use_kb: bool = True) -> str:
        """
        Send a chat message and get response using LangChain.

        Args:
            user_message: User's message
            use_kb: Whether to search KB for context

        Returns:
            Assistant's response
        """
        # Add user message to session
        self.session.add_message("user", user_message)

        # Search KB if enabled
        kb_results = []
        kb_context = ""

        if use_kb and self.kb_client:
            # Generate search terms from message
            search_terms = self._extract_search_terms(user_message)

            for term in search_terms[:2]:  # Search up to 2 terms
                results = self._search_kb(term, n=2)
                kb_results.extend(results)

            # Remove duplicates
            seen = set()
            unique_results = []
            for r in kb_results:
                if r['title'] not in seen:
                    seen.add(r['title'])
                    unique_results.append(r)
            kb_results = unique_results[:5]  # Max 5 results

            kb_context = self._format_kb_context(kb_results)

        # Prepare messages for LangChain
        messages = []

        # Add system message
        messages.append(SystemMessage(content=self.system_prompt))

        # Add KB context to system message if available
        if kb_context:
            messages[0].content += "\n\n" + kb_context

        # Add conversation history (convert to LangChain format)
        for msg in self.session.get_history_for_api():
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        try:
            # Use LangChain ChatOpenAI
            llm = create_openrouter_llm(
                api_key=self.api_key,
                model=self.model,
                temperature=0.7
            )

            # Invoke and get response
            response = llm.invoke(messages)
            assistant_message = response.content

            # Add assistant response to session
            self.session.add_message("assistant", assistant_message, sources=kb_results)

            return assistant_message

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            self.session.add_message("assistant", error_msg)
            return error_msg

    def _extract_search_terms(self, message: str) -> List[str]:
        """Extract search terms from user message."""
        # Simple extraction - look for key phrases
        import re

        # Remove common words
        stop_words = {
            "what", "how", "why", "when", "where", "who", "the", "a", "an",
            "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "can", "explain", "tell", "me", "about", "generate"
        }

        # Extract quoted phrases first
        quoted = re.findall(r'"([^"]+)"', message)
        if quoted:
            return quoted[:3]

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', message.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]

        # Look for trading-specific terms
        trading_terms = [
            "kelly", "optimal", "position", "sizing", "risk", "orb",
            "breakout", "scalping", "strategy", "indicator", "atr",
            "moving", "average", "rsi", "macd", "stop", "loss", "profit"
        ]

        # Prioritize trading terms
        found = [w for w in keywords if any(t in w for t in trading_terms)]

        # Add remaining keywords
        for w in keywords:
            if w not in found and len(found) < 5:
                found.append(w)

        return found[:3]

    def clear_history(self):
        """Clear conversation history."""
        self.session.messages = []

    def get_history_count(self) -> int:
        """Get number of messages in history."""
        return len(self.session.messages)


def print_assistant_message(content: str, sources: list = None):
    """Print assistant message with nice formatting."""
    # Parse content for markdown
    # For now, just print in a panel
    console.print("\n")

    # Handle code blocks
    if "```" in content:
        # Split by code blocks
        parts = content.split("```")
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code block
                lines = part.split("\n", 1)
                lang = lines[0].strip() if lines else ""
                code = lines[1] if len(lines) > 1 else part

                console.print(Panel(
                    code,
                    title=f"[dim]{lang}[/dim]" if lang else None,
                    border_style="cyan",
                    padding=(0, 1),
                ))
            else:  # Regular text
                if part.strip():
                    console.print(Panel(
                        Markdown(part.strip()),
                        border_style="green",
                        box=box.ROUNDED,
                        padding=(1, 2),
                    ))
    else:
        # No code blocks, print as markdown panel
        console.print(Panel(
            Markdown(content),
            title="[bold green]Assistant[/bold green]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        ))

    # Show sources if available
    if sources:
        console.print("\n[dim]📚 Sources:[/dim]")
        for s in sources:
            console.print(f"  [dim]•[/dim] [cyan]{s['title']}[/cyan] [dim]({s['score']:.2f})[/dim]")

    console.print()


def print_user_message(content: str):
    """Print user message."""
    console.print(Panel(
        Text(content, style="bold white"),
        title="[bold blue]You[/bold blue]",
        border_style="blue",
        box=box.ROUNDED,
        padding=(1, 2),
    ))
    console.print()
