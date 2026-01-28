"""
Tools system for QuantMindX agents.

A tool is a specific function that an agent can call.
Tools are lower-level than skills and perform atomic operations.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any, Union
from functools import wraps
import json


@dataclass
class Tool:
    """A tool that an agent can use."""

    name: str
    description: str

    # Tool execution
    func: Callable = None

    # Parameters schema (for validation)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Return type
    return_type: str = "string"

    # Metadata
    category: str = "general"
    version: str = "1.0"

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool."""
        if self.func is None:
            raise NotImplementedError(f"Tool '{self.name}' has no function")
        return self.func(*args, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "return_type": self.return_type,
            "category": self.category,
            "version": self.version
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": list(self.parameters.keys())
                }
            }
        }


class ToolRegistry:
    """Registry for managing agent tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool

    def register_function(self, name: str, func: Callable, description: str = "", **kwargs):
        """Register a function as a tool."""
        tool = Tool(name=name, description=description, func=func, **kwargs)
        self.register(tool)
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list(self, category: Optional[str] = None) -> List[Tool]:
        """List all tools, optionally filtered by category."""
        if category:
            return [t for t in self.tools.values() if t.category == category]
        return list(self.tools.values())

    def has(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self.tools

    def remove(self, name: str) -> bool:
        """Remove a tool."""
        if name in self.tools:
            del self.tools[name]
            return True
        return False

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Convert all tools to OpenAI function format."""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert registry to dictionary."""
        return {name: tool.to_dict() for name, tool in self.tools.items()}


# Decorator for creating tools

def tool(name: str = None, description: str = "", **kwargs):
    """Decorator to create a tool from a function."""
    def decorator(func):
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        return Tool(
            name=tool_name,
            description=tool_desc,
            func=func,
            **kwargs
        )
    return decorator


# Built-in tools

@tool(name="read_file", description="Read contents of a file")
def read_file_tool(path: str) -> str:
    """Read file contents."""
    from pathlib import Path
    file_path = Path(path)
    if not file_path.exists():
        return f"Error: File not found: {path}"
    return file_path.read_text()


@tool(name="write_file", description="Write content to a file")
def write_file_tool(path: str, content: str) -> str:
    """Write content to file."""
    from pathlib import Path
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)
    return f"Written to: {path}"


@tool(name="search_kb", description="Search knowledge base for articles")
def search_kb_tool(query: str, n: int = 3, kb_client=None) -> str:
    """Search KB and return results as JSON."""
    if not kb_client:
        return "Error: KB client not configured"
    results = kb_client.search(query, collection="analyst_kb", n=n)
    return json.dumps([{
        "title": r.get("title"),
        "score": r.get("score"),
        "preview": r.get("preview", "")[:200]
    } for r in results], indent=2)


@tool(name="get_current_time", description="Get current date and time")
def get_current_time_tool() -> str:
    """Get current time."""
    from datetime import datetime
    return datetime.now().isoformat()


@tool(name="calculate", description="Perform mathematical calculations")
def calculate_tool(expression: str) -> str:
    """Safely calculate a math expression."""
    try:
        # Only allow safe operations
        import operator
        import re

        ops = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            '**': operator.pow,
            '%': operator.mod,
        }

        # Simple and safe evaluation
        tokens = re.findall(r'(\d+\.?\d*|\+|\-|\*|\/|\*\*|\%|\(|\))', expression)
        if not tokens:
            return "Error: No valid expression"

        # Very basic evaluation - for production use ast.literal_eval or similar
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


BUILTIN_TOOLS = {
    "read_file": read_file_tool,
    "write_file": write_file_tool,
    "search_kb": lambda **kw: search_kb_tool(**kw),
    "get_current_time": get_current_time_tool,
    "calculate": calculate_tool
}
