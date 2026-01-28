"""
MCP (Model Context Protocol) client for QuantMindX agents.

Allows agents to access external tools and resources via MCP servers.
"""

import json
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from enum import Enum


class MCPServerState(Enum):
    """MCP server connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPTool:
    """An MCP tool definition."""

    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)

    # Server info
    server_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "server_name": self.server_name
        }

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": f"mcp_{self.server_name}_{self.name}",
                "description": f"[MCP:{self.server_name}] {self.description}",
                "parameters": self.input_schema
            }
        }


@dataclass
class MCPServer:
    """An MCP server connection."""

    name: str
    command: List[str]
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    # Runtime state
    process: Optional[Any] = None
    state: MCPServerState = MCPServerState.DISCONNECTED

    # Available tools
    tools: List[MCPTool] = field(default_factory=list)

    # Stdio for communication
    stdin = None
    stdout = None
    stderr = None


class MCPClient:
    """
    Client for connecting to and using MCP servers.

    Supports stdio transport for local MCP servers.
    """

    def __init__(self):
        self.servers: Dict[str, MCPServer] = {}
        self.tools: Dict[str, MCPTool] = {}

    def add_server(
        self,
        name: str,
        command: Union[str, List[str]],
        args: List[str] = None,
        env: Dict[str, str] = None
    ) -> None:
        """
        Add an MCP server configuration.

        Args:
            name: Server identifier
            command: Command to start server (string or list)
            args: Additional arguments
            env: Environment variables
        """
        if isinstance(command, str):
            command = [command]

        self.servers[name] = MCPServer(
            name=name,
            command=command,
            args=args or [],
            env=env or {}
        )

    def connect(self, name: str) -> bool:
        """
        Connect to an MCP server.

        Args:
            name: Server name

        Returns:
            True if connected successfully
        """
        if name not in self.servers:
            return False

        server = self.servers[name]
        server.state = MCPServerState.CONNECTING

        try:
            # Build command
            cmd = server.command + server.args

            # Start subprocess with stdio
            server.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**subprocess.os.environ, **server.env},
                text=True
            )

            server.stdin = server.process.stdin
            server.stdout = server.process.stdout
            server.stderr = server.process.stderr

            # Initialize handshake
            self._send_json(server, {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "quantmindx-agent",
                        "version": "1.0.0"
                    }
                }
            })

            # Get tools list
            response = self._send_json(server, {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/list",
                "params": {}
            })

            if response and "result" in response:
                for tool_def in response["result"].get("tools", []):
                    mcp_tool = MCPTool(
                        name=tool_def["name"],
                        description=tool_def.get("description", ""),
                        input_schema=tool_def.get("inputSchema", {}),
                        server_name=name
                    )
                    server.tools.append(mcp_tool)
                    self.tools[f"{name}.{mcp_tool.name}"] = mcp_tool

            server.state = MCPServerState.CONNECTED
            return True

        except Exception as e:
            server.state = MCPServerState.ERROR
            return False

    def disconnect(self, name: str) -> bool:
        """Disconnect from an MCP server."""
        if name not in self.servers:
            return False

        server = self.servers[name]

        # Send shutdown notification
        try:
            self._send_json(server, {
                "jsonrpc": "2.0",
                "method": "shutdown",
                "params": {}
            })
        except:
            pass

        # Close process
        if server.process:
            server.process.terminate()
            server.process.wait(timeout=5)

        server.state = MCPServerState.DISCONNECTED
        return True

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any] = None
    ) -> Any:
        """
        Call an MCP tool.

        Args:
            server_name: Server name
            tool_name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        key = f"{server_name}.{tool_name}"
        if key not in self.tools:
            return None

        server = self.servers.get(server_name)
        if not server or server.state != MCPServerState.CONNECTED:
            return None

        try:
            response = self._send_json(server, {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments or {}
                }
            })

            if response and "result" in response:
                return response["result"]

            return None

        except Exception as e:
            return {"error": str(e)}

    def list_tools(self, server_name: str = None) -> List[MCPTool]:
        """
        List available MCP tools.

        Args:
            server_name: Optional server filter

        Returns:
            List of MCP tools
        """
        if server_name:
            server = self.servers.get(server_name)
            return server.tools if server else []

        return list(self.tools.values())

    def get_tool(self, key: str) -> Optional[MCPTool]:
        """Get an MCP tool by server.tool key."""
        return self.tools.get(key)

    def _send_json(self, server: MCPServer, data: Dict[str, Any]) -> Optional[Dict]:
        """Send JSON request and read response."""
        import json

        try:
            # Send request
            request = json.dumps(data) + "\n"
            server.stdin.write(request)
            server.stdin.flush()

            # Read response
            response_line = server.stdout.readline()
            if not response_line:
                return None

            return json.loads(response_line.strip())

        except Exception as e:
            return None

    def connect_all(self) -> Dict[str, bool]:
        """Connect to all configured servers."""
        results = {}
        for name in self.servers:
            results[name] = self.connect(name)
        return results

    def disconnect_all(self) -> Dict[str, bool]:
        """Disconnect from all servers."""
        results = {}
        for name in list(self.servers.keys()):
            results[name] = self.disconnect(name)
        return results

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        """Convert all MCP tools to OpenAI function format."""
        return [tool.to_openai_format() for tool in self.tools.values()]

    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect_all()


# Built-in MCP server configurations

def get_quantmindx_mcp_servers() -> List[Dict[str, Any]]:
    """Get default MCP servers for QuantMindX."""
    return [
        {
            "name": "quantmindx-kb",
            "command": ["python", "-m", "quantmindx_kb.server"],
            "description": "Knowledge base access"
        },
        {
            "name": "filesystem",
            "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem"],
            "args": ["/home/mubarkahimself/Desktop/QUANTMINDX"],
            "description": "Filesystem access"
        }
    ]


def create_mcp_client(servers: List[Dict[str, Any]] = None) -> MCPClient:
    """
    Create and configure MCP client.

    Args:
        servers: List of server configurations

    Returns:
        Configured MCPClient
    """
    client = MCPClient()

    for server_config in servers or []:
        client.add_server(
            name=server_config["name"],
            command=server_config["command"],
            args=server_config.get("args", []),
            env=server_config.get("env", {})
        )

    return client
