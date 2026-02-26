"""
SDK MCP Server Integration

Loads MCP server configurations and provides integration with Claude Agent SDK.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MCPTransport(Enum):
    """MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    transport: MCPTransport
    description: str = ""

    # For STDIO transport
    command: Optional[str] = None
    args: List[str] = None
    env: Dict[str, str] = None

    # For HTTP/WebSocket transport
    url: Optional[str] = None

    def __post_init__(self):
        if self.args is None:
            self.args = []
        if self.env is None:
            self.env = {}


def load_mcp_servers(config_path: Path) -> List[MCPServerConfig]:
    """
    Load MCP server configurations from JSON file.

    Args:
        config_path: Path to MCP config JSON file

    Returns:
        List of MCPServerConfig objects
    """
    if not config_path.exists():
        logger.warning(f"MCP config not found: {config_path}")
        return []

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse MCP config: {e}")
        return []

    servers = []
    for name, server_config in config.get("mcpServers", {}).items():
        try:
            # Determine transport type
            if "url" in server_config:
                transport = MCPTransport.HTTP
            else:
                transport = MCPTransport.STDIO

            server = MCPServerConfig(
                name=name,
                transport=transport,
                description=server_config.get("description", ""),
                command=server_config.get("command"),
                args=server_config.get("args", []),
                env=server_config.get("env", {}),
                url=server_config.get("url"),
            )
            servers.append(server)
            logger.debug(f"Loaded MCP server: {name}")

        except Exception as e:
            logger.error(f"Failed to load MCP server {name}: {e}")

    logger.info(f"Loaded {len(servers)} MCP servers from {config_path}")
    return servers


def get_mcp_server_names(servers: List[MCPServerConfig]) -> List[str]:
    """Get list of MCP server names."""
    return [s.name for s in servers]


def get_mcp_server_by_name(servers: List[MCPServerConfig], name: str) -> Optional[MCPServerConfig]:
    """Get MCP server by name."""
    for server in servers:
        if server.name == name:
            return server
    return None


def validate_mcp_config(config_path: Path) -> Dict[str, Any]:
    """
    Validate MCP configuration file.

    Returns dict with validation results.
    """
    results = {
        "valid": True,
        "servers": [],
        "errors": [],
    }

    if not config_path.exists():
        results["valid"] = False
        results["errors"].append(f"Config file not found: {config_path}")
        return results

    servers = load_mcp_servers(config_path)

    for server in servers:
        server_info = {
            "name": server.name,
            "transport": server.transport.value,
            "valid": True,
        }

        # Validate based on transport
        if server.transport == MCPTransport.STDIO:
            if not server.command:
                server_info["valid"] = False
                server_info["error"] = "Missing command for STDIO transport"
                results["errors"].append(f"{server.name}: missing command")

        elif server.transport == MCPTransport.HTTP:
            if not server.url:
                server_info["valid"] = False
                server_info["error"] = "Missing URL for HTTP transport"
                results["errors"].append(f"{server.name}: missing URL")

        results["servers"].append(server_info)
        if not server_info["valid"]:
            results["valid"] = False

    return results
