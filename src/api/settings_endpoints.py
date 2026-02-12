"""
Settings API Endpoints

Manages user settings, API keys, MCP servers, agent configuration,
risk management, and database settings.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
from pathlib import Path

router = APIRouter(prefix="/api/settings", tags=["settings"])

# Settings file path
SETTINGS_DIR = Path("./config/settings")
SETTINGS_FILE = SETTINGS_DIR / "settings.json"
SETTINGS_DIR.mkdir(parents=True, exist_ok=True)

# Data models
class GeneralSettings(BaseModel):
    theme: str = "dark"
    language: str = "en"
    timezone: str = "UTC"
    autoSave: bool = True
    autoSaveInterval: int = 30
    debugMode: bool = False
    logLevel: str = "info"

class APIKey(BaseModel):
    id: str
    name: str
    key: str
    service: str
    created: str
    lastUsed: Optional[str] = None

class MCPServer(BaseModel):
    id: str
    name: str
    command: str
    args: List[str]
    status: str = "stopped"
    type: str = "custom"
    description: Optional[str] = None

class AgentSettings(BaseModel):
    defaultModel: str = "claude-sonnet-4"
    temperature: float = 0.7
    maxTokens: int = 4096
    enableMemory: bool = True
    memoryType: str = "hybrid"
    skillsEnabled: bool = True
    autoDelegate: bool = False

class RiskSettings(BaseModel):
    houseMoneyEnabled: bool = True
    houseMoneyThreshold: float = 0.5
    dailyLossLimit: float = 5.0
    maxDrawdown: float = 10.0
    riskMode: str = "dynamic"
    balanceZones: Dict[str, float] = {
        "danger": 200,
        "growth": 1000,
        "scaling": 5000,
        "guardian": 999999999  # Use large number instead of inf
    }

class DatabaseSettings(BaseModel):
    sqlitePath: str = "./data/quantmind.db"
    duckdbPath: str = "./data/analytics.duckdb"
    autoBackup: bool = True
    backupInterval: int = 3600
    maxBackups: int = 10

# Load settings from file
def load_settings() -> Dict[str, Any]:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {
        "general": GeneralSettings().dict(),
        "apiKeys": [],
        "mcpServers": [
            {
                "id": "context7",
                "name": "Context7",
                "command": "npx",
                "args": ["-y", "@context7/mcp-server"],
                "status": "stopped",
                "type": "builtin",
                "description": "Documentation lookup for LangChain, LangGraph, and libraries"
            }
        ],
        "agents": AgentSettings().dict(),
        "risk": RiskSettings().dict(),
        "database": DatabaseSettings().dict()
    }

# Save settings to file
def save_settings(settings: Dict[str, Any]):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

# General Settings
@router.get("/general")
async def get_general_settings():
    """Get general application settings."""
    settings = load_settings()
    return settings.get("general", GeneralSettings().dict())

@router.post("/general")
async def update_general_settings(settings: GeneralSettings):
    """Update general application settings."""
    current = load_settings()
    current["general"] = settings.dict()
    save_settings(current)
    return {"success": True, "settings": settings.dict()}

# API Keys
@router.get("/keys")
async def get_api_keys():
    """Get all API keys."""
    settings = load_settings()
    return settings.get("apiKeys", [])

@router.post("/keys")
async def add_api_key(api_key: APIKey):
    """Add a new API key."""
    current = load_settings()
    keys = current.get("apiKeys", [])

    # Check for duplicate
    if any(k["id"] == api_key.id for k in keys):
        raise HTTPException(status_code=400, detail="API key with this ID already exists")

    keys.append(api_key.dict())
    current["apiKeys"] = keys
    save_settings(current)
    return {"success": True, "key": api_key.dict()}

@router.delete("/keys/{key_id}")
async def delete_api_key(key_id: str):
    """Delete an API key."""
    current = load_settings()
    keys = current.get("apiKeys", [])
    keys = [k for k in keys if k["id"] != key_id]
    current["apiKeys"] = keys
    save_settings(current)
    return {"success": True}

# MCP Servers
@router.get("/mcp")
async def get_mcp_servers():
    """Get all MCP servers."""
    settings = load_settings()
    return settings.get("mcpServers", [])

@router.post("/mcp")
async def add_mcp_server(server: MCPServer):
    """Add a new MCP server."""
    current = load_settings()
    servers = current.get("mcpServers", [])

    # Check for duplicate
    if any(s["id"] == server.id for s in servers):
        raise HTTPException(status_code=400, detail="MCP server with this ID already exists")

    servers.append(server.dict())
    current["mcpServers"] = servers
    save_settings(current)
    return {"success": True, "server": server.dict()}

@router.delete("/mcp/{server_id}")
async def delete_mcp_server(server_id: str):
    """Delete an MCP server."""
    current = load_settings()
    servers = current.get("mcpServers", [])
    servers = [s for s in servers if s["id"] != server_id]
    current["mcpServers"] = servers
    save_settings(current)
    return {"success": True}

@router.patch("/mcp/{server_id}")
async def update_mcp_server_status(server_id: str, status: str):
    """Update MCP server status (running/stopped)."""
    current = load_settings()
    servers = current.get("mcpServers", [])

    for server in servers:
        if server["id"] == server_id:
            server["status"] = status
            break

    current["mcpServers"] = servers
    save_settings(current)
    return {"success": True}

# Agent Settings
@router.get("/agents")
async def get_agent_settings():
    """Get agent configuration settings."""
    settings = load_settings()
    return settings.get("agents", AgentSettings().dict())

@router.post("/agents")
async def update_agent_settings(settings: AgentSettings):
    """Update agent configuration settings."""
    current = load_settings()
    current["agents"] = settings.dict()
    save_settings(current)
    return {"success": True, "settings": settings.dict()}

# Skills
@router.get("/skills")
async def get_agent_skills():
    """Get all agent skills."""
    # Mock skills data - in production, this would come from the agent system
    return [
        {
            "id": "sk-1",
            "name": "Generate TRD",
            "agent": "analyst",
            "enabled": True,
            "description": "Generate Technical Requirements Document from NPRD"
        },
        {
            "id": "sk-2",
            "name": "Generate EA",
            "agent": "quantcode",
            "enabled": True,
            "description": "Generate MQL5 Expert Advisor code from TRD"
        },
        {
            "id": "sk-3",
            "name": "Run Backtest",
            "agent": "quantcode",
            "enabled": True,
            "description": "Execute backtest with specified parameters"
        }
    ]

class SkillToggleRequest(BaseModel):
    enabled: bool

@router.patch("/skills/{skill_id}")
async def toggle_agent_skill(skill_id: str, request: SkillToggleRequest):
    """Toggle a specific agent skill on/off."""
    # In production, this would update the agent configuration
    # For now, return success
    current = load_settings()
    skills = current.get("skills", [])

    # Find and update the skill
    for skill in skills:
        if skill.get("id") == skill_id:
            skill["enabled"] = request.enabled
            break

    current["skills"] = skills
    save_settings(current)
    return {"success": True, "enabled": request.enabled}

# Risk Settings
@router.get("/risk")
async def get_risk_settings():
    """Get risk management settings."""
    settings = load_settings()
    return settings.get("risk", RiskSettings().dict())

@router.post("/risk")
async def update_risk_settings(settings: RiskSettings):
    """Update risk management settings."""
    current = load_settings()
    current["risk"] = settings.dict()
    save_settings(current)
    return {"success": True, "settings": settings.dict()}

# Database Settings
@router.get("/database")
async def get_database_settings():
    """Get database configuration settings."""
    settings = load_settings()
    return settings.get("database", DatabaseSettings().dict())

@router.post("/database")
async def update_database_settings(settings: DatabaseSettings):
    """Update database configuration settings."""
    current = load_settings()
    current["database"] = settings.dict()
    save_settings(current)
    return {"success": True, "settings": settings.dict()}

# AGENTS.md Configuration
AGENTS_MD_PATH = SETTINGS_DIR / "agents.md"

@router.get("/agents-md")
async def get_agents_md():
    """Get the AGENTS.md file content."""
    if AGENTS_MD_PATH.exists():
        return {"content": AGENTS_MD_PATH.read_text()}
    else:
        # Return default content if file doesn't exist
        default_content = """# AGENTS.md

Agent configuration file for QuantMindX IDE.

## Agent Definitions

### copilot
**Role**: Trading Assistant & Workflow Guide

**Model Configuration**:
- Provider: openrouter
- Model: anthropic/claude-sonnet-4
- Temperature: 0.7
- Max Tokens: 4096

**System Prompt**:
```
You are a helpful trading assistant for QuantMindX, an AI-powered trading system.
```

**Skills**:
- `market-analysis`: Analyze market conditions and trends
- `strategy-guidance`: Guide users through strategy development
- `troubleshooting`: Identify and resolve common issues

**Tools**:
- get_market_data
- run_backtest
- get_position_size

---

### quantcode
**Role**: MQL5 Code Expert

**Model Configuration**:
- Provider: openrouter
- Model: anthropic/claude-sonnet-4
- Temperature: 0.3
- Max Tokens: 8192

**System Prompt**:
```
You are an MQ5 coding expert for QuantMindX.
```

**Skills**:
- `code-generation`: Generate MQL5 code from specifications
- `code-debugging`: Debug and fix MQL5 code issues
- `code-optimization`: Optimize code for performance

**Tools**:
- get_market_data
- run_backtest
- get_position_size

---

### analyst
**Role**: Trading Strategy Analyst

**Model Configuration**:
- Provider: openrouter
- Model: anthropic/claude-sonnet-4
- Temperature: 0.5
- Max Tokens: 6144

**System Prompt**:
```
You are a trading strategy analyst for QuantMindX.
```

**Skills**:
- `backtest-analysis`: Analyze backtesting results in depth
- `pattern-recognition`: Identify trading patterns and setups
- `risk-assessment`: Evaluate strategy risk profiles

**Tools**:
- get_market_data
- run_backtest
- get_position_size
"""
        return {"content": default_content}

@router.post("/agents-md")
async def save_agents_md(data: dict[str, str]):
    """Save the AGENTS.md file content."""
    content = data.get("content", "")
    AGENTS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    AGENTS_MD_PATH.write_text(content)
    return {"success": True, "message": "AGENTS.md saved successfully"}

@router.get("/agents-config")
async def get_agents_config():
    """Get parsed agent configurations from AGENTS.md."""
    if AGENTS_MD_PATH.exists():
        content = AGENTS_MD_PATH.read_text()
        return {"agents": parse_agents_md(content)}
    else:
        return {"agents": {}}

@router.post("/agents/{agent_id}/config")
async def update_agent_config(agent_id: str, config: dict[str, Any]):
    """Update a specific agent's configuration."""
    # Read current AGENTS.md
    if AGENTS_MD_PATH.exists():
        content = AGENTS_MD_PATH.read_text()
    else:
        content = "# AGENTS.md\n\n"

    # Parse and update the specific agent
    agents = parse_agents_md(content)

    if agent_id in agents:
        agents[agent_id].update(config)
        # Rebuild AGENTS.md content
        new_content = generate_agents_md(agents)
        AGENTS_MD_PATH.write_text(new_content)
        return {"success": True, "agent": agents[agent_id]}
    else:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

# Helper functions for AGENTS.md parsing
def parse_agents_md(content: str) -> dict[str, dict[str, Any]]:
    """Parse AGENTS.md content and return agent configurations."""
    agents = {}
    lines = content.split('\n')

    current_agent = None
    current_section = ''
    system_prompt_buffer = []
    skills_buffer = []
    tools_buffer = []

    for i, line in enumerate(lines):
        # Agent header
        if line.startswith('### ') and line[4:].islower():
            # Save previous agent
            if current_agent:
                if system_prompt_buffer:
                    current_agent['systemPrompt'] = '\n'.join(system_prompt_buffer).strip()
                    system_prompt_buffer = []
                current_agent['skills'] = skills_buffer.copy()
                current_agent['tools'] = tools_buffer.copy()
                agents[current_agent['name']] = current_agent
                skills_buffer = []
                tools_buffer = []

            current_agent = {
                'name': line[4:].strip(),
                'role': '',
                'provider': 'openrouter',
                'model': 'anthropic/claude-sonnet-4',
                'temperature': 0.7,
                'maxTokens': 4096,
                'systemPrompt': '',
                'skills': [],
                'tools': []
            }
            current_section = 'header'
            continue

        if not current_agent:
            continue

        # Section headers
        if line.startswith('**') and line.endswith('**'):
            section_name = line.replace('**', '').lower().replace(':', '').strip()

            if section_name == 'role':
                current_section = 'role'
                continue
            elif section_name == 'model configuration':
                current_section = 'config'
                continue
            elif section_name == 'system prompt':
                current_section = 'prompt'
                # Check for code block
                if i + 1 < len(lines) and lines[i + 1].startswith('```'):
                    i += 2  # Skip ```
                    while i < len(lines) and not lines[i].startswith('```'):
                        system_prompt_buffer.append(lines[i])
                        i += 1
                continue
            elif section_name == 'skills':
                current_section = 'skills'
                continue
            elif section_name == 'tools':
                current_section = 'tools'
                continue

        # Parse role
        if current_section == 'role' and line.strip() and not line.startswith('**'):
            current_agent['role'] = line.strip()

        # Parse model configuration
        if current_section == 'config':
            provider_match = line.match(r'Provider:\s*(.+)') if hasattr(line, 'match') else None
            if 'Provider:' in line:
                current_agent['provider'] = line.split(':', 1)[1].strip()
            if 'Model:' in line:
                current_agent['model'] = line.split(':', 1)[1].strip()
            if 'Temperature:' in line:
                current_agent['temperature'] = float(line.split(':', 1)[1].strip())
            if 'Max Tokens:' in line:
                current_agent['maxTokens'] = int(line.split(':', 1)[1].strip())

        # Parse skills
        if current_section == 'skills':
            if line.startswith('- '):
                parts = line[2:].split(':', 1)
                if len(parts) == 2:
                    skill_id = parts[0].strip().strip('`')
                    skill_desc = parts[1].strip()
                    skills_buffer.append({
                        'id': skill_id,
                        'name': skill_id.replace('-', ' ').title(),
                        'description': skill_desc
                    })

        # Parse tools
        if current_section == 'tools':
            if line.startswith('- '):
                tool_name = line[2:].strip().strip('`').split(':')[0].strip()
                tools_buffer.append(tool_name)

    # Save last agent
    if current_agent:
        if system_prompt_buffer:
            current_agent['systemPrompt'] = '\n'.join(system_prompt_buffer).strip()
        current_agent['skills'] = skills_buffer.copy()
        current_agent['tools'] = tools_buffer.copy()
        agents[current_agent['name']] = current_agent

    return agents

def generate_agents_md(agents: dict[str, dict[str, Any]]) -> str:
    """Generate AGENTS.md content from agent configurations."""
    lines = []
    lines.append("# AGENTS.md")
    lines.append("")
    lines.append("Agent configuration file for QuantMindX IDE.")
    lines.append("")
    lines.append("## Agent Definitions")
    lines.append("")

    for agent_name, agent in agents.items():
        lines.append(f"### {agent['name']}")
        lines.append("")
        lines.append(f"**Role**: {agent['role']}")
        lines.append("")
        lines.append("**Model Configuration**:")
        lines.append(f"- Provider: {agent['provider']}")
        lines.append(f"- Model: {agent['model']}")
        lines.append(f"- Temperature: {agent['temperature']}")
        lines.append(f"- Max Tokens: {agent['maxTokens']}")
        lines.append("")
        lines.append("**System Prompt**:")
        lines.append("```")
        lines.append(agent['systemPrompt'])
        lines.append("```")
        lines.append("")
        lines.append("**Skills**:")
        for skill in agent.get('skills', []):
            lines.append(f"- `{skill['id']}`: {skill['description']}")
        lines.append("")
        lines.append("**Tools**:")
        for tool in agent.get('tools', []):
            lines.append(f"- {tool}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return '\n'.join(lines)
