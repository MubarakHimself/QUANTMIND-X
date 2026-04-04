"""
Department Skill Registry — Maps agentic skills to departments.

Each department head and its sub-agents have access to a curated set of skills.
Skills are registered here and indexed in the system prompts (types.py).

Usage:
    from src.agents.skills.department_skills import get_department_skills, register_all_skills
    skills = get_department_skills("research")
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Department → Skill mapping
# ============================================================================

DEPARTMENT_SKILL_REGISTRY: Dict[str, List[Dict[str, Any]]] = {
    "research": [
        {
            "name": "financial_data_fetch",
            "description": "Fetch OHLCV, tick, or fundamental data for any symbol and timeframe",
            "slash_command": "/fetch-data",
            "parameters": ["symbol", "data_type", "timeframe", "start_date", "end_date"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "pattern_scanner",
            "description": "Scan price data for chart patterns (head_shoulders, triangles, double tops/bottoms)",
            "slash_command": "/scan-patterns",
            "parameters": ["prices", "pattern_type"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "statistical_edge",
            "description": "Calculate alpha, beta, Sharpe, win rate, profit factor from return series",
            "slash_command": "/stat-edge",
            "parameters": ["returns", "benchmark_returns"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "hypothesis_document_writer",
            "description": "Generate a structured TRD hypothesis document from research findings",
            "slash_command": "/write-hypothesis",
            "parameters": ["research_data"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "news_classifier",
            "description": "Classify news headlines by sentiment (positive/negative/neutral) and category",
            "slash_command": "/classify-news",
            "parameters": ["headlines"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "institutional_data_fetch",
            "description": "Fetch institutional-grade data (Bloomberg, Reuters, FactSet connectors)",
            "slash_command": "/institutional-data",
            "parameters": ["data_source", "query_params"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "knowledge_search",
            "description": "Search the knowledge base (PageIndex articles, Obsidian vault, semantic memory)",
            "slash_command": "/search-kb",
            "parameters": ["query", "top_k"],
            "module": "src.agents.skills.builtin_skills",
        },
        {
            "name": "research_summary",
            "description": "Generate a research summary on a market topic at brief or deep depth",
            "slash_command": "/research-summary",
            "parameters": ["topic", "depth"],
            "module": "src.agents.skills.builtin_skills",
        },
    ],
    "development": [
        {
            "name": "mql5_generator",
            "description": "Generate MQL5 Expert Advisor code from a strategy specification (TRD)",
            "slash_command": "/generate-ea",
            "parameters": ["strategy_spec"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "backtest_launcher",
            "description": "Queue and launch a backtest for a symbol/strategy/timeframe combo",
            "slash_command": "/run-backtest",
            "parameters": ["symbol", "strategy_params", "start_date", "end_date", "timeframe"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "strategy_optimizer",
            "description": "Optimize strategy parameters targeting Sharpe, profit, or win rate",
            "slash_command": "/optimize",
            "parameters": ["strategy_params", "optimization_target"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "pinescript_generate",
            "description": "Generate TradingView Pine Script v5 from natural language description",
            "slash_command": "/generate-pine",
            "parameters": ["strategy_description"],
            "module": "src.agents.departments.subagents.pinescript_subagent",
        },
        {
            "name": "pinescript_convert",
            "description": "Convert MQL5 code to Pine Script v5",
            "slash_command": "/convert-to-pine",
            "parameters": ["mql5_code"],
            "module": "src.agents.departments.subagents.pinescript_subagent",
        },
        {
            "name": "validate_mql5_syntax",
            "description": "Validate MQL5 code syntax and compilation readiness",
            "slash_command": "/validate-mql5",
            "parameters": ["code"],
            "module": "src.agents.tools.mcp.mt5_compiler",
        },
        {
            "name": "compile_mql5_code",
            "description": "Compile MQL5 code via MT5 compiler and return errors/warnings",
            "slash_command": "/compile-mql5",
            "parameters": ["code", "filename"],
            "module": "src.agents.tools.mcp.mt5_compiler",
        },
    ],
    "trading": [
        {
            "name": "calendar_gate_check",
            "description": "Check if trading is allowed (session hours, weekend, news blackout windows)",
            "slash_command": "/gate-check",
            "parameters": ["current_time", "calendar_config"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "calculate_position_size",
            "description": "Calculate risk-adjusted position size (lots) from account balance and risk %",
            "slash_command": "/position-size",
            "parameters": ["account_balance", "risk_percent", "stop_loss_pips", "pip_value"],
            "module": "src.agents.skills.builtin_skills",
        },
        {
            "name": "calculate_rsi",
            "description": "Calculate RSI indicator value and generate overbought/oversold signal",
            "slash_command": "/rsi",
            "parameters": ["prices", "period"],
            "module": "src.agents.skills.builtin_skills",
        },
        {
            "name": "detect_support_resistance",
            "description": "Detect support and resistance levels from price data",
            "slash_command": "/sr-levels",
            "parameters": ["highs", "lows", "closes", "lookback_period"],
            "module": "src.agents.skills.builtin_skills",
        },
        {
            "name": "bot_analysis",
            "description": "Analyse an underperforming bot and produce a Bot Analysis Brief",
            "slash_command": "/analyse-bot",
            "parameters": ["bot_metadata"],
            "module": "src.agents.departments.subagents.bot_analyst_subagent",
        },
    ],
    "risk": [
        {
            "name": "risk_evaluator",
            "description": "Evaluate trade risk: risk %, R:R ratio, risk level, approve/review/reject verdict",
            "slash_command": "/evaluate-risk",
            "parameters": ["position", "account_balance"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "report_writer",
            "description": "Generate performance, risk, trade, or summary reports from structured data",
            "slash_command": "/write-report",
            "parameters": ["report_type", "data"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "statistical_edge",
            "description": "Calculate Sharpe, win rate, profit factor for risk assessment",
            "slash_command": "/stat-edge",
            "parameters": ["returns", "benchmark_returns"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "backtest_report",
            "description": "Generate a structured backtest report with IS/OOS comparison and improvement suggestions",
            "slash_command": "/backtest-report",
            "parameters": ["strategy_id", "trd_data", "backtest_result", "sit_result"],
            "module": "src.agents.departments.subagents.backtest_report_subagent",
        },
        {
            "name": "calculate_position_size",
            "description": "Calculate risk-adjusted position size using fractional Kelly methodology",
            "slash_command": "/position-size",
            "parameters": ["account_balance", "risk_percent", "stop_loss_pips", "pip_value"],
            "module": "src.agents.skills.builtin_skills",
        },
    ],
    "portfolio": [
        {
            "name": "report_writer",
            "description": "Generate portfolio performance and attribution reports",
            "slash_command": "/write-report",
            "parameters": ["report_type", "data"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "statistical_edge",
            "description": "Calculate portfolio-level statistical metrics (alpha, beta, Sharpe)",
            "slash_command": "/stat-edge",
            "parameters": ["returns", "benchmark_returns"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "risk_evaluator",
            "description": "Evaluate portfolio-level risk exposure and concentration",
            "slash_command": "/evaluate-risk",
            "parameters": ["position", "account_balance"],
            "module": "src.agents.skills.core_skills",
        },
        {
            "name": "financial_data_fetch",
            "description": "Fetch market data for correlation analysis and performance tracking",
            "slash_command": "/fetch-data",
            "parameters": ["symbol", "data_type", "timeframe", "start_date", "end_date"],
            "module": "src.agents.skills.core_skills",
        },
    ],
}


# ============================================================================
# Department → MCP Config mapping
# ============================================================================

DEPARTMENT_MCP_CONFIG_SOURCES: Dict[str, str] = {
    "research": "analyst-mcp.json",
    "development": "quantcode-mcp.json",
    "trading": "executor-mcp.json",
    "risk": "analyst-mcp.json",
    "portfolio": "copilot-mcp.json",
    "floor_manager": "copilot-mcp.json",
}

DEPARTMENT_MCP_CONFIGS: Dict[str, str] = {
    "research": "research-mcp.json",
    "development": "development-mcp.json",
    "trading": "trading-mcp.json",
    "risk": "risk-mcp.json",
    "portfolio": "portfolio-mcp.json",
    "floor_manager": "floor-manager-mcp.json",
}

# Department → MCP server IDs that should be loaded
DEPARTMENT_MCP_SERVERS: Dict[str, List[str]] = {
    "research": [
        "filesystem", "github", "context7", "brave-search",
        "memory", "sequential-thinking", "pageindex-articles",
        "backtest-server", "obsidian",
    ],
    "development": [
        "filesystem", "github", "context7", "mt5-compiler",
        "backtest-server",
    ],
    "trading": [],  # Trading uses direct MT5 API, not MCP
    "risk": [
        "filesystem", "backtest-server", "sequential-thinking",
    ],
    "portfolio": [
        "filesystem", "pageindex-articles", "sequential-thinking",
        "obsidian",
    ],
    "floor_manager": [
        "filesystem", "github", "context7", "sequential-thinking",
        "pageindex-articles", "obsidian",
    ],
}


def get_department_skills(department: str) -> List[Dict[str, Any]]:
    """Get skill definitions for a department."""
    return DEPARTMENT_SKILL_REGISTRY.get(department, [])


def get_department_mcp_config(department: str) -> str:
    """Get the MCP config filename for a department."""
    return DEPARTMENT_MCP_CONFIGS.get(department, "floor-manager-mcp.json")


def get_department_mcp_config_source(department: str) -> str:
    """Get the compatibility MCP config source filename for a department."""
    return DEPARTMENT_MCP_CONFIG_SOURCES.get(department, "copilot-mcp.json")


def get_department_mcp_servers(department: str) -> List[str]:
    """Get the list of MCP server IDs for a department."""
    return DEPARTMENT_MCP_SERVERS.get(department, [])


def get_skill_index_for_prompt(department: str) -> str:
    """
    Generate a formatted skill index string for inclusion in system prompts.

    Returns a markdown-formatted list of skills with descriptions and slash commands.
    """
    skills = get_department_skills(department)
    if not skills:
        return "No skills registered for this department."

    lines = []
    for skill in skills:
        cmd = skill.get("slash_command", "")
        desc = skill.get("description", "")
        params = ", ".join(skill.get("parameters", []))
        lines.append(f"- **{skill['name']}** (`{cmd}`) — {desc}")
        if params:
            lines.append(f"  Parameters: {params}")

    return "\n".join(lines)


def get_mcp_index_for_prompt(department: str) -> str:
    """
    Generate a formatted MCP server index string for inclusion in system prompts.
    """
    servers = get_department_mcp_servers(department)
    if not servers:
        return "No MCP servers assigned to this department."

    return ", ".join(servers)


def register_all_skills() -> None:
    """
    Register all department skills with the global SkillManager.

    Call this at application startup to populate the skill registry.
    """
    try:
        from src.agents.skills.skill_manager import get_skill_manager, SkillMetadata
        manager = get_skill_manager()

        for department, skills in DEPARTMENT_SKILL_REGISTRY.items():
            for skill_def in skills:
                try:
                    import importlib
                    mod = importlib.import_module(skill_def["module"])
                    func = getattr(mod, skill_def["name"], None)
                    if func and callable(func):
                        metadata = SkillMetadata(
                            name=skill_def["name"],
                            description=skill_def["description"],
                            category=department,
                            departments=[department],
                            parameters={"required": skill_def.get("parameters", [])},
                            slash_command=skill_def.get("slash_command", ""),
                        )
                        manager.register(
                            name=skill_def["name"],
                            func=func,
                            metadata=metadata,
                        )
                except Exception as e:
                    logger.debug(f"Could not register skill {skill_def['name']}: {e}")

        logger.info(f"Registered skills for {len(DEPARTMENT_SKILL_REGISTRY)} departments")
    except Exception as e:
        logger.warning(f"Skill registration failed: {e}")
