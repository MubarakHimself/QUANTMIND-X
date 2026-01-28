"""
QuantMindX Analyst Agent - Strategy Parameter Extractor

ONLY extracts:
- Entry logic (conditions for entering trades)
- Exit logic (conditions for exiting trades)
- Strategy-specific parameters (indicators, timeframes, filters)

NOT responsible for:
- Risk management (handled by Kelly component)
- Position sizing (handled by position sizing component)
- Drawdown limits (handled separately)
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .base import BaseAgent, AgentConfig
from .skills import Skill, SkillRegistry, create_search_skill
from .tools import Tool, ToolRegistry
from .mcp import MCPClient


class AnalystAgent(BaseAgent):
    """
    QuantMindX Strategy Analyst - Extracts strategy parameters.

    ONLY responsible for:
    - Entry logic extraction
    - Exit logic extraction
    - Strategy parameters (indicators, timeframes, filters)

    Does NOT handle risk management, position sizing, etc.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = "qwen/qwen3-vl-30b-a3b-thinking",
        kb_client=None,
        mcp_client: MCPClient = None
    ):
        config = AgentConfig(
            name="analyst",
            description="QuantMindX strategy parameter extractor - entry/exit logic only",
            model=model,
            api_key=api_key,
            use_kb=True,
            kb_collection="analyst_kb",
            use_mcp=True
        )

        super().__init__(config, kb_client=kb_client, mcp_client=mcp_client)

    def _register_capabilities(self):
        """Register analyst-specific skills and tools."""
        # KB Search skill
        kb_search = create_search_skill(self.kb_client)
        self.add_skill(kb_search)

        # Strategy Extraction skill
        strategy_extract = Skill(
            name="extract_strategy",
            description="Extract entry/exit logic and strategy parameters from content",
            category="extraction",
            execute=self.extract_strategy,
            parameters={
                "content": {"type": "string", "description": "Video/NPRD content"},
                "source_name": {"type": "string", "description": "Strategy name"}
            }
        )
        self.add_skill(strategy_extract)

        # Register tools
        from .tools import (
            read_file_tool, write_file_tool, search_kb_tool
        )

        self.add_tool(read_file_tool)
        self.add_tool(write_file_tool)
        self.add_tool(search_kb_tool)

        # MCP tools
        if self.mcp_client:
            for mcp_tool in self.mcp_client.list_tools():
                self.capabilities.mcp_tools.append(
                    f"{mcp_tool.server_name}.{mcp_tool.name}"
                )

    def get_system_prompt(self) -> str:
        """Get analyst agent system prompt - focused ONLY on strategy extraction."""
        return """You are QuantMindX Strategy Analyst, an expert at extracting trading strategy parameters.

## YOUR RESPONSIBILITY (ONLY)

Extract from the content:
1. **Entry Logic** - Conditions for entering trades
2. **Exit Logic** - Conditions for exiting trades
3. **Strategy Parameters** - Indicators, timeframes, filters specific to THIS strategy

## NOT YOUR RESPONSIBILITY

These are handled by separate components - DO NOT extract:
- Risk management (Kelly criterion component handles this)
- Position sizing (position sizing component handles this)
- Drawdown limits (separate component)
- Stop loss levels (managed elsewhere)
- Take profit levels (managed elsewhere)

## EXTRACT ONLY

### Entry Logic
- Primary trigger conditions
- Confirmation signals
- Time filters (session, time of day)
- Market condition filters

### Exit Logic
- Exit trigger conditions
- Trailing conditions (if mentioned)

### Strategy Parameters
- Indicators (RSI, MACD, ATR, EMA, etc.)
- Timeframes
- Specific values/settings

## OUTPUT FORMAT

Return JSON:
```json
{
  "strategy_name": "Name from content",
  "entry_logic": {
    "primary_trigger": "condition",
    "confirmations": ["list"],
    "time_filters": {"session": "...", "time": "..."},
    "market_filters": {"volatility": "...", "trend": "..."}
  },
  "exit_logic": {
    "trigger": "condition",
    "trailing": "method if any"
  },
  "parameters": {
    "indicators": [
      {"name": "RSI", "period": 14, "purpose": "..."},
      {"name": "EMA", "period": 20, "purpose": "..."}
    ],
    "timeframe": "H1",
    "specific_settings": {"key": "value"}
  }
}
```

Be precise. Only extract what's in the content. Do NOT add risk management, position sizing, etc."""

    def extract_strategy(
        self,
        content: str,
        source_name: str = "Unknown Strategy"
    ) -> Dict[str, Any]:
        """
        Extract strategy parameters from content.

        ONLY extracts: entry logic, exit logic, strategy parameters.
        Does NOT extract: risk management, position sizing, etc.

        Args:
            content: Video/NPRD content
            source_name: Strategy name

        Returns:
            Strategy specification JSON
        """
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

        extract_prompt = f"""Extract the trading strategy parameters from this content.

**Source:** {source_name}

**Content:**
{content[:8000]}

Extract ONLY:
1. Entry logic (triggers, confirmations, filters)
2. Exit logic (triggers, trailing)
3. Strategy parameters (indicators, timeframes, settings)

DO NOT extract risk management, position sizing, stop loss, take profit - those are separate components.

Return as JSON."""

        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=extract_prompt)
        ]

        try:
            response = self._llm.invoke(messages)
            result = response.content

            # Extract JSON
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

            strategy = json.loads(result)
            strategy["source"] = source_name
            strategy["extracted_at"] = datetime.now().isoformat()

            return strategy

        except Exception as e:
            return {
                "error": f"Extraction failed: {str(e)}",
                "source": source_name,
                "entry_logic": {},
                "exit_logic": {},
                "parameters": {}
            }

    def extract_from_nprd(
        self,
        nprd_path: Path,
        output_path: Path = None
    ) -> Dict[str, Any]:
        """
        Extract strategy from NPRD file.

        Args:
            nprd_path: Path to NPRD JSON
            output_path: Optional output path for JSON

        Returns:
            Strategy specification
        """
        # Load NPRD
        with open(nprd_path, "r") as f:
            nprd_data = json.load(f)

        # Get content
        transcript = nprd_data.get("transcript", "")
        ocr_text = nprd_data.get("ocr_text", "")
        keywords = nprd_data.get("keywords", [])

        content = transcript or ocr_text or f"Video: {nprd_path.stem}\nKeywords: {keywords}"

        # Extract strategy
        strategy_name = nprd_path.stem
        strategy = self.extract_strategy(content, strategy_name)

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(strategy, f, indent=2)

        return strategy

    def search_kb(self, query: str, n: int = 3, collection: str = "analyst_kb") -> List[Dict]:
        """Search knowledge base for reference."""
        if not self.kb_client:
            return []
        return self.kb_client.search(query, collection=collection, n=n)

    def chat(self, user_message: str, use_kb: bool = True) -> str:
        """Chat about strategy extraction."""
        return super().chat(user_message, use_kb=use_kb)

    def list_capabilities(self) -> Dict[str, List[str]]:
        """List agent capabilities."""
        return {
            "skills": self.list_skills(),
            "tools": self.list_tools(),
            "mcp_tools": self.capabilities.mcp_tools
        }

    def get_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": self.config.name,
            "description": "Strategy parameter extractor - entry/exit logic only",
            "responsibilities": [
                "Entry logic extraction",
                "Exit logic extraction",
                "Strategy parameters (indicators, timeframes)"
            ],
            "not_responsible_for": [
                "Risk management (Kelly component)",
                "Position sizing (position sizing component)",
                "Drawdown limits",
                "Stop loss/take profit levels"
            ],
            "model": self.config.model,
            "capabilities": self.list_capabilities(),
            "kb_collection": self.config.kb_collection
        }


def create_analyst_agent(
    api_key: str = None,
    model: str = "qwen/qwen3-vl-30b-a3b-thinking",
    kb_client=None,
    mcp_client: MCPClient = None
) -> AnalystAgent:
    """Create AnalystAgent for strategy extraction."""
    return AnalystAgent(
        api_key=api_key,
        model=model,
        kb_client=kb_client,
        mcp_client=mcp_client
    )


# Singleton
_analyst_agent_instance: Optional[AnalystAgent] = None


def get_analyst_agent(
    api_key: str = None,
    kb_client=None,
    force_new: bool = False
) -> AnalystAgent:
    """Get singleton AnalystAgent."""
    global _analyst_agent_instance
    if _analyst_agent_instance is None or force_new:
        _analyst_agent_instance = create_analyst_agent(
            api_key=api_key,
            kb_client=kb_client
        )
    return _analyst_agent_instance
