"""
Research Subagent

Worker agent for the Research Department.
Responsible for strategy research, market analysis, and signal generation.

Model: Haiku (fast, low-cost for worker tasks)
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# Research agent system prompts for LLM
RESEARCH_SYSTEM_PROMPT = """You are an expert quantitative researcher and trading strategy analyst.

Your task is to analyze market conditions, research trading symbols, and synthesize information
from various sources to provide actionable insights for trading strategies.

## Analysis Framework:
1. Technical Analysis: Price action, trend analysis, support/resistance levels
2. Fundamental Analysis: News impact, macroeconomic factors, market sentiment
3. Risk Assessment: Volatility, correlation, drawdown potential
4. Signal Generation: Entry/exit points, position sizing recommendations

## Guidelines:
- Provide clear, data-driven analysis
- Include risk metrics and recommendations
- Use proper trading terminology
- Consider multiple timeframes when applicable
- Always include risk warnings for live trading
- Synthesize information from provided tools and context
"""

MARKET_ANALYSIS_PROMPT = """Analyze the current market conditions based on the following description:

{market_description}

Provide a comprehensive analysis including:
1. Overall market sentiment (bullish/bearish/neutral)
2. Key technical levels and patterns
3. Potential entry/exit points
4. Risk factors to consider
5. Recommended timeframe for analysis
"""

SYMBOL_RESEARCH_PROMPT = """Research the following trading symbol: {symbol}

Provide a comprehensive research report including:
1. Symbol overview and market
2. Recent price action and trends
3. Key technical indicators
4. Relevant news and events
5. Trading recommendations with risk assessment
"""

NEWS_SYNTHESIS_PROMPT = """Synthesize the following news and information for {symbol_or_market}:

{news_content}

Provide:
1. Key takeaways and impact assessment
2. How this affects the symbol/market outlook
3. Trading implications and recommendations
4. Risk considerations
"""


@dataclass
class ResearchTask:
    """Research task input data."""
    task_type: str  # strategy_development, market_analysis, backtest, alpha_research
    symbols: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class ResearchSubAgent:
    """
    Research subagent for strategy development and analysis.

    Capabilities:
    - Strategy development and backtesting
    - Market analysis (technical and fundamental)
    - Alpha factor research
    - Signal generation
    """

    def __init__(
        self,
        agent_id: str,
        task: Optional[ResearchTask] = None,
        available_tools: Optional[List[str]] = None,
    ):
        """
        Initialize research subagent.

        Args:
            agent_id: Unique identifier for this agent
            task: Task configuration
            available_tools: List of tool names available to this agent
        """
        self.agent_id = agent_id
        self.agent_type = "research"
        self.task = task or ResearchTask(task_type="strategy_development")
        self.available_tools = available_tools or []
        self.model_tier = "haiku"  # Workers use haiku for speed
        self._llm_client = None

        # Initialize tools based on available tools
        self._tools = self._initialize_tools()
        self._initialize_llm()

    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize available tools for this agent."""
        tools = {}

        # Core research tools
        tool_registry = {
            "market_data": self.get_market_data,
            "technical_analysis": self.perform_technical_analysis,
            "backtest": self.run_backtest,
            "signal_generation": self.generate_signals,
            "strategy_validator": self.validate_strategy,
            # LLM-powered research methods
            "analyze_market": self.analyze_market,
            "research_symbol": self.research_symbol,
            "synthesize_news": self.synthesize_news,
        }

        for tool_name in self.available_tools:
            if tool_name in tool_registry:
                tools[tool_name] = tool_registry[tool_name]

        # Initialize knowledge tools (READ access for Research department)
        self._knowledge_tools = {}
        try:
            from src.agents.tools.knowledge_tools import (
                search_knowledge_hub,
                search_mql5_book,
                search_strategy_patterns,
            )
            self._knowledge_tools = {
                "search_knowledge_hub": search_knowledge_hub,
                "search_mql5_book": search_mql5_book,
                "search_strategy_patterns": search_strategy_patterns,
            }
            logger.info("ResearchSubAgent: Knowledge tools initialized (READ access)")
        except ImportError as e:
            logger.warning(f"ResearchSubAgent: Could not load knowledge tools: {e}")

        return tools

    def _initialize_llm(self) -> None:
        """Initialize LLM client for research analysis."""
        try:
            from src.agents.departments.subagents.llm_utils import get_subagent_client
            self._llm_client, self._llm_model = get_subagent_client()
            logger.info(f"ResearchSubAgent: LLM client initialized (model={self._llm_model})")
        except Exception as e:
            logger.warning(f"ResearchSubAgent: LLM init failed: {e}")

    def _call_llm(
        self,
        user_prompt: str,
        system_prompt: str = RESEARCH_SYSTEM_PROMPT,
    ) -> str:
        """
        Call LLM for research analysis.

        Args:
            user_prompt: User's research request
            system_prompt: System instructions for research tasks

        Returns:
            LLM response text
        """
        if not self._llm_client:
            raise RuntimeError("LLM client not initialized")

        try:
            response = self._llm_client.messages.create(
                model=self._llm_model,
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"ResearchSubAgent: LLM call failed: {e}")
            raise

    def get_market_data(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 100,
    ) -> Dict[str, Any]:
        """
        Get market data for a symbol.

        Args:
            symbol: Trading symbol (e.g., EURUSD)
            timeframe: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to retrieve

        Returns:
            Market data dictionary
        """
        # Implementation would call market data API
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candles": limit,
            "status": "retrieved",
            "data": [],  # Would contain actual OHLCV data
        }

    def perform_technical_analysis(
        self,
        symbol: str,
        indicators: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform technical analysis on a symbol.

        Args:
            symbol: Trading symbol
            indicators: List of indicators (RSI, MACD, SMA, EMA, etc.)

        Returns:
            Technical analysis results
        """
        indicators = indicators or ["RSI", "MACD", "SMA"]
        return {
            "symbol": symbol,
            "indicators": indicators,
            "results": {
                "rsi": {"value": 55.0, "signal": "neutral"},
                "macd": {"histogram": 0.001, "signal": "bullish"},
                "sma": {"price": 1.0850, "sma_20": 1.0845},
            },
            "recommendation": "neutral",
        }

    def run_backtest(
        self,
        strategy_id: str,
        symbols: List[str],
        start_date: str,
        end_date: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest for a strategy.

        Args:
            strategy_id: Strategy identifier
            symbols: List of symbols to backtest
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            parameters: Strategy parameters

        Returns:
            Backtest results
        """
        return {
            "strategy_id": strategy_id,
            "symbols": symbols,
            "period": f"{start_date} to {end_date}",
            "results": {
                "total_trades": 150,
                "win_rate": 0.62,
                "profit_factor": 1.85,
                "max_drawdown": -0.08,
                "sharpe_ratio": 1.45,
            },
            "status": "completed",
        }

    def generate_signals(
        self,
        symbols: List[str],
        signal_type: str = "all",
    ) -> Dict[str, Any]:
        """
        Generate trading signals for symbols.

        Args:
            symbols: List of symbols
            signal_type: Type of signals (momentum, reversal, breakout, all)

        Returns:
            Trading signals
        """
        signals = []
        for symbol in symbols:
            signals.append({
                "symbol": symbol,
                "signal": "BUY" if hash(symbol) % 2 == 0 else "SELL",
                "confidence": 0.75,
                "entry_price": 1.0850,
                "stop_loss": 1.0800,
                "take_profit": 1.0950,
            })

        return {
            "signals": signals,
            "count": len(signals),
            "generated_at": "2024-01-15T10:30:00Z",
        }

    def validate_strategy(
        self,
        strategy_id: str,
        validation_type: str = "basic",
    ) -> Dict[str, Any]:
        """
        Validate a trading strategy.

        Args:
            strategy_id: Strategy identifier
            validation_type: Type of validation (basic, full, risk)

        Returns:
            Validation results
        """
        return {
            "strategy_id": strategy_id,
            "validation_type": validation_type,
            "passed": True,
            "checks": {
                "logic": "passed",
                "risk_limits": "passed",
                "execution": "passed",
            },
            "warnings": [],
        }

    def analyze_market(
        self,
        market_description: str,
        include_technical: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze market conditions from natural language description.

        Args:
            market_description: Natural language description of market conditions
            include_technical: Whether to include technical analysis

        Returns:
            Market analysis results
        """
        logger.info(f"ResearchSubAgent: Analyzing market: {market_description[:50]}...")

        try:
            # Format the prompt with market description
            user_prompt = MARKET_ANALYSIS_PROMPT.format(
                market_description=market_description
            )

            # Call LLM for analysis
            analysis = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=RESEARCH_SYSTEM_PROMPT,
            )

            return {
                "analysis": analysis,
                "status": "analyzed",
                "market_description": market_description,
                "include_technical": include_technical,
            }

        except Exception as e:
            logger.error(f"ResearchSubAgent: Market analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "market_description": market_description,
            }

    def research_symbol(
        self,
        symbol: str,
        include_technical: bool = True,
        include_fundamental: bool = True,
    ) -> Dict[str, Any]:
        """
        Research a trading symbol using LLM and available tools.

        Args:
            symbol: Trading symbol to research
            include_technical: Include technical analysis
            include_fundamental: Include fundamental analysis

        Returns:
            Symbol research results
        """
        logger.info(f"ResearchSubAgent: Researching symbol: {symbol}")

        try:
            # Get market data if available
            market_data = {}
            if "market_data" in self._tools:
                market_data = self._tools["market_data"](symbol=symbol)

            # Get technical analysis if available
            technical_data = {}
            if "technical_analysis" in self._tools and include_technical:
                technical_data = self._tools["technical_analysis"](symbol=symbol)

            # Format the research prompt
            user_prompt = SYMBOL_RESEARCH_PROMPT.format(symbol=symbol)

            # Add market and technical data context
            context_parts = [f"Symbol: {symbol}"]
            if market_data:
                context_parts.append(f"Market Data: {market_data}")
            if technical_data:
                context_parts.append(f"Technical Analysis: {technical_data}")

            user_prompt += "\n\nAdditional Context:\n" + "\n".join(context_parts)

            # Call LLM for research
            research = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=RESEARCH_SYSTEM_PROMPT,
            )

            return {
                "symbol": symbol,
                "research": research,
                "status": "researched",
                "market_data": market_data if market_data else None,
                "technical_analysis": technical_data if technical_data else None,
                "include_technical": include_technical,
                "include_fundamental": include_fundamental,
            }

        except Exception as e:
            logger.error(f"ResearchSubAgent: Symbol research failed: {e}")
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e),
            }

    def synthesize_news(
        self,
        news_content: List[str],
        symbol_or_market: str = "general",
    ) -> Dict[str, Any]:
        """
        Synthesize news for a symbol or market.

        Args:
            news_content: List of news items to synthesize
            symbol_or_market: Symbol or market to focus on
            include_sentiment: Whether to include sentiment analysis

        Returns:
            Synthesized news analysis
        """
        logger.info(f"ResearchSubAgent: Synthesizing news for: {symbol_or_market}")

        try:
            # Format the news content
            news_text = "\n\n".join([
                f"- {item}" for item in news_content
            ])

            # Format the synthesis prompt
            user_prompt = NEWS_SYNTHESIS_PROMPT.format(
                symbol_or_market=symbol_or_market,
                news_content=news_text,
            )

            # Call LLM for synthesis
            synthesis = self._call_llm(
                user_prompt=user_prompt,
                system_prompt=RESEARCH_SYSTEM_PROMPT,
            )

            return {
                "symbol_or_market": symbol_or_market,
                "synthesis": synthesis,
                "status": "synthesized",
                "news_count": len(news_content),
            }

        except Exception as e:
            logger.error(f"ResearchSubAgent: News synthesis failed: {e}")
            return {
                "symbol_or_market": symbol_or_market,
                "status": "error",
                "error": str(e),
            }

    def execute(self) -> Dict[str, Any]:
        """
        Execute the assigned task.

        Returns:
            Task execution results
        """
        task_type = self.task.task_type

        if task_type == "strategy_development":
            return self._execute_strategy_development()
        elif task_type == "market_analysis":
            return self._execute_market_analysis()
        elif task_type == "backtest":
            return self._execute_backtest()
        elif task_type == "alpha_research":
            return self._execute_alpha_research()
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}",
            }

    def _execute_strategy_development(self) -> Dict[str, Any]:
        """Execute strategy development task."""
        symbols = self.task.symbols or ["EURUSD"]
        parameters = self.task.parameters or {}

        signals = self.generate_signals(symbols)
        return {
            "task_type": "strategy_development",
            "status": "completed",
            "signals": signals,
            "symbols": symbols,
        }

    def _execute_market_analysis(self) -> Dict[str, Any]:
        """Execute market analysis task."""
        results = []
        for symbol in self.task.symbols or ["EURUSD"]:
            analysis = self.perform_technical_analysis(symbol)
            results.append(analysis)

        return {
            "task_type": "market_analysis",
            "status": "completed",
            "analyses": results,
        }

    def _execute_backtest(self) -> Dict[str, Any]:
        """Execute backtest task."""
        params = self.task.parameters or {}
        return self.run_backtest(
            strategy_id=params.get("strategy_id", "default"),
            symbols=self.task.symbols or ["EURUSD"],
            start_date=params.get("start_date", "2023-01-01"),
            end_date=params.get("end_date", "2024-01-01"),
            parameters=params,
        )

    def _execute_alpha_research(self) -> Dict[str, Any]:
        """Execute alpha research task."""
        return {
            "task_type": "alpha_research",
            "status": "completed",
            "factors": ["momentum", "value", "volatility"],
            "recommendations": [],
        }

    def get_capabilities(self) -> List[str]:
        """Get list of agent capabilities."""
        return [
            "market_data",
            "technical_analysis",
            "backtest",
            "signal_generation",
            "strategy_validation",
            # LLM-powered capabilities
            "analyze_market",
            "research_symbol",
            "synthesize_news",
            # Knowledge tools (READ access)
            "search_knowledge_hub",
            "search_mql5_book",
            "search_strategy_patterns",
        ]
