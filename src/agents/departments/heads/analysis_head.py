"""
Research Department Head

Responsible for:
- Strategy research and development
- Market analysis and signal generation
- Backtesting and validation
- Knowledge management

Workers:
- strategy_researcher: Develop new trading strategies
- market_analyst: Technical and fundamental analysis
- backtester: Run backtests on strategies
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import pandas as pd

from src.agents.departments.heads.base import DepartmentHead
from src.agents.departments.heads.signal_generator import (
    SignalGenerator,
    SignalConfig,
    SignalAction,
)
from src.agents.departments.types import Department, get_department_config

logger = logging.getLogger(__name__)


class ResearchHead(DepartmentHead):
    """
    Research Department Head.

    Handles strategy research, market analysis, and signal generation.
    """

    def __init__(self, mail_db_path: str = ".quantmind/department_mail.db"):
        """Initialize Research Department Head."""
        config = get_department_config(Department.RESEARCH)
        super().__init__(config=config, mail_db_path=mail_db_path)
        self.signal_generator = SignalGenerator()
        self.signal_config = SignalConfig()

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools for Research department.

        Returns:
            List of tool definitions
        """
        return [
            {
                "name": "analyze_market",
                "description": "Perform technical analysis on a symbol",
                "parameters": {
                    "symbol": "Trading symbol (e.g., EURUSD)",
                    "timeframe": "Timeframe (M1, M5, M15, H1, H4, D1)",
                },
            },
            {
                "name": "scan_sentiment",
                "description": "Scan market sentiment for a symbol",
                "parameters": {
                    "symbol": "Trading symbol",
                    "sources": "Sources to scan (news, social, all)",
                },
            },
            {
                "name": "monitor_news",
                "description": "Monitor news for trading signals",
                "parameters": {
                    "keywords": "Keywords to filter",
                    "symbols": "Symbols to track",
                },
            },
        ]

    def _fetch_price_data(
        self,
        symbol: str,
        timeframe: str = "H1",
        bars: int = 100,
    ) -> Optional[pd.Series]:
        """
        Fetch price data for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            bars: Number of bars to fetch

        Returns:
            Series of close prices or None if unavailable
        """
        # Try to get data from database or market data service
        # For now, return None to use stub data in tests
        try:
            # Attempt to get data from market_data module
            from src.database.repositories.market_data import MarketDataRepository

            repo = MarketDataRepository()
            df = repo.get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=bars
            )
            if df is not None and len(df) > 0:
                return df["close"]
        except Exception as e:
            logger.debug(f"Could not fetch market data: {e}")

        return None

    def analyze_market(
        self,
        symbol: str,
        timeframe: str = "H1",
    ) -> Dict[str, Any]:
        """
        Perform market analysis on a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe

        Returns:
            Analysis result
        """
        # Try to get real price data
        prices = self._fetch_price_data(symbol, timeframe)

        if prices is not None and len(prices) >= 50:
            # Generate signals using RSI and MACD
            signals = self.signal_generator.generate_all(prices)

            # Determine trend based on MACD
            macd_signal = signals["macd"]
            trend = "neutral"
            if macd_signal["histogram"] > 0:
                trend = "bullish"
            elif macd_signal["histogram"] < 0:
                trend = "bearish"

            return {
                "status": "analyzed",
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": {
                    "trend": trend,
                    "support": None,
                    "resistance": None,
                    "indicators": {
                        "rsi": signals["rsi"]["value"],
                        "macd": signals["macd"]["macd_line"],
                        "macd_signal": signals["macd"]["signal_line"],
                        "macd_histogram": signals["macd"]["histogram"],
                    },
                    "signal": signals["confluence"]["action"],
                    "signal_confidence": signals["confluence"]["confidence"],
                },
            }

        # Fallback: return structured placeholder when no data
        return {
            "status": "analyzed",
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {
                "trend": "neutral",
                "support": None,
                "resistance": None,
                "indicators": {},
                "signal": None,
            },
        }

    def scan_sentiment(
        self,
        symbol: str,
        sources: str = "all",
    ) -> Dict[str, Any]:
        """
        Scan sentiment for a symbol.

        Args:
            symbol: Trading symbol
            sources: Sources to scan

        Returns:
            Sentiment result
        """
        return {
            "status": "scanned",
            "symbol": symbol,
            "sources": sources,
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment": {
                "overall": "neutral",
                "score": 0.0,
                "news": None,
                "social": None,
            },
        }

    def generate_signal(
        self,
        symbol: str,
        analysis: Dict[str, Any],
        sentiment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a trading signal using RSI/MACD confluence.

        Args:
            symbol: Trading symbol
            analysis: Market analysis result with indicators
            sentiment: Optional sentiment result

        Returns:
            Trading signal
        """
        # Try to get price data for signal generation
        prices = self._fetch_price_data(symbol, analysis.get("timeframe", "H1"))

        if prices is not None and len(prices) >= 50:
            # Generate confluence signal
            signal = self.signal_generator.generate(prices, strategy="confluence")

            # Build trading signal with levels
            action = signal["action"]
            confidence = signal["confidence"]

            # Calculate entry, stop loss, take profit based on last price
            last_price = prices.iloc[-1]

            if action == SignalAction.BUY.value:
                entry = round(last_price * 1.001, 5)  # Slight above
                stop_loss = round(last_price * 0.99, 5)  # 1% stop
                take_profit = round(last_price * 1.02, 5)  # 2% target
            elif action == SignalAction.SELL.value:
                entry = round(last_price * 0.999, 5)
                stop_loss = round(last_price * 1.01, 5)
                take_profit = round(last_price * 0.98, 5)
            else:
                entry = None
                stop_loss = None
                take_profit = None

            return {
                "status": "generated",
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "signal": {
                    "action": action,
                    "confidence": confidence,
                    "entry": entry,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "reason": signal["reason"],
                    "rsi_signal": self.signal_generator.rsi_gen.generate(prices),
                    "macd_signal": self.signal_generator.macd_gen.generate(prices),
                },
            }

        # Fallback: return hold signal when no price data
        return {
            "status": "generated",
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "signal": {
                "action": "HOLD",
                "confidence": 0.0,
                "entry": None,
                "stop_loss": None,
                "take_profit": None,
                "reason": "Insufficient price data for signal generation",
            },
        }
