"""
TrendScanner: Trend-based trading opportunity detection.

Provides trend analysis and pattern detection including:
- Trend direction detection
- Moving average analysis
- Pattern recognition
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TrendScanner:
    """
    Trend-based scanner for detecting directional opportunities.

    Focuses on trend analysis including:
    - Trend direction detection
    - Moving average crossovers
    - Pattern recognition
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        db_manager: Optional[Any] = None,
    ):
        self.symbols = symbols or ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
        self.db_manager = db_manager

    def scan_trends(self) -> List[Dict[str, Any]]:
        """
        Scan all symbols for trend opportunities.

        Returns:
            List of trend scan results
        """
        results = []

        for symbol in self.symbols:
            result = self.scan_trend(symbol)
            if result:
                results.append(result)

        return results

    def scan_trend(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Scan a single symbol for trend opportunities.

        Args:
            symbol: Trading symbol to scan

        Returns:
            Trend scan result or None
        """
        try:
            # Get price data
            price_data = self._get_price_data(symbol)
            if not price_data:
                return None

            # Analyze trend
            analysis = {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": self._detect_trend_direction(price_data),
                "strength": self._calculate_trend_strength(price_data),
                "moving_averages": self._calculate_moving_averages(price_data),
            }

            return analysis

        except Exception as e:
            logger.error(f"Error scanning trend for {symbol}: {e}")
            return None

    def _get_price_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Get recent price data for symbol."""
        # Placeholder - actual implementation would query database
        return []

    def _detect_trend_direction(self, price_data: List[Dict[str, Any]]) -> str:
        """
        Detect trend direction using price action.

        Returns:
            "uptrend", "downtrend", or "neutral"
        """
        if not price_data or len(price_data) < 20:
            return "neutral"

        try:
            recent_prices = [bar["close"] for bar in price_data[-20:]]

            # Simple trend detection using linear regression slope
            n = len(recent_prices)
            x_sum = sum(range(n))
            y_sum = sum(recent_prices)
            xy_sum = sum(i * p for i, p in enumerate(recent_prices))
            x2_sum = sum(i * i for i in range(n))

            slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)

            # Normalize slope by price level
            avg_price = y_sum / n
            normalized_slope = slope / avg_price

            if normalized_slope > 0.001:
                return "uptrend"
            elif normalized_slope < -0.001:
                return "downtrend"
            else:
                return "neutral"

        except Exception:
            return "neutral"

    def _calculate_trend_strength(self, price_data: List[Dict[str, Any]]) -> float:
        """
        Calculate trend strength (0-1).

        Returns:
            Strength value between 0 and 1
        """
        if not price_data or len(price_data) < 20:
            return 0.0

        try:
            recent_prices = [bar["close"] for bar in price_data[-20:]]

            # Calculate using standard deviation of returns
            returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                      for i in range(1, len(recent_prices))]

            if not returns:
                return 0.0

            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5

            # Strength is related to consistency of direction
            positive_returns = sum(1 for r in returns if r > 0)
            strength = abs(2 * (positive_returns / len(returns)) - 1)

            return min(1.0, strength)

        except Exception:
            return 0.0

    def _calculate_moving_averages(self, price_data: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """
        Calculate common moving averages.

        Returns:
            Dictionary with MA values
        """
        if not price_data:
            return {"ma20": None, "ma50": None, "ma200": None}

        try:
            closes = [bar["close"] for bar in price_data]

            ma20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
            ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
            ma200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None

            return {
                "ma20": ma20,
                "ma50": ma50,
                "ma200": ma200,
            }

        except Exception:
            return {"ma20": None, "ma50": None, "ma200": None}

    def detect_ma_crossover(self, price_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Detect moving average crossovers.

        Returns:
            Crossover signal or None
        """
        if not price_data or len(price_data) < 50:
            return None

        try:
            ma_values = self._calculate_moving_averages(price_data)

            if ma_values["ma20"] is None or ma_values["ma50"] is None:
                return None

            # Get previous values
            closes = [bar["close"] for bar in price_data]
            prev_ma20 = sum(closes[-21:-1]) / 20
            prev_ma50 = sum(closes[-51:-1]) / 50

            current_ma20 = ma_values["ma20"]
            current_ma50 = ma_values["ma50"]

            # Detect crossover
            if prev_ma20 <= prev_ma50 and current_ma20 > current_ma50:
                return {
                    "type": "golden_cross",
                    "direction": "bullish",
                    "ma20": current_ma20,
                    "ma50": current_ma50,
                }
            elif prev_ma20 >= prev_ma50 and current_ma20 < current_ma50:
                return {
                    "type": "death_cross",
                    "direction": "bearish",
                    "ma20": current_ma20,
                    "ma50": current_ma50,
                }

            return None

        except Exception:
            return None
