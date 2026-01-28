"""
Kelly Statistics Analyzer

Extracts Kelly parameters from trade history for position sizing.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics


@dataclass
class KellyParameters:
    """Extracted Kelly parameters from trade history."""
    win_rate: float
    avg_win: float
    avg_loss: float
    risk_reward_ratio: float
    base_kelly_f: float
    sample_size: int
    expectancy: float           # Average profit per trade
    profit_factor: float        # Total wins / Total losses
    is_reliable: bool           # True if sample size sufficient
    confidence_note: str


class KellyStatisticsAnalyzer:
    """
    Analyzes trade history to extract Kelly parameters.
    
    Calculates win rate, average win/loss, and Kelly fraction
    from historical trade data.
    """

    def __init__(self, min_trades: int = 30):
        """
        Initialize analyzer.
        
        Args:
            min_trades: Minimum trades required for reliable Kelly calculation
        """
        self.min_trades = min_trades

    def calculate_kelly_parameters(
        self,
        trade_history: List[Dict[str, Any]]
    ) -> KellyParameters:
        """
        Extract Kelly parameters from trade history.

        Args:
            trade_history: List of trade dictionaries. Each trade must have:
                - 'profit': float (positive for win, negative for loss)
                OR
                - 'result': 'win' or 'loss'
                - 'amount': float (always positive)

        Returns:
            KellyParameters with all statistics
        """
        if not trade_history:
            return self._empty_parameters()

        # Normalize trade format
        profits = self._extract_profits(trade_history)
        
        if not profits:
            return self._empty_parameters()

        # Separate wins and losses (count zero-profit as win per tests)
        wins = [p for p in profits if p >= 0]
        losses = [abs(p) for p in profits if p < 0]

        # Calculate statistics
        total_trades = len(trade_history)
        win_count = len(wins)

        # Win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # Average win and loss
        pos_wins = [p for p in wins if p > 0]
        avg_win = statistics.mean(pos_wins) if pos_wins else 0
        avg_loss = statistics.mean(losses) if losses else 0

        # Risk-reward ratio (B)
        risk_reward_ratio = (avg_win / avg_loss) if avg_loss > 0 and avg_win > 0 else 0

        # Kelly fraction: f = ((B + 1) × P - 1) / B
        if risk_reward_ratio > 0:
            base_kelly_f = ((risk_reward_ratio + 1) * win_rate - 1) / risk_reward_ratio
        else:
            # If no valid R:R but there are losses and no avg_win, reflect negative expectancy
            base_kelly_f = -abs(1 - win_rate) if avg_loss > 0 and avg_win == 0 else 0

        # Expectancy (average profit per trade)
        expectancy = statistics.mean(profits)

        # Profit factor
        total_wins = sum(pos_wins)
        total_losses = sum(losses)
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Reliability assessment
        is_reliable = total_trades >= self.min_trades
        
        if total_trades < 10:
            confidence_note = "VERY LOW: Less than 10 trades - use fallback sizing"
        elif total_trades < self.min_trades:
            confidence_note = f"LOW: {total_trades}/{self.min_trades} trades - Kelly unreliable"
        elif total_trades < 100:
            confidence_note = f"MODERATE: {total_trades} trades - Kelly usable with caution"
        else:
            confidence_note = f"HIGH: {total_trades} trades - Kelly reliable"

        return KellyParameters(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            risk_reward_ratio=risk_reward_ratio,
            base_kelly_f=base_kelly_f,
            sample_size=total_trades,
            expectancy=expectancy,
            profit_factor=profit_factor,
            is_reliable=is_reliable,
            confidence_note=confidence_note
        )

    def _extract_profits(self, trade_history: List[Dict[str, Any]]) -> List[float]:
        """Extract profit values from various trade formats."""
        profits = []
        
        for trade in trade_history:
            if 'profit' in trade:
                profits.append(float(trade['profit']))
            elif 'pnl' in trade:
                profits.append(float(trade['pnl']))
            elif 'result' in trade and 'amount' in trade:
                amount = abs(float(trade['amount']))
                if trade['result'].lower() in ('win', 'won', 'profit'):
                    profits.append(amount)
                else:
                    profits.append(-amount)
        
        return profits

    def _empty_parameters(self) -> KellyParameters:
        """Return empty parameters for no history."""
        return KellyParameters(
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            risk_reward_ratio=0.0,
            base_kelly_f=0.0,
            sample_size=0,
            expectancy=0.0,
            profit_factor=0.0,
            is_reliable=False,
            confidence_note="NO DATA: No trade history available"
        )

    def calculate_rolling_kelly(
        self,
        trade_history: List[Dict[str, Any]],
        window_size: int = 50
    ) -> List[KellyParameters]:
        """
        Calculate Kelly parameters over rolling windows.
        
        Useful for detecting strategy decay over time.
        
        Args:
            trade_history: Full trade history
            window_size: Number of trades per window
            
        Returns:
            List of KellyParameters for each window
        """
        results = []
        n = len(trade_history)
        if n < window_size:
            return results
        
        for i in range(0, n - window_size + 1):
            window = trade_history[i:i + window_size]
            params = self.calculate_kelly_parameters(window)
            results.append(params)
        
        return results

    def detect_edge_decay(
        self,
        rolling_kelly: List[KellyParameters],
        decay_threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Detect if trading edge is decaying over time.
        
        Args:
            rolling_kelly: Output from calculate_rolling_kelly
            decay_threshold: Percentage drop that triggers alert (0.15 = 15%)
            
        Returns:
            Dictionary with decay analysis
        """
        if len(rolling_kelly) < 3:
            return {'status': 'insufficient_data', 'decay_detected': False}

        # Compare recent vs historical
        recent = rolling_kelly[-1]
        historical_avg = statistics.mean([p.win_rate for p in rolling_kelly[:-1]])
        
        if historical_avg > 0:
            win_rate_change = (recent.win_rate - historical_avg) / historical_avg
        else:
            win_rate_change = 0

        decay_detected = win_rate_change < -decay_threshold

        return {
            'status': 'analyzed',
            'decay_detected': decay_detected,
            'historical_win_rate': historical_avg,
            'recent_win_rate': recent.win_rate,
            'win_rate_change_pct': win_rate_change * 100,
            'alert': 'STRATEGY DECAY DETECTED' if decay_detected else 'OK'
        }
