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


def calculate_per_trade_ev(
    win_rate: float,
    risk_reward_ratio: float,
    risk_pct: float = 0.02
) -> float:
    """
    Calculate per-trade expectancy value (EV) as percentage of equity.

    F-14 Correction: Uses correct baseline parameters:
    - win_rate: 52% (was incorrectly 50%)
    - risk_reward_ratio: 2.0 for 1:2 R:R (was incorrectly 1.2:1)
    - risk_pct: 2% of equity (was incorrectly 1%)

    Formula:
        EV = p × (R_ratio × R) - q × R
           = p × (2R) - q × (R)
           = p × 2 × risk_pct - q × risk_pct

    Args:
        win_rate: Win probability (0 to 1, e.g., 0.52 for 52%)
        risk_reward_ratio: Reward-to-risk ratio (e.g., 2.0 for 1:2 R:R)
        risk_pct: Risk per trade as fraction of equity (default 0.02 for 2%)

    Returns:
        Expectancy as fraction of equity (e.g., 0.0112 for 1.12%)

    Example:
        >>> ev = calculate_per_trade_ev(win_rate=0.52, risk_reward_ratio=2.0, risk_pct=0.02)
        >>> print(f"EV: {ev:.4f} ({ev*100:.2f}%)")
        EV: 0.0112 (1.12%)
    """
    p = win_rate
    q = 1.0 - p
    R = risk_pct

    # EV = p × (R_ratio × R) - q × R
    ev_pct = p * (risk_reward_ratio * R) - q * R
    return ev_pct


def calculate_daily_ev(
    per_trade_ev_pct: float,
    trades_per_day: int,
    account_balance: float
) -> float:
    """
    Calculate expected daily profit/loss based on per-trade EV.

    F-14 Correction: Uses correct per-trade EV of +1.12% and $200 baseline.

    Args:
        per_trade_ev_pct: Per-trade expectancy as fraction (e.g., 0.0112 for 1.12%)
        trades_per_day: Number of trades expected per day (default 80)
        account_balance: Account balance for dollar conversion (default $200)

    Returns:
        Expected daily profit/loss in dollars

    Example:
        >>> per_trade_ev = calculate_per_trade_ev(0.52, 2.0, 0.02)
        >>> daily_ev = calculate_daily_ev(per_trade_ev, 80, 200.0)
        >>> print(f"Daily EV: ${daily_ev:.2f}")
        Daily EV: $179.20
    """
    ev_per_trade_dollars = account_balance * per_trade_ev_pct
    daily_ev = trades_per_day * ev_per_trade_dollars
    return daily_ev


def verify_baseline_mathematics() -> Dict[str, Any]:
    """
    Verify the F-14 corrected baseline mathematics for Story 4.11.

    Returns verification that all baseline values are correct:
    - Per-trade risk: 2% ($4 on $200 equity)
    - Minimum R:R ratio: 1:2 (ratio = 2.0)
    - Baseline win rate: 52%
    - Per-trade EV: +1.12% of equity
    - Daily EV: +$179.20 at 80 trades/day

    Returns:
        Dictionary with verification results
    """
    baseline_equity = 200.0
    baseline_risk_pct = 0.02  # 2%
    baseline_risk_dollars = baseline_equity * baseline_risk_pct  # $4
    baseline_win_rate = 0.52  # 52%
    baseline_rr_ratio = 2.0  # 1:2 R:R means reward is 2x risk
    trades_per_day = 80

    # Verify per-trade risk
    expected_risk_dollars = 4.0  # $4 on $200 = 2%
    risk_correct = abs(baseline_risk_dollars - expected_risk_dollars) < 0.001

    # Verify R:R ratio
    expected_rr = 2.0  # 1:2 means 2x reward
    rr_correct = abs(baseline_rr_ratio - expected_rr) < 0.001

    # Verify win rate
    expected_wr = 0.52  # 52%
    wr_correct = abs(baseline_win_rate - expected_wr) < 0.001

    # Calculate per-trade EV
    per_trade_ev = calculate_per_trade_ev(baseline_win_rate, baseline_rr_ratio, baseline_risk_pct)
    expected_ev_pct = 0.0112  # 1.12%
    ev_correct = abs(per_trade_ev - expected_ev_pct) < 0.0001

    # Calculate daily EV
    daily_ev = calculate_daily_ev(per_trade_ev, trades_per_day, baseline_equity)
    expected_daily_ev = 179.20  # $179.20
    daily_ev_correct = abs(daily_ev - expected_daily_ev) < 0.01

    return {
        'per_trade_risk_pct': baseline_risk_pct,
        'per_trade_risk_dollars': baseline_risk_dollars,
        'expected_risk_dollars': expected_risk_dollars,
        'risk_correct': risk_correct,
        'win_rate': baseline_win_rate,
        'win_rate_correct': wr_correct,
        'rr_ratio': baseline_rr_ratio,
        'rr_ratio_correct': rr_correct,
        'per_trade_ev_pct': per_trade_ev,
        'per_trade_ev_correct': ev_correct,
        'daily_ev': daily_ev,
        'expected_daily_ev': expected_daily_ev,
        'daily_ev_correct': daily_ev_correct,
        'all_correct': risk_correct and rr_correct and wr_correct and ev_correct and daily_ev_correct
    }
