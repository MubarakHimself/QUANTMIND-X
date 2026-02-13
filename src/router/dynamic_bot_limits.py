from typing import List, Tuple, Optional

class DynamicBotLimiter:
    """
    Calculates maximum bot count and risk per bot based on account tiers.
    """
    TIERS: List[Tuple[float, float, int]] = [
        (200, 500, 3),      # $200-500: 3 bots
        (500, 1000, 5),     # $500-1k: 5 bots
        (1000, 5000, 10),   # $1k-5k: 10 bots
        (5000, 10000, 20),  # $5k-10k: 20 bots
        (10000, float('inf'), 30),  # $10k+: 30 bots
    ]

    @classmethod
    def get_max_bots(cls, account_balance: Optional[float]) -> int:
        """Returns maximum bot count based on account balance tiers"""
        # Default to 100000 if None (same as governor fallback)
        if account_balance is None:
            account_balance = 100000.0
        for min_bal, max_bal, max_bots in cls.TIERS:
            if min_bal <= account_balance <= max_bal or max_bal == float('inf'):
                return max_bots
        return 30  # Default for large accounts

    @classmethod
    def get_recommended_risk_per_bot(cls, account_balance: Optional[float]) -> float:
        """Calculates risk per bot to maintain 3% total portfolio risk"""
        max_bots = cls.get_max_bots(account_balance)
        total_risk_pct = 0.03  # 3% total risk
        risk_per_bot_pct = total_risk_pct / max_bots
        return risk_per_bot_pct * 100  # Return as percentage
