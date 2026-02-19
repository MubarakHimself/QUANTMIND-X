"""
Dynamic Bot Limiter

Calculates maximum bot count and risk per bot based on account tiers.
Implements 7 tiers including small accounts under $200.

Tier Structure (from spec):
- Tier 0: $0-50: 0 bots (trading disabled)
- Tier 1: $50-100: 1 bot
- Tier 2: $100-200: 2 bots
- Tier 3: $200-500: 3 bots
- Tier 4: $500-1k: 5 bots
- Tier 5: $1k-5k: 10 bots
- Tier 6: $10k+: 30 bots

Safety Buffer:
- Minimum capital per bot: $50
- Safety buffer multiplier: 2x
- Required capital = num_bots × $50 × 2
"""

from typing import List, Tuple, Optional, Dict, Any


class DynamicBotLimiter:
    """
    Calculates maximum bot count and risk per bot based on account tiers.
    
    Provides tier-based limits with safety buffer calculations for
    responsible capital allocation across trading bots.
    """
    
    # 7-tier structure including small accounts
    TIERS: List[Tuple[float, float, int]] = [
        (0, 50, 0),           # Tier 0: $0-50: 0 bots (trading disabled)
        (50, 100, 1),         # Tier 1: $50-100: 1 bot
        (100, 200, 2),        # Tier 2: $100-200: 2 bots
        (200, 500, 3),        # Tier 3: $200-500: 3 bots
        (500, 1000, 5),       # Tier 4: $500-1k: 5 bots
        (1000, 5000, 10),     # Tier 5: $1k-5k: 10 bots
        (5000, float('inf'), 20),  # Tier 6: $5k+: 20 bots (per plan)
    ]
    
    # Safety buffer configuration
    MIN_CAPITAL_PER_BOT = 50.0  # Minimum $50 per bot
    SAFETY_BUFFER_MULTIPLIER = 2.0  # 2x safety buffer
    
    @classmethod
    def get_max_bots(cls, account_balance: Optional[float]) -> int:
        """
        Returns maximum bot count based on account balance tiers.
        
        Args:
            account_balance: Current account balance (defaults to 100k if None)
            
        Returns:
            Maximum number of bots allowed for this account tier
        """
        # Default to 100000 if None (same as governor fallback)
        if account_balance is None:
            account_balance = 100000.0
        for min_bal, max_bal, max_bots in cls.TIERS:
            # Handle the tier range check properly
            if max_bal == float('inf'):
                if account_balance >= min_bal:
                    return max_bots
            elif min_bal <= account_balance < max_bal:
                return max_bots
        return 20  # Default for large accounts (aligns with top tier)

    @classmethod
    def get_recommended_risk_per_bot(cls, account_balance: Optional[float]) -> float:
        """
        Calculates risk per bot to maintain 3% total portfolio risk.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Recommended risk percentage per bot
        """
        max_bots = cls.get_max_bots(account_balance)
        total_risk_pct = 0.03  # 3% total risk
        risk_per_bot_pct = total_risk_pct / max_bots if max_bots > 0 else 0
        return risk_per_bot_pct * 100  # Return as percentage
    
    @classmethod
    def can_add_bot(cls, account_balance: Optional[float], current_bots: int) -> Tuple[bool, str]:
        """
        Check if account can support adding another bot.
        
        Args:
            account_balance: Current account balance
            current_bots: Number of currently active bots
            
        Returns:
            Tuple of (can_add, reason) - can_add is True if bot can be added
        """
        if account_balance is None:
            account_balance = 100000.0
            
        max_bots = cls.get_max_bots(account_balance)
        
        # Check tier 0 (trading disabled)
        if max_bots == 0:
            return False, f"Trading disabled for accounts under ${cls.TIERS[0][1]}"
        
        # Check if already at max
        if current_bots >= max_bots:
            return False, f"Bot limit reached ({current_bots}/{max_bots}) for ${account_balance:.0f} account"
        
        # Check safety buffer
        safety_buffer = cls.calculate_safety_buffer(account_balance, current_bots + 1)
        if account_balance < safety_buffer:
            return False, f"Insufficient balance for safety buffer (need ${safety_buffer:.0f})"
        
        return True, f"OK - {max_bots - current_bots - 1} slots remaining"
    
    @classmethod
    def get_tier_info(cls, account_balance: Optional[float]) -> Dict[str, Any]:
        """
        Get detailed tier information for an account.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Dict with tier name, range, max bots, and risk info
        """
        if account_balance is None:
            account_balance = 100000.0
            
        for idx, (min_bal, max_bal, max_bots) in enumerate(cls.TIERS):
            if max_bal == float('inf'):
                in_tier = account_balance >= min_bal
            else:
                in_tier = min_bal <= account_balance < max_bal
                
            if in_tier:
                tier_name = cls._get_tier_name(idx)
                return {
                    "tier": tier_name,
                    "tier_number": idx,
                    "range": f"${min_bal:.0f}-${max_bal:.0f}" if max_bal != float('inf') else f"${min_bal:.0f}+",
                    "min_balance": min_bal,
                    "max_balance": max_bal,
                    "max_bots": max_bots,
                    "risk_per_bot_pct": cls.get_recommended_risk_per_bot(account_balance),
                }
        
        # Default to highest tier
        return {
            "tier": "Tier 6",
            "tier_number": 6,
            "range": "$5000+",
            "min_balance": 5000,
            "max_balance": float('inf'),
            "max_bots": 20,
            "risk_per_bot_pct": 0.15,
        }
    
    @classmethod
    def calculate_safety_buffer(cls, account_balance: float, num_bots: int) -> float:
        """
        Calculate required capital with 2x safety buffer.
        
        Safety Buffer Logic:
        - Minimum capital per bot: $50
        - Safety buffer multiplier: 2x
        - Required capital = num_bots × $50 × 2
        
        Args:
            account_balance: Current account balance (unused but kept for API consistency)
            num_bots: Number of bots to support
            
        Returns:
            Required capital with safety buffer
        """
        return num_bots * cls.MIN_CAPITAL_PER_BOT * cls.SAFETY_BUFFER_MULTIPLIER
    
    @classmethod
    def get_next_tier_threshold(cls, account_balance: Optional[float]) -> Optional[Dict[str, Any]]:
        """
        Get the next tier threshold for progress tracking.
        
        Args:
            account_balance: Current account balance
            
        Returns:
            Dict with next tier info, or None if at highest tier
        """
        if account_balance is None:
            account_balance = 100000.0
            
        for idx, (min_bal, max_bal, max_bots) in enumerate(cls.TIERS):
            if max_bal == float('inf'):
                in_tier = account_balance >= min_bal
            else:
                in_tier = min_bal <= account_balance < max_bal
                
            if in_tier:
                # Check if there's a next tier
                if idx < len(cls.TIERS) - 1:
                    next_min, next_max, next_bots = cls.TIERS[idx + 1]
                    return {
                        "next_tier": cls._get_tier_name(idx + 1),
                        "required_balance": next_min,
                        "additional_bots": next_bots - max_bots,
                        "amount_needed": next_min - account_balance,
                    }
                else:
                    return None  # Already at highest tier
        
        return None
    
    @classmethod
    def _get_tier_name(cls, tier_index: int) -> str:
        """Get human-readable tier name."""
        names = {
            0: "Tier 0 (Disabled)",
            1: "Tier 1 (Starter)",
            2: "Tier 2 (Basic)",
            3: "Tier 3 (Standard)",
            4: "Tier 4 (Growth)",
            5: "Tier 5 (Professional)",
            6: "Tier 6 (Enterprise)",
        }
        return names.get(tier_index, f"Tier {tier_index}")
