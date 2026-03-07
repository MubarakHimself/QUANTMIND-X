"""Prop firm risk overlay for tighter constraints and P_pass calculation."""
import math
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

# Prop firm constraints - each firm has different rules
PROP_FIRM_LIMITS: Dict[str, Dict[str, float]] = {
    "FTMO": {
        "max_drawdown": 0.03,    # 3% vs 5% personal
        "daily_loss": 0.04,      # 4% daily loss limit (per task)
        "profit_target": 0.10,   # 10% target
    },
    "Topstep": {
        "max_drawdown": 0.04,   # 4%
        "daily_loss": 0.04,     # 4% daily
        "profit_target": 0.10, # 10%
    },
    "FundedNext": {
        "max_drawdown": 0.03,   # 3%
        "daily_loss": 0.04,     # 4% daily
        "profit_target": 0.08, # 8%
    },
    "FundingPips": {
        "max_drawdown": 0.05,   # 5%
        "daily_loss": 0.04,     # 4% daily
        "profit_target": 0.10, # 10%
    },
}

# Default personal account limits
DEFAULT_PERSONAL_LIMITS = {
    "max_drawdown": 0.05,      # 5%
    "daily_loss": 0.03,        # 3%
    "profit_target": 0.15,     # 15%
}


@dataclass
class RiskLimits:
    """Risk limit values for an account."""
    max_drawdown: float
    daily_loss: float
    profit_target: float


class PropFirmRiskOverlay:
    """Applies prop firm-specific risk constraints.

    Prop firms have tighter risk limits than personal accounts.
    This overlay manages:
    - Tighter drawdown limits (3% vs 5%)
    - Daily loss limits (4% for prop firms)
    - P_pass (probability of passing) calculation
    - Recovery mode after drawdown breach
    """

    # Recovery mode configuration
    RECOVERY_DRAWDOWN_THRESHOLD = 0.025  # Exit recovery when under 2.5%
    RECOVERY_POSITION_REDUCTION = 0.5    # Reduce position by 50% in recovery

    def __init__(
        self,
        firm_name: Optional[str] = None,
        initial_balance: float = 100000.0
    ):
        """Initialize the overlay.

        Args:
            firm_name: Name of the prop firm (FTMO, Topstep, etc.)
            initial_balance: Starting account balance
        """
        self.firm_name = firm_name
        self.initial_balance = initial_balance
        self.current_balance = initial_balance

        # Get limits for this firm or use FTMO as default
        self.limits = PROP_FIRM_LIMITS.get(
            firm_name,
            PROP_FIRM_LIMITS["FTMO"]
        )

        # Recovery mode state
        self.in_recovery_mode = False
        self.recovery_start_drawdown = 0.0
        self.current_drawdown = 0.0

    def apply_risk_limits(
        self,
        account_book: str,
        max_drawdown: float = 0.05,
        daily_loss: float = 0.03,
        profit_target: float = 0.15
    ) -> Dict[str, Any]:
        """Apply appropriate risk limits based on account book.

        Args:
            account_book: "personal" or "prop_firm"
            max_drawdown: Personal max drawdown (default 5%)
            daily_loss: Personal daily loss limit (default 3%)
            profit_target: Personal profit target (default 15%)

        Returns:
            Dict with effective limits and metadata
        """
        if account_book == "prop_firm" and self.firm_name:
            return {
                "effective_max_drawdown": self.limits["max_drawdown"],
                "effective_daily_loss": self.limits["daily_loss"],
                "effective_profit_target": self.limits["profit_target"],
                "book_type": "prop_firm",
                "firm": self.firm_name
            }
        else:
            return {
                "effective_max_drawdown": max_drawdown,
                "effective_daily_loss": daily_loss,
                "effective_profit_target": profit_target,
                "book_type": "personal",
                "firm": None
            }

    def calculate_p_pass(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_book: str
    ) -> Dict[str, Any]:
        """Calculate probability of passing prop firm challenge.

        Uses a simplified Drift-Diffusion model to estimate the
        probability of passing the challenge within drawdown constraints.

        Args:
            win_rate: Win rate as decimal (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            account_book: "personal" or "prop_firm"

        Returns:
            Dict with p_pass, prop_score, and metadata
        """
        limits = self.apply_risk_limits(account_book)

        # Drift-Diffusion parameters
        mu = win_rate * avg_win - (1 - win_rate) * avg_loss
        sigma_sq = (
            win_rate * avg_win ** 2 +
            (1 - win_rate) * avg_loss ** 2 -
            mu ** 2
        )

        D = limits["effective_max_drawdown"]
        T = limits["effective_profit_target"]

        # Calculate P_pass
        if sigma_sq <= 0 or D <= 0 or T <= 0:
            p_pass = 0.0
        else:
            # P_pass formula from Bailey et al.
            # Probability of reaching profit target before drawdown limit
            try:
                numerator = 1 - math.exp(-2 * mu * D / sigma_sq)
                denominator = 1 - math.exp(-2 * mu * (D + T) / sigma_sq)
                p_pass = numerator / denominator if denominator != 0 else 0
            except (OverflowError, ZeroDivisionError):
                p_pass = 0.0

        # Prop score: normalized performance metric
        if sigma_sq > 0:
            prop_score = (mu / math.sqrt(sigma_sq)) * math.sqrt(D / T)
        else:
            prop_score = 0.0

        return {
            "p_pass": round(max(0, min(1, p_pass)), 3),
            "prop_score": round(prop_score, 3),
            "book_type": limits["book_type"],
            "firm": self.firm_name,
            "target": limits["effective_profit_target"],
            "max_dd": limits["effective_max_drawdown"]
        }

    def update_balance(self, new_balance: float) -> None:
        """Update current balance and calculate drawdown.

        Args:
            new_balance: New account balance
        """
        self.current_balance = new_balance
        self.current_drawdown = 1 - (new_balance / self.initial_balance)

    def check_recovery_mode(self) -> Dict[str, Any]:
        """Check if in recovery mode based on current drawdown.

        Returns:
            Dict with recovery status and details
        """
        # Check if should enter recovery mode
        effective_dd = self.get_effective_drawdown_limit()

        if not self.in_recovery_mode:
            should_enter = self.current_drawdown >= effective_dd
            if should_enter:
                self.enter_recovery_mode()

        return {
            "in_recovery": self.in_recovery_mode,
            "current_drawdown": round(self.current_drawdown, 4),
            "drawdown_limit": effective_dd,
            "firm": self.firm_name
        }

    def enter_recovery_mode(self) -> None:
        """Enter recovery mode after drawdown breach."""
        if not self.in_recovery_mode:
            self.in_recovery_mode = True
            self.recovery_start_drawdown = self.current_drawdown

    def can_exit_recovery(self) -> Dict[str, Any]:
        """Check if can exit recovery mode.

        Returns:
            Dict with exit status and requirements
        """
        if not self.in_recovery_mode:
            return {"can_exit": True, "reason": "Not in recovery"}

        # Can exit if drawdown is below threshold
        can_exit = self.current_drawdown < self.RECOVERY_DRAWDOWN_THRESHOLD

        return {
            "can_exit": can_exit,
            "current_drawdown": round(self.current_drawdown, 4),
            "required_drawdown": self.RECOVERY_DRAWDOWN_THRESHOLD,
            "reason": "Drawdown recovered" if can_exit else "Still in recovery"
        }

    def exit_recovery_mode(self) -> Dict[str, Any]:
        """Exit recovery mode if conditions are met.

        Returns:
            Dict with exit result
        """
        exit_status = self.can_exit_recovery()

        if exit_status["can_exit"]:
            self.in_recovery_mode = False
            self.recovery_start_drawdown = 0.0
            return {
                "exited": True,
                "message": "Exited recovery mode",
                "current_drawdown": round(self.current_drawdown, 4)
            }

        return {
            "exited": False,
            "message": exit_status["reason"],
            "current_drawdown": round(self.current_drawdown, 4)
        }

    def get_effective_drawdown_limit(self) -> float:
        """Get the effective drawdown limit based on account type.

        Returns:
            Effective drawdown limit as decimal
        """
        # In recovery, use a tighter limit
        if self.in_recovery_mode:
            return min(
                self.limits["max_drawdown"],
                self.recovery_start_drawdown
            )
        return self.limits["max_drawdown"]

    def get_recovery_risk_multiplier(self) -> Dict[str, Any]:
        """Get risk multiplier in recovery mode.

        Returns:
            Dict with multiplier value and details
        """
        if not self.in_recovery_mode:
            return {
                "multiplier": 1.0,
                "mode": "normal",
                "reason": "Not in recovery"
            }

        # Calculate how close to the breach
        breach_distance = self.recovery_start_drawdown - self.current_drawdown
        recovery_progress = min(1.0, breach_distance / self.RECOVERY_DRAWDOWN_THRESHOLD)

        # Reduce position as we recover
        multiplier = 1.0 - (self.RECOVERY_POSITION_REDUCTION * (1 - recovery_progress))

        return {
            "multiplier": round(multiplier, 3),
            "mode": "recovery",
            "recovery_progress": round(recovery_progress, 3),
            "original_multiplier": self.RECOVERY_POSITION_REDUCTION
        }

    def get_risk_limits(self) -> RiskLimits:
        """Get current risk limits as a RiskLimits object.

        Returns:
            RiskLimits dataclass
        """
        return RiskLimits(
            max_drawdown=self.limits["max_drawdown"],
            daily_loss=self.limits["daily_loss"],
            profit_target=self.limits["profit_target"]
        )
