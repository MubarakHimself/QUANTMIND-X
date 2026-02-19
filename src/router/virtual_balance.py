"""
Virtual Balance Manager Module

Manages virtual trading accounts for demo mode EAs.
Tracks virtual balances, margin, and equity for paper trading.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


@dataclass
class VirtualAccount:
    """
    Virtual trading account for demo mode.
    
    Attributes:
        ea_id: EA identifier
        initial_balance: Starting balance
        current_balance: Current balance after realized P&L
        equity: Current equity (balance + floating P&L)
        margin_used: Currently used margin
        free_margin: Available margin for new trades
        last_updated: Timestamp of last update
    """
    ea_id: str
    initial_balance: float
    current_balance: float
    equity: float
    margin_used: float = 0.0
    free_margin: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_balance(self, profit_loss: float) -> float:
        """
        Update balance after a trade closes.
        
        Args:
            profit_loss: Realized profit/loss from trade
            
        Returns:
            New balance
        """
        self.current_balance += profit_loss
        self.equity = self.current_balance + self.margin_used  # Update equity
        self.last_updated = datetime.now(timezone.utc)
        
        logger.info(
            f"VirtualAccount [{self.ea_id}]: Balance updated by {profit_loss:.2f} "
            f"-> New balance: {self.current_balance:.2f}"
        )
        return self.current_balance
    
    def can_trade(self, required_margin: float) -> bool:
        """
        Check if account has enough free margin.
        
        Args:
            required_margin: Margin required for the trade
            
        Returns:
            True if trade is possible
        """
        return self.free_margin >= required_margin
    
    def use_margin(self, margin: float) -> bool:
        """
        Reserve margin for a trade.
        
        Args:
            margin: Margin to reserve
            
        Returns:
            True if successful
        """
        if not self.can_trade(margin):
            return False
        
        self.margin_used += margin
        self.free_margin = self.equity - self.margin_used
        self.last_updated = datetime.now(timezone.utc)
        return True
    
    def release_margin(self, margin: float) -> None:
        """
        Release margin after a trade closes.
        
        Args:
            margin: Margin to release
        """
        self.margin_used = max(0, self.margin_used - margin)
        self.free_margin = self.equity - self.margin_used
        self.last_updated = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'ea_id': self.ea_id,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'equity': self.equity,
            'margin_used': self.margin_used,
            'free_margin': self.free_margin,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'pnl': self.current_balance - self.initial_balance,
            'pnl_pct': (self.current_balance - self.initial_balance) / self.initial_balance * 100 if self.initial_balance else 0
        }


class VirtualBalanceManager:
    """
    Manager for virtual trading accounts.
    
    Features:
    - Create virtual accounts for demo EAs
    - Track balances, equity, and margin
    - Update balances after trades
    - Persist to database (future enhancement)
    """
    
    def __init__(self):
        """Initialize Virtual Balance Manager."""
        self._accounts: Dict[str, VirtualAccount] = {}
        logger.info("VirtualBalanceManager initialized")
    
    def create_account(self, ea_id: str, initial_balance: float = 1000.0) -> VirtualAccount:
        """
        Create a new virtual account.
        
        Args:
            ea_id: EA identifier
            initial_balance: Starting balance
            
        Returns:
            Created VirtualAccount
        """
        if ea_id in self._accounts:
            logger.warning(f"Virtual account for {ea_id} already exists - returning existing")
            return self._accounts[ea_id]
        
        account = VirtualAccount(
            ea_id=ea_id,
            initial_balance=initial_balance,
            current_balance=initial_balance,
            equity=initial_balance,
            free_margin=initial_balance
        )
        
        self._accounts[ea_id] = account
        logger.info(f"Created virtual account for {ea_id} with balance {initial_balance}")
        return account
    
    def get_account(self, ea_id: str) -> Optional[VirtualAccount]:
        """
        Get virtual account by EA ID.
        
        Args:
            ea_id: EA identifier
            
        Returns:
            VirtualAccount if found, None otherwise
        """
        return self._accounts.get(ea_id)
    
    def update_after_trade(
        self,
        ea_id: str,
        profit_loss: float,
        margin_change: float = 0.0
    ) -> Optional[float]:
        """
        Update virtual account after a trade.
        
        Args:
            ea_id: EA identifier
            profit_loss: Realized profit/loss
            margin_change: Change in margin (negative = release, positive = use)
            
        Returns:
            New balance if account exists, None otherwise
        """
        account = self.get_account(ea_id)
        if account is None:
            logger.warning(f"No virtual account found for {ea_id}")
            return None
        
        # Update balance
        new_balance = account.update_balance(profit_loss)
        
        # Update margin
        if margin_change < 0:
            account.release_margin(abs(margin_change))
        elif margin_change > 0:
            account.use_margin(margin_change)
        
        return new_balance
    
    def reset_account(self, ea_id: str, new_balance: Optional[float] = None) -> bool:
        """
        Reset virtual account to initial or specified balance.
        
        Args:
            ea_id: EA identifier
            new_balance: New balance (if None, reset to initial)
            
        Returns:
            True if successful
        """
        account = self.get_account(ea_id)
        if account is None:
            return False
        
        target = new_balance if new_balance is not None else account.initial_balance
        account.current_balance = target
        account.equity = target
        account.margin_used = 0.0
        account.free_margin = target
        account.last_updated = datetime.now(timezone.utc)
        
        logger.info(f"Reset virtual account for {ea_id} to balance {target}")
        return True
    
    def delete_account(self, ea_id: str) -> bool:
        """
        Delete a virtual account.
        
        Args:
            ea_id: EA identifier
            
        Returns:
            True if successful
        """
        if ea_id in self._accounts:
            del self._accounts[ea_id]
            logger.info(f"Deleted virtual account for {ea_id}")
            return True
        return False
    
    def get_all_accounts(self) -> Dict[str, VirtualAccount]:
        """
        Get all virtual accounts.
        
        Returns:
            Dictionary of all accounts
        """
        return self._accounts.copy()
    
    def get_total_equity(self) -> float:
        """
        Get total equity across all virtual accounts.
        
        Returns:
            Total equity
        """
        return sum(acc.equity for acc in self._accounts.values())
    
    def get_summary(self) -> Dict:
        """
        Get summary of all virtual accounts.
        
        Returns:
            Summary dictionary
        """
        accounts = list(self._accounts.values())
        return {
            'total_accounts': len(accounts),
            'total_equity': sum(acc.equity for acc in accounts),
            'total_balance': sum(acc.current_balance for acc in accounts),
            'total_pnl': sum(acc.current_balance - acc.initial_balance for acc in accounts),
            'accounts': [acc.to_dict() for acc in accounts]
        }


# Global manager instance
_global_manager: Optional[VirtualBalanceManager] = None


def get_virtual_balance_manager() -> VirtualBalanceManager:
    """Get or create the global virtual balance manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = VirtualBalanceManager()
    return _global_manager