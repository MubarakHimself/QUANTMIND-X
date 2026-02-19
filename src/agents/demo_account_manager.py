"""
Demo Account Manager
====================
A thin wrapper around the existing AccountManager for demo account operations.

Provides convenience methods for managing MT5 demo accounts including:
- Adding new demo accounts
- Connecting to demo accounts
- Getting demo account balance
- Verifying demo account status
- Listing all demo accounts

The wrapper automatically unlocks the credential vault using a master password
from the DEMO_VAULT_PASSWORD environment variable.
"""

import os
import logging
from typing import List, Optional, Dict, Any

# Import from mcp_mt5 package
from mcp_mt5.account_manager import AccountManager, AccountType

logger = logging.getLogger(__name__)


class DemoAccountManager:
    """
    Manages MT5 demo accounts for paper trading deployments.
    
    This is a thin wrapper around the AccountManager that provides
    convenience methods specifically for demo account operations.
    
    Usage:
        manager = DemoAccountManager()
        accounts = manager.list_demo_accounts()
        manager.add_demo_account(login=123456, password="pass", server="broker-demo")
        manager.connect_demo_account(login=123456)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DemoAccountManager.
        
        Args:
            config_path: Optional path to the encrypted credentials file.
                        Defaults to ~/.quantmindx/accounts.json.enc
        """
        # Get master password from environment variable
        master_password = os.environ.get("DEMO_VAULT_PASSWORD")
        if not master_password:
            logger.warning(
                "DEMO_VAULT_PASSWORD environment variable not set. "
                "DemoAccountManager will not be able to unlock the vault. "
                "Set DEMO_VAULT_PASSWORD to enable demo account management."
            )
        
        self._account_manager = AccountManager(config_path=config_path)
        self._is_unlocked = False
        
        # Auto-unlock if password is available
        if master_password:
            self.unlock(master_password)
    
    @property
    def is_unlocked(self) -> bool:
        """Check if the credential vault is unlocked."""
        return self._is_unlocked
    
    def unlock(self, master_password: str) -> bool:
        """
        Unlock the credential vault with master password.
        
        Args:
            master_password: Master password for decryption.
            
        Returns:
            True if unlocked successfully.
        """
        result = self._account_manager.unlock(master_password)
        self._is_unlocked = result
        return result
    
    def lock(self) -> None:
        """Lock the credential vault and clear sensitive data."""
        self._account_manager.lock()
        self._is_unlocked = False
        logger.info("DemoAccountManager vault locked")
    
    def add_demo_account(
        self,
        login: int,
        password: str,
        server: str,
        broker: str = "generic",
        nickname: str = ""
    ) -> Dict[str, Any]:
        """
        Add a new demo trading account.
        
        Args:
            login: MT5 account number.
            password: Account password.
            server: Broker server name.
            broker: Broker identifier (exness, ftmo, etc.).
            nickname: Friendly name for the account.
            
        Returns:
            Dictionary with account info (safe, without password).
            
        Raises:
            ValueError: If vault is not unlocked or account already exists.
        """
        if not self._is_unlocked:
            raise ValueError("Unlock vault first with unlock(master_password)")
        
        account = self._account_manager.add_account(
            login=login,
            password=password,
            server=server,
            broker=broker,
            account_type="demo",  # Explicitly set as demo
            nickname=nickname or f"{broker}_demo_{login}"
        )
        
        logger.info(f"Added demo account: {account.to_safe_dict()}")
        return account.to_safe_dict()
    
    def remove_demo_account(self, login: int) -> bool:
        """
        Remove a demo account from the vault.
        
        Args:
            login: Account number to remove.
            
        Returns:
            True if removed successfully.
        """
        if not self._is_unlocked:
            raise ValueError("Unlock vault first")
        
        return self._account_manager.remove_account(login)
    
    def connect_demo_account(self, login: int) -> Dict[str, Any]:
        """
        Switch to a different MT5 demo account.
        
        Args:
            login: Demo account number to connect to.
            
 Dictionary with connection status        Returns:
           .
        """
        if not self._is_unlocked:
            raise ValueError("Unlock vault first")
        
        result = self._account_manager.switch_account(login)
        logger.info(f"Connected to demo account {login}: {result}")
        return result
    
    def get_demo_balance(self, login: int) -> float:
        """
        Get the balance of a demo account.
        
        Note: This connects to the account first to get the current balance.
        
        Args:
            login: Demo account login number.
            
        Returns:
            Current account balance.
            
        Raises:
            ValueError: If not connected or account not found.
        """
        # Connect to the account first
        self.connect_demo_account(login)
        
        # Import MT5 and get account info
        import MetaTrader5 as mt5
        
        if not mt5.account_info():
            raise ValueError(f"Failed to get account info for login {login}")
        
        return mt5.account_info().balance
    
    def verify_demo_account(self, login: int) -> Dict[str, Any]:
        """
        Verify demo account connection and get account details.
        
        Args:
            login: Demo account login number.
            
        Returns:
            Dictionary with:
                - balance: Account balance
                - equity: Account equity
                - margin: Used margin
                - leverage: Account leverage
                - server: Broker server
                - connected: Whether successfully connected
        """
        # Connect to the account
        connection_result = self.connect_demo_account(login)
        
        # Import MT5 and get account info
        import MetaTrader5 as mt5
        
        account_info = mt5.account_info()
        
        if not account_info:
            return {
                "login": login,
                "connected": False,
                "error": "Failed to get account info"
            }
        
        return {
            "login": login,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "leverage": account_info.leverage,
            "server": account_info.server,
            "connected": connection_result.get("connected", False),
            "currency": account_info.currency,
            "margin_mode": str(account_info.margin_mode)
        }
    
    def list_demo_accounts(self) -> List[Dict[str, Any]]:
        """
        List all configured demo accounts.
        
        Returns:
            List of demo account info dictionaries (without passwords).
        """
        if not self._is_unlocked:
            raise ValueError("Unlock vault first")
        
        all_accounts = self._account_manager.list_accounts()
        
        # Filter for demo accounts only
        demo_accounts = [
            acc for acc in all_accounts
            if acc.get("account_type") == "demo"
        ]
        
        return demo_accounts
    
    def get_account(self, login: int) -> Optional[Dict[str, Any]]:
        """
        Get account details by login number.
        
        Args:
            login: Account login number.
            
        Returns:
            Account info dictionary or None if not found.
        """
        if not self._is_unlocked:
            raise ValueError("Unlock vault first")
        
        account = self._account_manager.get_account(login)
        return account.to_safe_dict() if account else None
    
    def update_demo_account(
        self,
        login: int,
        password: Optional[str] = None,
        server: Optional[str] = None,
        nickname: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an existing demo account.
        
        Args:
            login: Account login number.
            password: New password (if updating).
            server: New server (if updating).
            nickname: New nickname (if updating).
            
        Returns:
            Updated account info dictionary.
        """
        if not self._is_unlocked:
            raise ValueError("Unlock vault first")
        
        # Get existing account
        existing = self._account_manager.get_account(login)
        if not existing:
            raise ValueError(f"Account {login} not found")
        
        # Update fields
        if password:
            existing.password = password
        if server:
            existing.server = server
        if nickname:
            existing.nickname = nickname
        
        # Save changes
        self._account_manager._accounts[login] = existing
        self._account_manager._save_accounts()
        
        logger.info(f"Updated demo account {login}")
        return existing.to_safe_dict()
    
    @property
    def connection_state(self) -> Dict[str, Any]:
        """Get current connection state."""
        state = self._account_manager.connection_state
        return {
            "connected": state.connected,
            "current_login": state.current_login,
            "current_server": state.current_server,
            "connected_at": state.connected_at,
            "last_error": state.last_error
        }


# Singleton instance for convenience
_demo_account_manager: Optional[DemoAccountManager] = None


def get_demo_account_manager() -> DemoAccountManager:
    """Get or create the global DemoAccountManager instance."""
    global _demo_account_manager
    if _demo_account_manager is None:
        _demo_account_manager = DemoAccountManager()
    return _demo_account_manager
