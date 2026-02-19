"""
Account Manager MCP Tools
=========================
MCP tool wrappers for multi-account management.

Security Note:
The vault must be unlocked before using account operations.
Use unlock_vault(master_password) first.
"""

from typing import Any, Optional

from .account_manager import get_account_manager, AccountManager


def register_account_tools(mcp):
    """
    Register account management tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance.
    """
    
    @mcp.tool()
    def unlock_vault(master_password: str, config_path: str = None) -> dict[str, Any]:
        """
        Unlock the credential vault to access saved accounts.
        
        This MUST be called before using any account management tools.
        The vault stores encrypted credentials for all your MT5 accounts.
        
        Args:
            master_password: Your master password for decryption.
            config_path: Optional custom path to credentials file.
                        Default: ~/.quantmindx/accounts.json.enc
        
        Returns:
            Dictionary with:
            - success: True if unlocked
            - accounts_loaded: Number of accounts available
            - error: Error message if failed
            
        Example:
            result = unlock_vault(master_password="my_secure_password")
            # Returns: {"success": true, "accounts_loaded": 3}
        """
        try:
            manager = get_account_manager(config_path)
            success = manager.unlock(master_password)
            
            if success:
                return {
                    "success": True,
                    "accounts_loaded": len(manager.list_accounts())
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid master password or corrupted vault"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    def lock_vault() -> dict[str, Any]:
        """
        Lock the credential vault and clear sensitive data from memory.
        
        Call this when you're done managing accounts to secure credentials.
        
        Returns:
            Dictionary with success status.
        """
        try:
            manager = get_account_manager()
            manager.lock()
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    def list_accounts() -> list[dict[str, Any]]:
        """
        List all configured MT5 accounts (without passwords).
        
        Returns account metadata including login, server, broker, 
        account type, and last used timestamp.
        
        Note: Vault must be unlocked first with unlock_vault().
        
        Returns:
            List of account dictionaries with:
            - login: Account number
            - server: Broker server name
            - broker: Broker identifier
            - account_type: live/demo/contest
            - nickname: Friendly name
            - is_active: Whether account is enabled
            - last_used: Last connection timestamp
            
        Example:
            accounts = list_accounts()
            # Returns: [
            #     {"login": 12345678, "server": "Exness-MT5Real", "broker": "exness", ...},
            #     {"login": 87654321, "server": "FTMO-Live", "broker": "ftmo", ...}
            # ]
        """
        try:
            manager = get_account_manager()
            return manager.list_accounts()
        except ValueError as e:
            return [{"error": str(e)}]
    
    @mcp.tool()
    def add_account(
        login: int,
        password: str,
        server: str,
        broker: str = "generic",
        account_type: str = "demo",
        nickname: str = "",
        terminal_path: str = ""
    ) -> dict[str, Any]:
        """
        Add a new MT5 trading account to the vault.
        
        The password will be encrypted and stored securely.
        
        Args:
            login: MT5 account number (e.g., 12345678).
            password: Account password (will be encrypted).
            server: Broker server name (e.g., "Exness-MT5Real").
            broker: Broker identifier: "exness", "ftmo", "the5ers", "mff", "generic".
            account_type: Account type: "live", "demo", or "contest".
            nickname: Friendly name for the account (optional).
            terminal_path: Full path to MT5 terminal executable (optional).
                          Use for multiple MT5 installations.
        
        Returns:
            Dictionary with added account info (without password).
            
        Example:
            result = add_account(
                login=12345678,
                password="my_trading_password",
                server="Exness-MT5Real",
                broker="exness",
                account_type="live",
                nickname="Main Exness Account"
            )
        """
        try:
            manager = get_account_manager()
            account = manager.add_account(
                login=login,
                password=password,
                server=server,
                broker=broker,
                account_type=account_type,
                nickname=nickname,
                terminal_path=terminal_path
            )
            return {
                "success": True,
                "account": account.to_safe_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    def remove_account(login: int) -> dict[str, Any]:
        """
        Remove an account from the vault.
        
        This permanently deletes the account credentials.
        
        Args:
            login: Account number to remove.
            
        Returns:
            Dictionary with success status.
        """
        try:
            manager = get_account_manager()
            success = manager.remove_account(login)
            return {
                "success": success,
                "message": f"Account {login} removed" if success else f"Account {login} not found"
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    def switch_account(login: int) -> dict[str, Any]:
        """
        Switch to a different MT5 trading account.
        
        This will:
        1. Disconnect from current account (if connected)
        2. Initialize MT5 terminal
        3. Login to the specified account
        
        After switching, all trading operations will use the new account.
        
        Args:
            login: Account number to switch to.
            
        Returns:
            Dictionary with:
            - success: True if connected
            - login: Connected account number
            - server: Connected server
            - balance: Current balance
            - equity: Current equity
            - currency: Account currency
            - error: Error message if failed
            
        Example:
            result = switch_account(login=12345678)
            # Returns: {
            #     "success": true,
            #     "login": 12345678,
            #     "balance": 10000.00,
            #     "equity": 10150.50,
            #     "currency": "USD"
            # }
        """
        try:
            manager = get_account_manager()
            return manager.switch_account(login)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    def get_connection_status() -> dict[str, Any]:
        """
        Get current MT5 connection status and account info.
        
        Returns:
            Dictionary with:
            - connected: True if connected to MT5
            - current_login: Currently connected account number
            - current_server: Connected server name
            - connected_at: Connection timestamp
            - balance: Current balance (if connected)
            - equity: Current equity (if connected)
            - margin: Used margin (if connected)
            - free_margin: Available margin (if connected)
            - profit: Open P&L (if connected)
            - currency: Account currency (if connected)
            - last_error: Last error message (if any)
        """
        try:
            manager = get_account_manager()
            return manager.get_connection_status()
        except Exception as e:
            return {"connected": False, "error": str(e)}
    
    @mcp.tool()
    def disconnect_account() -> dict[str, Any]:
        """
        Disconnect from the current MT5 account.
        
        Returns:
            Dictionary with success status.
        """
        try:
            manager = get_account_manager()
            success = manager.disconnect()
            return {"success": success}
        except Exception as e:
            return {"success": False, "error": str(e)}
