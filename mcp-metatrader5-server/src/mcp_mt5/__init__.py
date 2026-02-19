"""MetaTrader 5 MCP Server"""

# Lazy imports to avoid MetaTrader5 dependency when only using paper trading
def _get_mcp():
    from .main import mcp
    return mcp

from .ea_manager import EAManager, EAInfo, EAStatus, EAPerformance, get_ea_manager
from .account_manager import AccountManager, AccountCredentials, get_account_manager
from .alert_service import AlertService, AlertConfig, AlertSeverity, AlertCategory, get_alert_service

__version__ = "0.4.0"
__all__ = [
    "main", "mcp",
    # EA Manager
    "EAManager", "EAInfo", "EAStatus", "EAPerformance", "get_ea_manager",
    # Account Manager
    "AccountManager", "AccountCredentials", "get_account_manager",
    # Alert Service
    "AlertService", "AlertConfig", "AlertSeverity", "AlertCategory", "get_alert_service",
    # Trade Journal
    "TradeJournal", "JournalEntry", "get_trade_journal",
]


def main():
    """Entry point for the MCP server CLI"""
    import os

    from dotenv import load_dotenv

    # Load environment variables from .env file if it exists
    load_dotenv()

    # Determine transport mode from environment or default to stdio
    transport = os.getenv("MT5_MCP_TRANSPORT", "stdio")

    # Get MCP server
    mcp = _get_mcp()

    if transport == "http":
        host = os.getenv("MT5_MCP_HOST", "127.0.0.1")
        port = int(os.getenv("MT5_MCP_PORT", "8000"))
        mcp.run(transport="http", host=host, port=port)
    else:
        # Default to stdio for MCP clients like Claude Desktop
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
