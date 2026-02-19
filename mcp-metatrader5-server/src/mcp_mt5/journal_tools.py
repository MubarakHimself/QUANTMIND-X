"""
Trade Journal MCP Tools
=======================
MCP tool wrappers for the trade journaling system.
"""

from typing import Any

from .journal import (
    TradeJournal,
    JournalEntry,
    TradeStatus,
    get_trade_journal,
)


def register_journal_tools(mcp):
    """
    Register trade journal tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance.
    """
    
    @mcp.tool()
    def sync_journal_with_mt5(days: int = 30) -> dict[str, Any]:
        """
        Sync trades from MT5 history to the local journal database.
        
        This pulls recent deals from MT5 and saves them to the SQLite database.
        It handles deduplication automatically.
        
        Args:
            days: Number of days to look back for trades (default: 30).
            
        Returns:
            Dictionary with:
            - synced: Number of new trades imported
            - errors: List of any errors encountered
        """
        try:
            journal = get_trade_journal()
            return journal.sync_from_mt5(days=days)
        except Exception as e:
            return {"synced": 0, "errors": [str(e)]}
    
    @mcp.tool()
    def annotate_trade(
        ticket: int,
        notes: str = None,
        setup_type: str = None,
        rating: int = None,
        tags: list[str] = None,
        lessons: str = None
    ) -> dict[str, Any]:
        """
        Add notes and annotations to a trade.
        
        Args:
            ticket: MT5 ticket number of the trade.
            notes: Free-text notes about the trade.
            setup_type: Classification of the setup (e.g., "breakout", "reversal").
            rating: Self-assessment score (1-5).
            tags: List of tags (e.g., ["news", "impulsive"]).
            lessons: Lessons learned from this trade.
            
        Returns:
            Dictionary with success status.
        """
        try:
            journal = get_trade_journal()
            success = journal.annotate_trade(
                ticket=ticket,
                notes=notes,
                setup_type=setup_type,
                rating=rating,
                tags=tags,
                lessons=lessons
            )
            return {"success": success, "ticket": ticket}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    @mcp.tool()
    def get_journal_stats(days: int = 30) -> dict[str, Any]:
        """
        Get performance statistics from the trade journal.
        
        Args:
            days: Analysis period in days (default: 30).
            
        Returns:
            Dictionary with detailed stats:
            - win_rate: Percentage of winning trades
            - profit_factor: Gross profit / Gross loss
            - net_profit: Total net profit
            - total_trades: Number of trades
            - average_win: Average winning trade amount
            - average_loss: Average losing trade amount
            - best_day: Date and profit of best day
            - worst_day: Date and profit of worst day
        """
        try:
            journal = get_trade_journal()
            return journal.get_performance_stats(days=days)
        except Exception as e:
            return {"error": str(e)}
    
    @mcp.tool()
    def search_journal(
        symbol: str = None,
        setup_type: str = None,
        status: str = "closed",
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """
        Search the trade journal with filters.
        
        Args:
            symbol: Filter by symbol (e.g., "EURUSD").
            setup_type: Filter by setup type annotation.
            status: Filter by status ("open", "closed").
            limit: Maximum number of results.
            
        Returns:
            List of trade entries matching the filters.
        """
        try:
            journal = get_trade_journal()
            return journal.get_trades(
                symbol=symbol,
                setup_type=setup_type,
                status=status,
                limit=limit
            )
        except Exception as e:
            return [{"error": str(e)}]
    
    @mcp.tool()
    def export_journal(
        format: str = "csv",
        days: int = 30,
        filename: str = None
    ) -> dict[str, Any]:
        """
        Export trade journal to a file.
        
        Args:
            format: Export format ("csv" or "json").
            days: Number of days to export.
            filename: Optional custom filename.
            
        Returns:
            Dictionary with export path.
        """
        try:
            journal = get_trade_journal()
            import os
            
            # Default path: ~/Desktop/QUANTMINDX/exports/
            export_dir = os.path.join(
                os.path.expanduser("~"), 
                "Desktop", 
                "QUANTMINDX", 
                "exports"
            )
            
            if not filename:
                timestamp = os.popen("date +%Y%m%d_%H%M%S").read().strip()
                filename = f"journal_export_{timestamp}.{format}"
            
            filepath = os.path.join(export_dir, filename)
            
            if format.lower() == "json":
                path = journal.export_json(filepath, days)
            else:
                path = journal.export_csv(filepath, days)
                
            return {
                "success": True, 
                "path": path,
                "format": format
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
