"""
EA Manager MCP Tools
====================
MCP tool wrappers for EA management functionality.

These tools are registered with the FastMCP server and provide
AI-accessible interfaces to EA management operations.
"""

from typing import Any

from .ea_manager import (
    EAInfo,
    EAPerformance,
    EAStatus,
    get_ea_manager,
)


def register_ea_tools(mcp):
    """
    Register EA management tools with the MCP server.
    
    Args:
        mcp: FastMCP server instance.
    """
    
    @mcp.tool()
    def list_installed_eas() -> list[dict[str, Any]]:
        """
        List all installed Expert Advisors (EAs) in the MT5 Experts folder.
        
        Scans the MQL5/Experts folder for both compiled (.ex5) and source (.mq5) files.
        
        Returns:
            List of EAs with name, path, file_size, modified_time, and is_compiled flag.
            
        Example:
            eas = list_installed_eas()
            # Returns: [
            #     {"name": "MyScalper", "path": "MyScalper.ex5", "is_compiled": true, ...},
            #     {"name": "TrendFollower", "path": "Strategies/TrendFollower.ex5", ...}
            # ]
        """
        manager = get_ea_manager()
        eas = manager.list_installed_eas()
        return [ea.model_dump() for ea in eas]
    
    @mcp.tool()
    def get_ea_info(ea_name: str) -> dict[str, Any] | None:
        """
        Get detailed information about a specific Expert Advisor.
        
        Args:
            ea_name: Name of the EA (without .ex5 or .mq5 extension).
            
        Returns:
            EAInfo dictionary with name, path, file_size, modified_time, is_compiled.
            Returns None if the EA is not found.
            
        Example:
            info = get_ea_info("MyScalper")
            # Returns: {"name": "MyScalper", "path": "MyScalper.ex5", "file_size": 102400, ...}
        """
        manager = get_ea_manager()
        ea = manager.get_ea_info(ea_name)
        return ea.model_dump() if ea else None
    
    @mcp.tool()
    def get_ea_status(magic_number: int, days: int = 30) -> dict[str, Any]:
        """
        Check the status of an EA by its magic number.
        
        Analyzes recent trading activity to determine if the EA is active.
        An EA is considered "active" if it has open positions or traded within 7 days.
        
        Args:
            magic_number: The unique magic number assigned to the EA.
            days: Number of days to look back for trading activity (default: 30).
            
        Returns:
            EAStatus dictionary with:
            - magic_number: The queried magic number
            - is_active: True if EA has recent activity
            - last_trade_time: ISO timestamp of last trade (or None)
            - total_trades: Number of trades in the period
            - open_positions: Number of currently open positions
            - symbols_traded: List of symbols the EA has traded
            
        Example:
            status = get_ea_status(123456)
            # Returns: {
            #     "magic_number": 123456,
            #     "is_active": true,
            #     "last_trade_time": "2026-01-27T14:30:00",
            #     "total_trades": 45,
            #     "open_positions": 2,
            #     "symbols_traded": ["EURUSD", "GBPUSD"]
            # }
        """
        manager = get_ea_manager()
        status = manager.get_ea_status(magic_number, days)
        return status.model_dump()
    
    @mcp.tool()
    def get_ea_performance(magic_number: int, days: int = 30) -> dict[str, Any]:
        """
        Calculate comprehensive performance metrics for an EA.
        
        Analyzes trading history to compute win rate, profit factor, and other metrics.
        
        Args:
            magic_number: The unique magic number assigned to the EA.
            days: Number of days to analyze (default: 30).
            
        Returns:
            EAPerformance dictionary with:
            - total_trades: Number of completed trades
            - winning_trades / losing_trades: Win/loss count
            - win_rate: Percentage of winning trades (0.0 - 1.0)
            - total_profit / total_loss: Gross profit and loss
            - net_profit: Total profit minus total loss
            - profit_factor: total_profit / total_loss
            - average_win / average_loss: Average trade results
            - largest_win / largest_loss: Best and worst trades
            - max_consecutive_wins / max_consecutive_losses: Streak metrics
            
        Example:
            perf = get_ea_performance(123456, days=90)
            # Returns: {
            #     "magic_number": 123456,
            #     "period_days": 90,
            #     "total_trades": 150,
            #     "win_rate": 0.65,
            #     "profit_factor": 1.8,
            #     "net_profit": 2500.50,
            #     ...
            # }
        """
        manager = get_ea_manager()
        performance = manager.get_ea_performance(magic_number, days)
        return performance.model_dump()
    
    @mcp.tool()
    def create_ea_template(
        ea_name: str,
        symbol: str,
        timeframe: int,
        magic_number: int,
        inputs: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Create a chart template file with EA configuration.
        
        Since the MT5 Python API cannot directly attach EAs to charts,
        this creates a template file that can be applied manually:
        1. In MT5, open a chart for the specified symbol
        2. Right-click -> Templates -> Load Template
        3. Select the created template file
        
        Args:
            ea_name: Name of the EA (without .ex5 extension).
            symbol: Symbol to trade (e.g., "EURUSD").
            timeframe: Timeframe in minutes (e.g., 60 for H1, 1440 for D1).
            magic_number: Unique magic number to assign to the EA.
            inputs: Dictionary of EA input parameters (optional).
            
        Returns:
            Dictionary with:
            - success: True if template was created
            - template_path: Full path to the created template file
            - template_name: Name of the template file
            - instructions: How to apply the template
            
        Example:
            result = create_ea_template(
                ea_name="MyScalper",
                symbol="EURUSD",
                timeframe=60,
                magic_number=123456,
                inputs={"TakeProfit": 50, "StopLoss": 30}
            )
        """
        manager = get_ea_manager()
        
        try:
            template_path = manager.create_ea_template(
                ea_name=ea_name,
                symbol=symbol,
                timeframe=timeframe,
                magic_number=magic_number,
                inputs=inputs or {}
            )
            
            return {
                "success": True,
                "template_path": template_path,
                "template_name": template_path.split('/')[-1] if '/' in template_path else template_path.split('\\')[-1],
                "instructions": (
                    f"To deploy the EA:\n"
                    f"1. In MT5, open a {symbol} chart with {timeframe}min timeframe\n"
                    f"2. Right-click on chart -> Templates -> Load Template\n"
                    f"3. Select: QuantMindX_{ea_name}_{symbol}_{magic_number}.tpl\n"
                    f"4. The EA will be attached with magic number {magic_number}"
                )
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    def stop_ea_by_magic(magic_number: int) -> dict[str, Any]:
        """
        Emergency stop for an EA: close all positions and cancel pending orders.
        
        This finds all positions and orders with the specified magic number and:
        1. Closes all open positions at market price
        2. Cancels all pending orders
        
        Note: This does NOT remove the EA from the chart. The EA will continue
        running but will have no active trades. To fully stop, manually remove
        the EA from the chart.
        
        Args:
            magic_number: The magic number of the EA to stop.
            
        Returns:
            Dictionary with:
            - magic_number: The targeted magic number
            - positions_closed: Number of positions successfully closed
            - orders_cancelled: Number of orders successfully cancelled
            - errors: List of any errors encountered
            
        Example:
            result = stop_ea_by_magic(123456)
            # Returns: {
            #     "magic_number": 123456,
            #     "positions_closed": 3,
            #     "orders_cancelled": 1,
            #     "errors": []
            # }
        """
        manager = get_ea_manager()
        return manager.stop_ea_by_magic(magic_number)
    
    @mcp.tool()
    def get_daily_pnl(days: int = 7) -> list[dict[str, Any]]:
        """
        Get daily Profit & Loss summary for the account.
        
        Aggregates all trading activity by day to show daily performance.
        
        Args:
            days: Number of days to retrieve (default: 7).
            
        Returns:
            List of daily summaries with:
            - date: ISO date string
            - trades: Number of closed trades
            - gross_profit: Total profit from winning trades
            - gross_loss: Total loss from losing trades  
            - net_pnl: Net profit/loss for the day
            - commission: Total commission paid
            - swap: Total swap charges
            
        Example:
            pnl = get_daily_pnl(7)
            # Returns: [
            #     {"date": "2026-01-27", "net_pnl": 150.50, "trades": 5, ...},
            #     {"date": "2026-01-26", "net_pnl": -45.20, "trades": 3, ...},
            #     ...
            # ]
        """
        from datetime import datetime, timedelta
        import MetaTrader5 as mt5
        
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()
        
        deals = mt5.history_deals_get(from_date, to_date)
        
        if not deals:
            return []
        
        # Aggregate by date
        daily_data = {}
        
        for deal in deals:
            deal_date = datetime.fromtimestamp(deal.time).date().isoformat()
            
            if deal_date not in daily_data:
                daily_data[deal_date] = {
                    'date': deal_date,
                    'trades': 0,
                    'gross_profit': 0.0,
                    'gross_loss': 0.0,
                    'net_pnl': 0.0,
                    'commission': 0.0,
                    'swap': 0.0
                }
            
            if deal.profit != 0:
                daily_data[deal_date]['trades'] += 1
                daily_data[deal_date]['net_pnl'] += deal.profit
                
                if deal.profit > 0:
                    daily_data[deal_date]['gross_profit'] += deal.profit
                else:
                    daily_data[deal_date]['gross_loss'] += deal.profit
            
            daily_data[deal_date]['commission'] += deal.commission
            daily_data[deal_date]['swap'] += deal.swap
        
        # Sort by date descending
        result = list(daily_data.values())
        result.sort(key=lambda x: x['date'], reverse=True)
        
        return result
