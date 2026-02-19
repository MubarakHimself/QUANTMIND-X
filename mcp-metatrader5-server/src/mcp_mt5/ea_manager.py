"""
EA Manager MCP Tools
====================
MCP tools for managing Expert Advisors (EAs) in MetaTrader 5.

Provides functionality to:
- List installed EAs in the Experts folder
- Get EA metadata and status
- Deploy EAs to charts via templates
- Stop/detach EAs by magic number
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional

import MetaTrader5 as mt5
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================

class EAInfo(BaseModel):
    """Information about an installed Expert Advisor."""
    
    name: str
    path: str
    file_size: int
    modified_time: str
    is_compiled: bool


class EAStatus(BaseModel):
    """Status of a running EA identified by magic number."""
    
    magic_number: int
    is_active: bool
    last_trade_time: Optional[str] = None
    total_trades: int = 0
    open_positions: int = 0
    symbols_traded: list[str] = []


class EADeployRequest(BaseModel):
    """Request to deploy an EA to a chart."""
    
    ea_name: str
    symbol: str
    timeframe: int
    magic_number: int
    inputs: dict[str, Any] = {}


class EAPerformance(BaseModel):
    """Performance metrics for an EA."""
    
    magic_number: int
    period_days: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int


# ============================================================================
# EA Manager Class
# ============================================================================

class EAManager:
    """
    Manages Expert Advisors in MetaTrader 5.
    
    Note: Some operations have limitations due to MT5 Python API constraints:
    - Direct EA attachment via Python is not supported
    - Chart template workarounds are used for deployment
    - EA inputs can only be set via template files
    """
    
    def __init__(self):
        self._terminal_info = None
    
    def _get_terminal_path(self) -> str:
        """Get the MT5 terminal data path."""
        if self._terminal_info is None:
            self._terminal_info = mt5.terminal_info()
        if self._terminal_info is None:
            raise ValueError("MT5 not initialized. Call initialize() first.")
        return self._terminal_info.data_path
    
    def _get_experts_path(self) -> str:
        """Get the path to the Experts folder."""
        return os.path.join(self._get_terminal_path(), "MQL5", "Experts")
    
    def _get_templates_path(self) -> str:
        """Get the path to the Templates folder."""
        return os.path.join(self._get_terminal_path(), "MQL5", "Profiles", "Templates")
    
    def list_installed_eas(self) -> list[EAInfo]:
        """
        List all installed Expert Advisors.
        
        Scans the MQL5/Experts folder for .ex5 (compiled) and .mq5 (source) files.
        
        Returns:
            List of EAInfo objects with EA metadata.
        """
        experts_path = self._get_experts_path()
        eas = []
        
        if not os.path.exists(experts_path):
            logger.warning(f"Experts path does not exist: {experts_path}")
            return eas
        
        for root, dirs, files in os.walk(experts_path):
            for file in files:
                if file.endswith(('.ex5', '.mq5')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, experts_path)
                    
                    stat = os.stat(full_path)
                    mod_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    eas.append(EAInfo(
                        name=os.path.splitext(file)[0],
                        path=rel_path,
                        file_size=stat.st_size,
                        modified_time=mod_time.isoformat(),
                        is_compiled=file.endswith('.ex5')
                    ))
        
        # Sort by name
        eas.sort(key=lambda x: x.name.lower())
        return eas
    
    def get_ea_info(self, ea_name: str) -> Optional[EAInfo]:
        """
        Get detailed information about a specific EA.
        
        Args:
            ea_name: Name of the EA (without extension).
            
        Returns:
            EAInfo or None if not found.
        """
        all_eas = self.list_installed_eas()
        for ea in all_eas:
            if ea.name.lower() == ea_name.lower():
                return ea
        return None
    
    def get_ea_status(self, magic_number: int, days: int = 30) -> EAStatus:
        """
        Check the status of an EA by its magic number.
        
        Looks at recent trading history to determine if the EA is active.
        
        Args:
            magic_number: The magic number assigned to the EA.
            days: Number of days to look back for activity.
            
        Returns:
            EAStatus with activity information.
        """
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()
        
        # Get deals with this magic number
        deals = mt5.history_deals_get(from_date, to_date)
        
        ea_deals = []
        symbols_traded = set()
        last_trade_time = None
        
        if deals:
            for deal in deals:
                if deal.magic == magic_number:
                    ea_deals.append(deal)
                    symbols_traded.add(deal.symbol)
                    deal_time = datetime.fromtimestamp(deal.time)
                    if last_trade_time is None or deal_time > last_trade_time:
                        last_trade_time = deal_time
        
        # Get open positions with this magic number
        positions = mt5.positions_get()
        open_positions = 0
        if positions:
            for pos in positions:
                if pos.magic == magic_number:
                    open_positions += 1
                    symbols_traded.add(pos.symbol)
        
        # Determine if EA is "active" (traded in last 7 days or has open positions)
        is_active = open_positions > 0
        if last_trade_time and (datetime.now() - last_trade_time).days < 7:
            is_active = True
        
        return EAStatus(
            magic_number=magic_number,
            is_active=is_active,
            last_trade_time=last_trade_time.isoformat() if last_trade_time else None,
            total_trades=len(ea_deals),
            open_positions=open_positions,
            symbols_traded=list(symbols_traded)
        )
    
    def get_ea_performance(
        self, 
        magic_number: int, 
        days: int = 30
    ) -> EAPerformance:
        """
        Calculate performance metrics for an EA by magic number.
        
        Args:
            magic_number: The magic number assigned to the EA.
            days: Number of days to analyze.
            
        Returns:
            EAPerformance with detailed metrics.
        """
        from_date = datetime.now() - timedelta(days=days)
        to_date = datetime.now()
        
        # Get deals with this magic number
        deals = mt5.history_deals_get(from_date, to_date)
        
        profits = []
        if deals:
            for deal in deals:
                if deal.magic == magic_number and deal.profit != 0:
                    profits.append(deal.profit)
        
        if not profits:
            return EAPerformance(
                magic_number=magic_number,
                period_days=days,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_profit=0.0,
                total_loss=0.0,
                net_profit=0.0,
                profit_factor=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0
            )
        
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        total_profit = sum(wins) if wins else 0.0
        total_loss = abs(sum(losses)) if losses else 0.0
        
        # Calculate consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for p in profits:
            if p > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
        
        return EAPerformance(
            magic_number=magic_number,
            period_days=days,
            total_trades=len(profits),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(profits) if profits else 0.0,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=total_profit - total_loss,
            profit_factor=total_profit / total_loss if total_loss > 0 else float('inf'),
            average_win=sum(wins) / len(wins) if wins else 0.0,
            average_loss=sum(losses) / len(losses) if losses else 0.0,
            largest_win=max(wins) if wins else 0.0,
            largest_loss=min(losses) if losses else 0.0,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses
        )
    
    def create_ea_template(
        self,
        ea_name: str,
        symbol: str,
        timeframe: int,
        magic_number: int,
        inputs: dict[str, Any] = None
    ) -> str:
        """
        Create a chart template file with EA configuration.
        
        This is a workaround since MT5 Python API doesn't support direct EA attachment.
        The user can then apply this template to a chart manually.
        
        Args:
            ea_name: Name of the EA (without .ex5 extension).
            symbol: Symbol to trade (e.g., "EURUSD").
            timeframe: Timeframe in minutes.
            magic_number: Magic number to assign.
            inputs: Dictionary of EA input parameters.
            
        Returns:
            Path to the created template file.
        """
        templates_path = self._get_templates_path()
        os.makedirs(templates_path, exist_ok=True)
        
        template_name = f"QuantMindX_{ea_name}_{symbol}_{magic_number}.tpl"
        template_path = os.path.join(templates_path, template_name)
        
        # Build input parameters string
        inputs_str = ""
        if inputs:
            for key, value in inputs.items():
                if isinstance(value, bool):
                    value = 1 if value else 0
                inputs_str += f"{key}={value}\n"
        
        # Create template content
        # Note: This is a simplified template. Full templates have more parameters.
        template_content = f"""<chart>
id=0
symbol={symbol}
period={timeframe}
leftpos=0
digits=5
scale=8
graph=1
fore=1
grid=0
volume=0
scroll=1
shift=1
ohlc=1
autoshift=0
shift_size=10
fixed_pos=0

<expert>
name={ea_name}
magic={magic_number}
{inputs_str}</expert>

</chart>
"""
        
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        logger.info(f"Created EA template: {template_path}")
        return template_path
    
    def stop_ea_by_magic(self, magic_number: int) -> dict[str, Any]:
        """
        Close all positions and cancel orders for an EA by magic number.
        
        Note: This doesn't actually stop the EA from running, but closes
        all its positions and pending orders. The EA itself would need
        to be removed from the chart manually.
        
        Args:
            magic_number: The magic number of the EA to stop.
            
        Returns:
            Dictionary with closure results.
        """
        results = {
            'magic_number': magic_number,
            'positions_closed': 0,
            'orders_cancelled': 0,
            'errors': []
        }
        
        # Close all positions with this magic number
        positions = mt5.positions_get()
        if positions:
            for pos in positions:
                if pos.magic == magic_number:
                    # Determine close type based on position type
                    close_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    
                    tick = mt5.symbol_info_tick(pos.symbol)
                    if tick is None:
                        results['errors'].append(f"Failed to get tick for {pos.symbol}")
                        continue
                    
                    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": pos.symbol,
                        "volume": pos.volume,
                        "type": close_type,
                        "position": pos.ticket,
                        "price": price,
                        "magic": magic_number,
                        "comment": "QuantMindX EA Stop"
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        results['positions_closed'] += 1
                    else:
                        results['errors'].append(
                            f"Failed to close position {pos.ticket}: {result.comment}"
                        )
        
        # Cancel all pending orders with this magic number
        orders = mt5.orders_get()
        if orders:
            for order in orders:
                if order.magic == magic_number:
                    request = {
                        "action": mt5.TRADE_ACTION_REMOVE,
                        "order": order.ticket
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        results['orders_cancelled'] += 1
                    else:
                        results['errors'].append(
                            f"Failed to cancel order {order.ticket}: {result.comment}"
                        )
        
        return results


# ============================================================================
# Global EA Manager Instance
# ============================================================================

_ea_manager: Optional[EAManager] = None


def get_ea_manager() -> EAManager:
    """Get or create the global EA Manager instance."""
    global _ea_manager
    if _ea_manager is None:
        _ea_manager = EAManager()
    return _ea_manager
