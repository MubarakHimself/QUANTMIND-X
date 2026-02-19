import backtrader as bt
import pandas as pd
import io
import sys
import contextlib
import logging
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtesting.lean_slippage import SlippageModel
    from src.backtesting.lean_commission import CommissionModel

logger = logging.getLogger(__name__)


class BacktestResult(dict):
    """Structured result from a backtest run."""
    def __init__(
        self, 
        sharpe: float, 
        return_pct: float, 
        drawdown: float, 
        trades: int, 
        log: str,
        total_slippage: float = 0.0,
        total_commission: float = 0.0
    ):
        super().__init__(
            sharpe=sharpe, 
            return_pct=return_pct, 
            drawdown=drawdown, 
            trades=trades, 
            log=log,
            total_slippage=total_slippage,
            total_commission=total_commission
        )


class LeanSlippageCommission(bt.CommInfoBase):
    """
    Custom Backtrader commission class that uses LEAN-style slippage 
    and commission models.
    """
    
    params = (
        ('commission', 0.0),
        ('slippage_model', None),
        ('commission_model', None),
        ('stocklike', False),  # False for forex
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # Use COMM_FIXED for per-lot
    )
    
    def _getcommission(self, size, price, pseudoexec):
        """Calculate commission and slippage."""
        commission = 0.0
        if self.p.commission_model:
            commission = self.p.commission_model.calculate_commission(
                lots=abs(size),
                symbol='',
                price=price,
                side='buy' if size > 0 else 'sell'
            )
        elif self.p.commission > 0:
            # Use base commission rate when no commission model is provided
            # Apply as per-lot cost consistent with COMM_FIXED default
            commission = abs(size) * self.p.commission
        
        # Add slippage cost
        if self.p.slippage_model:
            slippage = self.p.slippage_model.calculate_slippage(
                order_volume=abs(size),
                price=price
            )
            # Slippage affects effective price, convert to cost
            commission += abs(size) * slippage
        
        return commission
    
    def get_slippage(self, size, price, pseudoexec):
        """Calculate slippage using the slippage model."""
        if self.p.slippage_model:
            return self.p.slippage_model.calculate_slippage(
                order_volume=abs(size),
                price=price
            )
        return 0.0


class QuantMindBacktester:
    """
    Enhanced backtester with LEAN-style slippage and commission support.
    
    Features:
    - Dynamic strategy loading from code strings
    - Configurable slippage models (constant, volume-based, volatility-based)
    - Configurable commission models (per-lot, per-share, tiered)
    - Comprehensive result tracking
    """
    
    def __init__(
        self, 
        initial_cash: float = 10000.0, 
        commission: float = 0.001,
        slippage_model: Optional['SlippageModel'] = None,
        commission_model: Optional['CommissionModel'] = None
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_cash: Starting capital
            commission: Default commission rate (used if no commission model)
            slippage_model: LEAN slippage model instance
            commission_model: LEAN commission model instance
        """
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage_model = slippage_model
        self.commission_model = commission_model

    def run(self, strategy_code: str, data: pd.DataFrame, strategy_name: str = "MyStrategy") -> BacktestResult:
        """
        Dynamically loads and runs a strategy against the provided data.
        WARNING: Executes arbitrary code. Ensure sandboxing in production.
        """
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # Register LEAN-style slippage and commission models if provided
        if self.slippage_model or self.commission_model:
            lean_comm_info = LeanSlippageCommission(
                commission=self.commission,
                slippage_model=self.slippage_model,
                commission_model=self.commission_model
            )
            cerebro.broker.addcommissioninfo(lean_comm_info)
            logger.info(f"Registered LEAN models: slippage={type(self.slippage_model).__name__ if self.slippage_model else 'None'}, "
                        f"commission={type(self.commission_model).__name__ if self.commission_model else 'None'}")

        # Feed Data
        # Assuming Data is OHLCV with Datetime index
        data_feed = bt.feeds.PandasData(dataname=data)
        cerebro.adddata(data_feed)

        # Dynamic Strategy Loading
        # We need to extract the class that inherits from bt.Strategy
        namespace = {}
        # Capture stdout to log strategy output
        log_capture = io.StringIO()
        
        try:
            with contextlib.redirect_stdout(log_capture):
                exec(strategy_code, globals(), namespace)
                
                strategy_class = None
                for name, obj in namespace.items():
                    if isinstance(obj, type) and issubclass(obj, bt.Strategy) and obj is not bt.Strategy:
                        strategy_class = obj
                        break
                
                if not strategy_class:
                    return BacktestResult(0, 0, 0, 0, "Error: No class inheriting from backtrader.Strategy found in code.")

                # Add Analyzers
                cerebro.addstrategy(strategy_class)
                cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
                cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
                cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
                cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

                results = cerebro.run()
                strat = results[0]

                # Extract Metrics
                sharpe_analysis = strat.analyzers.sharpe.get_analysis()
                drawdown_analysis = strat.analyzers.drawdown.get_analysis()
                trade_analysis = strat.analyzers.trades.get_analysis()
                return_analysis = strat.analyzers.returns.get_analysis()
                
                sharpe = sharpe_analysis.get('sharperatio', 0.0) or 0.0
                max_dd = drawdown_analysis.get('max', {}).get('drawdown', 0.0)
                total_return = return_analysis.get('rtot', 0.0)
                total_trades = trade_analysis.get('total', {}).get('total', 0)

                return BacktestResult(
                    sharpe=sharpe, 
                    return_pct=total_return, 
                    drawdown=max_dd, 
                    trades=total_trades,
                    log=log_capture.getvalue()
                )

        except Exception as e:
            return BacktestResult(0, 0, 0, 0, f"Runtime Error: {str(e)}\nLog: {log_capture.getvalue()}")

