import backtrader as bt
import pandas as pd
import io
import sys
import contextlib
from typing import Dict, Any, Optional

class BacktestResult(dict):
    """Structured result from a backtest run."""
    def __init__(self, sharpe: float, return_pct: float, drawdown: float, trades: int, log: str):
        super().__init__(sharpe=sharpe, return_pct=return_pct, drawdown=drawdown, trades=trades, log=log)

class QuantMindBacktester:
    def __init__(self, initial_cash=10000.0, commission=0.001):
        self.initial_cash = initial_cash
        self.commission = commission

    def run(self, strategy_code: str, data: pd.DataFrame, strategy_name: str = "MyStrategy") -> BacktestResult:
        """
        Dynamically loads and runs a strategy against the provided data.
        WARNING: Executes arbitrary code. Ensure sandboxing in production.
        """
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

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

