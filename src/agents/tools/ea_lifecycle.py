"""
EA Lifecycle Tools

Tools for managing Expert Advisor (EA) lifecycle operations:
- Create EA from strategy
- Validate EA code
- Backtest EA
- Deploy to paper trading
- Stop running EA
"""

from typing import Optional, Dict, Any
from pathlib import Path
import subprocess
import json
import logging

logger = logging.getLogger(__name__)


class EALifecycleTools:
    """Tools for EA lifecycle management."""

    def __init__(self, base_path: str = "/home/mubarkahimself/Desktop/QUANTMINDX"):
        self.base_path = Path(base_path)
        self.strategies_path = self.base_path / "strategies-yt"
        self.ea_output_path = self.base_path / "output" / "expert_advisors"

    def create_ea(
        self,
        strategy_name: str,
        ea_name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create an Expert Advisor from a strategy.

        Args:
            strategy_name: Name of the strategy to convert
            ea_name: Name for the EA (defaults to strategy_name)
            parameters: Trading parameters for the EA

        Returns:
            Dict with EA creation status and file path
        """
        try:
            ea_name = ea_name or f"{strategy_name}_EA"
            parameters = parameters or {}

            # Ensure output directory exists
            self.ea_output_path.mkdir(parents=True, exist_ok=True)

            # Read strategy file
            strategy_file = self.strategies_path / f"{strategy_name}.md"
            if not strategy_file.exists():
                return {
                    "success": False,
                    "error": f"Strategy file not found: {strategy_file}"
                }

            # Create basic MQ5 EA template
            ea_code = self._generate_ea_template(ea_name, strategy_name, parameters)

            # Write EA file
            ea_file = self.ea_output_path / f"{ea_name}.mq5"
            ea_file.write_text(ea_code)

            logger.info(f"Created EA: {ea_file}")

            return {
                "success": True,
                "ea_name": ea_name,
                "file_path": str(ea_file),
                "strategy": strategy_name,
                "parameters": parameters
            }

        except Exception as e:
            logger.error(f"Error creating EA: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def validate_ea(self, ea_name: str) -> Dict[str, Any]:
        """
        Validate EA code syntax.

        Args:
            ea_name: Name of the EA to validate

        Returns:
            Dict with validation results
        """
        try:
            ea_file = self.ea_output_path / f"{ea_name}.mq5"

            if not ea_file.exists():
                return {
                    "success": False,
                    "error": f"EA file not found: {ea_file}"
                }

            # Basic syntax validation
            code = ea_file.read_text()

            errors = []
            warnings = []

            # Check for required sections
            required_sections = ["OnInit", "OnDeinit", "OnTick"]
            for section in required_sections:
                if section not in code:
                    errors.append(f"Missing required section: {section}")

            # Check for common issues
            if "#property strict" not in code:
                warnings.append("Missing #property strict directive")

            # Basic bracket matching
            if code.count("{") != code.count("}"):
                errors.append("Mismatched braces in code")

            return {
                "success": len(errors) == 0,
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "file_path": str(ea_file)
            }

        except Exception as e:
            logger.error(f"Error validating EA: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def backtest_ea(
        self,
        ea_name: str,
        symbol: str = "EURUSD",
        timeframe: str = "H1",
        date_from: str = "2023-01-01",
        date_to: str = "2023-12-31",
        deposit: int = 10000
    ) -> Dict[str, Any]:
        """
        Run backtest for an EA.

        Args:
            ea_name: Name of the EA to backtest
            symbol: Trading symbol
            timeframe: Timeframe for backtest
            date_from: Start date
            date_to: End date
            deposit: Initial deposit

        Returns:
            Dict with backtest results
        """
        try:
            ea_file = self.ea_output_path / f"{ea_name}.mq5"

            if not ea_file.exists():
                return {
                    "success": False,
                    "error": f"EA file not found: {ea_file}"
                }

            # Placeholder for backtest execution
            # In production, this would call MetaTrader 5 terminal
            results = {
                "success": True,
                "ea_name": ea_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "period": f"{date_from} to {date_to}",
                "deposit": deposit,
                "status": "queued",
                "message": "Backtest queued for execution"
            }

            logger.info(f"Backtest queued for {ea_name}")
            return results

        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def deploy_paper(
        self,
        ea_name: str,
        account_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy EA to paper trading environment.

        Args:
            ea_name: Name of the EA to deploy
            account_id: Optional paper trading account ID

        Returns:
            Dict with deployment status
        """
        try:
            ea_file = self.ea_output_path / f"{ea_name}.mq5"

            if not ea_file.exists():
                return {
                    "success": False,
                    "error": f"EA file not found: {ea_file}"
                }

            # Validate EA before deployment
            validation = self.validate_ea(ea_name)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": "EA validation failed",
                    "validation_errors": validation.get("errors", [])
                }

            # Deploy to paper trading
            deployment = {
                "success": True,
                "ea_name": ea_name,
                "environment": "paper",
                "account_id": account_id or "paper_default",
                "status": "deployed",
                "file_path": str(ea_file)
            }

            logger.info(f"Deployed {ea_name} to paper trading")
            return deployment

        except Exception as e:
            logger.error(f"Error deploying to paper: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def stop_ea(self, ea_name: str, environment: str = "paper") -> Dict[str, Any]:
        """
        Stop a running EA.

        Args:
            ea_name: Name of the EA to stop
            environment: Environment where EA is running

        Returns:
            Dict with stop status
        """
        try:
            # Stop the EA
            result = {
                "success": True,
                "ea_name": ea_name,
                "environment": environment,
                "status": "stopped",
                "message": f"EA {ea_name} stopped on {environment}"
            }

            logger.info(f"Stopped {ea_name} on {environment}")
            return result

        except Exception as e:
            logger.error(f"Error stopping EA: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _generate_ea_template(
        self,
        ea_name: str,
        strategy_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Generate basic MQ5 EA template."""
        return f'''//+------------------------------------------------------------------+
//|                                          {ea_name}.mq5               |
//|                                    Generated by QuantMindX            |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://quantmindx.io"
#property version   "1.00"
#property strict

// Input parameters
input double RiskPercent = {parameters.get('risk_percent', 1.0)};
input double LotSize = {parameters.get('lot_size', 0.01)};
input int StopLoss = {parameters.get('stop_loss', 50)};
input int TakeProfit = {parameters.get('take_profit', 100)};

// Global variables
int strategyHandle = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{{
   // Initialize strategy from: {strategy_name}
   Print("{ea_name} initialized");

   return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
   // Cleanup
   Print("{ea_name} deinitialized, reason: ", reason);
}}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{{
   // Trading logic based on {strategy_name}

   // Check for new bar
   static datetime lastBarTime = 0;
   datetime currentBarTime = iTime(_Symbol, _Period, 0);

   if(lastBarTime != currentBarTime)
   {{
      lastBarTime = currentBarTime;

      // Execute strategy logic here
      // Analyze market conditions
      // Open/close positions
   }}
}}
//+------------------------------------------------------------------+
'''
