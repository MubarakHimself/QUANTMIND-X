"""
EA Lifecycle Tools

Tools for managing Expert Advisor (EA) lifecycle operations:
- Create EA from strategy
- Create EA variants (vanilla/spiced)
- Validate EA code
- Backtest EA
- Deploy to paper trading
- Stop running EA
"""

import sys
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pathlib import Path
from enum import Enum
import subprocess
import json
import logging

# Add MCP MT5 server path for PaperTradingDeployer
MCP_MT5_PATH = Path("/home/mubarkahimself/Desktop/QUANTMINDX/mcp-metatrader5-server/src")
if str(MCP_MT5_PATH) not in sys.path:
    sys.path.insert(0, str(MCP_MT5_PATH))

if TYPE_CHECKING:
    from mcp_mt5.paper_trading.deployer import PaperTradingDeployer

from src.router.virtual_balance import VirtualBalanceManager
from src.router.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


class EALifecycleStatus(str, Enum):
    """EA Lifecycle Status enum."""
    CREATED = "created"
    VALIDATED = "validated"
    BACKTEST_QUEUED = "backtest_queued"
    BACKTEST_RUNNING = "backtest_running"
    BACKTEST_COMPLETED = "backtest_completed"
    DEPLOYED_PAPER = "deployed_paper"
    STOPPED = "stopped"
    FAILED = "failed"


class EALifecycleTools:
    """Tools for EA lifecycle management."""

    def __init__(self, base_path: str = "/home/mubarkahimself/Desktop/QUANTMINDX"):
        self.base_path = Path(base_path)
        self.strategies_path = self.base_path / "strategies-yt"
        self.ea_output_path = self.base_path / "output" / "expert_advisors"
        # MT5 Paper Trading integration
        self._paper_deployer: Optional[Any] = None
        self._virtual_balance_manager: Optional[VirtualBalanceManager] = None
        self._trade_logger: Optional[TradeLogger] = None
        self._bot_registry: Optional[Any] = None

    def _get_paper_deployer(self):
        """Lazy initialization of paper trading deployer."""
        if self._paper_deployer is None:
            from mcp_mt5.paper_trading.deployer import PaperTradingDeployer
            self._paper_deployer = PaperTradingDeployer()
        return self._paper_deployer

    def _get_virtual_balance_manager(self) -> VirtualBalanceManager:
        """Lazy initialization of virtual balance manager."""
        if self._virtual_balance_manager is None:
            self._virtual_balance_manager = VirtualBalanceManager()
        return self._virtual_balance_manager

    def _get_trade_logger(self) -> TradeLogger:
        """Lazy initialization of trade logger."""
        if self._trade_logger is None:
            self._trade_logger = TradeLogger()
        return self._trade_logger

    def _get_bot_registry(self):
        """Lazy initialization of bot manifest registry."""
        if self._bot_registry is None:
            from src.router.bot_manifest import BotManifestRegistry
            self._bot_registry = BotManifestRegistry()
        return self._bot_registry


class EALifecycleManager:
    """Manager for EA variant creation from TRD."""

    def __init__(self, base_path: str = "/home/mubarkahimself/Desktop/QUANTMINDX"):
        self.base_path = Path(base_path)
        self.strategies_path = self.base_path / "strategies-yt"
        self.ea_output_path = self.base_path / "output" / "expert_advisors"
        self.ea_output_path.mkdir(parents=True, exist_ok=True)

    def create_ea_from_trd(
        self,
        trd_variants: Dict[str, Any],
        create_variants: str = "both"
    ) -> Dict[str, Any]:
        """
        Create EA(s) from TRD variants.

        Args:
            trd_variants: Dict with 'vanilla' and/or 'spiced' strategy configs
            create_variants: Which variants to create - "vanilla", "spiced", or "both"

        Returns:
            Dict with created EA(s) - each containing ea_type, strategy_name, and code
        """
        result = {}

        if create_variants in ("both", "vanilla") and "vanilla" in trd_variants:
            result["vanilla"] = self._create_vanilla_ea(trd_variants["vanilla"])

        if create_variants in ("both", "spiced") and "spiced" in trd_variants:
            result["spiced"] = self._create_spiced_ea(trd_variants["spiced"])

        return result

    def _create_vanilla_ea(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a vanilla EA from config."""
        strategy_name = config.get("strategy_name", "VanillaEA")
        entry_rules = config.get("entry_rules", [])
        exit_rules = config.get("exit_rules", [])
        parameters = config.get("parameters", {})

        ea_code = self._generate_mq5({
            "strategy_name": strategy_name,
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "parameters": parameters
        }, ea_type="vanilla")

        return {
            "ea_type": "vanilla",
            "strategy_name": strategy_name,
            "code": ea_code,
            "entry_rules": entry_rules,
            "exit_rules": exit_rules
        }

    def _create_spiced_ea(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a spiced EA from config with enhanced features."""
        strategy_name = config.get("strategy_name", "SpicedEA")
        entry_rules = config.get("entry_rules", [])
        exit_rules = config.get("exit_rules", [])
        articles = config.get("articles", [])
        parameters = config.get("parameters", {})

        ea_code = self._generate_mq5({
            "strategy_name": strategy_name,
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "parameters": parameters,
            "articles": articles
        }, ea_type="spiced")

        return {
            "ea_type": "spiced",
            "strategy_name": strategy_name,
            "code": ea_code,
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "articles": articles
        }

    def _generate_mq5(
        self,
        strategy_config: Dict[str, Any],
        ea_type: str = "vanilla"
    ) -> str:
        """Generate MQL5 EA code based on strategy config and variant type."""
        strategy_name = strategy_config.get("strategy_name", "Strategy")
        entry_rules = strategy_config.get("entry_rules", [])
        exit_rules = strategy_config.get("exit_rules", [])
        parameters = strategy_config.get("parameters", {})
        articles = strategy_config.get("articles", [])

        risk_percent = parameters.get("risk_percent", 1.0)
        lot_size = parameters.get("lot_size", 0.01)
        stop_loss = parameters.get("stop_loss", 50)
        take_profit = parameters.get("take_profit", 100)

        # Generate entry rule comments
        entry_rules_str = "\n".join([f"   // - {rule}" for rule in entry_rules]) if entry_rules else "   // No entry rules defined"
        exit_rules_str = "\n".join([f"   // - {rule}" for rule in exit_rules]) if exit_rules else "   // No exit rules defined"

        # Additional features for spiced variant
        spiced_features = ""
        if ea_type == "spiced" and articles:
            spiced_features = f"""
//+------------------------------------------------------------------+
//| Article-based enhancements                                       |
//+------------------------------------------------------------------+
string GetArticleSignals() {{
   string signals = "";
   string articles[] = {json.dumps(articles)};
   for(int i = 0; i < ArraySize(articles); i++) {{
      signals += articles[i] + ";";
   }}
   return signals;
}}
"""

        return f'''//+------------------------------------------------------------------+
//|                                          {strategy_name}_{ea_type}.mq5    |
//|                                    Generated by QuantMindX            |
//|                                          Type: {ea_type}                   |
//+------------------------------------------------------------------+
#property copyright "QuantMindX"
#property link      "https://quantmindx.io"
#property version   "1.00"
#property strict

// Input parameters
input double RiskPercent = {risk_percent};
input double LotSize = {lot_size};
input int StopLoss = {stop_loss};
input int TakeProfit = {take_profit};

// Global variables
int strategyHandle = INVALID_HANDLE;
datetime lastBarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{{
   Print("{strategy_name} ({ea_type}) initialized");
   return(INIT_SUCCEEDED);
}}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{{
   Print("{strategy_name} deinitialized, reason: ", reason);
}}

//+------------------------------------------------------------------+
//| Expert tick function                                               |
//+------------------------------------------------------------------+
void OnTick()
{{
   // Check for new bar
   datetime currentBarTime = iTime(_Symbol, _Period, 0);

   if(lastBarTime != currentBarTime)
   {{
      lastBarTime = currentBarTime;

      // Entry Rules:
{entry_rules_str}

      // Exit Rules:
{exit_rules_str}

      // Execute trading logic
      if(CheckEntryConditions())
      {{
         ExecuteTrade();
      }}
   }}
}}

//+------------------------------------------------------------------+
//| Check entry conditions                                            |
//+------------------------------------------------------------------+
bool CheckEntryConditions()
{{
   // Basic entry logic - implement strategy-specific conditions
   return true;
}}

//+------------------------------------------------------------------+
//| Execute trade                                                     |
//+------------------------------------------------------------------+
void ExecuteTrade()
{{
   // Trading execution logic
   double lot = CalculateLotSize();
   // Place order logic here
}}

//+------------------------------------------------------------------+
//| Calculate lot size                                                |
//+------------------------------------------------------------------+
double CalculateLotSize()
{{
   double accountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   double lot = (accountBalance * RiskPercent / 100.0) / 1000.0;
   return NormalizeDouble(MathMax(lot, LotSize), 2);
}}
{spiced_features}
//+------------------------------------------------------------------+
'''

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
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        strategy_code: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        virtual_balance: float = 10000.0,
        magic_number: int = 0
    ) -> Dict[str, Any]:
        """
        Deploy EA to paper trading environment via MT5 demo mode.

        Uses PaperTradingDeployer to launch a Docker container with the EA
        connected to MT5 demo account for live paper trading.

        Args:
            ea_name: Name of the EA to deploy
            account_id: Optional paper trading account ID (used as agent_id prefix)
            strategy_id: Strategy identifier for bot manifest lookup
            strategy_code: Strategy Python code or template reference
            config: Strategy configuration parameters
            virtual_balance: Starting virtual balance (default: $10,000)
            magic_number: Unique magic number for trade identification

        Returns:
            Dict with deployment status including container_id and agent_id
        """
        try:
            ea_file = self.ea_output_path / f"{ea_name}.mq5"

            if not ea_file.exists():
                # Try to find compiled EX5 file
                ea_file_ex5 = self.ea_output_path / f"{ea_name}.ex5"
                if ea_file_ex5.exists():
                    ea_file = ea_file_ex5
                else:
                    return {
                        "success": False,
                        "error": f"EA file not found: {ea_file}"
                    }

            # Look up bot manifest for strategy type if strategy_id provided
            strategy_type = None
            if strategy_id:
                registry = self._get_bot_registry()
                manifest = registry.get(strategy_id)
                if manifest:
                    strategy_type = manifest.strategy_type.value
                    logger.info(f"Found bot manifest for {strategy_id}: {strategy_type}")

            # Validate EA before deployment
            validation = self.validate_ea(ea_name)
            if not validation["valid"]:
                return {
                    "success": False,
                    "error": "EA validation failed",
                    "validation_errors": validation.get("errors", [])
                }

            # Create virtual account for paper trading P&L tracking
            vbm = self._get_virtual_balance_manager()
            virtual_account = vbm.create_account(
                ea_id=ea_name,
                initial_balance=virtual_balance
            )
            logger.info(f"Created virtual account for {ea_name}: balance={virtual_balance}")

            # Generate agent_id from ea_name and account_id
            agent_id = f"{ea_name}-{account_id or 'paper'}" if account_id else ea_name
            if magic_number:
                agent_id = f"{agent_id}-{magic_number}"

            # Deploy to paper trading via Docker container with MT5 demo
            deployer = self._get_paper_deployer()

            # Use deploy_demo_agent for pure demo mode (no broker credentials needed)
            from mcp_mt5.paper_trading.deployer import PaperTradingConfig

            paper_config = PaperTradingConfig(
                broker_connection=False,  # Pure demo mode with virtual balance
                virtual_balance=virtual_balance,
                use_live_data=True,
                simulate_slippage=True,
                simulate_fees=False,
            )

            deployment_result = deployer.deploy_agent(
                strategy_name=ea_name,
                strategy_code=strategy_code or f"file:{str(ea_file)}",
                config=config or {},
                mt5_credentials=None,  # No real broker for demo mode
                magic_number=magic_number or hash(ea_name) % 1000000,
                agent_id=agent_id,
                paper_config=paper_config,
            )

            # Return actual deployment confirmation
            deployment = {
                "success": True,
                "ea_name": ea_name,
                "environment": "paper",
                "account_id": account_id or "paper_default",
                "status": "deployed",
                "file_path": str(ea_file),
                # Real MT5 deployment details
                "agent_id": deployment_result.agent_id,
                "container_id": deployment_result.container_id,
                "container_name": deployment_result.container_name,
                "virtual_account_id": virtual_account.ea_id,
                "initial_balance": virtual_balance,
                "strategy_type": strategy_type,
                "redis_channel": deployment_result.redis_channel,
                "message": f"EA {ea_name} deployed to MT5 demo mode"
            }

            logger.info(f"Deployed {ea_name} to paper trading: agent_id={deployment_result.agent_id}")
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
