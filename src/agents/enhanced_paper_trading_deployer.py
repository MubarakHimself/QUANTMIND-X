"""
Enhanced Paper Trading Deployer
===============================
The central orchestrator for multi-format paper trading deployments.

Composes:
- PaperTradingDeployer (Docker-based Python bot deployment)
- PineScriptConverter (Pine Script → MQL5 conversion)
- BotRegistry (bot manifest registration)
- LifecycleManager (tag progression)
- DemoAccountManager (MT5 demo account management)

Supports deployment of:
- MQL5 EAs (from GitHub or local files)
- Pine Script strategies (converted to MQL5)
- Python trading bots (Docker containers)
"""

import os
import uuid
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field

from src.agents.demo_account_manager import DemoAccountManager
from src.integrations.pine_script_converter import PineScriptConverter

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================

class BotFormat(str, Enum):
    """Supported bot format types."""
    EA = "ea"
    PINE_SCRIPT = "pine_script"
    PYTHON = "python"


class BotSource(str, Enum):
    """Source of the bot code."""
    GITHUB = "github"
    LOCAL = "local"
    TRADINGVIEW = "tradingview"
    INLINE = "inline"


# ============================================================================
# Request/Response Models
# ============================================================================

class EnhancedDeploymentRequest(BaseModel):
    """
    Request to deploy a bot to enhanced paper trading.
    
    Fields:
        format: Bot format (EA, Pine Script, or Python)
        source: Source type (GitHub, Local, TradingView, Inline)
        ea_id: EA ID (for GitHub source)
        pine_script_code: Pine Script code (for Pine Script format)
        python_code: Python code (for Python format)
        file_path: Local file path (for Local source)
        strategy_name: Name for the strategy
        symbol: Trading symbol (e.g., "EURUSD")
        timeframe: Timeframe (e.g., "H1", "M15")
        config: Additional configuration
        mt5_demo_login: MT5 demo account login
        mt5_demo_password: MT5 demo account password
        mt5_demo_server: MT5 demo server
        initial_tag: Initial lifecycle tag (default: @primal)
    """
    format: BotFormat = Field(..., description="Bot format type")
    source: BotSource = Field(..., description="Source of the bot code")
    ea_id: Optional[int] = Field(None, description="EA ID for GitHub source")
    pine_script_code: Optional[str] = Field(None, description="Pine Script code")
    python_code: Optional[str] = Field(None, description="Python code for Python bots")
    file_path: Optional[str] = Field(None, description="Local file path")
    strategy_name: str = Field(..., description="Strategy name")
    symbol: str = Field(default="EURUSD", description="Trading symbol")
    timeframe: str = Field(default="H1", description="Timeframe")
    config: Dict[str, Any] = Field(default_factory=dict, description="Additional config")
    mt5_demo_login: Optional[int] = Field(None, description="MT5 demo login")
    mt5_demo_password: Optional[str] = Field(None, description="MT5 demo password")
    mt5_demo_server: Optional[str] = Field(None, description="MT5 demo server")
    initial_tag: str = Field(default="@primal", description="Initial lifecycle tag")


class DeploymentResult(BaseModel):
    """
    Result of a deployment operation.
    
    Fields:
        bot_id: Unique bot identifier
        agent_id: Container/agent ID (for Python bots)
        status: Deployment status
        format: Bot format that was deployed
    """
    bot_id: str
    agent_id: Optional[str] = None
    status: str
    format: BotFormat


# ============================================================================
# Demo Account Model
# ============================================================================

class DemoAccountInfo(BaseModel):
    """Information about a demo account."""
    login: int
    server: str
    broker: str
    nickname: str
    account_type: str = "demo"
    is_active: bool = True


# ============================================================================
# Enhanced Paper Trading Deployer
# ============================================================================

class EnhancedPaperTradingDeployer:
    """
    Central orchestrator for enhanced paper trading deployments.
    
    Handles multi-format bot deployment to paper trading with:
    - MQL5 EA deployment to MT5 demo accounts
    - Pine Script to MQL5 conversion and deployment
    - Python bot Docker container deployment
    - Bot manifest registration with lifecycle tracking
    
    Usage:
        deployer = EnhancedPaperTradingDeployer()
        
        # Deploy a Pine Script strategy
        result = deployer.deploy_bot({
            "format": "pine_script",
            "source": "inline",
            "pine_script_code": "...",
            "strategy_name": "My Strategy",
            "symbol": "EURUSD",
            "timeframe": "H1",
            "mt5_demo_login": 123456,
            "mt5_demo_password": "password",
            "mt5_demo_server": "broker-demo"
        })
    """
    
    def __init__(self):
        """Initialize the enhanced deployer with all required components."""
        # Lazy-load components to avoid import errors
        self._base_deployer = None
        self._bot_registry = None
        self._lifecycle_manager = None
        self._github_ea_sync = None
        
        # Initialize components
        self._demo_manager = DemoAccountManager()
        self._pine_converter = PineScriptConverter()
        
        logger.info("EnhancedPaperTradingDeployer initialized")
    
    @property
    def demo_account_manager(self) -> DemoAccountManager:
        """Get the demo account manager."""
        return self._demo_manager
    
    @property
    def pine_converter(self) -> PineScriptConverter:
        """Get the Pine Script converter."""
        return self._pine_converter
    
    def _get_base_deployer(self):
        """Lazy-load the base PaperTradingDeployer."""
        if self._base_deployer is None:
            try:
                from mcp_mt5.paper_trading.deployer import PaperTradingDeployer
                self._base_deployer = PaperTradingDeployer()
            except ImportError as e:
                logger.error(f"PaperTradingDeployer not available: {e}")
                raise RuntimeError("PaperTradingDeployer not available")
        return self._base_deployer
    
    def _get_bot_registry(self):
        """Lazy-load the BotRegistry."""
        if self._bot_registry is None:
            try:
                from src.router.bot_manifest import BotRegistry
                self._bot_registry = BotRegistry()
            except ImportError as e:
                logger.warning(f"BotRegistry not available: {e}")
                self._bot_registry = None
        return self._bot_registry
    
    def _get_lifecycle_manager(self):
        """Lazy-load the LifecycleManager."""
        if self._lifecycle_manager is None:
            try:
                from src.router.lifecycle_manager import LifecycleManager
                self._lifecycle_manager = LifecycleManager()
            except ImportError as e:
                logger.warning(f"LifecycleManager not available: {e}")
                self._lifecycle_manager = None
        return self._lifecycle_manager
    
    def _get_github_ea_sync(self):
        """Lazy-load the GitHubEASync."""
        if self._github_ea_sync is None:
            try:
                # Get the EA library path from config or use default
                library_path = os.environ.get("EA_LIBRARY_PATH", "/data/ea-library")
                
                # Get repo URL from config or use default
                repo_url = os.environ.get(
                    "EA_REPO_URL",
                    "https://github.com/quantmindx/ea-library"
                )
                
                from src.integrations.github_ea_sync import GitHubEASync
                self._github_ea_sync = GitHubEASync(
                    repo_url=repo_url,
                    library_path=library_path
                )
            except ImportError as e:
                logger.warning(f"GitHubEASync not available: {e}")
                self._github_ea_sync = None
        return self._github_ea_sync
    
    def deploy_bot(self, request: EnhancedDeploymentRequest) -> DeploymentResult:
        """
        Deploy a bot based on the request.
        
        Args:
            request: Enhanced deployment request
            
        Returns:
            Deployment result with bot_id, agent_id, status, and format
        """
        logger.info(f"Deploying bot: format={request.format}, source={request.source}")
        
        if request.format == BotFormat.EA:
            return self._deploy_ea(request)
        elif request.format == BotFormat.PINE_SCRIPT:
            return self._deploy_pine_script(request)
        elif request.format == BotFormat.PYTHON:
            return self._deploy_python_bot(request)
        else:
            raise ValueError(f"Unknown format: {request.format}")
    
    def _deploy_ea(self, request: EnhancedDeploymentRequest) -> DeploymentResult:
        """
        Deploy an MQL5 EA to MT5 demo account.
        
        Args:
            request: Deployment request with EA details
            
        Returns:
            Deployment result
        """
        logger.info(f"Deploying EA: source={request.source}")
        
        # Get EA file path
        if request.source == BotSource.GITHUB:
            ea_file = self._load_ea_from_github(request.ea_id)
        elif request.source == BotSource.LOCAL:
            if not request.file_path:
                raise ValueError("file_path required for local source")
            ea_file = Path(request.file_path)
        else:
            raise ValueError(f"Unsupported source for EA: {request.source}")
        
        # Connect to demo account
        if request.mt5_demo_login:
            self._demo_manager.connect_demo_account(request.mt5_demo_login)
        
        # Deploy to MT5 demo (this would use MT5 bridge in real implementation)
        # For now, we'll create the manifest
        source_path = str(ea_file)
        
        # Create and register bot manifest
        manifest = self._create_and_register_bot_manifest(
            name=request.strategy_name,
            source_type="imported_ea",
            source_path=source_path,
            symbol=request.symbol,
            timeframe=request.timeframe,
            tag=request.initial_tag
        )
        
        return DeploymentResult(
            bot_id=manifest.bot_id,
            agent_id=None,
            status="deployed",
            format=BotFormat.EA
        )
    
    def _deploy_pine_script(self, request: EnhancedDeploymentRequest) -> DeploymentResult:
        """
        Deploy a Pine Script strategy by converting to MQL5.
        
        Args:
            request: Deployment request with Pine Script code
            
        Returns:
            Deployment result
        """
        logger.info("Converting Pine Script to MQL5...")
        
        if not request.pine_script_code:
            raise ValueError("pine_script_code required for Pine Script format")
        
        # Convert Pine Script to MQL5
        mql5_code = self._pine_converter.convert(request.pine_script_code)
        
        # Save MQL5 code to temp file
        mq5_file = self._save_mq5_file(mql5_code, request.strategy_name)
        
        # Connect to demo account
        if request.mt5_demo_login:
            self._demo_manager.connect_demo_account(request.mt5_demo_login)
        
        # Deploy to MT5 demo (would use MT5 bridge)
        
        # Create and register bot manifest
        manifest = self._create_and_register_bot_manifest(
            name=request.strategy_name,
            source_type="pine_script",
            source_path=str(mq5_file),
            symbol=request.symbol,
            timeframe=request.timeframe,
            tag=request.initial_tag
        )
        
        return DeploymentResult(
            bot_id=manifest.bot_id,
            agent_id=None,
            status="deployed",
            format=BotFormat.PINE_SCRIPT
        )
    
    def _deploy_python_bot(self, request: EnhancedDeploymentRequest) -> DeploymentResult:
        """
        Deploy a Python trading bot as Docker container.
        
        Args:
            request: Deployment request with Python code
            
        Returns:
            Deployment result
        """
        logger.info("Deploying Python bot...")
        
        if not request.python_code:
            raise ValueError("python_code required for Python format")
        
        # Prepare MT5 credentials
        mt5_credentials = None
        if request.mt5_demo_login and request.mt5_demo_password and request.mt5_demo_server:
            mt5_credentials = {
                "account": request.mt5_demo_login,
                "password": request.mt5_demo_password,
                "server": request.mt5_demo_server
            }
        
        # Deploy using base deployer
        base_deployer = self._get_base_deployer()
        
        result = base_deployer.deploy_agent(
            strategy_name=request.strategy_name,
            strategy_code=request.python_code,
            config=request.config,
            mt5_credentials=mt5_credentials,
            symbol=request.symbol,
            timeframe=request.timeframe
        )
        
        # Create and register bot manifest
        manifest = self._create_and_register_bot_manifest(
            name=request.strategy_name,
            source_type="python_bot",
            source_path=f"container:{result.agent_id}",
            symbol=request.symbol,
            timeframe=request.timeframe,
            tag=request.initial_tag
        )
        
        return DeploymentResult(
            bot_id=manifest.bot_id,
            agent_id=result.agent_id,
            status=result.status,
            format=BotFormat.PYTHON
        )
    
    def _load_ea_from_github(self, ea_id: int) -> Path:
        """
        Load an EA file from the GitHub library by ID.
        
        Args:
            ea_id: EA ID in the database
            
        Returns:
            Path to the EA file
            
        Raises:
            ValueError: If EA not found
        """
        github_sync = self._get_github_ea_sync()
        if not github_sync:
            raise ValueError("GitHub EA Sync not available")
        
        # Sync repository first
        github_sync.sync_repository()
        
        # Query database for EA
        from src.database.models import ImportedEA
        from src.database.engine import get_session
        
        session = get_session()
        try:
            ea = session.query(ImportedEA).filter(ImportedEA.id == ea_id).first()
            if not ea:
                raise ValueError(f"EA with ID {ea_id} not found")
            
            # Construct file path
            ea_path = Path(github_sync.library_path) / ea.ea_filepath
            if not ea_path.exists():
                raise ValueError(f"EA file not found: {ea_path}")
            
            return ea_path
        finally:
            session.close()
    
    def _save_mq5_file(self, mql5_code: str, strategy_name: str) -> Path:
        """
        Save MQL5 code to a temporary .mq5 file.
        
        Args:
            mql5_code: MQL5 source code
            strategy_name: Name for the strategy
            
        Returns:
            Path to the saved file
        """
        # Use temp directory or configured path
        temp_dir = Path(os.environ.get("TEMP_MQ5_PATH", "/tmp/quantmindx/mq5"))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize filename
        safe_name = "".join(c for c in strategy_name if c.isalnum() or c in "-_")
        filename = f"{safe_name}_{uuid.uuid4().hex[:8]}.mq5"
        file_path = temp_dir / filename
        
        file_path.write_text(mql5_code, encoding="utf-8")
        logger.info(f"Saved MQL5 code to {file_path}")
        
        return file_path
    
    def _create_and_register_bot_manifest(
        self,
        name: str,
        source_type: str,
        source_path: str,
        symbol: str,
        timeframe: str,
        tag: str
    ):
        """
        Create and register a bot manifest.
        
        Args:
            name: Bot name
            source_type: Source type (imported_ea, pine_script, python_bot)
            source_path: Path to source
            symbol: Trading symbol
            timeframe: Timeframe
            tag: Initial tag
            
        Returns:
            Created BotManifest
        """
        # Lazy-load required classes
        try:
            from src.router.bot_manifest import (
                BotManifest,
                BotRegistry,
                TradingMode,
                StrategyType,
                TradeFrequency
            )
        except ImportError as e:
            logger.error(f"BotManifest classes not available: {e}")
            raise RuntimeError("Cannot create bot manifest")
        
        # Generate bot ID
        bot_id = f"bot_{uuid.uuid4().hex[:12]}"
        
        # Create manifest
        manifest = BotManifest(
            bot_id=bot_id,
            name=name,
            description=f"Paper trading bot: {name}",
            strategy_type=StrategyType.TRENDING,  # Default, can be inferred
            frequency=TradeFrequency.MEDIUM,
            symbols=[symbol],
            timeframes=[timeframe],
            tags=[tag],
            trading_mode=TradingMode.PAPER,
            source_type=source_type,
            source_path=source_path
        )
        
        # Register with BotRegistry
        registry = self._get_bot_registry()
        if registry:
            try:
                registry.register(manifest)
                logger.info(f"Registered bot manifest: {bot_id}")
            except Exception as e:
                logger.warning(f"Failed to register bot manifest: {e}")
        
        # Register with LifecycleManager
        lifecycle = self._get_lifecycle_manager()
        if lifecycle:
            try:
                lifecycle.register_bot(manifest.bot_id)
                logger.info(f"Registered bot with lifecycle manager: {bot_id}")
            except Exception as e:
                logger.warning(f"Failed to register with lifecycle manager: {e}")
        
        return manifest
    
    def list_bots_by_tag(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all bots grouped by tag.
        
        Returns:
            Dictionary mapping tags to lists of bot info
        """
        registry = self._get_bot_registry()
        if not registry:
            return {}
        
        bots_by_tag: Dict[str, List[Dict[str, Any]]] = {
            "@primal": [],
            "@pending": [],
            "@perfect": [],
            "@live": []
        }
        
        all_bots = registry.list_all()
        
        for bot in all_bots:
            for tag in bot.tags:
                if tag in bots_by_tag:
                    bots_by_tag[tag].append({
                        "bot_id": bot.bot_id,
                        "name": bot.name,
                        "format": self._infer_format(bot.source_type),
                        "symbol": bot.symbols[0] if bot.symbols else None,
                        "timeframe": bot.timeframes[0] if bot.timeframes else None,
                        "trading_mode": bot.trading_mode.value,
                        "source_type": bot.source_type
                    })
        
        return bots_by_tag
    
    def _infer_format(self, source_type: str) -> str:
        """Infer bot format from source type."""
        if source_type == "imported_ea":
            return "EA"
        elif source_type == "pine_script":
            return "Pine Script"
        elif source_type == "python_bot":
            return "Python"
        return "Unknown"
    
    def promote_bot(self, bot_id: str) -> Dict[str, Any]:
        """
        Manually promote a bot's tag.
        
        Args:
            bot_id: Bot ID to promote
            
        Returns:
            Promotion result
        """
        lifecycle = self._get_lifecycle_manager()
        if not lifecycle:
            return {"success": False, "error": "LifecycleManager not available"}
        
        try:
            result = lifecycle.manually_promote_bot(bot_id)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Failed to promote bot {bot_id}: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# Singleton Instance
# ============================================================================

_deployer: Optional[EnhancedPaperTradingDeployer] = None


def get_enhanced_deployer() -> EnhancedPaperTradingDeployer:
    """Get or create the global EnhancedPaperTradingDeployer instance."""
    global _deployer
    if _deployer is None:
        _deployer = EnhancedPaperTradingDeployer()
    return _deployer
