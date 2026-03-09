"""
The Commander (Execution Layer)
Responsible for selecting and dispatching Bots based on Regime.

V2: Integrated with BotManifest system for tag-based authorization.
Only @primal tagged bots can participate in the Strategy Auction.
"""

from typing import List, Dict, Optional, TYPE_CHECKING, Any
import logging
import json

if TYPE_CHECKING:
    from src.router.bot_manifest import BotRegistry, BotManifest
    from src.router.sentinel import RegimeReport
    from src.router.governor import Governor
    from src.router.multi_timeframe_sentinel import MultiTimeframeSentinel, Timeframe
    from src.router.routing_matrix import RoutingMatrix, RoutingDecision

from datetime import datetime, timezone
from src.router.session_detector import SessionDetector, TradingSession

from src.router.dynamic_bot_limits import DynamicBotLimiter
from src.router.routing_matrix import RoutingMatrix

# APScheduler for LifecycleManager scheduling
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False
    AsyncIOScheduler = None
    CronTrigger = None

logger = logging.getLogger(__name__)


class Commander:
    """
    The Base Commander implementing the Strategy Auction logic.
    
    V2 Integration:
    - Uses BotRegistry for bot lookup
    - Only @primal tagged bots can participate
    - Regime-aware filtering based on strategy type
    - Global 50-bot limit enforcement
    """
    
    def __init__(self, bot_registry: Optional["BotRegistry"] = None, governor: Optional["Governor"] = None):
        """
        Initialize Commander with optional BotRegistry and Governor integration.

        Args:
            bot_registry: BotRegistry instance for manifest-based bot management.
                         If None, falls back to legacy dict-based storage.
            governor: Optional Governor instance for position sizing (if None, defaults to EnhancedGovernor
                     for fee-aware Kelly sizing). If explicitly set to base Governor, position_size
                     filtering is skipped to allow legacy mode to work.
        """
        self._bot_registry = bot_registry

        # Initialize Routing Matrix for bot-to-account assignment
        self._routing_matrix = RoutingMatrix()
        logger.info("Commander: Initialized RoutingMatrix for bot-to-account routing")

        # Guard: Ensure _governor is never None - default to EnhancedGovernor for fee-aware Kelly sizing
        # This ensures bots are not filtered out by position_size > 0 check
        if governor is None:
            try:
                from src.router.enhanced_governor import EnhancedGovernor
                self._governor = EnhancedGovernor()
                self._use_enhanced_sizing = True
                logger.info("Commander: No governor provided - created EnhancedGovernor for fee-aware Kelly sizing")
            except ImportError:
                # Fallback to base Governor if EnhancedGovernor unavailable
                from src.router.governor import Governor
                self._governor = Governor()
                self._use_enhanced_sizing = False
                logger.info("Commander: EnhancedGovernor not available - created base Governor (position_size filtering disabled)")
        else:
            self._governor = governor
            # Detect if using base Governor (legacy mode) or EnhancedGovernor
            governor_class_name = governor.__class__.__name__
            self._use_enhanced_sizing = (governor_class_name == "EnhancedGovernor")
            logger.info(f"Commander: Using {governor_class_name} governor (enhanced_sizing={self._use_enhanced_sizing})")
        self.active_bots: Dict = {}  # Legacy support
        self.regime_map: Dict = self._load_regime_map()
        
        # Regime → Strategy Type mapping
        self.regime_strategy_map = {
            "TREND_STABLE": ["SCALPER", "STRUCTURAL", "HFT"],
            "RANGE_STABLE": ["SCALPER", "STRUCTURAL"],
            "BREAKOUT_PRIME": ["STRUCTURAL", "SWING"],
            "HIGH_CHAOS": [],  # No bots authorized
            "NEWS_EVENT": [],  # Kill zone - no new trades
            "UNCERTAIN": ["STRUCTURAL"],  # Conservative only
        }
        
        # Multi-timeframe sentinel (Comment 1: shared from StrategyRouter)
        self._multi_timeframe_sentinel: Optional["MultiTimeframeSentinel"] = None
        
        # Lifecycle Manager scheduler (runs daily at 3:00 AM UTC)
        self._scheduler: Optional[Any] = None

    def check_drawdown_limits(
        self,
        account_book: str,
        prop_firm_name: Optional[str],
        current_drawdown: float
    ) -> tuple[bool, str]:
        """Check if trade should be blocked due to drawdown limits.

        Args:
            account_book: "personal" or "prop_firm"
            prop_firm_name: Name of prop firm (required if account_book is "prop_firm")
            current_drawdown: Current drawdown as decimal (e.g., 0.04 for 4%)

        Returns:
            Tuple of (should_block: bool, reason: str)
                - should_block: True if trade should be blocked
                - reason: Explanation if blocked, empty string otherwise
        """
        from src.risk.prop_firm_overlay import PropFirmRiskOverlay

        overlay = PropFirmRiskOverlay(firm_name=prop_firm_name)
        return overlay.should_block_trade(account_book, prop_firm_name, current_drawdown)
        if APSCHEDULER_AVAILABLE:
            self._setup_lifecycle_scheduler()
    
    def _setup_lifecycle_scheduler(self):
        """Setup APScheduler for LifecycleManager daily checks."""
        try:
            from src.router.lifecycle_manager import LifecycleManager
            
            # Create LifecycleManager instance and keep reference to prevent GC
            self._lifecycle_manager = LifecycleManager()
            
            self._scheduler = AsyncIOScheduler()
            
            # Schedule daily check at 3:00 AM UTC using bound method
            self._scheduler.add_job(
                self._lifecycle_manager.run_daily_check,
                CronTrigger(hour=3, minute=0),
                id='lifecycle_daily_check',
                name='Lifecycle Manager Daily Check',
                replace_existing=True
            )
            
            # Start scheduler after job registration
            self._scheduler.start()
            logger.info("Commander: LifecycleManager scheduler started (runs daily at 3:00 AM UTC)")
        except Exception as e:
            logger.warning(f"Commander: Failed to start LifecycleManager scheduler: {e}")
    
    def _load_regime_map(self) -> Dict:
        """Load regime → bot mapping configuration."""
        # TODO: Load from config file
        return {}

    def _derive_mode_from_context(self, regime_report: "RegimeReport") -> str:
        """
        Derive trading mode from context (EA config, regime, etc.).
        
        Default to 'live' for production trading. Override for demo testing.
        
        Args:
            regime_report: Current regime report
            
        Returns:
            Trading mode string ('demo' or 'live')
        """
        # Check if regime report has mode hint
        if hasattr(regime_report, 'mode'):
            return regime_report.mode
        
        # Check environment variable
        import os
        env_mode = os.getenv("TRADING_MODE", "live").lower()
        if env_mode in ("demo", "paper"):
            return "demo"
        
        # Default to live
        return "live"

    def _get_account_for_mode(self, base_account_id: str, mode: Optional[str]) -> str:
        """
        Apply mode-aware account selection.
        
        For demo mode, routes to demo_account.
        For live mode, uses the base account from routing matrix.
        
        Args:
            base_account_id: Base account from routing decision
            mode: Trading mode ('demo' or 'live')
            
        Returns:
            Account ID to use for trading
        """
        if mode == "demo":
            return "demo_account"
        return base_account_id

    def _get_broker_for_account(self, account_id: str, mode: Optional[str] = None) -> str:
        """
        Map account ID to broker ID for fee-aware Kelly position sizing.

        Account-to-Broker Mapping from PDF specification:
        - Machine Gun account (HFT/Scalpers) → RoboForex Prime (raw ECN)
        - Sniper account (Structural/ICT) → Exness Raw (raw spread)
        - Demo account → MT5 Default

        Args:
            account_id: Account identifier from routing decision
            mode: Trading mode ('demo' or 'live') - demo uses mt5_default

        Returns:
            broker_id for fee calculation
        """
        # For demo mode, always use mt5_default broker
        if mode == "demo":
            return "mt5_default"
        
        account_to_broker = {
            "account_a_machine_gun": "roboforex_prime",  # HFT/Scalpers on RoboForex Prime
            "account_b_sniper": "exness_raw",            # Structural/ICT on Exness Raw
            "demo_account": "mt5_default",               # Demo account default
        }

        broker_id = account_to_broker.get(account_id, "mt5_default")

        if account_id not in account_to_broker:
            logger.warning(
                f"Unknown account_id '{account_id}' - defaulting to mt5_default broker"
            )
        else:
            logger.debug(f"Account-to-broker mapping: {account_id} → {broker_id}")

        return broker_id

    @property
    def bot_registry(self) -> Optional["BotRegistry"]:
        """Lazy load BotRegistry if not provided."""
        if self._bot_registry is None:
            try:
                from src.router.bot_manifest import BotRegistry
                self._bot_registry = BotRegistry()
                logger.info("Commander: Loaded BotRegistry")
            except Exception as e:
                logger.warning(f"Commander: Could not load BotRegistry: {e}")
        return self._bot_registry
    
    def run_auction(self, regime_report: "RegimeReport", account_balance: Optional[float] = None, broker_id: str = "mt5_default", current_utc: Optional[datetime] = None, mode: Optional[str] = None) -> List[dict]:
        """
        Conducts the Bot Auction for the current Regime.
        
        V2: Only @primal tagged bots can participate.
        
        SPEC: Session-Aware Bot Filtering
        ==================================
        Per spec requirement Comment 1, this method now accepts current_utc timestamp
        for session-aware filtering:
        
        1. If current_utc is provided (from backtest bar or live trading), it is used
           for session detection and bot filtering
        2. Session filtering happens BEFORE bot selection:
           - Bots with session preferences are filtered to matching sessions
           - Bots with time_windows are checked against current UTC time
           - Bots without preferences can trade in any session
        3. Only after session filtering are chaos score and other regime filters applied
        
        Backtest Integration (Comment 1):
        - Backtests pass each bar's UTC timestamp via current_utc parameter
        - SessionDetector.detect_session(current_utc) determines session
        - Regime history includes UTC timestamps for audit trail
        - This ensures backtests exercise session filters per spec
        
        Args:
            regime_report: Data from Sentinel (Regime Enum, Chaos Score)
            account_balance: Account balance for position sizing (optional)
            broker_id: Broker identifier for fee-aware Kelly (default: mt5_default)
            current_utc: UTC timestamp for session-aware filtering. If None, uses
                        current UTC time. If provided (e.g., from backtest bar),
                        enables session filtering.
            
        Returns:
            List of authorized Bot Dispatches with manifest data, filtered by:
            - Regime (from Sentinel)
            - Session (from SessionDetector)
            - Time windows (from bot preferences)
            - Chaos score
            - Win rate
        """
        # Default to current UTC time if not provided
        if current_utc is None:
            current_utc = datetime.now(timezone.utc)
        elif current_utc.tzinfo is None:
            current_utc = current_utc.replace(tzinfo=timezone.utc)
        elif current_utc.tzinfo != timezone.utc:
            current_utc = current_utc.astimezone(timezone.utc)
        else:
            current_utc = current_utc

        # Detect current session
        current_session = SessionDetector.detect_session(current_utc)
        
        # Derive mode if not provided (Comment 1: mode-aware routing)
        # Mode can come from: explicit parameter > EA config/registry > bot tags > default to 'live'
        if mode is None:
            mode = self._derive_mode_from_context(regime_report)
        
        mode_tag = "[DEMO]" if mode == "demo" else "[LIVE]"
        logger.info(f"{mode_tag} Running auction for session: {current_session.value}, UTC time: {current_utc.isoformat()}, mode={mode}")

        # Get eligible bots for current regime and session (Comment 1 fix)
        # V4: Pass mode to filter by trading_mode
        eligible_bots = self._get_bots_for_regime_and_session(
            regime_report.regime, current_session, current_utc, mode=mode
        )
        
        # Apply chaos-based filtering (high chaos reduces bot selection to lower frequency bots)
        if regime_report.chaos_score > 0.6:
            logger.info("High chaos detected - reducing bot selection")
            eligible_bots = [b for b in eligible_bots if b.get('frequency') in ['LOW', 'MEDIUM']]
        
        # Rank bots by score AND win_rate (higher is better) and sort descending
        ranked_bots = sorted(eligible_bots, key=lambda b: b.get('score', 0.0) * b.get('win_rate', 0.5), reverse=True)
        
        # Default account_balance if falsy (prevents crashes in legacy usage - Comment 1)
        account_balance = account_balance or getattr(self._governor, '_daily_start_balance', 100000.0)
        logger.debug(f"Commander: Defaulted account_balance to {account_balance} (multi-timeframe)")

        # Use DynamicBotLimiter for max_selection
        # Comment 2 fix: Use can_add_bot() API instead of hard cap
        limiter = DynamicBotLimiter()
        max_bots = limiter.get_max_bots(account_balance)
        active_positions = self._count_active_positions()
        
        # Use can_add_bot() for each candidate bot instead of hard cap
        # This respects tier-based limits and safety buffer
        selected_bots = []
        current_bot_count = active_positions
        
        for bot in ranked_bots:
            can_add, reason = limiter.can_add_bot(account_balance, current_bot_count)
            if can_add:
                selected_bots.append(bot)
                current_bot_count += 1
                logger.debug(f"Bot {bot.get('bot_id')} added: {reason}")
            else:
                logger.debug(f"Bot {bot.get('bot_id')} rejected: {reason}")
        
        top_bots = selected_bots

        # 5. Calculate position sizes for each bot using Governor (if available)
        # Track routing statistics for audit
        routing_stats = {"total": len(top_bots), "routed": 0, "rejected": 0}
        dispatches = []

        for bot in top_bots:
            bot_id = bot.get('bot_id', 'unknown')

            # === ROUTING MATRIX INTEGRATION ===
            # Route bot to appropriate account based on manifest
            manifest = bot.get('manifest')
            if manifest is None:
                # Comment 2 fix: Try to build a BotManifest from bot_data for legacy bots
                # instead of immediately defaulting to demo_account
                manifest = self._build_manifest_from_bot_data(bot_id, bot)
                if manifest is not None:
                    # Route through RoutingMatrix with constructed manifest
                    routing_decision = self._routing_matrix.route_bot(manifest)
                    if routing_decision.is_approved:
                        # Comment 1 fix: Enforce minimum routing compatibility score threshold
                        MIN_ROUTING_SCORE = 70
                        if routing_decision.priority_score < MIN_ROUTING_SCORE:
                            logger.warning(
                                f"Bot {bot_id} rejected by routing matrix: compatibility score "
                                f"{routing_decision.priority_score:.2f} below minimum threshold {MIN_ROUTING_SCORE}"
                            )
                            routing_stats["rejected"] += 1
                            continue
                        base_account_id = routing_decision.assigned_account
                        # Comment 1: Apply mode-aware account selection
                        account_id = self._get_account_for_mode(base_account_id, mode)
                        routed_broker_id = self._get_broker_for_account(account_id, mode)
                        routing_score = routing_decision.priority_score
                        logger.info(
                            f"{mode_tag} Bot {bot_id} (legacy) routed to account={account_id}, broker={routed_broker_id}, "
                            f"score={routing_score:.2f}, mode={mode}"
                        )
                        routing_stats["routed"] += 1
                    else:
                        logger.warning(
                            f"Bot {bot_id} rejected by routing matrix: {routing_decision.rejection_reason}"
                        )
                        routing_stats["rejected"] += 1
                        continue
                else:
                    # Only fall back to demo_account when routing genuinely cannot be performed
                    logger.warning(f"Bot {bot_id} has no manifest and cannot build one - falling back to demo_account")
                    account_id = "demo_account"
                    routed_broker_id = broker_id  # Use passed broker_id as fallback
                    routing_score = 0.0
            else:
                # Get routing decision from RoutingMatrix
                routing_decision = self._routing_matrix.route_bot(manifest)

                if not routing_decision.is_approved:
                    # Bot rejected by routing matrix - skip this bot
                    logger.warning(
                        f"Bot {bot_id} rejected by routing matrix: {routing_decision.rejection_reason}"
                    )
                    routing_stats["rejected"] += 1
                    continue

                # Get account assignment and map to broker
                base_account_id = routing_decision.assigned_account
                # Comment 1: Apply mode-aware account selection
                account_id = self._get_account_for_mode(base_account_id, mode)
                routed_broker_id = self._get_broker_for_account(account_id, mode)
                routing_score = routing_decision.priority_score

                # Comment 1 fix: Enforce minimum routing compatibility score threshold
                MIN_ROUTING_SCORE = 70
                if routing_score < MIN_ROUTING_SCORE:
                    logger.warning(
                        f"Bot {bot_id} rejected by routing matrix: compatibility score {routing_score:.2f} "
                        f"below minimum threshold {MIN_ROUTING_SCORE}"
                    )
                    routing_stats["rejected"] += 1
                    continue

                logger.info(
                    f"{mode_tag} Bot {bot_id} routed to account={account_id}, broker={routed_broker_id}, "
                    f"score={routing_score:.2f}, mode={mode}"
                )
                routing_stats["routed"] += 1

            # Build trade proposal for Governor with routing metadata
            # Comment 1: Include mode in trade_proposal for demo-specific risk scaling
            # V4: Include capital_allocated from PromotionManager for position sizing
            trade_proposal = {
                'symbol': bot.get('symbols', ['EURUSD'])[0],
                'current_balance': account_balance,
                'account_balance': account_balance,
                'broker_id': routed_broker_id,
                'account_id': account_id,  # Include account_id for account-specific limits
                'stop_loss_pips': bot.get('stop_loss_pips', 20.0),
                'win_rate': bot.get('win_rate', 0.55),
                'avg_win': bot.get('avg_win', 400.0),
                'avg_loss': bot.get('avg_loss', 200.0),
                'current_atr': bot.get('current_atr', 0.0012),
                'average_atr': bot.get('average_atr', 0.0010),
                'mode': mode,  # Comment 1: Include mode for Governor risk scaling
                'bot_id': bot_id,  # Include bot_id for virtual balance lookup
                'trading_mode': bot.get('trading_mode', 'paper'),  # V4: Trading mode
                'capital_allocated': bot.get('capital_allocated', 0.0),  # V4: From PromotionManager
            }

            # Get risk mandate with position sizing from Governor
            # NOTE: _governor is guaranteed to be non-None by __init__ guard
            # Comment 1: Pass mode to Governor for demo-specific risk scaling
            mandate = self._governor.calculate_risk(
                regime_report,
                trade_proposal,
                account_balance,
                routed_broker_id,
                account_id=account_id,
                mode=mode  # Comment 1: Pass mode for demo-specific risk scaling
            )

            # Filter by position size (for Kelly sizing)
            # Comment 2 fix: Position sizing now comes solely from Kelly result
            # Both base Governor and EnhancedGovernor use EnhancedKellyCalculator
            if mandate.position_size <= 0:
                # Kelly returned 0 (negative expectancy, fee kill switch, or calculation error)
                # Skip this bot - no fallback position size
                logger.debug(f"Bot {bot_id} skipped: position_size <= 0 (Kelly blocked)")
                continue
            position_size = mandate.position_size

            dispatch = bot.copy()
            dispatch.update({
                'position_size': position_size,
                'kelly_fraction': mandate.kelly_fraction,
                'risk_amount': mandate.risk_amount,
                'kelly_adjustments': mandate.kelly_adjustments,
                'notes': mandate.notes,
                'regime': regime_report.regime,
                'risk_mode': mandate.risk_mode,
                # === ROUTING METADATA ===
                'account_id': account_id,
                'broker_id': routed_broker_id,
                'routing_score': routing_score,
                # === MODE METADATA (Comment 1) ===
                'mode': mode,  # Include mode in dispatch for downstream logging/persistence
                # === V4: TRADING MODE METADATA ===
                'trading_mode': bot.get('trading_mode', 'paper'),
                'capital_allocated': bot.get('capital_allocated', 0.0),
            })
            dispatches.append(dispatch)

        # Log routing summary for audit trail
        # Comment 1: Include mode in auction result logging
        logger.info(
            f"{mode_tag} Auction result: {len(dispatches)} bots dispatched for {regime_report.regime} "
            f"(routed={routing_stats['routed']}, rejected={routing_stats['rejected']}, mode={mode})"
        )
        return dispatches
    
    async def execute_signal(self, execution_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a TradingView webhook signal.
        
        This method handles incoming TradingView alerts by:
        1. Validating the signal parameters
        2. Getting current market regime from Sentinel
        3. Finding suitable bot via auction
        4. Executing trade via MT5 adapter
        
        Args:
            execution_params: Dict containing signal parameters
                - symbol: Trading symbol (required)
                - action: 'long', 'short', 'close', 'close_all' (required)
                - volume: Lot size (required)
                - strategy: Strategy name (optional)
                - timeframe: Timeframe (optional)
                - price: Limit price (optional)
                - stop_loss: Stop loss price (optional)
                - take_profit: Take profit price (optional)
                - comment: Trade comment (optional)
                
        Returns:
            Dict with execution result:
                - status: 'success' or 'error'
                - message: Result message
                - bot_id: Selected bot ID (if success)
                - order_id: MT5 order ID (if success)
                - execution_price: Fill price (if success)
                - execution_time: ISO timestamp (if success)
        """
        try:
            # 1. Validate Parameters
            required_fields = ['symbol', 'action', 'volume']
            for field in required_fields:
                if field not in execution_params:
                    return {
                        'status': 'error',
                        'message': f'Missing required field: {field}',
                        'bot_id': None,
                        'order_id': None
                    }
            
            symbol = execution_params['symbol']
            action = execution_params['action']
            volume = execution_params['volume']
            
            # Validate action
            valid_actions = ['long', 'short', 'close', 'close_all']
            if action not in valid_actions:
                return {
                    'status': 'error',
                    'message': f'Invalid action: {action}. Must be one of {valid_actions}',
                    'bot_id': None,
                    'order_id': None
                }
            
            # Validate volume
            if volume <= 0:
                return {
                    'status': 'error',
                    'message': 'Volume must be greater than 0',
                    'bot_id': None,
                    'order_id': None
                }
            
            logger.info(f"Executing signal: {action} {volume} {symbol}")
            
            # 2. Get Current Regime
            from src.router.sentinel import Sentinel, get_sentinel
            sentinel = get_sentinel()
            regime_report = sentinel.current_report
            
            if regime_report is None:
                # Create a default regime report if none exists
                from src.router.sentinel import RegimeReport
                regime_report = RegimeReport(
                    regime='TREND_STABLE',
                    chaos_score=0.3,
                    regime_quality=0.7,
                    susceptibility=0.3,
                    is_systemic_risk=False,
                    news_state='SAFE',
                    timestamp=datetime.now(timezone.utc).timestamp()
                )
                logger.warning("No current regime found, using default TREND_STABLE")
            
            # 3. Get account balance for position sizing
            account_balance = 100000.0  # Default balance
            try:
                from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
                # Try to get balance from broker config
                broker_id = execution_params.get('broker_id', 'mt5_default')
                # Note: In production, you'd get actual balance from MT5
            except Exception as e:
                logger.warning(f"Could not get account balance: {e}")
            
            # 4. Run auction to find suitable bot
            # Filter by symbol and strategy if provided
            strategy = execution_params.get('strategy')
            bot_dispatches = self.run_auction(
                regime_report=regime_report,
                account_balance=account_balance,
                broker_id='mt5_default',
                mode=execution_params.get('mode', 'live')
            )
            
            # Filter by symbol
            bot_dispatches = [b for b in bot_dispatches if symbol in b.get('symbols', [symbol])]
            
            # Filter by strategy if provided
            if strategy:
                bot_dispatches = [b for b in bot_dispatches if b.get('strategy') == strategy]
            
            # Select top-ranked bot
            selected_bot = None
            if bot_dispatches:
                selected_bot = bot_dispatches[0]  # Already sorted by rank in run_auction
            
            if selected_bot is None:
                return {
                    'status': 'error',
                    'message': f'No suitable bot found for symbol {symbol}' + (f' and strategy {strategy}' if strategy else ''),
                    'bot_id': None,
                    'order_id': None
                }
            
            bot_id = selected_bot.get('bot_id', 'unknown')
            logger.info(f"Selected bot {bot_id} for signal execution")
            
            # 5. Execute Trade via MT5 Adapter
            try:
                from src.data.brokers.mt5_socket_adapter import MT5SocketAdapter
                
                # Map action to MT5 direction
                direction = 'buy' if action in ['long'] else 'sell'
                
                # For close/close_all, we'd need position handling
                # For now, handle long/short as buy/sell
                if action == 'close':
                    # Close specific position - would need position ID
                    logger.warning("Close action requires position ID - treating as market close")
                    direction = 'sell'  # Simplified
                elif action == 'close_all':
                    # Close all positions - would need position listing
                    logger.warning("Close_all action requires position enumeration - not implemented")
                    return {
                        'status': 'error',
                        'message': 'close_all action not yet implemented',
                        'bot_id': bot_id,
                        'order_id': None
                    }
                
                # Create adapter instance (in production, use singleton/config)
                # For now, we'll simulate the order since we don't have live MT5 connection
                config = {
                    'vps_host': 'localhost',
                    'vps_port': 5555,
                    'account_id': 'demo',
                    'timeout': 5.0
                }
                mt5_adapter = MT5SocketAdapter(config)
                
                # Place order
                order_result = await mt5_adapter.place_order(
                    symbol=symbol,
                    volume=volume,
                    direction=direction,
                    order_type='market',
                    price=execution_params.get('price'),
                    stop_loss=execution_params.get('stop_loss'),
                    take_profit=execution_params.get('take_profit')
                )
                
                return {
                    'status': 'success',
                    'bot_id': bot_id,
                    'order_id': order_result.get('order_id'),
                    'message': f"Order placed: {order_result.get('order_id')}",
                    'execution_price': order_result.get('filled_price'),
                    'execution_time': datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                logger.error(f"MT5 order execution failed: {e}")
                return {
                    'status': 'error',
                    'message': f'Order execution failed: {str(e)}',
                    'bot_id': bot_id,
                    'order_id': None
                }
                
        except Exception as e:
            logger.error(f"Signal execution failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'bot_id': None,
                'order_id': None
            }

    
    def _get_bots_for_regime_and_session(
        self, 
        regime: str, 
        current_session: TradingSession, 
        current_utc: datetime,
        mode: Optional[str] = None
    ) -> List[dict]:
        """
        Returns bots that are authorized to trade in this physics state and session.

        V2: Only @primal tagged bots are included.
        V3: Session-aware filtering for ICT-style strategies.
        V4: Trading mode filtering - only bots with matching trading_mode participate.

        Args:
            regime: Current market regime from Sentinel
            current_session: Current trading session
            current_utc: Current UTC time
            mode: Trading mode ('demo' or 'live'). If provided, filters bots by trading_mode.

        Returns:
            List of bot dicts with manifest data
        """
        from src.router.bot_manifest import TradingMode
        
        # No trading in extreme conditions
        if regime in ["HIGH_CHAOS", "NEWS_EVENT"]:
            return []

        # Get allowed strategy types for this regime
        allowed_strategies = self.regime_strategy_map.get(regime, ["STRUCTURAL"])

        # Determine target trading mode based on mode parameter
        # - 'demo' mode -> only DEMO bots
        # - 'live' mode -> only LIVE bots
        # - None (unspecified) -> all bots (backward compatible)
        target_trading_mode = None
        if mode == "demo":
            target_trading_mode = TradingMode.DEMO
        elif mode == "live":
            target_trading_mode = TradingMode.LIVE

        # Try BotRegistry first (V2)
        if self.bot_registry:
            logger.debug("Using BotRegistry path for bot selection")
            primal_bots = self.bot_registry.list_by_tag("@primal")

            eligible = []
            for manifest in primal_bots:
                # V4: Filter by trading_mode matching the requested mode
                if target_trading_mode is not None:
                    if manifest.trading_mode != target_trading_mode:
                        logger.debug(
                            f"Bot {manifest.bot_id} filtered by trading_mode: "
                            f"{manifest.trading_mode.value} != {target_trading_mode.value}"
                        )
                        continue
                
                # Skip PAPER bots in live auctions (they haven't been promoted yet)
                if mode == "live" and manifest.trading_mode == TradingMode.PAPER:
                    logger.debug(f"Bot {manifest.bot_id} skipped: PAPER mode in live auction")
                    continue

                # Filter by strategy type compatibility with regime
                if manifest.strategy_type.value not in allowed_strategies:
                    continue

                # V3: Filter by session compatibility
                if not self._is_bot_session_compatible(manifest, current_session, current_utc):
                    logger.debug(f"Bot {manifest.bot_id} filtered by session")
                    continue

                # Comment 3 fix: Include timeframe fields from BotManifest for multi-timeframe filtering
                # V4: Include trading_mode and capital_allocated for downstream use
                eligible.append({
                    "bot_id": manifest.bot_id,
                    "name": manifest.name,
                    "strategy_type": manifest.strategy_type.value,
                    "frequency": manifest.frequency.value,
                    "score": self._calculate_bot_score(manifest),
                    "win_rate": manifest.win_rate,
                    "prop_firm_safe": manifest.prop_firm_safe,
                    "symbols": manifest.symbols,
                    "manifest": manifest,  # Include full manifest for downstream
                    # Timeframe fields for multi-timeframe filtering (Comment 3)
                    "preferred_timeframe": manifest.preferred_timeframe.name if manifest.preferred_timeframe else None,
                    "use_multi_timeframe": manifest.use_multi_timeframe,
                    "secondary_timeframes": [tf.name for tf in manifest.secondary_timeframes] if manifest.secondary_timeframes else [],
                    # V4: Trading mode and capital allocation
                    "trading_mode": manifest.trading_mode.value,
                    "capital_allocated": manifest.capital_allocated,
                    "promotion_eligible": manifest.promotion_eligible,
                })

            return eligible
        
        # Fallback to legacy dict-based storage
        # DIAGNOSTIC: Log when legacy path is used
        logger.warning("LEGACY PATH: BotRegistry unavailable - using active_bots dict storage")
        logger.warning(f"LEGACY PATH: Found {len(self.active_bots)} bots in active_bots")
        
        # DIAGNOSTIC: Log current session and UTC time
        logger.warning(f"LEGACY PATH: Current session={current_session.value}, UTC={current_utc.isoformat()}")
        
        # Apply session filtering and @primal tag check to legacy bots
        eligible = []
        for bot_id, bot_data in self.active_bots.items():
            # V4: Filter by trading_mode if specified
            if target_trading_mode is not None:
                bot_mode_str = bot_data.get('trading_mode', 'paper')
                try:
                    bot_mode = TradingMode(bot_mode_str)
                    if bot_mode != target_trading_mode:
                        logger.debug(f"LEGACY PATH: Bot {bot_id} filtered by trading_mode")
                        continue
                except ValueError:
                    # Invalid trading_mode, skip
                    continue
            
            # Skip PAPER bots in live auctions
            if mode == "live" and bot_data.get('trading_mode', 'paper') == 'paper':
                logger.debug(f"LEGACY PATH: Bot {bot_id} skipped: PAPER mode in live auction")
                continue
            
            # Filter by strategy type compatibility with regime
            if bot_data.get('strategy_type') not in allowed_strategies:
                continue
            
            # Filter by @primal tag (V2 requirement)
            tags = bot_data.get('tags', [])
            if '@primal' not in tags:
                logger.debug(f"LEGACY PATH: Bot {bot_id} filtered - missing @primal tag")
                continue
            
            # V3: Filter by session compatibility
            if not self._is_bot_dict_session_compatible(bot_data, current_session, current_utc):
                logger.debug(f"LEGACY PATH: Bot {bot_id} filtered by session")
                continue
            
            eligible.append(bot_data)
        
        logger.warning(f"LEGACY PATH: Returning {len(eligible)} eligible bots after filtering")
        return eligible
    
    def _calculate_bot_score(self, manifest: "BotManifest") -> float:
        """
        Calculate bot's auction score based on manifest data.
        
        Score = win_rate * (total_trades / 100) * recency_bonus
        """
        base_score = manifest.win_rate
        
        # Experience bonus (more trades = more reliable stats)
        experience_factor = min(1.0, manifest.total_trades / 100)
        
        # Recency bonus (traded recently = warmed up)
        recency_bonus = 1.0
        if manifest.last_trade_at:
            from datetime import datetime
            hours_since = (datetime.now() - manifest.last_trade_at).total_seconds() / 3600
            if hours_since < 24:
                recency_bonus = 1.2
            elif hours_since > 168:  # 1 week
                recency_bonus = 0.8
        
        return base_score * experience_factor * recency_bonus

    def _is_bot_session_compatible(self, manifest: "BotManifest", current_session: TradingSession, current_utc: datetime) -> bool:
        """
        Check if bot is compatible with current session and time windows.

        V3: Session-aware filtering for ICT-style strategies.

        Args:
            manifest: Bot manifest with optional preferred_conditions
            current_session: Current trading session
            current_utc: Current UTC time

        Returns:
            True if bot can trade in current conditions
        """
        # If no preferred conditions, bot can trade anytime
        if manifest.preferred_conditions is None:
            return True

        prefs = manifest.preferred_conditions

        # Check session preference
        if prefs.sessions:
            # Check if current session matches any preferred session
            session_match = current_session.value in prefs.sessions
            if not session_match:
                # Special case: OVERLAP counts as both LONDON and NEW_YORK
                if current_session == TradingSession.OVERLAP:
                    if not (TradingSession.LONDON.value in prefs.sessions or
                           TradingSession.NEW_YORK.value in prefs.sessions):
                        return False
                else:
                    return False

        # Check time window preference
        if prefs.time_windows:
            in_any_window = False
            for window in prefs.time_windows:
                if SessionDetector.is_in_time_window(current_utc, window.start, window.end, window.timezone):
                    in_any_window = True
                    break
            if not in_any_window:
                return False

        # All checks passed
        return True

    def _is_bot_dict_session_compatible(self, bot_data: dict, current_session: TradingSession, current_utc: datetime) -> bool:
        """
        Check if bot (as dict) is compatible with current session and time windows.

        Legacy version for dict-based bot storage when BotRegistry is unavailable.
        Applies the same session filtering logic as _is_bot_session_compatible.

        Args:
            bot_data: Bot dictionary with optional preferred_conditions
            current_session: Current trading session
            current_utc: Current UTC time

        Returns:
            True if bot can trade in current conditions
        """
        # If no preferred conditions, bot can trade anytime
        prefs = bot_data.get('preferred_conditions')
        if prefs is None:
            return True

        # Handle both dict and object formats for preferred_conditions
        if isinstance(prefs, dict):
            sessions = prefs.get('sessions', [])
            time_windows_data = prefs.get('time_windows', [])
        else:
            # If it's an object, try to get sessions and time_windows
            sessions = getattr(prefs, 'sessions', [])
            time_windows_data = getattr(prefs, 'time_windows', [])

        # Check session preference
        if sessions:
            # Check if current session matches any preferred session
            session_match = current_session.value in sessions
            if not session_match:
                # Special case: OVERLAP counts as both LONDON and NEW_YORK
                if current_session == TradingSession.OVERLAP:
                    if not (TradingSession.LONDON.value in sessions or
                           TradingSession.NEW_YORK.value in sessions):
                        return False
                else:
                    return False

        # Check time window preference
        if time_windows_data:
            in_any_window = False
            for window_data in time_windows_data:
                # Handle both dict and object formats for time windows
                if isinstance(window_data, dict):
                    start = window_data.get('start', '00:00')
                    end = window_data.get('end', '23:59')
                    timezone = window_data.get('timezone', 'UTC')
                else:
                    start = getattr(window_data, 'start', '00:00')
                    end = getattr(window_data, 'end', '23:59')
                    timezone = getattr(window_data, 'timezone', 'UTC')

                if SessionDetector.is_in_time_window(current_utc, start, end, timezone):
                    in_any_window = True
                    break
            if not in_any_window:
                return False

        # All checks passed
        return True

    def _count_active_positions(self) -> int:
        """Count currently active positions across all bots."""
        # TODO: Get from socket_server or position tracker
        return 0

    def _build_manifest_from_bot_data(self, bot_id: str, bot_data: dict) -> Optional["BotManifest"]:
        """
        Build a BotManifest from legacy bot_data dict for routing purposes.

        Comment 2 fix: This allows legacy bots (without pre-attached manifests) to go through
        the RoutingMatrix instead of being forced to demo_account fallback.

        Args:
            bot_id: Bot identifier
            bot_data: Legacy bot dictionary data

        Returns:
            BotManifest if construction succeeds, None otherwise
        """
        try:
            from src.router.bot_manifest import BotManifest, StrategyType, BotFrequency

            # Map strategy_type string to enum
            strategy_str = bot_data.get('strategy_type', 'STRUCTURAL')
            try:
                strategy_type = StrategyType[strategy_str]
            except KeyError:
                strategy_type = StrategyType.STRUCTURAL

            # Map frequency string to enum
            freq_str = bot_data.get('frequency', 'MEDIUM')
            try:
                frequency = BotFrequency[freq_str]
            except KeyError:
                frequency = BotFrequency.MEDIUM

            # Build minimal manifest for routing
            manifest_data = {
                'bot_id': bot_id,
                'name': bot_data.get('name', bot_id),
                'strategy_type': strategy_type.value,
                'frequency': frequency.value,
                'symbols': bot_data.get('symbols', ['EURUSD']),
                'win_rate': bot_data.get('win_rate', 0.5),
                'total_trades': bot_data.get('total_trades', 0),
                'prop_firm_safe': bot_data.get('prop_firm_safe', False),
                'tags': bot_data.get('tags', []),
            }

            manifest = BotManifest.from_dict(manifest_data)
            logger.debug(f"Built manifest for legacy bot {bot_id} for routing")
            return manifest

        except Exception as e:
            logger.warning(f"Failed to build manifest for legacy bot {bot_id}: {e}")
            return None
    
    def dispatch(self, bot_id: str, instruction: str, mode: str = "live") -> bool:
        """
        Sends JSON command to the Interface via socket.
        
        Comment 1: Include mode in dispatch payload for downstream logging/persistence.
        V4: Include trading_mode from manifest for proper execution context.
        
        Args:
            bot_id: Target bot ID
            instruction: Command to send (e.g., "OPEN_BUY", "CLOSE_ALL")
            mode: Trading mode ('demo' or 'live') - affects which MT5 connection is used
            
        Returns:
            True if dispatch successful
        """
        try:
            # Get manifest to include trading_mode and capital_allocated
            trading_mode = "paper"
            capital_allocated = 0.0
            if self.bot_registry:
                manifest = self.bot_registry.get(bot_id)
                if manifest:
                    trading_mode = manifest.trading_mode.value
                    capital_allocated = manifest.capital_allocated
            
            # TODO: Integrate with socket_server
            # Comment 1: Include mode in dispatch command for downstream processing
            # V4: Include trading_mode and capital_allocated from manifest
            command = {
                "type": "DISPATCH",
                "bot_id": bot_id,
                "instruction": instruction,
                "mode": mode,  # Comment 1: Include mode for MT5 connection selection
                "trading_mode": trading_mode,  # V4: From manifest
                "capital_allocated": capital_allocated,  # V4: From PromotionManager
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            mode_tag = "[DEMO]" if mode == "demo" else "[LIVE]"
            logger.info(f"{mode_tag} Dispatching to {bot_id}: {instruction} (mode={mode}, trading_mode={trading_mode})")
            # socket_server.send(command)
            return True
        except Exception as e:
            logger.error(f"Dispatch failed for {bot_id}: {e}")
            return False
    
    def register_bot(self, bot_id: str, manifest_data: dict) -> bool:
        """
        Register a bot with the Commander.
        
        V2: Registers in both legacy dict and BotRegistry.
        
        Args:
            bot_id: Bot identifier
            manifest_data: Bot manifest as dict
            
        Returns:
            True if registration successful
        """
        self.active_bots[bot_id] = manifest_data
        
        if self.bot_registry:
            try:
                from src.router.bot_manifest import BotManifest
                manifest = BotManifest.from_dict(manifest_data)
                self.bot_registry.register(manifest)
                logger.info(f"Registered bot {bot_id} in BotRegistry")
            except Exception as e:
                logger.warning(f"Could not register in BotRegistry: {e}")
        
        return True
    
    def get_auction_status(self) -> dict:
        """Get current auction/commander status for monitoring."""
        primal_count = 0
        pending_count = 0
        
        if self.bot_registry:
            primal_count = len(self.bot_registry.list_by_tag("@primal"))
            pending_count = len(self.bot_registry.list_by_tag("@pending"))
        
        limiter = DynamicBotLimiter()
        default_balance = getattr(self._governor, '_daily_start_balance', 100000.0)
        max_bots = limiter.get_max_bots(default_balance)
        return {
            "active_bots": len(self.active_bots),
            "primal_bots": primal_count,
            "pending_bots": pending_count,
            "max_bots": max_bots,
            "regime_strategy_map": self.regime_strategy_map
        }
    
    # ========== Multi-Timeframe Auction (Comment 1) ==========
    
    def run_auction_with_timeframes(
        self,
        primary_regime_report: "RegimeReport",
        all_timeframe_regimes: Dict["Timeframe", "RegimeReport"],
        primary_timeframe: Optional["Timeframe"] = None,
        account_balance: Optional[float] = None,
        broker_id: str = "mt5_default",
        current_utc: Optional[datetime] = None,
        mode: Optional[str] = None
    ) -> List[dict]:
        """
        Multi-timeframe-aware bot auction.
        
        Comment 1: Extend run_auction to select bots by preferred_timeframe
        and enforce secondary_timeframes alignment when use_multi_timeframe is true.
        
        V3: Mode-aware routing for demo/live trading.
        
        This method:
        1. Gets eligible bots from the registry
        2. Filters bots by their preferred_timeframe vs available timeframe regimes
        3. For bots with use_multi_timeframe=True, enforces secondary_timeframes alignment
        4. Returns dispatches with timeframe context and mode
        
        Args:
            primary_regime_report: Primary regime report (fastest timeframe)
            all_timeframe_regimes: Dict of timeframe -> regime report for all timeframes
            primary_timeframe: The actual primary timeframe being used (e.g., fastest).
                             If None, derived from all_timeframe_regimes keys.
            account_balance: Account balance for position sizing
            broker_id: Broker identifier for fee-aware Kelly
            current_utc: UTC timestamp for session-aware filtering
            mode: Trading mode ('demo' or 'live'). If None, derives from EA config.
            
        Returns:
            List of bot dispatches with timeframe context and mode
        """
        # Default to current UTC time if not provided
        if current_utc is None:
            current_utc = datetime.now(timezone.utc)
        elif current_utc.tzinfo is None:
            current_utc = current_utc.replace(tzinfo=timezone.utc)
        elif current_utc.tzinfo != timezone.utc:
            current_utc = current_utc.astimezone(timezone.utc)

        # Comment 1 fix: Derive primary_timeframe if not provided
        # Use the fastest timeframe (smallest interval) as primary if not specified
        if primary_timeframe is None and all_timeframe_regimes:
            sorted_timeframes = sorted(all_timeframe_regimes.keys(), key=lambda tf: tf.seconds)
            primary_timeframe = sorted_timeframes[0]
        
        # Derive mode if not provided (Comment 1: mode-aware routing)
        if mode is None:
            mode = self._derive_mode_from_context(primary_regime_report)
        
        mode_tag = "[DEMO]" if mode == "demo" else "[LIVE]"
        
        # Detect current session
        current_session = SessionDetector.detect_session(current_utc)
        logger.info(
            f"{mode_tag} Running multi-timeframe auction for session: {current_session.value}, "
            f"UTC time: {current_utc.isoformat()}, timeframes: {[tf.name for tf in all_timeframe_regimes.keys()]}, "
            f"primary_timeframe: {primary_timeframe.name if primary_timeframe else 'None'}, mode={mode}"
        )

        # Get all eligible bots
        # V4: Pass mode to filter by trading_mode
        eligible_bots = self._get_bots_for_regime_and_session(
            primary_regime_report.regime, 
            current_session, 
            current_utc,
            mode=mode
        )

        if not eligible_bots:
            logger.debug(f"No eligible bots for primary regime: {primary_regime_report.regime}")
            return []

        # Filter bots based on multi-timeframe preferences
        filtered_bots = self._filter_bots_by_timeframe(
            eligible_bots,
            all_timeframe_regimes
        )

        if not filtered_bots:
            logger.debug("No bots passed multi-timeframe filtering")
            return []

        # Apply chaos-based filtering
        if primary_regime_report.chaos_score > 0.6:
            logger.info("High chaos detected - reducing bot selection")
            filtered_bots = [b for b in filtered_bots
                           if b.get('frequency') in ['LOW', 'MEDIUM']]

        # Rank by performance
        ranked_bots = sorted(
            filtered_bots,
            key=lambda x: x.get('score', 0) * x.get('win_rate', 0.5),
            reverse=True
        )

        # Default account_balance if falsy (prevents crashes in legacy usage - Comment 2)
        account_balance = account_balance or getattr(self._governor, '_daily_start_balance', 100000.0)
        logger.debug(f"Commander: Defaulted account_balance to {account_balance} (multi-timeframe)")

        # Select top N using DynamicBotLimiter API
        # Comment 2 fix: Use can_add_bot() API instead of hard cap
        limiter = DynamicBotLimiter()
        max_bots = limiter.get_max_bots(account_balance)
        active_positions = self._count_active_positions()
        
        # Use can_add_bot() for each candidate bot instead of hard cap
        # This respects tier-based limits and safety buffer
        selected_bots = []
        current_bot_count = active_positions
        
        for bot in ranked_bots:
            can_add, reason = limiter.can_add_bot(account_balance, current_bot_count)
            if can_add:
                selected_bots.append(bot)
                current_bot_count += 1
                logger.debug(f"Bot {bot.get('bot_id')} added: {reason}")
            else:
                logger.debug(f"Bot {bot.get('bot_id')} rejected: {reason}")
        
        top_bots = selected_bots

        # Calculate position sizes and build dispatches
        # Track routing statistics for audit
        routing_stats = {"total": len(top_bots), "routed": 0, "rejected": 0}
        dispatches = []

        for bot in top_bots:
            bot_id = bot.get('bot_id', 'unknown')

            # === ROUTING MATRIX INTEGRATION ===
            # Route bot to appropriate account based on manifest
            manifest = bot.get('manifest')
            if manifest is None:
                # Comment 2 fix: Try to build a BotManifest from bot_data for legacy bots
                # instead of immediately defaulting to demo_account
                manifest = self._build_manifest_from_bot_data(bot_id, bot)
                if manifest is not None:
                    # Route through RoutingMatrix with constructed manifest
                    routing_decision = self._routing_matrix.route_bot(manifest)
                    if routing_decision.is_approved:
                        # Comment 1 fix: Enforce minimum routing compatibility score threshold
                        MIN_ROUTING_SCORE = 70
                        if routing_decision.priority_score < MIN_ROUTING_SCORE:
                            logger.warning(
                                f"Bot {bot_id} rejected by routing matrix: compatibility score "
                                f"{routing_decision.priority_score:.2f} below minimum threshold {MIN_ROUTING_SCORE}"
                            )
                            routing_stats["rejected"] += 1
                            continue
                        base_account_id = routing_decision.assigned_account
                        # Comment 1: Apply mode-aware account selection
                        account_id = self._get_account_for_mode(base_account_id, mode)
                        routed_broker_id = self._get_broker_for_account(account_id, mode)
                        routing_score = routing_decision.priority_score
                        logger.info(
                            f"{mode_tag} Bot {bot_id} (legacy) routed to account={account_id}, broker={routed_broker_id}, "
                            f"score={routing_score:.2f}, mode={mode}"
                        )
                        routing_stats["routed"] += 1
                    else:
                        logger.warning(
                            f"Bot {bot_id} rejected by routing matrix: {routing_decision.rejection_reason}"
                        )
                        routing_stats["rejected"] += 1
                        continue
                else:
                    # Only fall back to demo_account when routing genuinely cannot be performed
                    logger.warning(f"Bot {bot_id} has no manifest and cannot build one - falling back to demo_account")
                    account_id = "demo_account"
                    routed_broker_id = broker_id  # Use passed broker_id as fallback
                    routing_score = 0.0
            else:
                # Get routing decision from RoutingMatrix
                routing_decision = self._routing_matrix.route_bot(manifest)

                if not routing_decision.is_approved:
                    # Bot rejected by routing matrix - skip this bot
                    logger.warning(
                        f"Bot {bot_id} rejected by routing matrix: {routing_decision.rejection_reason}"
                    )
                    routing_stats["rejected"] += 1
                    continue

                # Get account assignment and map to broker
                account_id = routing_decision.assigned_account
                routed_broker_id = self._get_broker_for_account(account_id)
                routing_score = routing_decision.priority_score

                logger.info(
                    f"Bot {bot_id} routed to account={account_id}, broker={routed_broker_id}, "
                    f"score={routing_score:.2f}"
                )
                routing_stats["routed"] += 1

            # Build trade proposal with routing metadata
            # Comment 1: Include mode in trade_proposal for demo-specific risk scaling
            # V4: Include capital_allocated from PromotionManager for position sizing
            trade_proposal = {
                'symbol': bot.get('symbols', ['EURUSD'])[0],
                'current_balance': account_balance,
                'account_balance': account_balance,
                'broker_id': routed_broker_id,
                'account_id': account_id,  # Include account_id for account-specific limits
                'stop_loss_pips': bot.get('stop_loss_pips', 20.0),
                'win_rate': bot.get('win_rate', 0.55),
                'avg_win': bot.get('avg_win', 400.0),
                'avg_loss': bot.get('avg_loss', 200.0),
                'current_atr': bot.get('current_atr', 0.0012),
                'average_atr': bot.get('average_atr', 0.0010),
                'mode': mode,  # Comment 1: Include mode for Governor risk scaling
                'bot_id': bot_id,  # Include bot_id for virtual balance lookup
                'trading_mode': bot.get('trading_mode', 'paper'),  # V4: Trading mode
                'capital_allocated': bot.get('capital_allocated', 0.0),  # V4: From PromotionManager
            }

            mandate = self._governor.calculate_risk(
                primary_regime_report,
                trade_proposal,
                account_balance,
                routed_broker_id,
                account_id=account_id,
                mode=mode  # Comment 1: Pass mode for demo-specific risk scaling
            )

            # Comment 2 fix: Position sizing now comes solely from Kelly result
            # Both base Governor and EnhancedGovernor use EnhancedKellyCalculator
            if mandate.position_size <= 0:
                # Kelly returned 0 (negative expectancy, fee kill switch, or calculation error)
                # Skip this bot - no fallback position size
                logger.debug(f"Bot {bot_id} skipped: position_size <= 0 (Kelly blocked)")
                continue
            position_size = mandate.position_size

            dispatch = bot.copy()
            dispatch.update({
                'position_size': position_size,
                'kelly_fraction': mandate.kelly_fraction,
                'risk_amount': mandate.risk_amount,
                'kelly_adjustments': mandate.kelly_adjustments,
                'notes': mandate.notes,
                'regime': primary_regime_report.regime,
                'risk_mode': mandate.risk_mode,
                # === ROUTING METADATA ===
                'account_id': account_id,
                'broker_id': routed_broker_id,
                'routing_score': routing_score,
                # === MODE METADATA (Comment 1) ===
                'mode': mode,  # Include mode in dispatch for downstream logging/persistence
                # === V4: TRADING MODE METADATA ===
                'trading_mode': bot.get('trading_mode', 'paper'),
                'capital_allocated': bot.get('capital_allocated', 0.0),
            })
            
            # Comment 1 fix: Use actual primary_timeframe instead of hardcoded 'M5'
            # Comment 3 fix: Add top-level timeframe and multi_timeframe_aligned fields
            bot_preferred_tf = bot.get('preferred_timeframe')
            actual_timeframe = bot_preferred_tf if bot_preferred_tf else (primary_timeframe.name if primary_timeframe else 'H1')
            
            # Determine multi_timeframe_alignment for multi-timeframe bots
            use_mtf = bot.get('use_multi_timeframe', False)
            multi_timeframe_aligned = False
            if use_mtf and bot.get('secondary_timeframes'):
                secondary_timeframes = bot.get('secondary_timeframes') or []
                multi_timeframe_aligned = self._check_secondary_timeframes_alignment(
                    secondary_timeframes,
                    all_timeframe_regimes
                )
            
            # Add top-level timeframe field (Comment 3)
            dispatch['timeframe'] = actual_timeframe
            
            # Add multi_timeframe_aligned field (Comment 3)
            dispatch['multi_timeframe_aligned'] = multi_timeframe_aligned
            
            # Add timeframe context to dispatch (Comment 1: ensure dispatch payloads include timeframe context)
            dispatch['timeframe_context'] = {
                'primary_timeframe': primary_timeframe.name if primary_timeframe else 'H1',
                'preferred_timeframe': bot_preferred_tf if bot_preferred_tf else 'H1',
                'use_multi_timeframe': use_mtf,
                'secondary_timeframes': bot.get('secondary_timeframes', []),
                'all_regimes': {tf.name: r.regime for tf, r in all_timeframe_regimes.items()}
            }
            
            dispatches.append(dispatch)

        # Log routing summary for audit trail
        # Comment 1: Include mode in multi-timeframe auction result logging
        logger.info(
            f"{mode_tag} Multi-timeframe auction result: {len(dispatches)} bots dispatched "
            f"(routed={routing_stats['routed']}, rejected={routing_stats['rejected']}, mode={mode})"
        )
        return dispatches
    
    def _filter_bots_by_timeframe(
        self,
        bots: List[dict],
        all_timeframe_regimes: Dict["Timeframe", "RegimeReport"]
    ) -> List[dict]:
        """
        Filter bots based on their preferred_timeframe and secondary_timeframes alignment.
        
        Comment 2: When a bot has preferred_timeframe set but that timeframe is absent
        from all_timeframe_regimes, skip the bot (or explicitly wait) instead of accepting it.
        Only include bots when their preferred timeframe has a regime report and passes
        chaos/extreme checks.
        
        Args:
            bots: List of eligible bot dicts
            all_timeframe_regimes: Dict of timeframe -> regime report
            
        Returns:
            Filtered list of bots
        """
        filtered = []
        
        for bot in bots:
            # Get bot's timeframe preferences from manifest
            preferred_tf = bot.get('preferred_timeframe')
            use_mtf = bot.get('use_multi_timeframe', False)
            secondary_tfs = bot.get('secondary_timeframes', [])
            
            # If no preferences, include the bot
            if preferred_tf is None:
                filtered.append(bot)
                continue
            
            # Convert preferred_timeframe to Timeframe enum if string
            if isinstance(preferred_tf, str):
                try:
                    from src.router.multi_timeframe_sentinel import Timeframe
                    preferred_tf = Timeframe[preferred_tf]
                except KeyError:
                    logger.warning(f"Unknown timeframe: {preferred_tf}, skipping bot")
                    continue
            
            # Comment 2 fix: Skip bot if preferred_timeframe is NOT in all_timeframe_regimes
            # This ensures bots only dispatch when their preferred timeframe has a regime report
            if preferred_tf not in all_timeframe_regimes:
                logger.debug(
                    f"Bot {bot.get('bot_id')} skipped: preferred timeframe {preferred_tf.name} "
                    f"not in available timeframes {list(all_timeframe_regimes.keys())}"
                )
                continue
            
            # Check if preferred timeframe's regime is suitable
            if preferred_tf in all_timeframe_regimes:
                tf_regime = all_timeframe_regimes[preferred_tf].regime
                
                # Check if regime is not extreme (HIGH_CHAOS, NEWS_EVENT)
                if tf_regime in ["HIGH_CHAOS", "NEWS_EVENT"]:
                    logger.debug(f"Bot {bot.get('bot_id')} filtered: preferred timeframe {preferred_tf.name} in {tf_regime}")
                    continue
            
            # For multi-timeframe bots, check secondary_timeframes alignment
            if use_mtf and secondary_tfs:
                alignment_ok = self._check_secondary_timeframes_alignment(
                    secondary_tfs, 
                    all_timeframe_regimes
                )
                if not alignment_ok:
                    logger.debug(f"Bot {bot.get('bot_id')} filtered: secondary timeframes not aligned")
                    continue
            
            filtered.append(bot)
        
        return filtered
    
    def _check_secondary_timeframes_alignment(
        self,
        secondary_timeframes: List,
        all_timeframe_regimes: Dict["Timeframe", "RegimeReport"]
    ) -> bool:
        """
        Check if secondary timeframes are aligned for multi-timeframe strategy.
        
        A bot with use_multi_timeframe=True requires secondary timeframes to be in
        compatible regimes (not extreme) before trading.
        
        Args:
            secondary_timeframes: List of secondary timeframes
            all_timeframe_regimes: Dict of timeframe -> regime report
            
        Returns:
            True if all secondary timeframes are aligned (not in extreme regimes)
        """
        for tf in secondary_timeframes:
            # Convert to Timeframe enum if string
            if isinstance(tf, str):
                try:
                    from src.router.multi_timeframe_sentinel import Timeframe
                    tf = Timeframe[tf]
                except KeyError:
                    continue
            
            if tf in all_timeframe_regimes:
                regime = all_timeframe_regimes[tf].regime
                # Filter out if any secondary timeframe is in extreme regime
                if regime in ["HIGH_CHAOS", "NEWS_EVENT"]:
                    return False
        
        return True
