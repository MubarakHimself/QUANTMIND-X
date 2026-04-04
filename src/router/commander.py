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
import src.router.sessions as session_module
SESSION_BOT_MIX = session_module.SESSION_BOT_MIX
from src.router.bot_manifest import BotManifest, PoolState
from enum import Enum
from src.router.sessions import TILT_TRANSITION_MINUTES


class CloseReason(Enum):
    """
    Reason codes for forced trade closes.

    Used to annotate why a position was closed involuntarily,
    enabling post-trade analysis of close patterns.
    """
    SESSION_END_CLOSE = "SESSION_END_CLOSE"       # Forced close at session boundary
    TILT_PROFIT_TAKE = "TILT_PROFIT_TAKE"         # Tilt: in profit, RVOL < 0.8, take profit
    TILT_LOSS_ACCEPT = "TILT_LOSS_ACCEPT"         # Tilt: at loss, close anyway
    TIME_EXIT_BREAKEVEN = "TIME_EXIT_BREAKEVEN"   # Time exit when SL hit breakeven
    TIME_EXIT_LIMIT = "TIME_EXIT_LIMIT"            # Trade stuck beyond max_hold_minutes
    RVOL_MOMENTUM_DEATH = "RVOL_MOMENTUM_DEATH"   # RVOL >= 1.4 triggered TP or momentum death
    NEWS_BLACKOUT_CLOSE = "NEWS_BLACKOUT_CLOSE"   # High-impact news event blackout
    CIRCUIT_BREAKER_CLOSE = "CIRCUIT_BREAKER_CLOSE"  # Circuit breaker triggered

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

        # Router mode selection: auction (default), priority, round_robin
        self.router_mode: str = "auction"
        self._last_selected_index: int = -1  # For round_robin mode

        # Current canonical window for session-based bot type filtering (set by Tilt ACTIVATE callback)
        self._current_canonical_window: Optional[str] = None
        
        # Regime → Strategy Type mapping
        self.regime_strategy_map = {
            "TREND_STABLE": ["SCALPER", "STRUCTURAL", "HFT"],
            "RANGE_STABLE": ["SCALPER", "STRUCTURAL"],
            "BREAKOUT_PRIME": ["ORB"],
            "HIGH_CHAOS": [],  # No bots authorized
            "NEWS_EVENT": [],  # Kill zone - no new trades
            "UNCERTAIN": ["STRUCTURAL"],  # Conservative only
        }

        # Regime → Pool State mapping (Story 8.10: Regime-Conditional Strategy Pool Framework)
        # Pool states: ACTIVE, MUTED, CONDITIONAL (requires volume confirmation)
        self.regime_pool_map = {
            "TREND_STABLE": {
                "scalping_long": PoolState.ACTIVE,
                "scalping_short": PoolState.MUTED,
                "orb_long": PoolState.CONDITIONAL,  # volume confirmation required
                "orb_short": PoolState.MUTED,
                "scalping_neutral": PoolState.MUTED,
            },
            "RANGE_STABLE": {
                "scalping_long": PoolState.MUTED,
                "scalping_short": PoolState.MUTED,
                "scalping_neutral": PoolState.ACTIVE,
                "orb_long": PoolState.MUTED,
                "orb_short": PoolState.MUTED,
            },
            "BREAKOUT_PRIME": {
                "orb_long": PoolState.ACTIVE,
                "orb_short": PoolState.ACTIVE,
                "orb_false_breakout": PoolState.ACTIVE,
                "scalping_long": PoolState.MUTED,  # per ORB session config
                "scalping_short": PoolState.MUTED,
                "scalping_neutral": PoolState.MUTED,
            },
            "HIGH_CHAOS": {
                "scalping_long": PoolState.MUTED,
                "scalping_short": PoolState.MUTED,
                "scalping_neutral": PoolState.MUTED,
                "orb_long": PoolState.MUTED,
                "orb_short": PoolState.MUTED,
                "orb_false_breakout": PoolState.MUTED,
                # Layer 3 CHAOS response triggered for open positions
            },
            "NEWS_EVENT": {
                # All pools muted — no new trades during kill zone
                # Existing behavior unchanged
                "scalping_long": PoolState.MUTED,
                "scalping_short": PoolState.MUTED,
                "scalping_neutral": PoolState.MUTED,
                "orb_long": PoolState.MUTED,
                "orb_short": PoolState.MUTED,
                "orb_false_breakout": PoolState.MUTED,
            },
            "UNCERTAIN": {
                # Conservative only — scalping pools muted, structural may trade
                "scalping_long": PoolState.MUTED,
                "scalping_short": PoolState.MUTED,
                "scalping_neutral": PoolState.MUTED,
                "orb_long": PoolState.MUTED,
                "orb_short": PoolState.MUTED,
                "orb_false_breakout": PoolState.MUTED,
            },
        }
        
        # Multi-timeframe sentinel (Comment 1: shared from StrategyRouter)
        self._multi_timeframe_sentinel: Optional["MultiTimeframeSentinel"] = None
        
        # Lifecycle Manager scheduler (runs daily at 3:00 AM UTC)
        self._scheduler: Optional[Any] = None

    def set_router_mode(self, mode: str) -> None:
        """Set the router mode for bot selection.

        Args:
            mode: One of 'auction', 'priority', 'round_robin'
                - auction: Sort by score (highest first) - default behavior
                - priority: Sort by priority value (highest first)
                - round_robin: Rotate through bots equally

        Raises:
            ValueError: If mode is not valid
        """
        valid_modes = ["auction", "priority", "round_robin"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid router mode: {mode}. Must be one of {valid_modes}")
        self.router_mode = mode
        logger.info(f"Commander: Router mode set to '{mode}'")

    def select_bots_for_auction(self, bots: List["BotManifest"]) -> List["BotManifest"]:
        """Select bots for auction based on the current router mode.

        Args:
            bots: List of BotManifest objects to select from

        Returns:
            List of BotManifest objects sorted/selected based on router mode
        """
        if not bots:
            return []

        if self.router_mode == "auction":
            # Current behavior: sort by score descending
            return sorted(bots, key=lambda b: b.score, reverse=True)

        elif self.router_mode == "priority":
            # Sort by priority value (highest first), ignore score
            return sorted(bots, key=lambda b: b.priority, reverse=True)

        elif self.router_mode == "round_robin":
            # Equal opportunity: rotate through bots
            next_index = (self._last_selected_index + 1) % len(bots)
            self._last_selected_index = next_index
            return [bots[next_index]]

        # Fallback: return as-is
        return bots

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

    def evaluate_sqs_gate(
        self,
        symbol: str,
        strategy_type: str = "scalping"
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Evaluate SQS gate for a symbol and strategy type.

        SQS = historical_avg_spread / current_spread
        Thresholds: scalping >0.75, ORB >0.80, hard block <0.50

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            strategy_type: "scalping" or "ORB"

        Returns:
            Tuple of (allowed: bool, result_data: dict)
                - allowed: True if trade is allowed
                - result_data: Contains sqs, threshold, reason, is_hard_block
        """
        try:
            from src.risk.sqs_engine import create_sqs_engine
            from src.risk.sqs_cache import create_sqs_cache
            from src.risk.sqs_calendar import create_calendar_integration
            from datetime import datetime as dt

            # Create SQS engine lazily
            if not hasattr(self, '_sqs_engine'):
                self._sqs_engine = create_sqs_engine()

            # Get current spread (would come from MT5 ZMQ feed in production)
            current_spread = self._get_current_spread(symbol)

            # Get historical buckets (would come from Redis in production)
            buckets = self._get_historical_buckets(symbol)

            # Get news override
            news_override = None
            if hasattr(self._sqs_engine, '_calendar') and self._sqs_engine._calendar:
                news_override = self._sqs_engine._calendar.get_threshold_override(symbol)

            # Evaluate SQS
            result = self._sqs_engine.evaluate(
                symbol=symbol,
                strategy_type=strategy_type,
                current_spread=current_spread,
                historical_buckets=buckets,
                news_override=news_override
            )

            # Log evaluation
            self._sqs_engine.log_evaluation(
                symbol=symbol,
                strategy_type=strategy_type,
                result=result,
                current_spread=current_spread,
                historical_avg=buckets.get(self._sqs_engine._get_bucket_key(dt.now()), None) if buckets else 0
            )

            result_data = {
                'sqs': result.sqs,
                'threshold': result.threshold,
                'reason': result.reason,
                'is_hard_block': result.is_hard_block,
                'current_spread': current_spread
            }

            return result.allowed, result_data

        except Exception as e:
            logger.warning(f"SQS gate evaluation failed for {symbol}: {e}")
            # NFR-R1: Graceful degradation - allow trade if SQS unavailable
            return True, {'sqs': 1.0, 'threshold': 0.75, 'reason': f'SQS unavailable - graceful degradation: {e}', 'is_hard_block': False}

    def _get_current_spread(self, symbol: str) -> float:
        """Get current spread from MT5 or demo data."""
        try:
            from src.risk.integrations.mt5_client import get_mt5_client
            client = get_mt5_client()
            if client and hasattr(client, 'is_connected') and client.is_connected():
                tick = client.get_tick(symbol)
                if tick and hasattr(tick, 'spread'):
                    return tick.spread
        except Exception:
            pass

        # Demo fallback
        demo_spreads = {
            "EURUSD": 0.9,
            "GBPUSD": 1.2,
            "USDJPY": 1.1,
            "GBPJPY": 1.8,
        }
        return demo_spreads.get(symbol, 1.0)

    def _get_historical_buckets(self, symbol: str) -> Dict[str, Any]:
        """Get historical spread buckets from cache or demo."""
        try:
            from src.risk.sqs_cache import create_sqs_cache
            from src.risk.sqs_engine import SpreadBucket
            from datetime import datetime as dt

            cache = create_sqs_cache()
            if cache.is_available:
                # Would fetch from Redis here
                pass
        except Exception:
            pass

        # Demo buckets
        from src.risk.sqs_engine import SpreadBucket
        now = dt.now()
        buckets = {}
        base_spreads = {
            "EURUSD": 0.8, "GBPUSD": 1.1, "USDJPY": 1.0, "GBPJPY": 1.6
        }
        base = base_spreads.get(symbol, 1.0)

        for i in range(-6, 7):
            test_time = now.replace(minute=(now.minute // 5) * 5) + __import__('datetime').timedelta(minutes=i * 5)
            dow = test_time.weekday()
            hour = test_time.hour
            minute_bucket = test_time.minute // 5
            key = f"{dow}:{hour}:{minute_bucket}"
            buckets[key] = SpreadBucket(
                avg_spread=base + (hash(f"{symbol}{key}") % 100) / 1000.0,
                sample_count=50,
                updated_at_utc=now
            )
        return buckets

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
            regime_report, current_session, current_utc, mode=mode
        )

        # Task 5: Apply session-based bot type filtering (SESSION_BOT_MIX)
        # This filters bots by type (ORB/MOM/MR/TC) based on the canonical window
        if self._current_canonical_window:
            eligible_bots = self._filter_bots_by_session_bot_type(
                eligible_bots, self._current_canonical_window
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
        4. Checking drawdown limits (blocks trade if exceeded)
        5. Executing trade via MT5 adapter

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
                - account_book: 'personal' or 'prop_firm' (optional, default: 'personal')
                - prop_firm_name: Name of prop firm if account_book is 'prop_firm' (optional)

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

            # =================================================================
            # FIX-015: Tilt Window Entry Blocking
            # Block new entries during T-5 tilt window before session end
            # =================================================================
            if action in ['long', 'short']:  # Only block new entries, not close actions
                utc_now = datetime.now(timezone.utc)
                if SessionDetector.is_tilt_window(utc_now):
                    logger.warning(
                        f"TILT WINDOW BLOCK: Signal rejected for {symbol} at "
                        f"{utc_now.isoformat()} - tilt window active, no new entries allowed"
                    )
                    return {
                        'status': 'error',
                        'message': 'Tilt window - no new entries allowed',
                        'bot_id': None,
                        'order_id': None,
                        'tilt_window': True,
                        'tilt_minutes_remaining': TILT_TRANSITION_MINUTES
                    }
            # =================================================================

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

            # FIX-014: Filter by symbol_affinity - skip bots that exclude this symbol
            # If "exclude": skip this bot for the symbol entirely
            bot_dispatches = [
                b for b in bot_dispatches
                if not (b.get('symbol_affinity') == 'exclude' and symbol in b.get('symbols', []))
            ]

            # FIX-014: Prioritize bots that prefer this symbol (preferred affinity)
            # Sort so preferred bots come first while preserving relative order within each affinity group
            def symbol_affinity_sort_key(bot):
                affinity = bot.get('symbol_affinity', 'agnostic')
                if affinity == 'preferred' and symbol in bot.get('symbols', []):
                    return 0  # highest priority
                elif affinity == 'agnostic':
                    return 1
                else:
                    return 2  # exclude (already filtered but just in case)
            bot_dispatches = sorted(bot_dispatches, key=symbol_affinity_sort_key)

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

                # Check Drawdown Limits before executing trade
                account_book = execution_params.get('account_book', 'personal')
                prop_firm_name = execution_params.get('prop_firm_name')

                # Calculate current drawdown from account equity vs balance
                current_drawdown = 0.0
                try:
                    # Try to get account info from MT5 adapter
                    account_info = mt5_adapter.get_account_info()
                    if account_info and account_info.balance > 0:
                        # Drawdown is (balance - equity) / balance when equity < balance
                        if account_info.equity < account_info.balance:
                            current_drawdown = (account_info.balance - account_info.equity) / account_info.balance
                except Exception as e:
                    logger.warning(f"Could not calculate drawdown: {e}")

                # Check if trade should be blocked due to drawdown limits
                should_block, block_reason = self.check_drawdown_limits(
                    account_book=account_book,
                    prop_firm_name=prop_firm_name,
                    current_drawdown=current_drawdown
                )

                if should_block:
                    logger.warning(f"Trade blocked due to drawdown limits: {block_reason}")
                    return {
                        'status': 'error',
                        'message': f'Trade blocked: {block_reason}',
                        'bot_id': bot_id,
                        'order_id': None
                    }

                # === SQS GATE CHECK (Story 4-7) ===
                # Evaluate SQS gate after drawdown check, before MT5 dispatch
                strategy_type = execution_params.get('strategy', 'scalping')
                # Infer scalping vs ORB from strategy name if not explicitly provided
                if strategy_type not in ('scalping', 'ORB'):
                    strategy_type = 'scalping'  # Default to scalping

                sqs_allowed, sqs_result = self.evaluate_sqs_gate(symbol, strategy_type)

                if not sqs_allowed:
                    logger.warning(
                        f"Trade blocked due to SQS gate: symbol={symbol} "
                        f"sqs={sqs_result.get('sqs', 0):.4f} "
                        f"threshold={sqs_result.get('threshold', 0):.2f} "
                        f"reason={sqs_result.get('reason', 'unknown')}"
                    )

                    # If hard block (SQS < 0.50), route to kill:pending for logging
                    if sqs_result.get('is_hard_block', False):
                        logger.error(
                            f"SQS HARD BLOCK: symbol={symbol} sqs={sqs_result.get('sqs', 0):.4f} "
                            f"- routed to kill:pending for audit logging"
                        )

                    return {
                        'status': 'error',
                        'message': f"Trade blocked by SQS gate: {sqs_result.get('reason', 'SQS below threshold')}",
                        'bot_id': bot_id,
                        'order_id': None,
                        'sqs': sqs_result.get('sqs', 0),
                        'threshold': sqs_result.get('threshold', 0),
                        'is_hard_block': sqs_result.get('is_hard_block', False)
                    }
                # === END SQS GATE CHECK ===

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
        regime_report: "RegimeReport",
        current_session: TradingSession,
        current_utc: datetime,
        mode: Optional[str] = None
    ) -> List[dict]:
        """
        Returns bots that are authorized to trade in this physics state and session.

        V2: Only @primal tagged bots are included.
        V3: Session-aware filtering for ICT-style strategies.
        V4: Trading mode filtering - only bots with matching trading_mode participate.
        V5: Pool-based filtering for regime-conditional strategy pool framework (Story 8.10).

        Args:
            regime_report: Current regime report from Sentinel (contains regime string + chaos_score)
            current_session: Current trading session
            current_utc: Current UTC time
            mode: Trading mode ('demo' or 'live'). If provided, filters bots by trading_mode.

        Returns:
            List of bot dicts with manifest data
        """
        from src.router.bot_manifest import TradingMode

        regime = regime_report.regime

        # HIGH_CHAOS: Trigger Layer 3 CHAOS response and block all new trades
        if regime == "HIGH_CHAOS":
            self._trigger_chaos_response(regime_report)
            logger.info("HIGH_CHAOS regime: all pools muted, Layer 3 CHAOS response triggered")
            return []

        # NEWS_EVENT: No new trades during kill zone
        if regime == "NEWS_EVENT":
            logger.info("NEWS_EVENT regime: kill zone - no new trades")
            return []

        # Get allowed strategy types for this regime (legacy fallback)
        allowed_strategies = self.regime_strategy_map.get(regime, ["STRUCTURAL"])

        # Determine target trading mode based on mode parameter
        # Note: DEMO mode was merged into PAPER mode
        target_trading_mode = None
        if mode == "demo":
            target_trading_mode = TradingMode.PAPER  # DEMO merged into PAPER
        elif mode == "live":
            target_trading_mode = TradingMode.LIVE

        # Get active pools for this regime (Story 8.10 pool framework)
        active_pools = self._get_active_pools_for_regime(regime)

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

                # Skip PAPER bots in live auctions
                if mode == "live" and manifest.trading_mode == TradingMode.PAPER:
                    logger.debug(f"Bot {manifest.bot_id} skipped: PAPER mode in live auction")
                    continue

                # V5: Pool-based filtering (Story 8.10)
                # Determine this bot's pool membership
                bot_pool_name = self._get_bot_pool_name(manifest)
                if bot_pool_name is not None:
                    # Bot belongs to a named pool - check if pool is active
                    if bot_pool_name not in active_pools:
                        logger.debug(
                            f"Bot {manifest.bot_id} filtered by pool: "
                            f"pool={bot_pool_name}, regime={regime}, active_pools={active_pools}"
                        )
                        continue
                else:
                    # Bot not in a named pool - use legacy strategy type filtering
                    if manifest.strategy_type.value not in allowed_strategies:
                        continue

                # V3: Filter by session compatibility
                if not self._is_bot_session_compatible(manifest, current_session, current_utc):
                    logger.debug(f"Bot {manifest.bot_id} filtered by session")
                    continue

                # Include pool name in bot data for downstream reference
                eligible.append({
                    "bot_id": manifest.bot_id,
                    "name": manifest.name,
                    "strategy_type": manifest.strategy_type.value,
                    "frequency": manifest.frequency.value,
                    "score": self._calculate_bot_score(manifest),
                    "win_rate": manifest.win_rate,
                    "prop_firm_safe": manifest.prop_firm_safe,
                    "symbols": manifest.symbols,
                    "symbol_affinity": getattr(manifest, 'symbol_affinity', 'agnostic'),
                    "manifest": manifest,
                    "pool_name": bot_pool_name,
                    # Timeframe fields
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

    # =============================================================================
    # Per-Session Bot Type Mix (Task 5: Wire per-session bot type mix into Commander)
    # =============================================================================

    # Mapping from StrategyType to bot_type for SESSION_BOT_MIX filtering
    # Bot types: ORB, MOM, MR, TC
    STRATEGY_TO_BOT_TYPE = {
        "ORB": "ORB",       # Opening Range Breakout
        "SCALPER": "MOM",  # Scalpers trade momentum
        "HFT": "MOM",      # HFT is momentum-based
        "STRUCTURAL": "MR", # Structural analysis is mean reversion
        "SWING": "TC",      # Swing trading is trend continuation
    }

    def _derive_bot_type(self, manifest: "BotManifest") -> str:
        """
        Derive bot_type (ORB/MOM/MR/TC) from manifest's strategy_type.

        Used for SESSION_BOT_MIX filtering during Tilt ACTIVATE.

        Args:
            manifest: Bot manifest

        Returns:
            Bot type string (ORB, MOM, MR, TC), defaults to "MR" if unknown
        """
        strategy_value = manifest.strategy_type.value
        bot_type = self.STRATEGY_TO_BOT_TYPE.get(strategy_value)
        if bot_type is None:
            logger.warning(
                f"Unknown strategy_type '{strategy_value}' for bot {manifest.bot_id}, "
                f"defaulting to MR for session filtering"
            )
            return "MR"
        return bot_type

    def _derive_bot_type_from_dict(self, bot_data: dict) -> str:
        """
        Derive bot_type (ORB/MOM/MR/TC) from legacy bot_data dict.

        Args:
            bot_data: Legacy bot dictionary

        Returns:
            Bot type string, defaults to "MR" if unknown
        """
        strategy_str = bot_data.get('strategy_type', 'STRUCTURAL')
        bot_type = self.STRATEGY_TO_BOT_TYPE.get(strategy_str)
        if bot_type is None:
            logger.warning(
                f"Unknown strategy_type '{strategy_str}' for bot {bot_data.get('bot_id', 'unknown')}, "
                f"defaulting to MR for session filtering"
            )
            return "MR"
        return bot_type

    def on_session_activate(self, canonical_window: str) -> None:
        """
        Handle Tilt ACTIVATE callback for session-based bot type filtering.

        Called by Tilt state machine when transitioning to ACTIVATE phase.
        Stores the canonical window name for use in subsequent auction runs.

        Args:
            canonical_window: The canonical window name (e.g., "LONDON_OPEN", "DEAD_ZONE")
        """
        self._current_canonical_window = canonical_window
        logger.info(
            f"Commander: Session activate callback received for window='{canonical_window}'"
        )

        # Log the allowed bot types for this window
        allowed_types = SessionDetector.get_session_bot_types(canonical_window)
        if allowed_types:
            logger.info(
                f"Commander: Bot types allowed for {canonical_window}: {allowed_types}"
            )
        else:
            logger.info(
                f"Commander: No trading allowed in {canonical_window} (DEAD_ZONE or unknown)"
            )

    def _filter_bots_by_session_bot_type(
        self,
        bots: List[dict],
        canonical_window: Optional[str]
    ) -> List[dict]:
        """
        Filter bots by session-based bot type mix.

        During Tilt ACTIVATE, filters the DPR-ranked queue so only bots of the
        correct types (as defined in SESSION_BOT_MIX) are slotted into the active roster.

        Args:
            bots: List of bot dicts (from _get_bots_for_regime_and_session)
            canonical_window: Current canonical window name

        Returns:
            Filtered list of bots
        """
        if not canonical_window:
            # No window set yet - skip filtering
            return bots

        # Check if trading is allowed in this window
        if not SessionDetector.is_trading_allowed(canonical_window):
            logger.info(
                f"Commander: No trading allowed in {canonical_window}, "
                f"filtering out all bots"
            )
            return []

        # Get allowed bot types for this window
        allowed_types = SessionDetector.get_session_bot_types(canonical_window)
        if not allowed_types:
            # Empty mix means no trading in this window (e.g., DEAD_ZONE)
            logger.info(
                f"Commander: Empty bot type mix for {canonical_window}, "
                f"no bots will be dispatched"
            )
            return []

        logger.debug(
            f"Commander: Filtering {len(bots)} bots by allowed types: {allowed_types}"
        )

        filtered = []
        for bot in bots:
            manifest = bot.get('manifest')
            if manifest is not None:
                # Bot has a manifest - derive bot_type from strategy_type
                bot_type = self._derive_bot_type(manifest)
            else:
                # Legacy bot - try to derive from bot_data
                bot_type = self._derive_bot_type_from_dict(bot)

            if bot_type in allowed_types:
                filtered.append(bot)
            else:
                logger.debug(
                    f"Commander: Bot {bot.get('bot_id')} filtered out - "
                    f"bot_type={bot_type} not in {allowed_types}"
                )

        logger.info(
            f"Commander: Bot type filtering result: {len(filtered)}/{len(bots)} bots allowed"
        )
        return filtered

    def _count_active_positions(self) -> int:
        """Count currently active positions across all bots."""
        # TODO: Get from socket_server or position tracker
        return 0

    # =============================================================================
    # FIX-015: Tilt Enforcement - Entry Blocking + Force Close
    # =============================================================================

    async def manage_running_positions(
        self,
        running_bots: List[Dict[str, Any]],
        utc_now: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        FIX-015: Enforce tilt actions on running positions during tilt window.

        During T-5 tilt window, evaluates each running position and:
        - If get_tilt_action() returns 'take_profit': close with TILT_PROFIT_TAKE
        - If get_tilt_action() returns 'force_close': close with SESSION_END_CLOSE

        This should be called from the main tick loop or position management loop.

        Args:
            running_bots: List of dicts with bot_id, current_profit, rvol, is_scalper
            utc_now: UTC datetime to check. Defaults to now.

        Returns:
            Dict with tilt enforcement results: closed_positions, errors
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        elif utc_now.tzinfo is None:
            utc_now = utc_now.replace(tzinfo=timezone.utc)

        results = {
            "tilt_window_active": False,
            "closed_positions": [],
            "errors": []
        }

        # Check if we're in tilt window
        if not SessionDetector.is_tilt_window(utc_now):
            return results

        results["tilt_window_active"] = True
        logger.info(f"TILT WINDOW: Managing {len(running_bots)} running positions")

        for bot in running_bots:
            bot_id = bot.get("bot_id")
            if not bot_id:
                continue

            current_profit = bot.get("current_profit", 0.0)
            rvol = bot.get("rvol", 1.0)

            # Get tilt action for this bot
            tilt_action = SessionDetector.get_tilt_action(
                bot_id=bot_id,
                current_profit=current_profit,
                rvol=rvol,
                utc_now=utc_now
            )

            if tilt_action is None:
                continue

            # Execute the tilt action
            if tilt_action == "take_profit":
                close_reason = CloseReason.TILT_PROFIT_TAKE
                logger.info(
                    f"TILT: Taking profit on bot {bot_id} - "
                    f"profit={current_profit:.2f}, rvol={rvol:.4f}"
                )
            elif tilt_action == "force_close":
                close_reason = CloseReason.SESSION_END_CLOSE
                logger.info(
                    f"TILT: Force closing bot {bot_id} at session boundary - "
                    f"profit={current_profit:.2f}, rvol={rvol:.4f}"
                )
            else:
                continue

            # Execute the close
            try:
                close_result = await self._close_position(
                    bot_id=bot_id,
                    close_reason=close_reason,
                    current_profit=current_profit
                )
                if close_result:
                    results["closed_positions"].append({
                        "bot_id": bot_id,
                        "action": tilt_action,
                        "close_reason": close_reason.value,
                        "profit": current_profit
                    })
            except Exception as e:
                error_msg = f"Failed to close bot {bot_id}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        return results

    async def check_session_boundary(
        self,
        running_bots: List[Dict[str, Any]],
        utc_now: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        FIX-015: Scalper force-close at session boundary (T-0).

        At session end, force close all scalping positions that are still open.
        This ensures no positions carry over between sessions.

        Args:
            running_bots: List of dicts with bot_id, is_scalper, is_open
            utc_now: UTC datetime to check. Defaults to now.

        Returns:
            Dict with session boundary results: scalpers_closed, errors
        """
        if utc_now is None:
            utc_now = datetime.now(timezone.utc)
        elif utc_now.tzinfo is None:
            utc_now = utc_now.replace(tzinfo=timezone.utc)

        results = {
            "session_end_detected": False,
            "scalpers_closed": [],
            "errors": []
        }

        # Check if we're at session end (within 1 minute)
        current_window = SessionDetector.detect_canonical_window(utc_now)
        if current_window is None:
            return results

        # Get time until session end
        session_template = SessionDetector._get_template_for_window(current_window)
        if session_template is None:
            return results

        # Parse session end time
        end_hour, end_minute = map(int, session_template.end_gmt.split(":"))
        end_time = utc_now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

        # Handle overnight sessions
        if end_time <= utc_now:
            end_time += __import__('datetime').timedelta(days=1)

        delta = end_time - utc_now
        minutes_until_end = delta.total_seconds() / 60

        # Only proceed if within 1 minute of session end
        if minutes_until_end > 1:
            return results

        results["session_end_detected"] = True
        logger.info(
            f"SESSION END: Force closing scalpers - "
            f"window={current_window}, minutes_until_end={minutes_until_end:.2f}"
        )

        # Force close all scalping bots
        for bot in running_bots:
            bot_id = bot.get("bot_id")
            if not bot_id:
                continue

            is_scalper = bot.get("is_scalper", False)
            is_open = bot.get("is_open", True)

            if not is_scalper or not is_open:
                continue

            current_profit = bot.get("current_profit", 0.0)

            logger.info(
                f"SESSION END: Force closing scalper {bot_id} - "
                f"profit={current_profit:.2f}"
            )

            try:
                close_result = await self._close_position(
                    bot_id=bot_id,
                    close_reason=CloseReason.SESSION_END_CLOSE,
                    current_profit=current_profit
                )
                if close_result:
                    results["scalpers_closed"].append({
                        "bot_id": bot_id,
                        "profit": current_profit,
                        "close_reason": CloseReason.SESSION_END_CLOSE.value
                    })
            except Exception as e:
                error_msg = f"Failed to force close scalper {bot_id}: {e}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

        return results

    async def _close_position(
        self,
        bot_id: str,
        close_reason: "CloseReason",
        current_profit: float
    ) -> bool:
        """
        Execute close for a single position.

        Args:
            bot_id: Bot identifier
            close_reason: CloseReason enum value
            current_profit: Current profit for logging

        Returns:
            True if close executed successfully
        """
        try:
            # Dispatch close command to the bot
            instruction = "CLOSE"
            success = self.dispatch(
                bot_id=bot_id,
                instruction=instruction,
                mode="live"  # Use live mode for closes during tilt
            )

            if success:
                logger.info(
                    f"Position closed: bot={bot_id}, reason={close_reason.value}, "
                    f"profit={current_profit:.2f}"
                )
            else:
                logger.warning(f"Failed to dispatch close for bot {bot_id}")

            return success

        except Exception as e:
            logger.error(f"Error closing position for bot {bot_id}: {e}")
            return False

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
            "regime_strategy_map": self.regime_strategy_map,
            "regime_pool_map": {
                regime: {pool: state.value for pool, state in pools.items()}
                for regime, pools in self.regime_pool_map.items()
            }
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
            primary_regime_report,
            current_session,
            current_utc,
            mode=mode
        )

        # Task 5: Apply session-based bot type filtering (SESSION_BOT_MIX)
        # This filters bots by type (ORB/MOM/MR/TC) based on the canonical window
        if self._current_canonical_window:
            eligible_bots = self._filter_bots_by_session_bot_type(
                eligible_bots, self._current_canonical_window
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

    # ========== Regime-Conditional Strategy Pool Framework (Story 8.10) ==========

    def _get_pool_states_for_regime(self, regime: str) -> Dict[str, PoolState]:
        """
        Get pool states for a given regime.

        Args:
            regime: Market regime string (e.g., "TREND_STABLE", "HIGH_CHAOS")

        Returns:
            Dict of pool_name -> PoolState for the given regime
        """
        return self.regime_pool_map.get(regime, {})

    def _get_active_pools_for_regime(
        self,
        regime: str,
        symbol: Optional[str] = None,
        volume: Optional[float] = None
    ) -> List[str]:
        """
        Get list of active pool names for a given regime.

        Handles CONDITIONAL pools by checking volume confirmation.

        Args:
            regime: Market regime string
            symbol: Trading symbol for volume check (optional)
            volume: Current volume for CONDITIONAL pool evaluation (optional)

        Returns:
            List of active pool names
        """
        pool_states = self._get_pool_states_for_regime(regime)
        active_pools = []

        for pool_name, state in pool_states.items():
            if state == PoolState.ACTIVE:
                active_pools.append(pool_name)
            elif state == PoolState.CONDITIONAL:
                # Check volume confirmation for CONDITIONAL pools
                if self._check_volume_confirmation(pool_name, symbol, volume):
                    active_pools.append(pool_name)
                    logger.debug(f"Pool {pool_name} activated: volume confirmed")
                else:
                    logger.debug(f"Pool {pool_name} not activated: volume not confirmed")

        return active_pools

    def _check_volume_confirmation(
        self,
        pool_name: str,
        symbol: Optional[str] = None,
        volume: Optional[float] = None
    ) -> bool:
        """
        Check if volume confirms a CONDITIONAL pool activation.

        For ORB Long pools under TREND_STABLE, requires volume to exceed
        the session average volume threshold.

        Args:
            pool_name: Pool name (e.g., "orb_long")
            symbol: Trading symbol
            volume: Current volume reading

        Returns:
            True if volume confirms activation, False otherwise
        """
        if "orb_long" not in pool_name:
            return False

        if symbol is None:
            symbol = "EURUSD"  # Default symbol

        if volume is None:
            # Get volume from market scanner or use demo threshold
            volume = self._get_current_volume(symbol)

        # Volume confirmation threshold: current volume > 1.2x session average
        VOLUME_CONFIRMATION_THRESHOLD = 1.2
        session_avg = self._get_session_average_volume(symbol)

        if session_avg <= 0:
            # No volume data available - be conservative and don't activate
            logger.debug(f"No session average volume for {symbol}, ORB Long not activated")
            return False

        confirmed = volume >= (session_avg * VOLUME_CONFIRMATION_THRESHOLD)
        logger.debug(
            f"Volume check for {pool_name}: current={volume:.2f}, "
            f"session_avg={session_avg:.2f}, threshold={session_avg * VOLUME_CONFIRMATION_THRESHOLD:.2f}, "
            f"confirmed={confirmed}"
        )
        return confirmed

    def _get_current_volume(self, symbol: str) -> float:
        """
        Get current volume for a symbol from market feed.

        Args:
            symbol: Trading symbol

        Returns:
            Current volume or demo value
        """
        try:
            from src.risk.integrations.mt5_client import get_mt5_client
            client = get_mt5_client()
            if client and hasattr(client, 'is_connected') and client.is_connected():
                tick = client.get_tick(symbol)
                if tick and hasattr(tick, 'volume'):
                    return float(tick.volume)
        except Exception:
            pass

        # Demo fallback
        demo_volumes = {
            "EURUSD": 15000.0,
            "GBPUSD": 12000.0,
            "USDJPY": 18000.0,
            "XAUUSD": 8000.0,
        }
        return demo_volumes.get(symbol, 10000.0)

    def _get_session_average_volume(self, symbol: str) -> float:
        """
        Get average volume for current session.

        Args:
            symbol: Trading symbol

        Returns:
            Average volume for the current session period
        """
        # In production, this would come from SVSS (Story 15) or historical data
        # For now, return demo values based on typical session volumes
        demo_session_avg = {
            "EURUSD": 12000.0,
            "GBPUSD": 9500.0,
            "USDJPY": 14000.0,
            "XAUUSD": 6500.0,
        }
        return demo_session_avg.get(symbol, 10000.0)

    def _get_bot_pool_name(self, manifest: "BotManifest") -> Optional[str]:
        """
        Determine the pool name for a bot based on its manifest.

        Pool membership is determined by strategy_type and trading direction.
        The direction is inferred from the bot's name or tags if not explicit.

        Args:
            manifest: Bot manifest

        Returns:
            Pool name string or None if not pool-assigned
        """
        strategy = manifest.strategy_type.value
        name_lower = manifest.name.lower() if manifest.name else ""
        tags = manifest.tags if manifest.tags else []

        # Determine direction from name or tags
        direction = "neutral"  # default
        if "long" in name_lower or "long" in tags:
            direction = "long"
        elif "short" in name_lower or "short" in tags:
            direction = "short"
        elif "false_breakout" in name_lower or "fb" in tags:
            direction = "false_breakout"

        # Map strategy + direction to pool name
        if strategy == "SCALPER":
            return f"scalping_{direction}"
        elif strategy == "ORB":
            return f"orb_{direction}"
        elif strategy == "STRUCTURAL":
            # Structural strategies don't belong to scalping/ORB pools
            return None
        elif strategy == "SWING":
            return None
        elif strategy == "HFT":
            return None
        return None

    def _trigger_chaos_response(self, regime_report: "RegimeReport") -> None:
        """
        Trigger Layer 3 CHAOS response when HIGH_CHAOS regime is detected.

        Wires to ProgressiveKillSwitch for coordinated position management
        and risk mitigation.

        Args:
            regime_report: Current regime report with chaos details
        """
        try:
            from src.router.progressive_kill_switch import get_progressive_kill_switch

            pks = get_progressive_kill_switch()
            logger.warning(
                f"HIGH_CHAOS detected: chaos_score={regime_report.chaos_score:.3f}, "
                f"regime_quality={regime_report.regime_quality:.3f}. "
                f"Triggering Layer 3 CHAOS response via ProgressiveKillSwitch."
            )

            # Trigger chaos response through the progressive kill switch
            # This evaluates all tiers and executes appropriate protective actions
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(pks.check_all_tiers({
                        'reason': 'HIGH_CHAOS_regime',
                        'chaos_score': regime_report.chaos_score,
                        'regime': regime_report.regime,
                    }))
                else:
                    loop.run_until_complete(pks.check_all_tiers({
                        'reason': 'HIGH_CHAOS_regime',
                        'chaos_score': regime_report.chaos_score,
                        'regime': regime_report.regime,
                    }))
            except RuntimeError:
                asyncio.run(pks.check_all_tiers({
                    'reason': 'HIGH_CHAOS_regime',
                    'chaos_score': regime_report.chaos_score,
                    'regime': regime_report.regime,
                }))

        except Exception as e:
            logger.error(f"Failed to trigger chaos response: {e}")
