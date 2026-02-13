"""
The Commander (Execution Layer)
Responsible for selecting and dispatching Bots based on Regime.

V2: Integrated with BotManifest system for tag-based authorization.
Only @primal tagged bots can participate in the Strategy Auction.
"""

from typing import List, Dict, Optional, TYPE_CHECKING
import logging
import json

if TYPE_CHECKING:
    from src.router.bot_manifest import BotRegistry, BotManifest
    from src.router.sentinel import RegimeReport
    from src.router.governor import Governor
    from src.router.multi_timeframe_sentinel import MultiTimeframeSentinel, Timeframe

from datetime import datetime, timezone
from src.router.sessions import SessionDetector, TradingSession

from src.router.dynamic_bot_limits import DynamicBotLimiter

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
    
    def _load_regime_map(self) -> Dict:
        """Load regime → bot mapping configuration."""
        # TODO: Load from config file
        return {}
    
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
    
    def run_auction(self, regime_report: "RegimeReport", account_balance: Optional[float] = None, broker_id: str = "mt5_default", current_utc: Optional[datetime] = None) -> List[dict]:
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
        logger.info(f"Running auction for session: {current_session.value}, UTC time: {current_utc.isoformat()}")

        # Get eligible bots for current regime and session (Comment 1 fix)
        eligible_bots = self._get_bots_for_regime_and_session(regime_report.regime, current_session, current_utc)
        
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
        limiter = DynamicBotLimiter()
        max_bots = limiter.get_max_bots(account_balance)
        max_selection = min(3, max_bots - self._count_active_positions())
        top_bots = ranked_bots[:max(0, max_selection)]

        # 5. Calculate position sizes for each bot using Governor (if available)
        dispatches = []
        for bot in top_bots:
            # Build trade proposal for Governor
            trade_proposal = {
                'symbol': bot.get('symbols', ['EURUSD'])[0],
                'current_balance': account_balance,
                'account_balance': account_balance,
                'broker_id': broker_id,
                'stop_loss_pips': bot.get('stop_loss_pips', 20.0),
                'win_rate': bot.get('win_rate', 0.55),
                'avg_win': bot.get('avg_win', 400.0),
                'avg_loss': bot.get('avg_loss', 200.0),
                'current_atr': bot.get('current_atr', 0.0012),
                'average_atr': bot.get('average_atr', 0.0010),
            }

            # Get risk mandate with position sizing from Governor
            # NOTE: _governor is guaranteed to be non-None by __init__ guard
            mandate = self._governor.calculate_risk(
                regime_report,
                trade_proposal,
                account_balance,
                broker_id
            )

            # Filter by position size (for enhanced Kelly sizing)
            # In legacy mode (_use_enhanced_sizing=False), inject minimal position to bypass filter
            # so bots still get dispatched even without Kelly sizing
            if self._use_enhanced_sizing:
                # EnhancedGovernor: Only include bots with valid position sizes from Kelly
                if mandate.position_size <= 0:
                    continue
                position_size = mandate.position_size
            else:
                # Base Governor: Skip position_size filter (legacy mode)
                # Inject minimal position size for compatibility
                position_size = mandate.position_size if mandate.position_size > 0 else 0.01

            dispatch = bot.copy()
            dispatch.update({
                'position_size': position_size,
                'kelly_fraction': mandate.kelly_fraction,
                'risk_amount': mandate.risk_amount,
                'kelly_adjustments': mandate.kelly_adjustments,
                'notes': mandate.notes,
                'regime': regime_report.regime,
                'risk_mode': mandate.risk_mode
            })
            dispatches.append(dispatch)

        logger.info(f"Auction result: {len(dispatches)} bots with position sizes for {regime_report.regime}")
        return dispatches
    
    def _get_bots_for_regime_and_session(self, regime: str, current_session: TradingSession, current_utc: datetime) -> List[dict]:
        """
        Returns bots that are authorized to trade in this physics state and session.

        V2: Only @primal tagged bots are included.
        V3: Session-aware filtering for ICT-style strategies.

        Args:
            regime: Current market regime from Sentinel
            current_session: Current trading session
            current_utc: Current UTC time

        Returns:
            List of bot dicts with manifest data
        """
        # No trading in extreme conditions
        if regime in ["HIGH_CHAOS", "NEWS_EVENT"]:
            return []

        # Get allowed strategy types for this regime
        allowed_strategies = self.regime_strategy_map.get(regime, ["STRUCTURAL"])

        # Try BotRegistry first (V2)
        if self.bot_registry:
            logger.debug("Using BotRegistry path for bot selection")
            primal_bots = self.bot_registry.list_by_tag("@primal")

            eligible = []
            for manifest in primal_bots:
                # Filter by strategy type compatibility with regime
                if manifest.strategy_type.value not in allowed_strategies:
                    continue

                # V3: Filter by session compatibility
                if not self._is_bot_session_compatible(manifest, current_session, current_utc):
                    logger.debug(f"Bot {manifest.bot_id} filtered by session")
                    continue

                # Comment 3 fix: Include timeframe fields from BotManifest for multi-timeframe filtering
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
    
    def dispatch(self, bot_id: str, instruction: str) -> bool:
        """
        Sends JSON command to the Interface via socket.
        
        Args:
            bot_id: Target bot ID
            instruction: Command to send (e.g., "OPEN_BUY", "CLOSE_ALL")
            
        Returns:
            True if dispatch successful
        """
        try:
            # TODO: Integrate with socket_server
            command = {
                "type": "DISPATCH",
                "bot_id": bot_id,
                "instruction": instruction,
                "timestamp": __import__('datetime').datetime.now().isoformat()
            }
            logger.info(f"Dispatching to {bot_id}: {instruction}")
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
        current_utc: Optional[datetime] = None
    ) -> List[dict]:
        """
        Multi-timeframe-aware bot auction.
        
        Comment 1: Extend run_auction to select bots by preferred_timeframe
        and enforce secondary_timeframes alignment when use_multi_timeframe is true.
        
        This method:
        1. Gets eligible bots from the registry
        2. Filters bots by their preferred_timeframe vs available timeframe regimes
        3. For bots with use_multi_timeframe=True, enforces secondary_timeframes alignment
        4. Returns dispatches with timeframe context
        
        Args:
            primary_regime_report: Primary regime report (fastest timeframe)
            all_timeframe_regimes: Dict of timeframe -> regime report for all timeframes
            primary_timeframe: The actual primary timeframe being used (e.g., fastest).
                             If None, derived from all_timeframe_regimes keys.
            account_balance: Account balance for position sizing
            broker_id: Broker identifier for fee-aware Kelly
            current_utc: UTC timestamp for session-aware filtering
            
        Returns:
            List of bot dispatches with timeframe context
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
        
        # Detect current session
        current_session = SessionDetector.detect_session(current_utc)
        logger.info(
            f"Running multi-timeframe auction for session: {current_session.value}, "
            f"UTC time: {current_utc.isoformat()}, timeframes: {[tf.name for tf in all_timeframe_regimes.keys()]}, "
            f"primary_timeframe: {primary_timeframe.name if primary_timeframe else 'None'}"
        )

        # Get all eligible bots
        eligible_bots = self._get_bots_for_regime_and_session(
            primary_regime_report.regime, 
            current_session, 
            current_utc
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

        # Select top N
        limiter = DynamicBotLimiter()
        max_bots = limiter.get_max_bots(account_balance)
        max_selection = min(3, max_bots - self._count_active_positions())
        top_bots = ranked_bots[:max(0, max_selection)]

        # Calculate position sizes and build dispatches
        dispatches = []
        for bot in top_bots:
            trade_proposal = {
                'symbol': bot.get('symbols', ['EURUSD'])[0],
                'current_balance': account_balance,
                'account_balance': account_balance,
                'broker_id': broker_id,
                'stop_loss_pips': bot.get('stop_loss_pips', 20.0),
                'win_rate': bot.get('win_rate', 0.55),
                'avg_win': bot.get('avg_win', 400.0),
                'avg_loss': bot.get('avg_loss', 200.0),
                'current_atr': bot.get('current_atr', 0.0012),
                'average_atr': bot.get('average_atr', 0.0010),
            }

            mandate = self._governor.calculate_risk(
                primary_regime_report,
                trade_proposal,
                account_balance,
                broker_id
            )

            if self._use_enhanced_sizing:
                if mandate.position_size <= 0:
                    continue
                position_size = mandate.position_size
            else:
                position_size = mandate.position_size if mandate.position_size > 0 else 0.01

            dispatch = bot.copy()
            dispatch.update({
                'position_size': position_size,
                'kelly_fraction': mandate.kelly_fraction,
                'risk_amount': mandate.risk_amount,
                'kelly_adjustments': mandate.kelly_adjustments,
                'notes': mandate.notes,
                'regime': primary_regime_report.regime,
                'risk_mode': mandate.risk_mode
            })
            
            # Comment 1 fix: Use actual primary_timeframe instead of hardcoded 'M5'
            # Comment 3 fix: Add top-level timeframe and multi_timeframe_aligned fields
            bot_preferred_tf = bot.get('preferred_timeframe')
            actual_timeframe = bot_preferred_tf if bot_preferred_tf else (primary_timeframe.name if primary_timeframe else 'H1')
            
            # Determine multi_timeframe_alignment for multi-timeframe bots
            use_mtf = bot.get('use_multi_timeframe', False)
            multi_timeframe_aligned = False
            if use_mtf and bot.get('secondary_timeframes'):
                multi_timeframe_aligned = self._check_secondary_timeframes_alignment(
                    bot.get('secondary_timeframes'),
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

        logger.info(f"Multi-timeframe auction result: {len(dispatches)} bots dispatched")
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
