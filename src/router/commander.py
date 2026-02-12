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

from datetime import datetime, timezone
from src.router.sessions import SessionDetector, TradingSession

logger = logging.getLogger(__name__)

# Global bot limit
MAX_ACTIVE_BOTS = 50


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

        # 1. Get eligible bots (filtered by regime, @primal tag, and session)
        eligible_bots = self._get_bots_for_regime_and_session(regime_report.regime, current_session, current_utc)

        if not eligible_bots:
            logger.debug(f"No eligible bots for regime: {regime_report.regime}")
            return []

        # 2. Apply chaos-based filtering
        if regime_report.chaos_score > 0.6:
            logger.info("High chaos detected - reducing bot selection")
            # In high chaos, only allow conservative bots
            eligible_bots = [b for b in eligible_bots
                           if b.get('frequency') in ['LOW', 'MEDIUM']]

        # 3. Rank by performance (win_rate * score)
        ranked_bots = sorted(
            eligible_bots,
            key=lambda x: x.get('score', 0) * x.get('win_rate', 0.5),
            reverse=True
        )

        # 4. Select top N (capped by global limit)
        max_selection = min(3, MAX_ACTIVE_BOTS - self._count_active_positions())
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

                eligible.append({
                    "bot_id": manifest.bot_id,
                    "name": manifest.name,
                    "strategy_type": manifest.strategy_type.value,
                    "frequency": manifest.frequency.value,
                    "score": self._calculate_bot_score(manifest),
                    "win_rate": manifest.win_rate,
                    "prop_firm_safe": manifest.prop_firm_safe,
                    "symbols": manifest.symbols,
                    "manifest": manifest  # Include full manifest for downstream
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
        
        return {
            "active_bots": len(self.active_bots),
            "primal_bots": primal_count,
            "pending_bots": pending_count,
            "max_bots": MAX_ACTIVE_BOTS,
            "regime_strategy_map": self.regime_strategy_map
        }
