"""
QuantMindX Strategy Router Engine
The coordination layer for the Sentient Loop.

V2: Integrated with Phase 2 components:
- BotRegistry for manifest-based bot management
- SmartKillSwitch for regime-aware exits
- TradeLogger for enhanced trade context logging
- RoutingMatrix for automatic bot-to-account assignment
"""

import logging
import asyncio
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.router.bot_manifest import BotRegistry, BotManifest
    from src.router.kill_switch import SmartKillSwitch
    from src.router.trade_logger import TradeLogger
    from src.router.routing_matrix import RoutingMatrix

from src.router.sentinel import Sentinel
from src.router.governor import Governor, RiskMandate
from src.router.enhanced_governor import EnhancedGovernor
from src.router.commander import Commander
from src.router.multi_timeframe_sentinel import MultiTimeframeSentinel, Timeframe

logger = logging.getLogger(__name__)


class StrategyRouter:
    """
    The Orchestrator of the Strategy Router system.
    Ties together the Intelligence, Compliance, and Decision layers.
    
    V2 Integration:
    - BotRegistry: Manifest-based bot management with @tags
    - SmartKillSwitch: Regime-aware exit strategies
    - TradeLogger: Enhanced "Why?" context logging
    - RoutingMatrix: Automatic bot → account assignment
    """
    
    def __init__(self, use_smart_kill: bool = True, use_kelly_governor: bool = True, 
                 use_multi_timeframe: bool = True, multi_timeframes: Optional[List[Timeframe]] = None):
        """
        Initialize Strategy Router with all components.

        Args:
            use_smart_kill: If True, use SmartKillSwitch (default)
            use_kelly_governor: If True, use EnhancedGovernor with Kelly (default)
            use_multi_timeframe: If True, use MultiTimeframeSentinel for multi-timeframe regime detection
            multi_timeframes: List of timeframes for multi-timeframe sentinel (default: M5, H1, H4)
        """
        # Core Sentient Loop components
        self.sentinel = Sentinel()

        # Multi-timeframe sentinel (Comment 1: instantiate and share with Commander)
        self._use_multi_timeframe = use_multi_timeframe
        if use_multi_timeframe:
            if multi_timeframes is None:
                multi_timeframes = [Timeframe.M5, Timeframe.H1, Timeframe.H4]
            self.multi_timeframe_sentinel = MultiTimeframeSentinel(timeframes=multi_timeframes)
            logger.info(f"StrategyRouter: MultiTimeframeSentinel initialized with timeframes={[tf.name for tf in multi_timeframes]}")
        else:
            self.multi_timeframe_sentinel = None

        # V3: Use Enhanced Governor with Kelly Calculator
        if use_kelly_governor:
            self.governor = EnhancedGovernor()
        else:
            self.governor = Governor()

        self.commander = Commander(governor=self.governor)
        
        # Share multi_timeframe_sentinel with Commander
        self.commander._multi_timeframe_sentinel = self.multi_timeframe_sentinel
        
        # Phase 2 components (lazy loaded)
        self._bot_registry: Optional["BotRegistry"] = None
        self._kill_switch: Optional["SmartKillSwitch"] = None
        self._trade_logger: Optional["TradeLogger"] = None
        self._routing_matrix: Optional["RoutingMatrix"] = None
        
        # Configuration
        self._use_smart_kill = use_smart_kill
        self._use_kelly_governor = use_kelly_governor
        
        # In-memory registration of bots (legacy support)
        self.registered_bots = {}
        
        # Current state
        self._last_regime = None
        self._tick_count = 0
        
        logger.info("StrategyRouter initialized with Phase 2 integration")
    
    # ========== Lazy Loading Properties ==========
    
    @property
    def bot_registry(self) -> Optional["BotRegistry"]:
        """Lazy load BotRegistry."""
        if self._bot_registry is None:
            try:
                from src.router.bot_manifest import BotRegistry
                self._bot_registry = BotRegistry()
                # Share with Commander
                self.commander._bot_registry = self._bot_registry
                logger.info("StrategyRouter: Loaded BotRegistry")
            except Exception as e:
                logger.warning(f"StrategyRouter: Could not load BotRegistry: {e}")
        return self._bot_registry
    
    @property
    def kill_switch(self) -> Optional["SmartKillSwitch"]:
        """Lazy load SmartKillSwitch."""
        if self._kill_switch is None:
            try:
                if self._use_smart_kill:
                    from src.router.kill_switch import SmartKillSwitch
                    self._kill_switch = SmartKillSwitch(
                        sentinel=self.sentinel,
                        governor=self.governor
                    )
                else:
                    from src.router.kill_switch import KillSwitch
                    self._kill_switch = KillSwitch()
                logger.info("StrategyRouter: Loaded KillSwitch")
            except Exception as e:
                logger.warning(f"StrategyRouter: Could not load KillSwitch: {e}")
        return self._kill_switch
    
    @property
    def trade_logger(self) -> Optional["TradeLogger"]:
        """Lazy load TradeLogger."""
        if self._trade_logger is None:
            try:
                from src.router.trade_logger import TradeLogger
                self._trade_logger = TradeLogger()
                logger.info("StrategyRouter: Loaded TradeLogger")
            except Exception as e:
                logger.warning(f"StrategyRouter: Could not load TradeLogger: {e}")
        return self._trade_logger
    
    @property
    def routing_matrix(self) -> Optional["RoutingMatrix"]:
        """Lazy load RoutingMatrix."""
        if self._routing_matrix is None:
            try:
                from src.router.routing_matrix import RoutingMatrix
                self._routing_matrix = RoutingMatrix()
                logger.info("StrategyRouter: Loaded RoutingMatrix")
            except Exception as e:
                logger.warning(f"StrategyRouter: Could not load RoutingMatrix: {e}")
        return self._routing_matrix
    
    # ========== Core Sentient Loop ==========
    
    def process_tick(self, symbol: str, price: float, account_data: Optional[Dict] = None) -> Dict:
        """
        The Full Sentient Loop execution.
        
        V2: Includes kill switch check, trade logging, and routing matrix.
        """
        self._tick_count += 1
        
        # 0. Kill Switch Check - if active, block all processing
        if self.kill_switch and self.kill_switch.is_active:
            logger.warning("Kill switch active - blocking tick processing")
            return {
                "regime": "HALTED",
                "quality": 0.0,
                "mandate": RiskMandate(allocation_scalar=0.0, risk_mode="HALTED"),
                "dispatches": [],
                "kill_switch_active": True
            }
        
        # 1. Observe (Sentinel)
        report = self.sentinel.on_tick(symbol, price)
        logger.debug(f"Sentinel Report: {report.regime} (Quality: {report.regime_quality:.2f})")

        # Track regime changes
        if self._last_regime != report.regime:
            logger.info(f"Regime changed: {self._last_regime} → {report.regime}")
            self._last_regime = report.regime

        # Extract account_balance and broker_id from account_data
        account_balance = account_data.get('account_balance', 10000.0) if account_data else 10000.0
        broker_id = account_data.get('broker_id', 'mt5_default') if account_data else 'mt5_default'

        # Get current UTC time for session-aware filtering
        from datetime import datetime, timezone
        current_utc = datetime.now(timezone.utc)

        # 2. Govern (Governor)
        # We issue a "Swarm Mandate" for the current physics
        # pass account_data for Tier 3 Prop rules
        proposal = {"symbol": symbol}
        if account_data:
            proposal.update(account_data)

        mandate = self.governor.calculate_risk(report, proposal, account_balance=account_balance, broker_id=broker_id)

        # 3. Command (Commander)
        # Identify which bots are authorized to trade in this regime
        # V2: Only @primal tagged bots participate
        # V3: Pass current_utc for session-aware filtering
        # Pass account_balance and broker_id for fee-aware position sizing
        dispatches = self.commander.run_auction(report, account_balance, broker_id, current_utc)
        
        # Apply Governor's Mandate to dispatches
        for bot in dispatches:
            bot['authorized_risk_scalar'] = mandate.allocation_scalar
            bot['risk_mode'] = mandate.risk_mode
            
            # V2: Apply routing matrix if available
            if self.routing_matrix and 'manifest' in bot:
                account_assignment = self.routing_matrix.get_account_for_bot(bot['manifest'])
                bot['assigned_account'] = account_assignment
        
        # 4. Log (TradeLogger) - V2 addition
        if self.trade_logger and dispatches:
            self.trade_logger.log_dispatch_context(
                regime=report.regime,
                chaos_score=report.chaos_score,
                mandate=mandate,
                dispatches=dispatches,
                symbol=symbol
            )
        
        return {
            "regime": report.regime,
            "quality": report.regime_quality,
            "chaos_score": report.chaos_score,
            "mandate": mandate,
            "dispatches": dispatches,
            "tick_count": self._tick_count
        }
    
    # ========== Multi-Timeframe Processing (Comment 1) ==========
    
    def process_tick_multi_timeframe(self, symbol: str, price: float, 
                                     account_data: Optional[Dict] = None) -> Dict:
        """
        Process tick with multi-timeframe regime detection.
        
        Comment 1: Feed ticks into MultiTimeframeSentinel and pass 
        timeframe-specific regime reports into run_auction.
        
        This method:
        1. Feeds ticks into the multi_timeframe_sentinel
        2. Gets regime reports for each timeframe
        3. Passes timeframe-specific regime reports to Commander.run_auction
        4. Returns dispatches with timeframe context
        
        Args:
            symbol: Trading symbol
            price: Current price
            account_data: Optional account data for position sizing
            
        Returns:
            Dict with multi-timeframe regime info and dispatches
        """
        if self.multi_timeframe_sentinel is None:
            logger.warning("Multi-timeframe sentinel not enabled, falling back to single timeframe")
            return self.process_tick(symbol, price, account_data)
        
        from datetime import datetime, timezone
        
        # Get current UTC time
        current_utc = datetime.now(timezone.utc)
        
        # Feed tick into multi-timeframe sentinel
        self.multi_timeframe_sentinel.on_tick(symbol, price, current_utc)
        
        # Comment 2 fix: Refresh self.sentinel to ensure auctions don't use stale/None primary regime
        # This also updates the main sentinel's current_report for consistency
        self.sentinel.on_tick(symbol, price)
        
        # Get regime reports for all timeframes
        all_regimes = self.multi_timeframe_sentinel.get_all_regimes()
        dominant_regime = self.multi_timeframe_sentinel.get_dominant_regime()
        
        logger.debug(f"Multi-timeframe regimes: {[(tf.name, r.regime) for tf, r in all_regimes.items()]}")
        
        # Comment 2 fix: Derive primary_regime_report from fastest entry in all_regimes if available
        # This ensures auctions pass a non-None regime aligned with the chosen primary timeframe
        if all_regimes:
            # Get the fastest timeframe (smallest interval) as primary
            sorted_timeframes = sorted(all_regimes.keys(), key=lambda tf: tf.seconds)
            primary_timeframe = sorted_timeframes[0]
            primary_regime_report = all_regimes[primary_timeframe]
        else:
            # Fallback to self.sentinel.current_report
            primary_regime_report = self.sentinel.current_report
        
        # Extract account_balance and broker_id from account_data
        account_balance = account_data.get('account_balance', 10000.0) if account_data else 10000.0
        broker_id = account_data.get('broker_id', 'mt5_default') if account_data else 'mt5_default'
        
        # Run auction with multi-timeframe context
        # Pass timeframe-specific regime reports and all_regimes dict
        dispatches = self.commander.run_auction_with_timeframes(
            primary_regime_report=primary_regime_report,
            all_timeframe_regimes=all_regimes,
            account_balance=account_balance,
            broker_id=broker_id,
            current_utc=current_utc
        )
        
        # Apply timeframe context to dispatches
        for dispatch in dispatches:
            dispatch['timeframe_context'] = {
                'dominant_regime': dominant_regime,
                'all_regimes': {tf.name: r.regime for tf, r in all_regimes.items()}
            }
        
        return {
            "regime": dominant_regime,
            "quality": self.sentinel.current_report.regime_quality if self.sentinel.current_report else 1.0,
            "chaos_score": self.sentinel.current_report.chaos_score if self.sentinel.current_report else 0.0,
            "all_timeframe_regimes": {tf.name: r.regime for tf, r in all_regimes.items()},
            "dispatches": dispatches,
            "tick_count": self._tick_count
        }
    
    # ========== Bot Lifecycle Management ==========

    
    def register_bot(self, bot_instance, initial_tag: str = "@pending") -> bool:
        """
        Register a new bot into the factory pipeline.
        
        V2: Registers with BotRegistry if available, sets initial tag.
        
        Args:
            bot_instance: Bot object with to_dict() method
            initial_tag: Initial lifecycle tag (default: @pending)
            
        Returns:
            True if registration successful
        """
        bot_id = bot_instance.bot_id
        
        # Legacy storage
        self.registered_bots[bot_id] = bot_instance
        
        # Convert to dict for Commander
        bot_dict = bot_instance.to_dict()
        
        # Set initial tag if not present
        if 'tags' not in bot_dict:
            bot_dict['tags'] = []
        if initial_tag not in bot_dict['tags']:
            bot_dict['tags'].append(initial_tag)
        
        # Register with Commander
        self.commander.register_bot(bot_id, bot_dict)
        
        # Register with KillSwitch for emergency stop
        if self.kill_switch:
            self.kill_switch.register_ea(bot_id)
        
        magic_number = getattr(bot_instance, 'magic_number', 'N/A')
        logger.info(f"Bot {bot_id} (@{magic_number}) registered with tag {initial_tag}")
        
        return True
    
    def promote_bot(self, bot_id: str, from_tag: str, to_tag: str) -> bool:
        """
        Promote a bot from one lifecycle stage to another.
        
        This is how bots go from @pending → @primal (human authorized).
        
        Args:
            bot_id: Bot identifier
            from_tag: Current tag (e.g., "@pending")
            to_tag: New tag (e.g., "@primal")
            
        Returns:
            True if promotion successful
        """
        if not self.bot_registry:
            logger.warning("BotRegistry not available for promotion")
            return False
        
        manifest = self.bot_registry.get(bot_id)
        if not manifest:
            logger.warning(f"Bot {bot_id} not found in registry")
            return False
        
        # Remove old tag, add new tag
        if from_tag in manifest.tags:
            manifest.tags.remove(from_tag)
        if to_tag not in manifest.tags:
            manifest.tags.append(to_tag)
        
        # Update registry
        self.bot_registry.register(manifest)
        
        logger.info(f"Bot {bot_id} promoted: {from_tag} → {to_tag}")
        return True
    
    def authorize_bot(self, bot_id: str) -> bool:
        """
        Convenience method: Authorize a bot for trading (@pending → @primal).
        
        This should be called after human review.
        """
        return self.promote_bot(bot_id, "@pending", "@primal")
    
    def quarantine_bot(self, bot_id: str, reason: str = "") -> bool:
        """
        Move bot to quarantine (disable trading).
        
        Args:
            bot_id: Bot identifier
            reason: Why bot is being quarantined
        """
        if not self.bot_registry:
            return False
        
        manifest = self.bot_registry.get(bot_id)
        if not manifest:
            return False
        
        # Remove active tags
        for tag in ["@primal", "@pending"]:
            if tag in manifest.tags:
                manifest.tags.remove(tag)
        
        # Add quarantine
        if "@quarantine" not in manifest.tags:
            manifest.tags.append("@quarantine")
        
        self.bot_registry.register(manifest)
        logger.warning(f"Bot {bot_id} quarantined: {reason}")
        return True
    
    # ========== Status & Monitoring ==========
    
    def get_status(self) -> Dict:
        """Get comprehensive router status for monitoring."""
        status = {
            "sentinel": {
                "current_regime": self._last_regime,
                "current_chaos": self.sentinel.current_report.chaos_score if self.sentinel.current_report else None
            },
            "commander": self.commander.get_auction_status(),
            "tick_count": self._tick_count,
            "kill_switch_active": self.kill_switch.is_active if self.kill_switch else False
        }
        
        if self.bot_registry:
            status["bots"] = {
                "total": len(self.bot_registry.list_all()),
                "primal": len(self.bot_registry.list_by_tag("@primal")),
                "pending": len(self.bot_registry.list_by_tag("@pending")),
                "quarantine": len(self.bot_registry.list_by_tag("@quarantine"))
            }
        
        return status
    
    async def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Trigger emergency stop via kill switch."""
        if self.kill_switch:
            from src.router.kill_switch import KillReason
            await self.kill_switch.trigger(
                reason=KillReason.MANUAL,
                triggered_by="strategy_router",
                message=reason
            )
    
    async def smart_exit(self, current_pnl_pct: float = 0.0, reason: str = "Session end") -> None:
        """Trigger smart exit (regime-aware)."""
        if self.kill_switch and hasattr(self.kill_switch, 'smart_trigger'):
            from src.router.kill_switch import KillReason
            await self.kill_switch.smart_trigger(
                reason=KillReason.SMART_EXIT,
                current_pnl_pct=current_pnl_pct,
                triggered_by="strategy_router",
                message=reason
            )

    # ========== Multi-Symbol Router Simulation (Task Group 5) ==========

    # Currency correlation matrix for cross-symbol exposure checks
    _CORRELATION_MATRIX = {
        ("EURUSD", "GBPUSD"): 0.85,     # Both USD majors
        ("EURUSD", "USDJPY"): -0.75,    # USD JPY inverse
        ("AUDUSD", "NZDUSD"): 0.80,     # Commodity dollars
        ("EURJPY", "GBPJPY"): 0.75,     # Both JPY crosses
        ("EURUSD", "XAUUSD"): 0.15,     # Low correlation
        ("BTCUSD", "ETHUSD"): 0.90,     # Crypto correlation
        ("GBPUSD", "AUDUSD"): 0.45,     # Moderate correlation
    }

    _HIGH_CORRELATION_THRESHOLD = 0.7
    _MAX_EXPOSURE_PER_CURRENCY = 2  # Max positions per currency pair

    async def process_multi_symbol_tick(
        self,
        symbols: List[str],
        prices: List[float],
        account_data: Optional[Dict] = None
    ) -> Dict:
        """
        Process multiple symbols in parallel for multi-symbol backtesting.

        V3: Multi-symbol extension with:
        - Asyncio parallel processing for all symbols
        - Cross-symbol correlation checks
        - Global auction across all symbols
        - Aggregated dispatch logging

        Args:
            symbols: List of symbol names (e.g., ['EURUSD', 'GBPUSD', 'XAUUSD'])
            prices: List of current prices (same length as symbols)
            account_data: Optional account information for Governor

        Returns:
            Dict with combined results from all symbols:
            {
                'combined_dispatches': List of all dispatches,
                'symbols_processed': Count of symbols handled,
                'total_dispatches': Total bot dispatches,
                'regime': Common regime,
                'correlation_warnings': List of correlation warnings
            }
        """
        if len(symbols) != len(prices):
            raise ValueError(f"Symbols ({len(symbols)}) and prices ({len(prices)}) must match")

        # Kill switch check
        if self.kill_switch and self.kill_switch.is_active:
            logger.warning("Kill switch active - blocking multi-symbol tick")
            return {
                "combined_dispatches": [],
                "symbols_processed": 0,
                "total_dispatches": 0,
                "regime": "HALTED",
                "correlation_warnings": [],
                "kill_switch_active": True
            }

        # Process all symbols in parallel
        tasks = [
            self._process_symbol_async(sym, price, account_data)
            for sym, price in zip(symbols, prices)
        ]

        # Wait for all symbols to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        combined_dispatches = []
        correlation_warnings = []
        common_regime = None
        symbols_processed = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing {symbols[i]}: {result}")
                continue

            symbols_processed += 1
            symbol = symbols[i]

            # Extract data from result
            if isinstance(result, dict):
                dispatches = result.get("dispatches", [])
                regime = result.get("regime")

                # Add symbol context to each dispatch
                for dispatch in dispatches:
                    dispatch["symbol"] = symbol
                    combined_dispatches.append(dispatch)

                # Track common regime
                if common_regime is None:
                    common_regime = regime

        # Cross-symbol correlation checks
        if combined_dispatches:
            correlation_warnings = self._check_correlation_risk(combined_dispatches)

        # Filter out correlated dispatches if warnings present
        if correlation_warnings:
            combined_dispatches = self._apply_correlation_filters(
                combined_dispatches,
                correlation_warnings
            )

        # Log combined dispatch context
        if self.trade_logger and combined_dispatches:
            self.trade_logger.log_dispatch_context(
                regime=common_regime or "UNKNOWN",
                chaos_score=0.2,  # Would need aggregation
                mandate=RiskMandate(allocation_scalar=1.0, risk_mode="STANDARD"),
                dispatches=combined_dispatches,
                symbol="MULTI"
            )

        return {
            "combined_dispatches": combined_dispatches,
            "symbols_processed": symbols_processed,
            "total_dispatches": len(combined_dispatches),
            "regime": common_regime,
            "correlation_warnings": correlation_warnings
        }

    async def _process_symbol_async(
        self,
        symbol: str,
        price: float,
        account_data: Optional[Dict] = None
    ) -> Dict:
        """
        Process a single symbol asynchronously.

        Wraps process_tick() for async execution.
        """
        # Run the blocking process_tick in an executor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.process_tick(symbol, price, account_data)
        )
        return result

    def check_symbol_correlation(self, symbol1: str, symbol2: str) -> bool:
        """
        Check if two symbols are highly correlated.

        Uses the correlation matrix to determine if trading both symbols
        would create excessive currency exposure.

        Args:
            symbol1: First symbol (e.g., 'EURUSD')
            symbol2: Second symbol (e.g., 'GBPUSD')

        Returns:
            True if symbols are highly correlated (|correlation| >= threshold)
        """
        # Check both orderings in matrix
        corr1 = self._CORRELATION_MATRIX.get((symbol1, symbol2))
        corr2 = self._CORRELATION_MATRIX.get((symbol2, symbol1))

        correlation = corr1 if corr1 is not None else corr2

        if correlation is None:
            # No correlation data - assume uncorrelated
            return False

        return abs(correlation) >= self._HIGH_CORRELATION_THRESHOLD

    def _check_correlation_risk(self, dispatches: List[Dict]) -> List[str]:
        """
        Check for correlated symbol exposure across dispatches.

        Args:
            dispatches: List of bot dispatches with 'symbol' key

        Returns:
            List of warning messages about correlated pairs
        """
        warnings = []
        symbols = [d.get("symbol") for d in dispatches if d.get("symbol")]

        # Check all pairs
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if self.check_symbol_correlation(sym1, sym2):
                    warning = f"High correlation detected: {sym1} + {sym2}"
                    warnings.append(warning)
                    logger.warning(warning)

        return warnings

    def _apply_correlation_filters(
        self,
        dispatches: List[Dict],
        warnings: List[str]
    ) -> List[Dict]:
        """
        Filter dispatches to remove correlated positions.

        Tie-breaking rules:
        1. Keep higher frequency bots
        2. Keep higher win rate
        3. Keep more experienced bots (total trades)

        Args:
            dispatches: All dispatches
            warnings: Correlation warnings

        Returns:
            Filtered dispatches with correlated positions removed
        """
        if not warnings:
            return dispatches

        # Parse warnings to get correlated pairs
        correlated_pairs = []
        for warning in warnings:
            if "High correlation detected:" in warning:
                pair = warning.replace("High correlation detected: ", "").split(" + ")
                if len(pair) == 2:
                    correlated_pairs.append(pair)

        # Group dispatches by symbol
        dispatches_by_symbol = {}
        for dispatch in dispatches:
            symbol = dispatch.get("symbol")
            if symbol:
                if symbol not in dispatches_by_symbol:
                    dispatches_by_symbol[symbol] = []
                dispatches_by_symbol[symbol].append(dispatch)

        # For each correlated pair, select winner
        to_remove = set()
        for sym1, sym2 in correlated_pairs:
            if sym1 in dispatches_by_symbol and sym2 in dispatches_by_symbol:
                # Get best dispatch from each symbol
                best1 = self._get_best_dispatch(dispatches_by_symbol[sym1])
                best2 = self._get_best_dispatch(dispatches_by_symbol[sym2])

                # Apply tie-breaking
                winner, loser = self._break_tie(best1, best2)

                # Mark loser's symbol for removal
                loser_symbol = loser.get("symbol")
                to_remove.add(loser_symbol)

        # Filter out removed symbols
        filtered = [
            d for d in dispatches
            if d.get("symbol") not in to_remove
        ]

        if to_remove:
            logger.info(f"Filtered correlated symbols: {to_remove}")

        return filtered

    def _get_best_dispatch(self, dispatches: List[Dict]) -> Dict:
        """Get the highest-scoring dispatch from a list."""
        if not dispatches:
            return {}

        return max(dispatches, key=lambda d: d.get("score", 0))

    def _break_tie(self, dispatch1: Dict, dispatch2: Dict) -> tuple:
        """
        Break tie between two correlated dispatches.

        Tie-breaking rules:
        1. Frequency rank (HFT > HIGH > MEDIUM > LOW)
        2. Win rate (higher is better)
        3. Total trades (more experience is better)

        Returns:
            (winner, loser) tuple
        """
        # Frequency ranking
        frequency_rank = {"HFT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}

        freq1 = frequency_rank.get(dispatch1.get("frequency", "LOW"), 0)
        freq2 = frequency_rank.get(dispatch2.get("frequency", "LOW"), 0)

        if freq1 != freq2:
            return (dispatch1, dispatch2) if freq1 > freq2 else (dispatch2, dispatch1)

        # Win rate comparison
        win1 = dispatch1.get("win_rate", 0.5)
        win2 = dispatch2.get("win_rate", 0.5)

        if abs(win1 - win2) > 0.01:  # Significant difference
            return (dispatch1, dispatch2) if win1 > win2 else (dispatch2, dispatch1)

        # Total trades (experience)
        trades1 = dispatch1.get("total_trades", 0)
        trades2 = dispatch2.get("total_trades", 0)

        return (dispatch1, dispatch2) if trades1 > trades2 else (dispatch2, dispatch1)
