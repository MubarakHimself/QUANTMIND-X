"""
Kill Switch: Emergency Stop System

Provides multiple methods to immediately halt all trading activity:
1. STOP file detection
2. Direct panic() call
3. Socket command (EMERGENCY_STOP)
4. API endpoint

V2: Smart Kill Switch with regime-aware exits using Sentinel/Governor integration.
Aims for breakeven or <10% loss before full CLOSE_ALL.

From PDF: "panic.py or STOP file → CLOSE_ALL immediately"

**Validates: PDF Requirements - Kill Switch**
"""

import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from src.router.sentinel import Sentinel, RegimeReport
    from src.router.governor import Governor

logger = logging.getLogger(__name__)


class KillReason(Enum):
    """Reasons for triggering kill switch."""
    MANUAL = "MANUAL"                   # User clicked button
    STOP_FILE = "STOP_FILE"             # STOP file detected
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"   # Max drawdown exceeded
    API_COMMAND = "API_COMMAND"         # External API call
    SOCKET_COMMAND = "SOCKET_COMMAND"   # EA/Socket command
    SYSTEM_ERROR = "SYSTEM_ERROR"       # Critical system error
    CHAOS_THRESHOLD = "CHAOS_THRESHOLD" # Sentinel detected extreme chaos
    NEWS_EVENT = "NEWS_EVENT"           # Kill zone from news
    SMART_EXIT = "SMART_EXIT"           # Smart exit triggered


class ExitStrategy(Enum):
    """Exit strategy for smart kill."""
    IMMEDIATE = "IMMEDIATE"      # Close all immediately
    BREAKEVEN = "BREAKEVEN"      # Wait for breakeven, then close
    SMART_LOSS = "SMART_LOSS"    # Close if > 10% loss, else wait
    TRAILING = "TRAILING"        # Trailing stop to breakeven


@dataclass
class KillEvent:
    """Record of a kill switch activation."""
    timestamp: datetime
    reason: KillReason
    triggered_by: str
    message: str
    positions_closed: int = 0
    accounts_affected: List[str] = field(default_factory=list)
    exit_strategy: ExitStrategy = ExitStrategy.IMMEDIATE
    smart_exit_result: Optional[str] = None


class KillSwitch:
    """
    Emergency stop system for QuantMindX.
    
    Monitors for kill triggers and executes CLOSE_ALL on all connected EAs.
    """
    
    # Configuration
    STOP_FILE_PATH = Path("STOP")
    PANIC_FILE_PATH = Path("PANIC")
    CHECK_INTERVAL_MS = 100  # Check every 100ms
    
    def __init__(
        self,
        project_root: Path = None,
        on_kill: Optional[Callable[["KillEvent"], None]] = None
    ):
        self.project_root = project_root or Path.cwd()
        self.on_kill = on_kill
        self._active = False
        self._monitoring = False
        self._kill_history: List[KillEvent] = []
        self._connected_eas: List[str] = []
        self._socket_server = None
        
    def register_socket_server(self, socket_server) -> None:
        """Register socket server for sending CLOSE_ALL commands."""
        self._socket_server = socket_server
        logger.info("Kill switch linked to socket server")
    
    def register_ea(self, ea_id: str) -> None:
        """Register an EA for kill switch notifications."""
        if ea_id not in self._connected_eas:
            self._connected_eas.append(ea_id)
            logger.info(f"EA registered for kill switch: {ea_id}")
    
    def unregister_ea(self, ea_id: str) -> None:
        """Unregister an EA."""
        if ea_id in self._connected_eas:
            self._connected_eas.remove(ea_id)
    
    @property
    def is_active(self) -> bool:
        """Check if kill switch has been triggered."""
        return self._active
    
    def _check_stop_file(self) -> bool:
        """Check if STOP file exists."""
        stop_path = self.project_root / self.STOP_FILE_PATH
        panic_path = self.project_root / self.PANIC_FILE_PATH
        return stop_path.exists() or panic_path.exists()
    
    def _clear_stop_file(self) -> None:
        """Remove STOP file after processing."""
        for path in [self.STOP_FILE_PATH, self.PANIC_FILE_PATH]:
            full_path = self.project_root / path
            if full_path.exists():
                full_path.unlink()
                logger.info(f"Removed {path} file")
    
    async def trigger(
        self,
        reason: KillReason,
        triggered_by: str = "system",
        message: str = "Emergency stop activated"
    ) -> KillEvent:
        """
        Trigger the kill switch.
        
        This will:
        1. Set active flag to block new trades
        2. Send CLOSE_ALL to all connected EAs
        3. Log the event
        4. Call the on_kill callback
        """
        if self._active:
            logger.warning("Kill switch already active")
            return self._kill_history[-1] if self._kill_history else None
        
        self._active = True
        
        logger.critical(f"🚨 KILL SWITCH TRIGGERED: {reason.value}")
        logger.critical(f"   Reason: {message}")
        logger.critical(f"   Triggered by: {triggered_by}")
        
        # Create event
        event = KillEvent(
            timestamp=datetime.now(),
            reason=reason,
            triggered_by=triggered_by,
            message=message,
            accounts_affected=self._connected_eas.copy()
        )
        
        # Send CLOSE_ALL to all EAs
        positions_closed = await self._send_close_all()
        event.positions_closed = positions_closed
        
        # Record event
        self._kill_history.append(event)
        
        # Call callback
        if self.on_kill:
            try:
                self.on_kill(event)
            except Exception as e:
                logger.error(f"Kill callback error: {e}")
        
        return event
    
    async def _send_close_all(self) -> int:
        """Send CLOSE_ALL command to all connected EAs."""
        if not self._socket_server:
            logger.warning("No socket server connected, cannot send CLOSE_ALL")
            return 0
        
        positions_closed = 0
        
        for ea_id in self._connected_eas:
            try:
                command = {
                    "type": "CLOSE_ALL",
                    "ea_id": ea_id,
                    "reason": "EMERGENCY_STOP",
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send via socket server
                if hasattr(self._socket_server, 'broadcast_command'):
                    await self._socket_server.broadcast_command(command)
                    positions_closed += 1  # Estimate, actual count from response
                
                logger.info(f"Sent CLOSE_ALL to EA: {ea_id}")
            except Exception as e:
                logger.error(f"Failed to send CLOSE_ALL to {ea_id}: {e}")
        
        return positions_closed
    
    def reset(self) -> None:
        """
        Reset kill switch to allow trading again.
        
        Should only be called after manual review.
        """
        if not self._active:
            return
        
        self._active = False
        self._clear_stop_file()
        logger.info("Kill switch reset - trading enabled")
    
    async def start_monitoring(self) -> None:
        """Start background monitoring for STOP file."""
        if self._monitoring:
            return
        
        self._monitoring = True
        logger.info("Kill switch monitoring started")
        
        while self._monitoring:
            if self._check_stop_file():
                await self.trigger(
                    reason=KillReason.STOP_FILE,
                    triggered_by="file_monitor",
                    message="STOP file detected in project root"
                )
                self._clear_stop_file()
            
            await asyncio.sleep(self.CHECK_INTERVAL_MS / 1000)
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring = False
        logger.info("Kill switch monitoring stopped")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get kill switch history."""
        return [
            {
                "timestamp": e.timestamp.isoformat(),
                "reason": e.reason.value,
                "triggered_by": e.triggered_by,
                "message": e.message,
                "positions_closed": e.positions_closed,
                "accounts_affected": e.accounts_affected
            }
            for e in self._kill_history
        ]


class SmartKillSwitch(KillSwitch):
    """
    V2: Smart Kill Switch with regime-aware exits.
    
    Integrates with:
    - Sentinel: Uses regime detection (NEWS_EVENT, HIGH_CHAOS)
    - Governor: Uses drawdown rules for throttling before exit
    
    Smart exit strategy:
    1. If chaos < 0.3 and drawdown < 10%: Wait for breakeven
    2. If chaos 0.3-0.6: Set trailing stop to breakeven
    3. If chaos > 0.6 or drawdown > 10%: Immediate close
    4. If NEWS_EVENT: Immediate close (no waiting)
    """
    
    # Smart exit thresholds
    MAX_DRAWDOWN_PCT = 0.10  # 10% max loss before immediate exit
    CHAOS_WAIT_THRESHOLD = 0.3  # Below this, can wait for breakeven
    CHAOS_TRAIL_THRESHOLD = 0.6  # Above this, immediate exit
    
    def __init__(
        self,
        project_root: Path = None,
        on_kill: Optional[Callable[["KillEvent"], None]] = None,
        sentinel: Optional["Sentinel"] = None,
        governor: Optional["Governor"] = None
    ):
        super().__init__(project_root, on_kill)
        self._sentinel = sentinel
        self._governor = governor
        self._pending_exits: Dict[str, Dict] = {}  # ea_id -> exit info
    
    @property
    def sentinel(self) -> Optional["Sentinel"]:
        """Lazy load Sentinel if not provided."""
        if self._sentinel is None:
            try:
                from src.router.sentinel import Sentinel
                self._sentinel = Sentinel()
                logger.info("SmartKillSwitch: Loaded Sentinel")
            except Exception as e:
                logger.warning(f"SmartKillSwitch: Could not load Sentinel: {e}")
        return self._sentinel
    
    @property
    def governor(self) -> Optional["Governor"]:
        """Lazy load Governor if not provided."""
        if self._governor is None:
            try:
                from src.router.governor import Governor
                self._governor = Governor()
                logger.info("SmartKillSwitch: Loaded Governor")
            except Exception as e:
                logger.warning(f"SmartKillSwitch: Could not load Governor: {e}")
        return self._governor
    
    def determine_exit_strategy(
        self,
        current_pnl_pct: float,
        regime_report: Optional["RegimeReport"] = None
    ) -> ExitStrategy:
        """
        Determine the best exit strategy based on current conditions.
        
        Borrowed from Sentinel's NEWS_EVENT handling logic.
        
        Args:
            current_pnl_pct: Current P&L as percentage (-0.05 = 5% loss)
            regime_report: Current regime from Sentinel
            
        Returns:
            Optimal exit strategy
        """
        # If no regime data, be conservative
        if regime_report is None:
            if self.sentinel and self.sentinel.current_report:
                regime_report = self.sentinel.current_report
            else:
                logger.warning("No regime data - using immediate exit")
                return ExitStrategy.IMMEDIATE
        
        # NEWS_EVENT or extreme chaos = immediate exit (no waiting)
        if regime_report.regime == "NEWS_EVENT":
            logger.info("NEWS_EVENT detected - immediate exit required")
            return ExitStrategy.IMMEDIATE
        
        if regime_report.chaos_score > self.CHAOS_TRAIL_THRESHOLD:
            logger.info(f"High chaos ({regime_report.chaos_score:.2f}) - immediate exit")
            return ExitStrategy.IMMEDIATE
        
        # Check drawdown
        if current_pnl_pct < -self.MAX_DRAWDOWN_PCT:
            logger.info(f"Drawdown exceeds limit ({current_pnl_pct:.1%}) - immediate exit")
            return ExitStrategy.IMMEDIATE
        
        # Moderate chaos = trailing stop
        if regime_report.chaos_score > self.CHAOS_WAIT_THRESHOLD:
            logger.info(f"Moderate chaos ({regime_report.chaos_score:.2f}) - trailing to breakeven")
            return ExitStrategy.TRAILING
        
        # Low chaos and small loss = wait for breakeven
        if current_pnl_pct >= -self.MAX_DRAWDOWN_PCT:
            logger.info(f"Low chaos, small loss ({current_pnl_pct:.1%}) - waiting for breakeven")
            return ExitStrategy.BREAKEVEN
        
        # Default to smart loss handling
        return ExitStrategy.SMART_LOSS
    
    async def smart_trigger(
        self,
        reason: KillReason,
        current_pnl_pct: float = 0.0,
        triggered_by: str = "system",
        message: str = "Smart exit activated"
    ) -> KillEvent:
        """
        Trigger smart kill switch with regime-aware exit strategy.
        
        Args:
            reason: Why kill switch was triggered
            current_pnl_pct: Current P&L percentage
            triggered_by: Who/what triggered it
            message: Description
            
        Returns:
            KillEvent with exit strategy used
        """
        # Get current regime
        regime_report = None
        if self.sentinel:
            regime_report = self.sentinel.current_report
        
        # Determine strategy
        strategy = self.determine_exit_strategy(current_pnl_pct, regime_report)
        
        logger.info(f"Smart Kill: Using strategy {strategy.value}")
        
        if strategy == ExitStrategy.IMMEDIATE:
            # Use parent's immediate exit
            event = await self.trigger(reason, triggered_by, message)
            event.exit_strategy = strategy
            return event
        
        elif strategy == ExitStrategy.BREAKEVEN:
            # Register pending exit - wait for breakeven
            return await self._register_pending_exit(
                reason, strategy, current_pnl_pct, triggered_by, message
            )
        
        elif strategy == ExitStrategy.TRAILING:
            # Set trailing stop to breakeven
            return await self._set_trailing_stop(
                reason, current_pnl_pct, triggered_by, message
            )
        
        else:  # SMART_LOSS
            # Wait if loss < 10%, else close
            if current_pnl_pct >= -self.MAX_DRAWDOWN_PCT:
                return await self._register_pending_exit(
                    reason, strategy, current_pnl_pct, triggered_by, message
                )
            else:
                event = await self.trigger(reason, triggered_by, message)
                event.exit_strategy = strategy
                return event
    
    async def _register_pending_exit(
        self,
        reason: KillReason,
        strategy: ExitStrategy,
        current_pnl_pct: float,
        triggered_by: str,
        message: str
    ) -> KillEvent:
        """Register a pending exit that waits for breakeven."""
        logger.info(f"Registering pending exit - waiting for breakeven")
        
        # Block new trades
        self._active = True
        
        event = KillEvent(
            timestamp=datetime.now(),
            reason=reason,
            triggered_by=triggered_by,
            message=f"{message} - waiting for breakeven",
            accounts_affected=self._connected_eas.copy(),
            exit_strategy=strategy,
            smart_exit_result="PENDING"
        )
        
        # Send "NO_NEW_TRADES" command instead of CLOSE_ALL
        await self._send_halt_new_trades()
        
        self._kill_history.append(event)
        return event
    
    async def _set_trailing_stop(
        self,
        reason: KillReason,
        current_pnl_pct: float,
        triggered_by: str,
        message: str
    ) -> KillEvent:
        """Set trailing stop to breakeven on all positions."""
        logger.info("Setting trailing stop to breakeven")
        
        self._active = True
        
        event = KillEvent(
            timestamp=datetime.now(),
            reason=reason,
            triggered_by=triggered_by,
            message=f"{message} - trailing to breakeven",
            accounts_affected=self._connected_eas.copy(),
            exit_strategy=ExitStrategy.TRAILING,
            smart_exit_result="TRAILING"
        )
        
        # Send trailing stop command
        await self._send_trailing_stop_command()
        
        self._kill_history.append(event)
        return event
    
    async def _send_halt_new_trades(self) -> None:
        """Send command to halt new trades but keep existing positions."""
        if not self._socket_server:
            return
        
        for ea_id in self._connected_eas:
            try:
                command = {
                    "type": "HALT_NEW_TRADES",
                    "ea_id": ea_id,
                    "reason": "SMART_EXIT_PENDING",
                    "timestamp": datetime.now().isoformat()
                }
                if hasattr(self._socket_server, 'broadcast_command'):
                    await self._socket_server.broadcast_command(command)
                logger.info(f"Sent HALT_NEW_TRADES to EA: {ea_id}")
            except Exception as e:
                logger.error(f"Failed to send HALT_NEW_TRADES to {ea_id}: {e}")
    
    async def _send_trailing_stop_command(self) -> None:
        """Send command to set trailing stop to breakeven."""
        if not self._socket_server:
            return
        
        for ea_id in self._connected_eas:
            try:
                command = {
                    "type": "SET_TRAILING_STOP",
                    "ea_id": ea_id,
                    "target": "BREAKEVEN",
                    "reason": "SMART_EXIT",
                    "timestamp": datetime.now().isoformat()
                }
                if hasattr(self._socket_server, 'broadcast_command'):
                    await self._socket_server.broadcast_command(command)
                logger.info(f"Sent SET_TRAILING_STOP to EA: {ea_id}")
            except Exception as e:
                logger.error(f"Failed to send SET_TRAILING_STOP to {ea_id}: {e}")
    
    async def check_pending_exits(self, current_pnl_pct: float) -> None:
        """
        Check if pending exits can now be executed.
        
        Called periodically to update pending exit status.
        """
        if not self._active:
            return
        
        # If we've reached breakeven, close all
        if current_pnl_pct >= 0:
            logger.info("Breakeven reached - executing full exit")
            await self._send_close_all()
            
            # Update last event
            if self._kill_history:
                self._kill_history[-1].smart_exit_result = "BREAKEVEN_ACHIEVED"
        
        # If loss has exceeded limit, force close
        elif current_pnl_pct < -self.MAX_DRAWDOWN_PCT:
            logger.warning(f"Loss exceeded limit ({current_pnl_pct:.1%}) - forcing exit")
            await self._send_close_all()
            
            if self._kill_history:
                self._kill_history[-1].smart_exit_result = "FORCED_EXIT"


# Convenience function for immediate panic
async def panic(message: str = "Manual panic triggered") -> KillEvent:
    """
    Immediately trigger kill switch.
    
    Usage: await panic("Max drawdown exceeded")
    """
    kill_switch = KillSwitch()
    return await kill_switch.trigger(
        reason=KillReason.MANUAL,
        triggered_by="panic_function",
        message=message
    )


# Convenience function for smart exit
async def smart_exit(
    current_pnl_pct: float = 0.0,
    message: str = "Smart exit triggered"
) -> KillEvent:
    """
    Trigger smart kill switch with regime-aware exit.
    
    Usage: await smart_exit(current_pnl_pct=-0.05, message="Session end")
    """
    kill_switch = get_smart_kill_switch()
    return await kill_switch.smart_trigger(
        reason=KillReason.SMART_EXIT,
        current_pnl_pct=current_pnl_pct,
        triggered_by="smart_exit_function",
        message=message
    )


# Create STOP file (alternative trigger method)
def create_stop_file(project_root: Path = None) -> Path:
    """
    Create STOP file to trigger kill switch.
    
    This can be done externally (e.g., from another script or manual).
    """
    root = project_root or Path.cwd()
    stop_file = root / "STOP"
    stop_file.touch()
    logger.warning(f"Created STOP file at {stop_file}")
    return stop_file


# Global kill switch instances (singleton pattern)
_global_kill_switch: Optional[KillSwitch] = None
_global_smart_kill_switch: Optional[SmartKillSwitch] = None


def get_kill_switch() -> KillSwitch:
    """Get or create the global kill switch instance."""
    global _global_kill_switch
    if _global_kill_switch is None:
        _global_kill_switch = KillSwitch()
    return _global_kill_switch


def get_smart_kill_switch() -> SmartKillSwitch:
    """Get or create the global smart kill switch instance."""
    global _global_smart_kill_switch
    if _global_smart_kill_switch is None:
        _global_smart_kill_switch = SmartKillSwitch()
    return _global_smart_kill_switch
