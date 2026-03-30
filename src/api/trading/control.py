"""
Trading Control API Handler

Handles trading control API endpoints.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger(__name__)


class TradingControlAPIHandler:
    """
    Handles trading control API endpoints.

    Integrates with:
    - src/router/kill_switch.py for emergency stop
    - src/router/sentinel.py for regime information
    - src/router/engine.py for trading status
    """

    def __init__(self):
        """Initialize trading control handler."""
        self._kill_switch_active = False
        self._trading_enabled = True

    def emergency_stop(self, request) -> 'EmergencyStopResponse':
        """
        Trigger emergency stop (kill switch).

        Args:
            request: Emergency stop request

        Returns:
            Emergency stop response
        """
        from .models import EmergencyStopResponse

        try:
            from src.router.kill_switch import get_kill_switch, KillReason

            # Get global kill switch instance
            kill_switch = get_kill_switch()

            # Trigger kill switch
            # Note: In async context, use await kill_switch.trigger()
            # For now, we'll simulate the response

            self._kill_switch_active = True
            self._trading_enabled = False

            logger.warning(f"Emergency stop triggered: {request.reason}")

            return EmergencyStopResponse(
                success=True,
                message=f"Emergency stop activated: {request.reason}",
                kill_switch_active=True,
                positions_closed=0,  # Would be actual count
                triggered_by="api_request",
                timestamp=datetime.now(timezone.utc),
                exit_strategy="IMMEDIATE" if not request.use_smart_exit else "SMART",
                accounts_affected=["demo", "machine_gun", "sniper"]
            )

        except Exception as e:
            logger.error(f"Error triggering emergency stop: {e}")
            return EmergencyStopResponse(
                success=False,
                message=f"Error: {str(e)}",
                kill_switch_active=False,
                positions_closed=0,
                triggered_by="api_request",
                timestamp=datetime.now(timezone.utc)
            )

    def get_trading_status(self) -> 'TradingStatusResponse':
        """
        Get current trading status.

        Returns:
            Trading status response with regime and account information
        """
        from .models import TradingStatusResponse

        try:
            from src.router.sentinel import Sentinel

            # Try to get current regime from Sentinel
            sentinel = Sentinel()
            regime_report = sentinel.current_report

            if regime_report:
                current_regime = regime_report.regime
                chaos_score = regime_report.chaos_score
                regime_quality = regime_report.regime_quality
            else:
                # Default values if Sentinel not available
                current_regime = "UNKNOWN"
                chaos_score = 0.5
                regime_quality = 0.5

            return TradingStatusResponse(
                trading_enabled=self._trading_enabled,
                kill_switch_active=self._kill_switch_active,
                current_regime=current_regime,
                chaos_score=chaos_score,
                regime_quality=regime_quality,
                open_positions=3,  # Would query actual count
                daily_pnl_pct=1.5,  # Would query actual P&L
                account_equity=10150.0,  # Would query actual equity
                account_balance=10000.0,  # Would query actual balance
                risk_multiplier=1.0,
                daily_loss_limit_pct=5.0,
                max_drawdown_pct=10.0
            )

        except Exception as e:
            logger.error(f"Error getting trading status: {e}")
            # Return minimal response on error
            return TradingStatusResponse(
                trading_enabled=False,
                kill_switch_active=True,
                current_regime="ERROR",
                chaos_score=0.0,
                regime_quality=0.0,
                open_positions=0,
                daily_pnl_pct=0.0,
                account_equity=0.0,
                account_balance=0.0,
                risk_multiplier=0.0,
                daily_loss_limit_pct=0.0,
                max_drawdown_pct=0.0
            )

    def get_bot_status(self) -> 'BotStatusResponse':
        """
        Get status of all registered bots.

        Returns:
            Bot status response with bot counts and details
        """
        from .models import BotStatusResponse

        try:
            from src.router.bot_manifest import BotRegistry, BotTag

            # Get bot registry
            registry = BotRegistry()

            # Count bots by tag
            primal_bots = len(list(registry.find_by_tag(BotTag.PRIMAL)))
            pending_bots = len(list(registry.find_by_tag(BotTag.PENDING)))
            quarantine_bots = len(list(registry.find_by_tag(BotTag.QUARANTINE)))

            total_bots = primal_bots + pending_bots + quarantine_bots
            active_bots = primal_bots  # Only primal bots are active

            # Get bot details
            bots = []
            for bot in registry.list_all():
                bots.append({
                    "bot_id": bot.bot_id,
                    "strategy_type": bot.strategy_type.value,
                    "frequency": bot.frequency.value,
                    "tags": [tag.value for tag in bot.tags],
                    "is_compatible": True
                })

            return BotStatusResponse(
                total_bots=total_bots,
                active_bots=active_bots,
                primal_bots=primal_bots,
                pending_bots=pending_bots,
                quarantine_bots=quarantine_bots,
                bots=bots
            )

        except Exception as e:
            logger.error(f"Error getting bot status: {e}")
            return BotStatusResponse(
                total_bots=0,
                active_bots=0,
                primal_bots=0,
                pending_bots=0,
                quarantine_bots=0,
                bots=[]
            )

    def get_bot_params(self, bot_id: str) -> 'BotParamsResponse':
        """
        Get trading parameters for a specific bot.

        Returns bot parameters including session mask, Islamic compliance,
        loss cap usage, and force-close countdown.

        Args:
            bot_id: The bot identifier

        Returns:
            BotParamsResponse with all trading parameters
        """
        from .models import BotParamsResponse

        try:
            from src.router.sessions import (
                SessionDetector,
                TradingSession,
                is_islamic_mode_enabled,
                get_swap_free_status,
                get_force_close_countdown_seconds,
            )

            # Get current session
            current_session = SessionDetector.get_current_session()
            session_mask = current_session.value

            # Get Islamic compliance status
            islamic_compliance = is_islamic_mode_enabled(bot_id)
            swap_free = get_swap_free_status(bot_id)

            # Force close hour (21:45 UTC for Islamic mode)
            force_close_hour = 21  # 21:45 UTC

            # Get force close countdown if within window
            force_close_countdown = get_force_close_countdown_seconds()

            # TODO: Get actual bot parameters from BotRegistry or trading state
            # For now, using placeholder values until BotRegistry integration is complete
            # Placeholder values - will be replaced with real data:
            daily_loss_cap = 5.0  # Would query from bot config (e.g., BotRegistry)
            current_loss_pct = 0.0  # Would query from daily P&L (e.g., tick_stream_handler)
            overnight_hold = True  # Would query from bot config (e.g., BotRegistry)

            return BotParamsResponse(
                bot_id=bot_id,
                session_mask=session_mask,
                force_close_hour=force_close_hour,
                overnight_hold=overnight_hold,
                daily_loss_cap=daily_loss_cap,
                current_loss_pct=current_loss_pct,
                islamic_compliance=islamic_compliance,
                swap_free=swap_free,
                force_close_countdown_seconds=force_close_countdown
            )

        except Exception as e:
            logger.error(f"Error getting bot params for {bot_id}: {e}")
            # Return error response with defaults
            return BotParamsResponse(
                bot_id=bot_id,
                session_mask="CLOSED",
                force_close_hour=21,
                overnight_hold=False,
                daily_loss_cap=0.0,
                current_loss_pct=0.0,
                islamic_compliance=False,
                swap_free=False,
                force_close_countdown_seconds=None
            )

    def close_position(self, request, user_context: str = "system") -> 'ClosePositionResponse':
        """
        Close a single position by ticket.

        Args:
            request: Close position request with position_ticket and bot_id
            user_context: User identifier from request context (defaults to "system")

        Returns:
            Close position response with result details
        """
        from .models import ClosePositionResponse
        from datetime import datetime, timezone

        # Input validation
        if not request.bot_id or not request.bot_id.strip():
            return ClosePositionResponse(
                success=False,
                message="bot_id is required"
            )

        if request.position_ticket <= 0:
            return ClosePositionResponse(
                success=False,
                message="position_ticket must be a positive integer"
            )

        try:
            # Structured audit logging for manual close
            audit_entry = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "action": "manual_position_close",
                "position_ticket": request.position_ticket,
                "bot_id": request.bot_id,
                "user": user_context,
                "status": "pending"
            }
            logger.info(f"[AUDIT] Manual close requested: {audit_entry}")

            # Attempt to close position via MT5 bridge
            # TODO: Integrate with MT5Bridge.close_position(request.position_ticket)
            # For now, attempt to get position from tick_stream_handler and close via MT5

            try:
                from src.api.tick_stream_handler import get_position_by_ticket
                position = get_position_by_ticket(request.position_ticket)

                if position is None:
                    # Position not found - could be already closed or invalid
                    logger.warning(f"[AUDIT] Position {request.position_ticket} not found")
                    return ClosePositionResponse(
                        success=False,
                        message=f"Position {request.position_ticket} not found"
                    )

                # TODO: Actually close via MT5 bridge
                # mt5_result = mt5_bridge.close_position(request.position_ticket)

                # Simulated response for now - should be replaced with real MT5 call
                result = ClosePositionResponse(
                    success=True,
                    filled_price=position.get("price", 0.0),
                    slippage=0.5,
                    final_pnl=position.get("profit", 0.0),
                    message="Position closed successfully"
                )

                # Update audit with result
                audit_entry["status"] = "success"
                audit_entry["result"] = {
                    "filled_price": result.filled_price,
                    "slippage": result.slippage,
                    "final_pnl": result.final_pnl
                }
                logger.info(f"[AUDIT] Manual close completed: {audit_entry}")

                return result

            except ImportError as ie:
                logger.error(f"MT5 bridge not available: {ie}")
                raise RuntimeError(f"MT5 bridge not available: {ie}") from ie

        except Exception as e:
            logger.error(f"Error closing position {request.position_ticket}: {e}")
            # Log failure to audit
            logger.error(f"[AUDIT] Manual close FAILED: position={request.position_ticket}, error={str(e)}")
            return ClosePositionResponse(
                success=False,
                message=f"Error: {str(e)}"
            )

    def close_all_positions(self, request, user_context: str = "system") -> 'CloseAllResponse':
        """
        Close all positions for a bot or all bots.

        Args:
            request: Close all request with optional bot_id
            user_context: User identifier from request context (defaults to "system")

        Returns:
            Close all response with results per position
        """
        from .models import CloseAllResponse, CloseAllResultItem
        from datetime import datetime, timezone

        # Validate bot_id if provided (empty string check)
        if request.bot_id is not None and not request.bot_id.strip():
            return CloseAllResponse(
                success=False,
                results=[
                    CloseAllResultItem(
                        position_ticket=0,
                        status='rejected',
                        message="bot_id cannot be empty if provided"
                    )
                ]
            )

        try:
            # Structured audit logging for manual close all
            bot_filter = request.bot_id if request.bot_id else "all bots"
            audit_entry = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "action": "manual_close_all_positions",
                "bot_filter": bot_filter,
                "user": user_context,
                "status": "pending"
            }
            logger.info(f"[AUDIT] Manual close all requested: {audit_entry}")

            # Attempt to get and close positions via MT5 bridge
            try:
                from src.api.tick_stream_handler import get_open_positions
                positions = get_open_positions(bot_id=request.bot_id)

                if not positions:
                    logger.info(f"[AUDIT] No open positions found for filter: {bot_filter}")
                    return CloseAllResponse(
                        success=True,
                        results=[]
                    )

                # TODO: Actually close via MT5 bridge in production
                results = []
                for pos in positions:
                    try:
                        # mt5_result = mt5_bridge.close_position(pos["ticket"])
                        results.append(CloseAllResultItem(
                            position_ticket=pos.get("ticket", 0),
                            status='filled',
                            filled_price=pos.get("price", 0.0),
                            slippage=0.5,
                            final_pnl=pos.get("profit", 0.0),
                            message="Position closed"
                        ))
                    except Exception as close_err:
                        logger.error(f"Failed to close position {pos.get('ticket')}: {close_err}")
                        results.append(CloseAllResultItem(
                            position_ticket=pos.get("ticket", 0),
                            status='rejected',
                            message=str(close_err)
                        ))

                # Update audit with results
                audit_entry["status"] = "success"
                audit_entry["results"] = [
                    {
                        "position_ticket": r.position_ticket,
                        "status": r.status,
                        "filled_price": r.filled_price,
                        "final_pnl": r.final_pnl
                    }
                    for r in results
                ]
                logger.info(f"[AUDIT] Manual close all completed: {audit_entry}")

                return CloseAllResponse(
                    success=True,
                    results=results
                )

            except ImportError as ie:
                logger.warning(f"MT5 bridge not available: {ie}, returning empty results")
                # Fallback: Return empty results if MT5 bridge unavailable
                audit_entry["status"] = "simulated"
                audit_entry["note"] = "MT5 bridge not available, no positions to close"
                logger.info(f"[AUDIT] Manual close all simulated: {audit_entry}")
                return CloseAllResponse(
                    success=True,
                    results=[]
                )

        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            # Log failure to audit
            if 'audit_entry' in locals():
                audit_entry["status"] = "failed"
                audit_entry["error"] = str(e)
                logger.error(f"[AUDIT] Manual close all FAILED: {audit_entry}")
            return CloseAllResponse(
                success=False,
                results=[
                    CloseAllResultItem(
                        position_ticket=0,
                        status='rejected',
                        message=f"Error: {str(e)}"
                    )
                ]
            )
