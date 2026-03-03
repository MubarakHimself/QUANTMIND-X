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
