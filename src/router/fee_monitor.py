from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
from datetime import date, datetime, timezone
import logging

from src.database.db_manager import DBManager
from src.database.models import DailyFeeTracking

logger = logging.getLogger(__name__)

@dataclass
class FeeReport:
    date: str
    total_fees: float
    total_trades: int
    fee_per_trade: float
    account_balance: float
    fee_burn_pct: float
    status: str

class FeeMonitor:
    MAX_DAILY_FEE_PCT = 0.10  # 10% of balance
    MAX_FEE_PER_TRADE = 50.0  # $50 per trade

    def __init__(self, account_id: str, db_manager: Optional[DBManager] = None, account_balance: float = 1000.0):
        self.account_id = account_id
        self.db = db_manager or DBManager()
        self.account_balance = account_balance

    def record_trade_fee(self, bot_id: str, fee: float, trade_date: Optional[date] = None):
        """Records fee for each trade (commission + spread cost)"""
        if trade_date is None:
            trade_date = date.today()
        date_str = trade_date.strftime('%Y-%m-%d')

        with self.db.get_session() as session:
            record = session.query(DailyFeeTracking).filter_by(
                account_id=self.account_id,
                date=date_str
            ).first()
            if not record:
                record = DailyFeeTracking(
                    account_id=self.account_id,
                    date=date_str,
                    total_fees=0.0,
                    total_trades=0,
                    fee_burn_pct=0.0,
                    account_balance=self.account_balance,
                    kill_switch_activated=False
                )
                session.add(record)
            
            # Update fee tracking - use SQLAlchemy's Python-side attribute access
            # These operations work at runtime but Pylance doesn't understand SQLAlchemy ORM
            record.total_fees = float(record.total_fees) + fee  # type: ignore[assignment]
            record.total_trades = int(record.total_trades) + 1  # type: ignore[assignment]
            record.fee_burn_pct = (float(record.total_fees) / self.account_balance) * 100 if self.account_balance > 0 else 0.0  # type: ignore[assignment]
            
            # Check if this trade triggers kill switch and persist status
            should_halt, reason = self._check_kill_switch_conditions(
                float(record.total_fees),  # type: ignore[arg-type]
                int(record.total_trades),  # type: ignore[arg-type]
                float(record.fee_burn_pct)  # type: ignore[arg-type]
            )
            
            if should_halt:
                record.kill_switch_activated = True  # type: ignore[assignment]
                logger.warning(f"Kill switch activated during fee recording: {reason}")
            
            session.commit()

            # Broadcast fee update via WebSocket
            try:
                from src.api.websocket_endpoints import broadcast_fee_update
                import asyncio

                fee_data = {
                    "daily_fees": float(record.total_fees),
                    "daily_fee_burn_pct": float(record.fee_burn_pct),
                    "kill_switch_active": bool(record.kill_switch_activated),
                    "fee_breakdown": self.get_fee_breakdown_by_bot(date_str)
                }

                # Run async broadcast in sync context
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(broadcast_fee_update(fee_data))
                except RuntimeError:
                    # No running loop, create new one
                    asyncio.run(broadcast_fee_update(fee_data))
            except Exception as e:
                logger.warning(f"Failed to broadcast fee update: {e}")

        logger.info(f"Recorded fee ${fee:.2f} for bot {bot_id} on {date_str}")
    
    def _check_kill_switch_conditions(self, total_fees: float, total_trades: int, fee_burn_pct: float) -> Tuple[bool, str]:
        """Check if kill switch conditions are met"""
        avg_fee = total_fees / max(1, total_trades)
        
        if fee_burn_pct > self.MAX_DAILY_FEE_PCT * 100:
            return True, f"Daily fee burn {fee_burn_pct:.1f}% exceeds {self.MAX_DAILY_FEE_PCT*100}% limit"
        
        if avg_fee > self.MAX_FEE_PER_TRADE:
            return True, f"Fee per trade ${avg_fee:.2f} > ${self.MAX_FEE_PER_TRADE}"
        
        return False, "Fees within limits"

    def get_daily_report(self, date_str: str) -> Optional[FeeReport]:
        """Returns FeeReport for specified date"""
        with self.db.get_session() as session:
            record = session.query(DailyFeeTracking).filter_by(
                account_id=self.account_id, date=date_str
            ).first()
            if record:
                avg_fee = float(record.total_fees) / max(1, int(record.total_trades))  # type: ignore[arg-type]
                return FeeReport(
                    date=str(record.date),  # type: ignore[arg-type]
                    total_fees=float(record.total_fees),  # type: ignore[arg-type]
                    total_trades=int(record.total_trades),  # type: ignore[arg-type]
                    fee_per_trade=avg_fee,
                    account_balance=float(record.account_balance),  # type: ignore[arg-type]
                    fee_burn_pct=float(record.fee_burn_pct),  # type: ignore[arg-type]
                    status="KILL_SWITCH_ACTIVE" if bool(record.kill_switch_activated) else "ACTIVE"  # type: ignore[arg-type]
                )
        return None

    def should_halt_trading(self) -> Tuple[bool, str]:
        """Checks if kill switch should activate and persists the status"""
        today_str = date.today().strftime('%Y-%m-%d')
        report = self.get_daily_report(today_str)
        if not report:
            return False, "No fee data available"
        
        should_halt = False
        reason = "Fees within limits"
        
        if report.fee_burn_pct > self.MAX_DAILY_FEE_PCT * 100:
            should_halt = True
            reason = f"Daily fee burn {report.fee_burn_pct:.1f}% exceeds {self.MAX_DAILY_FEE_PCT*100}% limit"
        elif report.fee_per_trade > self.MAX_FEE_PER_TRADE:
            should_halt = True
            reason = f"Fee per trade ${report.fee_per_trade:.2f} > ${self.MAX_FEE_PER_TRADE}"
        
        # Update kill switch status in database if halt is triggered
        if should_halt:
            self._update_kill_switch_status(today_str, True)
            logger.warning(f"Fee kill switch activated: {reason}")
        
        return should_halt, reason
    
    def _update_kill_switch_status(self, date_str: str, activated: bool) -> None:
        """Updates the kill_switch_activated flag in DailyFeeTracking"""
        with self.db.get_session() as session:
            record = session.query(DailyFeeTracking).filter_by(
                account_id=self.account_id,
                date=date_str
            ).first()
            if record:
                record.kill_switch_activated = activated  # type: ignore[assignment]
                session.commit()
                logger.info(f"Updated kill switch status to {activated} for {self.account_id} on {date_str}")
            else:
                logger.warning(f"No DailyFeeTracking record found for {self.account_id} on {date_str}")

    def calculate_scalping_fee_burn(self, trades_per_hour: int = 10, hours: int = 24, fee_per_trade: float = 5.0) -> float:
        """Analyzes fee burn for HFT scenarios"""
        total_trades = trades_per_hour * hours
        total_fees = total_trades * fee_per_trade
        return (total_fees / self.account_balance) * 100

    def get_fee_breakdown_by_bot(self, date_str: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get fee breakdown by bot for specified date.
        Returns list of dicts with bot_id, trades, fees_paid, fee_pct

        Args:
            date_str: Date string in YYYY-MM-DD format (defaults to today)

        Returns:
            List of dicts sorted by fees_paid (highest first)
        """
        from src.database.models import TradeJournal

        if date_str is None:
            date_str = date.today().strftime('%Y-%m-%d')

        with self.db.get_session() as session:
            # Query TradeJournal for the specified date
            # Group by bot_id and aggregate commission and count
            from sqlalchemy import func

            results = session.query(
                TradeJournal.bot_id,
                func.count(TradeJournal.id).label('trades'),
                func.sum(TradeJournal.commission).label('fees_paid')
            ).filter(
                func.date(TradeJournal.timestamp) == date_str,
                TradeJournal.account_id == self.account_id
            ).group_by(
                TradeJournal.bot_id
            ).all()

            breakdown = []
            for row in results:
                bot_id = row.bot_id
                trades = row.trades or 0
                fees_paid = float(row.fees_paid or 0.0)
                # Calculate fee percentage relative to account balance
                fee_pct = (fees_paid / self.account_balance) * 100 if self.account_balance > 0 else 0.0

                breakdown.append({
                    'bot_id': bot_id,
                    'trades': trades,
                    'fees_paid': fees_paid,
                    'fee_pct': round(fee_pct, 2)
                })

            # Sort by fees_paid (highest first)
            breakdown.sort(key=lambda x: x['fees_paid'], reverse=True)

            return breakdown
