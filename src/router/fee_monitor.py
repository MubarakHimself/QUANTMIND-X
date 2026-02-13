from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
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
            
            # Update fee tracking
            record.total_fees += fee
            record.total_trades += 1
            record.fee_burn_pct = (record.total_fees / self.account_balance) * 100 if self.account_balance > 0 else 0.0
            
            # Check if this trade triggers kill switch and persist status
            should_halt, reason = self._check_kill_switch_conditions(
                record.total_fees,
                record.total_trades,
                record.fee_burn_pct
            )
            
            if should_halt:
                record.kill_switch_activated = True
                logger.warning(f"Kill switch activated during fee recording: {reason}")
            
            session.commit()
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
                avg_fee = record.total_fees / max(1, record.total_trades)
                return FeeReport(
                    date=record.date,
                    total_fees=record.total_fees,
                    total_trades=record.total_trades,
                    fee_per_trade=avg_fee,
                    account_balance=record.account_balance,
                    fee_burn_pct=record.fee_burn_pct,
                    status="KILL_SWITCH_ACTIVE" if record.kill_switch_activated else "ACTIVE"
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
                record.kill_switch_activated = activated
                session.commit()
                logger.info(f"Updated kill switch status to {activated} for {self.account_id} on {date_str}")
            else:
                logger.warning(f"No DailyFeeTracking record found for {self.account_id} on {date_str}")

    def calculate_scalping_fee_burn(self, trades_per_hour: int = 10, hours: int = 24, fee_per_trade: float = 5.0) -> float:
        """Analyzes fee burn for HFT scenarios"""
        total_trades = trades_per_hour * hours
        total_fees = total_trades * fee_per_trade
        return (total_fees / self.account_balance) * 100
