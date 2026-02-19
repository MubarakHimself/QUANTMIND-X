"""
Trade Journal Service
=====================
Persistent trade logging and journaling system.

Features:
- SQLite database for trade storage
- Automatic trade capture from MT5
- Custom annotations and notes
- Performance analytics
- Export to CSV/JSON
- Trade replay capability
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Optional, List

import MetaTrader5 as mt5

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class TradeDirection(str, Enum):
    """Trade direction."""
    BUY = "buy"
    SELL = "sell"


class TradeStatus(str, Enum):
    """Trade status."""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class JournalEntry:
    """A trade journal entry with metadata and annotations."""
    
    # Core trade data
    id: Optional[int] = None
    ticket: int = 0
    symbol: str = ""
    direction: TradeDirection = TradeDirection.BUY
    volume: float = 0.0
    
    # Prices
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Timing
    entry_time: str = ""
    exit_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    
    # Results
    profit: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    net_profit: float = 0.0
    pips: float = 0.0
    
    # EA/Strategy info
    magic_number: int = 0
    strategy_name: str = ""
    timeframe: str = ""
    
    # Journal annotations
    setup_type: str = ""  # e.g., "breakout", "pullback", "reversal"
    notes: str = ""
    screenshots: List[str] = field(default_factory=list)  # Paths to screenshot files
    tags: List[str] = field(default_factory=list)
    rating: int = 0  # 1-5 self-assessment
    mistakes: List[str] = field(default_factory=list)
    lessons: str = ""
    
    # Market context
    session: str = ""  # "london", "new_york", "asian"
    day_of_week: str = ""
    news_events: str = ""
    
    status: TradeStatus = TradeStatus.OPEN
    created_at: str = ""
    updated_at: str = ""


# ============================================================================
# Database Manager
# ============================================================================

class TradeJournal:
    """
    Trade journaling system with SQLite persistence.
    
    Usage:
        journal = TradeJournal()
        
        # Sync trades from MT5
        journal.sync_from_mt5(days=7)
        
        # Add notes to a trade
        journal.annotate_trade(
            ticket=12345,
            notes="Entered on fair value gap",
            setup_type="FVG",
            rating=4
        )
        
        # Query trades
        trades = journal.get_trades(symbol="EURUSD", days=30)
        
        # Get analytics
        stats = journal.get_performance_stats(days=30)
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize Trade Journal.
        
        Args:
            db_path: Path to SQLite database file.
                    Defaults to ~/.quantmindx/trade_journal.db
        """
        if db_path is None:
            db_path = os.path.join(
                os.path.expanduser("~"),
                ".quantmindx",
                "trade_journal.db"
            )
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        logger.info(f"TradeJournal initialized at {self.db_path}")
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection context."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Main trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket INTEGER UNIQUE,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    volume REAL NOT NULL,
                    
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    duration_minutes INTEGER,
                    
                    profit REAL DEFAULT 0,
                    commission REAL DEFAULT 0,
                    swap REAL DEFAULT 0,
                    net_profit REAL DEFAULT 0,
                    pips REAL DEFAULT 0,
                    
                    magic_number INTEGER DEFAULT 0,
                    strategy_name TEXT DEFAULT '',
                    timeframe TEXT DEFAULT '',
                    
                    setup_type TEXT DEFAULT '',
                    notes TEXT DEFAULT '',
                    screenshots TEXT DEFAULT '[]',
                    tags TEXT DEFAULT '[]',
                    rating INTEGER DEFAULT 0,
                    mistakes TEXT DEFAULT '[]',
                    lessons TEXT DEFAULT '',
                    
                    session TEXT DEFAULT '',
                    day_of_week TEXT DEFAULT '',
                    news_events TEXT DEFAULT '',
                    
                    status TEXT DEFAULT 'open',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_magic ON trades(magic_number)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            
            logger.debug("Database schema initialized")
    
    def sync_from_mt5(self, days: int = 30) -> dict:
        """
        Sync trades from MT5 history.
        
        Args:
            days: Number of days to sync.
            
        Returns:
            Dictionary with sync results.
        """
        from_date = datetime.now() - timedelta(days=days)
        
        # Get deals from MT5
        deals = mt5.history_deals_get(from_date, datetime.now())
        
        if not deals:
            return {"synced": 0, "errors": []}
        
        synced = 0
        errors = []
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for deal in deals:
                # Skip balance operations
                if deal.type in [mt5.DEAL_TYPE_BALANCE, mt5.DEAL_TYPE_CREDIT]:
                    continue
                
                try:
                    # Check if exists
                    cursor.execute("SELECT id FROM trades WHERE ticket = ?", (deal.ticket,))
                    if cursor.fetchone():
                        continue  # Already exists
                    
                    # Determine direction
                    direction = TradeDirection.BUY.value if deal.type == mt5.DEAL_TYPE_BUY else TradeDirection.SELL.value
                    
                    # Create entry
                    now = datetime.now().isoformat()
                    entry_time = datetime.fromtimestamp(deal.time).isoformat()
                    
                    cursor.execute("""
                        INSERT INTO trades (
                            ticket, symbol, direction, volume,
                            entry_price, profit, commission, swap,
                            net_profit, magic_number, entry_time,
                            status, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        deal.ticket, deal.symbol, direction, deal.volume,
                        deal.price, deal.profit, deal.commission, deal.swap,
                        deal.profit + deal.commission + deal.swap,
                        deal.magic, entry_time,
                        TradeStatus.CLOSED.value, now, now
                    ))
                    synced += 1
                    
                except Exception as e:
                    errors.append(f"Deal {deal.ticket}: {str(e)}")
        
        logger.info(f"Synced {synced} trades from MT5")
        return {"synced": synced, "errors": errors}
    
    def add_trade(self, entry: JournalEntry) -> int:
        """
        Manually add a trade to the journal.
        
        Args:
            entry: JournalEntry object.
            
        Returns:
            ID of inserted trade.
        """
        now = datetime.now().isoformat()
        entry.created_at = now
        entry.updated_at = now
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    ticket, symbol, direction, volume,
                    entry_price, exit_price, stop_loss, take_profit,
                    entry_time, exit_time, duration_minutes,
                    profit, commission, swap, net_profit, pips,
                    magic_number, strategy_name, timeframe,
                    setup_type, notes, screenshots, tags, rating, mistakes, lessons,
                    session, day_of_week, news_events,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.ticket, entry.symbol, entry.direction.value, entry.volume,
                entry.entry_price, entry.exit_price, entry.stop_loss, entry.take_profit,
                entry.entry_time, entry.exit_time, entry.duration_minutes,
                entry.profit, entry.commission, entry.swap, entry.net_profit, entry.pips,
                entry.magic_number, entry.strategy_name, entry.timeframe,
                entry.setup_type, entry.notes, json.dumps(entry.screenshots),
                json.dumps(entry.tags), entry.rating, json.dumps(entry.mistakes), entry.lessons,
                entry.session, entry.day_of_week, entry.news_events,
                entry.status.value, entry.created_at, entry.updated_at
            ))
            
            return cursor.lastrowid
    
    def annotate_trade(
        self,
        ticket: int,
        notes: str = None,
        setup_type: str = None,
        rating: int = None,
        tags: List[str] = None,
        mistakes: List[str] = None,
        lessons: str = None,
        screenshots: List[str] = None
    ) -> bool:
        """
        Add annotations to an existing trade.
        
        Args:
            ticket: MT5 trade ticket number.
            notes: Trade notes.
            setup_type: Setup classification.
            rating: Self-assessment 1-5.
            tags: List of tags.
            mistakes: List of mistakes made.
            lessons: Lessons learned.
            screenshots: List of screenshot file paths.
            
        Returns:
            True if updated successfully.
        """
        updates = []
        params = []
        
        if notes is not None:
            updates.append("notes = ?")
            params.append(notes)
        
        if setup_type is not None:
            updates.append("setup_type = ?")
            params.append(setup_type)
        
        if rating is not None:
            updates.append("rating = ?")
            params.append(rating)
        
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        
        if mistakes is not None:
            updates.append("mistakes = ?")
            params.append(json.dumps(mistakes))
        
        if lessons is not None:
            updates.append("lessons = ?")
            params.append(lessons)
        
        if screenshots is not None:
            updates.append("screenshots = ?")
            params.append(json.dumps(screenshots))
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(ticket)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE trades SET {', '.join(updates)} WHERE ticket = ?",
                params
            )
            return cursor.rowcount > 0
    
    def get_trade(self, ticket: int) -> Optional[dict]:
        """Get a single trade by ticket."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trades WHERE ticket = ?", (ticket,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
    
    def get_trades(
        self,
        symbol: str = None,
        magic_number: int = None,
        days: int = None,
        status: str = None,
        setup_type: str = None,
        limit: int = 100
    ) -> List[dict]:
        """
        Query trades with filters.
        
        Args:
            symbol: Filter by symbol.
            magic_number: Filter by EA magic number.
            days: Only trades from last N days.
            status: Filter by status (open, closed).
            setup_type: Filter by setup type.
            limit: Maximum results.
            
        Returns:
            List of trade dictionaries.
        """
        conditions = []
        params = []
        
        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        
        if magic_number is not None:
            conditions.append("magic_number = ?")
            params.append(magic_number)
        
        if days:
            from_date = (datetime.now() - timedelta(days=days)).isoformat()
            conditions.append("entry_time >= ?")
            params.append(from_date)
        
        if status:
            conditions.append("status = ?")
            params.append(status)
        
        if setup_type:
            conditions.append("setup_type = ?")
            params.append(setup_type)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT * FROM trades 
                WHERE {where_clause}
                ORDER BY entry_time DESC
                LIMIT ?
            """, params + [limit])
            
            return [self._row_to_dict(row) for row in cursor.fetchall()]
    
    def get_performance_stats(
        self,
        days: int = 30,
        symbol: str = None,
        magic_number: int = None
    ) -> dict:
        """
        Calculate performance statistics.
        
        Args:
            days: Analysis period.
            symbol: Filter by symbol.
            magic_number: Filter by EA.
            
        Returns:
            Performance statistics dictionary.
        """
        trades = self.get_trades(
            days=days,
            symbol=symbol,
            magic_number=magic_number,
            status="closed",
            limit=10000
        )
        
        if not trades:
            return {
                "total_trades": 0,
                "message": "No trades found"
            }
        
        profits = [t["net_profit"] for t in trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p < 0]
        
        total_profit = sum(wins) if wins else 0
        total_loss = abs(sum(losses)) if losses else 0
        
        # Calculate by day
        trades_by_day = {}
        for t in trades:
            day = t["entry_time"][:10]
            if day not in trades_by_day:
                trades_by_day[day] = []
            trades_by_day[day].append(t["net_profit"])
        
        daily_pnl = {day: sum(pnls) for day, pnls in trades_by_day.items()}
        
        return {
            "period_days": days,
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": total_profit - total_loss,
            "profit_factor": total_profit / total_loss if total_loss > 0 else float("inf"),
            "average_win": sum(wins) / len(wins) if wins else 0,
            "average_loss": sum(losses) / len(losses) if losses else 0,
            "largest_win": max(wins) if wins else 0,
            "largest_loss": min(losses) if losses else 0,
            "best_day": max(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else None,
            "worst_day": min(daily_pnl.items(), key=lambda x: x[1]) if daily_pnl else None,
            "trading_days": len(trades_by_day)
        }
    
    def export_csv(self, filepath: str, days: int = 30) -> str:
        """Export trades to CSV file."""
        import csv
        
        trades = self.get_trades(days=days, limit=10000)
        
        if not trades:
            return ""
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=trades[0].keys())
            writer.writeheader()
            writer.writerows(trades)
        
        return filepath
    
    def export_json(self, filepath: str, days: int = 30) -> str:
        """Export trades to JSON file."""
        trades = self.get_trades(days=days, limit=10000)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(trades, f, indent=2)
        
        return filepath
    
    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert database row to dictionary."""
        d = dict(row)
        
        # Parse JSON fields
        for field in ['screenshots', 'tags', 'mistakes']:
            if field in d and d[field]:
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    d[field] = []
        
        return d


# ============================================================================
# Global Instance
# ============================================================================

_journal: Optional[TradeJournal] = None


def get_trade_journal(db_path: str = None) -> TradeJournal:
    """Get or create the global Trade Journal instance."""
    global _journal
    if _journal is None:
        _journal = TradeJournal(db_path)
    return _journal
