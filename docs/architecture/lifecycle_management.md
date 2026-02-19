# Lifecycle Management Architecture

## Overview

The Lifecycle Management system handles automatic bot progression through tags based on performance criteria. This document describes the architecture, components, and integration points.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         APScheduler                                  │
│                    (Daily at 3:00 AM UTC)                           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LifecycleManager                                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ - run_daily_check()                                           │  │
│  │ - _check_progression(bot) → TagProgression                    │  │
│  │ - _check_quarantine(bot) → bool                               │  │
│  │ - _promote_bot(bot, new_tag)                                  │  │
│  │ - _quarantine_bot(bot)                                        │  │
│  │ - _kill_bot(bot)                                              │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   BotRegistry   │  │PerformanceTracker│  │    PromQL      │
│ (BotManifest)   │  │ (PromotionManager)│  │   Metrics      │
└─────────────────┘  └─────────────────┘  └─────────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Database      │  │   Trade Journal │  │   Prometheus    │
│  (PostgreSQL)   │  │    (SQLite)     │  │    Server       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Components

### 1. LifecycleManager

**File:** `src/router/lifecycle_manager.py`

The core component responsible for evaluating and managing bot lifecycle.

**Key Responsibilities:**
- Daily evaluation of all registered bots
- Applying progression criteria
- Managing quarantine and dead states
- Emitting metrics and notifications

**Key Methods:**

```python
class LifecycleManager:
    def run_daily_check(self) -> Dict[str, Any]:
        """Run daily lifecycle evaluation for all bots."""
        
    def _check_progression(self, bot: BotManifest) -> Optional[TagProgression]:
        """Evaluate if bot meets criteria for next tag."""
        
    def _check_quarantine(self, bot: BotManifest) -> bool:
        """Evaluate if bot should be quarantined."""
        
    def _promote_bot(self, bot: BotManifest, new_tag: str) -> None:
        """Update bot tag and emit notifications."""
        
    def _quarantine_bot(self, bot: BotManifest, reason: str) -> None:
        """Move bot to quarantine state."""
```

### 2. BotRegistry (BotManifest)

**File:** `src/router/bot_manifest.py`

Manages bot registration and tag storage.

**Key Features:**
- Tag storage in `tags` field
- CRUD operations for bots
- Tag-based queries

**Integration:**
```python
from src.router.bot_manifest import BotRegistry

registry = BotRegistry()

# Get all bots
bots = registry.list_all()

# Update bot tags
registry.update_bot_tags(bot_id, ["@pending", "strategy_type"])

# Query by tag
pending_bots = registry.get_by_tag("@pending")
```

### 3. PerformanceTracker

**File:** `src/router/promotion_manager.py`

Calculates performance statistics used for progression evaluation.

**Key Metrics:**
- Total trades
- Win rate
- Sharpe ratio
- Maximum drawdown
- Profit factor
- Consecutive losing days

**Integration:**
```python
from src.router.promotion_manager import PerformanceTracker

tracker = PerformanceTracker()
stats = tracker.calculate_stats(bot_id)

# Returns:
{
    "total_trades": 45,
    "win_rate": 0.58,
    "sharpe_ratio": 1.6,
    "max_drawdown": 0.08,
    "profit_factor": 1.8,
    "consecutive_losing_days": 0,
    "critical_errors": 0
}
```

### 4. DynamicBotLimiter

**File:** `src/router/dynamic_bot_limits.py`

Enforces bot limits based on account balance.

**Tier Configuration:**
```python
TIERS = [
    (0, 50, 0),           # Tier 0: $0-50: 0 bots (disabled)
    (50, 100, 1),         # Tier 1: $50-100: 1 bot
    (100, 200, 2),        # Tier 2: $100-200: 2 bots
    (200, 500, 3),        # Tier 3: $200-500: 3 bots
    (500, 1000, 5),       # Tier 4: $500-1k: 5 bots
    (1000, 5000, 10),     # Tier 5: $1k-5k: 10 bots
    (5000, float('inf'), 20)  # Tier 6: $5k+: 20 bots
]
```

**Key Methods:**
```python
class DynamicBotLimiter:
    @staticmethod
    def get_max_bots(balance: float) -> int:
        """Get maximum bots for account balance."""
        
    @staticmethod
    def can_add_bot(balance: float, current_bots: int) -> Tuple[bool, str]:
        """Check if a new bot can be added."""
        
    @staticmethod
    def get_tier_info(balance: float) -> Dict[str, Any]:
        """Get detailed tier information."""
```

### 5. MarketScanner

**File:** `src/router/market_scanner.py`

Detects trading opportunities during specific market sessions.

**Scanner Types:**
1. Session Breakout Scanner
2. Volatility Scanner
3. News Event Scanner
4. ICT Setup Scanner

**Session-Aware Frequency:**
- Overlap: 1 minute
- London/NY: 5 minutes
- Asian: 15 minutes

## Data Models

### TagProgression

```python
@dataclass
class TagProgression:
    bot_id: str
    current_tag: str
    next_tag: str
    meets_criteria: bool
    criteria_details: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

### ProgressionCriteria

```python
@dataclass
class ProgressionCriteria:
    min_trades: int
    min_win_rate: float
    min_sharpe_ratio: Optional[float] = None
    min_days_active: int
    max_drawdown: Optional[float] = None
    min_profit_factor: Optional[float] = None
    max_critical_errors: int = 0
```

### QuarantineTrigger

```python
@dataclass
class QuarantineTrigger:
    max_win_rate_drop: float = 0.45
    max_sharpe_ratio: float = 0.5
    max_drawdown: float = 0.20
    max_consecutive_losing_days: int = 5
```

## Progression Criteria Configuration

The progression criteria are defined in `LifecycleManager.PROGRESSION_CRITERIA`:

```python
PROGRESSION_CRITERIA = {
    "@primal": ProgressionCriteria(
        min_trades=20,
        min_win_rate=0.50,
        min_days_active=7,
        max_critical_errors=0,
    ),
    "@pending": ProgressionCriteria(
        min_trades=50,
        min_win_rate=0.55,
        min_sharpe_ratio=1.5,
        min_days_active=30,
        max_drawdown=0.15,
    ),
    "@perfect": ProgressionCriteria(
        min_trades=100,
        min_win_rate=0.58,
        min_sharpe_ratio=2.0,
        min_days_active=60,
        max_drawdown=0.10,
        min_profit_factor=1.5,
    ),
}
```

## Scheduler Integration

The lifecycle check is scheduled via APScheduler.

**File:** `scripts/schedule_lifecycle_check.py`

```python
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

scheduler = BackgroundScheduler()
scheduler.add_job(
    run_lifecycle_check,
    trigger=CronTrigger(
        day_of_week='*',
        hour=3,
        minute=0,
        timezone='UTC'
    ),
    id='lifecycle_check',
    replace_existing=True
)
scheduler.start()
```

## Prometheus Metrics

The lifecycle manager emits the following metrics:

```python
BOT_LIFECYCLE_PROMOTIONS = Counter(
    'quantmind_bot_lifecycle_promotions_total',
    'Total bot promotions',
    ['from_tag', 'to_tag']
)

BOT_LIFECYCLE_QUARANTINES = Counter(
    'quantmind_bot_lifecycle_quarantines_total',
    'Total bot quarantines',
    ['reason']
)

BOT_LIFECYCLE_CHECKS = Counter(
    'quantmind_bot_lifecycle_checks_total',
    'Total lifecycle checks performed'
)

BOT_LIFECYCLE_CHECK_DURATION = Histogram(
    'quantmind_bot_lifecycle_check_duration_seconds',
    'Duration of lifecycle checks'
)
```

## WebSocket Integration

Lifecycle events are broadcast via WebSocket for real-time UI updates.

**Channel:** `ws://localhost:8000/ws/lifecycle`

**Event Types:**
- `bot_promoted`
- `bot_quarantined`
- `bot_killed`
- `lifecycle_check_complete`

**Implementation:**
```python
# In lifecycle_manager.py
async def _broadcast_promotion(self, bot_id: str, from_tag: str, to_tag: str):
    await websocket_manager.broadcast({
        "type": "bot_promoted",
        "bot_id": bot_id,
        "from_tag": from_tag,
        "to_tag": to_tag,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
```

## Database Schema

### Bot Tags Storage

Tags are stored in the `bot_manifests` table:

```sql
CREATE TABLE bot_manifests (
    bot_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    tags JSONB DEFAULT '[]',
    -- other fields...
);
```

### Tag History

Optional: Track tag changes over time:

```sql
CREATE TABLE bot_tag_history (
    id SERIAL PRIMARY KEY,
    bot_id VARCHAR(255) REFERENCES bot_manifests(bot_id),
    old_tag VARCHAR(50),
    new_tag VARCHAR(50),
    reason VARCHAR(255),
    changed_at TIMESTAMP DEFAULT NOW(),
    changed_by VARCHAR(50) DEFAULT 'system'
);
```

## Error Handling

The lifecycle manager implements graceful error handling:

```python
def run_daily_check(self) -> Dict[str, Any]:
    results = {
        "checked": 0,
        "promoted": 0,
        "quarantined": 0,
        "errors": []
    }
    
    for bot in self.registry.list_all():
        try:
            # Process bot
            ...
        except Exception as e:
            logger.error(f"Error processing bot {bot.bot_id}: {e}")
            results["errors"].append({
                "bot_id": bot.bot_id,
                "error": str(e)
            })
    
    return results
```

## Testing

Unit tests are in `tests/router/test_lifecycle_manager.py`:

```bash
pytest tests/router/test_lifecycle_manager.py -v
```

Key test classes:
- `TestProgressionCriteria` - Test criteria evaluation
- `TestQuarantineTriggers` - Test quarantine conditions
- `TestLifecycleManager` - Test main functionality
- `TestLifecycleMetrics` - Test metrics emission

## Related Documentation

- [User Guide: Bot Lifecycle](../user-guide/bot_lifecycle.md)
- [API: Lifecycle Endpoints](../api/lifecycle_endpoints.md)
- [Dynamic Bot Limits](./dynamic_bot_limits.md)