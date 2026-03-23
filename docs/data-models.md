# QUANTMINDX — Database Schema & Data Models

**Generated:** 2026-03-11 | **Databases:** SQLite (operational) + DuckDB (analytics)

---

## Database Architecture

| Database | File | Purpose |
|----------|------|---------|
| SQLite (SQLAlchemy) | `data/db/quantmind.db` | Operational data — bots, trades, agents, accounts |
| DuckDB | `data/db/analytics.duckdb` | Time-series market data, backtest results |
| DuckDB hybrid | `test_hybrid.duckdb.backup` | Test analytics DB |
| AgentDB | `agentdb.db` + `agentdb.rvf` | Agent memory (SQLite + vector) |
| Department Mail | `.quantmind/department_mail.db` | Inter-agent SQLite mailbox |

---

## SQLite Models (SQLAlchemy)

Located in `src/database/models/`

### Trading (`models/trading.py`)

#### `trade_proposals`
Proposed trades from bots awaiting analyst/quant review.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | |
| account_id | FK → prop_firm_accounts | Optional |
| bot_id | String(100) | Proposing bot |
| symbol | String(20) | e.g. "EURUSD" |
| kelly_score | Float | Kelly criterion score |
| regime | String(50) | Market regime at proposal |
| proposed_lot_size | Float | Suggested position size |
| status | String(20) | pending / approved / rejected |
| created_at | DateTime | |
| reviewed_at | DateTime | |

#### `risk_tier_transitions`
Audit log of risk tier changes (growth → scaling → guardian).

| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | |
| account_id | FK → prop_firm_accounts | |
| from_tier | String(20) | growth / scaling / guardian |
| to_tier | String(20) | |
| equity_at_transition | Float | |
| transition_timestamp | DateTime | |

#### `crypto_trades`
Crypto trades from Binance and other exchanges.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | |
| broker_type | String | binance_spot / binance_futures / mt5 |
| broker_id | String | Registry ID |
| order_id | String | Exchange order ID |
| symbol | String(20) | BTCUSDT / ETHUSDT / etc. |
| direction | String | buy / sell |
| volume | Float | Quantity (crypto) or lots (MT5) |
| entry_price | Float | |
| exit_price | Float | null if open |

---

### Bots (`models/bots.py`)

#### `bot_circuit_breaker`
Per-bot circuit breaker state.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | |
| bot_id | String(100) UNIQUE | |
| consecutive_losses | Integer | |
| daily_trade_count | Integer | |
| last_trade_time | DateTime | |
| is_quarantined | Boolean | |
| quarantine_reason | String(200) | |
| quarantine_start | DateTime | |
| mode | Enum(TradingMode) | LIVE / DEMO |

#### `bot_clone_history`
Tracks bot cloning operations.

| Column | Type | Description |
|--------|------|-------------|
| original_bot_id | String(100) | |
| clone_bot_id | String(100) | |
| original_symbol | String(20) | |
| clone_symbol | String(20) | |
| performance_at_clone | JSON | Metrics at clone time |
| allocation_strategy | String(50) | adaptive / equal |

#### `daily_fee_tracking`
Daily fee burn monitoring.

| Column | Type | Description |
|--------|------|-------------|
| account_id | String | |
| date | String | YYYY-MM-DD |
| total_fees | Float | |
| total_trades | Integer | |
| fee_burn_pct | Float | fees/balance × 100 |
| account_balance | Float | |
| kill_switch_activated | Boolean | |

---

### Accounts (`models/account.py`)

#### `prop_firm_accounts`
Prop firm trading accounts.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | |
| broker_id | FK → broker_registry | |
| account_number | String | MT5 account number |
| phase | String | evaluation / funded |
| firm | String | FTMO / The5ers / FundingPips |
| balance | Float | Current balance |
| equity | Float | Current equity |
| max_daily_dd | Float | Maximum daily drawdown % |
| max_total_dd | Float | Maximum total drawdown % |
| mode | Enum(TradingMode) | LIVE / DEMO |

---

### Market Data (`models/market.py`)

#### `market_opportunities`
Market scanner-detected opportunities.

| Column | Type | Description |
|--------|------|-------------|
| symbol | String(20) | |
| timeframe | String | |
| regime | String | |
| chaos_score | Float | |
| signal_type | String | |
| detected_at | DateTime | |

---

### Agents (`models/agents.py`)

#### `agent_sessions`
Active and historical agent task sessions.

| Column | Type | Description |
|--------|------|-------------|
| session_id | String UUID | |
| agent_type | String | analyst / quantcode / floor_manager / etc. |
| status | String | active / completed / failed |
| task_description | Text | |
| result | JSON | |
| started_at | DateTime | |
| completed_at | DateTime | |

---

### Agent Session (`models/agent_session.py`)

Full session state including checkpoints.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer PK | |
| agent_id | String | |
| checkpoint_data | JSON | Full state snapshot |
| created_at | DateTime | |

---

### HMM (`models/hmm.py`)

#### `hmm_models`
Trained HMM model metadata.

| Column | Type | Description |
|--------|------|-------------|
| model_id | String UUID | |
| version | String | semantic version |
| deployment_mode | String | shadow / hybrid / production |
| n_components | Integer | Number of hidden states |
| features | JSON | Feature list |
| trained_at | DateTime | |
| accuracy | Float | Validation accuracy |

#### `hmm_shadow_log`
Shadow mode prediction log for model comparison.

| Column | Type | Description |
|--------|------|-------------|
| symbol | String(20) | |
| timeframe | String | |
| ising_regime | String | |
| hmm_regime | String | |
| agreement | Boolean | Do Ising and HMM agree? |
| decision_source | String | ising / hmm / weighted |
| timestamp | DateTime | |

---

### Performance (`models/performance.py`)

#### `performance_snapshots`
Periodic performance snapshots per bot/account.

| Column | Type | Description |
|--------|------|-------------|
| bot_id | String | |
| account_id | Integer | |
| period | String | daily / weekly / monthly |
| win_rate | Float | |
| profit_factor | Float | |
| sharpe_ratio | Float | |
| max_drawdown | Float | |
| total_trades | Integer | |
| snapshot_at | DateTime | |

---

### Session Checkpoints (`models/session_checkpoint.py`)

Long-running agent operation checkpoints for resumability.

| Column | Type | Description |
|--------|------|-------------|
| checkpoint_id | String UUID | |
| agent_id | String | |
| operation_type | String | video_analysis / backtest / etc. |
| state | JSON | Resumable state snapshot |
| progress_pct | Float | |
| created_at | DateTime | |
| updated_at | DateTime | |

---

### Provider Config (`models/provider_config.py`)

LLM provider configurations.

| Column | Type | Description |
|--------|------|-------------|
| provider_id | String | anthropic / openai / openrouter |
| model_id | String | claude-opus-4-6 / etc. |
| api_key_env | String | env var name |
| base_url | String | API URL |
| is_active | Boolean | |
| max_tokens | Integer | |

---

### Migrations (`database/migrations/`)

| Migration | Table Added |
|-----------|-------------|
| `add_activity_events.py` | `activity_events` |
| `add_agent_sessions.py` | `agent_sessions` |
| `add_bot_clone_history.py` | `bot_clone_history` |
| `add_daily_fee_tracking.py` | `daily_fee_tracking` |
| `add_demo_mode.py` | `TradingMode` enum |
| `add_hmm_tables.py` | `hmm_models`, `hmm_shadow_log` |
| `add_lifecycle_log.py` | `lifecycle_log` |
| `add_market_opportunities.py` | `market_opportunities` |
| `add_progressive_kill_switch.py` | `kill_switch_state` |
| `add_session_checkpoints.py` | `session_checkpoints` |
| `add_tick_streaming_tables.py` | `tick_stream_log` |
| `add_trade_journal_mode.py` | `trade_journal_mode` |
| `add_webhook_and_ea_tables.py` | `webhooks`, `ea_registry` |

**Migration runner:** `src/database/migrations/migration_runner.py` — runs all pending migrations on startup.

---

## DuckDB Analytics Schema (`src/database/duckdb/`)

### `analytics.py` — Analytics tables

| Table | Purpose |
|-------|---------|
| `market_data` | OHLCV time-series data per symbol/timeframe |
| `backtest_results` | Backtest run results |
| `trade_analytics` | Enriched trade data for analytics queries |
| `regime_history` | Historical regime classifications per symbol |

### `market_data.py`

```sql
CREATE TABLE market_data (
    symbol      VARCHAR,
    timeframe   VARCHAR,
    timestamp   TIMESTAMP,
    open        DOUBLE,
    high        DOUBLE,
    low         DOUBLE,
    close       DOUBLE,
    volume      DOUBLE,
    PRIMARY KEY (symbol, timeframe, timestamp)
);
```

---

## BrokerRegistry Model (SQLAlchemy, `src/database/models/`)

#### `broker_registry`

| Column | Type | Description |
|--------|------|-------------|
| broker_id | String PK | e.g. "icmarkets_raw" |
| broker_name | String | Human-readable |
| spread_avg | Float | Average spread in points |
| commission_per_lot | Float | Commission per standard lot |
| lot_step | Float | Min lot step (default 0.01) |
| min_lot | Float | Min lot size |
| max_lot | Float | Max lot size |
| pip_values | JSON | {symbol: pip_value_per_lot} |
| preference_tags | JSON | Tag list |
| created_at | DateTime | |

**Default pip values:**
```json
{"EURUSD": 10.0, "GBPUSD": 10.0, "USDJPY": 9.09, "XAUUSD": 1.0, "XAGUSD": 50.0}
```

---

## AgentDB (SQLite + Vector)

`agentdb.db` — stores agent memory, department mail, task queues.
`agentdb.rvf` — vector store component for semantic memory search.

---

## Department Mail DB

`.quantmind/department_mail.db` — SQLite database for the inter-department message bus.

**Schema:**
```sql
CREATE TABLE messages (
    id          TEXT PRIMARY KEY,
    from_dept   TEXT,
    to_dept     TEXT,
    type        TEXT,   -- DISPATCH | RESULT | ALERT | REQUEST
    subject     TEXT,
    body        TEXT,
    priority    INTEGER, -- 0=LOW, 1=NORMAL, 2=HIGH, 3=URGENT
    is_read     BOOLEAN,
    created_at  TIMESTAMP,
    read_at     TIMESTAMP
);
```
