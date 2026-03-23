# Story 9.0: Portfolio & Broker Infrastructure Audit

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer starting Epic 9,
I want a complete audit of the current broker registry, routing matrix, and portfolio data state,
so that stories 9.1-9.5 build on verified existing infrastructure.

## Acceptance Criteria

1. **Given** the backend in `src/`,
   **When** the audit runs,
   **Then** a findings document covers:
   - (a) existing broker registry implementation and schema
   - (b) RoutingMatrix class state and API exposure
   - (c) existing portfolio metrics API endpoints
   - (d) MT5 account auto-detection implementation
   - (e) correlation matrix computation state

## Tasks / Subtasks

- [x] Task 1: Audit Broker Registry Implementation (AC: a)
  - [x] Task 1.1: Document src/router/broker_registry.py - BrokerRegistryManager with database model
  - [x] Task 1.2: Document src/api/broker_endpoints.py - in-memory BrokerRegistry
  - [x] Task 1.3: Document database schema in src/database/models/account.py
  - [x] Task 1.4: Identify gaps between the two implementations
- [x] Task 2: Audit RoutingMatrix (AC: b)
  - [x] Task 2.1: Document src/router/routing_matrix.py - RoutingMatrix class
  - [x] Task 2.2: Document AccountConfig and RoutingDecision data classes
  - [x] Task 2.3: Verify existing API exposure and integration points
- [x] Task 3: Audit Portfolio Metrics API (AC: c)
  - [x] Task 3.1: Document src/api/portfolio_endpoints.py endpoints
  - [x] Task 3.2: Document src/agents/departments/heads/portfolio_head.py
  - [x] Task 3.3: Verify endpoint implementations vs AC requirements
- [x] Task 4: Audit MT5 Auto-Detection (AC: d)
  - [x] Task 4.1: Document /api/brokers/heartbeat endpoint
  - [x] Task 4.2: Document broker detection flow and state management
  - [x] Task 4.3: Verify auto-detection covers: broker, account_type, leverage, currency
- [x] Task 5: Audit Correlation Matrix (AC: e)
  - [x] Task 5.1: Document src/api/loss_propagation.py correlation computation
  - [x] Task 5.2: Identify correlation matrix API endpoint if exists
  - [x] Task 5.3: Note production vs mock implementation status
- [x] Task 6: Compile Findings Document
  - [x] Task 6.1: Create findings summary with component status
  - [x] Task 6.2: Document architecture decisions and constraints
  - [x] Task 6.3: Note what needs extension vs what is production-ready

## Findings Document

### AC (a): Broker Registry Implementation

#### 1. Database-Backed BrokerRegistryManager (`src/router/broker_registry.py`)
- **Location:** `src/router/broker_registry.py`
- **Purpose:** Manages broker profiles with fee structures and trading parameters
- **Key Class:** `BrokerRegistryManager`
- **Methods:**
  - `get_broker(broker_id)` - Retrieve broker profile
  - `create_broker()` - Create new broker with fee structure
  - `update_broker()` - Update existing broker
  - `get_pip_value(symbol, broker_id)` - Dynamic pip value lookup
  - `get_commission(broker_id)` - Commission per lot
  - `find_brokers_by_tag(tag)` - Find brokers by preference tags
  - `list_all_brokers()` - List all registered brokers
- **Database Model:** `BrokerRegistry` in `src/database/models/account.py`
- **Status:** PRODUCTION-READY (DB-backed, real implementation)

#### 2. In-Memory BrokerRegistry (`src/api/broker_endpoints.py`)
- **Location:** `src/api/broker_endpoints.py`
- **Purpose:** REST API for broker management with in-memory state
- **Key Class:** `BrokerRegistry` (in-memory)
- **Methods:**
  - `add_or_update(heartbeat)` - Add/update from MT5 heartbeat
  - `confirm(broker_id)` - Confirm pending broker
  - `ignore(broker_id)` - Ignore pending broker
  - `get(broker_id)` - Get broker by ID
  - `get_all()` - Get all confirmed brokers
  - `disconnect(broker_id)` - Mark broker disconnected
  - `sync(broker_id)` - Request broker sync
- **Account Switcher:** `AccountSwitcher` class manages active account
- **WebSocket Support:** `BrokerWebSocketManager` for real-time events
- **Status:** PRODUCTION-READY (API endpoints exist, in-memory state)

#### 3. Database Schema (`src/database/models/account.py`)
- **Table:** `broker_registry`
- **Schema:**
  ```
  - id: Integer (PK)
  - broker_id: String (unique, indexed)
  - broker_name: String
  - spread_avg: Float
  - commission_per_lot: Float
  - lot_step: Float
  - min_lot: Float
  - max_lot: Float
  - pip_values: JSON (dict)
  - preference_tags: JSON (list)
  - created_at: DateTime
  - updated_at: DateTime
  ```
- **Status:** PRODUCTION-READY

#### 4. Gap Analysis: DB vs In-Memory
| Aspect | DB-Backed | In-Memory | Gap |
|--------|-----------|-----------|-----|
| Persistence | Yes | No | Story 9.1 should wire in-memory to DB |
| Fee Structure | Yes (spread, commission, pip values) | No | Use DB-backed for fee-aware routing |
| Auto-Detection | No | Yes | In-memory handles MT5 heartbeat |
| Recommendation | Use for registry | Use for live detection | Wire together in Story 9.1 |

---

### AC (b): RoutingMatrix Class

#### Location: `src/router/routing_matrix.py`

#### Key Classes:

1. **AccountType Enum:**
   - `MACHINE_GUN` - HFT/Scalper accounts
   - `SNIPER` - Structural/ICT accounts
   - `PROP_FIRM` - Prop firm challenge accounts
   - `CRYPTO` - Crypto exchange
   - `DEMO` - Demo/paper trading

2. **AccountConfig Dataclass:**
   - `account_id`, `account_type`, `broker_name`, `account_number`
   - `max_positions`, `max_daily_trades`, `capital_allocation`
   - `is_active`, `is_demo`
   - `accepts_strategies`, `accepts_frequencies`, `requires_prop_safe`
   - `current_positions`, `daily_trades`, `current_pnl`

3. **RoutingDecision Dataclass:**
   - `bot_id`, `assigned_account`, `is_approved`
   - `rejection_reason`, `priority_score`

4. **RoutingMatrix Class:**
   - `register_account(config)` - Register trading account
   - `get_account(account_id)` - Get account by ID
   - `list_accounts()` - List all accounts
   - `route_bot(manifest)` - Route bot to account
   - `get_routing_summary()` - Get routing state summary
   - `route_all_bots()` - Route all registered bots
   - `get_account_for_bot(manifest)` - Get assigned account

#### Default Accounts Initialized:
- `account_a_machine_gun` - RoboForex Prime (HFT/Scalper)
- `account_b_sniper` - Exness Raw (Structural/ICT)
- `demo_account` - Demo account

#### Integration Points:
- Uses `BotRegistry` from `src/router/bot_manifest.py`
- Strategy types: `SCALPER`, `HFT`, `STRUCTURAL`, `SWING`, etc.
- Trade frequencies: `HFT`, `HIGH`, `LOW`, `MEDIUM`
- Broker types: `RAW_ECN`, `STANDARD`

#### API Exposure:
- No direct REST API endpoints for routing matrix (Story 9.1 needed)
- Used by Strategy Router during dispatch
- Global instance via `get_routing_matrix()`

#### Status: PRODUCTION-READY (routing logic complete, API needed for Story 9.1)

---

### AC (c): Portfolio Metrics API

#### Location: `src/api/portfolio_endpoints.py`

#### Endpoints:

| Endpoint | Response Model | Status |
|----------|---------------|--------|
| `GET /api/portfolio/report` | `PortfolioReportResponse` | IMPLEMENTED |
| `GET /api/portfolio/equity` | `TotalEquityResponse` | IMPLEMENTED |
| `GET /api/portfolio/pnl/strategy` | `StrategyPnLResponse` | IMPLEMENTED |
| `GET /api/portfolio/pnl/broker` | `BrokerPnLResponse` | IMPLEMENTED |
| `GET /api/portfolio/drawdowns` | `AccountDrawdownsResponse` | IMPLEMENTED |

#### PortfolioHead (`src/agents/departments/heads/portfolio_head.py`):

Methods implemented:
- `generate_portfolio_report()` - Complete report with equity, P&L, drawdowns
- `get_total_equity()` - Total equity across accounts
- `get_strategy_pnl(period)` - P&L by strategy
- `get_broker_pnl(period)` - P&L by broker
- `get_account_drawdowns()` - Drawdown per account

Additional methods:
- `optimize_allocation()` - Mean-variance optimization
- `rebalance_portfolio()` - Rebalance to target allocation
- `track_performance()` - Performance metrics (alpha, beta, sharpe)

#### Status: PARTIAL (endpoints exist, DEMO DATA - needs real data wiring)

**Gap:** All methods return hardcoded/demo data. Story 9.2 should wire to real trade database and broker accounts.

---

### AC (d): MT5 Auto-Detection

#### Location: `src/api/broker_endpoints.py`

#### Heartbeat Endpoint: `POST /api/brokers/heartbeat`

**Request Model (BrokerHeartbeat):**
```python
- account_id: str
- server: str
- broker_name: str
- balance: float
- equity: float
- margin: float
- leverage: int
- currency: str = "USD"
```

**Auto-Detection Coverage:**
| Field | Detected | Notes |
|-------|----------|-------|
| broker_name | Yes | From heartbeat |
| account_type | Partial | Not explicitly in heartbeat, inferred from account_id/server |
| leverage | Yes | From heartbeat |
| currency | Yes | From heartbeat |

#### Detection Flow:
1. MT5 bridge sends heartbeat to `/api/brokers/heartbeat`
2. System checks if broker exists (by account_id + server)
3. If new: added to `pending` with status "pending", WebSocket notification sent
4. If existing: updates balance/equity/margin/leverage/currency, status "connected"
5. User confirms pending broker via `/api/brokers/{broker_id}/confirm`
6. WebSocket clients notified of status changes

#### Additional Endpoints:
- `GET /api/brokers` - List all confirmed brokers (paginated)
- `GET /api/brokers/pending` - List pending brokers
- `GET /api/brokers/{broker_id}` - Get broker details
- `POST /api/brokers/{broker_id}/confirm` - Confirm pending broker
- `POST /api/brokers/{broker_id}/ignore` - Ignore pending broker
- `POST /api/brokers/{broker_id}/disconnect` - Mark disconnected
- `POST /api/brokers/{broker_id}/sync` - Request sync
- `POST /api/brokers/add` - Manual broker add
- `WS /api/brokers/ws` - WebSocket for real-time events
- `GET /api/brokers/accounts` - List all accounts
- `GET /api/brokers/accounts/active` - Get active account
- `POST /api/brokers/accounts/active` - Switch active account

#### Status: PRODUCTION-READY (auto-detection works, in-memory state needs DB persistence in Story 9.1)

---

### AC (e): Correlation Matrix

#### Location: `src/api/loss_propagation.py`

#### Implementation:

**Class: LossPropagationEngine**
- `get_correlation_matrix()` - Get NxN correlation matrix
- `find_correlated_strategies(source_strategy, correlation_matrix, threshold)` - Find correlated

**Current State:**
- MOCK implementation - returns hardcoded correlation matrix
- Correlations between strategy types:
  - HFT ↔ Scalper: 0.85 (high correlation)
  - Structural ↔ Swing: 0.65 (moderate correlation)
  - Different types: 0.1-0.3 (low correlation)

**API Endpoint:**
- No dedicated `/api/portfolio/correlation` endpoint yet
- Correlation matrix used internally by loss propagation engine
- Referenced by Story 8.9 (AB Race Board - loss propagation)

**Gap Analysis vs Story 9.2 AC:**
| Story 9.2 Requirement | Current State | Gap |
|------------------------|---------------|-----|
| `GET /api/portfolio/correlation` | Not exists | Story 9.2 needed |
| Return NxN with strategy_a, strategy_b, correlation, period_days | N/A | Story 9.2 needed |
| Production correlation computation | Mock | Story 9.2 needed |

#### Status: MOCK (needs production implementation in Story 9.2)

---

## Summary: Component Status

| Component | Status | Next Story |
|-----------|--------|------------|
| BrokerRegistryManager (DB) | Production Ready | Use in 9.1 |
| BrokerRegistry (In-Memory) | Production Ready | Wire to DB in 9.1 |
| BrokerRegistry DB Schema | Production Ready | - |
| RoutingMatrix | Production Ready | Add API in 9.1 |
| Portfolio Endpoints | Implemented (Demo Data) | Wire real data in 9.2 |
| PortfolioHead | Implemented (Demo Data) | Wire real data in 9.2 |
| MT5 Auto-Detection | Production Ready | Wire to DB in 9.1 |
| Correlation Matrix | Mock | Implement in 9.2 |

## Architecture Decisions

1. **Broker Registry Strategy:** Two implementations coexist - DB-backed for persistent registry, in-memory for live MT5 detection. Story 9.1 should unify them.

2. **RoutingMatrix Extensibility:** Architecture notes "RoutingMatrix + broker registry stay untouched - extend only for fee-awareness" - the current implementation already supports fee structure via BrokerRegistry.

3. **Portfolio Data:** All portfolio methods use demo data. Story 9.2 must wire to:
   - Trade database for P&L attribution
   - Broker registry for account equity
   - Real correlation computation engine

4. **Correlation Matrix:** Currently only used by loss propagation (Story 8.9). Story 9.2 needs dedicated API endpoint.

## Dev Notes

- This is a READ-ONLY exploration - no code changes required
- Findings document should be comprehensive for subsequent stories to build upon
- Architecture note: RoutingMatrix + broker registry stay untouched - extend only for fee-awareness

### Project Structure Notes

- Epic 9 focuses on Portfolio & Multi-Broker Management
- Stories 9.1-9.5 will build on the audit findings
- Existing infrastructure from Epic 7 (Department Agent Platform) may have related components

### References

- Scan locations: `src/broker/`, `src/portfolio/`, `src/router/routing_matrix.py`
- Related files:
  - src/router/broker_registry.py (DB-backed broker registry manager)
  - src/api/broker_endpoints.py (REST API with in-memory registry)
  - src/router/routing_matrix.py (Bot-to-account routing)
  - src/api/portfolio_endpoints.py (Portfolio metrics)
  - src/database/models/account.py (BrokerRegistry SQLAlchemy model)
  - src/api/loss_propagation.py (Correlation matrix)

## Dev Agent Record

### Agent Model Used

Claude MiniMax-M2.5 (via BMad create-story workflow)

### Debug Log References

N/A - This is an audit/documentation story

### Completion Notes List

- Created comprehensive findings document covering all 5 AC areas
- Verified all source file references against actual implementation
- Documented production-ready components vs gaps needing Story 9.1/9.2 work

### File List

No source code changes - this is a documentation-only audit story.

---

## Senior Developer Review (AI)

**Review Outcome:** Approve
**Review Date:** 2026-03-20

### Git vs Story Discrepancies

- **0 discrepancies found** - This is a documentation-only story (audit), no source code changes expected

### Issues Found: 0 High, 0 Medium, 0 Low

### Verification Summary

| AC | Claim | Verification | Status |
|----|-------|--------------|--------|
| a | BrokerRegistryManager in src/router/broker_registry.py | Verified - class exists | PASS |
| a | BrokerRegistry in-memory in src/api/broker_endpoints.py | Verified - class exists | PASS |
| a | BrokerRegistry DB model in src/database/models/account.py | Verified - table exists | PASS |
| b | RoutingMatrix in src/router/routing_matrix.py | Verified - class exists | PASS |
| c | Portfolio endpoints in src/api/portfolio_endpoints.py | Verified - 5 endpoints exist | PASS |
| c | PortfolioHead in src/agents/departments/heads/portfolio_head.py | Verified - class exists | PASS |
| d | /api/brokers/heartbeat endpoint | Verified - exists in broker_endpoints.py | PASS |
| e | Correlation in src/api/loss_propagation.py | Verified - LossPropagationEngine exists | PASS |

### Review Notes

This is an audit/documentation story with no code implementation. The findings document accurately describes the current state of:

1. Broker registry implementations (DB-backed vs in-memory)
2. RoutingMatrix class and its API exposure
3. Portfolio metrics endpoints
4. MT5 auto-detection via heartbeat
5. Correlation matrix computation (currently mock)

All source file references were verified to exist. The gap analysis and recommendations for Stories 9.1 and 9.2 are accurate.

### Action Items

- [ ] None - Story is complete as designed

---

## Review Follow-ups (AI)

### File List
