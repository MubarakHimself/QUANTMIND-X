# Story 7.8: Risk, Trading & Portfolio Department — Real Implementations

Status: review

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer completing the department agent platform,
I want the Risk, Trading, and Portfolio Department Heads converted from stubs to real implementations,
So that evaluation, paper trade monitoring, and portfolio reporting work end-to-end.

## Acceptance Criteria

1. [AC-1] **Given** a compiled EA passes the SIT gate, **when** the Risk Department Head initiates backtesting, **then** all 6 modes queue and execute (Architecture: backtest engine production-ready — reuse), **And** a pass/fail verdict generates: ≥4/6 modes must pass (Sharpe ≥1.0, max_drawdown ≤15%, win_rate ≥50%).

2. [AC-2] **Given** a strategy is in paper trading, **when** the Trading Department Head monitors it, **then** it tracks P&L per trade, win/loss ratio, drawdown, avg hold time, regime correlation, **And** Copilot receives periodic updates: "Paper trading: 12 trades, +2.3% P&L, 65% win rate."

3. [AC-3] **Given** the Portfolio Department Head generates a report, **when** `GET /api/portfolio/report` is called, **then** it returns: total equity, P&L attribution per strategy, P&L attribution per broker, drawdown per account.

## Tasks / Subtasks

- [x] Task 1: Risk Department — Real Backtest Evaluation (AC: #1)
  - [x] Subtask 1.1: Implement backtest runner integration with 6-mode execution
  - [x] Subtask 1.2: Implement pass/fail verdict logic (Sharpe ≥1.0, max_drawdown ≤15%, win_rate ≥50%)
  - [x] Subtask 1.3: Implement mode-by-mode result aggregation

- [x] Task 2: Trading Department — Paper Trade Monitoring (AC: #2)
  - [x] Subtask 2.1: Implement P&L tracking per trade
  - [x] Subtask 2.2: Implement win/loss ratio, drawdown, hold time metrics
  - [x] Subtask 2.3: Implement Copilot periodic updates
  - [x] Subtask 2.4: Implement regime correlation tracking

- [x] Task 3: Portfolio Department — Report API (AC: #3)
  - [x] Subtask 3.1: Implement total equity calculation
  - [x] Subtask 3.2: Implement P&L attribution per strategy
  - [x] Subtask 3.3: Implement P&L attribution per broker
  - [x] Subtask 3.4: Implement drawdown per account

- [ ] Task 4: Integration (AC: all)
  - [ ] Subtask 4.1: Wire departments to FloorManager (re-use Story 7.7 task router)
  - [ ] Subtask 4.2: Add department-to-department messaging for results passing
  - [ ] Subtask 4.3: Test end-to-end flow from FloorManager dispatch to report generation

## Dev Notes

### Previous Story Intelligence

**From Story 7.7 (Concurrent Task Routing):**
- TaskRouter module created with concurrent dispatch to 5 departments
- Redis Streams-based task queues using established patterns
- TaskPriority enum (HIGH/MEDIUM/LOW) implemented
- Result aggregation with parallelism overhead calculation
- API endpoints: /concurrent, /tasks/status, /concurrent/execute

**Key Reuse Required:** Story 7.8 MUST use the TaskRouter from Story 7.7 for:
- Dispatching Risk, Trading, Portfolio tasks concurrently
- Tracking task status via Redis Streams
- Result aggregation patterns

### Architecture Context

**Department Heads Current State (from audit - Story 7.0):**
- RiskHead: `src/agents/departments/heads/risk_head.py` — STUB (tool definitions only, no implementation)
- TradingHead: `src/agents/departments/heads/execution_head.py` — STUB (tool definitions only, no implementation)
- PortfolioHead: `src/agents/departments/heads/portfolio_head.py` — STUB (tool definitions only, no implementation)

**Base Class Patterns (Story 7.0, 7.1, 7.2, 7.3):**
- All department heads inherit from `DepartmentHead` base class in `src/agents/departments/heads/base.py`
- Tool registry integration: `self.tool_registry.get_tools_for_department(department)`
- Redis mail service: `get_redis_mail_service()` for inter-department communication
- Memory manager: `DepartmentMemoryManager` for isolated markdown memory
- Worker spawning: `self.spawner.spawn(SubAgentConfig)` for Haiku-tier parallel workers

**FloorManager Current State:**
- File: `src/agents/departments/floor_manager.py`
- Has references to all 5 department heads (research_head, development_head, risk_head, trading_head, portfolio_head)
- Uses TaskRouter from Story 7.7 for concurrent dispatch
- Already uses RedisDepartmentMailService for department messaging

### Technical Requirements

1. **Risk Department Implementation:**
   - Integrate with existing backtest engine at `src/api/backtest_endpoints.py` (Story 4.4)
   - Execute all 6 backtest modes: scalping, momentum, mean_reversion, breakout, swing, position
   - Calculate metrics: Sharpe ratio, max drawdown, win rate
   - Pass/fail threshold: ≥4/6 modes pass with criteria (Sharpe≥1.0, DD≤15%, WR≥50%)
   - Return structured verdict: `{pass: bool, modes_passed: int, mode_results: [...], metrics: {...}}`

2. **Trading Department Implementation:**
   - Read paper trading state from existing endpoints (Story 3.4: LiveTradingCanvas)
   - Track per-trade P&L: entry_price, current_price, pnl_pct, pnl_value
   - Aggregate metrics: win_loss_ratio, avg_drawdown, avg_hold_time, regime_correlation
   - Push updates to Copilot via Redis Streams (topic: `copilot:updates`)
   - Update interval: configurable (default: 60 seconds)

3. **Portfolio Department Implementation:**
   - API endpoint: `GET /api/portfolio/report`
   - Query broker accounts via `src/router/broker_registry.py` (Story 9.1 already has schema)
   - Calculate attribution: per-strategy P&L, per-broker P&L, account drawdown
   - Response schema:
     ```python
     {
       "total_equity": float,
       "pnl_attribution": {
         "by_strategy": [{"strategy": str, "pnl": float, "percentage": float}],
         "by_broker": [{"broker": str, "pnl": float, "percentage": float}]
       },
       "drawdown_by_account": [{"account_id": str, "drawdown_pct": float}]
     }
     ```

4. **Redis Stream Patterns (re-use from Story 7.6, 7.7):**
   - Task queue: `task:dept:{dept_name}:{session_id}:queue`
   - Mail: `mail:dept:{dept_name}:{workflow_id}`
   - Consumer groups: `task:dept:{dept_name}:group`, `mail:dept:{dept_name}:group`
   - Ack pattern: `xack()` for task completion

5. **Key Namespacing Rules (from architecture):**
   - NEVER use underscores in key names
   - Use lowercase, colon-separated: `task:dept:risk:session_abc:queue`

### Source Tree Components to Touch

**Must Modify:**
- `src/agents/departments/heads/risk_head.py` — Add real backtest evaluation methods
- `src/agents/departments/heads/execution_head.py` (TradingHead) — Add paper trading monitoring
- `src/agents/departments/heads/portfolio_head.py` — Add portfolio report generation
- `src/api/portfolio_endpoints.py` — [NEW] Portfolio report API endpoint

**May Need to Update:**
- `src/agents/departments/floor_manager.py` — Add risk/trading/portfolio dispatch methods
- `src/agents/departments/task_router.py` — Ensure compatibility with new department implementations
- `src/api/floor_manager_endpoints.py` — Add portfolio report endpoint exposure

**Backend Infrastructure (already exists - wire only):**
- Backtest engine: `src/api/backtest_endpoints.py` (Story 4.4, tested)
- Broker registry: `src/router/broker_registry.py` (Story 2.2)
- Paper trading state: Story 3.7 (MorningDigestCard)
- Risk parameters: `src/database/models/risk_params.py`

**Tests:**
- `tests/agents/departments/test_risk_head.py` — [NEW] Risk department tests
- `tests/agents/departments/test_trading_head.py` — [NEW] Trading department tests
- `tests/agents/departments/test_portfolio_head.py` — [NEW] Portfolio department tests
- `tests/api/test_portfolio_report.py` — [NEW] Portfolio API tests

### Testing Standards

From project-context.md:
- Test file pattern: `test_*.py` in `/tests` directory
- Class pattern: `Test*`
- Function pattern: `test_*`
- Asyncio mode: `auto` (no need for `@pytest.mark.asyncio` decorator)
- Run from project root: `pytest`

### Project Structure Notes

- **File limit:** Python files under 500 lines (may need to split large department heads if they exceed)
- **Import convention:** Use `src.` prefix from project root
- **Pydantic v2:** Use `model_validate()`, NOT `parse_obj()`; use `model_dump()`, NOT `dict()`
- **Redis patterns:** Reuse established patterns from Story 7.6, 7.7

### References

- [Source: _bmad-output/planning-artifacts/epics.md#Story 7.8]
- [Source: _bmad-output/planning-artifacts/architecture.md#Risk Engine]
- [Source: _bmad-output/planning-artifacts/architecture.md#Strategy Router]
- [Source: _bmad-output/implementation-artifacts/7-7-concurrent-task-routing-5-simultaneous-tasks.md]
- [Source: _bmad-output/planning-artifacts/architecture.md#Redis Stream]

### Questions / Clarifications

1. What is the exact backtest mode execution order? (sequential or parallel?)
2. Should paper trading updates go to Copilot via Redis Streams or HTTP/WebSocket?
3. Is there a specific broker registry schema to follow for portfolio attribution?
4. Should failed backtest modes trigger a retry or immediately fail?

## Dev Agent Record

### Agent Model Used

Claude Mini Max 3.5 (via Claude Code)

### Debug Log References

- Implementation builds on Story 7.7 concurrent task routing foundation
- Wire to existing backtest engine from Story 4.4
- Re-use Redis Streams patterns from Story 7.6, 7.7

### Completion Notes List

- Implemented RiskHead with real backtest evaluation across all 6 modes (VANILLA, SPICED, VANILLA_FULL, SPICED_FULL, MODE_B, MODE_C)
- Implemented pass/fail verdict logic with thresholds: Sharpe >=1.0, max_drawdown <=15%, win_rate >=50%
- Pass verdict requires >= 4/6 modes to pass (per AC #1)
- Implemented TradingHead with paper trading monitoring (P&L tracking, win/loss ratio, drawdown, hold time, regime correlation)
- Implemented Copilot periodic updates structure (topic: copilot:updates)
- Implemented PortfolioHead with complete report generation
- Created new portfolio API endpoints: GET /api/portfolio/report, /equity, /pnl/strategy, /pnl/broker, /drawdowns
- Created tests for all three department heads
- Created API tests for portfolio endpoints

### File List

- Modified: `src/agents/departments/heads/risk_head.py` - Added real backtest evaluation
- Modified: `src/agents/departments/heads/execution_head.py` - Added paper trading monitoring
- Modified: `src/agents/departments/heads/portfolio_head.py` - Added portfolio report generation
- Created: `src/api/portfolio_endpoints.py` - New portfolio API endpoints
- Modified: `src/api/server.py` - Registered portfolio router
- Created: `tests/agents/departments/heads/test_risk_head.py` - RiskHead tests
- Created: `tests/agents/departments/heads/test_trading_head.py` - TradingHead tests
- Created: `tests/agents/departments/heads/test_portfolio_head.py` - PortfolioHead tests
- Created: `tests/api/test_portfolio_report.py` - Portfolio API tests
