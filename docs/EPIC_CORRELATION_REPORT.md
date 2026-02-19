# QuantMindX Epic Export — Master Correlation Report
**Generated:** 2026-02-19  
**Verified Against:** Live codebase (not the stale IMPLEMENTATION_REPORT.md from Feb 12)  
**Purpose:** Map all Traycer epic exports → actual source files, identify TRUE remaining gaps.

---

## ⚠️ Important Note on IMPLEMENTATION_REPORT.md

The `IMPLEMENTATION_REPORT.md` (dated Feb 12) flagged many "missing" items that have since been **fully implemented**. This report supersedes it with verified findings from direct source inspection.

---

## 📦 Epic Export Index

All epics live in `docs/epic-export/`. All spec files are **Traycer.AI stubs** — the actual spec content lives in the Traycer.AI platform. Only summaries and ticket statuses are available locally.

| Epic ID | Name | Specs | Tickets | True Coverage |
|---------|------|-------|---------|---------------|
| `epic-d81a1bce` | Trading Infrastructure Core | 11 | 6 | ✅ **~90%** |
| `epic-af37afc7` | Deployment Pipeline & Infrastructure | 12 | 0 | ✅ **~70%** |
| `epic-0aabe215` | Review System v3 Architecture | 5 | 0 | ✅ **~65%** |
| `epic-5b080ee3` | Agentic Sidebar UI Component | 3 | 0 | ✅ **~80%** |
| `epic-238cb09f` | Prop Firm Integration | 0 | 0 | ⚠️ **~60%** (wiring gap) |

---

## ✅ Items Previously Flagged as Missing — Now CONFIRMED IMPLEMENTED

The following were listed as critical gaps in the old report, but have been implemented:

| Item | File | Status |
|------|------|--------|
| `TradeJournal` DB table | `src/database/models.py:601-675` | ✅ DONE |
| `BotCircuitBreaker` DB table | `src/database/models.py:558-598` | ✅ DONE |
| `StrategyFolder` DB table | `src/database/models.py:381-424` | ✅ DONE |
| `HouseMoneyState` DB table | `src/database/models.py:517-555` | ✅ DONE |
| `SessionDetector` | `src/router/session_detector.py` | ✅ DONE |
| `sessions.py` | `src/router/sessions.py` | ✅ DONE |
| Balance-based bot limits | `src/router/dynamic_bot_limits.py` + `commander.py:312-329` | ✅ DONE (DynamicBotLimiter) |
| `ProgressiveKillSwitch` | `src/router/progressive_kill_switch.py` | ✅ DONE |
| House Money DB integration | `enhanced_governor.py:489-547` | ✅ DONE |
| Daily state DB load | `enhanced_governor.py:409-452` | ✅ DONE |
| Pip value from broker registry | `enhanced_governor.py:380-407` | ✅ DONE |
| Broker commission → Kelly | `enhanced_kelly.py:274-311` | ✅ DONE (auto-lookup) |
| NPRD file watcher | `src/router/nprd_watcher.py` | ✅ DONE |
| TRD file watcher | `src/router/trd_watcher.py` | ✅ DONE |
| Session-aware commander | `commander.py:230-296` | ✅ DONE |
| Multi-timeframe sentinel | `src/router/multi_timeframe_sentinel.py` | ✅ DONE |
| Account-type governor in engine | `engine.py:211-215` (EnhancedGovernor default) | ✅ DONE |
| Fee monitor | `src/router/fee_monitor.py` | ✅ DONE |
| Virtual balance (demo mode) | `src/router/virtual_balance.py` | ✅ DONE |
| Promotion manager | `src/router/promotion_manager.py` | ✅ DONE |
| Bot cloner | `src/router/bot_cloner.py` | ✅ DONE |
| Lifecycle manager | `src/router/lifecycle_manager.py` | ✅ DONE |
| Strategy monitor | `src/router/strategy_monitor.py` | ✅ DONE |
| Market scanner | `src/router/market_scanner.py` | ✅ DONE |

---

## 🔍 True Remaining Gaps (Verified)

### 🔴 FIX-01: PropGovernor NOT wired into engine.py

**The Gap:** `engine.py:211-215` always uses `EnhancedGovernor`, never `PropGovernor`, regardless of account type.

```python
# Current (engine.py:211-215)
if use_kelly_governor:
    self.governor = EnhancedGovernor()   # ← Always this
else:
    self.governor = Governor()           # ← Or this

# Required: Account-type selection
account_type = account_config.get('type', 'normal') if account_config else 'normal'
if account_type == 'prop_firm':
    self.governor = PropGovernor(account_config['account_id'])  # ← Never reached
else:
    self.governor = EnhancedGovernor(account_id=account_config.get('account_id') if account_config else None)
```

**Impact:** `PropGovernor` (with quadratic throttle, tiered risk, news guard) **never activates**. All prop firm accounts are treated as standard accounts.

**File:** `src/router/engine.py:187-215`  
**Effort:** 1 hour

---

### 🔴 FIX-02: PropGovernor.calculate_risk() signature mismatch

**The Gap:** `PropGovernor.calculate_risk()` (prop/governor.py:37) accepts only `(regime_report, trade_proposal)`, but `Commander.run_auction()` calls `self._governor.calculate_risk(regime_report, trade_proposal, account_balance, routed_broker_id, account_id=account_id, mode=mode)` — **PropGovernor will TypeError crash** if it ever gets called.

```python
# PropGovernor (prop/governor.py:37) — WRONG SIGNATURE
def calculate_risk(self, regime_report: 'RegimeReport', trade_proposal: dict) -> RiskMandate:

# EnhancedGovernor (enhanced_governor.py:134) — CORRECT SIGNATURE
def calculate_risk(self, regime_report, trade_proposal, account_balance=None, broker_id=None, account_id=None, mode="live", **kwargs) -> RiskMandate:
```

**Impact:** If `PropGovernor` is wired in (FIX-01), it will crash on the first auction because the signature doesn't accept the extra kwargs.

**File:** `src/router/prop/governor.py:37`  
**Effort:** 1 hour

---

### 🟠 FIX-03: Kelly `pip_value` has a default (legacy function only)

The `EnhancedKellyCalculator.calculate()` method (line 202) still has `pip_value: float = 10.0` as a default. The broker auto-lookup (lines 274-311) overrides this when `broker_id` and `symbol` are provided, so in practice this **works correctly** for the main path.

However, the legacy `enhanced_kelly_position_size()` function (line 92) also has `pip_value: float = 10.0` with no auto-lookup capability.

**Impact:** Low — the auto-lookup path in `EnhancedKellyCalculator.calculate()` correctly overrides this. Only pure function callers miss broker fees.

**File:** `src/position_sizing/enhanced_kelly.py:92`  
**Effort:** 30 min  

---

### 🟡 FIX-04: No explicit Mode B / Mode C backtest separation

**The Gap:** 4 backtest variants exist (VANILLA/SPICED/VANILLA_FULL/SPICED_FULL), but the spec requires explicit **Mode A** (EA only), **Mode B** (EA + Kelly), **Mode C** (EA + Full System: Kelly + Governor + Router).

Current modes:
- VANILLA ≈ Mode A ✅
- SPICED ≈ Mode A + regime filter (not Mode B)
- No Mode B (Kelly isolated)
- No Mode C (full system)

**File:** `src/backtesting/mode_runner.py`  
**Effort:** 4 hours

---

### 🟡 FIX-05: No GitHub Actions CI/CD pipeline

**The Gap:** `.githooks/` exists (pre-commit hooks), but no `.github/workflows/` automated CI/CD.

**Required:** At minimum:
- Build check workflow
- Unit test workflow
- Frontend Svelte build check

**File:** `.github/workflows/ci.yml` (NEW)  
**Effort:** 2 hours

---

### 🟡 FIX-06: No Redis caching layer

**The Gap:** Cache strategy mentioned in `epic-0aabe215` spec 04. Currently only Parquet file caching exists in `src/data/data_manager.py`. No Redis.

**Impact:** Medium — affects API response times, not correctness.

**File:** `src/cache/` (NEW)  
**Effort:** 4 hours

---

### 🟢 FIX-07: No complete OpenAPI spec exported

**The Gap:** FastAPI auto-generates OpenAPI at `/docs` but no static spec file is committed to repo.

**File:** `docs/api/openapi.json` (NEW)  
**Effort:** 30 min (run `curl http://localhost:8000/openapi.json > docs/api/openapi.json`)

---

### 🟢 FIX-08: WCAG 2.1 audit incomplete

**The Gap:** Some A11y warnings in `quantmind-ide` Svelte build. WCAG 2.1 compliance not formally verified for `epic-5b080ee3`.

**File:** All `.svelte` files  
**Effort:** 2-4 hours

---

### 🟢 FIX-09: Vault/secrets management not implemented

**The Gap:** `epic-af37afc7` spec 05 mentions HashiCorp Vault for secrets. Currently using `.env` files.

**Impact:** Low for current VPS deployment, needed for multi-tenant or cloud scaling.

**Effort:** 8 hours

---

## 🗓️ Immediate Action Items (Fixes for Broken Logic)

### Sprint 1 — Critical (Do Now)

**FIX-01 + FIX-02: Wire PropGovernor into engine.py (with signature fix)**

```python
# 1. engine.py — Add account_config parameter
def __init__(self, use_smart_kill: bool = True, use_kelly_governor: bool = True, 
             use_multi_timeframe: bool = True, multi_timeframes: Optional[List[Timeframe]] = None,
             account_config: Optional[dict] = None):  # ADD THIS
    ...
    # Replace lines 211-215:
    account_type = (account_config or {}).get('type', 'normal')
    account_id = (account_config or {}).get('account_id')
    if account_type == 'prop_firm' and account_id:
        from src.router.prop.governor import PropGovernor
        self.governor = PropGovernor(account_id)
    elif use_kelly_governor:
        self.governor = EnhancedGovernor(account_id=account_id)
    else:
        self.governor = Governor()
```

```python
# 2. prop/governor.py — Fix calculate_risk signature
def calculate_risk(self, regime_report: 'RegimeReport', trade_proposal: dict,
                   account_balance: Optional[float] = None,
                   broker_id: Optional[str] = None,
                   account_id: Optional[str] = None,
                   mode: str = "live",
                   **kwargs) -> RiskMandate:
    """..."""
    # Extract from kwargs or trade_proposal as needed
    return super().calculate_risk(regime_report, trade_proposal)  # existing logic
```

### Sprint 2 — High (This Week)

- **FIX-04**: Add Mode B and Mode C to backtest mode_runner.py
- **FIX-05**: Create `.github/workflows/ci.yml` 
- **FIX-07**: Export OpenAPI spec to `docs/api/openapi.json`

### Sprint 3 — Medium (Next Week)

- **FIX-06**: Redis caching layer
- **FIX-08**: WCAG 2.1 audit pass

---

## 🗂️ Full Code Coverage Map

### `epic-d81a1bce` — Trading Infrastructure Core

| Spec | Code Files | Status |
|------|-----------|--------|
| 01 Market Data Feed | `mcp-metatrader5-server/src/mcp_mt5/streaming.py`, `src/router/sessions.py` | ✅ |
| 02 Order Management | `src/router/engine.py`, `src/router/governor.py` | ✅ |
| 03 Risk Engine | `src/router/progressive_kill_switch.py`, `src/router/dynamic_bot_limits.py` | ✅ |
| 04 Execution Engine | `src/position_sizing/enhanced_kelly.py`, `src/router/enhanced_governor.py` | ✅ |
| 05 Strategy Framework | `src/backtesting/mode_runner.py`, `walk_forward.py`, `monte_carlo.py` | ⚠️ Mode B/C missing |
| 06 Portfolio Manager | `src/router/house_money.py`, `enhanced_governor.py` | ✅ |
| 07 Analytics Pipeline | `src/monitoring/`, Prometheus setup | ✅ |
| 08 Event Bus | `src/queues/` | ⚠️ Partial |
| 09 Data Storage | `src/database/models.py`, Parquet caching | ✅ |
| 10 API Gateway | `src/api/` (37 files), `phase5_endpoints.py` | ✅ |
| 11 Compliance Reporting | `src/database/models.py:TradeJournal`, `src/router/trade_logger.py` | ✅ |

### `epic-af37afc7` — Deployment Pipeline

| Spec | Code Files | Status |
|------|-----------|--------|
| 01 Docker Multi-Stage | `docker/`, `docker-compose*.yml` | ✅ |
| 02 CI/CD Pipeline | ❌ `.github/workflows/` missing | ❌ FIX-05 |
| 03 Kubernetes | `QuantMindX_Production_Deployment_Guide.md` | ⚠️ Docs only |
| 04 Environment Config | `.env`, `config/`, `.env.production` | ✅ |
| 05 Secrets Management | `.env` files only (no Vault) | ⚠️ FIX-09 |
| 06 Monitoring | `monitoring/`, Prometheus | ✅ |
| 07 Centralized Logging | File-based only (no ELK/Loki) | ⚠️ |
| 08 Auto-Scaling | Not implemented | ⚠️ Low priority |
| 09 Disaster Recovery | Manual process | ⚠️ |
| 10 Security Hardening | CORS configured, basic auth | ⚠️ |
| 11-12 Network/Runbook | `QuantMindX_Production_Deployment_Guide.md` | ✅ |

### `epic-0aabe215` — Review System v3

| Spec | Code Files | Status |
|------|-----------|--------|
| 01 Architecture | `docs/architecture/system_architecture.md` | ✅ |
| 02 API Design | `src/api/`, `docs/api/` | ⚠️ No static OpenAPI spec |
| 03 Database Schema | `src/database/models.py` (all tables present) | ✅ |
| 04 Caching | Parquet only (no Redis) | ⚠️ FIX-06 |
| 05 Performance | Partial (no benchmarks) | ⚠️ |

### `epic-5b080ee3` — Agentic Sidebar UI

| Spec | Code Files | Status |
|------|-----------|--------|
| 01 Component Architecture | `AgentPanel.svelte`, `CopilotPanel.svelte`, `Sidebar.svelte` | ✅ |
| 02 Conversation Flow | `CopilotPanel.svelte` | ✅ |
| 03 Accessibility WCAG 2.1 | All Svelte files | ⚠️ FIX-08 |

### `epic-238cb09f` — Prop Firm Integration

| Item | Code Files | Status |
|------|-----------|--------|
| PropGovernor logic | `src/router/prop/governor.py` | ✅ Logic complete |
| PropGovernor wiring | `src/router/engine.py` | ❌ **FIX-01** |
| PropGovernor signature | `src/router/prop/governor.py:37` | ❌ **FIX-02** |
| Kelly presets | `src/position_sizing/kelly_config.py` | ✅ |
| DB tables | `src/database/models.py` | ✅ |
| UI for prop accounts | `quantmind-ide/src/lib/` | ⚠️ Partial |

---

## 📊 Codebase Health Summary (2026-02-19)

```
TRUE REMAINING GAPS (after verification):

CRITICAL (broken if prop firm account used):
  ✗ PropGovernor not wired into engine.py (FIX-01)
  ✗ PropGovernor.calculate_risk() signature mismatch (FIX-02)

HIGH (missing features):
  ✗ Backtest Mode B/C not separated (FIX-04)
  ✗ No CI/CD pipeline (FIX-05)

MEDIUM (nice to have):
  ✗ No Redis cache (FIX-06)
  ✗ WCAG 2.1 audit (FIX-08)

LOW (housekeeping):
  ✗ No static OpenAPI spec committed (FIX-07)
  ✗ No Vault/secrets management (FIX-09)

WORKING WELL (verified):
  ✓ EnhancedGovernor + Kelly + broker auto-lookup
  ✓ House Money Effect (DB-backed)
  ✓ Session detector (London/NY/Asian/Pacific)
  ✓ Dynamic bot limits (DynamicBotLimiter — balance-based)
  ✓ Progressive kill switch (5-tier)
  ✓ Trade journal (full "why" context)
  ✓ 4-variant backtesting
  ✓ Multi-timeframe sentinel
  ✓ NPRD/TRD file watchers (auto-trigger)
  ✓ Strategy folders DB table
  ✓ Bot circuit breaker
  ✓ Virtual balance (demo mode)
  ✓ Promotion manager (paper→demo→live)
  ✓ Routing matrix (account-to-broker)
  ✓ Fee monitor
  ✓ Bot cloner
  ✓ Lifecycle manager
  ✓ Prometheus monitoring
  ✓ Docker deployment config
  ✓ 70+ UI components (SvelteKit)
  ✓ Agent system (Copilot/Analyst/QuantCode)
```

---

*This report was verified against the live source by direct file inspection on 2026-02-19. It supersedes all previous gap analyses.*
