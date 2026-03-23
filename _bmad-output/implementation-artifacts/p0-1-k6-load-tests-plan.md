# P0-1: k6 Load Tests — Implementation Plan

## Overview
This plan covers the creation of k6 load test scripts to validate 50-bot capacity and establish latency baselines for the QUANTMINDX trading platform.

---

## Phase 1: Discovery

### 1.1 API Endpoint Audit

**Trading Endpoints** (`src/api/trading/routes.py`):
| Endpoint | Method | Path | Auth | Description |
|----------|--------|------|------|-------------|
| Run Backtest | POST | `/api/v1/backtest/run` | Yes | Execute backtest |
| Get Backtest Results | GET | `/api/v1/backtest/results/{id}` | Yes | Retrieve results |
| Get Trading Status | GET | `/api/v1/trading/status` | Yes | Current trading status |
| Get Bot Status | GET | `/api/v1/trading/bots` | Yes | All bot statuses |
| Close Position | POST | `/api/v1/trading/close` | Yes | Close single position |
| Emergency Stop | POST | `/api/v1/trading/emergency_stop` | Yes | Kill switch trigger |
| Broker Connect | POST | `/api/v1/trading/broker/connect` | Yes | MT5 connection |

**Kill Switch Endpoints** (`src/api/kill_switch_endpoints.py`):
| Endpoint | Method | Path | Auth | Description |
|----------|--------|------|------|-------------|
| Kill Switch Status | GET | `/api/kill-switch/status` | No | Current kill switch state |
| Kill Switch Trigger | POST | `/api/kill-switch/trigger` | Yes | Trigger tier 1/2/3 |
| Kill Switch Health | GET | `/api/kill-switch/health` | No | Quick health check |
| Kill Switch Alerts | GET | `/api/kill-switch/alerts` | No | Current alerts |
| Kill Switch Audit | GET | `/api/kill-switch/audit` | No | Audit logs |

**Health Endpoints** (`src/api/health_endpoints.py`):
| Endpoint | Method | Path | Auth | Description |
|----------|--------|------|------|-------------|
| Full Health | GET | `/health` | No | All services health |
| API Health | GET | `/health/api` | No | API liveness |
| MT5 Health | GET | `/health/mt5` | No | MT5 bridge status |
| Database Health | GET | `/health/database` | No | DB connection |
| Redis Health | GET | `/health/redis` | No | Redis cache status |

**Risk Endpoints** (`src/api/risk_endpoints.py`):
| Endpoint | Method | Path | Auth | Description |
|----------|--------|------|------|-------------|
| Regime | GET | `/api/risk/regime` | No | Current regime |
| Risk Params | GET | `/api/risk/params/{tag}` | No | Risk parameters |
| Compliance | GET | `/api/risk/compliance` | No | Circuit breaker state |
| Physics | GET | `/api/risk/physics` | No | Sensor outputs |

**WebSocket Endpoints** (`src/api/websocket_endpoints.py`):
| Endpoint | Path | Description |
|----------|------|-------------|
| Main WS | `WS /ws` | General WebSocket |
| Trading WS | `WS /ws/trading` | Trading data stream |
| Chart WS | `WS /ws/chart/{sym}/{tf}` | Chart streaming |

**Floor Manager Endpoints** (`src/api/floor_manager_endpoints.py`):
| Endpoint | Method | Path | Auth | Description |
|----------|--------|------|------|-------------|
| Chat | POST | `/api/floor-manager/chat` | Yes | Copilot chat |
| Status | GET | `/api/floor-manager/status` | No | Floor manager status |
| Task Submit | POST | `/api/floor-manager/task` | Yes | Submit task |

**Agent Endpoints** (`src/api/agent_session_endpoints.py`):
| Endpoint | Method | Path | Auth | Description |
|----------|--------|------|------|-------------|
| Agent Stream | GET | `/api/agents/stream` | Yes | SSE agent events |
| Agent Health | GET | `/api/agents/{id}/health` | No | Agent health check |

### 1.2 Authentication Requirements
- **No Auth**: Health, Risk (read), Kill Switch (read), Floor Manager (status)
- **Auth Required**: Trading, Kill Switch (trigger), Chat, Agent operations
- Auth tokens passed via `Authorization: Bearer <token>` header

### 1.3 Request/Response Shapes

**Trade Execution** (`POST /api/v1/trading/close`):
```json
Request: { "bot_id": "string", "ticket": 12345 }
Response: { "success": true, "filled_price": 1.1234, "slippage": 0.0001, "pnl": 10.50 }
```

**Kill Switch Trigger** (`POST /api/kill-switch/trigger`):
```json
Request: { "tier": 1, "activator": "load-test" }
Response: { "success": true, "tier": 1, "audit_log_id": "uuid", "activated_at_utc": "ISO8601" }
```

**Risk Params** (`GET /api/risk/params/{account_tag}`):
```json
Response: { "account_tag": "prop-firm-001", "daily_loss_cap_pct": 5.0, "kelly_fraction": 0.5, ... }
```

---

## Phase 2: k6 Script Development

### 2.1 Script Structure
```
k6/
├── README.md                    # Usage instructions
├── config/
│   └── thresholds.json          # P50/P95/P99 targets
├── scenarios/
│   ├── trading-scenarios.js      # Trading endpoint tests
│   ├── kill-switch-scenarios.js # Kill switch tests
│   ├── health-scenarios.js       # Health endpoint tests
│   ├── risk-scenarios.js         # Risk endpoint tests
│   └── websocket-scenarios.js    # WebSocket tests
├── scripts/
│   ├── shared/
│   │   ├── auth.js              # Auth helpers
│   │   ├── checks.js             # Custom k6 checks
│   │   └── payload-generator.js  # Test data generators
│   └── smoke-test.js             # Quick smoke test
├── load-profiles/
│   ├── ramp-up.js                # Gradual load increase
│   ├── sustained-load.js         # Steady 50-bot load
│   └── spike-test.js             # Sudden load spikes
└── k6.conf.js                   # Main configuration
```

### 2.2 Load Profiles

**Profile 1: Ramp-Up (Warm-up)**
- 0-30s: 1-10 VUs (virtual users)
- 30-60s: 10-25 VUs
- 60-90s: 25-50 VUs
- Purpose: Establish baseline latency under increasing load

**Profile 2: Sustained Load (50-Bot Capacity Test)**
- 0-30s: Ramp to 50 VUs
- 30-120s: Hold at 50 VUs (simulates 50 concurrent bots)
- 120-150s: Ramp down
- Purpose: Validate 50-bot capacity requirement

**Profile 3: Spike Test**
- 0-10s: 10 VUs baseline
- 10-15s: Spike to 100 VUs
- 15-25s: Drop to 10 VUs
- 25-35s: Another spike to 75 VUs
- Purpose: Test circuit breaker and auto-scaling

### 2.3 Circuit Breaker Test Scenario
Tests bot circuit breaker triggers under load:
1. Send sustained high-load requests to trading endpoints
2. Monitor error rates and latency
3. Verify circuit breaker activates at threshold (5% error rate or P99 > 2s)
4. Validate graceful degradation

---

## Phase 3: Execution Infrastructure

### 3.1 k6 Runner Setup
- k6 binary installed locally or via Docker
- Environment variables:
  - `K6_BASE_URL`: Base URL for API (default: http://localhost:8000)
  - `K6_AUTH_TOKEN`: Authentication token for protected endpoints
  - `K6_DURATION`: Test duration override
  - `K6_VUS`: Virtual users override

### 3.2 Results Collection
- JSON output to `_bmad-output/test-artifacts/k6-results.json`
- HTML report generation: `k6 report --output k6-report.html`
- Prometheus remote write for continuous monitoring (optional)

### 3.3 Test Execution Commands
```bash
# Smoke test (quick validation)
k6 run k6/scripts/smoke-test.js

# Full load test with all profiles
k6 run k6/k6.conf.js

# Specific profile
k6 run k6/k6.conf.js --profile sustained

# With custom threshold
k6 run k6/k6.conf.js --threshold p99=800ms
```

---

## Phase 4: Reporting

### 4.1 Latency Report Metrics
| Metric | Target | Critical |
|--------|--------|----------|
| P50 Latency | < 100ms | < 500ms |
| P95 Latency | < 250ms | < 500ms |
| P99 Latency | < 500ms | < 1000ms |
| Error Rate | < 1% | < 5% |

### 4.2 Report Location
- File: `_bmad-output/test-artifacts/k6-load-test-report.md`
- Contents:
  - Test configuration (date, duration, VUs)
  - Latency percentiles per endpoint
  - Error rates and types
  - Circuit breaker activation events
  - Pass/Fail summary against targets

---

## Acceptance Criteria Validation

| Criteria | Test Method | Pass Condition |
|----------|-------------|----------------|
| k6 script covers all critical endpoints | Code review | All endpoints in audit covered |
| P50/P95/P99 latency < 500ms | k6 metrics | p(95) < 500ms for all endpoints |
| Bot circuit breaker triggers under load | Circuit breaker scenario | CB activates within 10s of threshold breach |
| Report generated and filed | File check | Report exists at specified path |

---

## Dependencies
- k6 >= 0.50.0
- Node.js >= 18 (for payload generators)
- Docker (optional, for containerized execution)
