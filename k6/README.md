# k6 Load Testing for QUANTMINDX

This directory contains k6 load test scripts for validating the QUANTMINDX trading platform's 50-bot capacity and establishing latency baselines.

## Directory Structure

```
k6/
├── README.md                    # This file
├── k6.conf.js                   # Main k6 configuration
├── config/
│   └── thresholds.json          # P50/P95/P99 latency targets
├── scenarios/
│   ├── trading-scenarios.js     # Trading endpoint tests
│   ├── kill-switch-scenarios.js # Kill switch tests
│   ├── health-scenarios.js      # Health endpoint tests
│   ├── risk-scenarios.js        # Risk endpoint tests
│   └── websocket-scenarios.js   # WebSocket tests
├── scripts/
│   ├── shared/
│   │   ├── auth.js              # Auth helpers
│   │   ├── checks.js            # Custom k6 checks
│   │   └── payload-generator.js # Test data generators
│   └── smoke-test.js             # Quick smoke test
└── load-profiles/
    ├── ramp-up.js               # Gradual load increase
    ├── sustained-load.js        # Steady 50-bot load
    └── spike-test.js            # Sudden load spikes
```

## Prerequisites

- k6 >= 0.50.0
- Node.js >= 18 (for payload generators, optional)

## Quick Start

### 1. Set Environment Variables

```bash
export K6_BASE_URL=http://localhost:8000
export K6_AUTH_TOKEN=your-auth-token-here  # Optional, for auth-protected endpoints
```

### 2. Run Smoke Test (Quick Validation)

```bash
k6 run k6/scripts/smoke-test.js
```

### 3. Run Full Load Test

```bash
k6 run k6/k6.conf.js
```

### 4. Run Specific Profile

```bash
# Sustained 50-bot load
k6 run k6/k6.conf.js --env PROFILE=sustained

# Spike test
k6 run k6/k6.conf.js --env PROFILE=spike

# Ramp-up
k6 run k6/k6.conf.js --env PROFILE=rampup
```

### 5. Generate HTML Report

```bash
k6 run k6/k6.conf.js --out json=results.json
k6 report results.json --output report.html
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| K6_BASE_URL | http://localhost:8000 | Base URL for the API |
| K6_AUTH_TOKEN | empty | Auth token for protected endpoints |
| K6_DURATION | 2m | Test duration |
| K6_VUS | 50 | Number of virtual users |
| PROFILE | sustained | Load profile: sustained, spike, rampup |

### Thresholds (from config/thresholds.json)

| Metric | Target | Critical |
|--------|--------|----------|
| P50 Latency | < 100ms | < 500ms |
| P95 Latency | < 250ms | < 500ms |
| P99 Latency | < 500ms | < 1000ms |
| Error Rate | < 1% | < 5% |

## Test Coverage

### Covered Endpoints

- **Trading**: backtest, trading status, bot status, position close, emergency stop
- **Kill Switch**: status, trigger (tiers 1-3), health, alerts, audit
- **Health**: full health, API, MT5, database, Redis
- **Risk**: regime, risk params, compliance, physics
- **WebSocket**: /ws, /ws/trading
- **Floor Manager**: chat, status, task submit
- **Agents**: stream, health

## Output

Results are written to:
- Console: Real-time metrics during test
- JSON: `_bmad-output/test-artifacts/k6-results.json` (if configured)
- HTML: User-specified output path

## Circuit Breaker Test

The spike test profile specifically tests bot circuit breaker activation:
1. Sends sustained high-load requests to trading endpoints
2. Monitors error rates and latency
3. Verifies circuit breaker activates at threshold (5% error rate or P99 > 2s)
4. Validates graceful degradation

## Notes

- Auth tokens should be set via environment variable, never hardcoded
- WebSocket tests require k6 version >= 0.46.0
- For CI/CD integration, use JSON output and parse results programmatically
