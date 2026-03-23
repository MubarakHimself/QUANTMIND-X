# APM On-Call Runbook

## QuantMindX APM Monitoring Runbook

This runbook provides guidance for on-call engineers responding to APM alerts in the QuantMindX trading platform.

---

## Alert Escalation Matrix

| Alert Name | Condition | Severity | On-Call | Escalation |
|------------|-----------|----------|---------|------------|
| HighAPIErrorRate | error_rate > 5% for 5m | P1 Critical | Primary | → Secondary → Engineering Lead |
| HighAPILatency | p95_latency > 1s for 5m | P2 Warning | Primary | → Secondary |
| MT5Disconnected | mt5_status == 0 for > 30s | P1 Critical | Primary | → Trading Team → Engineering Lead |
| HighChaosScore | chaos_score > 0.8 for 5m | P2 Warning | Primary | → Secondary |
| BotQuarantineSpike | quarantine_count > 5 | P2 Warning | Primary | → Secondary |
| TickStreamDrops | tick_drops > 10/min | P2 Warning | Primary | → Secondary |
| DBConnectionExhaustion | db_connections > 80% | P1 Critical | Primary | → DBA → Engineering Lead |

---

## Alert Response Procedures

### APM-001: High API Error Rate

**Condition:** `error_rate > 5% for 5 minutes`

**Impact:** Users cannot execute trades or access platform features

**Diagnosis:**
```bash
# Check recent errors in Loki
{c job="quantmind-api" } |= "ERROR" | json | level="error"

# Check trace for failed requests
# Search Grafana Tempo for spans with status=error
```

**Common Causes:**
1. Downstream service timeout (check MT5 bridge, database)
2. Code deployment with regression
3. Database connection pool exhaustion
4. Redis/cache unavailable

**Resolution:**
1. Check if recent deployment occurred
2. Review error traces in Grafana Tempo
3. If database issue: check connection pool metrics
4. If downstream service: check their health dashboards
5. If no clear cause: roll back recent deployment

**Rollback:**
```bash
# Rollback to previous version
kubectl rollout undo deployment/quantmind-api
```

---

### APM-002: High API Latency

**Condition:** `p95_latency > 1s for 5 minutes`

**Impact:** Slow user experience, potential timeout failures

**Diagnosis:**
```bash
# Check slow queries in database
# Look for high cardinality endpoints

# Check trace waterfall in Grafana Tempo
# Look for spans with duration > 1s
```

**Common Causes:**
1. N+1 query problem in new endpoint
2. Missing database index
3. External API slow response
4. Memory pressure causing GC pauses

**Resolution:**
1. Identify slow endpoints from dashboard
2. Check database query performance
3. Review recent code changes for inefficient queries
4. Scale horizontally if resource-bound

---

### APM-003: MT5 Disconnected

**Condition:** `mt5_status == 0 for > 30 seconds`

**Impact:** No live trading possible - CRITICAL for production

**Diagnosis:**
```bash
# Check MT5 bridge logs
{c job="quantmind-mt5-bridge" } |= "connection"

# Check MT5 terminal status manually
# Login to MT5 terminal and verify connection
```

**Common Causes:**
1. MT5 terminal crashed on Windows VPS
2. Network connectivity issue to MT5 bridge
3. MT5 bridge service restart
4. MT5 gateway blocked

**Resolution:**
1. Check MT5 terminal on trading VPS (Windows)
2. Restart MT5 bridge service if needed
3. Verify MT5 gateway credentials
4. Check firewall rules for port 5005

**Emergency Contact:**
- Trading VPS (Cloudzy): Check remote access credentials in secrets vault
- MT5 Terminal: Requires Windows RDP access

---

### APM-004: High Chaos Score

**Condition:** `chaos_score > 0.8 for 5 minutes`

**Impact:** Market conditions highly unpredictable

**Diagnosis:**
```bash
# Check regime detection in router logs
{c job="quantmind-router" } |= "regime"

# Review HMM model validation scores
```

**Common Causes:**
1. Major news event (NFP, FOMC, etc.)
2. Market open/close volatility
3. Flash crash scenario

**Resolution:**
1. Monitor but generally no action needed
2. Verify kill switch is functioning
3. Consider reducing position sizes manually

---

### APM-005: Bot Quarantine Spike

**Condition:** `quarantine_count > 5`

**Impact:** Multiple EAs being quarantined due to errors

**Diagnosis:**
```bash
# Check quarantine reason in logs
{c job="quantmind-api" } |= "quarantine"

# Look for common error patterns
```

**Resolution:**
1. Identify which EAs are quarantined
2. Check error patterns
3. Manually release if false positive
4. Fix EA code if systematic issue

---

### APM-006: Tick Stream Drops

**Condition:** `tick_drops > 10/min`

**Impact:** Missing price data may cause strategy errors

**Diagnosis:**
```bash
# Check tick drop reasons
{c job="quantmind-api" } |= "tick_drop"

# Check network latency to data source
```

**Resolution:**
1. Identify which symbols affected
2. Check data source health
3. Verify network connectivity
4. May need to switch to backup data source

---

## Grafana Dashboards

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| APM Overview | Grafana → Dashboards → QuantMindX - APM | Main APM metrics |
| Trading Operations | Grafana → Dashboards → Trading Ops | Live trading metrics |
| System Health | Grafana → Dashboards → System Health | Infrastructure health |
| Trace Explorer | Grafana → Explore → Tempo | Distributed traces |

---

## Trace Correlation

### Finding Traces from Logs

Every log entry includes `trace_id` for correlation:

```json
{
  "level": "error",
  "message": "Trade execution failed",
  "trace_id": "7e95d8f2a1b3c4d5e6f7a8b9c0d1e2f3",
  "span_id": "a1b2c3d4e5f6g7h8"
}
```

To find the trace:
1. Copy the `trace_id`
2. Go to Grafana → Explore → Tempo
3. Search for trace ID

### Adding Traces to Logs

```python
from src.monitoring import get_current_trace_id

# In your logging:
logger.error(f"Trade failed: {error}, trace_id={get_current_trace_id()}")
```

---

## Critical Metrics Reference

| Metric | Normal Range | Warning | Critical |
|--------|-------------|---------|----------|
| API Error Rate | < 0.5% | > 1% | > 5% |
| API P95 Latency | < 200ms | > 500ms | > 1s |
| MT5 Latency P95 | < 500ms | > 1s | > 2s |
| Chaos Score | 0.1 - 0.5 | > 0.6 | > 0.8 |
| Kelly Fraction | 0.2 - 0.6 | > 0.75 | > 0.9 |
| WebSocket Connections | 0 - 100 | > 150 | > 200 |
| Tick Stream Drops | 0 - 2/min | > 5/min | > 10/min |

---

## On-Call Rotation

Primary on-call: Rotate weekly (see PagerDuty schedule)

### Contact Information

- Engineering Lead: [See PagerDuty]
- Trading Team: [Slack #trading-alerts]
- DBA: [Slack #dba-support]

---

## Useful Commands

### Check API Health
```bash
curl http://localhost:8000/health
```

### Check Metrics
```bash
curl http://localhost:9090/metrics | grep quantmind
```

### Check Recent Traces in Tempo
```bash
# Using grafana-cli or curl
curl -G "http://localhost:3200/api/search" \
  --data-urlencode "q={service.name=\"quantmind-api\"}" \
  -H "Authorization: Bearer $GRAFANA_API_KEY"
```

### Force Metric Push to Grafana
```bash
# Restart the prometheus-agent to force re-scrape
docker-compose -f docker-compose.production.yml restart prometheus-agent
```

---

## Escalation Policy

1. **P1 (Critical):** Immediate response, 15-minute resolution target
   - Page primary on-call
   - Start incident channel in Slack
   - Consider customer notification

2. **P2 (Warning):** Respond within 30 minutes, 2-hour resolution target
   - Notify primary on-call
   - Monitor and prepare actions

3. **P3 (Info):** Monitor, no immediate action
   - Review in next business hours

---

## Post-Incident Review

After any P1 incident, complete PIR within 48 hours:

1. Timeline of events
2. Root cause analysis
3. Impact assessment
4. Action items with owners
5. Preventative measures

Template: [PIR Template](https://wiki.quantmindx.com/incidents/pir-template)
