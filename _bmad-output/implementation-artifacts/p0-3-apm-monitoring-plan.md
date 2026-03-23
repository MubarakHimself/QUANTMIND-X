# P0-3: APM Monitoring — Implementation Plan

## Phase 1: APM Vendor Selection

### Recommendation: Grafana Cloud with Grafana Tempo (OpenTelemetry)

**Justification:**
1. **Stack Fit:** Existing Grafana Cloud deployment with Prometheus + Loki — Tempo integrates seamlessly
2. **Cost:** Grafana Tempo is significantly cheaper than Datadog/New Relic for tracing ($0.50/million spans vs $0.10/million traces on Datadog)
3. **OpenTelemetry Native:** Vendor-neutral instrumentation that can be switched later if needed
4. **Unified View:** Single pane of glass — metrics, logs, and traces in one Grafana dashboard
5. **No Agent Overhead:** Uses existing prometheus-agent infrastructure

**Comparison:**
| Vendor | Ease of Use | Cost (100K spans/mo) | Existing Stack Fit |
|--------|-------------|---------------------|-------------------|
| Datadog | High | ~$30 + overage | Poor (new agent) |
| New Relic | High | ~$25 + overage | Poor (new agent) |
| Elastic APM | Medium | ~$35 + compute | Poor (ES cluster) |
| Grafana Tempo | Medium | ~$5 (storage only) | **Excellent** (existing) |

## Phase 2: Agent Deployment

### 2.1 Python Instrumentation (FastAPI)

**Install dependencies:**
```
opentelemetry-api>=1.21.0
opentelemetry-sdk>=1.21.0
opentelemetry-instrumentation-fastapi>=0.42b0
opentelemetry-exporter-otlp>=1.21.0
opentelemetry-ext-auto-attrs>=0.1b0
```

**Key instrumentation points:**
- FastAPI auto-instrumentation for HTTP request tracing
- PostgreSQL/SQLAlchemy instrumentation for database queries
- Redis instrumentation for cache operations
- Custom spans for strategy routing, trade execution, regime detection

### 2.2 Docker Integration

Add OpenTelemetry collector sidecar to `docker-compose.production.yml`:
```yaml
otel-collector:
  image: otel/opentelemetry-collector-contrib:latest
  ports:
    - "4317:4317"   # OTLP gRPC
    - "4318:4318"   # OTLP HTTP
    - "8888:8888"   # Prometheus metrics exposed by collector
  volumes:
    - ./docker/otel/collector-config.yml:/etc/otelcol-contrib/config.yaml
```

### 2.3 FastAPI Instrumentation

Add to `src/api/server.py` startup:
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Initialize tracer provider with OTLP exporter
trace.set_tracer_provider(TracerProvider())
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")))
)
```

## Phase 3: Distributed Tracing

### 3.1 Trace Context Propagation

**Cross-service tracing headers:**
- `traceparent` (W3C Trace Context)
- `tracestate` (W3C Trace Context)
- Propagators: W3C TraceContextPropagator + B3 for legacy

**WebSocket/SSE handling:**
- Inject trace context into WebSocket messages
- Extract trace context from incoming WebSocket connections
- Maintain trace correlation for SSE event streams

### 3.2 Custom Span Attributes

**Required attributes for trading context:**
- `trading.symbol` — symbol being traded
- `trading.action` — BUY/SELL
- `trading.mode` — live/demo/paper
- `trading.bot_id` — EA identifier
- `trading.strategy` — strategy name
- `trading.regime` — current market regime

### 3.3 Trace Correlation IDs

Add to existing logging middleware for log-trace correlation:
```python
from opentelemetry.trace import get_current_span

@app.middleware("http")
async def trace_context_middleware(request: Request, call_next):
    span = get_current_span()
    trace_id = span.get_span_context().trace_id
    request.state.trace_id = format(trace_id, '032x')
    # Add trace_id to log JSON for correlation
```

## Phase 4: Dashboard & Alerts

### 4.1 Key Metrics Dashboard

**Panels:**
1. **Request Latency (P50, P95, P99)**
   - Query: `histogram_quantile(0.95, rate(quantmind_api_request_duration_seconds_bucket[5m]))`
   - Alert threshold: P95 > 500ms

2. **Error Rate**
   - Query: `rate(quantmind_api_errors_total[5m]) / rate(quantmind_api_requests_total[5m]) * 100`
   - Alert threshold: > 1%

3. **Throughput (RPM)**
   - Query: `rate(quantmind_api_requests_total[1m]) * 60`
   - Baseline: establish from historical data

4. **Active EAs**
   - Query: `sum(quantmind_active_eas)`
   - Alert threshold: sudden drop > 20%

5. **MT5 Connection Status**
   - Query: `quantmind_mt5_connection_status`
   - Alert: = 0 for > 30s

6. **Trade Execution Latency**
   - Query: `histogram_quantile(0.95, rate(quantmind_mt5_latency_seconds_bucket{operation="trade"}[5m]))`
   - Alert threshold: P95 > 2s

### 4.2 Bot/Capacity Metrics Dashboard

**Panels:**
- Paper trading P&L by bot tag
- Strategy distribution (pie chart)
- Regime distribution
- Chaos score over time
- Tick stream rate by symbol

### 4.3 Alert Thresholds

| Alert | Condition | Severity | Runbook |
|-------|-----------|----------|---------|
| HighAPIErrorRate | error_rate > 5% for 5m | Critical | APM-001 |
| HighAPILatency | p95_latency > 1s for 5m | Warning | APM-002 |
| MT5Disconnected | mt5_status == 0 for > 30s | Critical | APM-003 |
| HighChaosScore | chaos_score > 0.8 for 5m | Warning | APM-004 |
| BotQuarantineSpike | quarantine_count > 5 | Warning | APM-005 |
| TickStreamDrops | tick_drops > 10/min | Warning | APM-006 |

## Phase 5: Documentation

### Runbook Location: `_bmad-output/test-artifacts/apm-runbook.md`

**Contents:**
- Alert escalation matrix (who to wake up)
- Common issue troubleshooting steps
- Trace ID correlation with logs
- Grafana dashboard URLs
- On-call rotation
- Critical metrics reference

## Implementation Files

### New Files:
1. `docker/otel/collector-config.yml` — OpenTelemetry collector config
2. `src/monitoring/tracing.py` — Tracing initialization and utilities
3. `src/monitoring/instrumentation.py` — Custom instrumentation
4. `docker/grafana/dashboards/apm-dashboard.json` — Grafana dashboard
5. `_bmad-output/test-artifacts/apm-runbook.md` — On-call runbook

### Modified Files:
1. `docker-compose.production.yml` — Add otel-collector service
2. `requirements.txt` — Add OpenTelemetry packages
3. `src/api/server.py` — Add tracing initialization

## Verification Checklist

- [ ] OpenTelemetry collector starts successfully
- [ ] Traces appear in Grafana Tempo within 30s of API calls
- [ ] Trace IDs correlate with log entries in Loki
- [ ] P95 latency dashboard panel shows data
- [ ] Alert rules fire correctly in test scenario
- [ ] Runbook accessible to on-call team
