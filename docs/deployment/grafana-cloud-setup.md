# Grafana Cloud Monitoring Setup Guide

This guide walks you through setting up Grafana Cloud monitoring for the QuantMindX trading system.

## Overview

QuantMindX integrates with Grafana Cloud's free tier for centralized observability:

- **Prometheus**: Metrics collection and alerting
- **Loki**: Log aggregation and querying
- **Grafana**: Dashboards and visualization

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  API Server     │     │  Strategy Router │     │   MT5 Bridge    │
│  :8000          │     │                  │     │   :5005         │
│  :9090/metrics  │     │                  │     │   :9091/metrics │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         │   Promtail Agent      │                        │
         │   :9080               │                        │
         └───────────────────────┼────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────────────┐
                    │        Grafana Cloud           │
                    │   ┌────────┐   ┌──────────┐   │
                    │   │Prometh.│   │  Loki    │   │
                    │   └────┬───┘   └────┬─────┘   │
                    │        │            │         │
                    │        └─────┬──────┘         │
                    │              │                │
                    │        ┌─────▼─────┐          │
                    │        │  Grafana  │          │
                    │        │ Dashboards│          │
                    │        └───────────┘          │
                    └────────────────────────────────┘
```

## Step 1: Create Grafana Cloud Account

1. Go to https://grafana.com/auth/sign-up/create-user
2. Sign up for a free account
3. Verify your email address

## Step 2: Create a Grafana Cloud Stack

1. After logging in, you'll be prompted to create a stack
2. Choose a stack name (e.g., `quantmindx-prod`)
3. Select a region closest to your VPS (e.g., `US East` for Cloudzy VPS)
4. Click **Create Stack**

## Step 3: Get Your Endpoints

Navigate to your stack and note down the following URLs:

1. **Prometheus URL**: 
   - Format: `https://prometheus-prod-XX-prod-XX-XX.grafana.net`
   - Found in: Configuration → Details → Prometheus

2. **Loki URL**: 
   - Format: `https://logs-prod-XX.grafana.net`
   - Found in: Configuration → Details → Loki

3. **Instance ID**:
   - This is your stack ID or username (e.g., `123456`)
   - Found in: Configuration → Details

## Step 4: Generate API Key

1. Go to **Configuration → API Keys**
2. Click **Create API Key**
3. Set the following:
   - **Name**: `quantmindx-metrics-pusher`
   - **Role**: `MetricsPublisher` (required for remote write)
   - **Expiration**: Choose a long period (1 year recommended)
4. Click **Create**
5. **Copy the API key immediately** - it won't be shown again

## Step 5: Configure Environment Variables

Copy the example file and update with your credentials:

```bash
cp .env.production .env
```

Edit `.env` with your Grafana Cloud details:

```bash
# Grafana Cloud Configuration
GRAFANA_PROMETHEUS_URL=https://prometheus-prod-10-prod-us-east-0.grafana.net
GRAFANA_LOKI_URL=https://logs-prod-006.grafana.net
GRAFANA_INSTANCE_ID=123456
GRAFANA_API_KEY=glc_eyJ...your-api-key-here

# Metrics Server Ports
PROMETHEUS_PORT=9090
MT5_PROMETHEUS_PORT=9091
```

## Step 6: Create Logs Directory

```bash
mkdir -p logs
```

## Step 7: Deploy with Docker Compose

```bash
docker-compose -f docker-compose.production.yml up -d
```

## Step 8: Verify Metrics Collection

### Check Metrics Endpoint

```bash
# API metrics
curl http://localhost:9090/metrics

# MT5 Bridge metrics
curl http://localhost:9091/metrics
```

### Check Promtail Status

```bash
docker logs quantmind-promtail
```

You should see Promtail successfully shipping logs to Grafana Cloud.

## Step 9: Create Dashboards in Grafana Cloud

### Dashboard 1: Trading Overview

1. In Grafana Cloud, go to **Dashboards → New Dashboard**
2. Add the following panels:

#### Active EAs by Status
```promql
quantmind_active_eas
```
- **Visualization**: Stat
- **Group by**: `status` label

#### Trades Executed (5m rate)
```promql
sum(rate(quantmind_trades_executed_total[5m])) by (mode)
```
- **Visualization**: Time series

#### Current Chaos Score
```promql
quantmind_chaos_score
```
- **Visualization**: Gauge
- **Thresholds**: 0.5 (warning), 0.75 (critical)

#### Regime Changes (1h)
```promql
increase(quantmind_regime_changes_total[1h])
```
- **Visualization**: Stat

#### P&L Distribution
```promql
sum(rate(quantmind_trade_profit_loss_sum[5m])) by (mode)
```
- **Visualization**: Time series

### Dashboard 2: System Health

#### API Request Rate
```promql
sum(rate(quantmind_api_requests_total[5m])) by (endpoint)
```
- **Visualization**: Time series

#### API Latency (p95)
```promql
histogram_quantile(0.95, rate(quantmind_api_request_duration_seconds_bucket[5m]))
```
- **Visualization**: Time series

#### API Error Rate
```promql
sum(rate(quantmind_api_requests_total{status=~"5.."}[5m])) / sum(rate(quantmind_api_requests_total[5m]))
```
- **Visualization**: Stat
- **Format**: Percent (0-1)

#### MT5 Connection Status
```promql
quantmind_mt5_connection_status
```
- **Visualization**: Stat
- **Value mappings**: 1="Connected", 0="Disconnected"

#### MT5 Latency (p95)
```promql
histogram_quantile(0.95, rate(quantmind_mt5_latency_seconds_bucket[5m])) by (operation)
```
- **Visualization**: Time series

### Dashboard 3: Logs Dashboard

1. Add a **Logs** panel
2. Configure Loki query:
```logql
{job="quantmind-api"} |= `` | json
```

#### Recent Errors
```logql
{job=~"quantmind.*"} | json | level =~ "error|critical"
```

#### Trade Logs
```logql
{job="quantmind-router"} | json | trade_id != ``
```

## Step 10: Configure Alerts

### Option A: Import Alert Rules

1. Go to **Alerting → Alert rules**
2. Click **New alert rule → Import**
3. Upload the `monitoring/alert-rules.yml` file

### Option B: Create Alerts Manually

Create the following alerts in Grafana Cloud:

#### MT5 Connection Down
- **Query**: `quantmind_mt5_connection_status == 0`
- **Condition**: For 2 minutes
- **Severity**: Critical
- **Message**: "MT5 Bridge has lost connection"

#### High Chaos Score
- **Query**: `quantmind_chaos_score > 0.8`
- **Condition**: For 10 minutes
- **Severity**: Critical
- **Message**: "Market chaos score is above 0.8"

#### API Error Rate High
- **Query**: `sum(rate(quantmind_api_requests_total{status=~"5.."}[5m])) / sum(rate(quantmind_api_requests_total[5m])) > 0.1`
- **Condition**: For 5 minutes
- **Severity**: Critical
- **Message**: "API 5xx error rate above 10%"

## Step 11: Configure Notification Channels

### Email Notifications

1. Go to **Alerting → Contact points**
2. Click **New contact point**
3. Select **Email**
4. Enter your email address
5. Click **Save**

### Slack Webhook (Optional)

1. Create a Slack incoming webhook URL
2. Go to **Alerting → Contact points**
3. Click **New contact point**
4. Select **Slack**
5. Enter the webhook URL
6. Click **Save**

### Notification Policies

1. Go to **Alerting → Notification policies**
2. Configure routing based on severity:
   - `severity=critical` → Email + Slack
   - `severity=warning` → Slack only
   - `severity=info` → Log only

## Step 12: Verify Everything Works

### Test Metrics Flow

```bash
# Make some API requests
curl http://localhost:8000/api/router/status

# Check metrics were recorded
curl http://localhost:9090/metrics | grep quantmind_api_requests_total
```

### Test Log Shipping

```bash
# Check Promtail is running
docker ps | grep promtail

# Check Promtail logs
docker logs quantmind-promtail --tail 50
```

### Test Alerts

1. Stop MT5 Bridge temporarily:
```bash
docker stop quantmind-mt5-bridge
```

2. Wait 2-3 minutes
3. Check if MT5ConnectionDown alert fires in Grafana Cloud
4. Restart the bridge:
```bash
docker start quantmind-mt5-bridge
```

## Free Tier Limits

Grafana Cloud free tier includes:

| Resource | Limit |
|----------|-------|
| Active Metrics Series | 10,000 |
| Logs per day | 1 GB |
| Traces | 1 GB |
| Data Retention | 14 days |
| Users | 3 (unlimited view-only) |

QuantMindX metrics should stay well under the 10K series limit:
- API metrics: ~50 series
- Trading metrics: ~30 series
- MT5 metrics: ~20 series
- System metrics: ~20 series

**Total: ~120 active series**

## Troubleshooting

### Metrics Not Appearing

1. Check API server logs:
```bash
docker logs quantmind-api | grep prometheus
```

2. Verify metrics endpoint is accessible:
```bash
curl http://localhost:9090/metrics
```

3. Check Grafana Cloud API key permissions

### Logs Not Appearing in Loki

1. Check Promtail is running:
```bash
docker ps | grep promtail
```

2. Check Promtail logs for errors:
```bash
docker logs quantmind-promtail 2>&1 | grep -i error
```

3. Verify logs directory is mounted correctly:
```bash
ls -la logs/
```

### High Memory Usage

If Promtail uses too much memory:

1. Reduce scrape frequency in `promtail-config.yml`
2. Limit the number of log files scraped
3. Adjust the positions file location

## Additional Resources

- [Grafana Cloud Documentation](https://grafana.com/docs/grafana-cloud/)
- [Prometheus Query Documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Loki Query Documentation](https://grafana.com/docs/loki/latest/query/)
- [QuantMindX Monitoring Module](../src/monitoring/README.md)