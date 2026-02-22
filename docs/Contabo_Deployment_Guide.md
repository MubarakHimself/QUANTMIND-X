# Contabo VPS Deployment Guide

## Server Information

### Contabo VPS (HMM + Storage)
- **IP:** `155.133.27.86`
- **Hostname:** `vmi2963912`
- **Role:** HMM Training, HMM Inference API, Cold Storage, Monitoring

### Services Running

| Service | Port | Status |
|---------|------|--------|
| HMM Inference API | 8001 | Running |
| HMM Scheduler | 9093 | Running |
| Grafana | 3001 | Running |
| Prometheus | 9090 | Running |
| Promtail | 9080 | Running |

### SSH Access

**For Claude Code (this machine):**
```bash
# SSH key location
~/.ssh/id_ed25519

# Public key (already added to server)
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHeHSkcALJVw5OJ4ONeINwYMjcEYvoCNiJeMjCRIOzvI quantmindx-claude

# Connect
ssh root@155.133.27.86
```

**To add SSH access for a new machine:**
```bash
# On new machine - generate key
ssh-keygen -t ed25519 -C "your-name"

# On Contabo - add public key
echo "PASTE_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
```

---

## Directory Structure

```
/opt/quantmindx/          # Project directory
/data/hmm/                # HMM models and metadata
  ├── models/             # Trained HMM models
  ├── metadata/           # Model version info
  └── logs/               # Training logs
/data/cold_storage/       # Cold data archive
/var/log/quantmindx/      # Cron job logs
```

---

## Cron Jobs

Installed cron jobs (run `crontab -l` to view):

| Schedule | Job | Purpose |
|----------|-----|---------|
| Sat 02:00 UTC | HMM Training | Trains regime detection models |
| Hourly | Hot→Warm Migration | Moves tick data to DuckDB |
| Daily 03:00 UTC | Warm→Cold Archive | Archives old data |
| Every 15 min | Config Sync | Syncs config from Cloudzy |
| Sun 04:00 UTC | Log Cleanup | Removes 30+ day logs |
| Sun 05:00 UTC | Docker Cleanup | Prunes unused resources |

---

## Docker Commands

```bash
# View running services
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# View logs
docker logs quantmind-hmm-inference-api --tail 50
docker logs quantmind-hmm-scheduler --tail 50

# Restart services
cd /opt/quantmindx
docker-compose -f docker-compose.contabo.yml -f docker-compose.contabo.override.yml restart

# Rebuild and restart
docker-compose -f docker-compose.contabo.yml -f docker-compose.contabo.override.yml up -d --build
```

---

## API Endpoints

### HMM Inference API

```bash
# Get regime for all symbols
curl http://155.133.27.86:8001/api/hmm/regime

# Get regime for specific symbol
curl http://155.133.27.86:8001/api/hmm/regime/EURUSD

# With API key (for external access)
curl -H "X-API-Key: YOUR_API_KEY" http://155.133.27.86:8001/api/hmm/regime
```

### Grafana

- **URL:** http://155.133.27.86:3001
- **Default credentials:** Check `.env` file

### Prometheus

- **URL:** http://155.133.27.86:9090

---

## Common Tasks

### Update Code

```bash
cd /opt/quantmindx
git pull
docker-compose -f docker-compose.contabo.yml -f docker-compose.contabo.override.yml up -d --build
```

### Train HMM Model Manually

```bash
cd /opt/quantmindx
docker exec quantmind-hmm-scheduler python scripts/schedule_hmm_training.py --run-now
```

### View Cron Job Logs

```bash
tail -f /var/log/quantmindx/hmm_training.log
tail -f /var/log/quantmindx/migration_hot_warm.log
tail -f /var/log/quantmindx/archive_warm_cold.log
```

### Check Disk Usage

```bash
df -h /data
df -h /
```

---

## Troubleshooting

### HMM API Not Responding

```bash
# Check if container is running
docker ps | grep hmm

# Check logs
docker logs quantmind-hmm-inference-api --tail 50

# Check if port is open
curl http://localhost:8001/api/hmm/regime
```

### HMM Scheduler Crashing

```bash
# Check if APScheduler is installed
docker exec quantmind-hmm-scheduler pip list | grep apscheduler

# Rebuild container
docker-compose -f docker-compose.contabo.yml -f docker-compose.contabo.override.yml up -d --build hmm-scheduler
```

### Cron Jobs Not Running

```bash
# Check cron service
systemctl status cron

# View cron logs
grep CRON /var/log/syslog | tail -20

# Verify crontab
crontab -l
```

---

## Environment Variables

Key environment variables in `.env`:

```bash
QUANTMIND_ENV=production
WARM_DB_PATH=/data/market_data.duckdb
COLD_STORAGE_PATH=/data/cold_storage
CONTABO_HMM_API_KEY=<api-key>
```

---

## Cloudzy VPS (Trading Server)

**Note:** Cloudzy VPS setup is separate. See `docs/Coding_Agent_Handoff_Plan.md` for details.

- Trading API: Port 8000
- MT5 Bridge: Port 5005
- GitHub EA Sync: Configured via environment variables
