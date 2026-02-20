#!/bin/bash
# =============================================================================
# Contabo Cron Jobs Setup Script
# =============================================================================
# Installs cron jobs for HMM training, data migration, and maintenance.
# Run this script on the Contabo VPS.
#
# Usage:
#   chmod +x scripts/setup_contabo_crons.sh
#   ./scripts/setup_contabo_crons.sh
# =============================================================================

set -e

# Configuration
PROJECT_DIR="/opt/quantmindx"
LOG_DIR="/var/log/quantmindx"
VENV_DIR="${PROJECT_DIR}/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== QuantMindX Contabo Cron Jobs Setup ===${NC}"
echo ""

# Create log directory
echo -e "${YELLOW}Creating log directory...${NC}"
sudo mkdir -p ${LOG_DIR}
sudo chown $USER:$USER ${LOG_DIR}

# Create cron job file
echo -e "${YELLOW}Creating cron jobs...${NC}"

CRON_FILE="/tmp/quantmindx_crons"

cat > ${CRON_FILE} << 'EOF'
# QuantMindX Cron Jobs - Contabo VPS
# Installed: $(date)

# Environment
SHELL=/bin/bash
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
PROJECT_DIR=/opt/quantmindx
VENV_DIR=/opt/quantmindx/venv
LOG_DIR=/var/log/quantmindx

# =============================================================================
# HMM Training - Weekly (Saturday 02:00 UTC)
# Trains new HMM models for regime detection
# =============================================================================
0 2 * * 6 cd ${PROJECT_DIR} && ${VENV_DIR}/bin/python scripts/schedule_hmm_training.py --run-now >> ${LOG_DIR}/hmm_training.log 2>&1

# =============================================================================
# Hot to Warm Migration - Hourly
# Migrates tick data from PostgreSQL (HOT) to DuckDB (WARM)
# =============================================================================
0 * * * * cd ${PROJECT_DIR} && ${VENV_DIR}/bin/python scripts/migrate_hot_to_warm.py >> ${LOG_DIR}/migration_hot_warm.log 2>&1

# =============================================================================
# Warm to Cold Archiving - Daily at 03:00 UTC
# Archives data older than 30 days to cold storage
# =============================================================================
0 3 * * * cd ${PROJECT_DIR} && ${VENV_DIR}/bin/python scripts/archive_warm_to_cold.py >> ${LOG_DIR}/archive_warm_cold.log 2>&1

# =============================================================================
# Config Sync - Every 15 minutes
# Syncs configuration from Cloudzy to Contabo
# =============================================================================
*/15 * * * * cd ${PROJECT_DIR} && bash scripts/sync_config.sh >> ${LOG_DIR}/config_sync.log 2>&1

# =============================================================================
# Cleanup Old Logs - Weekly (Sunday 04:00 UTC)
# Removes logs older than 30 days
# =============================================================================
0 4 * * 0 find ${LOG_DIR} -name "*.log" -mtime +30 -delete

# =============================================================================
# Docker Cleanup - Weekly (Sunday 05:00 UTC)
# Cleans up unused Docker resources
# =============================================================================
0 5 * * 0 docker system prune -f >> ${LOG_DIR}/docker_cleanup.log 2>&1

EOF

# Display the cron jobs
echo -e "${GREEN}Cron jobs to be installed:${NC}"
cat ${CRON_FILE}
echo ""

# Ask for confirmation
read -p "Install these cron jobs? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Backup existing crontab
    echo -e "${YELLOW}Backing up existing crontab...${NC}"
    crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || true

    # Install new crontab (appending to existing)
    echo -e "${YELLOW}Installing cron jobs...${NC}"
    (crontab -l 2>/dev/null | grep -v "QuantMindX"; cat ${CRON_FILE}) | crontab -

    echo -e "${GREEN}Cron jobs installed successfully!${NC}"
    echo ""
    echo -e "${YELLOW}Installed cron jobs:${NC}"
    crontab -l | grep -v "^#" | grep -v "^$"
else
    echo -e "${RED}Installation cancelled.${NC}"
fi

# Cleanup
rm -f ${CRON_FILE}

echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo "Log files will be written to: ${LOG_DIR}"
echo ""
echo "To view logs:"
echo "  tail -f ${LOG_DIR}/hmm_training.log"
echo "  tail -f ${LOG_DIR}/migration_hot_warm.log"
echo "  tail -f ${LOG_DIR}/archive_warm_cold.log"
echo "  tail -f ${LOG_DIR}/config_sync.log"
echo ""
echo "To edit cron jobs:"
echo "  crontab -e"
echo ""
echo "To remove all QuantMindX cron jobs:"
echo "  crontab -l | grep -v 'QuantMindX' | crontab -"
