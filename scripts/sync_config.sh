#!/bin/bash
# =============================================================================
# QuantMindX Config Sync Script
# =============================================================================
# 
# Pulls configuration changes from GitHub and restarts affected services.
# Runs every 15 minutes via cron.
#
# Usage:
#   ./scripts/sync_config.sh
#
# Environment:
#   DOCKER_COMPOSE_FILE: Override docker-compose file (default: docker-compose.production.yml)
#   IS_CONTABO: Set to "true" if running on Contabo VPS
#
# =============================================================================

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${LOG_DIR:-/var/log/quantmindx}"
LOG_FILE="$LOG_DIR/sync_config.log"

# Determine which docker-compose file to use
if [ "$IS_CONTABO" = "true" ]; then
    COMPOSE_FILE="docker-compose.contabo.yml"
    API_SERVICE="hmm-inference-api"
else
    COMPOSE_FILE="${DOCKER_COMPOSE_FILE:-docker-compose.production.yml}"
    API_SERVICE="quantmind-api"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "$1"
}

log_warn() {
    log "${YELLOW}WARN${NC}" "$1"
}

log_error() {
    log "${RED}ERROR${NC}" "$1"
}

log_success() {
    log "${GREEN}SUCCESS${NC}" "$1"
}

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Change to project directory
cd "$PROJECT_DIR"

log_info "Starting config sync..."

# Git pull
log_info "Pulling latest changes from GitHub..."
if git pull origin main >> "$LOG_FILE" 2>&1; then
    log_success "Git pull completed"
else
    log_error "Git pull failed - may be up to date or no network"
fi

# Check for .env changes and compare with .env.example
if [ -f ".env" ] && [ -f ".env.example" ]; then
    log_info "Checking for missing .env keys..."
    
    # Get keys from .env.example
    EXAMPLE_KEYS=$(grep -E "^[A-Z_]+=" .env.example | sed 's/=.*//g' | sort)
    
    # Get keys from .env
    ENV_KEYS=$(grep -E "^[A-Z_]+=" .env | sed 's/=.*//g' | sort)
    
    # Find missing keys
    MISSING_KEYS=$(comm -23 <(echo "$EXAMPLE_KEYS") <(echo "$ENV_KEYS"))
    
    if [ -n "$MISSING_KEYS" ]; then
        log_warn "Missing .env keys:"
        echo "$MISSING_KEYS" | while read -r key; do
            log_warn "  - $key"
        done
    else
        log_success "All required .env keys are present"
    fi
fi

# Check if config files changed
log_info "Checking for config file changes..."
CONFIG_CHANGED=false

# Get the commit hash before pull
PREV_COMMIT=$(git rev-parse HEAD@{1} 2>/dev/null || echo "HEAD")
CURRENT_COMMIT=$(git rev-parse HEAD)

if [ "$PREV_COMMIT" != "$CURRENT_COMMIT" ]; then
    # Check if config directory changed
    if git diff "$PREV_COMMIT" "$CURRENT_COMMIT" -- config/ >> "$LOG_FILE" 2>&1; then
        if git diff "$PREV_COMMIT" "$CURRENT_COMMIT" -- config/ --quiet; then
            log_info "Config files unchanged"
        else
            log_info "Config files changed - will restart services"
            CONFIG_CHANGED=true
        fi
    fi
else
    log_info "No new commits to process"
fi

# Restart affected services if config changed
if [ "$CONFIG_CHANGED" = true ]; then
    log_info "Restarting services with new configuration..."
    
    # Check if docker-compose file exists
    if [ -f "$COMPOSE_FILE" ]; then
        # Stop and restart the API service
        if docker-compose -f "$COMPOSE_FILE" restart "$API_SERVICE" >> "$LOG_FILE" 2>&1; then
            log_success "Restarted $API_SERVICE"
        else
            log_error "Failed to restart $API_SERVICE"
        fi
    else
        log_warn "Docker compose file not found: $COMPOSE_FILE"
    fi
else
    log_info "No config changes - skipping service restart"
fi

log_success "Config sync completed"
echo "" >> "$LOG_FILE"
