#!/bin/bash
# QuantMindX CLI - Central Deployment and Management Tool
# Provides update, rollback, status, restart, logs, version, history, and health commands

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAPSHOT_DIR="/opt/quantmindx/snapshots"
DEPLOY_LOG="$PROJECT_ROOT/data/logs/deployments.log"
MAX_SNAPSHOTS=5

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Service health check ports (from docker-compose.production.yml)
declare -A SERVICE_PORTS=(
    ["quantmind-api"]="8000/health"
    ["mt5-bridge"]="5005/health"
    ["prometheus-agent"]="9099/-/healthy"
    ["promtail"]="9080/ready"
    ["pageindex-articles"]="3000/health"
    ["pageindex-books"]="3001/health"
    ["pageindex-logs"]="3002/health"
)

# Staged restart order (as per spec)
RESTART_ORDER=("pageindex-articles" "pageindex-books" "pageindex-logs" "prometheus-agent" "promtail" "hmm-scheduler" "mt5-bridge" "quantmind-api")

# ============== Helper Functions ==============

run_health_check() {
    local service="$1"
    local port_path="${SERVICE_PORTS[$service]}"
    
    if [[ -z "$port_path" ]]; then
        echo -e "${YELLOW}⚠️  No health check defined for $service${NC}"
        return 0
    fi
    
    local port="${port_path%%/*}"
    local path="${port_path#*/}"
    
    if curl -sf "http://localhost:$port/$path" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $service healthy${NC}"
        return 0
    else
        echo -e "${RED}❌ $service unhealthy (http://localhost:$port/$path)${NC}"
        return 1
    fi
}

run_all_health_checks() {
    local failed=0
    echo -e "${BLUE}Running health checks for all services...${NC}"
    for service in "${RESTART_ORDER[@]}"; do
        if ! run_health_check "$service"; then
            ((failed++))
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        echo -e "${GREEN}All services healthy${NC}"
        return 0
    else
        echo -e "${RED}$failed service(s) unhealthy${NC}"
        return 1
    fi
}

get_current_version() {
    cd "$PROJECT_ROOT"
    python3 -c "from src.version import get_version; print(get_version())" 2>/dev/null || echo "unknown"
}

create_snapshot() {
    local version="$1"
    local snapshot_path="$SNAPSHOT_DIR/$version"
    
    mkdir -p "$SNAPSHOT_DIR"
    
    # Check if snapshot already exists
    if [[ -d "$snapshot_path" ]]; then
        echo -e "${YELLOW}Snapshot $version already exists, skipping...${NC}"
        return 0
    fi
    
    mkdir -p "$snapshot_path"
    
    # Write manifest
    cat > "$snapshot_path/manifest.json" << EOF
{
    "version": "$version",
    "timestamp": "$(date -Iseconds)",
    "created_by": "$(whoami)"
}
EOF
    
    # Create tarball of source (excluding venv, node_modules, data, logs)
    tar --exclude='venv' \
        --exclude='node_modules' \
        --exclude='data' \
        --exclude='logs' \
        --exclude='.git' \
        --exclude='__pycache__' \
        -czf "$snapshot_path/code.tar.gz" \
        -C "$PROJECT_ROOT" .
    
    echo -e "${GREEN}✅ Snapshot created: $snapshot_path${NC}"
    
    # Prune old snapshots
    prune_snapshots
}

prune_snapshots() {
    local count=$(ls -1 "$SNAPSHOT_DIR" 2>/dev/null | wc -l)
    
    if [[ $count -gt $MAX_SNAPSHOTS ]]; then
        echo -e "${YELLOW}Pruning old snapshots (keeping last $MAX_SNAPSHOTS)...${NC}"
        cd "$SNAPSHOT_DIR"
        ls -1t | tail -n +$((MAX_SNAPSHOTS + 1)) | xargs rm -rf
    fi
}

find_latest_snapshot() {
    if [[ ! -d "$SNAPSHOT_DIR" ]]; then
        return 1
    fi
    
    ls -1t "$SNAPSHOT_DIR" | head -1
}

log_deployment() {
    local version_from="$1"
    local version_to="$2"
    local status="$3"
    local duration="$4"
    local services="$5"
    
    mkdir -p "$(dirname "$DEPLOY_LOG")"
    
    # Append JSON line
    cat >> "$DEPLOY_LOG" << EOF
{"timestamp":"$(date -Iseconds)","version_from":"$version_from","version_to":"$version_to","vps":"$(hostname)","status":"$status","duration_seconds":$duration,"services":"$services"}
EOF
}

staged_restart() {
    local single_service="$1"
    local failed=0
    
    if [[ -n "$single_service" ]]; then
        # Restart single service
        echo -e "${BLUE}Restarting $single_service...${NC}"
        docker compose -f "$PROJECT_ROOT/docker-compose.production.yml" restart "$single_service"
        sleep 5
        
        if run_health_check "$single_service"; then
            echo -e "${GREEN}✅ $single_service restarted successfully${NC}"
            return 0
        else
            echo -e "${RED}❌ $single_service failed health check after restart${NC}"
            return 1
        fi
    fi
    
    # Staged restart in order
    echo -e "${BLUE}Starting staged restart...${NC}"
    for service in "${RESTART_ORDER[@]}"; do
        echo -e "${BLUE}Restarting $service...${NC}"
        
        if docker compose -f "$PROJECT_ROOT/docker-compose.production.yml" restart "$service" 2>/dev/null; then
            # Wait for health check
            local attempts=0
            local max_attempts=12
            while [[ $attempts -lt $max_attempts ]]; do
                sleep 5
                if run_health_check "$service"; then
                    break
                fi
                ((attempts++))
            done
            
            if [[ $attempts -eq $max_attempts ]]; then
                echo -e "${RED}❌ $service failed to become healthy after $((max_attempts * 5))s${NC}"
                ((failed++))
            fi
        else
            # Service might not exist in compose file
            echo -e "${YELLOW}⚠️  Could not restart $service (may not be running)${NC}"
        fi
    done
    
    if [[ $failed -eq 0 ]]; then
        echo -e "${GREEN}✅ All services restarted successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ $failed service(s) failed during restart${NC}"
        return 1
    fi
}

# ============== Commands ==============

cmd_update() {
    local target_version=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                target_version="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    local start_time=$(date +%s)
    local current_version=$(get_current_version)
    
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}       QuantMindX Update${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    
    # Validate environment
    echo -e "${BLUE}Step 1: Validating environment...${NC}"
    if ! "$PROJECT_ROOT/scripts/validate_env.sh"; then
        return 1
    fi
    
    # Pre-health check
    echo -e "${BLUE}Step 2: Pre-update health check...${NC}"
    if ! run_all_health_checks; then
        echo -e "${YELLOW}⚠️  Some services unhealthy before update, proceeding anyway...${NC}"
    fi
    
    # Create snapshot
    echo -e "${BLUE}Step 3: Creating snapshot of current version...${NC}"
    create_snapshot "$current_version"
    
    # Git operations
    echo -e "${BLUE}Step 4: Fetching updates...${NC}"
    cd "$PROJECT_ROOT"
    
    if ! git fetch --all; then
        echo -e "${RED}❌ Failed to fetch from remote${NC}"
        log_deployment "$current_version" "$target_version" "failed" 0 "git fetch"
        return 1
    fi
    
    if [[ -n "$target_version" ]]; then
        echo -e "${BLUE}Checking out $target_version...${NC}"
        if ! git checkout "$target_version"; then
            echo -e "${RED}❌ Failed to checkout $target_version${NC}"
            log_deployment "$current_version" "$target_version" "failed" 0 "git checkout"
            return 1
        fi
    else
        echo -e "${BLUE}Pulling latest...${NC}"
        if ! git pull; then
            echo -e "${RED}❌ Failed to pull latest${NC}"
            log_deployment "$current_version" "latest" "failed" 0 "git pull"
            return 1
        fi
    fi
    
    # Install dependencies
    echo -e "${BLUE}Step 5: Installing dependencies...${NC}"
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt --quiet
    fi
    
    local new_version=$(get_current_version)
    
    # Staged restart
    echo -e "${BLUE}Step 6: Staged restart of services...${NC}"
    if ! staged_restart; then
        echo -e "${RED}❌ Staged restart failed, attempting rollback...${NC}"
        # Rollback
        local latest_snapshot=$(find_latest_snapshot)
        if [[ -n "$latest_snapshot" ]]; then
            tar -xzf "$SNAPSHOT_DIR/$latest_snapshot/code.tar.gz" -C "$PROJECT_ROOT"
            staged_restart
        fi
        log_deployment "$current_version" "$new_version" "rolled_back" $(($(date +%s) - start_time)) "all"
        return 1
    fi
    
    # Post-health check
    echo -e "${BLUE}Step 7: Post-update health check...${NC}"
    if ! run_all_health_checks; then
        echo -e "${RED}❌ Post-update health check failed${NC}"
        log_deployment "$current_version" "$new_version" "failed" $(($(date +%s) - start_time)) "all"
        return 1
    fi
    
    local duration=$(($(date +%s) - start_time))
    log_deployment "$current_version" "$new_version" "success" "$duration" "all"
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Update complete!${NC}"
    echo -e "${GREEN}  Version: $current_version → $new_version${NC}"
    echo -e "${GREEN}  Duration: ${duration}s${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}"
}

cmd_rollback() {
    local target_version=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                target_version="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    local start_time=$(date +%s)
    local current_version=$(get_current_version)
    
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}       QuantMindX Rollback${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    
    # Find snapshot
    if [[ -z "$target_version" ]]; then
        target_version=$(find_latest_snapshot)
        if [[ -z "$target_version" ]]; then
            echo -e "${RED}❌ No snapshots found in $SNAPSHOT_DIR${NC}"
            return 1
        fi
    fi
    
    local snapshot_path="$SNAPSHOT_DIR/$target_version"
    if [[ ! -d "$snapshot_path" ]]; then
        echo -e "${RED}❌ Snapshot not found: $snapshot_path${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Rolling back to version: $target_version${NC}"
    
    # Extract snapshot
    echo -e "${BLUE}Extracting snapshot...${NC}"
    tar -xzf "$snapshot_path/code.tar.gz" -C "$PROJECT_ROOT"
    
    # Staged restart
    echo -e "${BLUE}Restarting services...${NC}"
    if ! staged_restart; then
        echo -e "${RED}❌ Failed to restart services after rollback${NC}"
        log_deployment "$current_version" "$target_version" "failed" $(($(date +%s) - start_time)) "all"
        return 1
    fi
    
    local duration=$(($(date +%s) - start_time))
    log_deployment "$current_version" "$target_version" "success" "$duration" "all"
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Rollback complete!${NC}"
    echo -e "${GREEN}  Version: $current_version → $target_version${NC}"
    echo -e "${GREEN}  Duration: ${duration}s${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}"
}

cmd_status() {
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}       QuantMindX Status${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    
    echo -e "Version: $(get_current_version)"
    echo -e "Environment: ${QUANTMIND_ENV:-development}"
    echo -e "Project Root: $PROJECT_ROOT"
    echo ""
    
    run_all_health_checks
}

cmd_restart() {
    local single_service=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --service)
                single_service="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    echo -e "${BLUE}       QuantMindX Restart${NC}"
    echo -e "${BLUE}════════════════════════════════════════${NC}"
    
    # Validate environment
    if ! "$PROJECT_ROOT/scripts/validate_env.sh"; then
        return 1
    fi
    
    staged_restart "$single_service"
}

cmd_logs() {
    local service=""
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --service)
                service="$2"
                shift 2
                ;;
            *)
                shift
                ;;
        esac
    done
    
    cd "$PROJECT_ROOT"
    if [[ -n "$service" ]]; then
        docker compose -f docker-compose.production.yml logs -f "$service"
    else
        docker compose -f docker-compose.production.yml logs -f
    fi
}

cmd_version() {
    get_current_version
}

cmd_history() {
    if [[ ! -f "$DEPLOY_LOG" ]]; then
        echo -e "${YELLOW}No deployment history found${NC}"
        return 0
    fi
    
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}                    Deployment History${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo ""
    printf "%-24s %-12s %-12s %-12s %-10s %s\n" "TIMESTAMP" "FROM" "TO" "STATUS" "DURATION" "VPS"
    echo "------------------------------------------------------------------------"
    
    # Parse JSON lines and format
    while IFS= read -r line; do
        timestamp=$(echo "$line" | jq -r '.timestamp' 2>/dev/null || echo "")
        version_from=$(echo "$line" | jq -r '.version_from' 2>/dev/null || echo "")
        version_to=$(echo "$line" | jq -r '.version_to' 2>/dev/null || echo "")
        status=$(echo "$line" | jq -r '.status' 2>/dev/null || echo "")
        duration=$(echo "$line" | jq -r '.duration_seconds' 2>/dev/null || echo "")
        vps=$(echo "$line" | jq -r '.vps' 2>/dev/null || echo "")
        
        # Color status
        case "$status" in
            success) status="${GREEN}$status${NC}" ;;
            failed) status="${RED}$status${NC}" ;;
            rolled_back) status="${YELLOW}$status${NC}" ;;
        esac
        
        printf "%-24s %-12s %-12s %-12s %-10s %s\n" "$timestamp" "$version_from" "$version_to" "$status" "${duration}s" "$vps"
    done < "$DEPLOY_LOG"
    
    echo ""
}

cmd_health() {
    if run_all_health_checks; then
        exit 0
    else
        exit 1
    fi
}

# ============== Main ==============

print_usage() {
    echo "QuantMindX CLI - Deployment and Management Tool"
    echo ""
    echo "Usage: quantmind <command> [options]"
    echo ""
    echo "Commands:"
    echo "  update [--version vX.Y.Z]   Update to latest or specific version"
    echo "  rollback [--version vX.Y.Z] Rollback to previous or specific version"
    echo "  status                      Show system status and health"
    echo "  restart [--service X]       Restart all or specific service"
    echo "  logs [--service X]          View logs (follow mode)"
    echo "  version                     Show current version"
    echo "  history                     Show deployment history"
    echo "  health                      Run health checks (exit 0/1)"
    echo ""
    echo "Options:"
    echo "  --version   Target version for update/rollback"
    echo "  --service   Target service for restart/logs"
    echo ""
    echo "Examples:"
    echo "  quantmind update"
    echo "  quantmind update --version v1.2.0"
    echo "  quantmind rollback"
    echo "  quantmind restart --service quantmind-api"
    echo "  quantmind logs --service mt5-bridge"
}

case "${1:-}" in
    update)
        shift
        cmd_update "$@"
        ;;
    rollback)
        shift
        cmd_rollback "$@"
        ;;
    status)
        cmd_status
        ;;
    restart)
        shift
        cmd_restart "$@"
        ;;
    logs)
        shift
        cmd_logs "$@"
        ;;
    version)
        cmd_version
        ;;
    history)
        cmd_history
        ;;
    health)
        cmd_health
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        print_usage
        exit 1
        ;;
esac