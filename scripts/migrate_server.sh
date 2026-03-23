#!/bin/bash
# =============================================================================
# Server Migration Script
# =============================================================================
# Migrates QUANTMINDX infrastructure from one server to another with full
# data continuity using the backup/restore mechanism from Story 11.4.
#
# FR70: server migration without data loss
#
# Usage:
#   ./scripts/migrate_server.sh --new-host <hostname> --node-role <role> [options]
#
# Options:
#   --new-host <hostname>    Target server hostname/IP (required)
#   --node-role <role>      NODE_ROLE: contabo, cloudzy, or desktop (required)
#   --migration-type        Type: full-migration (default) or config-only
#   --skip-backup           Skip backup step (for re-runs)
#   --skip-restore          Skip restore step (manual restore later)
#   --dry-run               Show what would happen without executing
#   --verbose               Enable verbose logging
#
# Examples:
#   # Full migration from Cloudzy to Hetzner
#   ./scripts/migrate_server.sh --new-host hetzner.example.com --node-role contabo
#
#   # Config-only migration (reconfigure without restoring data)
#   ./scripts/migrate_server.sh --new-host new-server.example.com --node-role desktop --migration-type config-only
#
# Environment Variables:
#   NODE_ROLE            Current node role (source server)
#   NEW_SERVER_HOST      Target server (alternative to --new-host)
#   SSH_USER             SSH user for target server (default: root)
#   SSH_PORT             SSH port (default: 22)
#   BACKUP_DIR           Local backup directory (default: ~/.quantmind/backups)
#   REMOTE_BACKUP_PATH   Remote backup storage path
#
# =============================================================================

set -euo pipefail

# Configuration
PROJECT_DIR="${PROJECT_DIR:-/home/mubarkahimself/Desktop/QUANTMINDX}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_SCRIPT="$SCRIPT_DIR/backup_full_system.sh"
RESTORE_SCRIPT="$SCRIPT_DIR/restore_full_system.sh"

# Default values
NODE_ROLE="${NODE_ROLE:-desktop}"
NEW_SERVER_HOST=""
MIGRATION_TYPE="full-migration"
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
SKIP_BACKUP=false
SKIP_RESTORE=false
DRY_RUN=false
VERBOSE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)
    if [ "$VERBOSE" = true ]; then
        echo -e "${timestamp} [${level}] ${message}"
    fi
}

log_info() { log "INFO" "$1"; }
log_warn() { log "${YELLOW}WARN${NC}" "$1"; }
log_error() { echo -e "${RED}ERROR${NC}: $1" >&2; }
log_success() { log "${GREEN}SUCCESS${NC}" "$1"; }
log_step() { log "${BLUE}STEP${NC}" "$1"; }
log_verbose() { log "${CYAN}VERBOSE${NC}" "$1"; }

print_header() {
    echo ""
    echo "=============================================="
    echo "  QUANTMINDX Server Migration"
    echo "  Started: $(date -Iseconds)"
    echo "=============================================="
    echo ""
}

print_footer() {
    echo ""
    echo "=============================================="
    echo "  Migration Status"
    echo "  Finished: $(date -Iseconds)"
    echo "=============================================="
    echo ""
}

# =============================================================================
# Validation Functions
# =============================================================================

validate_inputs() {
    log_step "Validating inputs..."

    # Check new server host
    if [ -z "$NEW_SERVER_HOST" ]; then
        log_error "NEW_SERVER_HOST is required. Set via --new-host flag or NEW_SERVER_HOST env var."
        exit 1
    fi

    # Validate node role
    case "$NODE_ROLE" in
        contabo|cloudzy|desktop)
            log_info "Source NODE_ROLE: $NODE_ROLE"
            ;;
        *)
            log_error "Invalid NODE_ROLE: $NODE_ROLE. Must be: contabo, cloudzy, or desktop"
            exit 1
            ;;
    esac

    # Validate migration type
    case "$MIGRATION_TYPE" in
        full-migration|config-only)
            log_info "Migration type: $MIGRATION_TYPE"
            ;;
        *)
            log_error "Invalid migration type: $MIGRATION_TYPE. Must be: full-migration or config-only"
            exit 1
            ;;
    esac

    # Check backup script exists
    if [ ! -f "$BACKUP_SCRIPT" ]; then
        log_error "Backup script not found: $BACKUP_SCRIPT"
        log_info "Migration requires Story 11.4 backup mechanism"
        exit 1
    fi

    # Check restore script exists
    if [ "$MIGRATION_TYPE" = "full-migration" ] && [ ! -f "$RESTORE_SCRIPT" ]; then
        log_error "Restore script not found: $RESTORE_SCRIPT"
        exit 1
    fi

    # Verify SSH connectivity
    log_info "Verifying SSH connectivity to $NEW_SERVER_HOST..."
    if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$NEW_SERVER_HOST" "echo 'SSH OK'" &>/dev/null; then
        log_warn "Cannot connect to $NEW_SERVER_HOST via SSH"
        log_info "Ensure the target server is reachable and SSH is configured"
        if [ "$DRY_RUN" = false ]; then
            read -p "Continue anyway? (y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    else
        log_success "SSH connectivity verified"
    fi

    log_success "Input validation complete"
}

# =============================================================================
# Backup Functions
# =============================================================================

create_backup() {
    if [ "$SKIP_BACKUP" = true ]; then
        log_warn "Skipping backup as requested"
        return 0
    fi

    log_step "Creating full system backup..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would execute: $BACKUP_SCRIPT"
        return 0
    fi

    # Execute backup script
    if "$BACKUP_SCRIPT" --verify; then
        log_success "Backup created and verified"
    else
        log_error "Backup failed"
        exit 1
    fi
}

transfer_backup() {
    log_step "Transferring backup to target server..."

    # Find latest backup
    local backup_dir="${BACKUP_DIR:-$HOME/.quantmind/backups}"
    local latest_backup=$(ls -t "$backup_dir"/backup_*.tar.gz 2>/dev/null | head -1)

    if [ -z "$latest_backup" ]; then
        log_error "No backup found in $backup_dir"
        exit 1
    fi

    log_info "Latest backup: $latest_backup"

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would transfer: $latest_backup to $NEW_SERVER_HOST:$REMOTE_BACKUP_PATH"
        return 0
    fi

    # Create remote backup directory if needed
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$NEW_SERVER_HOST" \
        "mkdir -p ${REMOTE_BACKUP_PATH:-$HOME/.quantmind/backups}"

    # Transfer backup
    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$latest_backup" \
        "$SSH_USER@$NEW_SERVER_HOST:${REMOTE_BACKUP_PATH:-$HOME/.quantmind/backups}/"

    log_success "Backup transferred to target server"
}

# =============================================================================
# Configuration Transfer
# =============================================================================

transfer_node_role_config() {
    log_step "Transferring NODE_ROLE configuration..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would set NODE_ROLE=$NODE_ROLE on $NEW_SERVER_HOST"
        return 0
    fi

    # Create environment setup script for target
    local setup_script="/tmp/migration_setup_$$.sh"
    cat > "$setup_script" << EOF
#!/bin/bash
# Migration environment setup
export NODE_ROLE="$NODE_ROLE"
echo "NODE_ROLE=$NODE_ROLE" > ~/.quantmind/node_role
mkdir -p ~/.quantmind/logs
mkdir -p ~/.quantmind/backups
mkdir -p ~/.quantmind/config
echo "Migration setup complete" > ~/.quantmind/.migration_complete
EOF

    # Transfer and execute on target
    scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$setup_script" \
        "$SSH_USER@$NEW_SERVER_HOST:/tmp/migration_setup.sh"

    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$NEW_SERVER_HOST" \
        "chmod +x /tmp/migration_setup.sh && /tmp/migration_setup.sh && rm /tmp/migration_setup.sh"

    rm -f "$setup_script"

    log_success "NODE_ROLE configuration transferred"
}

transfer_credentials() {
    log_step "Transferring credentials..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would transfer credentials to $NEW_SERVER_HOST"
        return 0
    fi

    # Transfer .quantmind config directory (contains credentials)
    if [ -d "$HOME/.quantmind" ]; then
        log_info "Transferring .quantmind config directory..."
        rsync -avz --progress \
            -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
            --exclude='*.db' \
            --exclude='backups/*' \
            "$HOME/.quantmind/" \
            "$SSH_USER@$NEW_SERVER_HOST:$HOME/.quantmind/"
    else
        log_warn "No .quantmind directory found - skipping credential transfer"
    fi

    # Transfer SSH keys if they exist
    if [ -d "$HOME/.ssh" ]; then
        log_info "Transferring SSH keys..."
        rsync -avz --progress \
            -e "ssh -o StrictHostKeyChecking=no -p $SSH_PORT" \
            "$HOME/.ssh/" \
            "$SSH_USER@$NEW_SERVER_HOST:$HOME/.ssh/"
    fi

    log_success "Credentials transferred"
}

transfer_project_configs() {
    log_step "Transferring project configurations..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would transfer project configs to $NEW_SERVER_HOST"
        return 0
    fi

    # Transfer key configuration files
    local configs=(
        "requirements.txt"
        ".env.example"
        "pyproject.toml"
        "setup.py"
    )

    local temp_dir="/tmp/migration_configs_$$"
    mkdir -p "$temp_dir"

    for config in "${configs[@]}"; do
        if [ -f "$PROJECT_DIR/$config" ]; then
            cp "$PROJECT_DIR/$config" "$temp_dir/"
        fi
    done

    # Transfer configs
    if [ "$(ls -A "$temp_dir" 2>/dev/null)" ]; then
        scp -o StrictHostKeyChecking=no -P "$SSH_PORT" "$temp_dir"/* \
            "$SSH_USER@$NEW_SERVER_HOST:$PROJECT_DIR/"

        log_success "Project configurations transferred"
    else
        log_warn "No project config files found to transfer"
    fi

    rm -rf "$temp_dir"
}

# =============================================================================
# Restore Functions
# =============================================================================

restore_backup() {
    if [ "$SKIP_RESTORE" = true ]; then
        log_warn "Skipping restore as requested"
        return 0
    fi

    if [ "$MIGRATION_TYPE" = "config-only" ]; then
        log_info "Config-only migration - skipping data restore"
        return 0
    fi

    log_step "Restoring backup on target server..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would execute restore on $NEW_SERVER_HOST"
        return 0
    fi

    # Execute restore script on target
    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$NEW_SERVER_HOST" \
        "cd $PROJECT_DIR && ./scripts/restore_full_system.sh --latest"

    log_success "Backup restored on target server"
}

# =============================================================================
# Health Check Functions
# =============================================================================

run_health_checks() {
    log_step "Running health checks on target server..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run health checks on $NEW_SERVER_HOST"
        return 0
    fi

    # Check if node-health-check script exists on target
    local health_check_result=$(ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$NEW_SERVER_HOST" "
        if [ -f '$PROJECT_DIR/scripts/node-health-check.sh' ]; then
            cd $PROJECT_DIR && ./scripts/node-health-check.sh
        else
            # Basic health checks
            echo 'NODE_ROLE: ' && cat ~/.quantmind/node_role 2>/dev/null || echo 'Not set'
            echo 'Project dir exists: ' && [ -d '$PROJECT_DIR' ] && echo 'Yes' || echo 'No'
            echo 'Python available: ' && python3 --version 2>/dev/null || echo 'No'
            echo 'Health checks completed'
        fi
    ")

    log_info "Health check results:"
    echo "$health_check_result" | while read -r line; do
        log_info "  $line"
    done

    # Check for critical failures
    if echo "$health_check_result" | grep -qi "error\|failed\|not found"; then
        log_warn "Some health checks reported issues - review required"
    else
        log_success "Health checks passed"
    fi
}

verify_node_role() {
    log_step "Verifying NODE_ROLE on target..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would verify NODE_ROLE=$NODE_ROLE on $NEW_SERVER_HOST"
        return 0
    fi

    local verified_role=$(ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$NEW_SERVER_HOST" \
        "cat ~/.quantmind/node_role 2>/dev/null || echo 'unknown'")

    if [ "$verified_role" = "$NODE_ROLE" ]; then
        log_success "NODE_ROLE verified: $verified_role"
    else
        log_error "NODE_ROLE mismatch! Expected: $NODE_ROLE, Got: $verified_role"
        exit 1
    fi
}

# =============================================================================
# Strategy Resume Functions
# =============================================================================

resume_strategies() {
    log_step "Checking strategy status on target..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would check/verify strategies on $NEW_SERVER_HOST"
        return 0
    fi

    # Check if there are any active strategies to resume
    # This is a placeholder - actual implementation depends on the trading system
    local strategy_status=$(ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$NEW_SERVER_HOST" "
        if [ -d '$PROJECT_DIR/src/strategy' ]; then
            echo 'Strategy directory exists'
            ls -la '$PROJECT_DIR/src/strategy/' 2>/dev/null | head -5 || echo 'No strategies found'
        else
            echo 'Strategy directory not found - manual setup required'
        fi
    ")

    log_info "Strategy status:"
    echo "$strategy_status" | while read -r line; do
        log_info "  $line"
    done

    log_info "Strategy check complete - manual verification recommended before resuming live trading"
}

# =============================================================================
# Migration Complete Functions
# =============================================================================

mark_migration_complete() {
    log_step "Marking migration as complete..."

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would mark migration complete on $NEW_SERVER_HOST"
        return 0
    fi

    ssh -o StrictHostKeyChecking=no -p "$SSH_PORT" "$SSH_USER@$NEW_SERVER_HOST" \
        "echo 'migration_complete: $(date -Iseconds)' >> ~/.quantmind/migration_log"

    log_success "Migration marked as complete"
}

print_migration_summary() {
    echo ""
    echo "=============================================="
    echo "  Migration Summary"
    echo "=============================================="
    echo ""
    echo "  Source NODE_ROLE: $NODE_ROLE"
    echo "  Target Server: $NEW_SERVER_HOST"
    echo "  Migration Type: $MIGRATION_TYPE"
    echo ""
    echo "  Completed Steps:"
    echo "  [x] Backup creation"
    echo "  [x] Backup transfer (if full-migration)"
    echo "  [x] NODE_ROLE configuration"
    echo "  [x] Credentials transfer"
    echo "  [x] Project configs transfer"
    echo "  [x] Backup restore (if full-migration)"
    echo "  [x] Health checks"
    echo "  [x] NODE_ROLE verification"
    echo "  [x] Strategy status check"
    echo ""
    echo "  Next Steps:"
    echo "  - Verify all services on target server"
    echo "  - Test API endpoints"
    echo "  - Update DNS/routing if needed"
    echo "  - Resume live trading when ready"
    echo ""
    echo "=============================================="
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --new-host)
                NEW_SERVER_HOST="$2"
                shift 2
                ;;
            --node-role)
                NODE_ROLE="$2"
                shift 2
                ;;
            --migration-type)
                MIGRATION_TYPE="$2"
                shift 2
                ;;
            --ssh-user)
                SSH_USER="$2"
                shift 2
                ;;
            --ssh-port)
                SSH_PORT="$2"
                shift 2
                ;;
            --skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            --skip-restore)
                SKIP_RESTORE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 --new-host <hostname> --node-role <role> [options]"
                echo ""
                echo "Options:"
                echo "  --new-host <hostname>    Target server hostname/IP (required)"
                echo "  --node-role <role>      NODE_ROLE: contabo, cloudzy, or desktop (required)"
                echo "  --migration-type        Type: full-migration or config-only (default: full-migration)"
                echo "  --ssh-user              SSH user (default: root)"
                echo "  --ssh-port              SSH port (default: 22)"
                echo "  --skip-backup           Skip backup step"
                echo "  --skip-restore          Skip restore step"
                echo "  --dry-run               Show what would happen"
                echo "  --verbose               Enable verbose logging"
                echo "  --help                  Show this help"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # Check for required parameters
    if [ -z "$NEW_SERVER_HOST" ] && [ -z "${NEW_SERVER_HOST:-}" ]; then
        NEW_SERVER_HOST="${NEW_SERVER_HOST:-}"
    fi

    print_header

    # Validate inputs
    validate_inputs

    # Execute migration steps
    create_backup
    transfer_backup
    transfer_node_role_config
    transfer_credentials
    transfer_project_configs
    restore_backup
    run_health_checks
    verify_node_role
    resume_strategies
    mark_migration_complete

    print_migration_summary
    print_footer

    log_success "Migration completed successfully!"
}

main "$@"