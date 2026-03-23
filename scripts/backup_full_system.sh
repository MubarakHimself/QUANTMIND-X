#!/bin/bash
# =============================================================================
# Full System Backup Script
# =============================================================================
# Backs up all system configurations, knowledge base, strategy artifacts,
# and graph memory for FR69: machine portability.
#
# Usage:
#   ./scripts/backup_full_system.sh [--dry-run] [--verify] [--incremental]
#
# Output:
#   backup_YYYYMMDD_HHMMSS.tar.gz - Full backup archive
#   backup_YYYYMMDD_HHMMSS_manifest.json - Backup manifest with checksums
#
# =============================================================================

set -euo pipefail

# Configuration
PROJECT_DIR="${PROJECT_DIR:-/home/mubarkahimself/Desktop/QUANTMINDX}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/.quantmind/backups}"
LOG_DIR="${LOG_DIR:-$HOME/.quantmind/logs}"
LOG_FILE="$LOG_DIR/backup_full_system.log"
NOTIFICATION_FILE="$LOG_DIR/backup_failure.notification"

# Remote backup configuration (for off-site backup)
REMOTE_BACKUP_ENABLED="${REMOTE_BACKUP_ENABLED:-false}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PATH="${REMOTE_PATH:-/backup/quantmindx}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/backup_key}"

# Backup retention (days)
RETENTION_DAYS="${RETENTION_DAYS:-30}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() { log "INFO" "$1"; }
log_warn() { log "${YELLOW}WARN${NC}" "$1"; }
log_error() { log "${RED}ERROR${NC}" "$1"; }
log_success() { log "${GREEN}SUCCESS${NC}" "$1"; }
log_step() { log "${BLUE}STEP${NC}" "$1"; }

cleanup_on_error() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        handle_failure "Script exited with code $exit_code"
    fi
}

handle_failure() {
    local failure_reason="$1"
    local timestamp=$(date -Iseconds)

    log_error "FAILURE: $failure_reason"

    # Log to audit trail
    local audit_log_entry="$LOG_DIR/backup_audit_$(date +%Y%m%d).log"
    echo "[$timestamp] BACKUP_FAILURE | reason: $failure_reason" >> "$audit_log_entry"

    # Create notification file
    echo "Full system backup failed — $failure_reason" > "$NOTIFICATION_FILE"
    echo "Timestamp: $timestamp" >> "$NOTIFICATION_FILE"

    log_info "Failure logged. Notification file created at $NOTIFICATION_FILE"

    # Cleanup partial backup if exists
    if [ -n "${backup_name:-}" ] && [ -d "$BACKUP_DIR/$backup_name" ]; then
        rm -rf "$BACKUP_DIR/$backup_name"
    fi

    exit 1
}

send_success_notification() {
    # Remove any existing failure notification
    if [ -f "$NOTIFICATION_FILE" ]; then
        rm -f "$NOTIFICATION_FILE"
    fi

    local audit_log_entry="$LOG_DIR/backup_audit_$(date +%Y%m%d).log"
    local timestamp=$(date -Iseconds)
    echo "[$timestamp] BACKUP_SUCCESS" >> "$audit_log_entry"
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

check_dependencies() {
    log_step "Checking dependencies..."

    local deps=("tar" "sha256sum" "date" "jq")
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            handle_failure "Required command not found: $dep"
        fi
    done

    log_success "All dependencies available"
}

check_disk_space() {
    log_step "Checking disk space..."

    # Calculate required space (rough estimate: 2x current data size)
    local data_size=$(du -sb "$PROJECT_DIR" 2>/dev/null | cut -f1 || echo "0")
    local required_space=$((data_size * 2 / 1024 / 1024 / 1024 + 1))  # GB

    local available_space=$(df -BG "$BACKUP_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')

    if [ "$available_space" -lt "$required_space" ] && [ "$available_space" -lt 5 ]; then
        handle_failure "Insufficient disk space: ${available_space}GB available (estimated need: ${required_space}GB)"
    fi

    log_success "Disk space OK: ${available_space}GB available"
}

# =============================================================================
# Backup Modules (sourced from separate files)
# =============================================================================

backup_configs() {
    log_step "Backing up configurations..."

    local config_dir="$WORK_DIR/configs"
    mkdir -p "$config_dir"

    # Backup database models (SQLite) - correct path is data/db/
    if [ -f "$PROJECT_DIR/data/db/quantmind.db" ]; then
        log_info "Backing up main SQLite database..."
        sqlite3 "$PROJECT_DIR/data/db/quantmind.db" ".backup $config_dir/quantmind.db"
    fi

    # Backup .quantmind directory (department_mail.db, configs, etc.)
    if [ -d "$PROJECT_DIR/.quantmind" ]; then
        log_info "Backing up .quantmind directory..."
        tar czf "$config_dir/quantmind_dir.tar.gz" -C "$PROJECT_DIR" .quantmind/
    fi

    # Backup provider configs (JSON/YAML)
    local provider_config_file="$PROJECT_DIR/src/database/models/provider_config.py"
    # Export provider configs as JSON
    if [ -f "$provider_config_file" ]; then
        log_info "Backing up provider configs..."
        # Note: Provider credentials are typically stored encrypted
        # We backup the model definition, not actual credentials
        cp "$provider_config_file" "$config_dir/provider_config.py"
    fi

    # Backup server config
    if [ -d "$PROJECT_DIR/src/database/models/server_config.py" ]; then
        cp "$PROJECT_DIR/src/database/models/"*.py "$config_dir/"
    fi

    # Backup broker account configs
    if [ -d "$PROJECT_DIR/src/database/models/broker_account.py" ]; then
        cp "$PROJECT_DIR/src/database/models/"*.py "$config_dir/"
    fi

    # Backup risk parameters
    if [ -d "$PROJECT_DIR/src/database/models/risk_params.py" ]; then
        cp "$PROJECT_DIR/src/database/models/"*.py "$config_dir/"
    fi

    log_success "Configuration backup complete"
}

backup_knowledge_base() {
    log_step "Backing up knowledge base..."

    local kb_dir="$WORK_DIR/knowledge_base"
    mkdir -p "$kb_dir"

    # Backup knowledge base directory (PageIndex data)
    if [ -d "$PROJECT_DIR/src/knowledge" ]; then
        log_info "Backing up knowledge directory..."
        tar czf "$kb_dir/knowledge.tar.gz" -C "$PROJECT_DIR" src/knowledge/
    fi

    # Backup news items database
    if [ -f "$PROJECT_DIR/src/database/news_items.db" ]; then
        log_info "Backing up news items database..."
        cp "$PROJECT_DIR/src/database/news_items.db" "$kb_dir/"
    fi

    # Backup notification config
    if [ -f "$PROJECT_DIR/src/database/notification_config.db" ]; then
        log_info "Backing up notification config..."
        cp "$PROJECT_DIR/src/database/notification_config.db" "$kb_dir/"
    fi

    log_success "Knowledge base backup complete"
}

backup_strategy_artifacts() {
    log_step "Backing up strategy artifacts..."

    local strategy_dir="$WORK_DIR/strategies"
    mkdir -p "$strategy_dir"

    # Backup TRD files
    if [ -d "$PROJECT_DIR/src/trd" ]; then
        log_info "Backing up TRD files..."
        tar czf "$strategy_dir/trd.tar.gz" -C "$PROJECT_DIR" src/trd/
    fi

    # Backup strategy templates
    if [ -d "$PROJECT_DIR/src/strategy" ]; then
        log_info "Backing up strategy templates..."
        tar czf "$strategy_dir/strategy_templates.tar.gz" -C "$PROJECT_DIR" src/strategy/
    fi

    # Backup backtesting results
    if [ -d "$PROJECT_DIR/src/backtesting" ]; then
        log_info "Backing up backtesting results..."
        tar czf "$strategy_dir/backtesting.tar.gz" -C "$PROJECT_DIR" src/backtesting/
    fi

    log_success "Strategy artifacts backup complete"
}

backup_graph_memory() {
    log_step "Backing up graph memory..."

    local memory_dir="$WORK_DIR/graph_memory"
    mkdir -p "$memory_dir"

    # Backup graph memory store
    if [ -d "$PROJECT_DIR/src/memory/graph" ]; then
        log_info "Backing up graph memory..."
        # Exclude __pycache__ directories
        tar czf "$memory_dir/graph_memory.tar.gz" \
            --exclude='*/__pycache__' \
            -C "$PROJECT_DIR" src/memory/graph/
    fi

    log_success "Graph memory backup complete"
}

backup_canvas_context() {
    log_step "Backing up canvas context..."

    local canvas_dir="$WORK_DIR/canvas_context"
    mkdir -p "$canvas_dir"

    # Backup canvas context templates
    if [ -d "$PROJECT_DIR/src/canvas_context" ]; then
        log_info "Backing up canvas context..."
        tar czf "$canvas_dir/canvas_context.tar.gz" \
            --exclude='*/__pycache__' \
            -C "$PROJECT_DIR" src/canvas_context/
    fi

    log_success "Canvas context backup complete"
}

# =============================================================================
# Checksum Generation & Verification
# =============================================================================

generate_checksums() {
    log_step "Generating checksums..."

    local checksum_file="$WORK_DIR/manifest.sha256"

    # Generate SHA256 checksums for all files
    find "$WORK_DIR" -type f -exec sha256sum {} \; > "$checksum_file"

    log_success "Checksums generated: $checksum_file"
}

verify_backup() {
    log_step "Verifying backup integrity..."

    local checksum_file="$WORK_DIR/manifest.sha256"

    if [ ! -f "$checksum_file" ]; then
        handle_failure "Checksum file not found"
    fi

    # Verify all files
    if ! sha256sum --check "$checksum_file"; then
        handle_failure "Checksum verification failed"
    fi

    log_success "Backup verification passed"
}

# =============================================================================
# Archive Creation
# =============================================================================

create_archive() {
    log_step "Creating final archive..."

    local timestamp=$(date +%Y%m%d_%H%M%S)
    backup_name="backup_$timestamp"

    # Create archive in backup directory
    tar czf "$BACKUP_DIR/${backup_name}.tar.gz" -C "$WORK_DIR" .

    # Store backup name for later use
    echo "$backup_name" > "$WORK_DIR/.backup_name"

    log_success "Archive created: ${backup_name}.tar.gz"
}

# =============================================================================
# Remote Backup (if configured)
# =============================================================================

sync_to_remote() {
    if [ "$REMOTE_BACKUP_ENABLED" != "true" ]; then
        log_info "Remote backup disabled - skipping"
        return 0
    fi

    log_step "Syncing to remote backup..."

    if [ -z "$REMOTE_HOST" ]; then
        log_warn "REMOTE_HOST not set - skipping remote sync"
        return 0
    fi

    # Check SSH key
    if [ ! -f "$SSH_KEY" ]; then
        log_warn "SSH key not found at $SSH_KEY - skipping remote sync"
        return 0
    fi

    # Create remote directory if needed
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE_HOST" \
        "mkdir -p $REMOTE_PATH" || true

    # Sync backup archive
    if rsync -avz --progress \
        -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
        "$BACKUP_DIR/${backup_name}.tar.gz" \
        "${REMOTE_HOST}:${REMOTE_PATH}/"; then
        log_success "Remote backup synced successfully"
    else
        log_warn "Remote sync failed - local backup still available"
    fi
}

# =============================================================================
# Cleanup Old Backups
# =============================================================================

cleanup_old_backups() {
    log_step "Cleaning up old backups (retention: $RETENTION_DAYS days)..."

    # Find and remove backups older than retention period
    find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

    # Also clean up old remote backups if configured
    if [ "$REMOTE_BACKUP_ENABLED" = "true" ] && [ -n "$REMOTE_HOST" ]; then
        ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$REMOTE_HOST" \
            "find $REMOTE_PATH -name 'backup_*.tar.gz' -mtime +$RETENTION_DAYS -delete" || true
    fi

    log_success "Old backups cleaned up"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local dry_run=false
    local verify=false
    local incremental=false

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run)
                dry_run=true
                shift
                ;;
            --verify)
                verify=true
                shift
                ;;
            --incremental)
                incremental=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--dry-run] [--verify] [--incremental]"
                exit 1
                ;;
        esac
    done

    echo ""
    echo "=============================================="
    echo "  Full System Backup"
    echo "  Started: $(date -Iseconds)"
    echo "=============================================="
    echo ""

    # Ensure directories exist
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"

    # Set up trap for cleanup
    trap cleanup_on_error EXIT

    # Pre-flight checks
    check_dependencies
    check_disk_space

    # Create working directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    backup_name="backup_$timestamp"
    WORK_DIR="$BACKUP_DIR/$backup_name"
    mkdir -p "$WORK_DIR"

    if [ "$dry_run" = true ]; then
        log_info "DRY RUN MODE - no actual backup will occur"
        exit 0
    fi

    # Execute backup modules
    backup_configs
    backup_knowledge_base
    backup_strategy_artifacts
    backup_graph_memory
    backup_canvas_context

    # Generate checksums
    generate_checksums

    # Verify if requested
    if [ "$verify" = true ]; then
        verify_backup
    fi

    # Create final archive
    create_archive

    # Sync to remote if configured
    sync_to_remote

    # Cleanup old backups
    cleanup_old_backups

    # Success!
    send_success_notification

    echo ""
    echo "=============================================="
    echo "  Backup Completed Successfully"
    echo "  Backup: ${backup_name}.tar.gz"
    echo "  Location: $BACKUP_DIR"
    echo "  Finished: $(date -Iseconds)"
    echo "=============================================="
    echo ""

    # Remove working directory (archive is sufficient)
    rm -rf "$WORK_DIR"
}

# Run main function
main "$@"