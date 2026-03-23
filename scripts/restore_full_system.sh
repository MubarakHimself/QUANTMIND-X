#!/bin/bash
# =============================================================================
# Full System Restore Script
# =============================================================================
# Restores system from a full backup archive for FR69: machine portability.
#
# Usage:
#   ./scripts/restore_full_system.sh <backup_file> [--dry-run] [--validate-only]
#
# Example:
#   ./scripts/restore_full_system.sh backup_20240315_143022.tar.gz
#
# =============================================================================

set -euo pipefail

# Configuration
PROJECT_DIR="${PROJECT_DIR:-/home/mubarkahimself/Desktop/QUANTMINDX}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/.quantmind/backups}"
LOG_DIR="${LOG_DIR:-$HOME/.quantmind/logs}"
LOG_FILE="$LOG_DIR/restore_full_system.log"
NOTIFICATION_FILE="$LOG_DIR/restore_completion.notification"

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
    local audit_log_entry="$LOG_DIR/restore_audit_$(date +%Y%m%d).log"
    echo "[$timestamp] RESTORE_FAILURE | reason: $failure_reason" >> "$audit_log_entry"

    exit 1
}

send_restore_completion() {
    local timestamp=$(date -Iseconds)

    # Log success
    local audit_log_entry="$LOG_DIR/restore_audit_$(date +%Y%m%d).log"
    echo "[$timestamp] RESTORE_SUCCESS | backup: $BACKUP_FILE" >> "$audit_log_entry"

    # Create completion notification for Copilot
    cat > "$NOTIFICATION_FILE" << EOF
=== Full System Restore Completed ===
Backup: $BACKUP_FILE
Completed: $timestamp
Status: SUCCESS

Restored Components:
- Configurations (database, provider configs, server connections)
- Knowledge base (PageIndex, news items)
- Strategy artifacts (TRDs, templates, backtest results)
- Graph memory
- Canvas context

The system is now fully operational.
EOF
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

check_backup_file() {
    log_step "Checking backup file..."

    if [ -z "${BACKUP_FILE:-}" ]; then
        echo "Usage: $0 <backup_file> [--dry-run] [--validate-only]"
        echo ""
        echo "Available backups:"
        ls -la "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "No backups found"
        exit 1
    fi

    # Check if backup file exists
    if [ ! -f "$BACKUP_FILE" ]; then
        # Try to find in backup directory
        if [ -f "$BACKUP_DIR/$BACKUP_FILE" ]; then
            BACKUP_FILE="$BACKUP_DIR/$BACKUP_FILE"
        else
            handle_failure "Backup file not found: $BACKUP_FILE"
        fi
    fi

    # Verify it's a valid tar.gz
    if ! tar tzf "$BACKUP_FILE" &>/dev/null; then
        handle_failure "Invalid backup file: not a valid tar.gz archive"
    fi

    log_success "Backup file valid: $BACKUP_FILE"
}

check_disk_space() {
    log_step "Checking disk space..."

    local available_space=$(df -BG "$PROJECT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')

    if [ "$available_space" -lt 5 ]; then
        handle_failure "Insufficient disk space: ${available_space}GB available (minimum: 5GB)"
    fi

    log_success "Disk space OK: ${available_space}GB available"
}

# =============================================================================
# Restore Modules
# =============================================================================

restore_configs() {
    log_step "Restoring configurations..."

    local restore_dir="$WORK_DIR/configs"

    # Ensure target directories exist
    mkdir -p "$PROJECT_DIR/data/db"
    mkdir -p "$PROJECT_DIR/.quantmind"

    if [ -f "$restore_dir/quantmind.db" ]; then
        log_info "Restoring main database..."
        # Create backup of existing database first
        if [ -f "$PROJECT_DIR/data/db/quantmind.db" ]; then
            cp "$PROJECT_DIR/data/db/quantmind.db" "$PROJECT_DIR/data/db/quantmind.db.pre_restore"
        fi
        sqlite3 "$PROJECT_DIR/data/db/quantmind.db" < "$restore_dir/quantmind.db" 2>/dev/null || \
            cp "$restore_dir/quantmind.db" "$PROJECT_DIR/data/db/quantmind.db"
    fi

    # Restore .quantmind directory from tarball
    if [ -f "$restore_dir/quantmind_dir.tar.gz" ]; then
        log_info "Restoring .quantmind directory..."
        tar xzf "$restore_dir/quantmind_dir.tar.gz" -C "$PROJECT_DIR"
    fi

    # Restore database model files
    if [ -f "$restore_dir/provider_config.py" ]; then
        log_info "Restoring provider config model..."
        cp "$restore_dir/provider_config.py" "$PROJECT_DIR/src/database/models/provider_config.py"
    fi

    # Restore other model files
    for model_file in broker_account.py server_config.py risk_params.py; do
        if [ -f "$restore_dir/$model_file" ]; then
            cp "$restore_dir/$model_file" "$PROJECT_DIR/src/database/models/$model_file"
        fi
    done

    log_success "Configuration restore complete"
}

restore_knowledge_base() {
    log_step "Restoring knowledge base..."

    local restore_dir="$WORK_DIR/knowledge_base"

    # Restore knowledge directory
    if [ -f "$restore_dir/knowledge.tar.gz" ]; then
        log_info "Restoring knowledge directory..."
        tar xzf "$restore_dir/knowledge.tar.gz" -C "$PROJECT_DIR/src"
    fi

    # Restore news items database
    if [ -f "$restore_dir/news_items.db" ]; then
        log_info "Restoring news items database..."
        mkdir -p "$PROJECT_DIR/src/database"
        cp "$restore_dir/news_items.db" "$PROJECT_DIR/src/database/news_items.db"
    fi

    # Restore notification config
    if [ -f "$restore_dir/notification_config.db" ]; then
        log_info "Restoring notification config..."
        cp "$restore_dir/notification_config.db" "$PROJECT_DIR/src/database/notification_config.db"
    fi

    log_success "Knowledge base restore complete"
}

restore_strategy_artifacts() {
    log_step "Restoring strategy artifacts..."

    local restore_dir="$WORK_DIR/strategies"

    # Restore TRD files
    if [ -f "$restore_dir/trd.tar.gz" ]; then
        log_info "Restoring TRD files..."
        tar xzf "$restore_dir/trd.tar.gz" -C "$PROJECT_DIR/src"
    fi

    # Restore strategy templates
    if [ -f "$restore_dir/strategy_templates.tar.gz" ]; then
        log_info "Restoring strategy templates..."
        tar xzf "$restore_dir/strategy_templates.tar.gz" -C "$PROJECT_DIR/src"
    fi

    # Restore backtesting results
    if [ -f "$restore_dir/backtesting.tar.gz" ]; then
        log_info "Restoring backtesting results..."
        tar xzf "$restore_dir/backtesting.tar.gz" -C "$PROJECT_DIR/src"
    fi

    log_success "Strategy artifacts restore complete"
}

restore_graph_memory() {
    log_step "Restoring graph memory..."

    local restore_dir="$WORK_DIR/graph_memory"

    if [ -f "$restore_dir/graph_memory.tar.gz" ]; then
        log_info "Restoring graph memory..."
        tar xzf "$restore_dir/graph_memory.tar.gz" -C "$PROJECT_DIR/src"
    fi

    log_success "Graph memory restore complete"
}

restore_canvas_context() {
    log_step "Restoring canvas context..."

    local restore_dir="$WORK_DIR/canvas_context"

    if [ -f "$restore_dir/canvas_context.tar.gz" ]; then
        log_info "Restoring canvas context..."
        tar xzf "$restore_dir/canvas_context.tar.gz" -C "$PROJECT_DIR/src"
    fi

    log_success "Canvas context restore complete"
}

# =============================================================================
# Validation
# =============================================================================

validate_restore() {
    log_step "Validating restore..."

    local errors=0

    # Check configs
    if [ -f "$PROJECT_DIR/src/database/quantmind.db" ]; then
        if ! sqlite3 "$PROJECT_DIR/src/database/quantmind.db" "SELECT name FROM sqlite_master WHERE type='table';" &>/dev/null; then
            log_warn "Database validation warning"
        fi
    fi

    # Check knowledge base
    if [ -d "$PROJECT_DIR/src/knowledge" ]; then
        log_info "Knowledge base restored"
    else
        log_warn "Knowledge base directory not found"
        ((errors++))
    fi

    # Check strategy artifacts
    if [ -d "$PROJECT_DIR/src/trd" ] || [ -d "$PROJECT_DIR/src/strategy" ]; then
        log_info "Strategy artifacts restored"
    else
        log_warn "Strategy artifacts directory not found"
        ((errors++))
    fi

    # Check graph memory
    if [ -d "$PROJECT_DIR/src/memory/graph" ]; then
        log_info "Graph memory restored"
    else
        log_warn "Graph memory directory not found"
        ((errors++))
    fi

    if [ $errors -gt 0 ]; then
        log_warn "Validation completed with $errors warnings"
    else
        log_success "Full validation passed"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local dry_run=false
    local validate_only=false

    # Check if backup file is provided
    if [ $# -lt 1 ]; then
        echo "Usage: $0 <backup_file> [--dry-run] [--validate-only]"
        echo ""
        echo "Available backups:"
        ls -la "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "No backups found"
        exit 1
    fi

    BACKUP_FILE="$1"
    shift

    # Parse remaining arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run)
                dry_run=true
                shift
                ;;
            --validate-only)
                validate_only=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    echo ""
    echo "=============================================="
    echo "  Full System Restore"
    echo "  Started: $(date -Iseconds)"
    echo "=============================================="
    echo ""

    # Ensure directories exist
    mkdir -p "$BACKUP_DIR"
    mkdir -p "$LOG_DIR"

    # Pre-flight checks
    check_backup_file
    check_disk_space

    # Create working directory
    WORK_DIR=$(mktemp -d)
    mkdir -p "$WORK_DIR"

    # Set up trap for cleanup
    trap cleanup_on_error EXIT

    if [ "$dry_run" = true ]; then
        log_info "DRY RUN MODE - extracting for validation only"
        tar xzf "$BACKUP_FILE" -C "$WORK_DIR"
        log_info "Backup extracted to: $WORK_DIR"
        log_info "No changes have been made to the system"
        exit 0
    fi

    if [ "$validate_only" = true ]; then
        log_info "VALIDATE ONLY MODE - checking backup contents"
        tar tzf "$BACKUP_FILE"
        log_info "Backup contents validated - no restore performed"
        exit 0
    fi

    # Ask for confirmation
    log_warn "This will restore from backup: $BACKUP_FILE"
    log_warn "Existing data may be overwritten. Continue? (yes/no)"
    read -r confirm
    if [ "$confirm" != "yes" ]; then
        log_info "Restore cancelled by user"
        rm -rf "$WORK_DIR"
        exit 0
    fi

    # Extract backup
    log_step "Extracting backup archive..."
    tar xzf "$BACKUP_FILE" -C "$WORK_DIR"
    log_success "Backup extracted"

    # Restore components
    restore_configs
    restore_knowledge_base
    restore_strategy_artifacts
    restore_graph_memory
    restore_canvas_context

    # Validate
    validate_restore

    # Cleanup
    rm -rf "$WORK_DIR"

    # Send completion notification
    send_restore_completion

    echo ""
    echo "=============================================="
    echo "  Restore Completed Successfully"
    echo "  Backup: $BACKUP_FILE"
    echo "  Finished: $(date -Iseconds)"
    echo "=============================================="
    echo ""

    log_success "Restore complete - system is now fully operational"
}

# Run main function
main "$@"