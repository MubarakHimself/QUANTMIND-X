#!/bin/bash
# =============================================================================
# Nightly rsync: Cloudzy → Contabo
# =============================================================================
# Syncs trade records, tick data, and configs from Cloudzy to Contabo.
# NFR-R6: 3-day backup cadence
# NFR-D5: Data integrity verification
#
# Usage:
#   ./scripts/sync_cloudzy_to_contabo.sh [--dry-run] [--verify-only]
#
# Environment:
#   CLOUDZY_HOST: SSH hostname for Cloudzy (default: from config)
#   CONTABO_HOST: SSH hostname for Contabo backup
#   SSH_KEY: Path to SSH key for rsync (default: ~/.ssh/rsync_key)
#
# =============================================================================

set -euo pipefail

# Configuration - should match environment-specific values
CLOUDZY_HOST="${CLOUDZY_HOST:-}"
CONTABO_HOST="${CONTABO_HOST:-}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/rsync_key}"
PROJECT_DIR="${PROJECT_DIR:-/opt/quantmindx}"
LOG_DIR="${LOG_DIR:-/var/log/quantmindx}"
LOG_FILE="$LOG_DIR/rsync_cloudzy_contabo.log"
NOTIFICATION_FILE="$LOG_DIR/rsync_failure.notification"

# Source directories on Cloudzy (remote)
CLOUDZY_TRADES_DIR="${CLOUDZY_TRADES_DIR:-/home/quantmind/data/trades}"
CLOUDZY_TICK_DATA_DIR="${CLOUDZY_TICK_DATA_DIR:-/home/quantmind/data/tick_data_warm}"
CLOUDZY_CONFIG_DIR="${CLOUDZY_CONFIG_DIR:-/home/quantmind/config}"

# Destination directory on Contabo (local backup)
CONTABO_BACKUP_DIR="${CONTABO_BACKUP_DIR:-/backup/cloudzy}"

# Retry configuration
MAX_RETRIES=3
RETRY_DELAY_BASE=30  # seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date -Iseconds)
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

log_step() {
    log "${BLUE}STEP${NC}" "$1"
}

# Cleanup function for trap
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        handle_failure "Script exited with code $exit_code"
    fi
}

# =============================================================================
# Failure Handling & Notifications
# =============================================================================

handle_failure() {
    local failure_reason="$1"
    local timestamp=$(date -Iseconds)

    log_error "FAILURE: $failure_reason"

    # Log to audit trail via file (will be processed by API later)
    local audit_log_entry="$LOG_DIR/rsync_audit_$(date +%Y%m%d).log"
    echo "[$timestamp] RSYNC_FAILURE | reason: $failure_reason | cloudzy_host: $CLOUDZY_HOST" >> "$audit_log_entry"

    # Create notification file for morning notification
    echo "Nightly rsync failed — $failure_reason. Manual sync recommended." > "$NOTIFICATION_FILE"
    echo "Timestamp: $timestamp" >> "$NOTIFICATION_FILE"
    echo "Cloudzy Host: $CLOUDZY_HOST" >> "$NOTIFICATION_FILE"

    log_info "Failure logged. Notification file created at $NOTIFICATION_FILE"

    # Exit with error
    exit 1
}

send_success_notification() {
    # Remove any existing failure notification
    if [ -f "$NOTIFICATION_FILE" ]; then
        rm -f "$NOTIFICATION_FILE"
    fi

    local audit_log_entry="$LOG_DIR/rsync_audit_$(date +%Y%m%d).log"
    local timestamp=$(date -Iseconds)
    echo "[$timestamp] RSYNC_SUCCESS" >> "$audit_log_entry"
}

# =============================================================================
# SSH and Connectivity Checks
# =============================================================================

check_ssh_key() {
    log_step "Checking SSH key..."

    if [ ! -f "$SSH_KEY" ]; then
        handle_failure "SSH key not found at $SSH_KEY"
    fi

    if [ ! -f "${SSH_KEY}.pub" ]; then
        log_warn "SSH public key not found at ${SSH_KEY}.pub"
    fi

    # Check key permissions
    chmod 600 "$SSH_KEY" 2>/dev/null || true

    log_success "SSH key found: $SSH_KEY"
}

test_ssh_connectivity() {
    log_step "Testing SSH connectivity to Cloudzy..."

    if [ -z "$CLOUDZY_HOST" ]; then
        handle_failure "CLOUDZY_HOST not configured"
    fi

    # Test SSH connection (with timeout)
    if ! ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 "$CLOUDZY_HOST" "echo 'Connection OK'" >/dev/null 2>&1; then
        handle_failure "Cannot connect to Cloudzy host: $CLOUDZY_HOST"
    fi

    log_success "SSH connectivity to Cloudzy OK"
}

check_disk_space() {
    log_step "Checking disk space on Contabo..."

    local available_space=$(df -BG "$CONTABO_BACKUP_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')

    # Require at least 10GB free
    if [ "$available_space" -lt 10 ]; then
        handle_failure "Insufficient disk space: ${available_space}GB available (minimum: 10GB)"
    fi

    log_success "Disk space OK: ${available_space}GB available"
}

# =============================================================================
# Source Directory Validation
# =============================================================================

validate_source_directories() {
    log_step "Validating source directories on Cloudzy..."

    local dirs=("$CLOUDZY_TRADES_DIR" "$CLOUDZY_TICK_DATA_DIR" "$CLOUDZY_CONFIG_DIR")

    for dir in "${dirs[@]}"; do
        if ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$CLOUDZY_HOST" "test -d $dir" 2>/dev/null; then
            log_info "Source directory exists: $dir"
        else
            log_warn "Source directory not found: $dir"
        fi
    done

    log_success "Source directory validation complete"
}

# =============================================================================
# Checksum Generation (Task 2.1)
# =============================================================================

generate_checksums() {
    local source_dir="$1"
    local checksum_file="$2"

    log_info "Generating checksums for $source_dir"

    # Generate SHA256 checksums for all files in the directory
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$CLOUDZY_HOST" \
        "find $source_dir -type f -name '*.db' -o -name '*.duckdb' -o -name '*.parquet' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' 2>/dev/null | xargs -I {} sha256sum {}" > "$checksum_file"

    log_success "Checksums generated: $checksum_file"
}

# =============================================================================
# Rsync Execution with Retry (Task 1 & Task 2.3)
# =============================================================================

run_rsync_with_retry() {
    local source_path="$1"
    local dest_path="$2"
    local attempt=1

    while [ $attempt -le $MAX_RETRIES ]; do
        log_info "Rsync attempt $attempt of $MAX_RETRIES: $source_path -> $dest_path"

        # Run rsync with:
        # -a: archive mode (preserves permissions, timestamps, etc.)
        # -v: verbose
        # -z: compress during transfer
        # --progress: show progress
        # --delete: delete files not in source
        # --checksum: use checksums (slower but more accurate)
        if rsync -avz --progress \
            --delete \
            --checksum \
            -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
            "${CLOUDZY_HOST}:${source_path}/" \
            "${dest_path}/" >> "$LOG_FILE" 2>&1; then

            log_success "Rsync completed successfully on attempt $attempt"
            return 0
        else
            log_warn "Rsync attempt $attempt failed"

            if [ $attempt -lt $MAX_RETRIES ]; then
                local delay=$((RETRY_DELAY_BASE * attempt))
                log_info "Waiting ${delay}s before retry..."
                sleep $delay
            fi
        fi

        attempt=$((attempt + 1))
    done

    handle_failure "Rsync failed after $MAX_RETRIES attempts"
}

# =============================================================================
# Checksum Validation (Task 2.2)
# =============================================================================

verify_checksums() {
    local source_dir="$1"
    local local_dir="$2"

    log_step "Verifying checksums..."

    # Generate checksums for source
    local source_checksum="/tmp/checksum_source_$$.txt"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no "$CLOUDZY_HOST" \
        "cd $source_dir && find . -type f \( -name '*.db' -o -name '*.duckdb' -o -name '*.parquet' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) -exec sha256sum {} \; | sort" > "$source_checksum"

    # Generate checksums for local copy
    local local_checksum="/tmp/checksum_local_$$.txt"
    cd "$local_dir" && find . -type f \( -name '*.db' -o -name '*.duckdb' -o -name '*.parquet' -o -name '*.yaml' -o -name '*.yml' -o -name '*.json' \) -exec sha256sum {} \; | sort > "$local_checksum"

    # Compare checksums
    if diff "$source_checksum" "$local_checksum" > /dev/null 2>&1; then
        log_success "Checksum verification passed"
        rm -f "$source_checksum" "$local_checksum"
        return 0
    else
        log_error "Checksum verification FAILED - files may be corrupted"
        rm -f "$source_checksum" "$local_checksum"
        return 1
    fi
}

# =============================================================================
# Main Sync Operations
# =============================================================================

sync_trade_records() {
    log_step "Syncing trade records (SQLite)..."

    local dest_dir="$CONTABO_BACKUP_DIR/trades"
    mkdir -p "$dest_dir"

    run_rsync_with_retry "$CLOUDZY_TRADES_DIR" "$dest_dir"

    # Verify checksums
    if ! verify_checksums "$CLOUDZY_TRADES_DIR" "$dest_dir"; then
        handle_failure "Checksum verification failed for trade records"
    fi

    log_success "Trade records synced and verified"
}

sync_tick_data() {
    log_step "Syncing tick data (DuckDB warm storage)..."

    local dest_dir="$CONTABO_BACKUP_DIR/tick_data_warm"
    mkdir -p "$dest_dir"

    run_rsync_with_retry "$CLOUDZY_TICK_DATA_DIR" "$dest_dir"

    # Verify checksums
    if ! verify_checksums "$CLOUDZY_TICK_DATA_DIR" "$dest_dir"; then
        handle_failure "Checksum verification failed for tick data"
    fi

    log_success "Tick data synced and verified"
}

sync_config_files() {
    log_step "Syncing configuration files..."

    local dest_dir="$CONTABO_BACKUP_DIR/config"
    mkdir -p "$dest_dir"

    run_rsync_with_retry "$CLOUDZY_CONFIG_DIR" "$dest_dir"

    # Verify checksums
    if ! verify_checksums "$CLOUDZY_CONFIG_DIR" "$dest_dir"; then
        handle_failure "Checksum verification failed for config files"
    fi

    log_success "Config files synced and verified"
}

# =============================================================================
# Main Execution
# =============================================================================

main() {
    local dry_run=false
    local verify_only=false

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --dry-run)
                dry_run=true
                shift
                ;;
            --verify-only)
                verify_only=true
                shift
                ;;
            *)
                echo "Unknown option: $1"
                echo "Usage: $0 [--dry-run] [--verify-only]"
                exit 1
                ;;
        esac
    done

    echo ""
    echo "=============================================="
    echo "  Nightly Rsync: Cloudzy → Contabo"
    echo "  Started: $(date -Iseconds)"
    echo "=============================================="
    echo ""

    # Ensure log directory exists
    mkdir -p "$LOG_DIR"

    # Set up trap for cleanup
    trap cleanup EXIT

    # Run pre-flight checks
    check_ssh_key
    test_ssh_connectivity
    check_disk_space
    validate_source_directories

    if [ "$dry_run" = true ]; then
        log_info "DRY RUN MODE - no actual sync will occur"
        exit 0
    fi

    if [ "$verify_only" = true ]; then
        log_info "VERIFY ONLY MODE - checking existing data"
        verify_checksums "$CLOUDZY_TRADES_DIR" "$CONTABO_BACKUP_DIR/trades" || handle_failure "Verification failed"
        verify_checksums "$CLOUDZY_TICK_DATA_DIR" "$CONTABO_BACKUP_DIR/tick_data_warm" || handle_failure "Verification failed"
        verify_checksums "$CLOUDZY_CONFIG_DIR" "$CONTABO_BACKUP_DIR/config" || handle_failure "Verification failed"
        exit 0
    fi

    # Perform syncs
    sync_trade_records
    sync_tick_data
    sync_config_files

    # Success!
    send_success_notification

    echo ""
    echo "=============================================="
    echo "  Rsync Completed Successfully"
    echo "  Finished: $(date -Iseconds)"
    echo "=============================================="
    echo ""
}

# Run main function
main "$@"