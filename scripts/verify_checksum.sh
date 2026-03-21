#!/bin/bash
# =============================================================================
# Checksum Verification Script
# =============================================================================
# Standalone checksum verification for rsync transfers.
# NFR-D5: Data integrity verification
#
# Usage:
#   ./scripts/verify_checksum.sh <source_dir> <local_dir>
#   ./scripts/verify_checksum.sh --generate <dir> <output_file>
#   ./scripts/verify_checksum.sh --verify <checksum_file> <dir>
#
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }

# =============================================================================
# Generate Checksums
# =============================================================================

generate_checksums() {
    local dir="$1"
    local output_file="$2"

    log_info "Generating checksums for: $dir"

    if [ ! -d "$dir" ]; then
        log_error "Directory not found: $dir"
        return 1
    fi

    # Find all data files and generate SHA256 checksums with relative paths
    cd "$dir"
    find . -type f \( \
        -name "*.db" -o \
        -name "*.duckdb" -o \
        -name "*.parquet" -o \
        -name "*.yaml" -o \
        -name "*.yml" -o \
        -name "*.json" -o \
        -name "*.csv" \
    \) -printf "./%f\n" | xargs -I{} sha256sum {} | sort > "$output_file"

    local count=$(wc -l < "$output_file")
    log_success "Generated $count checksums -> $output_file"
}

# =============================================================================
# Verify Checksums
# =============================================================================

verify_checksums() {
    local checksum_file="$1"
    local dir="$2"

    log_info "Verifying checksums for: $dir"

    if [ ! -f "$checksum_file" ]; then
        log_error "Checksum file not found: $checksum_file"
        return 1
    fi

    if [ ! -d "$dir" ]; then
        log_error "Directory not found: $dir"
        return 1
    fi

    # Change to directory for relative path matching
    local original_dir=$(pwd)
    cd "$dir"

    # Generate current checksums with relative paths (./file.db format)
    local temp_checksum=$(mktemp)
    find . -type f \( \
        -name "*.db" -o \
        -name "*.duckdb" -o \
        -name "*.parquet" -o \
        -name "*.yaml" -o \
        -name "*.yml" -o \
        -name "*.json" -o \
        -name "*.csv" \
    \) -printf "./%f\n" | xargs -I{} sha256sum {} | sort > "$temp_checksum"

    cd "$original_dir"

    # Compare checksums directly (both now use ./filename format)
    if diff -q "$checksum_file" "$temp_checksum" > /dev/null 2>&1; then
        rm -f "$temp_checksum"
        log_success "Checksum verification PASSED"
        return 0
    else
        log_error "Checksum verification FAILED"
        echo ""
        echo "Differences found:"
        diff "$checksum_file" "$temp_checksum" || true
        rm -f "$temp_checksum"
        return 1
    fi
}

# =============================================================================
# Compare Two Directories
# =============================================================================

compare_directories() {
    local dir1="$1"
    local dir2="$2"

    log_info "Comparing directories: $dir1 vs $dir2"

    if [ ! -d "$dir1" ]; then
        log_error "Directory not found: $dir1"
        return 1
    fi

    if [ ! -d "$dir2" ]; then
        log_error "Directory not found: $dir2"
        return 1
    fi

    # Generate checksums for both
    local checksum1=$(mktemp)
    local checksum2=$(mktemp)

    generate_checksums "$dir1" "$checksum1"
    generate_checksums "$dir2" "$checksum2"

    # Compare
    if diff -q "$checksum1" "$checksum2" > /dev/null 2>&1; then
        log_success "Directories are identical"
        rm -f "$checksum1" "$checksum2"
        return 0
    else
        log_warn "Directories differ"
        echo ""
        echo "Differences:"
        diff "$checksum1" "$checksum2" || true
        rm -f "$checksum1" "$checksum2"
        return 1
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    local mode="$1"

    case "$mode" in
        --generate)
            if [ $# -ne 3 ]; then
                echo "Usage: $0 --generate <directory> <output_file>"
                exit 1
            fi
            generate_checksums "$2" "$3"
            ;;
        --verify)
            if [ $# -ne 3 ]; then
                echo "Usage: $0 --verify <checksum_file> <directory>"
                exit 1
            fi
            verify_checksums "$2" "$3"
            ;;
        --compare)
            if [ $# -ne 3 ]; then
                echo "Usage: $0 --compare <dir1> <dir2>"
                exit 1
            fi
            compare_directories "$2" "$3"
            ;;
        *)
            echo "Unknown mode: $mode"
            echo ""
            echo "Usage:"
            echo "  $0 --generate <directory> <output_file>   Generate checksums"
            echo "  $0 --verify <checksum_file> <directory>    Verify checksums"
            echo "  $0 --compare <dir1> <dir2>                  Compare two directories"
            exit 1
            ;;
    esac
}

if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  $0 --generate <directory> <output_file>   Generate checksums"
    echo "  $0 --verify <checksum_file> <directory>   Verify checksums"
    echo "  $0 --compare <dir1> <dir2>                 Compare two directories"
    exit 1
fi

main "$@"