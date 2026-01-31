#!/bin/bash
# Test runner script for Docker security tests

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Change to script directory
cd "$(dirname "$0")"

# Check if Docker is available
log_info "Checking Docker availability..."
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    log_info "Start Docker with: sudo systemctl start docker"
    exit 1
fi

log_info "Docker is available: $(docker --version)"

# Check if pytest is installed
log_info "Checking pytest availability..."
if ! command -v pytest &> /dev/null; then
    log_error "pytest is not installed"
    log_info "Install pytest: pip install pytest"
    exit 1
fi

log_info "pytest is available: $(pytest --version)"

# Create minimal mock agent directory structure for testing
log_info "Setting up test environment..."
mkdir -p src/agent
cat > src/agent/__init__.py << 'EOF'
"""Mock agent module for testing"""
EOF

cat > src/agent/main.py << 'EOF'
"""Mock agent main module for testing"""
import sys
import time

def main():
    """Mock agent that keeps running"""
    print("Agent started successfully")
    sys.stdout.flush()

    # Keep running to allow tests to inspect container
    while True:
        time.sleep(60)

if __name__ == "__main__":
    main()
EOF

log_info "Test environment ready"

# Run tests
log_info "Running Docker security tests..."
echo ""

# Run pytest with verbose output
if pytest tests/ -v -s --tb=short; then
    echo ""
    log_info "All tests passed!"
    exit 0
else
    echo ""
    log_error "Some tests failed"
    exit 1
fi
