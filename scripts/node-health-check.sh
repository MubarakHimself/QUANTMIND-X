#!/bin/bash
# =============================================================================
# Node Health Check Script
#
# Checks health of a node by calling /api/health endpoint.
# Returns exit code 0 if healthy, non-zero otherwise.
#
# Usage: ./node-health-check.sh <node_url>
# Example: ./node-health-check.sh https://contabo.quantmindx.com
# =============================================================================

set -e

# Configuration
TIMEOUT=30
EXPECTED_STATUS="healthy"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    echo "Usage: $0 <node_url> [timeout_seconds]"
    echo "Example: $0 https://contabo.quantmindx.com 30"
    exit 1
}

# Function to check health
check_health() {
    local url="$1"
    local timeout="${2:-$TIMEOUT}"

    echo "Checking health of: $url"

    # Make request and capture response
    local response
    local http_code

    response=$(curl -s -w "\n%{http_code}" \
        --max-time "$timeout" \
        "$url/api/health" 2>/dev/null) || {
        echo -e "${RED}ERROR: Failed to connect to $url${NC}"
        return 1
    }

    # Extract HTTP code
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')

    # Check HTTP status
    if [ "$http_code" != "200" ]; then
        echo -e "${RED}ERROR: HTTP $http_code${NC}"
        echo "Response: $body"
        return 1
    fi

    # Parse JSON response
    local status
    status=$(echo "$body" | grep -o '"status"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
    local version
    version=$(echo "$body" | grep -o '"version"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
    local uptime
    uptime=$(echo "$body" | grep -o '"uptime_seconds"[[:space:]]*:[[:space:]]*[0-9]*' | grep -o '[0-9]*')

    # Display results
    echo -e "Status: ${GREEN}$status${NC}"
    [ -n "$version" ] && echo "Version: $version"
    [ -n "$uptime" ] && echo "Uptime: ${uptime}s"

    # Check if healthy
    if [ "$status" = "$EXPECTED_STATUS" ]; then
        echo -e "${GREEN}Health check PASSED${NC}"
        return 0
    elif [ "$status" = "degraded" ]; then
        echo -e "${YELLOW}WARNING: Node is degraded${NC}"
        return 0  # Still consider degraded as passing for updates
    else
        echo -e "${RED}ERROR: Node is unhealthy${NC}"
        return 1
    fi
}

# Main
main() {
    local url="${1:-}"
    local timeout="${2:-$TIMEOUT}"

    if [ -z "$url" ]; then
        usage
    fi

    check_health "$url" "$timeout"
}

main "$@"