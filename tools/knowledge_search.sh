#!/bin/bash
# Knowledge Base Search Tool
# Searches the PageIndex knowledge base for relevant content
#
# Usage:
#   echo '{"query": "RSI strategy", "namespace": "strategies"}' | ./knowledge_search.sh
#   ./knowledge_search.sh "RSI strategy" "strategies"
#
# Output: JSON with search results
# Exit codes: 0 = success, 1 = error

set -euo pipefail

# Configuration
PAGEINDEX_URL="${PAGEINDEX_BOOKS_URL:-http://localhost:3001}"
MAX_RESULTS="${MAX_RESULTS:-10}"

# Function to output JSON error
error_json() {
    echo "{\"error\": \"$1\", \"results\": []}"
    exit 1
}

# Function to search knowledge base
search_knowledge() {
    local query="$1"
    local namespace="${2:-default}"
    
    # Call PageIndex API
    local response
    response=$(curl -s -X POST \
        "${PAGEINDEX_URL}/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"${query}\", \"namespace\": \"${namespace}\", \"limit\": ${MAX_RESULTS}}" \
        2>/dev/null) || error_json "Failed to connect to PageIndex service"
    
    # Validate response is JSON
    if ! echo "$response" | jq -e . >/dev/null 2>&1; then
        error_json "Invalid response from PageIndex"
    fi
    
    echo "$response"
}

# Main logic
if [ $# -ge 1 ]; then
    # Positional arguments mode
    query="$1"
    namespace="${2:-default}"
    search_knowledge "$query" "$namespace"
elif [ ! -t 0 ]; then
    # Stdin JSON mode
    input=$(cat)
    query=$(echo "$input" | jq -r '.query // empty')
    namespace=$(echo "$input" | jq -r '.namespace // "default"')
    
    if [ -z "$query" ]; then
        error_json "Missing 'query' field in input"
    fi
    
    search_knowledge "$query" "$namespace"
else
    error_json "Usage: $0 'query' [namespace] OR echo '{\"query\": \"...\"}' | $0"
fi

exit 0