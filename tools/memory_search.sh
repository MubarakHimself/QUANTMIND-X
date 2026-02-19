#!/bin/bash
# Memory Search Tool
# Searches claude-mem for memories associated with an agent
# Wraps: claude-mem search --tag "agent:{name}" "{query}"
#
# Usage:
#   echo '{"agent": "analyst", "query": "RSI strategy"}' | ./memory_search.sh
#   ./memory_search.sh "analyst" "RSI strategy"
#
# Output: JSON with search results
# Exit codes: 0 = success, 1 = error

set -euo pipefail

# Configuration
CLAUDE_MEM_PORT="${CLAUDE_MEM_PORT:-37777}"

# Function to output JSON error
error_json() {
    echo "{\"error\": \"$1\", \"results\": []}"
    exit 1
}

# Function to search memories
search_memory() {
    local agent="$1"
    local query="$2"
    
    # Check if claude-mem is available
    if ! command -v claude-mem &> /dev/null; then
        # Fallback to direct API call
        local response
        response=$(curl -s "http://localhost:${CLAUDE_MEM_PORT}/search?tag=agent:${agent}&query=${query}" 2>/dev/null) || {
            # Return empty results if service not available
            echo "{\"results\": [], \"message\": \"Memory service not available\"}"
            return
        }
        echo "$response"
        return
    fi
    
    # Use claude-mem CLI
    claude-mem search --tag "agent:${agent}" "${query}" 2>/dev/null || {
        echo "{\"results\": [], \"message\": \"Search completed with no results\"}"
    }
}

# Main logic
if [ $# -ge 2 ]; then
    agent="$1"
    query="$2"
    search_memory "$agent" "$query"
elif [ ! -t 0 ]; then
    input=$(cat)
    agent=$(echo "$input" | jq -r '.agent // empty')
    query=$(echo "$input" | jq -r '.query // empty')
    
    if [ -z "$agent" ] || [ -z "$query" ]; then
        error_json "Missing 'agent' or 'query' field in input"
    fi
    
    search_memory "$agent" "$query"
else
    error_json "Usage: $0 'agent' 'query' OR echo '{\"agent\": \"...\", \"query\": \"...\"}' | $0"
fi

exit 0