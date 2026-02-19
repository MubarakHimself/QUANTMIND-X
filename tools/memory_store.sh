#!/bin/bash
# Memory Store Tool
# Stores memories to claude-mem associated with an agent
# Wraps: claude-mem add --tag "agent:{name}" "{content}"
#
# Usage:
#   echo '{"agent": "analyst", "content": "RSI strategy worked well in trending markets"}' | ./memory_store.sh
#   ./memory_store.sh "analyst" "RSI strategy worked well in trending markets"
#
# Output: JSON with memory ID
# Exit codes: 0 = success, 1 = error

set -euo pipefail

# Configuration
CLAUDE_MEM_PORT="${CLAUDE_MEM_PORT:-37777}"

# Function to output JSON error
error_json() {
    echo "{\"error\": \"$1\", \"memory_id\": null, \"success\": false}"
    exit 1
}

# Function to store memory
store_memory() {
    local agent="$1"
    local content="$2"
    local metadata="${3:-}"
    
    # Check if claude-mem is available
    if ! command -v claude-mem &> /dev/null; then
        # Fallback to direct API call
        local payload
        payload=$(cat <<EOF
{"tag": "agent:${agent}", "content": "${content}"}
EOF
)
        local response
        response=$(curl -s -X POST \
            "http://localhost:${CLAUDE_MEM_PORT}/memories" \
            -H "Content-Type: application/json" \
            -d "$payload" 2>/dev/null) || {
            error_json "Failed to store memory - service not available"
        }
        echo "$response"
        return
    fi
    
    # Use claude-mem CLI
    local result
    result=$(claude-mem add --tag "agent:${agent}" "${content}" 2>/dev/null) || {
        error_json "Failed to store memory"
    }
    
    # Return success with memory ID
    echo "{\"success\": true, \"memory_id\": \"$result\", \"agent\": \"$agent\"}"
}

# Main logic
if [ $# -ge 2 ]; then
    agent="$1"
    content="$2"
    metadata="${3:-}"
    store_memory "$agent" "$content" "$metadata"
elif [ ! -t 0 ]; then
    input=$(cat)
    agent=$(echo "$input" | jq -r '.agent // empty')
    content=$(echo "$input" | jq -r '.content // empty')
    metadata=$(echo "$input" | jq -r '.metadata // ""')
    
    if [ -z "$agent" ] || [ -z "$content" ]; then
        error_json "Missing 'agent' or 'content' field in input"
    fi
    
    store_memory "$agent" "$content" "$metadata"
else
    error_json "Usage: $0 'agent' 'content' [metadata] OR echo '{\"agent\": \"...\", \"content\": \"...\"}' | $0"
fi

exit 0