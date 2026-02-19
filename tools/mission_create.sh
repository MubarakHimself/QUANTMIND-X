#!/bin/bash
# Mission Create Tool
# Creates a new mission in the QuantMind system database
#
# Usage:
#   echo '{"name": "EURUSD Strategy", "description": "Create RSI strategy"}' | ./mission_create.sh
#   ./mission_create.sh "EURUSD Strategy" "Create RSI strategy"
#
# Output: JSON with mission ID and details
# Exit codes: 0 = success, 1 = error

set -euo pipefail

# Configuration
DB_PATH="${QUANTMIND_DB_PATH:-/app/data/quantmind.db}"

# Function to output JSON error
error_json() {
    echo "{\"error\": \"$1\", \"mission_id\": null, \"success\": false}"
    exit 1
}

# Function to generate UUID
generate_uuid() {
    if command -v uuidgen &> /dev/null; then
        uuidgen | tr '[:upper:]' '[:lower:]'
    else
        # Fallback: generate a simple unique ID
        echo "mission_$(date +%s)_$$"
    fi
}

# Function to create mission
create_mission() {
    local name="$1"
    local description="${2:-}"
    local mission_id
    mission_id=$(generate_uuid)
    
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    # Create mission JSON
    local mission_json
    mission_json=$(cat <<EOF
{
  "mission_id": "$mission_id",
  "name": "$name",
  "description": "$description",
  "status": "created",
  "created_at": "$timestamp",
  "updated_at": "$timestamp",
  "tasks": [],
  "context": {}
}
EOF
)
    
    # Try to store in database if available
    if command -v sqlite3 &> /dev/null && [ -f "$DB_PATH" ]; then
        sqlite3 "$DB_PATH" "INSERT INTO missions (mission_id, name, description, status, created_at) VALUES ('$mission_id', '$name', '$description', 'created', '$timestamp');" 2>/dev/null || {
            # Database not available, return mission anyway
            echo "$mission_json"
            return
        }
    fi
    
    # Return mission JSON
    echo "$mission_json"
}

# Main logic
if [ $# -ge 1 ]; then
    name="$1"
    description="${2:-}"
    create_mission "$name" "$description"
elif [ ! -t 0 ]; then
    input=$(cat)
    name=$(echo "$input" | jq -r '.name // empty')
    description=$(echo "$input" | jq -r '.description // ""')
    
    if [ -z "$name" ]; then
        error_json "Missing 'name' field in input"
    fi
    
    create_mission "$name" "$description"
else
    error_json "Usage: $0 'name' ['description'] OR echo '{\"name\": \"...\", \"description\": \"...\"}' | $0"
fi

exit 0