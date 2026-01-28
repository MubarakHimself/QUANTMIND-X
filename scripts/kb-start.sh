#!/bin/bash
# QuantMindX Knowledge Base Starter
# Automatically called by directory hook

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Only start if we're in QuantMindX directory
if [[ "$(basename "$PROJECT_ROOT")" != "QUANTMINDX" ]]; then
    echo "❌ Not in QuantMindX directory"
    exit 1
fi

cd "$PROJECT_ROOT"

# Check if already running
if pgrep -f "quantmindx-kb.*server.py" > /dev/null; then
    echo "✅ Knowledge base already running"
    exit 0
fi

# Start MCP server in background
nohup python3 "$PROJECT_ROOT/mcp-servers/quantmindx-kb/server.py" \
    > "$PROJECT_ROOT/data/logs/mcp-server.log" 2>&1 &

echo "🚀 Knowledge base MCP server started"
