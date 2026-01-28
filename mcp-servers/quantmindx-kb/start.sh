#!/bin/bash
# QuantMindX Knowledge Base MCP Server Launcher
# Only works when run from the QuantMindX directory

# Check we're in the QuantMindX directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "$(basename "$PROJECT_ROOT")" != "QUANTMINDX" ]]; then
    echo "❌ Error: Must be run from QuantMindX directory"
    echo "   Current: $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT"

# Check dependencies and install if needed
python3 -c "import chromadb" 2>/dev/null || {
    echo "⏳ Installing dependencies..."
    TMPDIR="$PROJECT_ROOT/tmp" pip3 install --user chromadb mcp
}

# Start the MCP server
echo "🚀 Starting QuantMindX Knowledge Base MCP Server (ChromaDB)..."
python3 "$PROJECT_ROOT/mcp-servers/quantmindx-kb/server_chroma.py"
