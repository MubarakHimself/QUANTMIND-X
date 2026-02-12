#!/bin/bash
# NPRD Docker Entrypoint
# Handles OAuth authentication check on startup

set -e

echo "🚀 Starting NPRD Processor..."

# Check if Gemini credentials exist
if [ -f "/root/.gemini/credentials.json" ]; then
    echo "✅ Gemini OAuth credentials found"
else
    echo "⚠️  Gemini OAuth credentials not found!"
    echo ""
    echo "To authenticate, run this ONCE on your host machine:"
    echo "  gemini auth"
    echo ""
    echo "Then copy credentials to container volume:"
    echo "  cp ~/.gemini/credentials.json ./gemini-credentials/"
    echo ""
    echo "Or run interactive auth inside container:"
    echo "  docker exec -it quantmind-nprd gemini auth"
    echo ""
fi

# Check for Qwen credentials (if using cloud API)
if [ -n "$QWEN_API_KEY" ]; then
    echo "✅ Qwen API key configured"
elif [ -f "/root/.qwen/credentials.json" ]; then
    echo "✅ Qwen OAuth credentials found"
else
    echo "ℹ️  Qwen not configured (optional - using Gemini)"
fi

echo ""
echo "📁 Data directory: /app/data"
echo "📁 Gemini config: /root/.gemini"
echo ""

# Execute the main command
exec "$@"
