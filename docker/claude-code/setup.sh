#!/bin/bash
# Setup script for Claude Code container

# Ensure config directory exists
mkdir -p ~/.claude

# Copy and expand settings if mounted
if [ -f /config/settings.json ]; then
    # Expand environment variables in settings
    envsubst < /config/settings.json > ~/.claude/settings.json
fi

# Copy MCP config if mounted
if [ -f /config/mcp.json ]; then
    cp /config/mcp.json ~/.claude/mcp.json
fi

# Copy hooks if mounted
if [ -d /config/hooks ]; then
    cp -r /config/hooks ~/.claude/
fi

# Copy commands if mounted
if [ -d /config/commands ]; then
    cp -r /config/commands ~/.claude/
fi

# Set permissions
chmod 600 ~/.claude/settings.json 2>/dev/null
chmod 600 ~/.claude/mcp.json 2>/dev/null

# Execute the command
exec "$@"
