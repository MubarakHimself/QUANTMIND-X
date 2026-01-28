# QuantMindX Knowledge Base Auto-Start Hooks

## Overview

The knowledge base automatically activates when you enter the QuantMindX directory and deactivates when you leave.

## Installation

### One-Time Setup

```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
./scripts/setup-kb-hooks.sh
```

This installs hooks into your shell config (~/.bashrc or ~/.zshrc).

### Reload Shell

```bash
source ~/.bashrc   # or ~/.zshrc
```

## How It Works

1. **Directory Detection**: Shell hook detects when you cd into QuantMindX
2. **Auto-Activation**: KB service activates automatically
3. **Claude Code Integration**: MCP server starts only when in QuantMindX
4. **Auto-Deactivation**: Service stops when you leave the directory

## Usage

```bash
# Enter directory - auto-activates
cd ~/Desktop/QUANTMINDX

# Check status
./scripts/kb.sh status

# Search knowledge base
./scripts/kb.sh search "RSI divergence"

# Leave directory - auto-deactivates
cd ..
```

## Manual Control

```bash
# Always available commands
./scripts/kb.sh index      # Index articles
./scripts/kb.sh status     # Check status
./scripts/kb.sh search     # Search KB
./scripts/kb.sh start      # Start MCP server
```

## Claude Code Integration

When in QuantMindX directory, Claude Code has access to:
- `search_knowledge_base` - Semantic search
- `get_article_content` - Full article retrieval
- `kb_stats` - Database statistics

## Troubleshooting

**Hooks not working:**
```bash
# Verify installation
grep "QuantMindX" ~/.bashrc
# or
grep "QuantMindX" ~/.zshrc

# Reinstall
./scripts/setup-kb-hooks.sh
```

**MCP server not starting:**
```bash
# Check logs
tail -f data/logs/mcp-server.log

# Manual start
./scripts/kb.sh start
```

**Uninstall hooks:**
Edit ~/.bashrc or ~/.zshrc and remove the QuantMindX hook lines.
