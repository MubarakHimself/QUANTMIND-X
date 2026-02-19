#!/bin/bash
#
# QuantMind TUI SSH Setup Script
# 
# This script sets up SSH aliases for easy access to the QuantMind TUI and CLI.
# Run this script on your local machine to configure SSH aliases.
#

set -e

PROJECT_DIR="/home/mubarkahimself/Desktop/QUANTMINDX"

echo "========================================"
echo "QuantMind TUI SSH Setup"
echo "========================================"
echo ""

# Check if .bashrc exists
BASHRC="$HOME/.bashrc"
if [ ! -f "$BASHRC" ]; then
    echo "Creating ~/.bashrc..."
    touch "$BASHRC"
fi

# Add aliases to .bashrc
echo "" >> "$BASHRC"
echo "# QuantMind TUI Aliases" >> "$BASHRC"
echo "alias quantmind-tui='cd $PROJECT_DIR && python -m src.tui.tui_server'" >> "$BASHRC"
echo "alias qm='cd $PROJECT_DIR && python -m src.tui.cli'" >> "$BASHRC"

echo "Aliases added to ~/.bashrc:"
echo "  - quantmind-tui : Launch the Textual TUI"
echo "  - qm            : Run the QuantMind CLI"
echo ""

# Make script executable
chmod +x "$0"

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To use the TUI:"
echo "  1. SSH into your VPS"
echo "  2. Run: quantmind-tui"
echo "  3. Use keyboard shortcuts:"
echo "       q - Quit"
echo "       r - Refresh"
echo "       b - Bots view"
echo "       t - Trades view"
echo "       s - Sync view"
echo "       h - Health view"
echo ""
echo "To use the CLI:"
echo "  1. Run: qm"
echo "  2. Available commands:"
echo "       qm status [trading|contabo]"
echo "       qm bots [list|start|stop|lifecycle]"
echo "       qm sync [status|hmm|data]"
echo "       qm trades [recent|today|bot]"
echo "       qm health [--service api|mt5|database|redis|prometheus]"
echo ""
echo "NOTE: Run 'source ~/.bashrc' or restart your terminal"
echo "      to load the new aliases."
echo ""
