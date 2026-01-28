#!/bin/bash
# QuantMindX Knowledge Base Hook Setup
# Installs directory-aware hooks for auto-starting KB

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUANTMINDX_HOOK="$PROJECT_ROOT/.env.sh"

echo "🔧 QuantMindX Knowledge Base Hook Setup"
echo "========================================"

# Detect shell
if [[ -n "$ZSH_VERSION" ]]; then
    SHELL_CONFIG="$HOME/.zshrc"
    SHELL_NAME="zsh"
elif [[ -n "$BASH_VERSION" ]]; then
    SHELL_CONFIG="$HOME/.bashrc"
    SHELL_NAME="bash"
else
    echo "❌ Unsupported shell"
    exit 1
fi

echo "📁 Detected shell: $SHELL_NAME"
echo "📝 Config file: $SHELL_CONFIG"

# Check if already installed
if grep -q "QuantMindX.*Hook" "$SHELL_CONFIG" 2>/dev/null; then
    echo "⚠️  Hooks already installed in $SHELL_CONFIG"
    echo "   Remove the lines manually to reinstall"
    exit 0
fi

# Add hook to shell config
echo "" >> "$SHELL_CONFIG"
echo "# QuantMindX Knowledge Base Hook" >> "$SHELL_CONFIG"
echo "source \"$QUANTMINDX_HOOK\"" >> "$SHELL_CONFIG"

echo "✅ Hooks installed!"
echo ""
echo "📋 Next steps:"
echo "   1. Reload your shell: source $SHELL_CONFIG"
echo "   2. Or restart your terminal"
echo "   3. cd into QuantMindX directory"
echo "   4. The knowledge base will auto-activate"
echo ""
echo "🔍 To verify: ./scripts/kb.sh status"
