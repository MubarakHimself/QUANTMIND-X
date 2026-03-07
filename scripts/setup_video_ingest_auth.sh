#!/bin/bash
# Setup script for NPRD authentication

echo "🔧 Setting up NPRD Authentication..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.nprd.example .env
fi

# Gemini CLI Setup
echo ""
echo "📝 Gemini CLI Setup:"
echo "1. Install Gemini CLI: npm install -g @google/generative-ai-cli"
echo "2. Authenticate: gemini auth"
echo "   OR set GEMINI_API_KEY in .env file"

# Check if Gemini CLI is installed
if command -v gemini &> /dev/null; then
    echo "✅ Gemini CLI is installed"
else
    echo "❌ Gemini CLI not found - install with: npm install -g @google/generative-ai-cli"
fi

# Qwen VL Setup
echo ""
echo "📝 Qwen VL Setup:"
echo "1. Get API key from https://dashscope.aliyun.com/"
echo "2. Set QWEN_API_KEY in .env file"
echo "   OR use local model (set QWEN_MODEL_PATH)"

# Test authentication
echo ""
echo "🧪 Testing authentication..."

# Test Gemini
if [ ! -z "$GEMINI_API_KEY" ]; then
    echo "✅ GEMINI_API_KEY is set"
else
    echo "⚠️  GEMINI_API_KEY not set - run 'gemini auth' or add to .env"
fi

# Test Qwen
if [ ! -z "$QWEN_API_KEY" ]; then
    echo "✅ QWEN_API_KEY is set"
else
    echo "⚠️  QWEN_API_KEY not set - add to .env if using cloud API"
fi

echo ""
echo "✅ NPRD setup complete!"
echo "Edit .env file to configure your API keys"
