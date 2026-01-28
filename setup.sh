#!/bin/bash
# QuantMindX One-Time Setup (Fast)
# Installs all dependencies and initializes the project

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "🚀 QuantMindX Setup"
echo "===================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Step 1: Check Python
echo -e "${BLUE}1. Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "   ✅ Python $PYTHON_VERSION found"

# Step 2: Create virtual environment
echo ""
echo -e "${BLUE}2. Setting up virtual environment...${NC}"
if [[ ! -d "venv" ]]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ✅ Virtual environment exists"
fi

# Activate venv
source venv/bin/activate

# Step 3: Install dependencies (with progress)
echo ""
echo -e "${BLUE}3. Installing Python dependencies...${NC}"
echo "   This may take 5-10 minutes (PyTorch is large)..."
echo ""

# Upgrade pip
pip install --upgrade pip > /dev/null

# Install requirements with progress (remove --quiet)
echo "   Installing base packages..."
pip install qdrant-client tqdm python-dotenv

echo ""
echo "   Installing ML packages (PyTorch + sentence-transformers)..."
pip install sentence-transformers

echo ""
echo "   Installing remaining packages..."
pip install -r requirements.txt || echo "   ⚠️  Some packages may need manual install"

echo "   ✅ Python packages installed"

# Step 4: Create directories
echo ""
echo -e "${BLUE}4. Creating directories...${NC}"
mkdir -p data/{inputs,scraped_articles,qdrant_db,logs}
echo "   ✅ Directories created"

# Step 5: Environment file
echo ""
echo -e "${BLUE}5. Setting up environment...${NC}"
if [[ ! -f ".env" ]]; then
    cat > .env <<EOF
# QuantMindX Environment Variables
FIRECRAWL_API_KEY=your_api_key_here
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
EOF
    echo "   ⚠️  .env file created - add your API keys"
else
    echo "   ✅ .env file exists"
fi

# Step 6: Initialize Qdrant
echo ""
echo -e "${BLUE}6. Initializing Qdrant database...${NC}"
python -c "
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(path='data/qdrant_db')
collections = [c.name for c in client.get_collections().collections]

if 'mql5_knowledge' not in collections:
    client.create_collection(
        collection_name='mql5_knowledge',
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print('   ✅ Qdrant collection created')
else:
    print('   ✅ Qdrant collection exists')
" 2>/dev/null || echo "   ⚠️  Qdrant will initialize on first run"

# Step 7: Make scripts executable
echo ""
echo -e "${BLUE}7. Setting up scripts...${NC}"
chmod +x scripts/*.sh 2>/dev/null || true
echo "   ✅ Scripts executable"

# Done
echo ""
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo "📋 Next Steps:"
echo "   1. (Optional) Add your API keys to .env file"
echo "   2. Index articles: python3 scripts/index_to_qdrant.py"
echo ""
echo "💡 Note: Embedding model downloads on first index (~100MB)"
