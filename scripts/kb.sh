#!/bin/bash
# QuantMindX Knowledge Base Manager
# Only works when run from the QuantMindX directory

set -e

# Check we're in the right directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(basename "$PROJECT_ROOT")" != "QUANTMINDX" ]]; then
    echo "❌ Error: Must be run from QuantMindX directory"
    echo "   Current: $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

case "${1:-help}" in
    index)
        echo -e "${BLUE}📚 Indexing Knowledge Base...${NC}"
        python3 scripts/index_to_qdrant.py
        ;;
    start)
        echo -e "${GREEN}🚀 Starting Knowledge Base MCP Server...${NC}"
        ./mcp-servers/quantmindx-kb/start.sh
        ;;
    status)
        echo -e "${BLUE}📊 Knowledge Base Status:${NC}"
        python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient(path='data/qdrant_db')
collections = client.get_collections()
for c in collections.collections:
    info = client.get_collection(c.name)
    print(f'  {c.name}: {info.points_count} vectors')
"
        ;;
    search)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: $0 search '<query>'"
            exit 1
        fi
        echo -e "${BLUE}🔍 Searching: $2${NC}"
        python3 -c "
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(path='data/qdrant_db')
model = SentenceTransformer('all-MiniLM-L6-v2')
query_vector = model.encode('$2').tolist()

results = client.search(
    collection_name='mql5_knowledge',
    query_vector=query_vector,
    limit=5
)

for i, r in enumerate(results, 1):
    print(f'{i}. {r.payload[\"title\"][:60]}...')
    print(f'   Score: {r.score:.3f}')
    print(f'   {r.payload[\"text\"][:100]}...')
    print()
"
        ;;
    help|--help|-h)
        cat <<EOF
${GREEN}QuantMindX Knowledge Base Manager${NC}

Usage: $0 <command> [args]

Commands:
  index      Index all scraped articles to Qdrant
  start      Start the MCP server (for Claude Code)
  status     Show knowledge base statistics
  search     Search the knowledge base

Examples:
  $0 index              # Index articles
  $0 status             # Check status
  $0 search "RSI"       # Search for RSI strategies

Note: Only works from QuantMindX directory
EOF
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
