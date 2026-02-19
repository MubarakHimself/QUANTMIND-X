#!/bin/bash
# QuantMindX Knowledge Base Manager
# Uses PageIndex for reasoning-based retrieval
# Only works when run from the QuantMindX directory

set -e

# Check we're in the right directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$(basename "$PROJECT_ROOT")" != "QUANTMINDX" ]]; then
    echo "Error: Must be run from QuantMindX directory"
    echo "   Current: $PROJECT_ROOT"
    exit 1
fi

cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# PageIndex URLs (can be overridden via environment)
PAGEINDEX_ARTICLES_URL="${PAGEINDEX_ARTICLES_URL:-http://localhost:3000}"
PAGEINDEX_BOOKS_URL="${PAGEINDEX_BOOKS_URL:-http://localhost:3001}"
PAGEINDEX_LOGS_URL="${PAGEINDEX_LOGS_URL:-http://localhost:3002}"

case "${1:-help}" in
    index)
        echo -e "${BLUE}Indexing Knowledge Base to PageIndex...${NC}"
        shift
        python3 scripts/index_to_pageindex.py --check-health "$@"
        ;;
    start)
        echo -e "${GREEN}Starting Knowledge Base MCP Server...${NC}"
        ./mcp-servers/quantmindx-kb/start.sh
        ;;
    start-pageindex)
        echo -e "${GREEN}Starting PageIndex services...${NC}"
        docker-compose up -d
        echo ""
        echo "PageIndex services starting..."
        sleep 3
        $0 status
        ;;
    stop-pageindex)
        echo -e "${YELLOW}Stopping PageIndex services...${NC}"
        docker-compose down
        ;;
    status)
        echo -e "${BLUE}Knowledge Base Status (PageIndex):${NC}"
        echo ""
        
        # Check PageIndex services
        echo "PageIndex Services:"
        for service in articles books logs; do
            url_var="PAGEINDEX_${service^^}_URL"
            url="${!url_var}"
            if curl -s -o /dev/null -w "%{http_code}" "$url/health" 2>/dev/null | grep -q "200"; then
                echo -e "  ${GREEN}OK${NC} $service ($url)"
            else
                echo -e "  ${YELLOW}OFFLINE${NC} $service ($url)"
            fi
        done
        
        echo ""
        echo "Data Directories:"
        for dir in scraped_articles knowledge_base/books logs; do
            path="data/$dir"
            if [[ -d "$path" ]]; then
                count=$(find "$path" -type f \( -name "*.md" -o -name "*.txt" -o -name "*.pdf" -o -name "*.json" \) 2>/dev/null | wc -l)
                echo -e "  ${GREEN}$count${NC} files in $path"
            else
                echo -e "  ${YELLOW}NOT FOUND${NC} $path"
            fi
        done
        ;;
    search)
        if [[ -z "${2:-}" ]]; then
            echo "Usage: $0 search '<query>' [collection]"
            echo "Collections: articles (default), books, logs, all"
            exit 1
        fi
        
        query="$2"
        collection="${3:-articles}"
        
        echo -e "${BLUE}Searching $collection for: $query${NC}"
        echo ""
        
        python3 -c "
import httpx
import json
import sys

collection = '$collection'
query = '$query'

if collection == 'all':
    collections = ['articles', 'books', 'logs']
else:
    collections = [collection]

urls = {
    'articles': '$PAGEINDEX_ARTICLES_URL',
    'books': '$PAGEINDEX_BOOKS_URL',
    'logs': '$PAGEINDEX_LOGS_URL',
}

for coll in collections:
    url = urls.get(coll)
    if not url:
        continue
    
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f'{url}/search',
                json={'query': query, 'limit': 5}
            )
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            if results:
                print(f'=== {coll.upper()} ({len(results)} results) ===')
                for i, r in enumerate(results, 1):
                    title = r.get('title', 'Unknown')[:60]
                    score = r.get('score', 0)
                    page = r.get('page', '')
                    source = r.get('source', '')
                    
                    print(f'{i}. {title}')
                    print(f'   Score: {score:.3f}  Page: {page}  Source: {source}')
                    content = r.get('content', '')[:150]
                    print(f'   {content}...')
                    print()
    except Exception as e:
        print(f'Error searching {coll}: {e}', file=sys.stderr)
"
        ;;
    health)
        echo -e "${BLUE}PageIndex Health Check:${NC}"
        echo ""
        for service in articles books logs; do
            url_var="PAGEINDEX_${service^^}_URL"
            url="${!url_var}"
            echo "Checking $service at $url..."
            
            response=$(curl -s "$url/health" 2>/dev/null)
            if [[ $? -eq 0 ]]; then
                echo -e "  ${GREEN}Healthy${NC}"
                echo "  Response: $response"
            else
                echo -e "  ${YELLOW}Unreachable${NC}"
            fi
            echo ""
        done
        ;;
    help|--help|-h)
        cat <<EOF
${GREEN}QuantMindX Knowledge Base Manager${NC}

Uses PageIndex for reasoning-based retrieval with page/section references.

Usage: $0 <command> [args]

Commands:
  index [collection]    Index documents to PageIndex
                        Collections: articles, books, logs, all (default: all)
  start                 Start the MCP server (for Claude Code)
  start-pageindex       Start PageIndex Docker services
  stop-pageindex        Stop PageIndex Docker services
  status                Show knowledge base statistics
  search <query> [col]  Search the knowledge base
                        Collections: articles (default), books, logs, all
  health                Check PageIndex service health

Examples:
  $0 index                     # Index all collections
  $0 index articles            # Index only articles
  $0 start-pageindex           # Start PageIndex services
  $0 status                    # Check status
  $0 search "RSI strategy"     # Search articles for RSI
  $0 search "MACD" books       # Search books for MACD
  $0 search "error" logs       # Search logs for errors
  $0 health                    # Check service health

PageIndex Services:
  - Articles: $PAGEINDEX_ARTICLES_URL
  - Books:    $PAGEINDEX_BOOKS_URL
  - Logs:     $PAGEINDEX_LOGS_URL

Note: Only works from QuantMindX directory
EOF
        ;;
    *)
        echo "Unknown command: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac