#!/bin/bash
# QuantMindX One-Click Startup
# Starts all services for local development

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# PID tracking for cleanup
PIDS=()

cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Shutting down services...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       ${GREEN}QuantMindX Local Development${BLUE}          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# Check virtual environment
if [[ ! -d "venv" ]]; then
    echo -e "${RED}❌ Virtual environment not found. Run ./setup.sh first${NC}"
    exit 1
fi

# Ensure logs directory exists
mkdir -p data/logs

# Activate venv
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Start Socket Server
echo -e "${BLUE}Starting Socket Server...${NC}"
python src/router/socket_server.py > data/logs/socket_server.log 2>&1 &
PIDS+=($!)
sleep 1
echo -e "${GREEN}✅ Socket Server started (PID: ${PIDS[-1]})${NC}"

# Start NPRD Server (if exists)
if [[ -f "server/nprd_server.js" ]]; then
    echo -e "${BLUE}Starting NPRD Server...${NC}"
    cd server && node nprd_server.js > ../data/logs/nprd_server.log 2>&1 &
    PIDS+=($!)
    cd ..
    sleep 1
    echo -e "${GREEN}✅ NPRD Server started (PID: ${PIDS[-1]})${NC}"
fi

# Start LangGraph Dev Server
echo -e "${BLUE}Starting LangGraph Dev Server...${NC}"
langgraph dev > /dev/null 2>&1 &
# specific log file causing issues, redirecting to null for now
PIDS+=($!)
sleep 2
echo -e "${GREEN}✅ LangGraph started (PID: ${PIDS[-1]})${NC}"

# Start Backend API Server (FastAPI)
echo -e "${BLUE}Starting Backend API Server...${NC}"
python src/api/server.py > data/logs/backend.log 2>&1 &
PIDS+=($!)
sleep 2
echo -e "${GREEN}✅ Backend API started (PID: ${PIDS[-1]})${NC}"

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All services running!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BLUE}Socket Server:${NC}   tcp://*:5555"
echo -e "  ${BLUE}NPRD Server:${NC}     http://localhost:3000"
echo -e "  ${BLUE}LangGraph API:${NC}   http://localhost:2024"
echo -e "  ${BLUE}Backend API:${NC}     http://localhost:8000"
echo -e "  ${BLUE}LangGraph UI:${NC}    https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024"
echo ""
echo -e "${YELLOW}Starting IDE...${NC}"
echo ""

# Start IDE in foreground
cd "$PROJECT_ROOT/quantmind-ide"
npm run dev

# When IDE exits, cleanup
cleanup
