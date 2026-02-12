# QuantMind IDE - Setup Instructions

## Prerequisites
- Node.js 20+ (for SvelteKit)
- Rust (for Tauri)
- Python 3.11+ (backend)

## Phase 1: NPRD Authentication

### Gemini CLI Setup
```bash
# Install Gemini CLI
npm install -g @google/generative-ai-cli

# Authenticate
gemini auth

# OR set API key manually
export GEMINI_API_KEY="your_key_here"
```

### Qwen VL Setup
```bash
# Get API key from: https://dashscope.aliyun.com/
export QWEN_API_KEY="your_key_here"
```

### Test NPRD
```bash
# Run setup script
./scripts/setup_nprd_auth.sh

# Test NPRD processing
python -m src.nprd.cli process test_video.mp4
```

## Phase 2: UI Shell

### Initialize SvelteKit + Tauri
```bash
# Create UI directory
cd quantmind-ide

# Initialize SvelteKit
npm create svelte@latest .
# Choose: Skeleton project, TypeScript, ESLint, Prettier

# Install dependencies
npm install

# Add Tauri
npm install --save-dev @tauri-apps/cli @tauri-apps/api
npx tauri init

# Install UI dependencies
npm install -D tailwindcss@next @tailwindcss/typography
npm install @tanstack/svelte-query
npm install lucide-svelte
npm install monaco-editor
```

### Run Dev Server
```bash
# Start backend (terminal 1)
cd ..
python -m uvicorn src.api.trading_endpoints:create_fastapi_app --reload --port 8000

# Start PageIndex containers (terminal 2)
docker-compose -f docker-compose.pageindex.yml up

# Start Tauri dev (terminal 3)
cd quantmind-ide
npm run tauri dev
```

## Verification

### Backend Health Check
```bash
curl http://localhost:8000/health
```

### PageIndex Check
```bash
curl http://localhost:3000/health  # Articles
curl http://localhost:3001/health  # Books
curl http://localhost:3002/health  # Logs
```

### UI Check
- Tauri window should open
- No console errors
- Dark theme loads correctly
