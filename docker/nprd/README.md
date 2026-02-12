# NPRD Docker Setup

## Quick Start

### 1. One-time Gemini OAuth (on host)
```bash
npm install -g @anthropic-ai/gemini-cli
gemini auth  # Opens browser, login with Google
cp ~/.gemini/* docker/nprd/gemini-credentials/
```

### 2. Start NPRD
```bash
cd docker/nprd
docker-compose up -d
```

### 3. Test
```bash
curl http://localhost:8001/health
```

## How It Works

- **Primary**: Gemini CLI with YOLO mode (headless + bypass permissions)
- **Backup**: Qwen API (cloud) - only if Gemini fails

YOLO mode means Gemini runs without needing to ask for permissions - it's free to process files as directed by our NPRD pipeline.

## Qwen Backup (Optional)

Set `QWEN_API_KEY` in your environment to enable Qwen as backup:
```bash
export QWEN_API_KEY="your_key_here"
docker-compose up -d
```
