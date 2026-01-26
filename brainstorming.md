# QUANTMINDX Brainstorming & Decisions Log

> **Last Updated:** 2026-01-23  
> **Project:** QUANTMINDX - Algorithmic Trading Knowledge System

---

## 📋 Project Overview

Building an AI-powered system to:
1. **Ingest** trading knowledge from articles + videos
2. **Extract** strategies without bias
3. **Generate** MQL5 Expert Advisors

---

## 🎯 Key Decisions Made

### 1. Data Sources

| Source | Type | Status | Notes |
|--------|------|--------|-------|
| MQL5.com Articles | Text | ✅ 499/500 scraped | Firecrawl API |
| Trading Videos | Video | 🔧 NPRD tool built | Fabio/ICT content |
| Books/PDFs | Text | ⏳ Future | TBD |

### 2. Model Providers

| Provider | Model | Use Case | Status |
|----------|-------|----------|--------|
| Google | Gemini 2.0 Flash | Video analysis | ⚠️ Quota limit hit |
| ZhipuAI | GLM-4.5V | Video analysis | ✅ Added as alternative |
| OpenAI | GPT-4 | Strategy extraction | ⏳ Future |

**Decision:** Dual model support in NPRD - user selects at runtime.

### 3. NPRD Design Principles

> **DUMB EXTRACTION** - No strategy interpretation!

The NPRD tool is a "Video Transcriptionist" NOT a "Trading Analyst":
- ✅ Transcribes audio
- ✅ Describes visuals objectively  
- ✅ Extracts text from screen (OCR)
- ✅ Records keywords mentioned
- ❌ Does NOT interpret trading strategies
- ❌ Does NOT analyze trade signals

**Why:** Keeps extraction unbiased. A separate "Strategy Extraction Agent" does the thinking later.

---

## 🔗 Resources & Links

### API Documentation
- **Gemini SDK:** https://github.com/googleapis/python-genai
- **Gemini Migration Guide:** https://ai.google.dev/gemini-api/docs/migrate
- **ZAI GLM-4.5V:** https://docs.z.ai/guides/vlm/glm-4.5v
- **ZAI GLM-4.6V:** https://docs.z.ai/guides/vlm/glm-4.6v
- **ZAI Vision MCP:** https://docs.z.ai/devpack/mcp/vision-mcp-server
- **ZAI Chat API:** https://docs.z.ai/api-reference/llm/chat-completion

### API Keys
- **Gemini:** Set `GEMINI_API_KEY` env var
- **ZAI:** Get from https://z.ai/manage-apikey/apikey-list → Set `Z_AI_API_KEY`
- **Firecrawl:** Already configured in project

### Package Installs
```bash
# Scraping
pip install firecrawl-py

# NPRD
pip install yt-dlp ffmpeg-python google-genai click tqdm zai-sdk
```

---

## 📊 MQL5 Article Scraping

### Scraping Stats
- **Total Articles:** 500 (batch 1)
- **Scraped:** 499 ✅
- **Failed:** 1
- **Time:** ~28 minutes
- **Rate:** 3.3s/article

### Categories Scraped
| Category | Count |
|----------|-------|
| trading | 191 |
| trading_systems | 103 |
| integration | 28 |
| expert_advisors | 11 |

### Output Location
```
data/scraped_articles/
├── trading/*.md
├── trading_systems/*.md
├── integration/*.md
└── expert_advisors/*.md
```

### Resume Feature
Added checkpoint system - scraper skips already-scraped articles:
```bash
python scripts/firecrawl_scraper.py --batch-size 500 --start-index 0
```

---

## 🎬 NPRD Video Indexer

### Architecture
```
INPUT                    PROCESSING              OUTPUT
┌────────┐            ┌──────────────┐        ┌────────────┐
│ Video  │ ──────────▶│  Download    │───────▶│  Chunks    │
│ URL    │  yt-dlp    │  Split 45min │ ffmpeg │ (if >45m)  │
└────────┘            └──────────────┘        └────────────┘
                             │
                             ▼
                      ┌──────────────┐        ┌────────────┐
                      │  Analyze     │───────▶│ JSON+MD    │
                      │  Gemini/ZAI  │        │ Timeline   │
                      └──────────────┘        └────────────┘
```

### Files Created
```
tools/
├── nprd_cli.py              # Main CLI
├── README_NPRD.md
└── nprd/
    ├── __init__.py
    ├── video_downloader.py  # yt-dlp
    ├── video_splitter.py    # ffmpeg
    ├── playlist_handler.py
    ├── gemini_client.py     # Gemini 2.0 Flash
    ├── zai_client.py        # GLM-4.5V
    ├── prompts.py           # Librarian prompt
    ├── retry_handler.py     # Exponential backoff
    └── markdown_generator.py
```

### Usage
```bash
# Interactive mode
python tools/nprd_cli.py

# Direct with model selection
python tools/nprd_cli.py "https://youtu.be/VIDEO" --model zai
python tools/nprd_cli.py "PLAYLIST_URL" --playlist --model gemini
```

### Output Format
Agent-optimized JSON with:
- `timeline[]` - 30-60 second semantic clips
- `visual.scene_type` - chart_analysis, slide, intro, discussion
- `ocr_content[]` - All text visible on screen
- `transcript` - Verbatim speech
- `keywords[]` - Mentioned terms for tagging

---

## 🔄 Retry & Recovery

### Retry Logic
- 3 attempts with exponential backoff
- Delays: 2s → 4s → 8s (with jitter)
- Chunk-level recovery (save immediately)

### Resume Capability
```bash
# If interrupted, just run again
python tools/nprd_cli.py "URL" --resume
```
Skips already-processed chunks.

---

## 📝 Playlist/Course Handling

For multi-video courses:
- Creates `playlist.json` manifest
- Tracks each video status
- Supports `--max-videos` limit
- All outputs in organized folders

---

## ⚠️ Issues Encountered

### 1. Gemini API Quota
**Error:** `429 RESOURCE_EXHAUSTED`  
**Solution:** Added ZAI GLM-4.5V as alternative provider

### 2. Deprecated SDK
**Error:** `google.generativeai` deprecated  
**Solution:** Migrated to `google-genai` SDK

### 3. Model Not Found
**Error:** `models/gemini-1.5-pro not found`  
**Solution:** Updated to `gemini-2.0-flash`

### 4. YouTube Download Empty
**Error:** `The downloaded file is empty`  
**Solution:** Updated format selection, added retries

---

## 🚀 Next Steps

1. **Test NPRD with ZAI** - Run video analysis with GLM-4.5V
2. **Continue Scraping** - Start batch 2 (articles 500-999)
3. **Strategy Extraction Agent** - Build downstream processor
4. **MQL5 Code Generator** - Convert strategies to EAs

---

## 💡 Ideas & Notes

- Consider frame sampling rate for Gemini (impacts cost)
- ZAI Vision MCP has 8MB video limit - too small, use API directly
- Google AI Studio free tier sufficient for initial testing
- Token budget: GLM-4.6V supports 32K output, GLM-4.5V supports 16K

---

## 📁 Project Structure

```
/home/mubarkahimself/Desktop/QUANTMINDX/
├── data/
│   ├── exports/engineering_ranked.json  # 1000 articles metadata
│   ├── scraped_articles/                # Markdown articles
│   └── logs/                            # Failure logs
├── outputs/
│   └── videos/                          # NPRD outputs
├── scripts/
│   └── firecrawl_scraper.py            # Article scraper
├── tools/
│   ├── nprd_cli.py                     # Video indexer
│   └── nprd/                           # NPRD modules
└── venv/                               # Python environment
```
