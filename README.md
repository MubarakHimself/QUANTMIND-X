# QUANTMIND-X

An AI-powered quantitative trading platform with automated strategy extraction, coding, and deployment.

## Project Structure

```
QUANTMINDX/
├── quantmindx-extension/   # Chrome extension for article collection
├── scripts/                # Python scripts for scraping & processing
├── notebooks/              # Jupyter notebooks for exploration
├── data/                   # Data storage
│   ├── raw_articles/       # Raw article HTML (gitignored)
│   ├── markdown/           # Processed markdown files (gitignored)
│   └── exports/            # Exported JSON from extension
├── config/                 # Configuration files
└── requirements.txt        # Python dependencies
```

## Setup

### 1. Install Extension

See `quantmindx-extension/README.md` for installation instructions.

### 2. Install Python Dependencies

```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the project root:

```
FIRECRAWL_API_KEY=your_api_key_here
```

Get your Firecrawl API key: https://www.firecrawl.dev/

## Workflow

1. **Collect Articles**: Use Chrome extension to save MQL5 article URLs
2. **Export**: Click "Export JSON" in extension popup
3. **Categorize**: AI reviews and categorizes articles
4. **Scrape**: Firecrawl extracts content to markdown
5. **Store**: Articles embedded and stored in Qdrant

## Current Status

- [x] Chrome extension built
- [ ] Firecrawl scraper (in progress)
- [ ] Qdrant setup
- [ ] Categorization system
- [ ] NPRD (video processing)
- [ ] Coding agent
- [ ] Code review agent

## Components

### Knowledge Base Scraper
Extracts MQL5 articles using Firecrawl, handles multi-part series automatically.

### Chrome Extension
Browser extension for collecting article URLs with right-click menu.

### Vector Database
Qdrant instance for storing categorized knowledge with semantic search.
