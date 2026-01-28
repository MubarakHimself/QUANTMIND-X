# Scraper Usage Guide

## Setup

1. **Get Firecrawl API Key**:
   - Go to https://www.firecrawl.dev/
   - Sign up (free plan: 500 pages/month)
   - Copy your API key

2. **Create .env file**:
   ```bash
   cp .env.example .env
   # Edit .env and add your FIRECRAWL_API_KEY
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Export Articles from Extension

- Open Chrome extension
- Click "Export JSON"
- Save to: `data/exports/quantmindx_articles_YYYY-MM-DD.json`

### 2. Run Scraper

```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
python scripts/scraper.py
```

### 3. Check Output

Markdown files will be saved to: `data/markdown/{category}/article_name.md`

## Multi-Part Articles

The scraper automatically:
- Detects articles that are part of a series (e.g., "Part 1", "Part 2")
- Attempts to find related parts by scanning article links
- Saves series information in the markdown frontmatter

Example output for series:
```markdown
# Building a Trading Strategy - Part 1

**URL**: https://www.mql5.com/en/articles/12345
**Series**: Building a Trading Strategy
**Part**: 1

**Related Parts**:
- https://www.mql5.com/en/articles/12346
- https://www.mql5.com/en/articles/12347

---

[Article content here...]
```

## Programmatic Usage

```python
from scripts.scraper import MQL5Scraper

scraper = MQL5Scraper(api_key="your_firecrawl_key")

# Scrape single article
result = scraper.scrape_with_series_detection(
    url="https://www.mql5.com/en/articles/12345",
    title="Article Title"
)

# Scrape from export file
results = scraper.scrape_from_export("data/exports/articles.json")
```

## Notes

- Firecrawl free plan: 500 pages/month
- Rate limiting handled automatically
- Failed scrapes logged in results
- Markdown saved with metadata for easy processing
