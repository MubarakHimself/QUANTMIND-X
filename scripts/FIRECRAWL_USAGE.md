# Firecrawl MQL5 Scraper Usage

## Setup

Make sure `firecrawl-py` is installed:
```bash
pip show firecrawl-py
```

If not installed:
```bash
pip install firecrawl-py
```

---

## Usage

### Basic: Scraper prompts for API key
```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
source venv/bin/activate
python scripts/firecrawl_scraper.py --batch-size 500 --start-index 0
```

**You'll see:**
```
🔑 Enter your Firecrawl API key:
API Key: [paste your key]
```

The key input is hidden for security (uses `getpass`).

---

### Alternative: Provide API key as argument
```bash
python scripts/firecrawl_scraper.py --batch-size 500 --start-index 0 --api-key "fc-YOUR_KEY_HERE"
```

> ⚠️ **Not recommended** - exposes key in terminal history

---

## Batch Strategy (1,818 articles with 1,000 page limit)

### Account 1: First 500 articles
```bash
python scripts/firecrawl_scraper.py --batch-size 500 --start-index 0
```
**Enter API key when prompted**

### Account 2: Next 500 articles (501-1000)
```bash
python scripts/firecrawl_scraper.py --batch-size 500 --start-index 500
```
**Enter second API key when prompted**

### Account 3: Remaining 818 articles (1001-1818)
```bash
python scripts/firecrawl_scraper.py --batch-size 818 --start-index 1000
```
**Enter third API key when prompted**

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--batch-size` | 500 | Number of articles to scrape |
| `--start-index` | 0 | Starting position in ranked list |
| `--api-key` | None | Optional: Firecrawl API key |

---

## Output

### File Structure
```
data/scraped_articles/
├── trading_systems/
│   ├── article_1.md
│   ├── article_2.md
│   └── ...
├── trading/
├── expert_advisors/
├── integration/
├── indicators/
└── machine_learning/
```

### Markdown Format
```markdown
---
title: Article Title
url: https://www.mql5.com/en/articles/xxxxx
categories: Category 1, Category 2
relevance_score: 15
scraped_at: 2026-01-22T16:15:00
---

Clean markdown content from Firecrawl...
```

---

## Timing Estimates

- **500 articles:** ~15-25 minutes
- **1,818 articles (3 batches):** ~60-90 minutes total

Firecrawl handles rate limiting automatically, so it's faster than our manual delays.

---

## Failed Articles

If any articles fail, they're logged to:
```
data/logs/failed_firecrawl_batch_{start_index}.json
```

You can retry failed articles by creating a custom retry script if needed.

---

## Tips

1. **Keep terminal open** - Scraping takes time
2. **Don't interrupt** - Let it finish the batch
3. **Check output** - Validate markdown quality after first batch
4. **Monitor credits** - Firecrawl shows credit usage at end

---

## Example Session

```bash
$ python scripts/firecrawl_scraper.py --batch-size 500 --start-index 0

🔑 Enter your Firecrawl API key:
API Key: ●●●●●●●●●●●●●●●●

✅ Firecrawl initialized successfully
📦 Loaded batch: 500 articles
   Range: [0:500]

======================================================================
🚀 STARTING FIRECRAWL SCRAPER
======================================================================
Batch size: 500
Start index: 0
======================================================================

📄 Scraping: Implementing a Rapid-Fire Trading Strategy...
   Score: 18 | Categories: Trading Systems, Indicators
   ✅ Saved: trading_systems/implementing_rapid_fire_trading.md
   Progress: 1/500

...

======================================================================
✨ SCRAPING COMPLETE
======================================================================
✅ Successfully scraped: 500/500
❌ Failed: 0
⏱️  Time elapsed: 18.50 minutes
📊 Average time per article: 2.2s
💾 Saved to: data/scraped_articles
🔥 Firecrawl credits used: ~500
======================================================================
```

---

## Ready to Start!

Run your first batch:
```bash
python scripts/firecrawl_scraper.py --batch-size 500 --start-index 0
```
