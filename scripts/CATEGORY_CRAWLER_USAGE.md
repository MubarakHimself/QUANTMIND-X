# Category Crawler Usage Guide

## Purpose

Instead of manually saving each article, provide category page URLs and automatically extract all article links.

## Usage

### Option 1: Simple Category Crawl

```python
from scripts.category_crawler import MQL5CategoryCrawler

crawler = MQL5CategoryCrawler()

# Crawl all pages in Trading category
articles = crawler.crawl_category(
    category_url="https://www.mql5.com/en/articles/trading"
)

# Save to JSON
import json
with open('data/exports/trading_articles.json', 'w') as f:
    json.dump(articles, f, indent=2)
```

### Option 2: Specific Page Range

```python
# Only pages 2-5
articles = crawler.crawl_category(
    category_url="https://www.mql5.com/en/articles/trading",
    start_page=2,
    end_page=5
)
```

### Option 3: Multiple Categories at Once

```python
categories = [
    {
        'name': 'Trading Systems',
        'url': 'https://www.mql5.com/en/articles/trading',
        'start_page': 1,
        'end_page': 5
    },
    {
        'name': 'Expert Advisors',
        'url': 'https://www.mql5.com/en/articles/experts',
        'start_page': 1,
        'end_page': 3
    }
]

results = crawler.crawl_multiple_categories(categories)

# Results is a dict: {'Trading Systems': [...], 'Expert Advisors': [...]}
```

## What You Provide

Just tell me:
1. **Category name** (e.g., "Trading Systems")
2. **Category URL** (e.g., "https://www.mql5.com/en/articles/trading")
3. **Page range** (e.g., "pages 1-10" or "all pages")

## Example

"I want all articles from:
- Trading category, pages 1-16
- Expert Advisors category, pages 1-8"

I'll run the crawler and extract all article links for you.

## CLI Usage

```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX

# Crawl Trading category (all pages)
python -c "
from scripts.category_crawler import MQL5CategoryCrawler
import json

crawler = MQL5CategoryCrawler()
articles = crawler.crawl_category('https://www.mql5.com/en/articles/trading')

with open('data/exports/articles.json', 'w') as f:
    json.dump(articles, f, indent=2)

print(f'Saved {len(articles)} articles')
"
```

## Output Format

```json
[
  {
    "url": "https://www.mql5.com/en/articles/12345",
    "title": "Building a Trading Strategy - Part 1",
    "page": 2
  },
  {
    "url": "https://www.mql5.com/en/articles/12346",
    "title": "Advanced Risk Management",
    "page": 2
  }
]
```

This output can be directly fed to the scraper!
