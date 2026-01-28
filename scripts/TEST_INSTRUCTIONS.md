# 🔧 FIXED: Simple Scraper Test Instructions

## Problem Identified:
MQL5.com is blocking Crawlee (Playwright-based) scraping with **HTTP 403**

## Solution:
Created `simple_scraper.py` - uses `requests` + `BeautifulSoup` (more lightweight, less detectable)

---

## Test New Simple Scraper

### Step 1: Test with 1 Article
```bash
cd /home/mubarkahimself/Desktop/QUANTMINDX
source venv/bin/activate
python scripts/simple_scraper.py --batch-size 1 --start-index 0
```

**What to look for:**
- ✅ Should show HTTP 200 (not 403!)
- ✅ File created in `data/scraped_articles/`
- ✅ Markdown conversion looks good

**If still 403:** MQL5 might need more aggressive anti-detection (we can add more tricks)

---

### Step 2: Test with 10 Articles
If Step 1 works:
```bash
python scripts/simple_scraper.py --batch-size 10 --start-index 0
```

**Expected:**
- Prime delays (2-47s) between articles
- Different user agents per request
- Should take ~3-7 minutes

---

### Step 3: Full Batch (300 articles)
If Step 2 works perfectly:
```bash
python scripts/simple_scraper.py --batch-size 300 --start-index 0
```

**Time:** ~60-90 minutes

---

## Why This Should Work Better:

1. **Requests library** - Standard HTTP, not headless browser
2. **Rotating User Agents** - 5 different realistic user agents
3. **Realistic Headers** - Accept, Accept-Language, DNT, etc.
4. **Prime delays** - 2-47s random delays (natural pattern)
5. **Connection pooling** - Session reuse (more natural)
6. **Less resource intensive** - No browser overhead

---

## If Simple Scraper Also Fails (403):

We have 3 backup options:

### Option A: **Use Firecrawl** ($$)
- Handles anti-bot professionally
- Limit: 500 pages/month free

### Option B: **Add rotating proxies** ($)
- Use free proxy list or pay for residential proxies
- Rotate IP per request

### Option C: **Manual + Extension**
- Use the Chrome extension we built
- Manually save articles while browsing
- More time-consuming but 100% reliable

---

## Test This Now:
```bash
python scripts/simple_scraper.py --batch-size 1 --start-index 0
```

Tell me:
1. HTTP status code (200 = good, 403 = blocked again)
2. Did file get created?
3. Quality of markdown output?
