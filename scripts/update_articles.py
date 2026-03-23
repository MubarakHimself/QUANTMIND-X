#!/usr/bin/env python3
"""
Incremental Article Updater — Firecrawl

Scrapes unscraped articles from engineering_ranked.json (highest relevance first)
and appends them to data/scraped_articles/. Already-scraped URLs are skipped.

Usage:
    python scripts/update_articles.py                    # scrape next 50 articles
    python scripts/update_articles.py --batch 100        # custom batch size
    python scripts/update_articles.py --sync-contabo     # rsync to server after
    python scripts/update_articles.py --api-key fc-...   # pass key directly
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
RANKED_FILE = ROOT / "data" / "exports" / "engineering_ranked.json"
SCRAPED_DIR = ROOT / "data" / "scraped_articles"
LOGS_DIR = ROOT / "data" / "logs"
SCRAPED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# File that records which URLs we have already scraped (url → filename)
INDEX_FILE = SCRAPED_DIR / "_index.json"

CONTABO_HOST = "root@155.133.27.86"
CONTABO_PATH = "/opt/quantmindx/data/scraped_articles/"
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")


# ─── Index helpers ────────────────────────────────────────────────────────────

def load_index() -> dict:
    """Load the url→filename index, or rebuild it from existing files."""
    if INDEX_FILE.exists():
        return json.loads(INDEX_FILE.read_text())

    # Rebuild: scan existing .md files for their source URL in frontmatter
    index = {}
    for md_file in SCRAPED_DIR.glob("*.md"):
        text = md_file.read_text(errors="ignore")
        for line in text.splitlines()[:10]:
            if line.startswith("source_url:") or line.startswith("url:"):
                url = line.split(":", 1)[1].strip().strip('"').strip("'")
                index[url] = md_file.name
                break
    INDEX_FILE.write_text(json.dumps(index, indent=2))
    return index


def save_index(index: dict) -> None:
    INDEX_FILE.write_text(json.dumps(index, indent=2))


def url_to_filename(url: str) -> str:
    """Stable filename from URL hash."""
    return hashlib.md5(url.encode()).hexdigest() + ".md"


# ─── Core scraper ─────────────────────────────────────────────────────────────

def scrape_batch(articles: list, api_key: str) -> tuple[int, int]:
    """
    Scrape a list of article dicts and save to SCRAPED_DIR.

    Returns:
        (success_count, fail_count)
    """
    try:
        from firecrawl import FirecrawlApp
    except ImportError:
        print("ERROR: firecrawl-py not installed. Run: pip install firecrawl-py")
        sys.exit(1)

    firecrawl = FirecrawlApp(api_key=api_key)
    index = load_index()
    success = 0
    fail = 0
    failed_items = []

    for i, article in enumerate(articles, 1):
        url = article.get("url", "")
        title = article.get("title", "untitled")
        score = article.get("relevance_score", 0)

        if not url:
            fail += 1
            continue

        print(f"[{i}/{len(articles)}] {title[:60]}... (score: {score:.3f})")

        try:
            result = firecrawl.scrape_url(url, params={"formats": ["markdown"]})
            markdown = result.get("markdown", "")

            if not markdown or len(markdown) < 100:
                print(f"  ⚠  Empty response, skipping")
                fail += 1
                failed_items.append({"url": url, "reason": "empty_response"})
                continue

            # Build frontmatter + content
            filename = url_to_filename(url)
            scraped_at = datetime.utcnow().isoformat()
            content = (
                f"---\n"
                f"title: {json.dumps(title)}\n"
                f"source_url: {url}\n"
                f"relevance_score: {score}\n"
                f"scraped_at: {scraped_at}\n"
                f"categories: {json.dumps(article.get('categories', []))}\n"
                f"---\n\n"
                f"{markdown}"
            )

            (SCRAPED_DIR / filename).write_text(content, encoding="utf-8")
            index[url] = filename
            save_index(index)
            success += 1
            print(f"  ✅ Saved → {filename}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            fail += 1
            failed_items.append({"url": url, "reason": str(e)})

        # Respect Firecrawl rate limits (1 req/s on free tier)
        if i < len(articles):
            time.sleep(1.2)

    if failed_items:
        log_path = LOGS_DIR / f"update_failed_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        log_path.write_text(json.dumps(failed_items, indent=2))
        print(f"\nFailed URLs logged → {log_path}")

    return success, fail


def sync_to_contabo() -> None:
    """Rsync scraped articles to Contabo server."""
    print("\nSyncing to Contabo...")
    cmd = [
        "rsync", "-avz", "--progress",
        "-e", f"ssh -i {SSH_KEY} -o StrictHostKeyChecking=no",
        str(SCRAPED_DIR) + "/",
        f"{CONTABO_HOST}:{CONTABO_PATH}",
    ]
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        print("✅ Sync complete")
    else:
        print(f"❌ Sync failed (exit code {result.returncode})")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Incremental Firecrawl article updater")
    parser.add_argument("--batch", type=int, default=50, help="Articles to scrape this run (default: 50)")
    parser.add_argument("--api-key", type=str, default=os.getenv("FIRECRAWL_API_KEY"), help="Firecrawl API key")
    parser.add_argument("--sync-contabo", action="store_true", help="Rsync to Contabo after scraping")
    parser.add_argument("--status", action="store_true", help="Show progress stats and exit")
    args = parser.parse_args()

    # Load ranked article list
    if not RANKED_FILE.exists():
        print(f"ERROR: {RANKED_FILE} not found. Run the category crawler first.")
        sys.exit(1)

    all_articles = json.loads(RANKED_FILE.read_text())
    index = load_index()
    already_scraped = set(index.keys())

    remaining = [a for a in all_articles if a.get("url") not in already_scraped]
    total = len(all_articles)
    done = total - len(remaining)

    print(f"📊 Article progress: {done}/{total} scraped ({done/total*100:.1f}%)")
    print(f"   Remaining: {len(remaining)}")

    if args.status:
        return

    if not remaining:
        print("✅ All articles already scraped!")
        if args.sync_contabo:
            sync_to_contabo()
        return

    # Pick top N by relevance (already sorted in engineering_ranked.json)
    batch = remaining[: args.batch]
    print(f"\n🔥 Scraping next {len(batch)} articles (highest relevance first)...\n")

    if not args.api_key:
        import getpass
        args.api_key = getpass.getpass("Firecrawl API key: ").strip()

    if not args.api_key:
        print("ERROR: No API key provided.")
        sys.exit(1)

    start = time.time()
    success, fail = scrape_batch(batch, args.api_key)
    elapsed = time.time() - start

    print(f"\n{'─'*60}")
    print(f"✅ Scraped: {success}   ❌ Failed: {fail}   ⏱ {elapsed:.0f}s")
    print(f"{'─'*60}")

    if args.sync_contabo:
        sync_to_contabo()
    else:
        print("\nTip: Run with --sync-contabo to upload to server automatically")


if __name__ == "__main__":
    main()
