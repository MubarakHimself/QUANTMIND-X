#!/usr/bin/env python3
"""
Simple MQL5 Article Scraper using requests + BeautifulSoup
More lightweight than Crawlee, less likely to trigger bot detection
"""

import time
import json
import random
from pathlib import Path
from typing import List, Dict
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from markitdown import MarkItDown

# Prime numbers for natural delays (2-47 seconds)
PRIME_DELAYS = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

# Paths
ENGINEERING_RANKED_FILE = Path("data/exports/engineering_ranked.json")
SCRAPED_DIR = Path("data/scraped_articles")
LOGS_DIR = Path("data/logs")

# Create directories
SCRAPED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# User agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:134.0) Gecko/20100101 Firefox/134.0',
]


def get_prime_delay() -> int:
    """Get a random prime number delay."""
    return random.choice(PRIME_DELAYS)


def get_random_user_agent() -> str:
    """Get a random user agent."""
    return random.choice(USER_AGENTS)


def sanitize_filename(title: str, max_length: int = 100) -> str:
    """Convert article title to safe filename."""
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    safe_title = safe_title.replace(' ', '_')
    if len(safe_title) > max_length:
        safe_title = safe_title[:max_length]
    return safe_title.strip('_').lower()


class SimpleMQL5Scraper:
    """Lightweight scraper using requests + BeautifulSoup."""
    
    def __init__(self, batch_size: int = 300, start_index: int = 0):
        self.batch_size = batch_size
        self.start_index = start_index
        self.markdown_converter = MarkItDown()
        self.scraped_count = 0
        self.failed_urls = []
        self.start_time = None
        
        # Session for connection pooling
        self.session = requests.Session()
        
        # Load articles
        with open(ENGINEERING_RANKED_FILE, 'r') as f:
            all_articles = json.load(f)
        
        end_index = start_index + batch_size
        self.articles = all_articles[start_index:end_index]
        
        print(f"📦 Loaded batch: {len(self.articles)} articles")
        print(f"   Range: [{start_index}:{end_index}]")
    
    def scrape_article(self, article: Dict) -> bool:
        """
        Scrape a single article.
        
        Returns:
            True if successful, False otherwise
        """
        url = article['url']
        title = article['title']
        categories = article['categories']
        relevance_score = article['relevance_score']
        
        try:
            print(f"\n📄 Scraping: {title[:60]}...")
            print(f"   Score: {relevance_score} | Categories: {', '.join(categories[:2])}")
            
            # Random user agent
            headers = {
                'User-Agent': get_random_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
            }
            
            # Make request
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            print(f"   ✅ HTTP {response.status_code}")
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract article content
            article_body = soup.find('div', class_='content')
            if not article_body:
                article_body = soup.find('article')
            
            if article_body:
                article_html = str(article_body)
            else:
                print(f"   ⚠️ Could not find article body, using full page")
                article_html = response.text
            
            # Convert to markdown
            temp_html_file = SCRAPED_DIR / f"temp_{sanitize_filename(title)}.html"
            temp_html_file.write_text(article_html, encoding='utf-8')
            
            try:
                result = self.markdown_converter.convert(str(temp_html_file))
                markdown_content = result.text_content
            except Exception as e:
                print(f"   ⚠️ MarkItDown failed: {e}, using text fallback")
                text_soup = BeautifulSoup(article_html, 'html.parser')
                markdown_content = text_soup.get_text(separator='\n', strip=True)
            finally:
                if temp_html_file.exists():
                    temp_html_file.unlink()
            
            # Create metadata header
            metadata_header = f"""---
title: {title}
url: {url}
categories: {', '.join(categories)}
relevance_score: {relevance_score}
scraped_at: {datetime.now().isoformat()}
---

"""
            
            # Combine
            full_markdown = metadata_header + markdown_content
            
            # Save
            filename = sanitize_filename(title) + ".md"
            primary_category = categories[0].replace(' ', '_').lower()
            category_dir = SCRAPED_DIR / primary_category
            category_dir.mkdir(exist_ok=True)
            
            output_file = category_dir / filename
            output_file.write_text(full_markdown, encoding='utf-8')
            
            self.scraped_count += 1
            print(f"   💾 Saved: {output_file.relative_to(SCRAPED_DIR)}")
            print(f"   Progress: {self.scraped_count}/{len(self.articles)}")
            
            return True
            
        except requests.exceptions.HTTPError as e:
            print(f"   ❌ HTTP Error {e.response.status_code}: {url}")
            self.failed_urls.append(url)
            return False
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.failed_urls.append(url)
            return False
    
    def run(self):
        """Run the scraper."""
        self.start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"🚀 STARTING SIMPLE SCRAPER")
        print(f"{'='*70}")
        print(f"Batch size: {self.batch_size}")
        print(f"Start index: {self.start_index}")
        print(f"Prime delays: {min(PRIME_DELAYS)}-{max(PRIME_DELAYS)}s")
        print(f"{'='*70}\n")
        
        for i, article in enumerate(self.articles, 1):
            success = self.scrape_article(article)
            
            # Prime delay between articles (except last one)
            if i < len(self.articles):
                if success:
                    delay = get_prime_delay()
                    print(f"   ⏳ Waiting {delay}s (prime delay)...")
                    time.sleep(delay)
                else:
                    # Shorter delay on failure
                    delay = random.choice([2, 3, 5])
                    print(f"   ⏳ Waiting {delay}s (failure delay)...")
                    time.sleep(delay)
        
        # Print summary
        elapsed_time = time.time() - self.start_time
        elapsed_minutes = elapsed_time / 60
        
        print(f"\n{'='*70}")
        print(f"✨ SCRAPING COMPLETE")
        print(f"{'='*70}")
        print(f"✅ Successfully scraped: {self.scraped_count}/{len(self.articles)}")
        print(f"❌ Failed: {len(self.failed_urls)}")
        print(f"⏱️  Time elapsed: {elapsed_minutes:.2f} minutes")
        if self.scraped_count > 0:
            print(f"📊 Average time per article: {elapsed_time/self.scraped_count:.1f}s")
        print(f"💾 Saved to: {SCRAPED_DIR}")
        print(f"{'='*70}\n")
        
        # Save failed URLs
        if self.failed_urls:
            failed_log = LOGS_DIR / f"failed_batch_{self.start_index}.json"
            failed_log.write_text(json.dumps(self.failed_urls, indent=2))
            print(f"❌ Failed URLs saved to: {failed_log}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape MQL5 articles (simple version)')
    parser.add_argument('--batch-size', type=int, default=300, help='Number of articles')
    parser.add_argument('--start-index', type=int, default=0, help='Starting index')
    
    args = parser.parse_args()
    
    scraper = SimpleMQL5Scraper(
        batch_size=args.batch_size,
        start_index=args.start_index
    )
    
    scraper.run()


if __name__ == "__main__":
    main()
