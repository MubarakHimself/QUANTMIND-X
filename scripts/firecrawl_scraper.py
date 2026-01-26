#!/usr/bin/env python3
"""
MQL5 Article Scraper using Firecrawl
Prompts for API key at runtime
"""

import json
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import getpass

from firecrawl import FirecrawlApp

# Paths
ENGINEERING_RANKED_FILE = Path("data/exports/engineering_ranked.json")
SCRAPED_DIR = Path("data/scraped_articles")
LOGS_DIR = Path("data/logs")

# Create directories
SCRAPED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_filename(title: str, max_length: int = 100) -> str:
    """Convert article title to safe filename."""
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    safe_title = safe_title.replace(' ', '_')
    if len(safe_title) > max_length:
        safe_title = safe_title[:max_length]
    return safe_title.strip('_').lower()


class FirecrawlMQL5Scraper:
    """Firecrawl-based scraper for MQL5 articles."""
    
    def __init__(self, api_key: str, batch_size: int = 500, start_index: int = 0):
        """
        Initialize scraper.
        
        Args:
            api_key: Firecrawl API key
            batch_size: Number of articles to scrape (max 500 for free tier)
            start_index: Starting index in ranked list
        """
        self.api_key = api_key
        self.batch_size = batch_size
        self.start_index = start_index
        self.scraped_count = 0
        self.failed_urls = []
        self.start_time = None
        
        # Initialize Firecrawl
        try:
            self.firecrawl = FirecrawlApp(api_key=api_key)
            print("✅ Firecrawl initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Firecrawl: {e}")
            raise
        
        # Load articles
        with open(ENGINEERING_RANKED_FILE, 'r') as f:
            all_articles = json.load(f)
        
        end_index = start_index + batch_size
        self.articles = all_articles[start_index:end_index]
        
        print(f"📦 Loaded batch: {len(self.articles)} articles")
        print(f"   Range: [{start_index}:{end_index}]")
    
    def is_already_scraped(self, article: Dict) -> bool:
        """Check if article already exists."""
        title = article['title']
        categories = article['categories']
        
        filename = sanitize_filename(title) + ".md"
        primary_category = categories[0].replace(' ', '_').lower()
        category_dir = SCRAPED_DIR / primary_category
        output_file = category_dir / filename
        
        return output_file.exists()
    
    def scrape_article(self, article: Dict) -> bool:
        """
        Scrape a single article using Firecrawl.
        
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
            print(f"   URL: {url}")
            
            # Retry logic for rate limits
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # Scrape with Firecrawl (v4 SDK)
                    result = self.firecrawl.scrape(
                        url,
                        formats=['markdown']
                    )
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    error_msg = str(e)
                    
                    # Check if it's a rate limit error
                    if 'rate limit' in error_msg.lower():
                        retry_count += 1
                        if retry_count < max_retries:
                            # Extract wait time from error message if available
                            wait_time = 10  # Default 10 seconds
                            if 'retry after' in error_msg.lower():
                                # Try to extract seconds from error message
                                import re
                                match = re.search(r'retry after (\d+)s', error_msg)
                                if match:
                                    wait_time = int(match.group(1)) + 2  # Add 2 seconds buffer
                            
                            print(f"   ⏳ Rate limit hit, waiting {wait_time}s (retry {retry_count}/{max_retries})...")
                            time.sleep(wait_time)
                        else:
                            print(f"   ❌ Rate limit error after {max_retries} retries")
                            raise
                    else:
                        # Not a rate limit error, raise immediately
                        raise
            
            # Extract markdown content - Firecrawl v4 returns different structure
            markdown_content = None
            
            # Try different response structures
            if isinstance(result, dict):
                # Try direct markdown field
                if 'markdown' in result:
                    markdown_content = result['markdown']
                # Try data.markdown
                elif 'data' in result and isinstance(result['data'], dict):
                    markdown_content = result['data'].get('markdown')
                # Try content field
                elif 'content' in result:
                    markdown_content = result['content']
            elif hasattr(result, 'markdown'):
                markdown_content = result.markdown
            elif hasattr(result, 'data') and hasattr(result.data, 'markdown'):
                markdown_content = result.data.markdown
            
            if not markdown_content:
                print(f"   ⚠️  No markdown content in response, using full result")
                markdown_content = str(result)
            
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
            
            # Save to file
            filename = sanitize_filename(title) + ".md"
            primary_category = categories[0].replace(' ', '_').lower()
            category_dir = SCRAPED_DIR / primary_category
            category_dir.mkdir(exist_ok=True)
            
            output_file = category_dir / filename
            output_file.write_text(full_markdown, encoding='utf-8')
            
            self.scraped_count += 1
            print(f"   ✅ Saved: {output_file.relative_to(SCRAPED_DIR)}")
            print(f"   Progress: {self.scraped_count}/{len(self.articles)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.failed_urls.append(url)
            return False
    
    def run(self):
        """Run the scraper."""
        self.start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"🚀 STARTING FIRECRAWL SCRAPER")
        print(f"{'='*70}")
        print(f"Batch size: {self.batch_size}")
        print(f"Start index: {self.start_index}")
        print(f"{'='*70}\n")
        
        for i, article in enumerate(self.articles, 1):
            # Check if already scraped (resume functionality)
            if self.is_already_scraped(article):
                print(f"\n⏭️  Skipping (already scraped): {article['title'][:60]}...")
                self.scraped_count += 1
                continue
            
            success = self.scrape_article(article)
            
            # Rate limit: 10 requests/min = 1 request every 6 seconds
            # Use 7 seconds to be safe
            if i < len(self.articles):
                time.sleep(7)
        
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
        print(f"🔥 Firecrawl credits used: ~{self.scraped_count}")
        print(f"{'='*70}\n")
        
        # Save failed URLs
        if self.failed_urls:
            failed_log = LOGS_DIR / f"failed_firecrawl_batch_{self.start_index}.json"
            failed_log.write_text(json.dumps(self.failed_urls, indent=2))
            print(f"❌ Failed URLs saved to: {failed_log}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape MQL5 articles with Firecrawl')
    parser.add_argument('--batch-size', type=int, default=500, help='Number of articles (max 500)')
    parser.add_argument('--start-index', type=int, default=0, help='Starting index')
    parser.add_argument('--api-key', type=str, help='Firecrawl API key (if not provided, will prompt)')
    
    args = parser.parse_args()
    
    # Get API key
    if args.api_key:
        api_key = args.api_key
    else:
        print("\n🔑 Enter your Firecrawl API key:")
        api_key = getpass.getpass("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    # Validate batch size
    if args.batch_size > 500:
        print(f"⚠️  Warning: Firecrawl free tier limit is 500 pages/month")
        print(f"   Requested: {args.batch_size}")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting.")
            return
    
    scraper = FirecrawlMQL5Scraper(
        api_key=api_key,
        batch_size=args.batch_size,
        start_index=args.start_index
    )
    
    scraper.run()


if __name__ == "__main__":
    main()
