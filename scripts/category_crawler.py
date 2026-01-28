"""
MQL5 Category Crawler
Extracts all article links from MQL5 category pages
"""

import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse, parse_qs
import time
from tqdm import tqdm

class MQL5CategoryCrawler:
    """Crawls MQL5 category pages to extract article links"""
    
    BASE_URL = "https://www.mql5.com"
    
    def __init__(self, delay: float = 1.0):
        """
        Args:
            delay: Seconds to wait between requests (be polite)
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })
    
    def extract_articles_from_page(self, url: str) -> List[Dict]:
        """
        Extract all article links from a single category page
        Returns list of {url, title}
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = []
            
            # Find all article links
            # MQL5 uses <div class="item"> for articles
            article_items = soup.find_all('div', class_='item')
            
            if not article_items:
                # Try alternative selectors
                article_items = soup.find_all('a', href=re.compile(r'/en/articles/\d+'))
            
            for item in article_items:
                # Find the article link
                link = item.find('a', href=re.compile(r'/en/articles/\d+'))
                
                if link:
                    article_url = urljoin(self.BASE_URL, link['href'])
                    title = link.get_text(strip=True)
                    
                    # Clean up title
                    title = re.sub(r'\s+', ' ', title)
                    
                    articles.append({
                        'url': article_url,
                        'title': title
                    })
            
            # Remove duplicates
            seen = set()
            unique_articles = []
            for article in articles:
                if article['url'] not in seen:
                    seen.add(article['url'])
                    unique_articles.append(article)
            
            return unique_articles
            
        except Exception as e:
            print(f"Error extracting from {url}: {e}")
            return []
    
    def get_total_pages(self, category_url: str) -> int:
        """
        Determine total number of pages in a category
        """
        try:
            response = self.session.get(category_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for pagination info
            # MQL5 typically shows "Page X of Y"
            pagination = soup.find('div', class_='pagination')
            if pagination:
                text = pagination.get_text()
                match = re.search(r'of\s+(\d+)', text)
                if match:
                    return int(match.group(1))
            
            # Alternative: count page number links
            page_links = soup.find_all('a', href=re.compile(r'page=\d+'))
            if page_links:
                page_numbers = []
                for link in page_links:
                    match = re.search(r'page=(\d+)', link['href'])
                    if match:
                        page_numbers.append(int(match.group(1)))
                if page_numbers:
                    return max(page_numbers)
            
            return 1  # Default to single page
            
        except Exception as e:
            print(f"Error getting page count: {e}")
            return 1
    
    def build_page_url(self, base_url: str, page_num: int) -> str:
        """Build URL for specific page number"""
        parsed = urlparse(base_url)
        
        if 'page=' in base_url:
            # Replace existing page number
            return re.sub(r'page=\d+', f'page={page_num}', base_url)
        else:
            # Add page parameter
            separator = '&' if '?' in base_url else '?'
            return f"{base_url}{separator}page={page_num}"
    
    def crawl_category(self, 
                      category_url: str, 
                      start_page: int = 1, 
                      end_page: Optional[int] = None,
                      auto_detect_pages: bool = True) -> List[Dict]:
        """
        Crawl all pages in a category
        
        Args:
            category_url: Base category URL
            start_page: Page to start from (default: 1)
            end_page: Page to end at (default: None = auto-detect)
            auto_detect_pages: Automatically detect total pages
        
        Returns:
            List of {url, title, page}
        """
        # Auto-detect total pages if needed
        if auto_detect_pages and end_page is None:
            print("Detecting total pages...")
            total_pages = self.get_total_pages(category_url)
            end_page = total_pages
            print(f"Found {total_pages} pages")
        
        if end_page is None:
            end_page = start_page
        
        all_articles = []
        
        for page_num in tqdm(range(start_page, end_page + 1), desc="Crawling pages"):
            page_url = self.build_page_url(category_url, page_num)
            
            articles = self.extract_articles_from_page(page_url)
            
            # Add page number to each article
            for article in articles:
                article['page'] = page_num
            
            all_articles.extend(articles)
            
            # Be polite - wait between requests
            if page_num < end_page:
                time.sleep(self.delay)
        
        print(f"\nFound {len(all_articles)} total articles")
        return all_articles
    
    def crawl_multiple_categories(self, categories: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Crawl multiple categories
        
        Args:
            categories: List of {
                'name': str,
                'url': str,
                'start_page': int (optional),
                'end_page': int (optional)
            }
        
        Returns:
            Dict mapping category name to list of articles
        """
        results = {}
        
        for category in categories:
            name = category['name']
            url = category['url']
            start = category.get('start_page', 1)
            end = category.get('end_page')
            
            print(f"\n{'='*60}")
            print(f"Crawling: {name}")
            print(f"{'='*60}")
            
            articles = self.crawl_category(url, start, end)
            results[name] = articles
        
        return results


def main():
    """Example usage"""
    crawler = MQL5CategoryCrawler(delay=1.0)
    
    # Example: Crawl Trading category
    articles = crawler.crawl_category(
        category_url="https://www.mql5.com/en/articles/trading",
        start_page=1,
        end_page=3  # First 3 pages only
    )
    
    print(f"\nExtracted {len(articles)} articles")
    
    # Print first few
    for article in articles[:5]:
        print(f"  - {article['title']}")
        print(f"    {article['url']}")


if __name__ == "__main__":
    main()
