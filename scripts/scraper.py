"""
MQL5 Article Scraper using Firecrawl
Handles multi-part article series automatically
"""

import os
import json
import re
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from tqdm import tqdm

# Load environment variables
load_dotenv()

class MQL5Scraper:
    """Scrapes MQL5 articles using Firecrawl with multi-part detection"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        if not self.api_key:
            raise ValueError("Firecrawl API key required. Set FIRECRAWL_API_KEY in .env")
        
        self.firecrawl = FirecrawlApp(api_key=self.api_key)
        self.output_dir = Path("data/markdown")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_series(self, title: str, url: str) -> Dict:
        """
        Detect if article is part of a series
        Returns: {
            'is_series': bool,
            'part_number': int or None,
            'series_name': str or None
        }
        """
        # Patterns for detecting series
        patterns = [
            r'[Pp]art\s+(\d+)',           # Part 1, part 2
            r'\((\d+)/\d+\)',             # (1/5), (2/5)
            r'[Cc]hapter\s+(\d+)',        # Chapter 1
            r'Episode\s+(\d+)',           # Episode 1
            r'\[(\d+)\]',                 # [1], [2]
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title)
            if match:
                part_num = int(match.group(1))
                # Extract series name (remove part number)
                series_name = re.sub(pattern, '', title).strip()
                series_name = re.sub(r'\s+', ' ', series_name)  # Clean whitespace
                
                return {
                    'is_series': True,
                    'part_number': part_num,
                    'series_name': series_name
                }
        
        return {
            'is_series': False,
            'part_number': None,
            'series_name': None
        }
    
    def find_series_parts(self, url: str, markdown_content: str, series_name: str, current_part: int) -> List[str]:
        """
        Attempt to find other parts of the series
        Looks for links in the article content
        """
        # Extract article ID from URL
        article_id_match = re.search(r'/articles/(\d+)', url)
        if not article_id_match:
            return []
        
        current_id = int(article_id_match.group(1))
        
        # Look for sequential article IDs in content (simple heuristic)
        found_urls = []
        
        # Search for common patterns indicating next/previous parts
        link_patterns = [
            r'\[Part\s+(\d+)\]\((https://www\.mql5\.com/en/articles/\d+)\)',  # Markdown links
            r'https://www\.mql5\.com/en/articles/(\d+)',  # Direct URLs
        ]
        
        for pattern in link_patterns:
            matches = re.finditer(pattern, markdown_content)
            for match in matches:
                if 'Part' in pattern:
                    found_urls.append(match.group(2))
                else:
                    found_urls.append(f"https://www.mql5.com/en/articles/{match.group(1)}")
        
        # Remove duplicates and current URL
        found_urls = list(set(found_urls))
        found_urls = [u for u in found_urls if u != url]
        
        return found_urls
    
    def scrape_article(self, url: str, title: str = "") -> Dict:
        """
        Scrape a single article using Firecrawl
        Returns: {
            'url': str,
            'title': str,
            'markdown': str,
            'success': bool,
            'error': str or None
        }
        """
        try:
            # Use Firecrawl to scrape the page
            result = self.firecrawl.scrape_url(
                url,
                params={'formats': ['markdown', 'html']}
            )
            
            if not result or 'markdown' not in result:
                return {
                    'url': url,
                    'title': title,
                    'markdown': '',
                    'success': False,
                    'error': 'No markdown content returned'
                }
            
            return {
                'url': url,
                'title': result.get('title', title),
                'markdown': result['markdown'],
                'success': True,
                'error': None
            }
            
        except Exception as e:
            return {
                'url': url,
                'title': title,
                'markdown': '',
                'success': False,
                'error': str(e)
            }
    
    def scrape_with_series_detection(self, url: str, title: str = "") -> Dict:
        """
        Scrape article and detect/handle series
        """
        # Scrape main article
        result = self.scrape_article(url, title)
        
        if not result['success']:
            return result
        
        # Detect if part of series
        series_info = self.detect_series(result['title'], url)
        result['series_info'] = series_info
        
        if series_info['is_series']:
            # Try to find other parts
            related_urls = self.find_series_parts(
                url,
                result['markdown'],
                series_info['series_name'],
                series_info['part_number']
            )
            result['series_related_urls'] = related_urls
        else:
            result['series_related_urls'] = []
        
        return result
    
    def save_markdown(self, article_data: Dict, category: str = "uncategorized") -> str:
        """Save article markdown to file"""
        # Create category directory
        category_dir = self.output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename from title
        title = article_data['title']
        series_info = article_data.get('series_info', {})
        
        if series_info.get('is_series'):
            part_num = series_info['part_number']
            filename = f"{title.replace('/', '_').replace(' ', '_')}_part{part_num}.md"
        else:
            filename = f"{title.replace('/', '_').replace(' ', '_')}.md"
        
        # Clean filename
        filename = re.sub(r'[<>:"|?*]', '', filename)
        filename = filename[:200]  # Limit length
        
        filepath = category_dir / filename
        
        # Write markdown with metadata
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {article_data['title']}\n\n")
            f.write(f"**URL**: {article_data['url']}\n\n")
            
            if series_info.get('is_series'):
                f.write(f"**Series**: {series_info['series_name']}\n")
                f.write(f"**Part**: {series_info['part_number']}\n\n")
                
                if article_data.get('series_related_urls'):
                    f.write("**Related Parts**:\n")
                    for related_url in article_data['series_related_urls']:
                        f.write(f"- {related_url}\n")
                    f.write("\n")
            
            f.write("---\n\n")
            f.write(article_data['markdown'])
        
        return str(filepath)
    
    def scrape_from_export(self, export_file: str, save_markdown: bool = True) -> List[Dict]:
        """
        Scrape all articles from extension export JSON
        """
        # Load export file
        with open(export_file, 'r') as f:
            articles = json.load(f)
        
        print(f"Found {len(articles)} articles to scrape")
        
        results = []
        for article in tqdm(articles, desc="Scraping articles"):
            url = article['url']
            title = article['title']
            category = article.get('category', 'uncategorized')
            
            # Scrape with series detection
            result = self.scrape_with_series_detection(url, title)
            result['category'] = category
            
            if result['success'] and save_markdown:
                filepath = self.save_markdown(result, category)
                result['saved_to'] = filepath
            
            results.append(result)
        
        return results


def main():
    """Example usage"""
    scraper = MQL5Scraper()
    
    # Example: scrape from export
    export_file = "data/exports/quantmindx_articles_2026-01-22.json"
    
    if os.path.exists(export_file):
        results = scraper.scrape_from_export(export_file)
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        series_detected = sum(1 for r in results if r.get('series_info', {}).get('is_series'))
        
        print(f"\nScraping complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Series detected: {series_detected}")
    else:
        print(f"Export file not found: {export_file}")
        print("Please export articles from the Chrome extension first.")


if __name__ == "__main__":
    main()
