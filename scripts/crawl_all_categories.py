"""
Quick script to crawl user's selected categories and get stats
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.category_crawler import MQL5CategoryCrawler
import json

# User's categories
categories = [
    {
        'name': 'Trading',
        'url': 'https://www.mql5.com/en/articles/trading'
    },
    {
        'name': 'Trading Systems',
        'url': 'https://www.mql5.com/en/articles/trading_systems'
    },
    {
        'name': 'Integration',
        'url': 'https://www.mql5.com/en/articles/integration'
    },
    {
        'name': 'Indicators',
        'url': 'https://www.mql5.com/en/articles/indicators'
    },
    {
        'name': 'Expert Advisors',
        'url': 'https://www.mql5.com/en/articles/expert_advisors'
    },
    {
        'name': 'Machine Learning',
        'url': 'https://www.mql5.com/en/articles/machine_learning'
    },
    {
        'name': 'Strategy Tester',
        'url': 'https://www.mql5.com/en/articles/strategy_tester'
    }
]

print("Crawling MQL5 categories...")
print("="*60)

crawler = MQL5CategoryCrawler(delay=0.5)
results = crawler.crawl_multiple_categories(categories)

# Deduplicate across all categories
all_articles = []
seen_urls = set()
category_counts = {}

for category_name, articles in results.items():
    category_counts[category_name] = len(articles)
    
    for article in articles:
        if article['url'] not in seen_urls:
            seen_urls.add(article['url'])
            article['categories'] = [category_name]
            all_articles.append(article)
        else:
            # Article already seen, add this category to its list
            for existing in all_articles:
                if existing['url'] == article['url']:
                    existing['categories'].append(category_name)
                    break

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nPer-Category Counts (before deduplication):")
for name, count in category_counts.items():
    print(f"  {name:25s}: {count:4d} articles")

total_before = sum(category_counts.values())
print(f"\n{'TOTAL (with duplicates)':25s}: {total_before:4d} articles")

print(f"\n{'UNIQUE ARTICLES':25s}: {len(all_articles):4d} articles")
print(f"{'DUPLICATES REMOVED':25s}: {total_before - len(all_articles):4d} articles")

print(f"\n{'FIRECRAWL PAGES NEEDED':25s}: {len(all_articles):4d}/{500}")

if len(all_articles) > 500:
    print(f"\n⚠️  WARNING: Exceeds 500-page limit by {len(all_articles) - 500} articles")
    print("   Will need to prioritize categories or use second account")
else:
    print(f"\n✅  Within 500-page limit! ({500 - len(all_articles)} pages to spare)")

# Save deduped articles
output_file = 'data/exports/all_categories_deduped.json'
with open(output_file, 'w') as f:
    json.dump(all_articles, f, indent=2)

print(f"\nSaved to: {output_file}")

# Show articles that appear in multiple categories
multi_category = [a for a in all_articles if len(a['categories']) > 1]
if multi_category:
    print(f"\nArticles in multiple categories: {len(multi_category)}")
    print("\nTop 5 examples:")
    for article in multi_category[:5]:
        print(f"  - {article['title']}")
        print(f"    Categories: {', '.join(article['categories'])}")
