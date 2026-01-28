#!/usr/bin/env python3
"""
Document Summary Index Generator
Scans all scraped articles and creates a summary index for AI analysis.
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
SCRAPED_DIR = Path("data/scraped_articles")
OUTPUT_DIR = Path("data/knowledge_index")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown."""
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter = parts[1].strip()
            meta = {}
            for line in frontmatter.split('\n'):
                if ':' in line:
                    key, val = line.split(':', 1)
                    meta[key.strip()] = val.strip()
            return meta, parts[2].strip()
    return {}, content

def extract_keywords(content: str) -> list:
    """Extract trading-related keywords from content."""
    # Common trading terms to look for
    keywords = []
    terms = [
        'RSI', 'MACD', 'Moving Average', 'Bollinger', 'Fibonacci',
        'Stop Loss', 'Take Profit', 'Trailing Stop', 'Expert Advisor',
        'Scalping', 'Swing Trading', 'Day Trading', 'Price Action',
        'Candlestick', 'Support', 'Resistance', 'Trend', 'Breakout',
        'Neural Network', 'Machine Learning', 'Backtest', 'Optimization',
        'Risk Management', 'Money Management', 'Entry', 'Exit',
        'Order', 'Position', 'Lot', 'Spread', 'Pip', 'Drawdown',
        'MQL5', 'MQL4', 'MetaTrader', 'MT5', 'MT4', 'Indicator',
        'Python', 'DLL', 'API', 'WebSocket', 'JSON', 'REST'
    ]
    
    content_lower = content.lower()
    for term in terms:
        if term.lower() in content_lower:
            keywords.append(term)
    
    return keywords[:15]  # Limit to 15

def classify_article(content: str, title: str, category: str) -> str:
    """Classify article by primary use case."""
    content_lower = content.lower()
    title_lower = title.lower()
    
    # Classification rules
    if any(t in content_lower for t in ['neural network', 'machine learning', 'deep learning', 'tensorflow', 'pytorch']):
        return 'ml_strategy'
    if any(t in content_lower for t in ['indicator', 'oscillator', 'signal']):
        if 'custom' in title_lower or 'create' in title_lower:
            return 'indicator_development'
        return 'indicator_usage'
    if any(t in content_lower for t in ['expert advisor', 'trading robot', 'automated']):
        return 'ea_development'
    if any(t in content_lower for t in ['backtest', 'optimization', 'strategy tester']):
        return 'backtesting'
    if any(t in content_lower for t in ['python', 'dll', 'api', 'socket', 'json']):
        return 'integration'
    if any(t in content_lower for t in ['entry', 'exit', 'strategy', 'trading system']):
        return 'trading_logic'
    if any(t in content_lower for t in ['risk', 'money management', 'position size']):
        return 'risk_management'
    
    return 'general'

def process_article(file_path: Path) -> dict:
    """Process a single article and extract summary info."""
    try:
        content = file_path.read_text(encoding='utf-8')
        meta, body = extract_frontmatter(content)
        
        # Get first 300 chars of body (cleaned)
        preview = re.sub(r'\[.*?\]|\(.*?\)|[#*_\n]', ' ', body[:500])
        preview = ' '.join(preview.split())[:300]
        
        return {
            'file': str(file_path.relative_to(SCRAPED_DIR)),
            'title': meta.get('title', file_path.stem.replace('_', ' ')),
            'url': meta.get('url', ''),
            'categories': meta.get('categories', 'Unknown'),
            'relevance_score': int(meta.get('relevance_score', 0)),
            'keywords': extract_keywords(content),
            'classification': classify_article(content, meta.get('title', ''), meta.get('categories', '')),
            'preview': preview,
            'word_count': len(body.split())
        }
    except Exception as e:
        return {
            'file': str(file_path),
            'error': str(e)
        }

def main():
    print("📚 Document Summary Index Generator")
    print("=" * 60)
    
    all_articles = []
    category_stats = defaultdict(int)
    classification_stats = defaultdict(int)
    
    # Find all markdown files
    md_files = list(SCRAPED_DIR.rglob("*.md"))
    print(f"Found {len(md_files)} articles\n")
    
    for i, file_path in enumerate(md_files, 1):
        if i % 100 == 0:
            print(f"   Processing {i}/{len(md_files)}...")
        
        article = process_article(file_path)
        all_articles.append(article)
        
        # Stats
        if 'error' not in article:
            category_stats[article['categories']] += 1
            classification_stats[article['classification']] += 1
    
    # Sort by relevance score
    all_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    # Create summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'total_articles': len(all_articles),
        'category_breakdown': dict(category_stats),
        'classification_breakdown': dict(classification_stats),
        'articles': all_articles
    }
    
    # Save full index
    full_index_path = OUTPUT_DIR / "document_index.json"
    full_index_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n✅ Full index saved: {full_index_path}")
    
    # Create compact summary (top 200 by relevance)
    compact = {
        'generated_at': summary['generated_at'],
        'total_articles': summary['total_articles'],
        'category_breakdown': summary['category_breakdown'],
        'classification_breakdown': summary['classification_breakdown'],
        'top_articles': all_articles[:200]
    }
    compact_path = OUTPUT_DIR / "document_summary.json"
    compact_path.write_text(json.dumps(compact, indent=2, ensure_ascii=False))
    print(f"✅ Compact summary saved: {compact_path}")
    
    # Print stats
    print("\n📊 Classification Breakdown:")
    for cls, count in sorted(classification_stats.items(), key=lambda x: -x[1]):
        print(f"   {cls}: {count}")
    
    print("\n📂 Category Breakdown:")
    for cat, count in sorted(category_stats.items(), key=lambda x: -x[1])[:10]:
        print(f"   {cat}: {count}")
    
    print(f"\n✨ Done! Index ready for analysis.")

if __name__ == "__main__":
    main()
