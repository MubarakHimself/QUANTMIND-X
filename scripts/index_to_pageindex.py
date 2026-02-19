#!/usr/bin/env python3
"""
PageIndex Indexing Script for QuantMindX Knowledge Base

Indexes documents into PageIndex services for reasoning-based retrieval.
Supports articles, books, and logs collections.

Usage:
    python scripts/index_to_pageindex.py --collection articles
    python scripts/index_to_pageindex.py --collection books
    python scripts/index_to_pageindex.py --collection logs
    python scripts/index_to_pageindex.py --all
"""

import os
import sys
import argparse
import json
import re
import httpx
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# PageIndex service URLs
PAGEINDEX_URLS = {
    "articles": os.environ.get("PAGEINDEX_ARTICLES_URL", "http://localhost:3000"),
    "books": os.environ.get("PAGEINDEX_BOOKS_URL", "http://localhost:3001"),
    "logs": os.environ.get("PAGEINDEX_LOGS_URL", "http://localhost:3002"),
}

HTTP_TIMEOUT = httpx.Timeout(60.0)


def parse_frontmatter(content: str) -> Dict[str, Any]:
    """Extract YAML frontmatter from markdown content."""
    frontmatter = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            for line in fm_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    frontmatter[key.strip()] = value.strip().strip('"\'')
    return frontmatter


def extract_title_from_content(content: str, file_path: Path) -> str:
    """Extract title from content or filename."""
    # Try frontmatter first
    fm = parse_frontmatter(content)
    if "title" in fm:
        return fm["title"]
    
    # Try first heading
    for line in content.split("\n"):
        if line.startswith("# "):
            return line[2:].strip()
    
    # Fall back to filename
    return file_path.stem.replace("-", " ").replace("_", " ").title()


def extract_sections(content: str) -> List[Dict[str, Any]]:
    """Extract sections from markdown content based on headings."""
    sections = []
    current_section = {"title": "Introduction", "content": "", "level": 0}
    
    for line in content.split("\n"):
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading_match:
            # Save previous section
            if current_section["content"].strip():
                sections.append(current_section)
            
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            current_section = {"title": title, "content": "", "level": level}
        else:
            current_section["content"] += line + "\n"
    
    # Save last section
    if current_section["content"].strip():
        sections.append(current_section)
    
    return sections


def index_articles() -> Dict[str, Any]:
    """Index scraped articles to PageIndex articles service."""
    articles_dir = PROJECT_ROOT / "data" / "scraped_articles"
    
    if not articles_dir.exists():
        return {"error": f"Articles directory not found: {articles_dir}"}
    
    pageindex_url = PAGEINDEX_URLS["articles"]
    indexed_count = 0
    errors = []
    
    # Find all markdown files
    md_files = list(articles_dir.rglob("*.md"))
    
    print(f"Found {len(md_files)} article files in {articles_dir}")
    
    for file_path in md_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            
            # Extract metadata
            title = extract_title_from_content(content, file_path)
            frontmatter = parse_frontmatter(content)
            sections = extract_sections(content)
            
            # Create document for PageIndex
            document = {
                "id": str(file_path.relative_to(articles_dir)),
                "title": title,
                "content": content,
                "source": str(file_path.relative_to(PROJECT_ROOT)),
                "url": frontmatter.get("url", ""),
                "categories": frontmatter.get("categories", ""),
                "date": frontmatter.get("date", ""),
                "sections": [
                    {
                        "title": s["title"],
                        "content": s["content"][:2000],  # Truncate long sections
                    }
                    for s in sections
                ],
                "indexed_at": datetime.utcnow().isoformat(),
            }
            
            # POST to PageIndex
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                response = client.post(
                    f"{pageindex_url}/index",
                    json=document
                )
                response.raise_for_status()
                indexed_count += 1
                print(f"  Indexed: {title} ({file_path.relative_to(articles_dir)})")
                
        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")
            print(f"  Error indexing {file_path}: {e}")
    
    return {
        "collection": "articles",
        "indexed": indexed_count,
        "total_files": len(md_files),
        "errors": errors,
    }


def index_books() -> Dict[str, Any]:
    """Index books/PDFs to PageIndex books service."""
    books_dir = PROJECT_ROOT / "data" / "knowledge_base" / "books"
    
    if not books_dir.exists():
        return {"error": f"Books directory not found: {books_dir}"}
    
    pageindex_url = PAGEINDEX_URLS["books"]
    indexed_count = 0
    pdf_indexed_count = 0
    errors = []
    
    # Find all PDF and markdown files
    book_files = list(books_dir.rglob("*.md")) + list(books_dir.rglob("*.txt"))
    
    # Also check for PDF files to index
    pdf_files = list(books_dir.rglob("*.pdf"))
    
    print(f"Found {len(book_files)} text files in {books_dir}")
    if pdf_files:
        print(f"Found {len(pdf_files)} PDF files to index")
    
    # Index text files (markdown and txt)
    for file_path in book_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            title = extract_title_from_content(content, file_path)
            sections = extract_sections(content)
            
            document = {
                "id": str(file_path.relative_to(books_dir)),
                "title": title,
                "content": content,
                "source": str(file_path.relative_to(PROJECT_ROOT)),
                "sections": [
                    {
                        "title": s["title"],
                        "content": s["content"][:2000],
                    }
                    for s in sections
                ],
                "indexed_at": datetime.utcnow().isoformat(),
            }
            
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                response = client.post(
                    f"{pageindex_url}/index",
                    json=document
                )
                response.raise_for_status()
                indexed_count += 1
                print(f"  Indexed: {title}")
                
        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")
            print(f"  Error indexing {file_path}: {e}")
    
    # Index PDF files by uploading to PageIndex books service
    for pdf_path in pdf_files:
        try:
            pdf_title = pdf_path.stem.replace("-", " ").replace("_", " ").title()
            print(f"  Indexing PDF: {pdf_title}...")
            
            # Read PDF file in binary mode and upload
            with open(pdf_path, "rb") as pdf_file:
                files = {
                    "file": (pdf_path.name, pdf_file, "application/pdf")
                }
                data = {
                    "id": str(pdf_path.relative_to(books_dir)),
                    "title": pdf_title,
                    "source": str(pdf_path.relative_to(PROJECT_ROOT)),
                    "indexed_at": datetime.utcnow().isoformat(),
                }
                
                with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                    # Try multipart file upload endpoint first
                    response = client.post(
                        f"{pageindex_url}/index/pdf",
                        files=files,
                        data=data
                    )
                    
                    # If /index/pdf endpoint doesn't exist, try /index with file path
                    if response.status_code == 404:
                        # Fallback: send file path/metadata for server-side processing
                        document = {
                            "id": str(pdf_path.relative_to(books_dir)),
                            "title": pdf_title,
                            "source": str(pdf_path.relative_to(PROJECT_ROOT)),
                            "file_path": str(pdf_path.absolute()),
                            "file_type": "pdf",
                            "indexed_at": datetime.utcnow().isoformat(),
                        }
                        response = client.post(
                            f"{pageindex_url}/index",
                            json=document
                        )
                    
                    response.raise_for_status()
                    pdf_indexed_count += 1
                    print(f"    Indexed PDF: {pdf_title}")
                    
        except httpx.HTTPStatusError as e:
            error_msg = f"{pdf_path}: HTTP {e.response.status_code} - {e}"
            errors.append(error_msg)
            print(f"  Error indexing PDF {pdf_path}: {error_msg}")
        except Exception as e:
            errors.append(f"{pdf_path}: {str(e)}")
            print(f"  Error indexing PDF {pdf_path}: {e}")
    
    total_indexed = indexed_count + pdf_indexed_count
    total_files = len(book_files) + len(pdf_files)
    
    return {
        "collection": "books",
        "indexed": total_indexed,
        "text_files_indexed": indexed_count,
        "pdfs_indexed": pdf_indexed_count,
        "total_files": total_files,
        "pdfs_found": len(pdf_files),
        "errors": errors,
    }


def index_logs() -> Dict[str, Any]:
    """Index trading logs to PageIndex logs service."""
    logs_dir = PROJECT_ROOT / "data" / "logs"
    
    if not logs_dir.exists():
        # Create directory if it doesn't exist
        logs_dir.mkdir(parents=True, exist_ok=True)
        return {"message": f"Created logs directory: {logs_dir}", "indexed": 0}
    
    pageindex_url = PAGEINDEX_URLS["logs"]
    indexed_count = 0
    errors = []
    
    # Find all log and JSON files
    log_files = (
        list(logs_dir.rglob("*.log")) +
        list(logs_dir.rglob("*.json")) +
        list(logs_dir.rglob("*.csv"))
    )
    
    print(f"Found {len(log_files)} log files in {logs_dir}")
    
    for file_path in log_files:
        try:
            content = file_path.read_text(encoding="utf-8")
            
            # For JSON files, parse and structure
            if file_path.suffix == ".json":
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        content = "\n".join(json.dumps(item) for item in data)
                except json.JSONDecodeError:
                    pass
            
            document = {
                "id": str(file_path.relative_to(logs_dir)),
                "title": file_path.name,
                "content": content[:50000],  # Limit log size
                "source": str(file_path.relative_to(PROJECT_ROOT)),
                "type": file_path.suffix,
                "indexed_at": datetime.utcnow().isoformat(),
            }
            
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                response = client.post(
                    f"{pageindex_url}/index",
                    json=document
                )
                response.raise_for_status()
                indexed_count += 1
                print(f"  Indexed: {file_path.name}")
                
        except Exception as e:
            errors.append(f"{file_path}: {str(e)}")
            print(f"  Error indexing {file_path}: {e}")
    
    return {
        "collection": "logs",
        "indexed": indexed_count,
        "total_files": len(log_files),
        "errors": errors,
    }


def check_service_health(collection: str) -> bool:
    """Check if a PageIndex service is healthy."""
    if collection not in PAGEINDEX_URLS:
        return False
    
    url = PAGEINDEX_URLS[collection]
    try:
        with httpx.Client(timeout=httpx.Timeout(5.0)) as client:
            response = client.get(f"{url}/health")
            return response.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Index documents to PageIndex services"
    )
    parser.add_argument(
        "--collection",
        choices=["articles", "books", "logs", "all"],
        default="all",
        help="Collection to index (default: all)"
    )
    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Check PageIndex service health before indexing"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("QuantMindX PageIndex Indexer")
    print("=" * 60)
    
    # Check health if requested
    if args.check_health:
        print("\nChecking PageIndex service health...")
        for collection in ["articles", "books", "logs"]:
            healthy = check_service_health(collection)
            status = "OK" if healthy else "UNREACHABLE"
            print(f"  {collection}: {status} ({PAGEINDEX_URLS[collection]})")
            if not healthy:
                print(f"    Warning: {collection} service may not be running")
        print()
    
    # Index based on collection argument
    results = {}
    
    if args.collection == "all":
        print("\nIndexing all collections...\n")
        
        print("--- Articles ---")
        results["articles"] = index_articles()
        
        print("\n--- Books ---")
        results["books"] = index_books()
        
        print("\n--- Logs ---")
        results["logs"] = index_logs()
    else:
        print(f"\nIndexing {args.collection}...\n")
        if args.collection == "articles":
            results["articles"] = index_articles()
        elif args.collection == "books":
            results["books"] = index_books()
        elif args.collection == "logs":
            results["logs"] = index_logs()
    
    # Summary
    print("\n" + "=" * 60)
    print("Indexing Summary")
    print("=" * 60)
    
    total_indexed = 0
    total_errors = 0
    
    for collection, result in results.items():
        if "error" in result:
            print(f"{collection}: ERROR - {result['error']}")
        else:
            indexed = result.get("indexed", 0)
            errors = len(result.get("errors", []))
            total_indexed += indexed
            total_errors += errors
            print(f"{collection}: {indexed} documents indexed, {errors} errors")
    
    print(f"\nTotal: {total_indexed} documents indexed, {total_errors} errors")
    
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())