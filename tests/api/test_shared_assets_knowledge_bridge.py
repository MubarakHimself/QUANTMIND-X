import io
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def bridge_app(tmp_path, monkeypatch):
    import src.api.ide_assets as ide_assets
    import src.api.ide_handlers_assets as assets_handlers
    import src.api.ide_handlers_knowledge as knowledge_handlers
    import src.api.ide_knowledge as ide_knowledge

    assets_dir = tmp_path / "shared_assets"
    knowledge_dir = tmp_path / "knowledge"
    scraped_dir = tmp_path / "scraped_articles"

    monkeypatch.setattr(assets_handlers, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(assets_handlers, "KNOWLEDGE_DIR", knowledge_dir, raising=False)
    monkeypatch.setattr(assets_handlers, "SCRAPED_ARTICLES_DIR", scraped_dir, raising=False)
    monkeypatch.setattr(knowledge_handlers, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(knowledge_handlers, "SCRAPED_ARTICLES_DIR", scraped_dir)
    monkeypatch.setattr(ide_knowledge, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(ide_knowledge, "SCRAPED_ARTICLES_DIR", scraped_dir)

    ide_assets.assets_handler = assets_handlers.AssetsAPIHandler()
    ide_knowledge.knowledge_handler = knowledge_handlers.KnowledgeAPIHandler()

    app = FastAPI()
    app.include_router(ide_assets.router)
    app.include_router(ide_knowledge.router)
    app.include_router(ide_knowledge.knowledge_upload_router)
    return app


@pytest.fixture
def client(bridge_app):
    return TestClient(bridge_app)


def test_shared_assets_lists_scraped_articles_and_uploaded_files(client, tmp_path):
    scraped_dir = tmp_path / "scraped_articles" / "trading_systems"
    scraped_dir.mkdir(parents=True, exist_ok=True)
    article_path = scraped_dir / "market_structure.md"
    article_path.write_text(
        "---\n"
        "title: Market Structure Notes\n"
        "source_url: https://example.com/market-structure\n"
        "relevance_tags:\n"
        "  - smc\n"
        "  - structure\n"
        "---\n\n"
        "This article explains market structure and BOS behaviour.",
        encoding="utf-8",
    )
    article_path.with_suffix(".json").write_text(
        json.dumps({"title": "Market Structure Notes", "url": "https://example.com/market-structure"}),
        encoding="utf-8",
    )

    upload_response = client.post(
        "/api/assets/upload",
        data={"category": "Books", "description": "Execution playbook"},
        files={"file": ("execution-playbook.pdf", io.BytesIO(b"%PDF-1.4\nbook"), "application/pdf")},
    )
    assert upload_response.status_code == 200, upload_response.text

    docs_response = client.post(
        "/api/assets/upload",
        data={"category": "Docs", "description": "Deployment runbook"},
        files={"file": ("runbook.md", io.BytesIO(b"# Deploy\nSteps"), "text/markdown")},
    )
    assert docs_response.status_code == 200, docs_response.text

    response = client.get("/api/assets/shared")
    assert response.status_code == 200, response.text

    assets = response.json()
    ids = {asset["id"] for asset in assets}
    categories = {asset["category"] for asset in assets}

    assert "scraped/trading_systems/market_structure.md" in ids
    assert "knowledge/books/execution-playbook.pdf" in ids
    assert "docs/runbook.md" in ids
    assert {"Articles", "Books", "Docs"}.issubset(categories)


def test_research_articles_endpoint_returns_merged_live_article_cards(client, tmp_path):
    knowledge_articles = tmp_path / "knowledge" / "articles"
    knowledge_articles.mkdir(parents=True, exist_ok=True)
    uploaded_article = knowledge_articles / "desk-note.md"
    uploaded_article.write_text("# Desk Note\nLondon session observations.", encoding="utf-8")
    uploaded_article.with_name("desk-note.md.metadata.json").write_text(
        json.dumps(
            {
                "title": "Desk Note",
                "url": "https://desk.local/note",
                "subcategory": "london-session",
                "uploaded_at": "2026-04-02T10:00:00",
            }
        ),
        encoding="utf-8",
    )

    scraped_dir = tmp_path / "scraped_articles" / "trading_systems"
    scraped_dir.mkdir(parents=True, exist_ok=True)
    scraped_article = scraped_dir / "volatility-breakout.md"
    scraped_article.write_text(
        "---\n"
        "title: Volatility Breakout\n"
        "relevance_tags:\n"
        "  - breakout\n"
        "---\n\n"
        "Breakout article excerpt.",
        encoding="utf-8",
    )

    response = client.get("/api/knowledge/articles")
    assert response.status_code == 200, response.text

    articles = response.json()
    assert isinstance(articles, list)
    titles = {article["title"] for article in articles}
    assert "Desk Note" in titles
    assert "Volatility Breakout" in titles

    uploaded = next(article for article in articles if article["title"] == "Desk Note")
    scraped = next(article for article in articles if article["title"] == "Volatility Breakout")

    assert uploaded["source"] == "desk.local"
    assert "london-session" in uploaded["tags"]
    assert "breakout" in scraped["tags"]


def test_research_articles_endpoint_recovers_title_and_category_from_scraped_markdown(client, tmp_path):
    scraped_root = tmp_path / "scraped_articles"
    scraped_root.mkdir(parents=True, exist_ok=True)
    scraped_article = scraped_root / "6f2e0ae3-a2b2-4fed-ab54-aca5deb5afa6.md"
    scraped_article.write_text(
        "---\n"
        "job_id: 6f2e0ae3-a2b2-4fed-ab54-aca5deb5afa6\n"
        "source_url: https://example.com/retry-test\n"
        "relevance_tags: []\n"
        "---\n\n"
        "# Success After Retry\n\n"
        "Recovered article body.",
        encoding="utf-8",
    )

    response = client.get("/api/knowledge/articles")
    assert response.status_code == 200, response.text

    articles = response.json()
    recovered = next(article for article in articles if article["id"].endswith("6f2e0ae3-a2b2-4fed-ab54-aca5deb5afa6.md"))

    assert recovered["title"] == "Success After Retry"
    assert recovered["source"] == "example.com"
    assert recovered["tags"] == ["root"]


def test_research_books_endpoint_lists_uploaded_books(client):
    response = client.post(
        "/api/assets/upload",
        data={"category": "Books", "author": "N. Taleb", "description": "Risk book"},
        files={"file": ("fooled-by-randomness.pdf", io.BytesIO(b"%PDF-1.4\nbook"), "application/pdf")},
    )
    assert response.status_code == 200, response.text

    books_response = client.get("/api/knowledge/books")
    assert books_response.status_code == 200, books_response.text

    books = books_response.json()
    assert books == [
        {
            "id": "books/fooled-by-randomness.pdf",
            "title": "fooled-by-randomness",
            "author": "N. Taleb",
            "topics": [],
            "year": None,
        }
    ]
