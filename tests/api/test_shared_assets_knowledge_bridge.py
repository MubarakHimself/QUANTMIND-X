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
    import src.api.wf1_artifacts as wf1_artifacts

    assets_dir = tmp_path / "shared_assets"
    knowledge_dir = tmp_path / "knowledge"
    scraped_dir = tmp_path / "scraped_articles"

    monkeypatch.setattr(assets_handlers, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(assets_handlers, "KNOWLEDGE_DIR", knowledge_dir, raising=False)
    monkeypatch.setattr(assets_handlers, "SCRAPED_ARTICLES_DIR", scraped_dir, raising=False)
    monkeypatch.setattr(assets_handlers, "WF1_STRATEGIES_ROOT", assets_dir / "strategies", raising=False)
    monkeypatch.setattr(wf1_artifacts, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(wf1_artifacts, "WF1_STRATEGIES_ROOT", assets_dir / "strategies")
    monkeypatch.setattr(wf1_artifacts, "STRATEGIES_DIR", tmp_path / "strategies")
    monkeypatch.setattr(knowledge_handlers, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(knowledge_handlers, "SCRAPED_ARTICLES_DIR", scraped_dir)
    monkeypatch.setattr(ide_knowledge, "KNOWLEDGE_DIR", knowledge_dir)
    monkeypatch.setattr(ide_knowledge, "SCRAPED_ARTICLES_DIR", scraped_dir)
    monkeypatch.setenv("QMX_BOOTSTRAP_REFERENCE_BOOKS", "0")

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


def test_shared_assets_counts_endpoint_groups_books_articles_and_strategies(client, tmp_path, monkeypatch):
    import src.api.wf1_artifacts as wf1_artifacts

    response = client.post(
        "/api/assets/upload",
        data={"category": "Books", "author": "MetaQuotes", "description": "Developer guide"},
        files={"file": ("mql5-reference.pdf", io.BytesIO(b"%PDF-1.4\nbook"), "application/pdf")},
    )
    assert response.status_code == 200, response.text

    knowledge_articles = tmp_path / "knowledge" / "articles"
    knowledge_articles.mkdir(parents=True, exist_ok=True)
    article = knowledge_articles / "desk-note.md"
    article.write_text("# Desk note", encoding="utf-8")

    assets_dir = tmp_path / "shared_assets"
    strategies_root = assets_dir / "strategies"
    legacy_root = tmp_path / "strategies"
    monkeypatch.setattr(wf1_artifacts, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(wf1_artifacts, "WF1_STRATEGIES_ROOT", strategies_root)
    monkeypatch.setattr(wf1_artifacts, "STRATEGIES_DIR", legacy_root)

    wf1_artifacts.ensure_bundle(
        strategy_name="London Scalper",
        is_playlist=False,
        workflow_id="wf1_test_counts_001",
        source_url="https://youtube.com/watch?v=abc123",
    )

    counts_response = client.get("/api/assets/counts")
    assert counts_response.status_code == 200, counts_response.text
    assert counts_response.json() == {
        "docs": 2,
        "strategy-templates": 0,
        "indicators": 0,
        "skills": 0,
        "flow-components": 0,
        "mcp-configs": 0,
        "strategies": 1,
    }


def test_shared_assets_docs_category_returns_books_articles_and_docs_bucket(client, tmp_path):
    books_response = client.post(
        "/api/assets/upload",
        data={"category": "Books", "author": "MetaQuotes", "description": "Developer guide"},
        files={"file": ("mql5-reference.pdf", io.BytesIO(b"%PDF-1.4\nbook"), "application/pdf")},
    )
    assert books_response.status_code == 200, books_response.text

    docs_response = client.post(
        "/api/assets/upload",
        data={"category": "Docs", "description": "Terminal runbook"},
        files={"file": ("terminal-runbook.md", io.BytesIO(b"# Runbook\n"), "text/markdown")},
    )
    assert docs_response.status_code == 200, docs_response.text

    knowledge_articles = tmp_path / "knowledge" / "articles"
    knowledge_articles.mkdir(parents=True, exist_ok=True)
    article = knowledge_articles / "desk-note.md"
    article.write_text("# Desk note", encoding="utf-8")

    scraped_root = tmp_path / "scraped_articles"
    scraped_root.mkdir(parents=True, exist_ok=True)
    bogus = scraped_root / "a8ca233e-96fc-4a51-aa8c-1e684fcc9d9c.md"
    bogus.write_text(
        "---\n"
        "job_id: a8ca233e-96fc-4a51-aa8c-1e684fcc9d9c\n"
        "source_url: https://example.com/retry\n"
        "---\n\n"
        "# Success\n",
        encoding="utf-8",
    )
    category_dir = scraped_root / "trading_systems"
    category_dir.mkdir(parents=True, exist_ok=True)
    categorized = category_dir / "volatility-breakout.md"
    categorized.write_text(
        "---\n"
        "title: Volatility Breakout\n"
        "---\n\n"
        "Breakout article.",
        encoding="utf-8",
    )

    response = client.get("/api/assets?category=docs")
    assert response.status_code == 200, response.text

    assets = response.json()
    ids = {asset["id"] for asset in assets}
    types = {asset["type"] for asset in assets}
    names = {asset["name"] for asset in assets}

    assert "knowledge/books/mql5-reference.pdf" in ids
    assert "knowledge/articles/desk-note.md" in ids
    assert "docs/terminal-runbook.md" in ids
    assert "scraped/trading_systems/volatility-breakout.md" in ids
    assert "scraped/a8ca233e-96fc-4a51-aa8c-1e684fcc9d9c.md" not in ids
    assert {"books", "articles", "docs"}.issubset(types)
    assert "Volatility Breakout" in names


def test_shared_assets_docs_hides_root_level_example_scrape_artifacts(client, tmp_path):
    scraped_dir = tmp_path / "scraped_articles"
    scraped_dir.mkdir(parents=True, exist_ok=True)

    bogus = scraped_dir / "a8ca233e-96fc-4a51-aa8c-1e684fcc9d9c.md"
    bogus.write_text(
        """---
job_id: a8ca233e-96fc-4a51-aa8c-1e684fcc9d9c
source_url: 'https://example.com/backoff-test'
scraped_at_utc: 2026-03-21T16:22:11.908385+00:00
---

# Success
""",
        encoding="utf-8",
    )

    real_dir = scraped_dir / "trading"
    real_dir.mkdir(parents=True, exist_ok=True)
    real_article = real_dir / "real-article.md"
    real_article.write_text(
        """---
source_url: 'https://www.mql5.com/en/articles/12345'
---

# Real MQL5 Article
""",
        encoding="utf-8",
    )

    response = client.get("/api/assets/shared")
    assert response.status_code == 200, response.text
    payload = response.json()

    names = {item["name"] for item in payload}
    ids = {item["id"] for item in payload}

    assert "Success" not in names
    assert "scraped/a8ca233e-96fc-4a51-aa8c-1e684fcc9d9c.md" not in ids
    assert "Real MQL5 Article" in names


def test_shared_assets_docs_hides_root_level_uuid_scrapes_even_when_they_have_real_titles(client, tmp_path):
    scraped_dir = tmp_path / "scraped_articles"
    scraped_dir.mkdir(parents=True, exist_ok=True)

    root_article = scraped_dir / "6f2e0ae3-a2b2-4fed-ab54-aca5deb5afa6.md"
    root_article.write_text(
        """---
job_id: 6f2e0ae3-a2b2-4fed-ab54-aca5deb5afa6
source_url: 'https://www.mql5.com/en/articles/99999'
relevance_tags:
  - scalping
---

# Scalping Recovery Notes
""",
        encoding="utf-8",
    )

    categorized_dir = scraped_dir / "trading_systems"
    categorized_dir.mkdir(parents=True, exist_ok=True)
    categorized_article = categorized_dir / "volatility-breakout.md"
    categorized_article.write_text(
        """---
source_url: 'https://www.mql5.com/en/articles/12345'
---

# Volatility Breakout
""",
        encoding="utf-8",
    )

    response = client.get("/api/assets/shared")
    assert response.status_code == 200, response.text
    payload = response.json()

    names = {item["name"] for item in payload}
    ids = {item["id"] for item in payload}

    assert "Scalping Recovery Notes" not in names
    assert "scraped/6f2e0ae3-a2b2-4fed-ab54-aca5deb5afa6.md" not in ids
    assert "Volatility Breakout" in names


def test_assets_strategies_category_returns_strategy_roots_not_internal_manifest_files(client, tmp_path, monkeypatch):
    import src.api.wf1_artifacts as wf1_artifacts

    assets_dir = tmp_path / "shared_assets"
    strategies_root = assets_dir / "strategies"
    legacy_root = tmp_path / "strategies"
    monkeypatch.setattr(wf1_artifacts, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(wf1_artifacts, "WF1_STRATEGIES_ROOT", strategies_root)
    monkeypatch.setattr(wf1_artifacts, "STRATEGIES_DIR", legacy_root)

    bundle = wf1_artifacts.ensure_bundle(
        strategy_name="London Scalper",
        is_playlist=False,
        workflow_id="wf1_test_bridge_001",
        source_url="https://youtube.com/watch?v=abc123",
    )
    (bundle.workflow_dir / "manifest.json").write_text('{"status":"pending"}', encoding="utf-8")

    response = client.get("/api/assets", params={"category": "strategies"})
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload == [
        {
            "id": "strategies/scalping/single-videos/london_scalper",
            "name": "London Scalper",
            "type": "strategies",
            "path": str(bundle.root),
            "description": "scalping · single-videos · pending",
            "used_in": [],
        }
    ]


def test_assets_strategies_category_hides_placeholder_wf1_roots_without_operator_visible_signal(client, tmp_path, monkeypatch):
    import src.api.wf1_artifacts as wf1_artifacts

    assets_dir = tmp_path / "shared_assets"
    strategies_root = assets_dir / "strategies"
    legacy_root = tmp_path / "strategies"
    monkeypatch.setattr(wf1_artifacts, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(wf1_artifacts, "WF1_STRATEGIES_ROOT", strategies_root)
    monkeypatch.setattr(wf1_artifacts, "STRATEGIES_DIR", legacy_root)

    wf1_artifacts.ensure_bundle(
        strategy_name="probe",
        is_playlist=False,
        workflow_id="wf1_probe_hidden",
    )
    visible = wf1_artifacts.ensure_bundle(
        strategy_name="London Scalper",
        is_playlist=False,
        workflow_id="wf1_visible_asset",
        source_url="https://youtube.com/watch?v=abc123",
    )

    response = client.get("/api/assets", params={"category": "strategies"})
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload == [
        {
            "id": "strategies/scalping/single-videos/london_scalper",
            "name": "London Scalper",
            "type": "strategies",
            "path": str(visible.root),
            "description": "scalping · single-videos · pending",
            "used_in": [],
        }
    ]


def test_strategy_asset_content_includes_strategy_detail_summary(client, tmp_path, monkeypatch):
    import src.api.wf1_artifacts as wf1_artifacts

    assets_dir = tmp_path / "shared_assets"
    strategies_root = assets_dir / "strategies"
    legacy_root = tmp_path / "strategies"
    monkeypatch.setattr(wf1_artifacts, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(wf1_artifacts, "WF1_STRATEGIES_ROOT", strategies_root)
    monkeypatch.setattr(wf1_artifacts, "STRATEGIES_DIR", legacy_root)

    bundle = wf1_artifacts.ensure_bundle(
        strategy_name="Caption Probe",
        is_playlist=False,
        workflow_id="wf1_test_bridge_002",
        source_url="https://youtube.com/watch?v=xyz456",
    )
    (bundle.root / "source" / "captions").mkdir(parents=True, exist_ok=True)
    (bundle.root / "source" / "captions" / "captions.en.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nHello", encoding="utf-8")
    (bundle.root / "source" / "chunks").mkdir(parents=True, exist_ok=True)
    (bundle.root / "source" / "chunks" / "job_001.json").write_text('{"chunk_count":1}', encoding="utf-8")

    response = client.get("/api/assets/strategies/scalping/single-videos/caption_probe/content")
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["content"]
    parsed = json.loads(payload["content"])
    assert parsed["type"] == "strategy_tree"
    assert parsed["detail"]["id"] == "caption_probe"
    assert parsed["detail"]["asset_id"] == "strategies/scalping/single-videos/caption_probe"
    assert parsed["detail"]["has_source_captions"] is True
    assert parsed["detail"]["has_chunk_manifest"] is True


def test_strategy_asset_content_uses_canonical_path_for_duplicate_names(client, tmp_path, monkeypatch):
    import src.api.wf1_artifacts as wf1_artifacts

    assets_dir = tmp_path / "shared_assets"
    strategies_root = assets_dir / "strategies"
    legacy_root = tmp_path / "strategies"
    monkeypatch.setattr(wf1_artifacts, "ASSETS_DIR", assets_dir)
    monkeypatch.setattr(wf1_artifacts, "WF1_STRATEGIES_ROOT", strategies_root)
    monkeypatch.setattr(wf1_artifacts, "STRATEGIES_DIR", legacy_root)

    single_bundle = wf1_artifacts.ensure_bundle(
        strategy_name="Collision Probe",
        is_playlist=False,
        workflow_id="wf1_collision_single",
        source_url="https://youtube.com/watch?v=single",
    )
    playlist_bundle = wf1_artifacts.ensure_bundle(
        strategy_name="Collision Probe",
        is_playlist=True,
        workflow_id="wf1_collision_playlist",
        source_url="https://youtube.com/playlist?list=playlist",
    )

    (single_bundle.root / "source" / "captions").mkdir(parents=True, exist_ok=True)
    (single_bundle.root / "source" / "captions" / "single.en.vtt").write_text("WEBVTT", encoding="utf-8")
    (playlist_bundle.root / "source" / "audio").mkdir(parents=True, exist_ok=True)
    (playlist_bundle.root / "source" / "audio" / "playlist.mp3").write_bytes(b"audio")

    single_response = client.get("/api/assets/strategies/scalping/single-videos/collision_probe/content")
    playlist_response = client.get("/api/assets/strategies/scalping/playlists/collision_probe/content")

    assert single_response.status_code == 200, single_response.text
    assert playlist_response.status_code == 200, playlist_response.text

    single_payload = json.loads(single_response.json()["content"])
    playlist_payload = json.loads(playlist_response.json()["content"])

    assert single_payload["detail"]["asset_id"] == "strategies/scalping/single-videos/collision_probe"
    assert single_payload["detail"]["has_source_captions"] is True
    assert single_payload["detail"]["has_source_audio"] is False
    assert playlist_payload["detail"]["asset_id"] == "strategies/scalping/playlists/collision_probe"
    assert playlist_payload["detail"]["has_source_audio"] is True
    assert playlist_payload["detail"]["has_source_captions"] is False
