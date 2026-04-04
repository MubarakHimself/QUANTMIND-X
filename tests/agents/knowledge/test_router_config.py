from src.agents.knowledge.router import PageIndexClient


def test_pageindex_client_defaults_to_container_service_urls(monkeypatch):
    monkeypatch.delenv("PAGEINDEX_ARTICLES_URL", raising=False)
    monkeypatch.delenv("PAGEINDEX_BOOKS_URL", raising=False)
    monkeypatch.delenv("PAGEINDEX_LOGS_URL", raising=False)

    client = PageIndexClient()

    assert client.articles_url == "http://pageindex-articles:3000"
    assert client.books_url == "http://pageindex-books:3001"
    assert client.logs_url == "http://pageindex-logs:3002"


def test_pageindex_client_prefers_env_urls(monkeypatch):
    monkeypatch.setenv("PAGEINDEX_ARTICLES_URL", "https://articles.quantmindx.local")
    monkeypatch.setenv("PAGEINDEX_BOOKS_URL", "https://books.quantmindx.local")
    monkeypatch.setenv("PAGEINDEX_LOGS_URL", "https://logs.quantmindx.local")

    client = PageIndexClient()

    assert client.articles_url == "https://articles.quantmindx.local"
    assert client.books_url == "https://books.quantmindx.local"
    assert client.logs_url == "https://logs.quantmindx.local"
