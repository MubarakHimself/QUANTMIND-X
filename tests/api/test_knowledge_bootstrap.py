import json

from src.api.knowledge_bootstrap import bootstrap_reference_books


def test_bootstrap_reference_books_mirrors_repo_guides_into_knowledge_books(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "mql5.pdf").write_bytes(b"%PDF-1.4\nmql5")
    (repo_root / "mql5book.pdf").write_bytes(b"%PDF-1.4\nmql5book")

    knowledge_dir = tmp_path / "knowledge"
    monkeypatch.setenv("QMX_BOOTSTRAP_REFERENCE_BOOKS", "1")

    synced = bootstrap_reference_books(repo_root=repo_root, knowledge_dir=knowledge_dir)

    assert {path.name for path in synced} == {"mql5.pdf", "mql5book.pdf"}

    books_dir = knowledge_dir / "books"
    assert (books_dir / "mql5.pdf").exists()
    assert (books_dir / "mql5book.pdf").exists()

    mql5_meta = json.loads((books_dir / "mql5.pdf.metadata.json").read_text(encoding="utf-8"))
    assert mql5_meta["title"] == "MQL5 Reference"
    assert mql5_meta["author"] == "MetaQuotes"
    assert "mql5" in mql5_meta["topics"]
    assert mql5_meta["category"] == "Books"
    assert mql5_meta["bootstrap_source"] == "repo-reference-book"
