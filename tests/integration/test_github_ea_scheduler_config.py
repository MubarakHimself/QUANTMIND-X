from src.integrations.github_ea_scheduler import is_scheduler_configured


def test_scheduler_not_configured_without_repo_url(monkeypatch):
    monkeypatch.delenv("GITHUB_EA_REPO_URL", raising=False)
    assert is_scheduler_configured() is False


def test_scheduler_not_configured_for_placeholder_repo_url(monkeypatch):
    monkeypatch.setenv("GITHUB_EA_REPO_URL", "https://github.com/your-org/your-ea-repo")
    assert is_scheduler_configured() is False


def test_scheduler_configured_for_real_repo_url(monkeypatch):
    monkeypatch.setenv("GITHUB_EA_REPO_URL", "https://github.com/acme/quantmind-eas")
    assert is_scheduler_configured() is True
