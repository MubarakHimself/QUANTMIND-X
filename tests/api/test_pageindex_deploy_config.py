from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_compose() -> dict:
    compose_path = PROJECT_ROOT / "docker-compose.production.yml"
    with compose_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_quantmind_api_does_not_hard_depend_on_pageindex_services() -> None:
    compose = _load_compose()
    depends_on = compose["services"]["quantmind-api"].get("depends_on", [])

    assert "pageindex-articles" not in depends_on
    assert "pageindex-books" not in depends_on
    assert "pageindex-logs" not in depends_on


def test_pageindex_services_are_opt_in_profile_only() -> None:
    compose = _load_compose()

    for service_name in ("pageindex-articles", "pageindex-books", "pageindex-logs"):
        service = compose["services"][service_name]
        assert service.get("profiles") == ["pageindex"]


def test_default_pageindex_mcp_package_matches_official_package_name() -> None:
    from src.api.mcp_endpoints import DEFAULT_MCP_SERVERS
    from src.agents.mcp.discovery import MCPServerDiscovery
    from src.agents.tools.mcp.manager import MCPClientManager

    pageindex_default = next(
        server for server in DEFAULT_MCP_SERVERS if server["server_id"] == "pageindex"
    )

    assert pageindex_default["args"] == ["-y", "@pageindex/mcp"]
    assert "@pageindex/mcp" in MCPServerDiscovery.KNOWN_MCP_PACKAGES
    assert "@pageindex/mcp-server" not in MCPServerDiscovery.KNOWN_MCP_PACKAGES
    assert MCPClientManager()._get_stdio_command("pageindex", {"type": "pageindex"}) == (
        "npx",
        ["-y", "@pageindex/mcp"],
    )


def test_server_dockerfile_does_not_install_nonexistent_prometheus_remote_write() -> None:
    dockerfile_path = PROJECT_ROOT / "server" / "Dockerfile"
    content = dockerfile_path.read_text(encoding="utf-8")

    assert "prometheus-remote-write" not in content
    assert "python-snappy" in content


def test_deploy_workflows_use_targeted_contabo_service_bringup() -> None:
    deploy_workflow = (PROJECT_ROOT / ".github" / "workflows" / "deploy-contabo.yml").read_text(encoding="utf-8")
    rollback_workflow = (PROJECT_ROOT / ".github" / "workflows" / "rollback.yml").read_text(encoding="utf-8")

    expected_services = "redis quantmind-api prefect-server prefect-worker hmm-inference-api hmm-scheduler"

    assert expected_services in deploy_workflow
    assert expected_services in rollback_workflow
    assert "--remove-orphans" not in deploy_workflow
    assert "--remove-orphans" not in rollback_workflow
