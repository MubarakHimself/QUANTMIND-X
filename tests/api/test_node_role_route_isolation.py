import importlib
import os


def _load_server_with_role(role: str):
    os.environ["NODE_ROLE"] = role
    from src.api import server

    return importlib.reload(server)


def _paths_for_role(role: str) -> set[str]:
    server = _load_server_with_role(role)
    return {
        getattr(route, "path", "")
        for route in server.app.routes
        if getattr(route, "path", "")
    }


def test_node_trading_excludes_backend_management_routes():
    paths = _paths_for_role("node_trading")

    assert "/api/copilot/chat" not in paths
    assert "/api/floor-manager/chat" not in paths
    assert not any(path.startswith("/api/settings") for path in paths)
    assert not any(path.startswith("/api/providers") for path in paths)
    assert "/api/trading/bots" in paths


def test_node_backend_keeps_backend_management_routes():
    paths = _paths_for_role("node_backend")

    assert "/api/copilot/chat" in paths
    assert "/api/floor-manager/chat" in paths
    assert any(path.startswith("/api/settings") for path in paths)
    assert any(path.startswith("/api/providers") for path in paths)
    assert "/api/trading/bots" not in paths
