from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api import graph_memory_endpoints
from src.api.server import app
from src.memory.graph.facade import get_graph_memory
from src.memory.graph.types import MemoryCategory, MemoryNode, MemoryNodeType, MemoryTier


def test_hot_nodes_endpoint_returns_recent_nodes(tmp_path, monkeypatch):
    db_path = tmp_path / "graph_memory.db"
    facade = get_graph_memory(db_path=str(db_path))
    monkeypatch.setattr(graph_memory_endpoints, "_facade", facade)

    node = MemoryNode(
        title="Recent opinion",
        content="Recent content",
        node_type=MemoryNodeType.OPINION,
        category=MemoryCategory.SUBJECTIVE,
        tier=MemoryTier.HOT,
        created_at_utc=datetime.now(timezone.utc),
        updated_at_utc=datetime.now(timezone.utc),
    )
    facade.store.create_node(node)

    client = TestClient(app)
    response = client.get("/api/graph-memory/nodes/hot", params={"limit": 5})

    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["id"] == str(node.id)
    assert data[0]["tier"] == "hot"
