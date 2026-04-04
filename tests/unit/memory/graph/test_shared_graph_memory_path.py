from pathlib import Path

from src.agents.departments.heads.research_head import Hypothesis, ResearchHead
from src.memory.graph import facade as facade_module
from src.memory.graph.types import MemoryNodeType


def reset_graph_memory_singleton() -> None:
    facade_module._facade_instance = None
    facade_module._facade_db_path = None


def test_graph_memory_facade_uses_shared_env_db_path(monkeypatch, tmp_path):
    db_path = tmp_path / "graph_memory.db"
    monkeypatch.setenv("GRAPH_MEMORY_DB", str(db_path))
    reset_graph_memory_singleton()

    facade = facade_module.get_graph_memory()

    assert facade.db_path == db_path
    assert db_path.exists()


def test_research_opinion_writes_to_shared_graph_memory_db(monkeypatch, tmp_path):
    db_path = tmp_path / "graph_memory.db"
    monkeypatch.setenv("GRAPH_MEMORY_DB", str(db_path))
    reset_graph_memory_singleton()

    head = ResearchHead()
    head._current_session_id = "shared-db-test"
    hypothesis = Hypothesis(
        symbol="EURUSD",
        timeframe="H4",
        hypothesis="Research opinion should land in the shared graph memory database.",
        supporting_evidence=["test evidence"],
        confidence_score=0.81,
        recommended_next_steps=["verify workshop graph memory"],
    )

    head._write_research_opinion(hypothesis, "test shared graph db")

    facade = facade_module.get_graph_memory()
    opinions = facade.store.query_nodes(node_type=MemoryNodeType.OPINION, limit=20)

    assert any(
        node.content == "Research opinion should land in the shared graph memory database."
        and node.session_id == "shared-db-test"
        for node in opinions
    )
