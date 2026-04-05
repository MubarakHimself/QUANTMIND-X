import asyncio
import json

import pytest
from fastapi import HTTPException

from src.api import task_sse_endpoints


class _FakePubSub:
    def __init__(self, message=None):
        self.message = message
        self.closed = False
        self.unsubscribed = []

    def subscribe(self, _channel):
        return None

    def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        assert ignore_subscribe_messages is True
        assert timeout == 1.0
        message = self.message
        self.message = None
        return message

    def listen(self):
        raise AssertionError("pubsub.listen() must not be used in async SSE streams")

    def unsubscribe(self, channel):
        self.unsubscribed.append(channel)

    def close(self):
        self.closed = True


class _FakeRedisClient:
    def __init__(self, pubsub):
        self._pubsub = pubsub

    def ping(self):
        return True

    def pubsub(self):
        return self._pubsub


def test_task_sse_redis_stream_uses_non_blocking_get_message(monkeypatch):
    pubsub = _FakePubSub(
        {
            "type": "message",
            "data": json.dumps({"task_id": "todo-1", "status": "DONE"}),
        }
    )
    redis_client = _FakeRedisClient(pubsub)

    monkeypatch.setattr(task_sse_endpoints, "_get_department_tasks", lambda department: [])

    import sys
    import types

    fake_redis_module = types.ModuleType("redis")
    fake_redis_module.Redis = lambda **kwargs: redis_client
    sys.modules["redis"] = fake_redis_module

    async def run_test():
        stream = task_sse_endpoints._stream_task_updates("research")

        initial_event = await anext(stream)
        assert '"type": "initial"' in initial_event

        await stream.aclose()

    asyncio.run(run_test())

    assert pubsub.unsubscribed == ["task:dept:research:updates"]
    assert pubsub.closed is True


def test_map_todo_to_task_preserves_provenance_fields():
    task = task_sse_endpoints._map_todo_to_task(
        {
            "id": "todo_123",
            "title": "Review live risk routing",
            "description": "Validate router state and MT5 availability before deploy.",
            "department": "risk",
            "priority": "high",
            "status": "in_progress",
            "source_dept": "floor_manager",
            "message_type": "dispatch",
            "workflow_id": "wf_alpha",
            "kanban_card_id": "kb_987",
            "mail_message_id": "msg_555",
            "created_at": "2026-04-01T20:00:00Z",
            "updated_at": "2026-04-01T21:00:00Z",
        }
    )

    assert task["description"] == "Validate router state and MT5 availability before deploy."
    assert task["source_dept"] == "floor_manager"
    assert task["message_type"] == "dispatch"
    assert task["workflow_id"] == "wf_alpha"
    assert task["kanban_card_id"] == "kb_987"
    assert task["mail_message_id"] == "msg_555"
    assert task["updated_at"] == "2026-04-01T21:00:00Z"
    assert task["started_at"] == "2026-04-01T21:00:00Z"


def test_get_department_tasks_includes_canonical_workflow_tasks(monkeypatch):
    class _FakeTaskManager:
        DEPARTMENTS = ["research", "development", "risk", "trading", "portfolio"]

        def get_todos(self, department):
            return []

    monkeypatch.setattr(task_sse_endpoints, "get_task_manager", lambda: _FakeTaskManager())
    monkeypatch.setattr(
        task_sse_endpoints,
        "_load_canonical_workflows",
        lambda: [
            {
                "id": "wf1_demo",
                "name": "WF1 Creation",
                "department": "Development",
                "state": "RUNNING",
                "strategy_id": "vwap_breakout",
                "current_stage": "Development",
                "next_step": "Backtesting",
                "updated_at": "2026-04-04T09:10:00Z",
                "started_at": "2026-04-04T09:00:00Z",
                "latest_artifact": {"path": "development/ea/VWAPBreakout.mq5"},
            }
        ],
    )

    tasks = task_sse_endpoints._get_department_tasks("development")

    assert len(tasks) == 1
    task = tasks[0]
    assert task["task_id"] == "workflow:wf1_demo:development"
    assert task["status"] == "IN_PROGRESS"
    assert task["workflow_id"] == "wf1_demo"
    assert task["strategy_id"] == "vwap_breakout"
    assert task["latest_artifact"]["path"] == "development/ea/VWAPBreakout.mq5"
    assert task["read_only"] is True


def test_get_department_tasks_merges_mail_task_with_workflow_context(monkeypatch):
    class _Todo:
        def to_dict(self):
            return {
                "id": "todo_123",
                "department": "development",
                "mail_message_id": "msg_123",
                "title": "Compile variant",
                "description": "Compile the latest variant and report back.",
                "priority": "high",
                "status": "in_progress",
                "source_dept": "research",
                "message_type": "dispatch",
                "workflow_id": "wf1_demo",
                "kanban_card_id": "kb_321",
                "created_at": "2026-04-04T08:55:00Z",
                "updated_at": "2026-04-04T09:11:00Z",
            }

    class _FakeTaskManager:
        DEPARTMENTS = ["research", "development", "risk", "trading", "portfolio"]

        def get_todos(self, department):
            return [_Todo()] if department == "development" else []

    monkeypatch.setattr(task_sse_endpoints, "get_task_manager", lambda: _FakeTaskManager())
    monkeypatch.setattr(
        task_sse_endpoints,
        "_load_canonical_workflows",
        lambda: [
            {
                "id": "wf1_demo",
                "name": "WF1 Creation",
                "department": "Development",
                "state": "RUNNING",
                "strategy_id": "vwap_breakout",
                "current_stage": "Development",
                "next_step": "Backtesting",
                "updated_at": "2026-04-04T09:10:00Z",
                "started_at": "2026-04-04T09:00:00Z",
                "blocking_error": None,
                "latest_artifact": {"path": "development/ea/VWAPBreakout.mq5"},
            }
        ],
    )

    tasks = task_sse_endpoints._get_department_tasks("development")

    assert len(tasks) == 1
    task = tasks[0]
    assert task["task_id"] == "todo_123"
    assert task["mail_message_id"] == "msg_123"
    assert task["kanban_card_id"] == "kb_321"
    assert task["workflow_id"] == "wf1_demo"
    assert task["latest_artifact"]["path"] == "development/ea/VWAPBreakout.mq5"
    assert task["source_kind"] == "workflow+mail"
    assert task["read_only"] is False


def test_get_department_tasks_decomposes_workflow_tasks_per_department(monkeypatch):
    class _FakeTaskManager:
        DEPARTMENTS = ["research", "development", "risk", "trading", "portfolio"]

        def get_todos(self, department):
            return []

    monkeypatch.setattr(task_sse_endpoints, "get_task_manager", lambda: _FakeTaskManager())
    monkeypatch.setattr(
        task_sse_endpoints,
        "_load_canonical_workflows",
        lambda: [
            {
                "id": "wf_live_1",
                "name": "WF1 Creation",
                "department": "Development",
                "state": "RUNNING",
                "strategy_id": "vwap_breakout",
                "current_stage": "Development",
                "next_step": "Backtesting",
                "updated_at": "2026-04-04T09:10:00Z",
                "started_at": "2026-04-04T09:00:00Z",
                "latest_artifact": {"path": "development/ea/VWAPBreakout.mq5"},
                "tasks": [
                    {
                        "task_id": "task_1",
                        "stage": "research",
                        "from_dept": "floor_manager",
                        "to_dept": "research",
                        "status": "completed",
                        "priority": "medium",
                        "created_at": "2026-04-04T09:00:00Z",
                        "completed_at": "2026-04-04T09:02:00Z",
                        "payload": {},
                    },
                    {
                        "task_id": "task_2",
                        "stage": "development",
                        "from_dept": "research",
                        "to_dept": "development",
                        "status": "running",
                        "priority": "high",
                        "created_at": "2026-04-04T09:03:00Z",
                        "payload": {},
                    },
                ],
            }
        ],
    )

    tasks = task_sse_endpoints._get_department_tasks("development")

    assert len(tasks) == 1
    task = tasks[0]
    assert task["task_id"] == "workflow:wf_live_1:development:task_2"
    assert task["workflow_id"] == "wf_live_1"
    assert task["source_dept"] == "research"
    assert task["status"] == "IN_PROGRESS"
    assert task["read_only"] is True
    assert task["source_kind"] == "workflow"


def test_get_department_tasks_matches_raw_coordinator_workflow_without_top_level_department(monkeypatch):
    class _FakeTaskManager:
        DEPARTMENTS = ["research", "development", "risk", "trading", "portfolio"]

        def get_todos(self, department):
            return []

    monkeypatch.setattr(task_sse_endpoints, "get_task_manager", lambda: _FakeTaskManager())
    monkeypatch.setattr(
        task_sse_endpoints,
        "_load_canonical_workflows",
        lambda: [
            {
                "workflow_id": "wf_live_raw",
                "workflow_type": "wf1_creation",
                "status": "running",
                "current_stage": "research",
                "updated_at": "2026-04-04T09:10:00Z",
                "created_at": "2026-04-04T09:00:00Z",
                "tasks": [
                    {
                        "task_id": "task_1",
                        "stage": "research",
                        "from_dept": "floor_manager",
                        "to_dept": "research",
                        "status": "running",
                        "priority": "medium",
                        "created_at": "2026-04-04T09:00:00Z",
                        "completed_at": None,
                        "payload": {},
                    }
                ],
            }
        ],
    )

    tasks = task_sse_endpoints._get_department_tasks("research")

    assert len(tasks) == 1
    task = tasks[0]
    assert task["workflow_id"] == "wf_live_raw"
    assert task["status"] == "IN_PROGRESS"
    assert task["department"] == "research"


def test_load_canonical_workflows_prefers_full_coordinator_payload(monkeypatch):
    coordinator_payload = [
        {
            "workflow_id": "wf_live_1",
            "workflow_type": "wf1_creation",
            "status": "running",
            "current_stage": "development",
            "created_at": "2026-04-04T09:00:00Z",
            "updated_at": "2026-04-04T09:10:00Z",
            "strategy_id": "vwap_breakout",
            "tasks": [
                {
                    "task_id": "task_2",
                    "stage": "development",
                    "from_dept": "research",
                    "to_dept": "development",
                    "status": "running",
                    "priority": "high",
                    "created_at": "2026-04-04T09:03:00Z",
                    "completed_at": None,
                    "payload": {},
                }
            ],
        }
    ]

    monkeypatch.setattr(
        "src.api.prefect_workflow_endpoints._load_db_workflows",
        lambda: {
            "wf_live_1": {
                "id": "wf_live_1",
                "name": "WF1 Creation",
                "department": "Development",
                "state": "RUNNING",
                "current_stage": "Development",
                "tasks": [{"id": "wf_live_1-task-1", "name": "Development", "state": "RUNNING"}],
            }
        },
    )
    monkeypatch.setattr(
        "src.api.prefect_workflow_endpoints._load_wf1_manifest_workflows",
        lambda: {},
    )

    class _FakeCoordinator:
        def get_workflow_status(self, workflow_id):
            if workflow_id == "wf_live_1":
                return coordinator_payload[0]
            return None

        def get_all_workflows(self):
            return coordinator_payload

    monkeypatch.setattr(
        "src.agents.departments.workflow_coordinator.get_workflow_coordinator",
        lambda: _FakeCoordinator(),
    )

    workflows = task_sse_endpoints._load_canonical_workflows()

    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["workflow_id"] == "wf_live_1"
    assert workflow["tasks"][0]["stage"] == "development"
    assert workflow["tasks"][0]["status"] == "running"
    assert workflow["tasks"][0]["to_dept"] == "development"


def test_update_task_status_rejects_workflow_derived_tasks():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            task_sse_endpoints.update_task_status(
                "workflow:wf1_demo:development",
                task_sse_endpoints.TaskStatusUpdate(
                    task_id="workflow:wf1_demo:development",
                    status="DONE",
                    timestamp="2026-04-04T09:30:00Z",
                ),
            )
        )

    assert exc_info.value.status_code == 409
    assert "read-only" in exc_info.value.detail
