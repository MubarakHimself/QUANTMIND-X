import asyncio
import json

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

        update_event = await anext(stream)
        assert '"task_id": "todo-1"' in update_event
        assert '"status": "DONE"' in update_event

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
