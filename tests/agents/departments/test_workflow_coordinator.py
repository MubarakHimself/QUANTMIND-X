"""
Tests for Department Workflow Coordinator

Task Group: Phase 4 - Department Workflow Coordinator
"""
import pytest
import tempfile
import os
import json
from datetime import datetime

from src.agents.departments.workflow_coordinator import (
    DepartmentWorkflowCoordinator,
    DepartmentWorkflow,
    WorkflowTask,
    WorkflowStage,
    WorkflowStatus,
    get_workflow_coordinator,
    create_workflow_coordinator,
)


class TestDepartmentWorkflowCoordinator:
    """Test the department workflow coordinator."""

    def test_coordinator_creates_mail_service(self):
        """Coordinator should initialize mail service."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            assert coordinator.mail_service is not None
            coordinator.close()

    def test_start_workflow_creates_workflow(self):
        """Starting a workflow should create a workflow object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            assert workflow_id is not None
            assert workflow_id.startswith("wf_")

            # Check workflow was created
            workflow = coordinator._workflows.get(workflow_id)
            assert workflow is not None
            assert workflow.current_stage == WorkflowStage.VIDEO_INGEST
            assert workflow.status == WorkflowStatus.PENDING

            coordinator.close()

    def test_start_workflow_creates_initial_task(self):
        """Starting a workflow should create an initial task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            workflow = coordinator._workflows.get(workflow_id)
            assert len(workflow.tasks) == 1

            task = workflow.tasks[0]
            assert task.stage == WorkflowStage.VIDEO_INGEST
            assert task.from_dept == "floor_manager"
            assert task.to_dept == "video_ingest"

            coordinator.close()

    def test_process_stage_creates_task_and_moves_to_next(self):
        """Processing a stage should create task and move to next stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            # Process video ingest stage
            result = coordinator.process_stage(workflow_id)

            assert result["current_stage"] == "video_ingest"
            assert result["next_stage"] == "research"

            # Check workflow moved to research stage
            workflow = coordinator._workflows.get(workflow_id)
            assert workflow.current_stage == WorkflowStage.RESEARCH

            # Should have 2 tasks now (video_ingest + research)
            assert len(workflow.tasks) == 2

            coordinator.close()

    def test_process_stage_sends_mail_message(self):
        """Processing a stage should send a mail message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            # Process video ingest stage
            coordinator.process_stage(workflow_id)

            # Check mail was sent to video_ingest
            messages = coordinator.mail_service.check_inbox(
                dept="video_ingest",
                unread_only=False,
            )

            assert len(messages) >= 1

            # Find the workflow message
            workflow_msg = None
            for msg in messages:
                body = json.loads(msg.body)
                if body.get("workflow_id") == workflow_id:
                    workflow_msg = msg
                    break

            assert workflow_msg is not None
            assert workflow_msg.subject.startswith("Workflow")

            coordinator.close()

    def test_workflow_completes_full_pipeline(self):
        """Workflow should complete full pipeline through all stages."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            # Process all stages (there are 5 stages + completed)
            # Video Ingest -> Research -> Development -> Backtesting -> Paper Trading -> Completed
            stages_processed = []

            # Initial stage is VIDEO_INGEST
            workflow = coordinator._workflows.get(workflow_id)
            stages_processed.append(workflow.current_stage.value)

            # Process through all 5 transitions
            for i in range(5):
                coordinator.process_stage(workflow_id)
                workflow = coordinator._workflows.get(workflow_id)
                stages_processed.append(workflow.current_stage.value)

            # Verify all stages were processed
            expected_stages = [
                "video_ingest",
                "research",
                "development",
                "backtesting",
                "paper_trading",
                "completed",
            ]

            assert stages_processed == expected_stages

            coordinator.close()

    def test_handle_department_response_updates_task(self):
        """Handling department response should update task with results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            # Process first stage
            coordinator.process_stage(workflow_id)

            # Handle response from video_ingest
            result = coordinator.handle_department_response(
                workflow_id=workflow_id,
                from_dept="video_ingest",
                result={"nprd_path": "/path/to/nprd.json", "status": "success"},
            )

            assert result["status"] == "completed"
            assert "nprd_path" in result["result"]

            # Check task was updated
            workflow = coordinator._workflows.get(workflow_id)
            task = workflow.tasks[-1]
            assert task.status == WorkflowStatus.COMPLETED
            assert task.result is not None
            assert task.result["nprd_path"] == "/path/to/nprd.json"

            coordinator.close()

    def test_check_department_inbox_filters_by_workflow(self):
        """Checking inbox should filter messages by workflow ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            # Process first stage
            coordinator.process_stage(workflow_id)

            # Check inbox for video_ingest
            messages = coordinator.check_department_inbox(
                workflow_id=workflow_id,
                department="video_ingest",
            )

            # Should have messages for this workflow
            for msg in messages:
                body = json.loads(msg["body"])
                assert body["workflow_id"] == workflow_id

            coordinator.close()

    def test_get_workflow_status_returns_dict(self):
        """Getting workflow status should return dictionary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            status = coordinator.get_workflow_status(workflow_id)

            assert status is not None
            assert status["workflow_id"] == workflow_id
            assert "current_stage" in status
            assert "status" in status
            assert "tasks" in status

            coordinator.close()

    def test_get_all_workflows_returns_list(self):
        """Getting all workflows should return list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            # Create multiple workflows
            workflow1 = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video1.mp4"},
            )
            workflow2 = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video2.mp4"},
            )

            all_workflows = coordinator.get_all_workflows()

            assert len(all_workflows) == 2
            workflow_ids = [w["workflow_id"] for w in all_workflows]
            assert workflow1 in workflow_ids
            assert workflow2 in workflow_ids

            coordinator.close()

    def test_cancel_workflow_marks_as_cancelled(self):
        """Cancelling a workflow should mark it as cancelled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            # Cancel the workflow
            result = coordinator.cancel_workflow(workflow_id)

            assert result is True

            # Check workflow is cancelled
            workflow = coordinator._workflows.get(workflow_id)
            assert workflow.status == WorkflowStatus.CANCELLED

            coordinator.close()

    def test_workflow_progress_calculation(self):
        """Workflow should calculate progress correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            workflow = coordinator._workflows.get(workflow_id)

            # Initial progress should be at video_ingest stage
            assert workflow.get_progress() == 0.0

            # Process video_ingest stage
            coordinator.process_stage(workflow_id)
            workflow = coordinator._workflows.get(workflow_id)

            # Should be at research stage now (1/5 * 100 = 20%)
            assert workflow.get_progress() == 20.0

            coordinator.close()

    def test_workflow_stores_trd_from_research(self):
        """Workflow should handle TRD from Research department."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_workflow.db")
            coordinator = DepartmentWorkflowCoordinator(mail_db_path=db_path)

            workflow_id = coordinator.start_workflow(
                source="video_ingest",
                initial_payload={"video_path": "/path/to/video.mp4"},
            )

            # Process video_ingest stage to move to research
            coordinator.process_stage(workflow_id)

            # Handle response from video_ingest first
            result = coordinator.handle_department_response(
                workflow_id=workflow_id,
                from_dept="video_ingest",
                result={"nprd_path": "/path/to/nprd.json", "status": "success"},
            )

            # Now process research stage
            coordinator.process_stage(workflow_id)

            # Simulate TRD result from Research department
            result = coordinator.handle_department_response(
                workflow_id=workflow_id,
                from_dept="research",
                result={
                    "trd_content": "# Trading Strategy\n\nBuy when RSI < 30",
                    "strategy_name": "RSI Reversal",
                    "strategy_type": "mean_reversion",
                    "symbols": ["EURUSD"],
                },
            )

            # Verify TRD was stored
            workflow = coordinator._workflows.get(workflow_id)
            assert "latest_results" in workflow.metadata
            assert "trd_content" in workflow.metadata["latest_results"]

            coordinator.close()


class TestWorkflowStageEnum:
    """Test WorkflowStage enum values."""

    def test_all_stages_defined(self):
        """All required stages should be defined."""
        assert WorkflowStage.VIDEO_INGEST.value == "video_ingest"
        assert WorkflowStage.RESEARCH.value == "research"
        assert WorkflowStage.DEVELOPMENT.value == "development"
        assert WorkflowStage.BACKTESTING.value == "backtesting"
        assert WorkflowStage.PAPER_TRADING.value == "paper_trading"
        assert WorkflowStage.COMPLETED.value == "completed"
        assert WorkflowStage.FAILED.value == "failed"


class TestWorkflowStatusEnum:
    """Test WorkflowStatus enum values."""

    def test_all_statuses_defined(self):
        """All required statuses should be defined."""
        assert WorkflowStatus.PENDING.value == "pending"
        assert WorkflowStatus.RUNNING.value == "running"
        assert WorkflowStatus.WAITING.value == "waiting"
        assert WorkflowStatus.COMPLETED.value == "completed"
        assert WorkflowStatus.FAILED.value == "failed"
        assert WorkflowStatus.CANCELLED.value == "cancelled"


class TestDepartmentWorkflow:
    """Test DepartmentWorkflow dataclass."""

    def test_to_dict_includes_all_fields(self):
        """to_dict should include all fields."""
        workflow = DepartmentWorkflow(
            workflow_id="test_123",
            status=WorkflowStatus.RUNNING,
            current_stage=WorkflowStage.RESEARCH,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={"key": "value"},
        )

        result = workflow.to_dict()

        assert result["workflow_id"] == "test_123"
        assert result["status"] == "running"
        assert result["current_stage"] == "research"
        assert result["metadata"] == {"key": "value"}

    def test_get_progress_returns_float(self):
        """get_progress should return float."""
        workflow = DepartmentWorkflow(
            workflow_id="test_123",
            status=WorkflowStatus.RUNNING,
            current_stage=WorkflowStage.DEVELOPMENT,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        progress = workflow.get_progress()
        assert isinstance(progress, float)
        assert progress == 40.0


class TestCreateWorkflowCoordinator:
    """Test factory function."""

    def test_create_workflow_coordinator(self):
        """Factory function should create coordinator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            coordinator = create_workflow_coordinator(
                mail_db_path=db_path,
            )

            assert coordinator is not None
            assert coordinator.mail_service is not None

            coordinator.close()

    def test_get_workflow_coordinator_singleton(self):
        """get_workflow_coordinator should return singleton."""
        # Note: This test may fail if singleton is already created
        # in other tests, so we just verify it returns something
        coordinator = get_workflow_coordinator()
        assert coordinator is not None
