"""
End-to-End Workflow Integration Tests

Tests the complete workflow: Video Ingest -> TRD -> EA -> Backtest -> Paper Trading.
"""
import pytest
from unittest.mock import MagicMock, patch

from src.agents.departments.workflow_coordinator import (
    DepartmentWorkflowCoordinator,
    WorkflowStage,
    WorkflowStatus,
)


@pytest.fixture
def coordinator():
    """Create a workflow coordinator for testing."""
    with patch('src.agents.departments.workflow_coordinator.DepartmentMailService'):
        coord = DepartmentWorkflowCoordinator()
        yield coord
        coord.close()


def test_full_video_to_paper_trading_workflow(coordinator):
    """
    Test complete workflow: Video -> TRD -> EA -> Backtest -> Paper Trading.

    This is the main integration test.
    """
    # Start with video ingest
    video_url = "https://youtube.com/watch?v=test123"
    workflow_id = coordinator.start_workflow(
        source="video_ingest",
        initial_payload={"video_url": video_url, "title": "Test Strategy"},
    )

    assert workflow_id is not None

    # Get initial workflow status
    status = coordinator.get_workflow_status(workflow_id)
    assert status is not None
    assert status["current_stage"] == WorkflowStage.VIDEO_INGEST.value
    assert status["status"] == WorkflowStatus.PENDING.value

    # Simulate TRD generation (would normally come from Research department)
    trd_variants = {
        "vanilla": {
            "strategy_name": "TestVanilla",
            "entry_rules": ["rule1", "rule2"],
            "exit_rules": ["exit1"],
        },
        "spiced": {
            "strategy_name": "TestSpiced",
            "entry_rules": ["rule1", "rule2"],
            "exit_rules": ["exit1"],
            "articles": [
                {"title": "Article 1", "url": "https://example.com/1"},
                {"title": "Article 2", "url": "https://example.com/2"},
            ],
        },
    }

    # Update workflow with TRD results
    workflow = coordinator._workflows[workflow_id]
    workflow.metadata["latest_results"] = {"trd_variants": trd_variants}
    workflow.current_stage = WorkflowStage.RESEARCH

    # Advance to development stage
    with patch.object(coordinator.mail_service, 'send') as mock_send:
        mock_send.return_value = MagicMock(message_id="msg_123")
        coordinator.process_stage(workflow_id)

    # Verify workflow advanced to development
    status = coordinator.get_workflow_status(workflow_id)
    assert status["current_stage"] == WorkflowStage.DEVELOPMENT.value

    # Simulate EA creation
    ea_variants = {
        "vanilla": {
            "ea_name": "TestVanillaEA",
            "trd_ref": "TestVanilla",
            "mq5_file": "TestVanillaEA.mq5",
        },
        "spiced": {
            "ea_name": "TestSpicedEA",
            "trd_ref": "TestSpiced",
            "mq5_file": "TestSpicedEA.mq5",
        },
    }

    workflow.metadata["latest_results"] = {"ea_variants": ea_variants}
    workflow.current_stage = WorkflowStage.DEVELOPMENT

    # Advance to backtesting stage
    with patch.object(coordinator.mail_service, 'send') as mock_send:
        mock_send.return_value = MagicMock(message_id="msg_124")
        coordinator.process_stage(workflow_id)

    # Verify workflow advanced to backtesting
    status = coordinator.get_workflow_status(workflow_id)
    assert status["current_stage"] == WorkflowStage.BACKTESTING.value

    # Simulate backtest results
    backtest_results = {
        "vanilla": {"profit_factor": 1.5, "max_drawdown": 10.2},
        "spiced": {"profit_factor": 1.8, "max_drawdown": 8.5},
    }

    workflow.metadata["latest_results"] = {"backtest_results": backtest_results}
    workflow.current_stage = WorkflowStage.BACKTESTING

    # Advance to paper trading stage
    with patch.object(coordinator.mail_service, 'send') as mock_send:
        mock_send.return_value = MagicMock(message_id="msg_125")
        coordinator.process_stage(workflow_id)

    # Verify workflow advanced to paper trading
    status = coordinator.get_workflow_status(workflow_id)
    assert status["current_stage"] == WorkflowStage.PAPER_TRADING.value


def test_workflow_tracks_variant_history(coordinator):
    """
    Test that workflow maintains history of both vanilla and spiced variants.
    """
    workflow_id = coordinator.start_workflow(
        source="video_ingest",
        initial_payload={"video_url": "https://youtube.com/watch?v=abc"},
    )

    # Simulate TRD with both variants
    trd_variants = {
        "vanilla": {"strategy_name": "VanillaStrategy", "entry_rules": ["r1"]},
        "spiced": {"strategy_name": "SpicedStrategy", "entry_rules": ["r1"], "articles": []},
    }

    workflow = coordinator._workflows[workflow_id]
    workflow.metadata["latest_results"] = {"trd_variants": trd_variants}
    workflow.current_stage = WorkflowStage.RESEARCH

    # Verify both variants are tracked
    assert "trd_variants" in workflow.metadata["latest_results"]
    assert "vanilla" in workflow.metadata["latest_results"]["trd_variants"]
    assert "spiced" in workflow.metadata["latest_results"]["trd_variants"]


def test_workflow_progress_calculation(coordinator):
    """
    Test that workflow progress is calculated correctly across stages.
    """
    workflow_id = coordinator.start_workflow(
        source="video_ingest",
        initial_payload={"video_url": "https://youtube.com/watch?v=xyz"},
    )

    # Video ingest = 0% (just started)
    progress = coordinator._workflows[workflow_id].get_progress()
    assert progress == 0.0

    # Advance through stages
    # Stage order: VIDEO_INGEST(0), RESEARCH(1), DEVELOPMENT(2), BACKTESTING(3), PAPER_TRADING(4), COMPLETED(5)
    # Progress formula: index / 5 * 100
    expected_progress = [
        (1 / 5) * 100,  # RESEARCH = 20%
        (2 / 5) * 100,  # DEVELOPMENT = 40%
        (3 / 5) * 100,  # BACKTESTING = 60%
        (4 / 5) * 100,  # PAPER_TRADING = 80%
    ]

    stages = [
        WorkflowStage.RESEARCH,
        WorkflowStage.DEVELOPMENT,
        WorkflowStage.BACKTESTING,
        WorkflowStage.PAPER_TRADING,
    ]

    for i, stage in enumerate(stages):
        coordinator._workflows[workflow_id].current_stage = stage
        coordinator._workflows[workflow_id].status = WorkflowStatus.RUNNING
        progress = coordinator._workflows[workflow_id].get_progress()
        assert abs(progress - expected_progress[i]) < 1.0


def test_workflow_cancellation(coordinator):
    """
    Test that workflow can be cancelled at any stage.
    """
    workflow_id = coordinator.start_workflow(
        source="video_ingest",
        initial_payload={"video_url": "https://youtube.com/watch?v=cancel"},
    )

    # Cancel the workflow
    result = coordinator.cancel_workflow(workflow_id)
    assert result is True

    # Verify status is cancelled
    status = coordinator.get_workflow_status(workflow_id)
    assert status["status"] == WorkflowStatus.CANCELLED.value


def test_get_all_workflows(coordinator):
    """
    Test listing all workflows.
    """
    # Create multiple workflows
    wf1 = coordinator.start_workflow(
        source="video_ingest",
        initial_payload={"video_url": "https://youtube.com/watch?v=1"},
    )
    wf2 = coordinator.start_workflow(
        source="video_ingest",
        initial_payload={"video_url": "https://youtube.com/watch?v=2"},
    )

    # Get all workflows
    workflows = coordinator.get_all_workflows()
    assert len(workflows) >= 2
    workflow_ids = [w["workflow_id"] for w in workflows]
    assert wf1 in workflow_ids
    assert wf2 in workflow_ids
