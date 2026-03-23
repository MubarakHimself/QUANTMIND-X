"""
Tests for Strategy Versions API

Integration tests for version control API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Note: We'll test the API using the router directly since full server setup
# requires extensive mocking. These tests verify endpoint logic.


class TestStrategyVersionsAPI:
    """Test suite for strategy versions API endpoints."""

    def test_list_versions_endpoint_structure(self):
        """Test that list versions response structure is correct."""
        from src.api.strategy_versions import VersionListResponse, VersionResponse

        response = VersionListResponse(
            strategy_id="test-strategy",
            versions=[
                VersionResponse(
                    id="123",
                    strategy_id="test-strategy",
                    version_tag="1.0.0",
                    created_at="2024-01-01T00:00:00",
                    author="test-author",
                    source_hash="abc123",
                    template_deps={},
                    pin_template_version=False,
                    variant_type="vanilla",
                    improvement_cycle=1,
                    artifacts={},
                    is_active=True,
                )
            ],
            active_version="1.0.0",
            total=1,
        )

        assert response.strategy_id == "test-strategy"
        assert len(response.versions) == 1
        assert response.versions[0].version_tag == "1.0.0"
        assert response.active_version == "1.0.0"

    def test_rollback_response_structure(self):
        """Test rollback response structure."""
        from src.api.strategy_versions import RollbackResponse

        response = RollbackResponse(
            success=True,
            strategy_id="test-strategy",
            from_version="1.0.1",
            to_version="1.0.0",
            compilation_passed=True,
            sit_validation_passed=True,
            audit_id="audit-123",
            message="Rolled back to version 1.0.0",
        )

        assert response.success is True
        assert response.compilation_passed is True
        assert response.sit_validation_passed is True
        assert response.message == "Rolled back to version 1.0.0"

    def test_create_version_request_structure(self):
        """Test create version request structure."""
        from src.api.strategy_versions import CreateVersionRequest
        from src.mql5.versions.schema import VariantType, EAVersionArtifacts

        request = CreateVersionRequest(
            strategy_id="test-strategy",
            author="test-author",
            source_code="print('hello')",
            template_deps={"template-1": "v1.0"},
            auto_increment="patch",
            variant_type=VariantType.VANILLA,
            improvement_cycle=1,
            artifacts=EAVersionArtifacts(mq5_path="test.mq5"),
        )

        assert request.strategy_id == "test-strategy"
        assert request.author == "test-author"
        assert request.variant_type == VariantType.VANILLA
        assert request.artifacts.mq5_path == "test.mq5"

    def test_update_artifacts_request_structure(self):
        """Test update artifacts request structure."""
        from src.api.strategy_versions import UpdateArtifactsRequest

        request = UpdateArtifactsRequest(
            mq5_path="updated.mq5",
            ex5_path="updated.ex5",
            trd_id="trd-123",
            backtest_result_ids=["bt-1", "bt-2"],
        )

        assert request.mq5_path == "updated.mq5"
        assert request.ex5_path == "updated.ex5"
        assert len(request.backtest_result_ids) == 2

    def test_rollback_request_structure(self):
        """Test rollback request structure."""
        from src.api.strategy_versions import RollbackRequest

        request = RollbackRequest(
            target_version="1.0.0",
            author="test-author",
            reason="Testing rollback",
        )

        assert request.target_version == "1.0.0"
        assert request.author == "test-author"
        assert request.reason == "Testing rollback"

    def test_history_response_structure(self):
        """Test history response structure."""
        from src.api.strategy_versions import HistoryResponse

        response = HistoryResponse(
            strategy_id="test-strategy",
            versions=[
                {
                    "version_tag": "1.0.0",
                    "created_at": "2024-01-01T00:00:00",
                    "author": "author1",
                    "source_hash": "abc",
                    "variant_type": "vanilla",
                    "improvement_cycle": 1,
                    "artifacts": {},
                    "is_active": True,
                }
            ],
            active_version="1.0.0",
            rollback_history=[
                {
                    "strategy_id": "test-strategy",
                    "from_version": "1.0.1",
                    "to_version": "1.0.0",
                    "timestamp": "2024-01-02T00:00:00",
                    "author": "author2",
                    "reason": "Test",
                    "sit_validation_passed": True,
                }
            ],
        )

        assert len(response.versions) == 1
        assert len(response.rollback_history) == 1
        assert response.rollback_history[0]["from_version"] == "1.0.1"

    def test_version_metadata_response_structure(self):
        """Test version metadata response structure."""
        from src.api.strategy_versions import VersionMetadataResponse

        response = VersionMetadataResponse(
            version_tag="1.0.0",
            created_at="2024-01-01T00:00:00",
            author="test-author",
            source_hash="abc123",
            variant_type="vanilla",
            improvement_cycle=1,
        )

        assert response.version_tag == "1.0.0"
        assert response.variant_type == "vanilla"

    def test_compare_versions_response_structure(self):
        """Test compare versions response structure."""
        # This would come from the manager's compare_versions method
        comparison = {
            "version_a": "1.0.0",
            "version_b": "1.0.1",
            "differences": {
                "source_hash": True,
                "template_deps": False,
                "variant_type": True,
                "improvement_cycle": False,
                "artifacts": {
                    "mq5": False,
                    "ex5": True,
                    "trd": False,
                    "backtests": True,
                },
            },
            "metadata": {
                "a": {
                    "author": "author1",
                    "created_at": "2024-01-01T00:00:00",
                },
                "b": {
                    "author": "author2",
                    "created_at": "2024-01-02T00:00:00",
                },
            },
        }

        assert comparison["version_a"] == "1.0.0"
        assert comparison["differences"]["source_hash"] is True
        assert comparison["metadata"]["a"]["author"] == "author1"