"""
Tests for MQL5 Compile API Endpoints

Tests model structures and endpoint request/response logic.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestCompileModels:
    """Test compile request/response model structures."""

    def test_compile_request_required_fields(self):
        from src.api.compile_endpoints import CompileRequest

        req = CompileRequest(strategy_id="strat-001")
        assert req.strategy_id == "strat-001"
        assert req.version is None

    def test_compile_request_with_version(self):
        from src.api.compile_endpoints import CompileRequest

        req = CompileRequest(strategy_id="strat-002", version=3)
        assert req.strategy_id == "strat-002"
        assert req.version == 3

    def test_compile_response_success(self):
        from src.api.compile_endpoints import CompileResponse

        resp = CompileResponse(
            success=True,
            strategy_id="strat-001",
            version=1,
            compile_status="success",
            mq5_path="src/mql5/Experts/strat-001.mq5",
            ex5_path="src/mql5/Experts/strat-001.ex5",
            errors=[],
            warnings=[],
        )
        assert resp.success is True
        assert resp.compile_status == "success"
        assert resp.escalated is False

    def test_compile_response_failure_with_errors(self):
        from src.api.compile_endpoints import CompileResponse

        resp = CompileResponse(
            success=False,
            strategy_id="strat-001",
            version=2,
            compile_status="failed",
            mq5_path="src/mql5/Experts/strat-001.mq5",
            errors=["Error on line 42: undeclared identifier"],
            warnings=["Warning: unused variable"],
            escalated=True,
            escalation_reason="Max retries exceeded",
        )
        assert resp.success is False
        assert len(resp.errors) == 1
        assert resp.escalated is True
        assert resp.escalation_reason == "Max retries exceeded"

    def test_compilation_status_response_defaults(self):
        from src.api.compile_endpoints import CompilationStatusResponse

        resp = CompilationStatusResponse(
            strategy_id="strat-001",
            version=1,
            compile_status="pending",
            mq5_path="test.mq5",
        )
        assert resp.compile_status == "pending"
        assert resp.ex5_path is None
        assert resp.errors is None
        assert resp.compile_attempts == 0


class TestCompileEndpoints:
    """Test compile API endpoints with mocked service."""

    def setup_method(self):
        from fastapi import FastAPI
        from src.api.compile_endpoints import router

        self.app = FastAPI()
        self.app.include_router(router)
        self.client = TestClient(self.app)

    def test_get_compile_status_not_found_returns_404(self):
        # EAOutputStorage is lazily imported inside the handler, patch it at source
        with patch("src.strategy.output.EAOutputStorage") as MockStorage:
            mock_storage_instance = MockStorage.return_value
            mock_storage_instance.get_ea.return_value = None

            response = self.client.get("/compile/strat-missing")

        # 404 expected because EA not found
        assert response.status_code == 404

    def test_compile_post_returns_response(self):
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.strategy_id = "strat-001"
        mock_result.version = 1
        mock_result.compile_status = "success"
        mock_result.mq5_path = "test.mq5"
        mock_result.ex5_path = "test.ex5"
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.auto_correction_attempts = 0
        mock_result.escalated_to_floor_manager = False
        mock_result.escalation_reason = None

        with patch("src.api.compile_endpoints.get_service") as mock_svc:
            mock_service = MagicMock()
            mock_service.compile_ea.return_value = mock_result
            mock_svc.return_value = mock_service

            response = self.client.post(
                "/compile",
                json={"strategy_id": "strat-001", "version": 1},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_id"] == "strat-001"

    def test_escalate_not_found_returns_404(self):
        with patch("src.api.compile_endpoints.get_service") as mock_svc:
            mock_service = MagicMock()
            mock_service.ea_storage.get_ea.return_value = None
            mock_svc.return_value = mock_service

            response = self.client.post(
                "/compile/nonexistent/escalate",
                params={"reason": "test reason"},
            )

        assert response.status_code == 404
