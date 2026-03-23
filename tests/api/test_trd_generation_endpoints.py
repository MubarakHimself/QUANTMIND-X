"""
Tests for TRD Generation API Endpoints

Tests TRD generation request/response model structures and endpoint logic.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI


def _make_client():
    from src.api.trd_generation_endpoints import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestTRDModels:
    """Test TRD request/response model structures."""

    def test_hypothesis_input_defaults(self):
        from src.api.trd_generation_endpoints import HypothesisInput

        h = HypothesisInput(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="RSI divergence predicts reversals",
        )
        assert h.symbol == "EURUSD"
        assert h.timeframe == "H4"
        assert h.supporting_evidence == []
        assert h.confidence_score == 0.5

    def test_trd_generate_request_defaults(self):
        from src.api.trd_generation_endpoints import TRDGenerateRequest, HypothesisInput

        req = TRDGenerateRequest(
            hypothesis=HypothesisInput(
                symbol="GBPUSD",
                timeframe="H1",
                hypothesis="Moving average crossover strategy",
            )
        )
        assert req.run_validation is True
        assert req.auto_save is True
        assert req.strategy_name is None

    def test_trd_generate_response_structure(self):
        from src.api.trd_generation_endpoints import TRDGenerateResponse

        resp = TRDGenerateResponse(
            success=True,
            strategy_id="trd-001",
            trd={"strategy_id": "trd-001"},
            requires_clarification=False,
            is_valid=True,
            is_complete=True,
            message="TRD generated successfully",
        )
        assert resp.success is True
        assert resp.strategy_id == "trd-001"
        assert resp.is_valid is True

    def test_trd_generate_response_failure_defaults(self):
        from src.api.trd_generation_endpoints import TRDGenerateResponse

        resp = TRDGenerateResponse(
            success=False,
            strategy_id="",
            message="TRD generation failed: some error",
        )
        assert resp.success is False
        assert resp.trd is None
        assert resp.requires_clarification is False

    def test_trd_validate_request(self):
        from src.api.trd_generation_endpoints import TRDValidateRequest

        req = TRDValidateRequest(strategy_id="strat-001")
        assert req.strategy_id == "strat-001"

    def test_trd_validate_response_structure(self):
        from src.api.trd_generation_endpoints import TRDValidateResponse

        resp = TRDValidateResponse(
            success=True,
            is_valid=True,
            is_complete=False,
            missing_parameters=["stop_loss"],
        )
        assert resp.is_valid is True
        assert "stop_loss" in resp.missing_parameters
        assert resp.errors == []


class TestTRDGenerationEndpoints:
    """Test TRD generation API endpoints with mocked dependencies."""

    def test_generate_trd_success(self):
        client = _make_client()
        mock_trd = MagicMock()
        mock_trd.strategy_id = "strat-generated"
        mock_trd.to_dict.return_value = {"strategy_id": "strat-generated"}

        mock_result = {
            "trd": mock_trd,
            "validation": {"is_valid": True},
            "requires_clarification": False,
            "is_valid": True,
            "is_complete": True,
        }

        with patch("src.api.trd_generation_endpoints.TRDGenerator") as MockGen:
            mock_gen_instance = MockGen.return_value
            mock_gen_instance.generate_and_validate.return_value = mock_result

            with patch("src.api.trd_generation_endpoints.save_trd"):
                response = client.post(
                    "/api/trd/generation/generate",
                    json={
                        "hypothesis": {
                            "symbol": "EURUSD",
                            "timeframe": "H4",
                            "hypothesis": "Trend following based on EMA crossover",
                        },
                        "auto_save": False,
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["strategy_id"] == "strat-generated"

    def test_generate_trd_exception_returns_failure(self):
        client = _make_client()

        with patch("src.api.trd_generation_endpoints.TRDGenerator") as MockGen:
            MockGen.return_value.generate_and_validate.side_effect = Exception("LLM error")

            response = client.post(
                "/api/trd/generation/generate",
                json={
                    "hypothesis": {
                        "symbol": "EURUSD",
                        "timeframe": "H4",
                        "hypothesis": "Test",
                    },
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "failed" in data["message"].lower()

    def test_validate_trd_not_found_returns_404(self):
        client = _make_client()

        with patch("src.api.trd_generation_endpoints.load_trd_from_storage") as mock_load:
            mock_load.return_value = None

            response = client.post(
                "/api/trd/generation/validate",
                json={"strategy_id": "nonexistent-trd"},
            )

        assert response.status_code == 404

    def test_validate_trd_success(self):
        client = _make_client()
        mock_trd = MagicMock()
        mock_validation = MagicMock()
        mock_validation.is_valid = True
        mock_validation.is_complete = True
        mock_validation.errors = []
        mock_validation.ambiguities = []
        mock_validation.missing_parameters = []
        mock_validation.requires_clarification.return_value = False

        with patch("src.api.trd_generation_endpoints.load_trd_from_storage", return_value=mock_trd):
            with patch("src.api.trd_generation_endpoints.TRDValidator") as MockValidator:
                MockValidator.return_value.validate.return_value = mock_validation

                response = client.post(
                    "/api/trd/generation/validate",
                    json={"strategy_id": "strat-001"},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_valid"] is True

    def test_clarification_not_found_returns_404(self):
        client = _make_client()

        with patch("src.api.trd_generation_endpoints.load_trd_from_storage", return_value=None):
            response = client.get("/api/trd/generation/clarification/nonexistent")

        assert response.status_code == 404

    def test_simple_generate_endpoint(self):
        client = _make_client()
        mock_trd = MagicMock()
        mock_trd.to_dict.return_value = {"strategy_id": "simple-strat"}

        mock_result = {
            "trd": mock_trd,
            "validation": {"is_valid": True},
            "requires_clarification": False,
            "is_valid": True,
            "is_complete": True,
        }

        with patch("src.api.trd_generation_endpoints.TRDGenerator") as MockGen:
            MockGen.return_value.generate_and_validate.return_value = mock_result

            with patch("src.api.trd_generation_endpoints.save_trd"):
                response = client.post(
                    "/api/trd/generation/from-hypothesis-simple",
                    json={
                        "hypothesis": {
                            "symbol": "USDJPY",
                            "timeframe": "M15",
                            "hypothesis": "Breakout pattern",
                        },
                        "auto_save": False,
                    },
                )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
