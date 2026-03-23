"""
Tests for Server Health Metrics API (Story 10-5)
"""
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.server_health_endpoints import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def make_mock_metrics():
    return {
        "cpu": 25.0,
        "memory": 50.0,
        "disk": 40.0,
        "latency_ms": 10.0,
        "uptime_seconds": 86400,
        "last_heartbeat": "2026-03-21T10:00:00+00:00",
        "status": "healthy"
    }


def test_get_server_health_metrics_returns_200(client):
    """GET /api/server/health/metrics returns 200 with both nodes."""
    with patch("src.api.server_health_endpoints.get_system_metrics", return_value=make_mock_metrics()), \
         patch("src.api.server_health_endpoints.get_cloudzy_metrics", return_value=make_mock_metrics()):
        response = client.get("/api/server/health/metrics")
    assert response.status_code == 200


def test_get_server_health_metrics_structure(client):
    """Response contains contabo, cloudzy, and timestamp keys."""
    with patch("src.api.server_health_endpoints.get_system_metrics", return_value=make_mock_metrics()), \
         patch("src.api.server_health_endpoints.get_cloudzy_metrics", return_value=make_mock_metrics()):
        response = client.get("/api/server/health/metrics")
    data = response.json()
    assert "contabo" in data
    assert "cloudzy" in data
    assert "timestamp" in data


def test_node_metrics_fields(client):
    """Each node exposes all 6 required metrics."""
    with patch("src.api.server_health_endpoints.get_system_metrics", return_value=make_mock_metrics()), \
         patch("src.api.server_health_endpoints.get_cloudzy_metrics", return_value=make_mock_metrics()):
        response = client.get("/api/server/health/metrics")
    node = response.json()["contabo"]
    for field in ("cpu", "memory", "disk", "latency_ms", "uptime_seconds", "last_heartbeat", "status"):
        assert field in node


def test_critical_status_when_cpu_above_threshold(client):
    """CPU above 85% marks node as critical."""
    high_cpu = {**make_mock_metrics(), "cpu": 90.0, "status": "critical"}
    with patch("src.api.server_health_endpoints.get_system_metrics", return_value=high_cpu), \
         patch("src.api.server_health_endpoints.get_cloudzy_metrics", return_value=make_mock_metrics()):
        response = client.get("/api/server/health/metrics")
    assert response.json()["contabo"]["status"] == "critical"
    assert response.json()["contabo"]["cpu"] == 90.0


def test_get_contabo_metrics_endpoint(client):
    """GET /api/server/health/metrics/contabo returns contabo metrics."""
    with patch("src.api.server_health_endpoints.get_system_metrics", return_value=make_mock_metrics()):
        response = client.get("/api/server/health/metrics/contabo")
    assert response.status_code == 200
    assert "cpu" in response.json()


def test_get_thresholds_endpoint(client):
    """GET /api/server/health/thresholds returns threshold config."""
    response = client.get("/api/server/health/thresholds")
    assert response.status_code == 200
    data = response.json()
    assert data["cpu"] == 85.0
    assert data["disk"] == 90.0
    assert data["memory"] == 90.0
    assert data["latency"] == 500.0
