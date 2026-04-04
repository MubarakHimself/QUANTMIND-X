import builtins
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from src.api.ide_mt5 import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class _StubInfo:
    def __init__(self, **kwargs):
        self._payload = kwargs

    def _asdict(self):
        return dict(self._payload)


def test_mt5_status_reports_missing_package(client, monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "MetaTrader5":
            raise ImportError("missing for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    response = client.get("/api/ide/mt5/status")

    assert response.status_code == 200
    data = response.json()
    assert data["connected"] is False
    assert data["accounts"] == []
    assert "not installed" in data["error"].lower()


def test_mt5_status_reports_linux_host_limitation(client, monkeypatch):
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "MetaTrader5":
            raise ImportError("missing for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    response = client.get("/api/ide/mt5/status")

    assert response.status_code == 200
    data = response.json()
    assert data["connected"] is False
    assert "linux" in data["error"].lower()
    assert "windows" in data["error"].lower()


def test_mt5_status_reports_real_runtime_state(client, monkeypatch):
    stub = ModuleType("MetaTrader5")
    stub.initialize = lambda: True
    stub.shutdown = lambda: True
    stub.last_error = lambda: (0, "ok")
    stub.version = lambda: (500, 4200, "01 Apr 2026")
    stub.terminal_info = lambda: _StubInfo(path="C:/MT5/terminal64.exe", connected=True, trade_allowed=True)
    stub.account_info = lambda: _StubInfo(
        login=123456,
        server="Broker-Demo",
        currency="USD",
        balance=1000.0,
        equity=1010.0,
        margin=10.0,
        margin_free=1000.0,
    )

    monkeypatch.setitem(__import__("sys").modules, "MetaTrader5", stub)

    response = client.get("/api/ide/mt5/status")

    assert response.status_code == 200
    data = response.json()
    assert data["connected"] is True
    assert data["terminal_path"] == "C:/MT5/terminal64.exe"
    assert data["version"] == [500, 4200, "01 Apr 2026"]
    assert data["accounts"][0]["login"] == 123456
    assert data["accounts"][0]["server"] == "Broker-Demo"


def test_mt5_scan_returns_real_filesystem_hits(client, tmp_path):
    terminal = tmp_path / "terminal64.exe"
    terminal.write_text("stub-binary")

    response = client.post(
        "/api/ide/mt5/scan",
        json={"custom_paths": [str(terminal)]},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 1
    assert any(row["path"] == str(terminal) and row["exists"] is True for row in data["terminals"])
