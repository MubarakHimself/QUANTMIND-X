import builtins
import importlib
import sys


def test_grafana_cloud_pusher_imports_without_requests_at_module_import(monkeypatch):
    module_name = "src.monitoring.grafana_cloud_pusher"
    sys.modules.pop(module_name, None)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "requests":
            raise ModuleNotFoundError("broken requests")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = importlib.import_module(module_name)

    assert hasattr(module, "GrafanaCloudPusher")

