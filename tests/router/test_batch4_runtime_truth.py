from types import SimpleNamespace

from src.router.broker_registry import BrokerRegistryManager
from src.router.progressive_kill_switch import load_progressive_config


def test_resolve_execution_profile_uses_registry_values(monkeypatch):
    manager = BrokerRegistryManager.__new__(BrokerRegistryManager)
    manager.get_broker = lambda broker_id: SimpleNamespace(
        broker_id=broker_id,
        broker_name="Exness Raw",
        spread_avg=0.2,
        commission_per_lot=3.5,
        lot_step=0.01,
        min_lot=0.01,
        max_lot=100.0,
        pip_values={"EURUSD": 10.0, "XAUUSD": 1.0},
        preference_tags=["RAW_ECN"],
    )

    profile = manager.resolve_execution_profile("exness_raw", symbol="XAUUSD")

    assert profile.broker_id == "exness_raw"
    assert profile.broker_name == "Exness Raw"
    assert profile.commission_per_lot == 3.5
    assert profile.pip_value == 1.0
    assert profile.source == "registry"


def test_resolve_execution_profile_falls_back_without_registry():
    manager = BrokerRegistryManager.__new__(BrokerRegistryManager)
    manager.get_broker = lambda broker_id: None

    profile = manager.resolve_execution_profile("missing_broker", symbol="EURUSD")

    assert profile.broker_id == "missing_broker"
    assert profile.pip_value == 10.0
    assert profile.min_lot == 0.01
    assert profile.source == "fallback"


def test_progressive_config_overlays_runtime_risk_settings(monkeypatch, tmp_path):
    config_path = tmp_path / "trading_system.yaml"
    config_path.write_text(
        "kill_switch:\n"
        "  progressive:\n"
        "    tier3:\n"
        "      max_daily_loss_pct: 0.03\n"
        "      max_weekly_loss_pct: 0.10\n"
    )

    runtime_risk = SimpleNamespace(
        daily_loss_limit_pct=0.045,
        weekly_loss_limit_pct=0.08,
    )

    monkeypatch.setattr(
        "src.api.settings_endpoints.load_runtime_risk_config",
        lambda: runtime_risk,
    )

    config = load_progressive_config(str(config_path))

    assert config.tier3_max_daily_loss_pct == 0.045
    assert config.tier3_max_weekly_loss_pct == 0.08
