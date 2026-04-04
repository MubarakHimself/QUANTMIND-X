from src.api.ide_backtest import (
    _build_parameter_sweep_grid,
    _request_backtest_approval,
)
from src.api.ide_models import BacktestRunRequest


def test_build_parameter_sweep_grid_uses_known_placeholders():
    strategy_code = """
ma_fast = {{ma_fast}}
ma_slow = {{ma_slow}}
take_profit = {{take_profit}}
unknown_value = {{custom_value}}
"""

    grid = _build_parameter_sweep_grid(strategy_code)

    assert grid == {
        "ma_fast": [10, 20, 30],
        "ma_slow": [50, 100, 200],
        "take_profit": [20, 40, 60],
    }


def test_request_backtest_approval_uses_workflow_gate(monkeypatch):
    captured = {}

    class MockApprovalManager:
        def request_approval(self, **kwargs):
            captured.update(kwargs)
            return type("Approval", (), {"id": "apr_test"})()

    monkeypatch.setattr(
        "src.agents.approval_manager.get_approval_manager",
        lambda: MockApprovalManager(),
    )

    request = BacktestRunRequest(
        symbol="EURUSD",
        timeframe="H1",
        start_date="2024-01-01",
        end_date="2024-06-01",
        variant="spiced",
        strategy_name="Momentum",
        strategy_code="print('hello')",
    )

    _request_backtest_approval(
        backtest_id="bt_123",
        request=request,
        report_text="report body",
        optimization_results=[{"pass": 1, "net_profit": 10.0}],
    )

    assert captured["workflow_id"] == "bt_123"
    assert captured["department"] == "trading"
    assert captured["context"]["report"] == "report body"
    assert captured["context"]["optimization_results"] == [{"pass": 1, "net_profit": 10.0}]
