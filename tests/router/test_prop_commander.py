import pytest
from types import SimpleNamespace

from src.router.prop.commander import PropCommander


class DummyCommander(PropCommander):
    """
    Helper subclass to inject deterministic PropState metrics for testing.
    """
    def __init__(self, account_id: str = "TEST"):
        super().__init__(account_id)
        # Replace prop_state with an object exposing get_metrics() returning a SimpleNamespace
        self._metrics = None

        class _State:
            def __init__(self, outer):
                self._outer = outer
            def get_metrics(self):
                return outer._metrics
        outer = self
        self.prop_state = _State(self)

    def set_metrics(self, **kwargs):
        self._metrics = SimpleNamespace(**kwargs)


@pytest.mark.router
class TestPropCommander:
    def test_standard_mode_calls_base_when_target_not_reached(self, monkeypatch):
        cmd = DummyCommander()
        cmd.set_metrics(daily_start_balance=100_000.0, current_equity=104_000.0, trading_days=3)

        # stub base run_auction to observe call and return
        called = {"count": 0}
        def base_auction(_self, regime_report):
            called["count"] += 1
            return [{"id": 1, "score": 0.5}, {"id": 2, "score": 0.4}]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)  # patch Commander.run_auction

        result = cmd.run_auction(SimpleNamespace(regime="RISK_ON"))
        assert called["count"] == 1
        assert isinstance(result, list)

    def test_preservation_mode_filters_kelly(self, monkeypatch):
        cmd = DummyCommander()
        # Target reached: +8% or more
        cmd.set_metrics(daily_start_balance=100_000.0, current_equity=108_500.0, trading_days=5)

        # Base returns mixed kelly_scores
        def base_auction(_self, regime_report):
            return [
                {"name": "A", "score": 0.9, "kelly_score": 0.85},
                {"name": "B", "score": 0.8, "kelly_score": 0.75},
                {"name": "C", "score": 0.7, "kelly_score": 0.95},
            ]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="RISK_ON"))
        names = [b["name"] for b in result]
        assert names == ["A", "C"]

    def test_coin_flip_bot_when_min_days_not_met(self, monkeypatch):
        cmd = DummyCommander()
        # Target reached but insufficient days
        cmd.set_metrics(daily_start_balance=100_000.0, current_equity=109_000.0, trading_days=3)

        # Base auction would normally return bots, but CoinFlip should override
        def base_auction(_self, regime_report):
            return [{"name": "X", "score": 1.0, "kelly_score": 0.99}]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="QUIET"))
        assert isinstance(result, list) and len(result) == 1
        bot = result[0]
        assert bot.get("name") == "CoinFlip_Bot"
        assert bot.get("risk_mode") == "MINIMAL"

    def test_preservation_mode_threshold_exact(self, monkeypatch):
        cmd = DummyCommander()
        # Exactly 8% gain triggers preservation
        cmd.set_metrics(daily_start_balance=100_000.0, current_equity=108_000.0, trading_days=10)

        def base_auction(_self, regime_report):
            return [
                {"name": "A", "score": 0.9, "kelly_score": 0.8},
                {"name": "B", "score": 0.8, "kelly_score": 0.79},
            ]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="RISK_ON"))
        names = [b["name"] for b in result]
        assert names == ["A"]

    def test_no_metrics_safe_default(self, monkeypatch):
        cmd = DummyCommander()
        # No metrics -> standard mode path
        def base_auction(_self, regime_report):
            return [{"id": 1, "score": 0.5}]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="RISK_ON"))
        assert isinstance(result, list)


    def test_preservation_mode_includes_boundary_kelly(self, monkeypatch):
        cmd = DummyCommander()
        # Target reached
        cmd.set_metrics(daily_start_balance=100_000.0, current_equity=110_000.0, trading_days=7)

        # Base returns a bot exactly at the boundary 0.8
        def base_auction(_self, regime_report):
            return [
                {"name": "Boundary", "kelly_score": 0.8},
                {"name": "Below", "kelly_score": 0.79},
            ]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="ANY"))
        names = [b["name"] for b in result]
        assert names == ["Boundary"]

    def test_preservation_mode_drops_missing_kelly(self, monkeypatch):
        cmd = DummyCommander()
        # Target reached
        cmd.set_metrics(daily_start_balance=50_000.0, current_equity=55_000.0, trading_days=6)

        # One bot has no kelly_score -> treated as 0 and dropped
        def base_auction(_self, regime_report):
            return [
                {"name": "NoKelly"},
                {"name": "High", "kelly_score": 0.95},
            ]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="RISK_ON"))
        names = [b["name"] for b in result]
        assert names == ["High"]

    def test_preservation_mode_handles_empty_base_output(self, monkeypatch):
        cmd = DummyCommander()
        # Target reached and enough trading days
        cmd.set_metrics(daily_start_balance=10_000.0, current_equity=10_900.0, trading_days=9)

        def base_auction(_self, regime_report):
            return []
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="RISK_OFF"))
        assert result == []

    @pytest.mark.parametrize("start_balance", [0.0, None])
    def test_no_preservation_when_start_balance_falsy(self, start_balance, monkeypatch):
        cmd = DummyCommander()
        # Falsy start balance should not trigger preservation
        cmd.set_metrics(daily_start_balance=start_balance, current_equity=999_999.0, trading_days=99)

        called = {"count": 0}
        def base_auction(_self, regime_report):
            called["count"] += 1
            return [{"name": "Any"}]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="ANY"))
        assert called["count"] == 1
        assert result == [{"name": "Any"}]

    def test_no_coin_flip_when_not_in_preservation(self, monkeypatch):
        cmd = DummyCommander()
        # Below target but trading days also below min -> should still call base and not coin flip
        cmd.set_metrics(daily_start_balance=100_000.0, current_equity=103_000.0, trading_days=1)

        def base_auction(_self, regime_report):
            return [
                {"name": "NormalBot", "kelly_score": 0.5}
            ]
        monkeypatch.setattr(PropCommander.__mro__[1], "run_auction", base_auction)

        result = cmd.run_auction(SimpleNamespace(regime="ANY"))
        assert isinstance(result, list) and len(result) == 1
        assert result[0]["name"] == "NormalBot"
