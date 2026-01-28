import pytest
from types import SimpleNamespace

from src.router.prop.governor import PropGovernor
from src.router.governor import RiskMandate


class DummyGovernor(PropGovernor):
    """
    Helper subclass to inject a deterministic PropState for testing without external persistence.
    """
    def __init__(self, account_id: str = "TEST"):
        super().__init__(account_id)
        # Inject a simple namespace as state with overridable attributes
        self.prop_state = SimpleNamespace(daily_start_balance=None)


def make_trade_proposal(**kwargs):
    base = {
        "symbol": "EURUSD",
        "systemic_correlation": 0.0,
        "news_state": None,
        "current_balance": 100_000.0,
    }
    base.update(kwargs)
    return base


@pytest.mark.router
class TestPropGovernor:
    def test_pass_through_when_no_loss_and_no_news(self):
        gov = DummyGovernor()
        # No daily_start_balance set -> falls back to current_balance, resulting in 0 loss
        trade = make_trade_proposal()
        mandate = gov.calculate_risk(trade)
        assert isinstance(mandate, RiskMandate)
        assert mandate.allocation_scalar == pytest.approx(1.0)
        assert mandate.risk_mode == "STANDARD"
        assert (mandate.notes is None) or ("Throttle" not in (mandate.notes or ""))

    def test_news_guard_halts_risk(self):
        gov = DummyGovernor()
        trade = make_trade_proposal(news_state="KILL_ZONE")
        mandate = gov.calculate_risk(trade)
        assert mandate.allocation_scalar == 0.0
        assert mandate.risk_mode == "HALTED_NEWS"
        assert "News" in (mandate.notes or "")

    def test_quadratic_throttle_applies_under_effective_limit(self):
        gov = DummyGovernor()
        # Set a fixed daily start, simulate 2% loss
        gov.prop_state.daily_start_balance = 100_000.0
        current_balance = 98_000.0  # 2% loss
        trade = make_trade_proposal(current_balance=current_balance)
        mandate = gov.calculate_risk(trade)
        # Effective limit = 4%, so throttle should be between (0,1)
        assert 0.0 < mandate.allocation_scalar < 1.0
        assert mandate.risk_mode == "THROTTLED"
        assert "Prop Throttle Active" in (mandate.notes or "")

    def test_zero_throttle_when_beyond_effective_limit(self):
        gov = DummyGovernor()
        gov.prop_state.daily_start_balance = 100_000.0
        # Breach: > 4% loss (effective limit is 5% - 1% buffer = 4%)
        current_balance = 95_000.0  # 5% loss
        trade = make_trade_proposal(current_balance=current_balance)
        mandate = gov.calculate_risk(trade)
        # Base scalar is 1.0, then multiplied by throttle (0.0)
        assert mandate.allocation_scalar == 0.0
        # THROTTLED or STANDARD not applicable since throttle 0.0 via combine – risk_mode remains default unless news
        # We do not assert risk_mode exact value beyond ensuring scalar is 0.0

    def test_base_clamp_then_throttle_combination(self):
        gov = DummyGovernor()
        gov.prop_state.daily_start_balance = 100_000.0
        # Induce base governor clamp via high systemic correlation
        trade = make_trade_proposal(systemic_correlation=0.95, current_balance=98_000.0)  # 2% loss
        mandate = gov.calculate_risk(trade)
        # Base clamps to 0.5 then throttle (<1) applies multiplicatively
        assert 0.0 < mandate.allocation_scalar < 0.5
        assert mandate.risk_mode in {"CLAMPED", "THROTTLED"}

