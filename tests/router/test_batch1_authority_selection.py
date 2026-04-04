from types import SimpleNamespace
from unittest.mock import Mock, patch


def test_strategy_router_uses_enhanced_governor_for_normal_accounts():
    from src.router.engine import StrategyRouter

    settings = SimpleNamespace(routerMode="auction")
    governor = Mock(name="enhanced_governor")
    commander = Mock(name="commander")

    with (
        patch("src.router.engine.Sentinel"),
        patch("src.router.engine.load_risk_settings", return_value=settings),
        patch("src.router.engine.EnhancedGovernor", return_value=governor) as mock_enhanced_governor,
        patch("src.router.engine.Commander", return_value=commander) as mock_commander,
    ):
        router = StrategyRouter(
            use_smart_kill=False,
            use_multi_timeframe=False,
            account_config={"type": "normal", "account_id": "ACC-1"},
        )

        mock_enhanced_governor.assert_called_once_with(
            account_id="ACC-1",
            settings=settings,
        )
        mock_commander.assert_called_once_with(governor=governor)
        assert router.governor is governor


def test_strategy_router_uses_enhanced_governor_for_prop_firm_accounts():
    from src.router.engine import StrategyRouter

    settings = SimpleNamespace(routerMode="auction")
    governor = Mock(name="enhanced_governor")
    commander = Mock(name="commander")

    with (
        patch("src.router.engine.Sentinel"),
        patch("src.router.engine.load_risk_settings", return_value=settings),
        patch("src.router.engine.EnhancedGovernor", return_value=governor) as mock_enhanced_governor,
        patch("src.router.engine.Commander", return_value=commander) as mock_commander,
    ):
        router = StrategyRouter(
            use_smart_kill=False,
            use_multi_timeframe=False,
            account_config={"type": "prop_firm", "account_id": "FTMO-1"},
        )

        mock_enhanced_governor.assert_called_once_with(
            account_id="FTMO-1",
            settings=settings,
        )
        mock_commander.assert_called_once_with(governor=governor)
        assert router.governor is governor


@patch("src.router.commander.RoutingMatrix")
def test_commander_without_governor_uses_explicit_compatibility_mode(mock_routing_matrix):
    from src.router.commander import Commander

    commander = Commander()

    assert commander._governor.__class__.__name__ == "Governor"
    assert commander._use_enhanced_sizing is False
    assert commander._compatibility_governor is True
