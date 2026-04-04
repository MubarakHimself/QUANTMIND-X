from unittest.mock import Mock, patch

from src.router.commander import Commander


@patch("src.router.commander.RoutingMatrix")
def test_get_legacy_dynamic_limit_status_reports_legacy_source(mock_routing_matrix):
    commander = Commander()
    commander._governor._daily_start_balance = 2500.0

    status = commander._get_legacy_dynamic_limit_status()

    assert status["limit_source"] == "dynamic_bot_limits_legacy"
    assert status["account_balance"] == 2500.0
    assert "max_bots" in status
    assert "available_slots" in status


@patch("src.router.commander.RoutingMatrix")
def test_select_funded_bots_with_legacy_limits_stops_when_limiter_blocks(mock_routing_matrix):
    commander = Commander()
    ranked_bots = [
        {"bot_id": "bot-1"},
        {"bot_id": "bot-2"},
        {"bot_id": "bot-3"},
    ]

    fake_limiter = Mock()
    fake_limiter.get_max_bots.return_value = 2
    fake_limiter.can_add_bot.side_effect = [
        (True, "slot 1"),
        (True, "slot 2"),
        (False, "limit reached"),
    ]

    with patch.object(commander, "_count_active_positions", return_value=0):
        selected_bots, limit_status = commander._select_funded_bots_with_legacy_limits(
            ranked_bots,
            account_balance=500.0,
            limiter=fake_limiter,
        )

    assert [bot["bot_id"] for bot in selected_bots] == ["bot-1", "bot-2"]
    assert limit_status["limit_source"] == "dynamic_bot_limits_legacy"
    assert limit_status["max_bots"] == 2
    assert limit_status["active_positions"] == 0
    assert limit_status["available_slots"] == 2


@patch("src.router.commander.RoutingMatrix")
def test_get_legacy_dynamic_limit_status_preserves_explicit_zero_balance(mock_routing_matrix):
    commander = Commander()
    commander._governor._daily_start_balance = 2500.0

    status = commander._get_legacy_dynamic_limit_status(account_balance=0.0, active_positions=0)

    assert status["account_balance"] == 0.0
    assert status["limit_source"] == "dynamic_bot_limits_legacy"
