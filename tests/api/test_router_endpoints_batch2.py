import types

from src.api import router_endpoints


class _FakeLimiter:
    def get_max_bots(self, account_balance):
        return 7 if account_balance == 2500.0 else 3

    def get_tier_info(self, account_balance):
        return {
            "tier": 5,
            "tier_label": "Tier 5 (Compatibility)",
            "range": (1000, 5000),
            "range_label": "$1000-$5000",
        }

    def calculate_safety_buffer(self, account_balance, active_bots):
        return active_bots * 100.0


class _FakeRegistry:
    def __init__(self, bots):
        self._bots = bots

    def list_all(self):
        return self._bots


def test_build_bot_limit_status_prefers_commander_legacy_status():
    live_bot = types.SimpleNamespace(tags=[])
    dead_bot = types.SimpleNamespace(tags=["@dead"])

    commander = types.SimpleNamespace(
        active_bots={"a": object(), "b": object()},
        _get_legacy_dynamic_limit_status=lambda account_balance=None: {
            "limit_source": "dynamic_bot_limits_legacy",
            "account_balance": account_balance,
            "active_positions": 2,
            "max_bots": 6,
            "available_slots": 4,
        },
    )
    strategy_router = types.SimpleNamespace(
        commander=commander,
        progressive_kill_switch=None,
    )

    payload = router_endpoints._build_bot_limit_status_payload(
        account_balance=2500.0,
        strategy_router=strategy_router,
        limiter=_FakeLimiter(),
        bot_registry_factory=lambda: _FakeRegistry([live_bot, dead_bot]),
    )

    assert payload["account_balance"] == 2500.0
    assert payload["max_bots"] == 6
    assert payload["active_bots"] == 1
    assert payload["available_slots"] == 4
    assert payload["limit_source"] == "dynamic_bot_limits_legacy"
    assert payload["tier"] == "Tier 5 (Compatibility)"


def test_build_bot_limit_status_falls_back_when_commander_missing():
    payload = router_endpoints._build_bot_limit_status_payload(
        account_balance=400.0,
        strategy_router=None,
        limiter=_FakeLimiter(),
        bot_registry_factory=lambda: _FakeRegistry([]),
    )

    assert payload["account_balance"] == 400.0
    assert payload["max_bots"] == 3
    assert payload["active_bots"] == 0
    assert payload["available_slots"] == 3
    assert payload["limit_source"] == "dynamic_bot_limits_legacy"
