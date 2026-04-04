from src.api.ide_handlers_broker import BrokerAccountsAPIHandler


def test_list_broker_accounts_returns_empty_without_runtime_registry_entries(monkeypatch):
    class _Registry:
        brokers = {}
        pending = {}

    class _Switcher:
        active_account_id = None

    import src.api.broker_endpoints as broker_endpoints

    monkeypatch.setattr(broker_endpoints, "broker_registry", _Registry())
    monkeypatch.setattr(broker_endpoints, "account_switcher", _Switcher())

    handler = BrokerAccountsAPIHandler()
    assert handler.list_broker_accounts() == []
