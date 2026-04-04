from src.api import journal_endpoints


def test_build_review_summary_prioritizes_quarantined_and_losing_bots(monkeypatch):
    monkeypatch.setattr(
        journal_endpoints,
        "query_trade_journal",
        lambda limit=200, **kwargs: [
            {
                "eaName": "bot_a",
                "status": "closed",
                "pnl": -10.0,
                "entryTime": "2026-04-03T10:00:00+00:00",
            },
            {
                "eaName": "bot_a",
                "status": "closed",
                "pnl": -5.0,
                "entryTime": "2026-04-03T11:00:00+00:00",
            },
            {
                "eaName": "bot_b",
                "status": "closed",
                "pnl": 15.0,
                "entryTime": "2026-04-03T12:00:00+00:00",
            },
        ],
    )

    class StubCircuitBreakerManager:
        def get_all_states(self):
            return [
                {
                    "bot_id": "bot_a",
                    "consecutive_losses": 2,
                    "is_quarantined": True,
                    "quarantine_reason": "bad session",
                },
                {
                    "bot_id": "bot_b",
                    "consecutive_losses": 0,
                    "is_quarantined": False,
                    "quarantine_reason": None,
                },
            ]

    class StubAccountMonitor:
        def get_accounts_at_risk(self):
            return [{"account_id": "acct-1"}]

    monkeypatch.setattr(
        "src.router.bot_circuit_breaker.BotCircuitBreakerManager",
        StubCircuitBreakerManager,
    )
    monkeypatch.setattr(
        "src.router.account_monitor.get_account_monitor",
        lambda: StubAccountMonitor(),
    )

    summary = journal_endpoints.build_review_summary(limit=50)

    assert summary.total_reviewed == 2
    assert summary.quarantined_bots == 1
    assert summary.pressure_accounts == 1
    assert summary.review_watchlist[0].bot_id == "bot_a"
    assert summary.review_watchlist[0].is_quarantined is True
    assert summary.review_watchlist[0].consecutive_losses == 2
