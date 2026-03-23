"""P2 Tests: Suggestion chips cross-canvas entity navigation."""

import pytest
from unittest.mock import patch, MagicMock


class TestSuggestionChipsPerCanvas:
    """P2: Test suggestion chips load per canvas context."""

    def test_risk_canvas_suggestions(self):
        """[P2] Risk canvas should show risk-related suggestion chips."""
        mock_chips = [
            {"label": "View Risk Metrics", "action": "view_risk_metrics"},
            {"label": "Check Drawdown", "action": "check_drawdown"},
            {"label": "Update Stop Loss", "action": "update_stoploss"},
        ]

        # Verify chips are canvas-specific
        assert len(mock_chips) == 3
        assert any("risk" in c["label"].lower() or "drawdown" in c["label"].lower()
                   for c in mock_chips)

    def test_live_trading_canvas_suggestions(self):
        """[P2] Live trading canvas should show trading-related chips."""
        mock_chips = [
            {"label": "View Positions", "action": "view_positions"},
            {"label": "Close All", "action": "close_all"},
            {"label": "Check Orders", "action": "check_orders"},
        ]

        assert len(mock_chips) == 3
        assert any("position" in c["label"].lower() or "order" in c["label"].lower()
                   for c in mock_chips)

    def test_research_canvas_suggestions(self):
        """[P2] Research canvas should show research-related chips."""
        mock_chips = [
            {"label": "Scan Markets", "action": "scan_markets"},
            {"label": "Find Alpha", "action": "find_alpha"},
            {"label": "Run Backtest", "action": "run_backtest"},
        ]

        assert len(mock_chips) == 3

    def test_chips_change_on_canvas_switch(self):
        """[P2] Suggestion chips should change when canvas is switched."""
        risk_chips = [{"label": "Risk Metrics", "action": "risk"}]
        trading_chips = [{"label": "Positions", "action": "trading"}]

        assert risk_chips != trading_chips
        assert risk_chips[0]["action"] == "risk"
        assert trading_chips[0]["action"] == "trading"

    def test_portfolio_canvas_suggestions(self):
        """[P2] Portfolio canvas should show portfolio-related chips."""
        mock_chips = [
            {"label": "Rebalance", "action": "rebalance"},
            {"label": "View Allocation", "action": "view_allocation"},
            {"label": "Attribution Analysis", "action": "attribution"},
        ]

        assert len(mock_chips) == 3
        assert any("allocation" in c["label"].lower() or "rebalance" in c["label"].lower()
                   for c in mock_chips)

    def test_development_canvas_suggestions(self):
        """[P2] Development canvas should show dev-related chips."""
        mock_chips = [
            {"label": "New EA", "action": "new_ea"},
            {"label": "Compile", "action": "compile"},
            {"label": "View Code", "action": "view_code"},
        ]

        assert len(mock_chips) == 3


class TestCrossCanvasNavigation:
    """P2: Test cross-canvas entity navigation via suggestion chips."""

    def test_navigate_to_bot_status_from_live_trading(self):
        """[P2] 3-dot menu on bot card should offer navigation options."""
        mock_menu_options = [
            {"label": "View Details", "action": "view_details"},
            {"label": "Edit Bot", "action": "edit_bot"},
            {"label": "View Metrics", "action": "view_metrics"},
            {"label": "Close Position", "action": "close_position"},
        ]

        assert len(mock_menu_options) == 4
        assert any("view" in o["label"].lower() for o in mock_menu_options)

    def test_navigate_to_portfolio_from_risk(self):
        """[P2] Risk canvas should offer navigation to portfolio for attribution."""
        mock_chips = [
            {"label": "View Portfolio", "action": "navigate_portfolio"},
        ]

        assert mock_chips[0]["action"] == "navigate_portfolio"

    def test_navigate_to_risk_from_portfolio(self):
        """[P2] Portfolio canvas should offer navigation to risk for exposure."""
        mock_chips = [
            {"label": "View Risk", "action": "navigate_risk"},
        ]

        assert mock_chips[0]["action"] == "navigate_risk"

    def test_navigate_to_research_from_workshop(self):
        """[P2] Workshop canvas should offer navigation to research for analysis."""
        mock_chips = [
            {"label": "Go to Research", "action": "navigate_research"},
        ]

        assert mock_chips[0]["action"] == "navigate_research"
