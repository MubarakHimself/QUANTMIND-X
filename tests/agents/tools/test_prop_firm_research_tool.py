"""
Tests for Prop Firm Research Tool

Tests for firm_analysis, get_rules, compare_firms, and other PropFirmResearch methods.
"""

import pytest
from src.agents.tools.prop_firm_research_tool import (
    PropFirmResearch,
    PropFirm,
    AccountTier,
    PROP_FIRM_TOOL_SCHEMAS,
    get_prop_firm_tool_schemas,
)


class TestPropFirmResearch:
    """Test suite for PropFirmResearch class."""

    @pytest.fixture
    def tools(self):
        """Create PropFirmResearch instance."""
        return PropFirmResearch()

    # =========================================================================
    # get_firms tests
    # =========================================================================

    def test_get_firms(self, tools):
        """Test getting list of supported firms."""
        firms = tools.get_firms()
        assert isinstance(firms, list)
        assert "fundednext" in firms
        assert "trueforex" in firms
        assert "mycryptobuddy" in firms

    # =========================================================================
    # get_tiers tests
    # =========================================================================

    def test_get_tiers_fundednext(self, tools):
        """Test getting tiers for FundedNext."""
        tiers = tools.get_tiers("fundednext")
        assert isinstance(tiers, list)
        assert "starter" in tiers
        assert "standard" in tiers
        assert "pro" in tiers
        assert "elite" in tiers

    def test_get_tiers_trueforex(self, tools):
        """Test getting tiers for TrueForex."""
        tiers = tools.get_tiers("trueforex")
        assert isinstance(tiers, list)
        assert "starter" in tiers
        assert "standard" in tiers
        assert "pro" in tiers

    def test_get_tiers_mcryptobuddy(self, tools):
        """Test getting tiers for MyCryptoBuddy."""
        tiers = tools.get_tiers("mycryptobuddy")
        assert isinstance(tiers, list)
        assert "starter" in tiers
        assert "standard" in tiers
        assert "pro" in tiers

    def test_get_tiers_unknown_firm(self, tools):
        """Test getting tiers for unknown firm."""
        tiers = tools.get_tiers("unknown_firm")
        assert tiers == []

    # =========================================================================
    # get_rules tests
    # =========================================================================

    def test_get_rules_fundednext_standard(self, tools):
        """Test getting rules for FundedNext standard tier."""
        rules = tools.get_rules("fundednext", "standard")
        assert rules is not None
        assert rules["firm"] == "fundednext"
        assert rules["tier"] == "standard"
        assert rules["initial_balance"] == 10000
        assert rules["funded_balance"] == 10000
        assert rules["max_drawdown_pct"] == 10.0
        assert rules["daily_drawdown_pct"] == 5.0
        assert rules["profit_target_pct"] == 8.0
        assert rules["leverage"] == 100
        assert rules["ea_allowed"] is True
        assert rules["hedge_allowed"] is True

    def test_get_rules_trueforex_starter(self, tools):
        """Test getting rules for TrueForex starter tier."""
        rules = tools.get_rules("trueforex", "starter")
        assert rules is not None
        assert rules["firm"] == "trueforex"
        assert rules["tier"] == "starter"
        assert rules["initial_balance"] == 5000
        assert rules["max_drawdown_pct"] == 8.0
        assert rules["news_trading_allowed"] is True

    def test_get_rules_mcryptobuddy_pro(self, tools):
        """Test getting rules for MyCryptoBuddy pro tier."""
        rules = tools.get_rules("mycryptobuddy", "pro")
        assert rules is not None
        assert rules["firm"] == "mycryptobuddy"
        assert rules["allowed_instruments"] == ["Crypto"]
        assert rules["leverage"] == 50
        assert rules["hedge_allowed"] is False

    def test_get_rules_unknown_firm(self, tools):
        """Test getting rules for unknown firm."""
        rules = tools.get_rules("unknown_firm", "standard")
        assert rules is None

    def test_get_rules_unknown_tier(self, tools):
        """Test getting rules for unknown tier."""
        rules = tools.get_rules("fundednext", "unknown_tier")
        assert rules is None

    # =========================================================================
    # firm_analysis tests
    # =========================================================================

    def test_firm_analysis_fundednext(self, tools):
        """Test firm analysis for FundedNext."""
        result = tools.firm_analysis("fundednext", "standard")
        assert "error" not in result
        assert result["firm"] == "fundednext"
        assert result["tier"] == "standard"
        assert "overall_score" in result
        assert isinstance(result["pros"], list)
        assert isinstance(result["cons"], list)
        assert isinstance(result["recommendations"], list)
        assert "risk_assessment" in result
        assert "suitability" in result

    def test_firm_analysis_trueforex(self, tools):
        """Test firm analysis for TrueForex."""
        result = tools.firm_analysis("trueforex", "standard")
        assert "error" not in result
        assert result["firm"] == "trueforex"
        assert result["overall_score"] > 0

    def test_firm_analysis_mcryptobuddy(self, tools):
        """Test firm analysis for MyCryptoBuddy."""
        result = tools.firm_analysis("mycryptobuddy", "standard")
        assert "error" not in result
        assert result["firm"] == "mycryptobuddy"
        assert "crypto-focused" in result["suitability"].lower()

    def test_firm_analysis_unknown_firm(self, tools):
        """Test firm analysis for unknown firm."""
        result = tools.firm_analysis("unknown_firm", "standard")
        assert "error" in result

    def test_firm_analysis_unknown_tier(self, tools):
        """Test firm analysis for unknown tier."""
        result = tools.firm_analysis("fundednext", "unknown_tier")
        assert "error" in result

    def test_firm_analysis_caching(self, tools):
        """Test that firm analysis results are cached."""
        result1 = tools.firm_analysis("fundednext", "standard")
        result2 = tools.firm_analysis("fundednext", "standard")
        assert result1 == result2

    # =========================================================================
    # compare_firms tests
    # =========================================================================

    def test_compare_firms_single(self, tools):
        """Test comparing a single firm."""
        result = tools.compare_firms(["fundednext"], "standard")
        assert result["tier"] == "standard"
        assert len(result["firms"]) == 1
        assert result["firms"][0]["firm"] == "fundednext"

    def test_compare_firms_multiple(self, tools):
        """Test comparing multiple firms."""
        result = tools.compare_firms(["fundednext", "trueforex", "mycryptobuddy"], "standard")
        assert result["tier"] == "standard"
        assert len(result["firms"]) == 3
        # Should be sorted by score descending
        scores = [f["score"] for f in result["firms"]]
        assert scores == sorted(scores, reverse=True)

    def test_compare_firms_sorted_by_score(self, tools):
        """Test that comparison results are sorted by score."""
        result = tools.compare_firms(["mycryptobuddy", "fundednext", "trueforex"], "standard")
        firms = result["firms"]
        assert firms[0]["score"] >= firms[1]["score"]
        assert firms[1]["score"] >= firms[2]["score"]

    def test_compare_firms_unknown_tier(self, tools):
        """Test comparing firms with unknown tier returns empty."""
        result = tools.compare_firms(["fundednext"], "unknown_tier")
        assert result["firms"] == []

    # =========================================================================
    # Tool schemas tests
    # =========================================================================

    def test_tool_schemas_exist(self):
        """Test that tool schemas are defined."""
        assert len(PROP_FIRM_TOOL_SCHEMAS) > 0

    def test_tool_schemas_structure(self):
        """Test tool schemas have correct structure."""
        for schema in PROP_FIRM_TOOL_SCHEMAS:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_get_tool_schemas_function(self):
        """Test get_prop_firm_tool_schemas function."""
        schemas = get_prop_firm_tool_schemas()
        assert isinstance(schemas, list)
        assert len(schemas) > 0

    # =========================================================================
    # Edge cases and integration tests
    # =========================================================================

    def test_case_insensitive_firm_name(self, tools):
        """Test that firm names are case insensitive."""
        rules1 = tools.get_rules("FUNDEDNEXT", "standard")
        rules2 = tools.get_rules("fundednext", "standard")
        rules3 = tools.get_rules("FundedNext", "standard")
        assert rules1 == rules2 == rules3

    def test_case_insensitive_tier(self, tools):
        """Test that tier names are case insensitive."""
        rules1 = tools.get_rules("fundednext", "STANDARD")
        rules2 = tools.get_rules("fundednext", "standard")
        rules3 = tools.get_rules("fundednext", "Standard")
        assert rules1 == rules2 == rules3

    def test_all_tiers_have_monthly_fee(self, tools):
        """Test that all tiers have monthly fee defined."""
        for firm in ["fundednext", "trueforex", "mycryptobuddy"]:
            tiers = tools.get_tiers(firm)
            for tier in tiers:
                rules = tools.get_rules(firm, tier)
                assert rules is not None
                assert rules["monthly_fee"] is not None

    def test_all_tiers_have_profit_target(self, tools):
        """Test that all tiers have profit target defined."""
        for firm in ["fundednext", "trueforex", "mycryptobuddy"]:
            tiers = tools.get_tiers(firm)
            for tier in tiers:
                rules = tools.get_rules(firm, tier)
                assert rules is not None
                assert rules["profit_target_pct"] > 0
