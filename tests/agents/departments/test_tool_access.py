"""
Test suite for ToolAccessController functionality

Tests verify:
1. Tool permission levels (READ/WRITE)
2. Live trading is prohibited for all agents
3. Strategy Router, Risk/Position, Broker are READ-ONLY for ALL departments
4. Research has WRITE access to EA lifecycle
5. Other departments have READ-ONLY access to EA lifecycle
"""
import pytest
from src.agents.departments.tool_access import ToolPermission, ToolAccessController, LIVE_TRADING_PROHIBITED
from src.agents.departments.types import Department


class TestToolPermissionEnum:
    """Test ToolPermission enum values."""

    def test_read_permission_value(self):
        """Test READ permission has correct value."""
        assert ToolPermission.READ.value == "read"

    def test_write_permission_value(self):
        """Test WRITE permission has correct value."""
        assert ToolPermission.WRITE.value == "write"


class TestLiveTradingProhibition:
    """Test that live trading is prohibited for all departments."""

    @pytest.mark.parametrize("department", [
        Department.ANALYSIS,
        Department.RESEARCH,
        Department.RISK,
        Department.EXECUTION,
        Department.PORTFOLIO,
    ])
    def test_live_trading_always_prohibited(self, department):
        """Test live trading is prohibited for all departments."""
        controller = ToolAccessController(department)
        assert controller.is_live_trading_prohibited() is True

    def test_live_trading_module_constant(self):
        """Test module-level LIVE_TRADING_PROHIBITED constant."""
        assert LIVE_TRADING_PROHIBITED is True

    def test_live_trading_tool_name_returns_false(self):
        """Test that 'live_trading' tool name always returns False."""
        controller = ToolAccessController(Department.EXECUTION)
        assert controller.can_access("live_trading", ToolPermission.READ) is False
        assert controller.can_access("live_trading", ToolPermission.WRITE) is False


class TestResearchDepartmentAccess:
    """Test Research department has WRITE access to EA lifecycle."""

    def test_research_has_write_ea_lifecycle(self):
        """Test Research has WRITE access to EA lifecycle."""
        controller = ToolAccessController(Department.RESEARCH)
        assert controller.has_write_access("ea_lifecycle") is True

    def test_research_read_only_shared_tools(self):
        """Test Research has READ-ONLY access to shared tools."""
        controller = ToolAccessController(Department.RESEARCH)

        # Strategy Router - READ-ONLY
        assert controller.has_read_access("strategy_router") is True
        assert controller.has_write_access("strategy_router") is False

        # Risk Tools - READ-ONLY
        assert controller.has_read_access("risk_tools") is True
        assert controller.has_write_access("risk_tools") is False

        # Broker Tools - READ-ONLY
        assert controller.has_read_access("broker_tools") is True
        assert controller.has_write_access("broker_tools") is False

    def test_research_write_access_to_dev_tools(self):
        """Test Research has WRITE access to development tools."""
        controller = ToolAccessController(Department.RESEARCH)

        assert controller.has_write_access("pinescript_tools") is True
        assert controller.has_write_access("mql5_tools") is True
        assert controller.has_write_access("backtest_tools") is True


class TestAnalysisDepartmentAccess:
    """Test Analysis department only has READ access to EA lifecycle."""

    def test_analysis_read_only_ea_lifecycle(self):
        """Test Analysis has READ-ONLY access to EA lifecycle."""
        controller = ToolAccessController(Department.ANALYSIS)
        assert controller.has_read_access("ea_lifecycle") is True
        assert controller.has_write_access("ea_lifecycle") is False

    def test_analysis_read_only_shared_tools(self):
        """Test Analysis has READ-ONLY access to shared tools."""
        controller = ToolAccessController(Department.ANALYSIS)

        assert controller.has_read_access("strategy_router") is True
        assert controller.has_write_access("strategy_router") is False

        assert controller.has_read_access("risk_tools") is True
        assert controller.has_write_access("risk_tools") is False

        assert controller.has_read_access("broker_tools") is True
        assert controller.has_write_access("broker_tools") is False

    def test_analysis_write_strategy_extraction(self):
        """Test Analysis has WRITE access to strategy extraction."""
        controller = ToolAccessController(Department.ANALYSIS)
        assert controller.has_write_access("strategy_extraction") is True


class TestAllDepartmentsReadOnlyAccess:
    """Test all departments have READ-ONLY access to critical tools."""

    @pytest.mark.parametrize("department", [
        Department.ANALYSIS,
        Department.RESEARCH,
        Department.RISK,
        Department.EXECUTION,
        Department.PORTFOLIO,
    ])
    @pytest.mark.parametrize("tool", [
        "strategy_router",
        "risk_tools",
        "broker_tools",
    ])
    def test_critical_tools_read_only_for_all(self, department, tool):
        """Test Strategy Router, Risk/Position, Broker are READ-ONLY for ALL departments."""
        controller = ToolAccessController(department)

        # Should have READ access
        assert controller.has_read_access(tool) is True, \
            f"{department.value} should have READ access to {tool}"

        # Should NOT have WRITE access
        assert controller.has_write_access(tool) is False, \
            f"{department.value} should NOT have WRITE access to {tool}"


class TestDepartmentSpecificPermissions:
    """Test department-specific permission variations."""

    def test_risk_department_limited_access(self):
        """Test Risk department has limited access."""
        controller = ToolAccessController(Department.RISK)

        # Should have READ access to knowledge
        assert controller.has_read_access("knowledge_tools") is True
        assert controller.has_write_access("knowledge_tools") is False

    def test_execution_department_limited_access(self):
        """Test Execution department has limited access."""
        controller = ToolAccessController(Department.EXECUTION)

        # Should have READ access to knowledge
        assert controller.has_read_access("knowledge_tools") is True
        assert controller.has_write_access("knowledge_tools") is False

    def test_portfolio_department_strategy_extraction_read_only(self):
        """Test Portfolio has READ-ONLY access to strategy extraction."""
        controller = ToolAccessController(Department.PORTFOLIO)

        assert controller.has_read_access("strategy_extraction") is True
        assert controller.has_write_access("strategy_extraction") is False

    def test_research_write_strategy_extraction(self):
        """Test Research has WRITE access to strategy extraction."""
        controller = ToolAccessController(Department.RESEARCH)

        assert controller.has_read_access("strategy_extraction") is True
        assert controller.has_write_access("strategy_extraction") is True


class TestFloorManagerAccess:
    """Test Floor Manager has special cross-department access."""

    def test_floor_manager_can_read_all_tools(self):
        """Test Floor Manager has READ access to monitoring tools."""
        # Note: Floor Manager is not a Department enum value
        # It's accessed via the tool_access.py module directly
        from src.agents.departments.tool_access import TOOL_ACCESS

        floor_manager_perms = TOOL_ACCESS.get("floor_manager", {})

        assert ToolPermission.READ in floor_manager_perms.get("memory_all_depts", set())
        assert ToolPermission.READ in floor_manager_perms.get("strategy_router", set())
        assert ToolPermission.READ in floor_manager_perms.get("risk_tools", set())


class TestToolAccessControllerMethods:
    """Test ToolAccessController utility methods."""

    def test_controller_initialization(self):
        """Test controller initializes correctly."""
        controller = ToolAccessController(Department.RESEARCH)
        assert controller.department == Department.RESEARCH
        assert isinstance(controller.permissions, dict)

    def test_get_available_tools(self):
        """Test getting available tools for department."""
        controller = ToolAccessController(Department.RESEARCH)
        tools = controller.get_available_tools()

        assert isinstance(tools, list)
        assert "memory_tools" in tools
        assert "strategy_router" in tools
        assert "ea_lifecycle" in tools

    def test_filter_tools(self):
        """Test filtering tools by permissions."""
        controller = ToolAccessController(Department.RESEARCH)

        # All these tools should pass through
        input_tools = ["memory_tools", "strategy_router", "ea_lifecycle"]
        filtered = controller.filter_tools(input_tools)

        assert set(filtered) == set(input_tools)

    def test_can_access_with_unknown_tool(self):
        """Test unknown tools return False."""
        controller = ToolAccessController(Department.RESEARCH)

        assert controller.can_access("unknown_tool", ToolPermission.READ) is False
        assert controller.can_access("unknown_tool", ToolPermission.WRITE) is False


class TestMailSystemAccess:
    """Test all departments have READ/WRITE access to mail system."""

    @pytest.mark.parametrize("department", [
        Department.ANALYSIS,
        Department.RESEARCH,
        Department.RISK,
        Department.EXECUTION,
        Department.PORTFOLIO,
    ])
    def test_all_departments_have_mail_access(self, department):
        """Test all departments have READ/WRITE access to mail system."""
        controller = ToolAccessController(department)

        assert controller.has_read_access("mail") is True
        assert controller.has_write_access("mail") is True
