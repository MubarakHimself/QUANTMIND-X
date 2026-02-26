# tests/agents/departments/heads/test_all_heads.py
"""
Tests for all Department Heads

Task Group: Department Heads - All departments
"""
import pytest
import tempfile
import os


class TestAllDepartmentHeads:
    """Test all department heads have correct configuration."""

    @pytest.mark.parametrize("dept,agent_type,expected_workers", [
        ("research", "research_head", ["strategy_researcher", "backtester", "data_scientist"]),
        ("risk", "risk_head", ["position_sizer", "drawdown_monitor", "var_calculator"]),
        ("execution", "executor_head", ["order_router", "fill_tracker", "slippage_monitor"]),
        ("portfolio", "portfolio_head", ["allocation_manager", "rebalancer", "performance_tracker"]),
    ])
    def test_department_head_configuration(self, dept, agent_type, expected_workers):
        """Each department head should have correct configuration."""
        from src.agents.departments.types import Department
        from src.agents.departments.heads.research_head import ResearchHead
        from src.agents.departments.heads.risk_head import RiskHead
        from src.agents.departments.heads.execution_head import ExecutionHead
        from src.agents.departments.heads.portfolio_head import PortfolioHead

        head_classes = {
            "research": ResearchHead,
            "risk": RiskHead,
            "execution": ExecutionHead,
            "portfolio": PortfolioHead,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            HeadClass = head_classes[dept]
            head = HeadClass(mail_db_path=db_path)

            assert head.department.value == dept
            assert head.agent_type == agent_type
            assert head.sub_agents == expected_workers
            assert head.model_tier == "sonnet"

            head.close()

    def test_all_heads_have_tools(self):
        """All department heads should define tools."""
        from src.agents.departments.heads.research_head import ResearchHead
        from src.agents.departments.heads.risk_head import RiskHead
        from src.agents.departments.heads.execution_head import ExecutionHead
        from src.agents.departments.heads.portfolio_head import PortfolioHead

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")

            for HeadClass in [ResearchHead, RiskHead, ExecutionHead, PortfolioHead]:
                head = HeadClass(mail_db_path=db_path)
                tools = head.get_tools()
                assert len(tools) > 0, f"{HeadClass.__name__} should have tools"
                head.close()
