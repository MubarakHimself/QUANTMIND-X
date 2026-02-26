# tests/agents/departments/test_floor_manager.py
"""
Tests for Floor Manager

Task Group: Core Infrastructure - Department routing
"""
import pytest


class TestDepartmentTypes:
    """Test department type definitions."""

    def test_department_enum_values(self):
        """Department enum should have 5 trading floor departments."""
        from src.agents.departments.types import Department

        assert Department.ANALYSIS.value == "analysis"
        assert Department.RESEARCH.value == "research"
        assert Department.RISK.value == "risk"
        assert Department.EXECUTION.value == "execution"
        assert Department.PORTFOLIO.value == "portfolio"

    def test_department_model_tier_mapping(self):
        """Each department should have a model tier."""
        from src.agents.departments.types import Department, get_model_tier

        # Department heads use sonnet
        assert get_model_tier(Department.ANALYSIS) == "sonnet"
        assert get_model_tier(Department.RESEARCH) == "sonnet"
        assert get_model_tier(Department.RISK) == "sonnet"
        assert get_model_tier(Department.EXECUTION) == "sonnet"
        assert get_model_tier(Department.PORTFOLIO) == "sonnet"


class TestDepartmentHeadConfig:
    """Test department head configuration."""

    def test_department_head_config_creation(self):
        """Should create department head config with all required fields."""
        from src.agents.departments.types import DepartmentHeadConfig, Department

        config = DepartmentHeadConfig(
            department=Department.ANALYSIS,
            agent_type="analyst_head",
            system_prompt="You are the Analysis Department Head...",
            sub_agents=["market_analyst", "sentiment_scanner", "news_monitor"],
            memory_namespace="analysis",
        )

        assert config.department == Department.ANALYSIS
        assert config.agent_type == "analyst_head"
        assert len(config.sub_agents) == 3
        assert "market_analyst" in config.sub_agents

    def test_all_departments_have_configs(self):
        """All 5 departments should have default configs."""
        from src.agents.departments.types import get_department_configs

        configs = get_department_configs()
        assert len(configs) == 5

        dept_names = {c.department.value for c in configs.values()}
        assert dept_names == {"analysis", "research", "risk", "execution", "portfolio"}


class TestFloorManagerRouting:
    """Test Floor Manager task routing."""

    def test_route_task_to_analysis(self):
        """Should route analysis-related tasks to Analysis department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            tasks = [
                "Analyze EURUSD for trading opportunities",
                "What is the market sentiment for gold?",
                "Scan news for GBPUSD",
            ]

            for task in tasks:
                dept = manager.classify_task(task)
                assert dept == Department.ANALYSIS, f"Expected ANALYSIS for: {task}"

            manager.close()

    def test_route_task_to_research(self):
        """Should route research-related tasks to Research department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            tasks = [
                "Develop a new moving average strategy",
                "Backtest the RSI strategy on EURUSD",
                "Research alpha factors for gold",
            ]

            for task in tasks:
                dept = manager.classify_task(task)
                assert dept == Department.RESEARCH, f"Expected RESEARCH for: {task}"

            manager.close()

    def test_route_task_to_risk(self):
        """Should route risk-related tasks to Risk department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            tasks = [
                "Calculate position size for EURUSD trade",
                "Check drawdown limits",
                "Compute Value at Risk for portfolio",
            ]

            for task in tasks:
                dept = manager.classify_task(task)
                assert dept == Department.RISK, f"Expected RISK for: {task}"

            manager.close()

    def test_route_task_to_execution(self):
        """Should route execution-related tasks to Execution department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            tasks = [
                "Execute buy order for EURUSD",
                "Route order to best venue",
                "Track fill for GBPUSD order",
            ]

            for task in tasks:
                dept = manager.classify_task(task)
                assert dept == Department.EXECUTION, f"Expected EXECUTION for: {task}"

            manager.close()

    def test_route_task_to_portfolio(self):
        """Should route portfolio-related tasks to Portfolio department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            tasks = [
                "Rebalance portfolio allocation",
                "Track portfolio performance",
                "Optimize asset allocation",
            ]

            for task in tasks:
                dept = manager.classify_task(task)
                assert dept == Department.PORTFOLIO, f"Expected PORTFOLIO for: {task}"

            manager.close()

    def test_route_unknown_task_to_analysis_default(self):
        """Should route unknown tasks to Analysis department as default."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            # Task with no matching keywords
            task = "Do something completely unrelated"
            dept = manager.classify_task(task)
            assert dept == Department.ANALYSIS, "Unknown tasks should default to ANALYSIS"

            manager.close()

    def test_route_multi_keyword_task(self):
        """Should route task with multiple keywords to highest scoring department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            # Task with analysis keywords (should score higher than research)
            task = "Analyze the market technical indicators and sentiment"
            dept = manager.classify_task(task)
            # Has "analyze", "technical", "sentiment" -> 3 analysis keywords
            # This should route to ANALYSIS
            assert dept == Department.ANALYSIS, f"Expected ANALYSIS for multi-keyword task"

            manager.close()
