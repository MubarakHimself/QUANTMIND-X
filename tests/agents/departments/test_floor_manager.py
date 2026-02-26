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


class TestFloorManagerInit:
    """Test Floor Manager initialization."""

    def test_floor_manager_initializes_with_dependencies(self):
        """Floor Manager should initialize with mail service and spawner."""
        from src.agents.departments.floor_manager import FloorManager
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            assert manager.mail_service is not None
            assert manager.spawner is not None
            assert len(manager.departments) == 5

            manager.close()

    def test_floor_manager_has_all_departments(self):
        """Floor Manager should have all 5 departments configured."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            dept_values = {d.value for d in manager.departments.keys()}
            expected = {"analysis", "research", "risk", "execution", "portfolio"}
            assert dept_values == expected

            manager.close()

    def test_floor_manager_model_is_opus(self):
        """Floor Manager should use opus tier."""
        from src.agents.departments.floor_manager import FloorManager
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            assert manager.model_tier == "opus"

            manager.close()


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


class TestFloorManagerDispatch:
    """Test Floor Manager dispatch functionality."""

    def test_dispatch_creates_mail_message(self):
        """Dispatch should create a mail message to department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        from src.agents.departments.department_mail import MessageType
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            result = manager.dispatch(
                to_dept=Department.ANALYSIS,
                task="Analyze EURUSD for opportunities",
                priority="normal",
            )

            assert result["status"] == "dispatched"
            assert result["to_dept"] == "analysis"

            messages = manager.mail_service.check_inbox("analysis")
            assert len(messages) == 1
            assert messages[0].type == MessageType.DISPATCH
            assert "EURUSD" in messages[0].body

            manager.close()

    def test_dispatch_with_high_priority(self):
        """Dispatch with high priority should be reflected in mail."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        from src.agents.departments.department_mail import Priority
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            manager.dispatch(
                to_dept=Department.EXECUTION,
                task="URGENT: Execute order immediately",
                priority="high",
            )

            messages = manager.mail_service.check_inbox("execution")
            assert len(messages) == 1
            assert messages[0].priority == Priority.HIGH

            manager.close()

    def test_process_incoming_task(self):
        """Process should classify and dispatch task to correct department."""
        from src.agents.departments.floor_manager import FloorManager
        from src.agents.departments.types import Department
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            manager = FloorManager(mail_db_path=db_path)

            result = manager.process("Backtest the RSI strategy on EURUSD")

            assert result["classified_dept"] == "research"
            assert result["dispatch"]["to_dept"] == "research"
            assert result["dispatch"]["status"] == "dispatched"

            messages = manager.mail_service.check_inbox("research")
            assert len(messages) == 1

            manager.close()
