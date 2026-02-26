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
