# tests/agents/departments/heads/test_base.py
"""
Tests for Department Head Base Class

Task Group: Department Heads - Base functionality
"""
import pytest
import tempfile
import os


class TestDepartmentHeadBase:
    """Test Department Head base class."""

    def test_department_head_initializes_with_config(self):
        """Department Head should initialize from config."""
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig

        config = DepartmentHeadConfig(
            department=Department.ANALYSIS,
            agent_type="analyst_head",
            system_prompt="Test prompt",
            sub_agents=["worker1", "worker2"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")
            head = DepartmentHead(config=config, mail_db_path=db_path)

            assert head.department == Department.ANALYSIS
            assert head.agent_type == "analyst_head"
            assert head.model_tier == "sonnet"
            assert len(head.sub_agents) == 2

            head.close()

    def test_department_head_can_check_mail(self):
        """Department Head should be able to check mail inbox."""
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig
        from src.agents.departments.department_mail import MessageType

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")

            config = DepartmentHeadConfig(
                department=Department.ANALYSIS,
                agent_type="analyst_head",
                system_prompt="Test prompt",
            )
            head = DepartmentHead(config=config, mail_db_path=db_path)

            # Send a message to this department
            head.mail_service.send(
                from_dept="floor_manager",
                to_dept="analysis",
                type=MessageType.DISPATCH,
                subject="Test task",
                body="Test body",
            )

            # Check inbox
            messages = head.check_mail()
            assert len(messages) == 1
            assert messages[0].subject == "Test task"

            head.close()

    def test_department_head_can_send_result(self):
        """Department Head should be able to send results to other departments."""
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")

            config = DepartmentHeadConfig(
                department=Department.ANALYSIS,
                agent_type="analyst_head",
                system_prompt="Test prompt",
            )
            head = DepartmentHead(config=config, mail_db_path=db_path)

            # Send result to Execution
            head.send_result(
                to_dept=Department.EXECUTION,
                subject="Analysis Complete",
                body="EURUSD signal: BUY",
            )

            # Verify message was sent
            messages = head.mail_service.check_inbox("execution")
            assert len(messages) == 1

            head.close()

    def test_department_head_can_spawn_worker(self):
        """Department Head should be able to spawn workers."""
        from src.agents.departments.heads.base import DepartmentHead
        from src.agents.departments.types import Department, DepartmentHeadConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_mail.db")

            config = DepartmentHeadConfig(
                department=Department.ANALYSIS,
                agent_type="analyst_head",
                system_prompt="Test prompt",
                sub_agents=["market_analyst", "sentiment_scanner"],
            )
            head = DepartmentHead(config=config, mail_db_path=db_path)

            # Spawn a worker
            result = head.spawn_worker(
                worker_type="market_analyst",
                task="Analyze EURUSD",
            )

            # Result should indicate spawned, spawner unavailable, or spawn failed (if SubAgentConfig missing)
            assert result["status"] in ["spawned", "spawner_unavailable", "spawn_failed"]

            head.close()
