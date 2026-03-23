"""P0 Tests: FloorManager Department Routing Accuracy (Epic 5 - R-007)

These tests verify that FloorManager correctly routes tasks to the
appropriate department based on keyword classification.

Test Design Reference: _bmad-output/test-artifacts/test-design-epic-5.md
Risk: R-007 (Score 6) - FloorManager routing misclassification
Priority: P0 - High risk (>=6), no workaround
"""

import pytest
import tempfile
from unittest.mock import MagicMock, patch

from src.agents.departments.floor_manager import FloorManager
from src.agents.departments.types import Department


class TestFloorManagerDepartmentRouting:
    """Test R-007: FloorManager department routing accuracy."""

    def test_routes_research_task_to_research_department(self):
        """P0: Task containing 'research' MUST be routed to RESEARCH department.

        FloorManager.classify_task is the core routing method that must
        correctly identify department based on keywords.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.departments.floor_manager.get_redis_mail_service"):
                floor_manager = FloorManager(
                    mail_db_path=f"{tmpdir}/test_mail.db",
                    use_redis_mail=False,
                )

                # Test various research-related tasks
                research_tasks = [
                    "run research on EURUSD",
                    "analyze market sentiment for GBPUSD",
                    "develop a new trading strategy",
                    "backtest this approach on 2023 data",
                    "scan for trading opportunities",
                    "find alpha factors",
                ]

                for task in research_tasks:
                    dept = floor_manager.classify_task(task)

                    assert dept == Department.RESEARCH, (
                        f"Task '{task}' was routed to {dept.value} instead of research! "
                        f"Routing accuracy is critical for correct system operation."
                    )

    def test_routes_development_task_to_development_department(self):
        """P0: Task containing development keywords MUST be routed to DEVELOPMENT.

        Tasks about building EAs, bots, code, etc. must go to Development.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.departments.floor_manager.get_redis_mail_service"):
                floor_manager = FloorManager(
                    mail_db_path=f"{tmpdir}/test_mail.db",
                    use_redis_mail=False,
                )

                # Test various development-related tasks
                development_tasks = [
                    "compile EA for EURUSD",
                    "build an Expert Advisor",
                    "implement this strategy in MQL5",
                    "write Python script for analysis",
                    "code a trading bot",
                    "automate this manual process",
                ]

                for task in development_tasks:
                    dept = floor_manager.classify_task(task)

                    assert dept == Department.DEVELOPMENT, (
                        f"Task '{task}' was routed to {dept.value} instead of development! "
                        f"Routing accuracy is critical for correct system operation."
                    )

    def test_routes_risk_task_to_risk_department(self):
        """P0: Task containing risk keywords MUST be routed to RISK department.

        Tasks about risk management, position sizing, drawdown must
        go to the Risk department.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.departments.floor_manager.get_redis_mail_service"):
                floor_manager = FloorManager(
                    mail_db_path=f"{tmpdir}/test_mail.db",
                    use_redis_mail=False,
                )

                # Test various risk-related tasks
                risk_tasks = [
                    "show risk metrics",
                    "calculate position size for this trade",
                    "what is the current drawdown",
                    "check var exposure",
                    "update stop loss limits",
                    "analyze margin requirements",
                ]

                for task in risk_tasks:
                    dept = floor_manager.classify_task(task)

                    assert dept == Department.RISK, (
                        f"Task '{task}' was routed to {dept.value} instead of risk! "
                        f"Routing accuracy is critical for correct system operation."
                    )

    def test_routes_trading_task_to_trading_department(self):
        """P0: Task containing trading execution keywords MUST go to TRADING.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.departments.floor_manager.get_redis_mail_service"):
                floor_manager = FloorManager(
                    mail_db_path=f"{tmpdir}/test_mail.db",
                    use_redis_mail=False,
                )

                # Test various trading-related tasks
                trading_tasks = [
                    "execute buy order for EURUSD",
                    "close all positions",
                    "check order fill status",
                    "route order to broker",
                    "paper trade this strategy",
                    "submit sell order",
                ]

                for task in trading_tasks:
                    dept = floor_manager.classify_task(task)

                    assert dept == Department.TRADING, (
                        f"Task '{task}' was routed to {dept.value} instead of trading! "
                        f"Routing accuracy is critical for correct system operation."
                    )

    def test_routes_portfolio_task_to_portfolio_department(self):
        """P0: Task containing portfolio keywords MUST go to PORTFOLIO.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.departments.floor_manager.get_redis_mail_service"):
                floor_manager = FloorManager(
                    mail_db_path=f"{tmpdir}/test_mail.db",
                    use_redis_mail=False,
                )

                # Test various portfolio-related tasks
                portfolio_tasks = [
                    "rebalance portfolio",
                    "check allocation",
                    "analyze performance",
                    "diversify holdings",
                    "calculate portfolio value",
                    "attribution analysis",
                ]

                for task in portfolio_tasks:
                    dept = floor_manager.classify_task(task)

                    assert dept == Department.PORTFOLIO, (
                        f"Task '{task}' was routed to {dept.value} instead of portfolio! "
                        f"Routing accuracy is critical for correct system operation."
                    )

    def test_default_to_research_for_unknown_tasks(self):
        """P0: Unclassifiable tasks MUST default to RESEARCH department.

        When no keywords match, the system should default to Research
        rather than failing or returning None.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.departments.floor_manager.get_redis_mail_service"):
                floor_manager = FloorManager(
                    mail_db_path=f"{tmpdir}/test_mail.db",
                    use_redis_mail=False,
                )

                # These tasks have no clear keyword matches
                unknown_tasks = [
                    "hello",
                    "what can you do",
                    "status check",
                    "help",
                    "just checking in",
                ]

                for task in unknown_tasks:
                    dept = floor_manager.classify_task(task)

                    assert dept == Department.RESEARCH, (
                        f"Unknown task '{task}' was routed to {dept.value} instead of "
                        f"defaulting to research! Unknown tasks must have a default."
                    )

    def test_task_classification_is_deterministic(self):
        """P0: Same task MUST always be routed to the same department.

        Routing must be deterministic for the system to be predictable.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.agents.departments.floor_manager.get_redis_mail_service"):
                floor_manager = FloorManager(
                    mail_db_path=f"{tmpdir}/test_mail.db",
                    use_redis_mail=False,
                )

                task = "analyze the market for trading opportunities"

                # Classify the same task multiple times
                results = [floor_manager.classify_task(task) for _ in range(10)]

                # All results should be the same
                assert all(r == results[0] for r in results), (
                    f"Task classification is not deterministic! "
                    f"Same task '{task}' was routed to different departments: {results}"
                )
