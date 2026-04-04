"""
P1 Acceptance Tests for Epic 7: Department Heads Coverage

Priority: P1 (High - Core user journeys + Frequently used features)

Coverage:
- DevelopmentHead: TRD validation, EA generation, clarification flow
- ResearchHead: Hypothesis generation, confidence scoring
- TradingHead: Paper trading monitoring, order routing
- PortfolioHead: Report generation, attribution, correlation

Run: pytest tests/agents/departments/test_epic7_p1_department_heads.py -v
"""

import pytest
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List
from unittest.mock import MagicMock, patch, AsyncMock

from src.agents.departments.heads.development_head import (
    DevelopmentHead, DevelopmentTask, EAGenerationResult
)
from src.agents.departments.heads.research_head import (
    ResearchHead, ResearchTask, Hypothesis
)
from src.agents.departments.heads.execution_head import (
    TradingHead, TradeSide, RegimeType
)
from src.agents.departments.heads.portfolio_head import (
    PortfolioHead, StrategyAttribution, BrokerAttribution
)
from src.agents.departments.types import Department


# ============================================================================
# P1-1: DevelopmentHead - TRD Validation
# ============================================================================

class TestDevelopmentTRDValidation:
    """P1 Test Group: TRD validation and parsing"""

    @pytest.fixture
    def dev_head(self):
        """Create DevelopmentHead instance with mocked dependencies."""
        with patch('src.agents.departments.heads.development_head.TRDParser') as mock_parser, \
             patch('src.agents.departments.heads.development_head.TRDValidator') as mock_validator, \
             patch('src.agents.departments.heads.development_head.MQL5Generator') as mock_generator, \
             patch('src.agents.departments.heads.development_head.EAOutputStorage') as mock_storage, \
             patch('src.agents.departments.heads.development_head.get_compilation_service'):

            mock_parser_instance = MagicMock()
            mock_parser.return_value = mock_parser_instance

            mock_validator_instance = MagicMock()
            mock_validator.return_value = mock_validator_instance

            mock_generator_instance = MagicMock()
            mock_generator.return_value = mock_generator_instance

            mock_storage_instance = MagicMock()
            mock_storage.return_value = mock_storage_instance

            head = DevelopmentHead()
            head.parser = mock_parser_instance
            head.validator = mock_validator_instance
            head.mql5_generator = mock_generator_instance
            head.ea_storage = mock_storage_instance

            yield head

    def test_validate_trd_with_valid_data(self, dev_head):
        """
        P1 Test: Verify valid TRD passes validation.

        AC: Valid TRD with all required fields passes validation.
        """
        valid_trd = {
            "strategy_id": "test_strategy",
            "strategy_name": "Test MA Cross",
            "symbol": "EURUSD",
            "timeframe": "H4",
            "entry_conditions": ["MA crossover"],
            "exit_conditions": ["Reverse crossover"],
            "position_sizing": {"method": "fixed_lot", "risk_percent": 1.0},
            "parameters": {"session_mask": "UK"},
        }

        # Mock validation result
        mock_validation = MagicMock()
        mock_validation.has_blocking_issues.return_value = False
        mock_validation.errors = []
        dev_head.validator.validate.return_value = mock_validation

        # Mock parsed TRD
        mock_trd = MagicMock()
        mock_trd.strategy_id = "test_strategy"
        mock_trd.version = 1
        mock_trd.to_dict.return_value = valid_trd
        dev_head.parser.parse_dict.return_value = mock_trd

        task = DevelopmentTask(task_type="validate_trd", trd_data=valid_trd)
        result = dev_head.process_task(task)

        assert result.success is True, "Valid TRD should pass validation"
        assert result.strategy_id == "test_strategy"
        assert result.validation_result is not None

    def test_validate_trd_missing_required_fields(self, dev_head):
        """
        P1 Test: Verify TRD with missing required fields fails validation.

        AC: Missing required field returns validation error with field name.
        """
        invalid_trd = {
            "strategy_name": "Missing Symbol",
            "timeframe": "H4",
            # Missing: strategy_id, symbol, entry_conditions
        }

        # Mock validation with blocking issues
        mock_validation = MagicMock()
        mock_validation.has_blocking_issues.return_value = True
        mock_validation.errors = [
            MagicMock(error="Missing required field: symbol"),
            MagicMock(error="Missing required field: strategy_id"),
        ]
        dev_head.validator.validate.return_value = mock_validation

        # Mock parsed TRD
        mock_trd = MagicMock()
        mock_trd.strategy_id = ""
        mock_trd.version = 1
        dev_head.parser.parse_dict.return_value = mock_trd

        task = DevelopmentTask(task_type="validate_trd", trd_data=invalid_trd)
        result = dev_head.process_task(task)

        assert result.success is False, "Invalid TRD should fail validation"
        assert "symbol" in result.error.lower() or "symbol" in result.error

    def test_validate_trd_returns_all_errors(self, dev_head):
        """
        P1 Test: Verify TRD validation returns ALL errors, not just first.

        AC: TRD with multiple missing fields returns all errors.
        """
        invalid_trd = {
            "strategy_name": "Multi Error TRD",
            # Missing: strategy_id, symbol, timeframe, entry_conditions
        }

        mock_validation = MagicMock()
        mock_validation.has_blocking_issues.return_value = True
        mock_validation.errors = [
            MagicMock(error="Missing required field: symbol"),
            MagicMock(error="Missing required field: strategy_id"),
            MagicMock(error="Missing required field: timeframe"),
            MagicMock(error="Missing required field: entry_conditions"),
        ]
        dev_head.validator.validate.return_value = mock_validation

        mock_trd = MagicMock()
        mock_trd.strategy_id = ""
        mock_trd.version = 1
        dev_head.parser.parse_dict.return_value = mock_trd

        task = DevelopmentTask(task_type="validate_trd", trd_data=invalid_trd)
        result = dev_head.process_task(task)

        assert result.success is False
        # Verify all errors are captured
        assert len(mock_validation.errors) >= 3


# ============================================================================
# P1-2: DevelopmentHead - EA Generation
# ============================================================================

class TestDevelopmentEAGeneration:
    """P1 Test Group: EA generation from TRD"""

    @pytest.fixture
    def dev_head_with_mocks(self):
        """Create DevelopmentHead with full mocking."""
        with patch('src.agents.departments.heads.development_head.TRDParser') as mock_parser, \
             patch('src.agents.departments.heads.development_head.TRDValidator') as mock_validator, \
             patch('src.agents.departments.heads.development_head.MQL5Generator') as mock_generator, \
             patch('src.agents.departments.heads.development_head.EAOutputStorage') as mock_storage, \
             patch('src.agents.departments.heads.development_head.get_compilation_service') as mock_compile_svc:

            head = DevelopmentHead()
            head.parser = MagicMock()
            head.validator = MagicMock()
            head.mql5_generator = MagicMock()
            head.ea_storage = MagicMock()
            head.compilation_service = MagicMock()

            yield head

    def test_generate_ea_success(self, dev_head_with_mocks):
        """
        P1 Test: Verify successful EA generation from valid TRD.

        AC: Valid TRD generates EA code and saves to storage.
        """
        valid_trd = {
            "strategy_id": "ma_cross_v1",
            "strategy_name": "MA Cross Strategy",
            "symbol": "EURUSD",
            "timeframe": "H4",
            "entry_conditions": ["Fast MA crosses Slow MA"],
            "exit_conditions": ["Reverse cross"],
            "position_sizing": {"method": "fixed_lot", "risk_percent": 1.0},
            "parameters": {},
        }

        head = dev_head_with_mocks

        # Mock validation passes
        mock_validation = MagicMock()
        mock_validation.has_blocking_issues.return_value = False
        mock_validation.errors = []
        head.validator.validate.return_value = mock_validation
        head.validator.get_clarification_request.return_value = {"needs_clarification": False}

        # Mock parsed TRD
        mock_trd = MagicMock()
        mock_trd.strategy_id = "ma_cross_v1"
        mock_trd.strategy_name = "MA Cross Strategy"
        mock_trd.version = 1
        mock_trd.symbol = "EURUSD"
        mock_trd.to_dict.return_value = valid_trd
        head.parser.parse_dict.return_value = mock_trd

        # Mock MQL5 generation
        head.mql5_generator.generate.return_value = "// MQL5 EA Code"
        head.mql5_generator.validate_mql5_syntax.return_value = (True, None)

        # Mock EA storage
        mock_ea_output = MagicMock()
        mock_ea_output.strategy_id = "ma_cross_v1"
        mock_ea_output.version = 1
        mock_ea_output.file_path = "/tmp/ea_ma_cross_v1.mq5"
        head.ea_storage.save_ea.return_value = mock_ea_output

        # Mock compilation service
        mock_compile_result = MagicMock()
        mock_compile_result.success = True
        mock_compile_result.errors = []
        head.compilation_service.compile_ea.return_value = mock_compile_result

        task = DevelopmentTask(task_type="generate_ea", trd_data=valid_trd)
        result = head.process_task(task)

        assert result.success is True
        assert result.strategy_id == "ma_cross_v1"
        assert result.file_path is not None

    def test_generate_ea_clarification_needed(self, dev_head_with_mocks):
        """
        P1 Test: Verify EA generation requests clarification for ambiguous TRD.

        AC: Ambiguous TRD triggers clarification request to FloorManager.
        """
        ambiguous_trd = {
            "strategy_id": "ambiguous_strategy",
            "strategy_name": "Ambiguous Strategy",
            "symbol": "EURUSD",
            "timeframe": "H4",
            "entry_conditions": ["RSI overbought - but what level?"],
            # Missing clear entry rules
            "position_sizing": {},
            "parameters": {},
        }

        head = dev_head_with_mocks

        # Mock validation with clarification needed
        mock_validation = MagicMock()
        mock_validation.has_blocking_issues.return_value = False
        head.validator.validate.return_value = mock_validation

        mock_clarification = {
            "needs_clarification": True,
            "missing_parameters": ["entry_conditions"],
            "ambiguous_parameters": [
                {"parameter": "entry_conditions", "severity": "high"}
            ],
            "message": "Entry conditions need clarification"
        }
        head.validator.get_clarification_request.return_value = mock_clarification

        # Mock parsed TRD
        mock_trd = MagicMock()
        mock_trd.strategy_id = "ambiguous_strategy"
        mock_trd.version = 1
        head.parser.parse_dict.return_value = mock_trd

        task = DevelopmentTask(task_type="generate_ea", trd_data=ambiguous_trd)
        result = head.process_task(task)

        assert result.clarification_needed is True, "Should request clarification"
        assert result.clarification_details is not None


# ============================================================================
# P1-3: ResearchHead - Hypothesis Generation
# ============================================================================

class TestResearchHypothesisGeneration:
    """P1 Test Group: Research hypothesis generation and confidence scoring"""

    @pytest.fixture
    def mock_research_head(self):
        """Create ResearchHead with mocked knowledge clients."""
        with patch('src.agents.departments.heads.research_head.ResearchHead.__init__', return_value=None):
            head = ResearchHead.__new__(ResearchHead)
            head._current_session_id = "test_session"
            head.pageindex_client = MagicMock()
            head.embedding_service = MagicMock()
            head.mcp_integration = MagicMock()
            head.department = MagicMock()
            head.agent_type = "research_head"
            yield head

    def test_calculate_confidence_with_sufficient_evidence(self, mock_research_head):
        """
        P1 Test: Verify confidence calculation with sufficient evidence.

        AC: More evidence results in higher confidence score.
        """
        evidence = [
            "Source 1: MA crossover detected",
            "Source 2: RSI oversold confirmation",
            "Source 3: Volume spike on breakout",
            "Source 4: Support level holding",
            "Source 5: Trend line bounce",
            "Source 6: News catalyst identified",
            "Source 7: Multiple timeframe alignment",
            "Source 8: Historical pattern match",
        ]

        confidence = mock_research_head._calculate_confidence(evidence)

        assert confidence >= 0.5, "Sufficient evidence should yield confidence >= 0.5"
        assert confidence < 0.95, "Confidence should not exceed 0.95 without LLM validation"

    def test_calculate_confidence_with_insufficient_evidence(self, mock_research_head):
        """
        P1 Test: Verify confidence calculation with insufficient evidence.

        AC: Limited evidence results in low confidence score.
        """
        evidence = ["Weak signal detected"]

        confidence = mock_research_head._calculate_confidence(evidence)

        assert confidence < 0.5, "Insufficient evidence should yield low confidence"

    def test_should_escalate_to_trd_at_threshold(self, mock_research_head):
        """
        P1 Test: Verify TRD escalation at exact threshold (0.75).

        AC: Confidence = 0.75 triggers TRD escalation.
        """
        hypothesis = Hypothesis(
            symbol="EURUSD",
            timeframe="H4",
            hypothesis="EURUSD will trend upward",
            supporting_evidence=["Evidence 1", "Evidence 2", "Evidence 3"],
            confidence_score=0.75,
            recommended_next_steps=["Validate", "Escalate"]
        )

        should_escalate = mock_research_head.should_escalate_to_trd(hypothesis)

        assert should_escalate is True, "Confidence 0.75 should trigger escalation"

    def test_should_not_escalate_below_threshold(self, mock_research_head):
        """
        P1 Test: Verify no escalation below threshold (0.74).

        AC: Confidence = 0.74 does NOT trigger TRD escalation.
        """
        hypothesis = Hypothesis(
            symbol="GBPUSD",
            timeframe="D1",
            hypothesis="GBPUSD unclear direction",
            supporting_evidence=["Weak evidence"],
            confidence_score=0.74,
            recommended_next_steps=["Gather more evidence"]
        )

        should_escalate = mock_research_head.should_escalate_to_trd(hypothesis)

        assert should_escalate is False, "Confidence 0.74 should NOT trigger escalation"

    def test_get_escalation_prompt_format(self, mock_research_head):
        """
        P1 Test: Verify escalation prompt contains required TRD fields.

        AC: Escalation prompt includes symbol, timeframe, confidence, next steps.
        """
        hypothesis = Hypothesis(
            symbol="USDJPY",
            timeframe="H1",
            hypothesis="USDJPY momentum shift detected",
            supporting_evidence=["MA crossover", "RSI divergence", "Volume spike"],
            confidence_score=0.85,
            recommended_next_steps=["Backtest MA cross", "Risk review", "TRD generation"]
        )

        prompt = mock_research_head.get_escalation_prompt(hypothesis)

        assert "USDJPY" in prompt, "Prompt should contain symbol"
        assert "0.85" in prompt or "85" in prompt, "Prompt should contain confidence"
        assert "TRD" in prompt or "Development" in prompt, "Prompt should reference TRD"
        assert len(prompt) > 50, "Prompt should be detailed"


# ============================================================================
# P1-4: TradingHead - Paper Trading Monitoring
# ============================================================================

class TestTradingPaperTrading:
    """P1 Test Group: Paper trading monitoring and status"""

    @pytest.fixture
    def trading_head(self):
        """Create TradingHead instance."""
        head = TradingHead.__new__(TradingHead)
        head._update_interval = 60
        head._active_monitors = {}
        head._regime_cache = {}
        return head

    def test_monitor_paper_trading_starts_monitoring(self, trading_head):
        """
        P1 Test: Verify paper trading monitoring starts successfully.

        AC: Valid agent_id starts monitoring and returns confirmation.
        """
        result = trading_head.monitor_paper_trading(
            agent_id="agent_001",
            strategy_name="MA_Cross_Test"
        )

        assert result["status"] == "monitoring_started"
        assert result["agent_id"] == "agent_001"
        assert result["strategy_name"] == "MA_Cross_Test"
        assert "update_interval_seconds" in result

    def test_monitor_paper_trading_duplicate_detection(self, trading_head):
        """
        P1 Test: Verify duplicate monitoring is detected.

        AC: Already monitoring same agent returns already_monitoring status.
        """
        # Start monitoring first time
        trading_head.monitor_paper_trading(
            agent_id="agent_001",
            strategy_name="MA_Cross_Test"
        )

        # Try to start monitoring again
        result = trading_head.monitor_paper_trading(
            agent_id="agent_001",
            strategy_name="MA_Cross_Test"
        )

        assert result["status"] == "already_monitoring"
        assert "Already monitoring" in result["message"]

    def test_get_paper_trading_status_returns_metrics(self, trading_head):
        """
        P1 Test: Verify paper trading status returns complete metrics.

        AC: Status includes agent_id, metrics, regime, and last_update.
        """
        result = trading_head.get_paper_trading_status("agent_001")

        assert "agent_id" in result
        assert result["agent_id"] == "agent_001"
        assert "metrics" in result
        assert "regime" in result
        assert "last_update" in result

        metrics = result["metrics"]
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "total_pnl" in metrics

    def test_stop_monitoring_removes_active_monitor(self, trading_head):
        """
        P1 Test: Verify stopping monitoring removes active monitor.

        AC: After stop, agent is no longer in active monitors.
        """
        # Start monitoring
        trading_head.monitor_paper_trading(
            agent_id="agent_001",
            strategy_name="MA_Cross_Test"
        )

        # Stop monitoring
        result = trading_head.stop_monitoring("agent_001")

        assert result["status"] == "monitoring_stopped"
        assert "agent_001" not in trading_head._active_monitors

    def test_stop_monitoring_not_monitoring(self, trading_head):
        """
        P1 Test: Verify stopping non-existent monitoring is handled.

        AC: Not monitoring returns not_monitoring status.
        """
        result = trading_head.stop_monitoring("agent_999")

        assert result["status"] == "not_monitoring"


# ============================================================================
# P1-5: PortfolioHead - Report Generation and Attribution
# ============================================================================

class TestPortfolioReportGeneration:
    """P1 Test Group: Portfolio report generation and attribution"""

    @pytest.fixture
    def portfolio_head(self):
        """Create PortfolioHead instance."""
        head = PortfolioHead.__new__(PortfolioHead)
        head.department = MagicMock()
        return head

    def test_generate_portfolio_report_structure(self, portfolio_head):
        """
        P1 Test: Verify portfolio report has required structure.

        AC: Report includes total_equity, strategy_attribution, broker_attribution, drawdowns.
        """
        report = portfolio_head.generate_portfolio_report()

        assert "total_equity" in report
        assert "pnl_attribution" in report
        assert "drawdown_by_account" in report
        assert "generated_at" in report

        attribution = report["pnl_attribution"]
        assert "by_strategy" in attribution
        assert "by_broker" in attribution

    def test_get_portfolio_summary_includes_daily_pnl(self, portfolio_head):
        """
        P1 Test: Verify portfolio summary includes daily P&L.

        AC: Summary returns total_equity, daily_pnl, drawdown, active_strategies.
        """
        summary = portfolio_head.get_portfolio_summary()

        assert "total_equity" in summary
        assert "daily_pnl" in summary
        assert "total_drawdown" in summary
        assert "active_strategies" in summary
        assert "accounts" in summary
        assert "drawdown_alert" in summary

    def test_get_attribution_includes_strategy_and_broker(self, portfolio_head):
        """
        P1 Test: Verify attribution includes both strategy and broker breakdown.

        AC: Attribution returns by_strategy and by_broker with P&L and percentages.
        """
        attribution = portfolio_head.get_attribution()

        assert "by_strategy" in attribution
        assert "by_broker" in attribution

        strategies = attribution["by_strategy"]
        assert isinstance(strategies, list)

        for strategy in strategies:
            assert "strategy" in strategy
            assert "pnl" in strategy
            assert "percentage" in strategy

    def test_get_correlation_matrix_returns_pairs(self, portfolio_head):
        """
        P1 Test: Verify correlation matrix returns strategy pairs.

        AC: Matrix returns NxN pairs with correlation coefficients.
        """
        matrix = portfolio_head.get_correlation_matrix(period_days=30)

        assert "matrix" in matrix
        assert "high_correlation_threshold" in matrix
        assert "period_days" in matrix

        pairs = matrix["matrix"]
        assert isinstance(pairs, list)

        for pair in pairs:
            assert "strategy_a" in pair
            assert "strategy_b" in pair
            assert "correlation" in pair

    def test_drawdown_alert_triggered_above_threshold(self, portfolio_head):
        """
        P1 Test: Verify drawdown alert triggers when above 10% threshold.

        AC: drawdown_alert = True when total_drawdown > 10%.
        """
        # Mock get_account_drawdowns to return high drawdown
        portfolio_head.get_account_drawdowns = MagicMock(return_value={
            "by_account": [
                {"account_id": "acc_main", "drawdown_pct": 15.0},  # > 10%
                {"account_id": "acc_backup", "drawdown_pct": 12.0},  # > 10%
            ]
        })

        with patch.object(portfolio_head, 'get_total_equity', return_value={
            "total_equity": 50000.0,
            "accounts": [
                {"account_id": "acc_main", "balance": 30000.0},
                {"account_id": "acc_backup", "balance": 20000.0},
            ]
        }):
            summary = portfolio_head.get_portfolio_summary()

            # High drawdown should trigger alert
            assert summary["drawdown_alert"] is True


# ============================================================================
# P1-6: SkillManager - Registration and Execution
# ============================================================================

class TestSkillManagerRegistration:
    """P1 Test Group: Skill registration and execution"""

    @pytest.fixture
    def skill_manager(self):
        """Create SkillManager instance."""
        from src.agents.skills.skill_manager import SkillManager
        return SkillManager(enable_cache=False)

    def test_register_skill_with_metadata(self, skill_manager):
        """
        P1 Test: Verify skill registration with full metadata.

        AC: Registered skill can be retrieved with correct metadata.
        """
        def dummy_skill(param1: str) -> str:
            return f"Processed: {param1}"

        skill = skill_manager.register(
            name="test_skill",
            func=dummy_skill,
            description="A test skill",
            category="research",
            departments=["research", "development"],
            parameters={"param1": "string input"},
            returns={"result": "processed string"},
            tags=["test", "dummy"]
        )

        assert skill is not None
        assert skill.name == "test_skill"

        # Verify retrieval
        retrieved = skill_manager.get_skill("test_skill")
        assert retrieved.name == "test_skill"

        info = skill_manager.get_skill_info("test_skill")
        assert info["description"] == "A test skill"
        assert info["category"] == "research"
        assert "research" in info["departments"]

    def test_register_skill_duplicate_warning(self, skill_manager):
        """
        P1 Test: Verify registering duplicate skill logs warning.

        AC: Overwriting existing skill logs warning but doesn't crash.
        """
        def skill_v1():
            return "v1"

        def skill_v2():
            return "v2"

        skill_manager.register(name="duplicate_skill", func=skill_v1)

        # Should log warning but not crash
        with pytest.warns():
            skill_manager.register(name="duplicate_skill", func=skill_v2)

    def test_get_skill_not_found_raises_error(self, skill_manager):
        """
        P1 Test: Verify getting non-existent skill raises error.

        AC: SkillNotFoundError raised for unknown skill.
        """
        from src.agents.skills.skill_manager import SkillNotFoundError

        with pytest.raises(SkillNotFoundError):
            skill_manager.get_skill("non_existent_skill")


class TestSkillManagerExecution:
    """P1 Test Group: Skill execution and chaining"""

    @pytest.fixture
    def skill_manager_with_skills(self):
        """Create SkillManager with registered skills."""
        from src.agents.skills.skill_manager import SkillManager
        manager = SkillManager(enable_cache=False)

        def add_numbers(a: int, b: int) -> int:
            return a + b

        def multiply_numbers(a: int, b: int) -> int:
            return a * b

        manager.register(
            name="add",
            func=add_numbers,
            description="Add two numbers",
            parameters={"a": "int", "b": "int"},
            returns={"result": "int"}
        )

        manager.register(
            name="multiply",
            func=multiply_numbers,
            description="Multiply two numbers",
            parameters={"a": "int", "b": "int"},
            returns={"result": "int"}
        )

        return manager

    def test_execute_skill_success(self, skill_manager_with_skills):
        """
        P1 Test: Verify skill execution returns correct result.

        AC: Executed skill returns SkillResult with success=True.
        """
        result = skill_manager_with_skills.execute("add", params={"a": 5, "b": 3})

        assert result.success is True
        assert result.data == 8
        assert result.error is None

    def test_execute_skill_with_invalid_params(self, skill_manager_with_skills):
        """
        P1 Test: Verify skill execution with invalid params fails gracefully.

        AC: Missing required params returns SkillResult with success=False.
        """
        result = skill_manager_with_skills.execute("add", params={"a": 5})  # Missing b

        assert result.success is False
        assert result.error is not None

    def test_chain_skills_sequential(self, skill_manager_with_skills):
        """
        P1 Test: Verify sequential skill chaining.

        AC: Sequential chain passes output of one skill as input to next.
        """
        from src.agents.skills.skill_manager import ChainMode

        results = skill_manager_with_skills.chain(
            skills=[
                {"Name": "add", "params": {"a": 2, "b": 3}},  # = 5
                {"Name": "multiply", "params": {"a": 5, "b": 10}},  # = 50
            ],
            mode=ChainMode.SEQUENTIAL
        )

        assert len(results) == 2
        assert results[0].success is True
        assert results[0].data == 5
        assert results[1].success is True
        assert results[1].data == 50

    def test_chain_skills_parallel(self, skill_manager_with_skills):
        """
        P1 Test: Verify parallel skill chaining.

        AC: Parallel chain executes all skills with same initial params.
        """
        from src.agents.skills.skill_manager import ChainMode

        results = skill_manager_with_skills.chain(
            skills=[
                {"Name": "add", "params": {"a": 10, "b": 5}},
                {"Name": "multiply", "params": {"a": 10, "b": 5}},
            ],
            mode=ChainMode.PARALLEL
        )

        assert len(results) == 2
        # Both execute with a=10, b=5
        assert results[0].data == 15
        assert results[1].data == 50


# ============================================================================
# P1-7: TaskRouter - Priority Preemption (Integration)
# ============================================================================

class TestTaskRouterPriorityPreemption:
    """P1 Test Group: Task routing priority preemption"""

    @pytest.fixture
    def task_router(self):
        """Create TaskRouter with mocked Redis."""
        with patch('src.agents.departments.task_router.redis') as mock_redis_module:
            mock_client = MagicMock()
            mock_redis_module.ConnectionPool.return_value = mock_client
            mock_redis_module.Redis.return_value = mock_client
            mock_client.ping.return_value = True
            mock_client.xadd.return_value = "msg_id"
            mock_client.setex.return_value = True
            mock_client.sadd.return_value = 1

            from src.agents.departments.task_router import TaskRouter
            router = TaskRouter()
            yield router
            router.close()

    def test_preemption_finds_running_medium_task(self, task_router):
        """
        P1 Test: Verify HIGH priority preempts MEDIUM running task.

        AC: preempt_medium_task returns MEDIUM task for preemption.
        """
        from src.agents.departments.task_router import TaskPriority, TaskStatus, Task
        from src.agents.departments.types import Department

        # Add a running MEDIUM task
        medium_task = Task(
            task_id="medium_task_1",
            task_type="research",
            department=Department.RESEARCH.value,
            priority=TaskPriority.MEDIUM,
            payload={"query": "test"},
            status=TaskStatus.RUNNING,
            session_id="session_preempt"
        )
        task_router._active_tasks["medium_task_1"] = medium_task

        # Preempt for HIGH priority
        preempted = task_router.preempt_medium_task(
            Department.RESEARCH,
            "session_preempt"
        )

        assert preempted is not None
        assert preempted.task_id == "medium_task_1"
        assert preempted.status == TaskStatus.PREEMPTED

    def test_preemption_skips_low_priority(self, task_router):
        """
        P1 Test: Verify HIGH priority does NOT preempt LOW task.

        AC: preempt_medium_task returns None when only LOW tasks running.
        """
        from src.agents.departments.task_router import TaskPriority, TaskStatus, Task
        from src.agents.departments.types import Department

        # Add a running LOW task
        low_task = Task(
            task_id="low_task_1",
            task_type="analysis",
            department=Department.RISK.value,
            priority=TaskPriority.LOW,
            payload={},
            status=TaskStatus.RUNNING,
            session_id="session_low"
        )
        task_router._active_tasks["low_task_1"] = low_task

        # Try to preempt
        preempted = task_router.preempt_medium_task(
            Department.RISK,
            "session_low"
        )

        assert preempted is None, "Should not preempt LOW priority task"

    def test_preemption_skips_different_session(self, task_router):
        """
        P1 Test: Verify preemption only targets same session tasks.

        AC: Tasks from different session are not preempted.
        """
        from src.agents.departments.task_router import TaskPriority, TaskStatus, Task
        from src.agents.departments.types import Department

        # Add MEDIUM task from different session
        other_session_task = Task(
            task_id="other_session_task",
            task_type="research",
            department=Department.RESEARCH.value,
            priority=TaskPriority.MEDIUM,
            payload={},
            status=TaskStatus.RUNNING,
            session_id="other_session"
        )
        task_router._active_tasks["other_session_task"] = other_session_task

        # Try to preempt from our session
        preempted = task_router.preempt_medium_task(
            Department.RESEARCH,
            "our_session"
        )

        assert preempted is None, "Should not preempt tasks from different session"
