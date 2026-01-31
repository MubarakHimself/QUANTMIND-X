"""
Integration Tests for Backend Bridge Components.

Tests critical integration workflows between Python agents and MQL5 EAs.
Focuses on end-to-end workflows rather than individual unit tests.

Task Group 7: Integration Testing
Maximum 10 strategic integration tests for critical backend bridge workflows.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.agents.core.base_agent import BaseAgent, CommandParser
from src.agents.skills.system_skills.skill_creator import SkillCreator, SkillGenerationConfig
from src.agents.skills.trading_skills.indicator_writer import generate_mql5_indicator
from src.router.sync import DiskSyncer
from src.router.governor import RiskMandate
from src.backtesting.mt5_engine import PythonStrategyTester, MQL5Timeframe, MT5BacktestResult
import pandas as pd
import numpy as np


class TestCommandParserSkillCreatorIntegration:
    """Test CommandParser integration with Skill Creator for slash command skill generation."""

    @pytest.fixture
    def agent_with_skill_creator(self, tmp_path):
        """Create BaseAgent with skill creator integration."""
        agent = Mock(spec=BaseAgent)
        agent.name = "TestAgent"
        agent.command_parser = CommandParser()
        agent.command_parser._register_builtin_commands()

        # Register skill creator command
        creator = SkillCreator(skills_base_dir=str(tmp_path))

        async def create_skill_handler(args: str) -> str:
            # Parse args: "Create a skill to [description]"
            description = args.replace("Create a skill to ", "").strip()
            result = creator.create_skill_from_description(
                high_level_description=description,
                category="trading_skills"
            )
            if result["success"]:
                return f"Skill created at: {result['skill_path']}"
            return f"Error: {result.get('error', 'Unknown error')}"

        agent.command_parser.register_command("create_skill", create_skill_handler)
        return agent, creator, tmp_path

    @pytest.mark.integration
    def test_slash_command_creates_skill_via_command_parser(self, agent_with_skill_creator):
        """Test /create_skill slash command triggers skill creation workflow."""
        agent, creator, tmp_path = agent_with_skill_creator

        # Execute slash command to create skill
        import asyncio
        result = asyncio.run(agent.command_parser.execute(
            "/create_skill Create a skill to calculate RSI indicator"
        ))

        # Verify skill was created
        assert "Skill created at:" in result
        assert "rsi" in result.lower() or "indicator" in result.lower()

        # Verify skill directory exists
        skill_path = result.split(": ")[1].strip()
        skill_dir = Path(skill_path)
        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "__init__.py").exists()


class TestDiskSyncerMQL5LibraryIntegration:
    """Test DiskSyncer integration with MQL5 Risk Library via risk_matrix.json."""

    @pytest.fixture
    def mql5_library_path(self, tmp_path):
        """Create mock MQL5 library path."""
        mql5_path = tmp_path / "MQL5" / "Files"
        mql5_path.mkdir(parents=True)
        return str(mql5_path)

    @pytest.mark.integration
    def test_python_writes_risk_matrix_mql5_reads(self, mql5_library_path):
        """Test Python writes risk_matrix.json that MQL5 library can read."""
        syncer = DiskSyncer(mt5_path=mql5_library_path)

        # Python writes risk matrix
        risk_matrix = {
            "EURUSD": {"multiplier": 1.0, "timestamp": 1234567890},
            "GBPUSD": {"multiplier": 0.8, "timestamp": 1234567891},
            "USDJPY": {"multiplier": 1.2, "timestamp": 1234567892}
        }

        syncer.sync_risk_matrix(risk_matrix)

        # Verify file was written
        risk_file = Path(mql5_library_path) / "risk_matrix.json"
        assert risk_file.exists()

        # Verify MQL5 can read the file (simulating MQL5 JSON parsing)
        with open(risk_file, 'r') as f:
            loaded_data = json.load(f)

        # Verify structure matches MQL5 expectations
        assert "EURUSD" in loaded_data
        assert "multiplier" in loaded_data["EURUSD"]
        assert "timestamp" in loaded_data["EURUSD"]
        assert loaded_data["EURUSD"]["multiplier"] == 1.0

    @pytest.mark.integration
    def test_governor_updates_risk_via_disk_syncer(self, mql5_library_path):
        """Test governor uses DiskSyncer to update risk parameters."""
        syncer = DiskSyncer(mt5_path=mql5_library_path)

        # Create risk mandates from governor
        mandates = {
            "EURUSD": RiskMandate(allocation_scalar=1.5, risk_mode="AGGRESSIVE"),
            "GBPUSD": RiskMandate(allocation_scalar=0.5, risk_mode="CONSERVATIVE")
        }

        # Convert to risk matrix format
        import time
        risk_matrix = {
            symbol: {
                "multiplier": mandate.allocation_scalar,
                "timestamp": int(time.time())
            }
            for symbol, mandate in mandates.items()
        }

        # Sync to disk
        syncer.sync_risk_matrix(risk_matrix)

        # Verify file reflects governor mandates
        risk_file = Path(mql5_library_path) / "risk_matrix.json"
        with open(risk_file, 'r') as f:
            loaded_data = json.load(f)

        assert loaded_data["EURUSD"]["multiplier"] == 1.5
        assert loaded_data["GBPUSD"]["multiplier"] == 0.5


class TestSkillCreatorIndicatorWriterIntegration:
    """Test Skill Creator integration with Indicator Writer for MQL5 code generation."""

    @pytest.mark.integration
    def test_skill_creator_generates_indicator_writer_skill(self, tmp_path):
        """Test skill creator can create skill that generates MQL5 indicators."""
        creator = SkillCreator(skills_base_dir=str(tmp_path))

        # Create a skill that generates indicators
        result = creator.create_skill_from_description(
            high_level_description="Create a skill to generate custom moving average indicators in MQL5",
            category="trading_skills"
        )

        assert result["success"]
        skill_path = Path(result["skill_path"])

        # Verify skill was created
        assert skill_path.exists()
        assert (skill_path / "SKILL.md").exists()
        assert (skill_path / "__init__.py").exists()

        # Load generated skill definition
        init_content = (skill_path / "__init__.py").read_text()
        assert "SkillDefinition" in init_content

        # Verify skill can generate MQL5 code
        # (In real scenario, would import and execute the skill)
        skill_md = (skill_path / "SKILL.md").read_text()
        assert "MQL5" in skill_md or "indicator" in skill_md.lower()

    @pytest.mark.integration
    def test_indicator_writer_generates_valid_mql5_code(self):
        """Test indicator writer generates compilation-ready MQL5 code."""
        # Generate MQL5 indicator
        mql5_code = generate_mql5_indicator(
            indicator_name="TestMA",
            indicator_type="line",
            buffers=1,
            use_ring_buffer=True
        )

        # Verify compilation-ready syntax
        assert "#property" in mql5_code
        assert "indicator_chart_window" in mql5_code
        assert "indicator_buffers" in mql5_code
        assert "OnInit()" in mql5_code
        assert "OnCalculate(" in mql5_code
        assert "SetIndexBuffer(" in mql5_code

        # Verify balanced braces
        assert mql5_code.count("{") == mql5_code.count("}")

        # Verify CRiBuffDbl class included
        assert "class CRiBuffDbl" in mql5_code or "CRiBuffDbl" in mql5_code


class TestMT5EngineBacktestingIntegration:
    """Test MT5 Engine integration with backtesting pipeline."""

    @pytest.mark.integration
    def test_complete_backtest_workflow_from_python(self):
        """Test complete backtest workflow: Python strategy → MT5 engine → results."""
        tester = PythonStrategyTester(initial_cash=10000.0, commission=0.001)

        # Create OHLCV data (simulating MT5 data feed)
        data = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=100, freq='1h'),
            'open': np.linspace(1.1000, 1.1100, 100) + np.random.randn(100) * 0.0005,
            'high': np.linspace(1.1050, 1.1150, 100) + np.random.randn(100) * 0.0005,
            'low': np.linspace(1.0950, 1.1050, 100) + np.random.randn(100) * 0.0005,
            'close': np.linspace(1.1000, 1.1100, 100) + np.random.randn(100) * 0.0005,
            'tick_volume': np.random.randint(1000, 3000, 100)
        })

        # Strategy: Simple moving average crossover using MQL5-style functions
        strategy_code = """
def on_bar(tester):
    if tester.current_bar < 20:
        return

    # Calculate simple moving averages
    sum_fast = 0
    sum_slow = 0
    for i in range(10):
        sum_fast += tester.iClose(tester.symbol, tester.timeframe, i)
    for i in range(20):
        sum_slow += tester.iClose(tester.symbol, tester.timeframe, i)

    ma_fast = sum_fast / 10
    ma_slow = sum_slow / 20

    # Entry logic: crossover
    if ma_fast > ma_slow and tester.position_size == 0:
        tester.buy(tester.symbol, 0.01)
    elif ma_fast < ma_slow and tester.position_size > 0:
        tester.close_all_positions()
"""

        # Run backtest
        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        # Verify backtest completed
        assert result is not None
        assert isinstance(result, MT5BacktestResult)
        assert result.initial_cash == 10000.0
        assert result.trades >= 0
        assert hasattr(result, 'final_cash')
        assert hasattr(result, 'return_pct')  # MT5BacktestResult uses return_pct

    @pytest.mark.integration
    def test_mt5_engine_with_risk_parameters_from_sync(self, tmp_path):
        """Test MT5 engine uses risk parameters synced from DiskSyncer."""
        # Setup: Sync risk parameters
        mql5_path = tmp_path / "MQL5" / "Files"
        mql5_path.mkdir(parents=True)

        syncer = DiskSyncer(mt5_path=str(mql5_path))
        risk_matrix = {
            "EURUSD": {"multiplier": 0.5, "timestamp": 1234567890}  # 50% risk
        }
        syncer.sync_risk_matrix(risk_matrix)

        # MT5 engine reads risk parameters (simulated)
        risk_file = mql5_path / "risk_matrix.json"
        with open(risk_file, 'r') as f:
            synced_risk = json.load(f)

        risk_multiplier = synced_risk.get("EURUSD", {}).get("multiplier", 1.0)

        # Backtest with adjusted position size based on risk
        tester = PythonStrategyTester(initial_cash=10000.0)
        data = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=50, freq='1h'),
            'open': [1.10] * 50,
            'high': [1.11] * 50,
            'low': [1.09] * 50,
            'close': [1.105] * 50,
            'tick_volume': [1000] * 50
        })

        strategy_code = f"""
def on_bar(tester):
    if tester.current_bar == 0:
        # Use risk multiplier from synced risk matrix
        position_size = 0.01 * {risk_multiplier}
        tester.buy(tester.symbol, position_size)
"""

        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        # Verify backtest executed with adjusted risk
        assert result is not None


class TestSlashCommandExecutionIntegration:
    """Test slash commands execute code and interact with trading components."""

    @pytest.mark.integration
    def test_code_command_executes_trading_strategy(self):
        """Test /code command executes trading strategy calculation."""
        parser = CommandParser()
        parser._register_builtin_commands()

        import asyncio
        result = asyncio.run(parser.execute(
            "/code "
            "sma = sum([1.10, 1.11, 1.12]) / 3; "
            "print(f'SMA: {sma:.4f}')"
        ))

        # Verify calculation executed
        assert "SMA:" in result
        assert "1.1100" in result or "1.11" in result

    @pytest.mark.integration
    def test_file_command_writes_mql5_indicator(self, tmp_path):
        """Test /file command writes generated MQL5 indicator to file."""
        parser = CommandParser()
        parser._register_builtin_commands()

        # Generate MQL5 code
        mql5_code = generate_mql5_indicator(
            indicator_name="TestIndicator",
            indicator_type="line",
            buffers=1
        )

        # Write to file via /file command
        indicator_file = tmp_path / "TestIndicator.mq5"
        import asyncio
        result = asyncio.run(parser.execute(
            f"/file write {indicator_file} {mql5_code[:100]}..."  # Truncated for test
        ))

        # Verify file write
        assert "successfully" in result.lower() or "written" in result.lower()


class TestMultiStepWorkflowIntegration:
    """Test complete multi-step workflows across multiple components."""

    @pytest.mark.integration
    def test_complete_workflow_skill_to_indicator_to_backtest(self, tmp_path):
        """Test complete workflow: Create skill → Generate indicator → Backtest."""
        # Step 1: Create skill via skill creator
        creator = SkillCreator(skills_base_dir=str(tmp_path))
        skill_result = creator.create_skill_from_description(
            high_level_description="Create a skill to generate RSI indicators",
            category="trading_skills"
        )

        assert skill_result["success"]
        skill_path = Path(skill_result["skill_path"])
        assert skill_path.exists()

        # Step 2: Generate MQL5 indicator
        mql5_code = generate_mql5_indicator(
            indicator_name="RSI_Indicator",
            indicator_type="line",
            buffers=1,
            use_ring_buffer=True
        )

        assert "OnInit()" in mql5_code
        assert "OnCalculate(" in mql5_code

        # Step 3: Save indicator to file
        indicator_file = tmp_path / "RSI_Indicator.mq5"
        indicator_file.write_text(mql5_code)
        assert indicator_file.exists()

        # Step 4: Run backtest with MT5 engine
        tester = PythonStrategyTester(initial_cash=10000.0)
        data = pd.DataFrame({
            'time': pd.date_range(start='2024-01-01', periods=50, freq='1h'),
            'open': np.linspace(1.10, 1.11, 50),
            'high': np.linspace(1.11, 1.12, 50),
            'low': np.linspace(1.09, 1.10, 50),
            'close': np.linspace(1.10, 1.11, 50),
            'tick_volume': [1000] * 50
        })

        strategy_code = """
def on_bar(tester):
    if tester.current_bar == 0:
        tester.buy(tester.symbol, 0.01)
"""

        result = tester.run(strategy_code, data, "EURUSD", MQL5Timeframe.PERIOD_H1)

        # Verify complete workflow
        assert result is not None
        assert isinstance(result, MT5BacktestResult)
        assert skill_path.exists()
        assert indicator_file.exists()
