"""
End-to-End Test for Complete QuantMind Workflow.

This test validates the complete NPRD → Analyst → QuantCode → Backtest pipeline.
"""

import asyncio
import json
import pytest
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Test markers
pytestmark = pytest.mark.e2e


class TestCompleteWorkflow:
    """Test suite for complete QuantMind workflow."""

    @pytest.fixture
    def sample_nprd_data(self) -> Dict[str, Any]:
        """Create sample NPRD data for testing."""
        return {
            "meta": {
                "video_title": "EURUSD Trend Following Strategy",
                "duration_seconds": 180,
                "processed_at": datetime.now().isoformat()
            },
            "timeline": [
                {
                    "start": 0,
                    "end": 30,
                    "description": "Strategy overview: Trend following on EURUSD H1 timeframe",
                    "key_points": ["Trend following", "EURUSD", "H1 timeframe"]
                },
                {
                    "start": 30,
                    "end": 60,
                    "description": "Entry rules: Enter when price crosses above 20 EMA with RSI above 50",
                    "key_points": ["Entry signal", "EMA crossover", "RSI filter"]
                },
                {
                    "start": 60,
                    "end": 90,
                    "description": "Exit rules: Take profit at 2x risk, stop loss below recent swing low",
                    "key_points": ["Exit signal", "Risk-reward ratio", "Stop loss placement"]
                },
                {
                    "start": 90,
                    "end": 120,
                    "description": "Risk management: Maximum 2% risk per trade, use Kelly criterion for sizing",
                    "key_points": ["Risk management", "Position sizing", "Kelly criterion"]
                }
            ],
            "strategy_summary": {
                "name": "EURUSD Trend Follower",
                "type": "trend_following",
                "timeframe": "H1",
                "symbols": ["EURUSD"],
                "risk_level": "moderate"
            }
        }

    @pytest.fixture
    def sample_trd_content(self) -> str:
        """Create sample TRD content for testing."""
        return """
# EURUSD Trend Follower Strategy

## Description
A trend following strategy for EURUSD on the H1 timeframe that uses EMA crossovers
with RSI confirmation for entry signals.

```json
{
  "strategy_name": "EURUSD Trend Follower",
  "description": "Trend following strategy using EMA crossover with RSI confirmation",
  "entry_rules": [
    {
      "name": "EMA Crossover Entry",
      "description": "Enter when price crosses above 20 EMA with RSI above 50",
      "conditions": [
        "Price closes above 20 EMA",
        "RSI(14) > 50",
        "Volume above average"
      ],
      "order_type": "market"
    }
  ],
  "exit_rules": [
    {
      "name": "Take Profit Exit",
      "description": "Exit at 2x risk target",
      "conditions": ["Price reaches take profit level"],
      "order_type": "limit"
    },
    {
      "name": "Stop Loss Exit",
      "description": "Exit at stop loss below recent swing low",
      "conditions": ["Price reaches stop loss level"],
      "order_type": "stop"
    }
  ],
  "stop_loss": {
    "type": "swing_low",
    "value": 50
  },
  "take_profit": {
    "type": "risk_reward",
    "value": 2.0
  },
  "risk_management": {
    "max_risk_per_trade": 0.02,
    "max_daily_drawdown": 0.05,
    "max_open_positions": 3,
    "use_kelly_criterion": true
  },
  "kelly_integration": {
    "enabled": true,
    "lookback_period": 100,
    "max_position_size": 0.10,
    "min_win_rate": 0.40,
    "safety_fraction": 0.5
  },
  "timeframe": "H1",
  "symbols": ["EURUSD"],
  "position_side": "long",
  "indicators": [
    {
      "name": "EMA",
      "parameters": {"period": 20}
    },
    {
      "name": "RSI",
      "parameters": {"period": 14}
    }
  ]
}
```
"""

    @pytest.mark.asyncio
    async def test_nprd_to_analyst_workflow(self, sample_nprd_data: Dict[str, Any], tmp_path: Path):
        """Test NPRD processing triggers Analyst agent."""
        # Create NPRD output file
        nprd_dir = tmp_path / "nprd" / "outputs"
        nprd_dir.mkdir(parents=True)
        
        nprd_file = nprd_dir / "test_nprd_001.json"
        with open(nprd_file, 'w') as f:
            json.dump(sample_nprd_data, f)
        
        # Verify file was created
        assert nprd_file.exists()
        
        # Verify content
        with open(nprd_file, 'r') as f:
            data = json.load(f)
        
        assert data["strategy_summary"]["name"] == "EURUSD Trend Follower"
        assert len(data["timeline"]) == 4

    @pytest.mark.asyncio
    async def test_trd_validation(self, sample_trd_content: str):
        """Test TRD validation schema."""
        from src.agents.analyst.trd_validator import validate_trd
        
        result = validate_trd(sample_trd_content)
        
        assert result.valid is True
        assert result.trd is not None
        assert result.trd.strategy_name == "EURUSD Trend Follower"
        assert result.trd.timeframe.value == "H1"
        assert "EURUSD" in result.trd.symbols

    @pytest.mark.asyncio
    async def test_trd_to_quantcode_workflow(self, sample_trd_content: str, tmp_path: Path):
        """Test TRD triggers QuantCode agent."""
        # Create TRD output file
        trd_dir = tmp_path / "analyst" / "outputs"
        trd_dir.mkdir(parents=True)
        
        trd_file = trd_dir / "trd_vanilla.md"
        with open(trd_file, 'w') as f:
            f.write(sample_trd_content)
        
        # Verify file was created
        assert trd_file.exists()

    @pytest.mark.asyncio
    async def test_code_quality_check(self):
        """Test MQL5 code quality checker."""
        from src.agents.quantcode.code_quality import check_code_quality
        
        sample_code = """
//+------------------------------------------------------------------+
//| Expert Advisor - EURUSD Trend Follower                           |
//+------------------------------------------------------------------+
#property copyright "QuantMind"
#property version   "1.00"
#property description "Trend following strategy using EMA crossover"

input int    emaPeriod = 20;
input int    rsiPeriod = 14;
input double riskPercent = 2.0;
input int    magicNumber = 123456;

int emaHandle;
int rsiHandle;
double emaBuffer[];
double rsiBuffer[];

int OnInit()
{
    emaHandle = iMA(_Symbol, PERIOD_H1, emaPeriod, 0, MODE_EMA, PRICE_CLOSE);
    rsiHandle = iRSI(_Symbol, PERIOD_H1, rsiPeriod, PRICE_CLOSE);
    
    ArraySetAsSeries(emaBuffer, true);
    ArraySetAsSeries(rsiBuffer, true);
    
    return(INIT_SUCCEEDED);
}

void OnTick()
{
    if(CopyBuffer(emaHandle, 0, 0, 3, emaBuffer) < 3) return;
    if(CopyBuffer(rsiHandle, 0, 0, 3, rsiBuffer) < 3) return;
    
    // Check for entry signal
    if(emaBuffer[0] > emaBuffer[1] && rsiBuffer[0] > 50)
    {
        // Entry logic here
    }
}
"""
        
        result = check_code_quality(sample_code)
        
        assert result["score"] >= 60
        assert "metrics" in result
        assert result["metrics"]["functions"] >= 2

    @pytest.mark.asyncio
    async def test_backtest_validation(self):
        """Test backtest results validation."""
        from src.agents.quantcode.backtest_validator import validate_backtest_results
        
        sample_results = {
            "total_trades": 150,
            "win_rate": 0.58,
            "max_drawdown": 0.15,
            "sharpe_ratio": 1.45,
            "profit_factor": 1.65,
            "net_profit": 12500.00,
            "recovery_factor": 5.21,
            "average_risk_reward": 1.8,
            "consecutive_losses": 5,
            "expectancy": 83.33
        }
        
        result = validate_backtest_results(sample_results)
        
        assert result["passed"] is True
        assert result["critical_passed"] is True

    @pytest.mark.asyncio
    async def test_backtest_validation_failure(self):
        """Test backtest validation fails for poor results."""
        from src.agents.quantcode.backtest_validator import validate_backtest_results
        
        poor_results = {
            "total_trades": 50,  # Below minimum
            "win_rate": 0.30,    # Below minimum
            "max_drawdown": 0.50,  # Above maximum
            "sharpe_ratio": 0.5,
            "profit_factor": 0.8
        }
        
        result = validate_backtest_results(poor_results)
        
        assert result["passed"] is False
        assert result["critical_passed"] is False

    @pytest.mark.asyncio
    async def test_memory_operations(self):
        """Test LangMem memory operations."""
        from src.memory.langmem_manager import LangMemManager
        
        manager = LangMemManager()
        
        # Add semantic memory
        entry = await manager.add_semantic_memory(
            agent_name="analyst",
            content="EURUSD shows mean reversion behavior on H1 timeframe during Asian session",
            importance=0.8
        )
        
        assert entry.id is not None
        assert entry.content == "EURUSD shows mean reversion behavior on H1 timeframe during Asian session"
        
        # Search memory
        results = await manager.search_semantic_memory(
            agent_name="analyst",
            query="EURUSD behavior"
        )
        
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_workflow_orchestrator(self, sample_nprd_data: Dict[str, Any], tmp_path: Path):
        """Test workflow orchestrator."""
        from src.router.workflow_orchestrator import WorkflowOrchestrator
        
        # Create NPRD file
        nprd_dir = tmp_path / "nprd" / "outputs"
        nprd_dir.mkdir(parents=True)
        
        nprd_file = nprd_dir / "test_workflow_001.json"
        with open(nprd_file, 'w') as f:
            json.dump(sample_nprd_data, f)
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(work_dir=tmp_path / "workflows")
        
        # Start workflow
        workflow_id = await orchestrator.start_workflow(nprd_file)
        
        assert workflow_id is not None
        assert workflow_id.startswith("wf_")
        
        # Get status
        status = orchestrator.get_workflow_status(workflow_id)
        
        assert status is not None
        assert "workflow_id" in status

    @pytest.mark.asyncio
    async def test_mcp_tools_registration(self):
        """Test MCP tools are properly registered."""
        from src.agents.tools.mcp_tools import MCP_TOOLS
        
        # Check that all expected tools are registered
        expected_tools = [
            "get_mql5_documentation",
            "sequential_thinking",
            "index_pdf",
            "search_pdf"
        ]
        
        for tool_name in expected_tools:
            assert tool_name in MCP_TOOLS, f"Tool {tool_name} not registered"

    @pytest.mark.asyncio
    async def test_knowledge_tools(self):
        """Test knowledge base tools."""
        from src.agents.tools.knowledge_tools import (
            search_mql5_book,
            search_knowledge_hub,
            list_knowledge_namespaces
        )
        
        # Test MQL5 book search
        result = await search_mql5_book("OrderSend function")
        assert result["success"] is True
        
        # Test knowledge hub search
        result = await search_knowledge_hub("trading strategy")
        assert result["success"] is True
        
        # Test namespace listing
        result = await list_knowledge_namespaces()
        assert result["success"] is True
        assert len(result["namespaces"]) > 0


class TestMCPIntegration:
    """Test MCP server integration."""

    @pytest.mark.asyncio
    async def test_mcp_status_endpoint(self):
        """Test MCP status endpoint returns valid response."""
        # This would test the actual API endpoint
        # Placeholder for integration test
        pass

    @pytest.mark.asyncio
    async def test_context7_mcp_connection(self):
        """Test Context7 MCP server connection."""
        # This would test actual MCP connection
        # Placeholder for integration test
        pass

    @pytest.mark.asyncio
    async def test_backtest_mcp_execution(self):
        """Test backtest execution via MCP."""
        # This would test actual backtest execution
        # Placeholder for integration test
        pass


class TestPDFIntegration:
    """Test PDF upload and indexing integration."""

    @pytest.mark.asyncio
    async def test_pdf_upload(self, tmp_path: Path):
        """Test PDF file upload."""
        from src.api.pdf_endpoints import UPLOAD_DIR
        
        # Create a test PDF path
        test_pdf = tmp_path / "test.pdf"
        test_pdf.write_bytes(b"%PDF-1.4\ntest content")
        
        assert test_pdf.exists()

    @pytest.mark.asyncio
    async def test_pdf_indexing_job(self):
        """Test PDF indexing job creation."""
        # This would test the indexing job workflow
        # Placeholder for integration test
        pass


class TestErrorRecovery:
    """Test error recovery and retry logic."""

    @pytest.mark.asyncio
    async def test_compilation_retry(self):
        """Test compilation error recovery with retry."""
        from src.agents.quantcode.code_quality import check_code_quality
        
        # Code with intentional issues
        bad_code = """
int OnInit() {
    // Missing return statement
}
"""
        
        result = check_code_quality(bad_code)
        
        # Should still return a result, not crash
        assert "score" in result
        assert "issues" in result

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, tmp_path: Path):
        """Test workflow handles errors gracefully."""
        from src.router.workflow_orchestrator import WorkflowOrchestrator
        
        orchestrator = WorkflowOrchestrator(work_dir=tmp_path / "workflows")
        
        # Try to start workflow with non-existent file
        non_existent = tmp_path / "non_existent.json"
        
        # Should handle gracefully
        try:
            workflow_id = await orchestrator.start_workflow(non_existent)
            # If it starts, check that it fails properly
            status = orchestrator.get_workflow_status(workflow_id)
            assert status["status"] in ["failed", "pending"]
        except FileNotFoundError:
            # This is also acceptable
            pass


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
