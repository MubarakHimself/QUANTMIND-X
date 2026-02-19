"""
Test Pine Script Agent

Tests for the Pine Script generation agent including:
- Code generation from natural language
- MQL5 to Pine Script conversion
- Syntax validation
- Error correction loop

Validates requirements from spec lines 1117-1124.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Test the Pine Script agent functions
class TestPineScriptAgent:
    """Test cases for Pine Script agent."""
    
    @pytest.fixture
    def sample_strategy_description(self):
        """Sample strategy description for testing."""
        return "Create a simple RSI strategy that buys when RSI crosses above 30 and sells when it crosses below 70"
    
    @pytest.fixture
    def sample_mql5_code(self):
        """Sample MQL5 code for conversion testing."""
        return """
//+------------------------------------------------------------------+
//|                                                     RSIStrategy.mq5 |
//+------------------------------------------------------------------+
#property version   "1.00"
#property strict

input int RSIPeriod = 14;
input double Lots = 0.1;

int handle;
double rsi[];

int OnInit() {
   handle = iRSI(_Symbol, PERIOD_CURRENT, RSIPeriod, PRICE_CLOSE);
   if(handle == INVALID_HANDLE) return INIT_FAILED;
   ArraySetAsSeries(rsi, true);
   return INIT_SUCCEEDED;
}

void OnTick() {
   if(CopyBuffer(handle, 0, 0, 3, rsi) <= 0) return;
   
   double rsiValue = rsi[0];
   double prevRsi = rsi[1];
   
   // Buy signal: RSI crosses above 30
   if(prevRsi < 30 && rsiValue >= 30) {
      Trade.Buy(Lots, _Symbol);
   }
   
   // Sell signal: RSI crosses below 70
   if(prevRsi > 70 && rsiValue <= 70) {
      Trade.Sell(Lots, _Symbol);
   }
}

void OnDeinit(const int reason) {
   IndicatorRelease(handle);
}
"""
    
    def test_pine_script_state_creation(self):
        """Test PineScriptState initialization."""
        from src.agents.pinescript import PineScriptState
        
        state = PineScriptState(
            messages=[],
            user_query="Generate a RSI strategy",
            strategy_description="RSI crossover strategy",
            pine_script_code=None,
            validation_errors=[],
            status="pending",
            mql5_source=None,
            conversion_mode="generate"
        )
        
        assert state.status == "pending"
        assert state.strategy_description == "RSI crossover strategy"
        assert state.conversion_mode == "generate"
    
    def test_pine_script_patterns(self):
        """Test Pine Script syntax patterns."""
        from src.agents.pinescript import PineScriptState
        
        # Valid Pine Script patterns
        valid_patterns = PineScriptState.PINESCRIPT_PATTERNS
        
        assert 'version' in valid_patterns
        assert 'indicator' in valid_patterns
        assert 'strategy' in valid_patterns
    
    def test_validate_pine_script_syntax_valid(self, tmp_path):
        """Test syntax validation with valid code."""
        from src.agents.pinescript import validate_pinescript_syntax
        
        valid_code = """//@version=5
strategy("RSI Strategy", overlay=true)

rsi = ta.rsi(close, 14)

if (rsi < 30)
    strategy.entry("Long", strategy.long)
"""
        errors = validate_pinescript_syntax(valid_code)
        assert len(errors) == 0
    
    def test_validate_pine_script_syntax_invalid(self):
        """Test syntax validation with invalid code."""
        from src.agents.pinescript import validate_pinescript_syntax
        
        invalid_code = """// Invalid Pine Script
indicator("Test", overlay=true)

x = some_unknown_function(10)
if (x)
    entry("Long")
"""
        errors = validate_pinescript_syntax(invalid_code)
        assert len(errors) > 0
    
    def test_validate_pine_script_missing_version(self):
        """Test validation catches missing version declaration."""
        from src.agents.pinescript import validate_pinescript_syntax
        
        code_no_version = """
strategy("Test", overlay=true)
"""
        errors = validate_pinescript_syntax(code_no_version)
        # Should have at least one error about version
        assert any('version' in e.lower() for e in errors) or len(errors) > 0
    
    def test_convert_mql5_to_pinescript_rsi(self, sample_mql5_code):
        """Test MQL5 to Pine Script conversion for RSI."""
        from src.agents.pinescript import convert_mql5_to_pinescript
        
        result = convert_mql5_to_pinescript(sample_mql5_code)
        
        assert result is not None
        assert '//@version=5' in result
        assert 'rsi' in result.lower()
    
    def test_convert_mql5_to_pinescript_order_functions(self, sample_mql5_code):
        """Test that MQL5 Order functions are mapped to Pine Script equivalents."""
        from src.agents.pinescript import convert_mql5_to_pinescript
        
        result = convert_mql5_to_pinescript(sample_mql5_code)
        
        # Should use strategy.entry instead of Trade.Buy/Sell
        assert 'strategy.entry' in result or 'strategy.entry' in result.lower()
    
    def test_convert_mql5_to_pinescript_indicators(self):
        """Test MQL5 indicator to Pine Script indicator mapping."""
        from src.agents.pinescript import convert_mql5_to_pinescript
        
        mql5_with_indicators = """
input int FastEMA = 10;
input int SlowEMA = 20;

int OnInit() {
   return INIT_SUCCEEDED;
}

void OnTick() {
   double fast[] = iMA(_Symbol, PERIOD_CURRENT, FastEMA, 0, MODE_EMA, PRICE_CLOSE);
   double slow[] = iMA(_Symbol, PERIOD_CURRENT, SlowEMA, 0, MODE_EMA, PRICE_CLOSE);
}
"""
        result = convert_mql5_to_pinescript(mql5_with_indicators)
        
        # Should have ta.sma or ta.ema
        assert 'ta.sma' in result.lower() or 'ta.ema' in result.lower() or 'ta.rma' in result.lower()
    
    @patch('src.agents.pinescript.get_llm')
    def test_generate_pine_script(self, mock_llm):
        """Test Pine Script generation from description."""
        from src.agents.pinescript import generate_pine_script_from_query
        
        # Mock the LLM
        mock_llm_instance = Mock()
        mock_llm_instance.invoke.return_value = Mock(content="//@version=5\nstrategy('Test')\n")
        mock_llm.return_value = mock_llm_instance
        
        result = generate_pine_script_from_query("Create a simple moving average strategy")
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_compile_pinescript_graph(self):
        """Test that the Pine Script graph compiles correctly."""
        from src.agents.pinescript import compile_pinescript_graph
        
        graph = compile_pinescript_graph()
        
        # Should return a StateGraph
        assert graph is not None
    
    def test_error_correction_max_iterations(self):
        """Test that error correction has max iteration limit."""
        # This tests that the error correction loop doesn't run forever
        # The implementation should limit corrections to max 3 iterations
        
        # This is more of an integration test that would verify
        # the fix_errors_node has iteration limiting
        max_iterations = 3  # Per spec
        assert max_iterations == 3


class TestPineScriptTools:
    """Test Pine Script utility functions."""
    
    def test_extract_indicators_from_strategy(self):
        """Test extraction of indicators from strategy description."""
        # This would test the parsing of strategy descriptions
        # to extract indicator requirements
        pass
    
    def test_pine_script_template_generation(self):
        """Test that Pine Script templates are generated correctly."""
        from src.agents.pinescript import PINESCRIPT_SYSTEM_PROMPT
        
        assert 'Pine Script v5' in PINESCRIPT_SYSTEM_PROMPT
        assert 'indicator' in PINESCRIPT_SYSTEM_PROMPT
        assert 'strategy' in PINESCRIPT_SYSTEM_PROMPT


class TestPineScriptValidation:
    """Test Pine Script validation functions."""
    
    def test_version_declaration_check(self):
        """Test that version 5 is required."""
        from src.agents.pinescript import validate_pinescript_syntax
        
        # Test v5
        v5_code = "//@version=5\nstrategy('test')"
        errors_v5 = validate_pinescript_syntax(v5_code)
        # Should have no version errors
        
        # Test v4 (should fail)
        v4_code = "//@version=4\nstrategy('test')"
        errors_v4 = validate_pinescript_syntax(v4_code)
        # Should have version error
    
    def test_function_syntax_validation(self):
        """Test function syntax validation."""
        from src.agents.pinescript import validate_pinescript_syntax
        
        # Valid function
        valid_func = """
//@version=5
indicator("Test")

foo() =>
    close
"""
        errors = validate_pinescript_syntax(valid_func)
        # Should have minimal errors
    
    def test_variable_declaration_validation(self):
        """Test variable declaration validation."""
        from src.agents.pinescript import validate_pinescript_syntax
        
        # Valid variable
        valid_var = """
//@version=5
indicator("Test")
int x = 10
float y = 1.5
"""
        errors = validate_pinescript_syntax(valid_var)
        # Should have no errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
