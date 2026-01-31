"""
Test Core Modules Compilation

This test verifies that all Core QSL modules exist and have proper structure.
Since we cannot compile MQL5 files directly in Python, we check for:
1. File existence
2. Proper header guards
3. Required includes
4. Basic syntax structure
"""

import pytest
from pathlib import Path
import re


class TestCoreModulesCompilation:
    """Test suite for Core QSL modules compilation verification"""
    
    @pytest.fixture
    def core_modules_path(self):
        """Get path to Core modules directory"""
        return Path("src/mql5/Include/QuantMind/Core")
    
    @pytest.fixture
    def test_ea_path(self):
        """Get path to test EA"""
        return Path("src/mql5/Experts/TestCoreModules.mq5")
    
    def test_core_directory_exists(self, core_modules_path):
        """Test that Core directory exists"""
        assert core_modules_path.exists(), "Core modules directory does not exist"
        assert core_modules_path.is_dir(), "Core path is not a directory"
    
    def test_base_agent_exists(self, core_modules_path):
        """Test that BaseAgent.mqh exists"""
        base_agent_file = core_modules_path / "BaseAgent.mqh"
        assert base_agent_file.exists(), "BaseAgent.mqh does not exist"
    
    def test_constants_exists(self, core_modules_path):
        """Test that Constants.mqh exists"""
        constants_file = core_modules_path / "Constants.mqh"
        assert constants_file.exists(), "Constants.mqh does not exist"
    
    def test_types_exists(self, core_modules_path):
        """Test that Types.mqh exists"""
        types_file = core_modules_path / "Types.mqh"
        assert types_file.exists(), "Types.mqh does not exist"
    
    def test_test_ea_exists(self, test_ea_path):
        """Test that TestCoreModules.mq5 exists"""
        assert test_ea_path.exists(), "TestCoreModules.mq5 does not exist"
    
    def test_base_agent_structure(self, core_modules_path):
        """Test BaseAgent.mqh has proper structure"""
        base_agent_file = core_modules_path / "BaseAgent.mqh"
        content = base_agent_file.read_text()
        
        # Check header guard
        assert "#ifndef __QSL_BASE_AGENT_MQH__" in content, "Missing header guard ifndef"
        assert "#define __QSL_BASE_AGENT_MQH__" in content, "Missing header guard define"
        assert "#endif" in content, "Missing header guard endif"
        
        # Check class definition
        assert "class CBaseAgent" in content, "Missing CBaseAgent class definition"
        
        # Check key methods
        assert "Initialize(" in content, "Missing Initialize method"
        assert "Deinitialize(" in content, "Missing Deinitialize method"
        assert "IsInitialized(" in content, "Missing IsInitialized method"
        assert "ValidateSymbol(" in content, "Missing ValidateSymbol method"
        assert "ValidateTimeframe(" in content, "Missing ValidateTimeframe method"
        
        # Check logging methods
        assert "LogInfo(" in content, "Missing LogInfo method"
        assert "LogWarning(" in content, "Missing LogWarning method"
        assert "LogError(" in content, "Missing LogError method"
        
        # Check utility methods
        assert "GetAccountBalance(" in content, "Missing GetAccountBalance method"
        assert "GetAccountEquity(" in content, "Missing GetAccountEquity method"
        assert "IsTradingAllowed(" in content, "Missing IsTradingAllowed method"
        assert "NormalizeLot(" in content, "Missing NormalizeLot method"
    
    def test_constants_structure(self, core_modules_path):
        """Test Constants.mqh has proper structure"""
        constants_file = core_modules_path / "Constants.mqh"
        content = constants_file.read_text()
        
        # Check header guard
        assert "#ifndef __QSL_CONSTANTS_MQH__" in content, "Missing header guard ifndef"
        assert "#define __QSL_CONSTANTS_MQH__" in content, "Missing header guard define"
        assert "#endif" in content, "Missing header guard endif"
        
        # Check key constants
        assert "QM_VERSION_STRING" in content, "Missing version string constant"
        assert "QM_DAILY_LOSS_LIMIT_PCT" in content, "Missing daily loss limit constant"
        assert "QM_KELLY_THRESHOLD" in content, "Missing Kelly threshold constant"
        assert "QM_MAX_RISK_PER_TRADE_PCT" in content, "Missing max risk per trade constant"
        assert "QM_RISK_MULTIPLIER_MIN" in content, "Missing risk multiplier min constant"
        assert "QM_RISK_MULTIPLIER_MAX" in content, "Missing risk multiplier max constant"
        
        # Check magic number ranges
        assert "QM_MAGIC_BASE" in content, "Missing magic base constant"
        assert "QM_MAGIC_ANALYST_START" in content, "Missing analyst magic start"
        assert "QM_MAGIC_QUANT_START" in content, "Missing quant magic start"
        assert "QM_MAGIC_EXECUTOR_START" in content, "Missing executor magic start"
        
        # Check communication constants
        assert "QM_BRIDGE_HOST" in content, "Missing bridge host constant"
        assert "QM_BRIDGE_PORT" in content, "Missing bridge port constant"
        assert "QM_HEARTBEAT_INTERVAL_SEC" in content, "Missing heartbeat interval constant"
        
        # Check global variable names
        assert "QM_GV_RISK_MULTIPLIER" in content, "Missing risk multiplier GV name"
        assert "QM_GV_TRADING_ALLOWED" in content, "Missing trading allowed GV name"
    
    def test_types_structure(self, core_modules_path):
        """Test Types.mqh has proper structure"""
        types_file = core_modules_path / "Types.mqh"
        content = types_file.read_text()
        
        # Check header guard
        assert "#ifndef __QSL_TYPES_MQH__" in content, "Missing header guard ifndef"
        assert "#define __QSL_TYPES_MQH__" in content, "Missing header guard define"
        assert "#endif" in content, "Missing header guard endif"
        
        # Check enums
        assert "enum ENUM_TRADE_DECISION" in content, "Missing ENUM_TRADE_DECISION"
        assert "enum ENUM_AGENT_TYPE" in content, "Missing ENUM_AGENT_TYPE"
        assert "enum ENUM_RISK_STATUS" in content, "Missing ENUM_RISK_STATUS"
        assert "enum ENUM_ACCOUNT_STATUS" in content, "Missing ENUM_ACCOUNT_STATUS"
        assert "enum ENUM_SIGNAL_TYPE" in content, "Missing ENUM_SIGNAL_TYPE"
        assert "enum ENUM_STRATEGY_QUALITY" in content, "Missing ENUM_STRATEGY_QUALITY"
        
        # Check structures
        assert "struct STradeProposal" in content, "Missing STradeProposal structure"
        assert "struct SAccountState" in content, "Missing SAccountState structure"
        assert "struct SRiskParameters" in content, "Missing SRiskParameters structure"
        assert "struct SPositionInfo" in content, "Missing SPositionInfo structure"
        assert "struct SOrderInfo" in content, "Missing SOrderInfo structure"
        assert "struct SHeartbeatPayload" in content, "Missing SHeartbeatPayload structure"
        assert "struct SStrategyPerformance" in content, "Missing SStrategyPerformance structure"
        
        # Check conversion functions
        assert "TradeDecisionToString(" in content, "Missing TradeDecisionToString function"
        assert "AgentTypeToString(" in content, "Missing AgentTypeToString function"
        assert "RiskStatusToString(" in content, "Missing RiskStatusToString function"
        assert "SignalTypeToString(" in content, "Missing SignalTypeToString function"
        assert "StrategyQualityToString(" in content, "Missing StrategyQualityToString function"
        assert "GetStrategyQualityFromKelly(" in content, "Missing GetStrategyQualityFromKelly function"
    
    def test_test_ea_structure(self, test_ea_path):
        """Test TestCoreModules.mq5 has proper structure"""
        content = test_ea_path.read_text()
        
        # Check includes
        assert "#include <QuantMind/Core/BaseAgent.mqh>" in content, "Missing BaseAgent include"
        assert "#include <QuantMind/Core/Constants.mqh>" in content, "Missing Constants include"
        assert "#include <QuantMind/Core/Types.mqh>" in content, "Missing Types include"
        
        # Check EA functions
        assert "int OnInit()" in content, "Missing OnInit function"
        assert "void OnDeinit(" in content, "Missing OnDeinit function"
        assert "void OnTick()" in content, "Missing OnTick function"
        
        # Check test implementation
        assert "CBaseAgent" in content, "Missing CBaseAgent usage"
        assert "STradeProposal" in content, "Missing STradeProposal usage"
        assert "SAccountState" in content, "Missing SAccountState usage"
        assert "SRiskParameters" in content, "Missing SRiskParameters usage"
    
    def test_no_syntax_errors_in_base_agent(self, core_modules_path):
        """Test BaseAgent.mqh has no obvious syntax errors"""
        base_agent_file = core_modules_path / "BaseAgent.mqh"
        content = base_agent_file.read_text()
        
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"
        
        # Check for balanced parentheses in function definitions
        # This is a simple check - not perfect but catches obvious errors
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'void ' in line or 'bool ' in line or 'double ' in line or 'int ' in line or 'string ' in line:
                if '(' in line and ')' not in line and '{' not in line:
                    # Multi-line function definition is okay
                    continue
                if '(' in line:
                    open_paren = line.count('(')
                    close_paren = line.count(')')
                    if open_paren != close_paren:
                        # Check next few lines for closing paren
                        found_close = False
                        for j in range(i, min(i + 5, len(lines))):
                            if ')' in lines[j]:
                                found_close = True
                                break
                        if not found_close:
                            pytest.fail(f"Unbalanced parentheses at line {i}: {line.strip()}")
    
    def test_no_syntax_errors_in_constants(self, core_modules_path):
        """Test Constants.mqh has no obvious syntax errors"""
        constants_file = core_modules_path / "Constants.mqh"
        content = constants_file.read_text()
        
        # Check all #define statements are properly formatted
        # Note: Header guards have only 2 parts (#define NAME), which is valid
        define_lines = [line for line in content.split('\n') if line.strip().startswith('#define')]
        for line in define_lines:
            parts = line.strip().split()
            assert len(parts) >= 2, f"Invalid #define statement: {line.strip()}"
    
    def test_no_syntax_errors_in_types(self, core_modules_path):
        """Test Types.mqh has no obvious syntax errors"""
        types_file = core_modules_path / "Types.mqh"
        content = types_file.read_text()
        
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"
        
        # Check all struct definitions are properly closed
        struct_count = content.count('struct ')
        struct_end_count = content.count('};')
        # Note: struct_end_count might be higher due to nested structs or other uses
        assert struct_end_count >= struct_count, "Some struct definitions may not be properly closed"
    
    def test_core_modules_integration(self, core_modules_path):
        """Test that Core modules can work together"""
        # Read all three files
        base_agent = (core_modules_path / "BaseAgent.mqh").read_text()
        constants = (core_modules_path / "Constants.mqh").read_text()
        types = (core_modules_path / "Types.mqh").read_text()
        
        # BaseAgent should not depend on Types or Constants (minimal dependencies)
        # This is a design principle check
        assert "#include" not in base_agent or \
               "#include <QuantMind/Core/Types.mqh>" not in base_agent, \
               "BaseAgent should not include Types (violates minimal dependency principle)"
        
        # Constants should not depend on other modules
        assert "#include <QuantMind/" not in constants, \
               "Constants should not include other QSL modules"
        
        # Types should not depend on BaseAgent
        assert "#include <QuantMind/Core/BaseAgent.mqh>" not in types, \
               "Types should not include BaseAgent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
