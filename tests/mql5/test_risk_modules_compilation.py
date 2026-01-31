"""
Test Risk Modules Compilation

This test verifies that all Risk QSL modules exist and have proper structure.
Tests PropManager, RiskClient, and KellySizer modules.
"""

import pytest
from pathlib import Path


class TestRiskModulesCompilation:
    """Test suite for Risk QSL modules compilation verification"""
    
    @pytest.fixture
    def risk_modules_path(self):
        """Get path to Risk modules directory"""
        return Path("src/mql5/Include/QuantMind/Risk")
    
    @pytest.fixture
    def test_ea_path(self):
        """Get path to test EA"""
        return Path("src/mql5/Experts/TestRiskModules.mq5")
    
    def test_risk_directory_exists(self, risk_modules_path):
        """Test that Risk directory exists"""
        assert risk_modules_path.exists(), "Risk modules directory does not exist"
        assert risk_modules_path.is_dir(), "Risk path is not a directory"
    
    def test_prop_manager_exists(self, risk_modules_path):
        """Test that PropManager.mqh exists"""
        prop_manager_file = risk_modules_path / "PropManager.mqh"
        assert prop_manager_file.exists(), "PropManager.mqh does not exist"
    
    def test_risk_client_exists(self, risk_modules_path):
        """Test that RiskClient.mqh exists"""
        risk_client_file = risk_modules_path / "RiskClient.mqh"
        assert risk_client_file.exists(), "RiskClient.mqh does not exist"
    
    def test_kelly_sizer_exists(self, risk_modules_path):
        """Test that KellySizer.mqh exists"""
        kelly_sizer_file = risk_modules_path / "KellySizer.mqh"
        assert kelly_sizer_file.exists(), "KellySizer.mqh does not exist"
    
    def test_test_ea_exists(self, test_ea_path):
        """Test that TestRiskModules.mq5 exists"""
        assert test_ea_path.exists(), "TestRiskModules.mq5 does not exist"
    
    def test_prop_manager_structure(self, risk_modules_path):
        """Test PropManager.mqh has proper structure"""
        prop_manager_file = risk_modules_path / "PropManager.mqh"
        content = prop_manager_file.read_text()
        
        # Check header guard
        assert "#ifndef __QSL_PROP_MANAGER_MQH__" in content, "Missing header guard ifndef"
        assert "#define __QSL_PROP_MANAGER_MQH__" in content, "Missing header guard define"
        assert "#endif" in content, "Missing header guard endif"
        
        # Check class definition
        assert "class CPropManager" in content, "Missing CPropManager class definition"
        
        # Check key methods
        assert "Initialize(" in content, "Missing Initialize method"
        assert "Update(" in content, "Missing Update method"
        assert "CalculateDailyDrawdown(" in content, "Missing CalculateDailyDrawdown method"
        assert "CheckHardStop(" in content, "Missing CheckHardStop method"
        assert "SetNewsGuard(" in content, "Missing SetNewsGuard method"
        assert "CalculateQuadraticThrottle(" in content, "Missing CalculateQuadraticThrottle method"
        assert "IsTradingAllowed(" in content, "Missing IsTradingAllowed method"
        assert "GetRiskMultiplier(" in content, "Missing GetRiskMultiplier method"
        assert "GetAccountState(" in content, "Missing GetAccountState method"
        
        # Check includes
        assert "#include <QuantMind/Core/Constants.mqh>" in content, "Missing Constants include"
        assert "#include <QuantMind/Core/Types.mqh>" in content, "Missing Types include"
    
    def test_risk_client_structure(self, risk_modules_path):
        """Test RiskClient.mqh has proper structure"""
        risk_client_file = risk_modules_path / "RiskClient.mqh"
        content = risk_client_file.read_text()
        
        # Check header guard
        assert "#ifndef __QSL_RISK_CLIENT_MQH__" in content, "Missing header guard ifndef"
        assert "#define __QSL_RISK_CLIENT_MQH__" in content, "Missing header guard define"
        assert "#endif" in content, "Missing header guard endif"
        
        # Check key functions
        assert "GetRiskMultiplier(" in content, "Missing GetRiskMultiplier function"
        assert "ReadRiskFromFile(" in content, "Missing ReadRiskFromFile function"
        assert "SendHeartbeat(" in content, "Missing SendHeartbeat function"
        
        # Check constants
        assert "QM_RISK_MULTIPLIER_VAR" in content, "Missing risk multiplier variable constant"
        assert "QM_RISK_MATRIX_FILE" in content, "Missing risk matrix file constant"
        
        # Check includes
        assert "#include <QuantMind/Utils/JSON.mqh>" in content, "Missing JSON include"
    
    def test_kelly_sizer_structure(self, risk_modules_path):
        """Test KellySizer.mqh has proper structure"""
        kelly_sizer_file = risk_modules_path / "KellySizer.mqh"
        content = kelly_sizer_file.read_text()
        
        # Check header guard
        assert "#ifndef __QSL_KELLY_SIZER_MQH__" in content, "Missing header guard ifndef"
        assert "#define __QSL_KELLY_SIZER_MQH__" in content, "Missing header guard define"
        assert "#endif" in content, "Missing header guard endif"
        
        # Check class definition
        assert "class QMKellySizer" in content, "Missing QMKellySizer class definition"
        
        # Check key methods
        assert "CalculateKellyFraction(" in content, "Missing CalculateKellyFraction method"
        assert "CalculateLotSize(" in content, "Missing CalculateLotSize method"
        assert "GetLastError(" in content, "Missing GetLastError method"
        assert "GetErrorMessage(" in content, "Missing GetErrorMessage method"
    
    def test_test_ea_structure(self, test_ea_path):
        """Test TestRiskModules.mq5 has proper structure"""
        content = test_ea_path.read_text()
        
        # Check includes
        assert "#include <QuantMind/Risk/PropManager.mqh>" in content, "Missing PropManager include"
        assert "#include <QuantMind/Risk/RiskClient.mqh>" in content, "Missing RiskClient include"
        assert "#include <QuantMind/Risk/KellySizer.mqh>" in content, "Missing KellySizer include"
        
        # Check EA functions
        assert "int OnInit()" in content, "Missing OnInit function"
        assert "void OnDeinit(" in content, "Missing OnDeinit function"
        assert "void OnTick()" in content, "Missing OnTick function"
        
        # Check test implementation
        assert "CPropManager" in content, "Missing CPropManager usage"
        assert "QMKellySizer" in content, "Missing QMKellySizer usage"
        assert "GetRiskMultiplier" in content, "Missing GetRiskMultiplier usage"
        assert "SendHeartbeat" in content, "Missing SendHeartbeat usage"
    
    def test_prop_manager_daily_drawdown_logic(self, risk_modules_path):
        """Test PropManager has daily drawdown tracking logic"""
        prop_manager_file = risk_modules_path / "PropManager.mqh"
        content = prop_manager_file.read_text()
        
        # Check for daily tracking variables
        assert "m_startBalance" in content, "Missing start balance tracking"
        assert "m_highWaterMark" in content, "Missing high water mark tracking"
        assert "m_dailyPnL" in content, "Missing daily P&L tracking"
        assert "m_currentDrawdown" in content, "Missing current drawdown tracking"
        
        # Check for reset functionality
        assert "ResetDailyMetrics(" in content, "Missing daily metrics reset"
    
    def test_prop_manager_news_guard_logic(self, risk_modules_path):
        """Test PropManager has news guard functionality"""
        prop_manager_file = risk_modules_path / "PropManager.mqh"
        content = prop_manager_file.read_text()
        
        # Check for news guard variables
        assert "m_newsGuardActive" in content, "Missing news guard active flag"
        
        # Check for news guard methods
        assert "SetNewsGuard(" in content, "Missing SetNewsGuard method"
        assert "IsNewsGuardActive(" in content, "Missing IsNewsGuardActive method"
        
        # Check for KILL_ZONE reference
        assert "KILL_ZONE" in content or "NEWS_GUARD" in content, "Missing KILL_ZONE/NEWS_GUARD reference"
    
    def test_prop_manager_quadratic_throttle_logic(self, risk_modules_path):
        """Test PropManager has quadratic throttle calculation"""
        prop_manager_file = risk_modules_path / "PropManager.mqh"
        content = prop_manager_file.read_text()
        
        # Check for quadratic throttle method
        assert "CalculateQuadraticThrottle(" in content, "Missing CalculateQuadraticThrottle method"
        
        # Check for formula components
        assert "MathPow" in content, "Missing power calculation for quadratic formula"
        assert "remainingCapacity" in content or "maxLoss" in content, "Missing capacity calculation"
    
    def test_prop_manager_hard_stop_logic(self, risk_modules_path):
        """Test PropManager has hard stop enforcement"""
        prop_manager_file = risk_modules_path / "PropManager.mqh"
        content = prop_manager_file.read_text()
        
        # Check for hard stop variables
        assert "m_hardStopActive" in content, "Missing hard stop active flag"
        
        # Check for hard stop methods
        assert "CheckHardStop(" in content, "Missing CheckHardStop method"
        assert "IsHardStopActive(" in content, "Missing IsHardStopActive method"
        
        # Check for threshold reference
        assert "QM_EFFECTIVE_LIMIT_PCT" in content or "4.5" in content or "4.0" in content, \
               "Missing hard stop threshold reference"
    
    def test_risk_client_heartbeat_logic(self, risk_modules_path):
        """Test RiskClient has heartbeat transmission"""
        risk_client_file = risk_modules_path / "RiskClient.mqh"
        content = risk_client_file.read_text()
        
        # Check for heartbeat function
        assert "SendHeartbeat(" in content, "Missing SendHeartbeat function"
        
        # Check for WebRequest usage
        assert "WebRequest" in content, "Missing WebRequest call"
        
        # Check for JSON payload construction
        assert "ea_name" in content, "Missing ea_name in payload"
        assert "symbol" in content, "Missing symbol in payload"
        assert "magic_number" in content, "Missing magic_number in payload"
        assert "risk_multiplier" in content, "Missing risk_multiplier in payload"
    
    def test_risk_client_global_variable_fast_path(self, risk_modules_path):
        """Test RiskClient has GlobalVariable fast path"""
        risk_client_file = risk_modules_path / "RiskClient.mqh"
        content = risk_client_file.read_text()
        
        # Check for GlobalVariable usage
        assert "GlobalVariableCheck" in content or "GlobalVariableGet" in content, \
               "Missing GlobalVariable fast path"
        
        # Check for fast path comment or logic
        assert "Fast path" in content or "fast path" in content, "Missing fast path documentation"
    
    def test_risk_client_file_fallback(self, risk_modules_path):
        """Test RiskClient has file fallback mechanism"""
        risk_client_file = risk_modules_path / "RiskClient.mqh"
        content = risk_client_file.read_text()
        
        # Check for file reading
        assert "ReadRiskFromFile(" in content, "Missing ReadRiskFromFile function"
        assert "FileOpen" in content, "Missing FileOpen call"
        assert "risk_matrix.json" in content, "Missing risk_matrix.json reference"
        
        # Check for fallback comment or logic
        assert "Fallback" in content or "fallback" in content, "Missing fallback documentation"
    
    def test_kelly_sizer_position_sizing_logic(self, risk_modules_path):
        """Test KellySizer has position sizing logic"""
        kelly_sizer_file = risk_modules_path / "KellySizer.mqh"
        content = kelly_sizer_file.read_text()
        
        # Check for Kelly formula components
        assert "winRate" in content, "Missing win rate parameter"
        assert "avgWin" in content, "Missing average win parameter"
        assert "avgLoss" in content, "Missing average loss parameter"
        
        # Check for Kelly formula: f* = (bp - q) / b
        assert "payoff" in content or "b =" in content, "Missing payoff ratio calculation"
        
        # Check for lot size calculation
        assert "CalculateLotSize(" in content, "Missing CalculateLotSize method"
        assert "equity" in content, "Missing equity parameter"
        assert "stopLoss" in content or "stopLossPips" in content, "Missing stop loss parameter"
    
    def test_no_syntax_errors_in_prop_manager(self, risk_modules_path):
        """Test PropManager.mqh has no obvious syntax errors"""
        prop_manager_file = risk_modules_path / "PropManager.mqh"
        content = prop_manager_file.read_text()
        
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"
    
    def test_no_syntax_errors_in_risk_client(self, risk_modules_path):
        """Test RiskClient.mqh has no obvious syntax errors"""
        risk_client_file = risk_modules_path / "RiskClient.mqh"
        content = risk_client_file.read_text()
        
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"
    
    def test_no_syntax_errors_in_kelly_sizer(self, risk_modules_path):
        """Test KellySizer.mqh has no obvious syntax errors"""
        kelly_sizer_file = risk_modules_path / "KellySizer.mqh"
        content = kelly_sizer_file.read_text()
        
        # Check for balanced braces
        open_braces = content.count('{')
        close_braces = content.count('}')
        assert open_braces == close_braces, f"Unbalanced braces: {open_braces} open, {close_braces} close"
    
    def test_risk_modules_integration(self, risk_modules_path):
        """Test that Risk modules have proper dependencies"""
        prop_manager = (risk_modules_path / "PropManager.mqh").read_text()
        risk_client = (risk_modules_path / "RiskClient.mqh").read_text()
        kelly_sizer = (risk_modules_path / "KellySizer.mqh").read_text()
        
        # PropManager should include Core modules
        assert "#include <QuantMind/Core/" in prop_manager, \
               "PropManager should include Core modules"
        
        # RiskClient should include Utils/JSON
        assert "#include <QuantMind/Utils/JSON.mqh>" in risk_client, \
               "RiskClient should include JSON utility"
        
        # KellySizer should be self-contained (no QSL includes)
        assert "#include <QuantMind/" not in kelly_sizer, \
               "KellySizer should be self-contained"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
