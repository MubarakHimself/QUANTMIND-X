"""
Unit tests for Agent Configuration

Tests the AgentConfig class and YAML loading functionality.

**Validates: Phase 7.1 - Config Tests**
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.agents.config import AgentConfig, load_agent_config


class TestAgentConfig:
    """Test suite for AgentConfig."""
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "agent_id": "test_agent",
            "agent_type": "analyst",
            "name": "Test Agent",
            "llm_model": "gpt-4",
            "temperature": 0.5,
        }
        
        config = AgentConfig.from_dict(data)
        
        assert config.agent_id == "test_agent"
        assert config.agent_type == "analyst"
        assert config.name == "Test Agent"
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.5
    
    def test_config_with_defaults(self):
        """Test config with default values."""
        config = AgentConfig(
            agent_id="test",
            agent_type="analyst",
            name="Test"
        )
        
        assert config.llm_provider == "openrouter"
        assert config.max_tokens == 4096
        assert config.enable_tracing is True
        assert config.checkpointer_type == "memory"
    
    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(ValueError):
            AgentConfig(agent_id="", agent_type="analyst", name="Test")
        
        with pytest.raises(ValueError):
            AgentConfig(agent_id="test", agent_type="", name="Test")
    
    def test_config_from_yaml(self):
        """Test loading config from YAML."""
        yaml_content = """
agent_id: "yaml_agent"
agent_type: "quantcode"
name: "YAML Agent"
llm_model: "claude-3"
temperature: 0.7
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = AgentConfig.from_yaml(temp_path)
            
            assert config.agent_id == "yaml_agent"
            assert config.agent_type == "quantcode"
            assert config.name == "YAML Agent"
            assert config.llm_model == "claude-3"
            assert config.temperature == 0.7
        finally:
            os.unlink(temp_path)
    
    def test_config_with_overrides(self):
        """Test config with runtime overrides."""
        config = AgentConfig(
            agent_id="base",
            agent_type="analyst",
            name="Base",
            temperature=0.0
        )
        
        override_config = config.with_overrides(
            temperature=0.5,
            max_tokens=8192
        )
        
        assert override_config.agent_id == "base"
        assert override_config.temperature == 0.5
        assert override_config.max_tokens == 8192
    
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = AgentConfig(
            agent_id="test",
            agent_type="analyst",
            name="Test"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["agent_id"] == "test"
        assert config_dict["agent_type"] == "analyst"
    
    def test_get_tools_for_agent_type(self):
        """Test getting default tools for agent type."""
        config = AgentConfig(
            agent_id="test",
            agent_type="analyst",
            name="Test"
        )
        
        tools = config.get_tools_for_agent_type()
        
        assert "research_market_data" in tools
        assert "parse_nprd" in tools
    
    def test_load_agent_config_convenience(self):
        """Test convenience function."""
        yaml_content = """
agent_id: "convenience_test"
agent_type: "copilot"
name: "Convenience Test"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = load_agent_config(temp_path, temperature=0.9)
            
            assert config.agent_id == "convenience_test"
            assert config.temperature == 0.9
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
