"""
Agent Configuration System

Provides YAML-based configuration for factory-created agents with validation,
overrides, and flexible customization options.

**Validates: Phase 1.1 - Configuration System**
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Configuration dataclass for factory-created agents.
    
    Supports loading from YAML, dict, and runtime overrides.
    """
    agent_id: str
    agent_type: str
    name: str
    llm_provider: str = "openrouter"
    llm_model: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    tools: List[str] = field(default_factory=list)
    tool_config: Dict[str, Any] = field(default_factory=dict)
    state_class: str = "MessagesState"
    checkpointer_type: str = "memory"
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_streaming: bool = False
    timeout_seconds: int = 300
    max_retries: int = 3
    custom: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields with defaults
    checkpointer_config: Optional[Dict[str, Any]] = None
    memory_namespace: tuple = ("memories", "default", "default")
    workspace_path: str = "workspaces/default"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate required fields."""
        if not self.agent_id:
            raise ValueError("agent_id is required")
        if not self.agent_type:
            raise ValueError("agent_type is required")
        if not self.name:
            raise ValueError("name is required")
        
        # Validate agent_type
        valid_types = ["analyst", "quantcode", "copilot", "router", "executor"]
        if self.agent_type not in valid_types:
            logger.warning(
                f"Unknown agent_type: {self.agent_type}, expected one of {valid_types}"
            )
        
        # Validate checkpointer_type
        valid_checkpointers = ["memory", "postgres", "redis"]
        if self.checkpointer_type not in valid_checkpointers:
            logger.warning(
                f"Unknown checkpointer_type: {self.checkpointer_type}, "
                f"defaulting to memory"
            )
            self.checkpointer_type = "memory"
        
        # Validate log_level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            logger.warning(f"Invalid log_level: {self.log_level}, defaulting to INFO")
            self.log_level = "INFO"
    
    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            AgentConfig instance
            
        Example:
            >>> config = AgentConfig.from_yaml("config/agents/analyst.yaml")
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            raise ValueError(f"Empty configuration file: {path}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        """
        Create configuration from a dictionary.
        
        Args:
            data: Configuration dictionary
            
        Returns:
            AgentConfig instance
        """
        # Extract known fields
        known_fields = {
            'agent_id', 'agent_type', 'name', 'llm_provider', 'llm_model',
            'temperature', 'max_tokens', 'tools', 'tool_config', 'state_class',
            'checkpointer_type', 'enable_tracing', 'enable_metrics', 'enable_logging',
            'log_level', 'enable_streaming', 'timeout_seconds', 'max_retries',
            'custom', 'checkpointer_config', 'memory_namespace', 'workspace_path'
        }
        
        # Separate known and custom fields
        config_data = {}
        custom_data = {}
        
        for key, value in data.items():
            if key in known_fields:
                config_data[key] = value
            else:
                custom_data[key] = value
        
        # Add custom fields to 'custom' if not already a dict
        if custom_data:
            if 'custom' in config_data and isinstance(config_data['custom'], dict):
                config_data['custom'].update(custom_data)
            else:
                config_data['custom'] = custom_data
        
        # Convert memory_namespace from list to tuple if needed
        if 'memory_namespace' in config_data:
            ns = config_data['memory_namespace']
            if isinstance(ns, list):
                config_data['memory_namespace'] = tuple(ns)
        
        return cls(**config_data)
    
    def with_overrides(self, **kwargs) -> "AgentConfig":
        """
        Create a new config with runtime overrides.
        
        Args:
            **kwargs: Configuration fields to override
            
        Returns:
            New AgentConfig with overrides applied
            
        Example:
            >>> config = base_config.with_overrides(temperature=0.5, max_tokens=8192)
        """
        # Get current config as dict
        config_dict = asdict(self)
        
        # Update with overrides
        for key, value in kwargs.items():
            if key in config_dict:
                config_dict[key] = value
            else:
                # Add to custom if not a known field
                if 'custom' not in config_dict:
                    config_dict['custom'] = {}
                config_dict['custom'][key] = value
        
        return AgentConfig.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: str = None) -> str:
        """
        Export configuration to YAML format.
        
        Args:
            path: Optional path to save YAML file
            
        Returns:
            YAML string representation
        """
        # Convert to dict, handling special types
        config_dict = self.to_dict()
        
        # Convert tuple to list for YAML
        if 'memory_namespace' in config_dict:
            config_dict['memory_namespace'] = list(config_dict['memory_namespace'])
        
        yaml_str = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        
        if path:
            with open(path, 'w') as f:
                f.write(yaml_str)
        
        return yaml_str
    
    def get_tools_for_agent_type(self) -> List[str]:
        """
        Get default tools for the configured agent type.
        
        Returns:
            List of tool names for this agent type
        """
        # Default tool mappings by agent type
        default_tools = {
            "analyst": [
                "research_market_data",
                "extract_insights",
                "parse_nprd",
                "validate_nprd",
                "generate_trd",
                "analyze_backtest",
                "compare_strategies",
                "generate_optimization_report",
            ],
            "quantcode": [
                "create_strategy_plan",
                "generate_mql5_code",
                "validate_code",
                "compile_code",
                "run_backtest",
                "analyze_strategy_performance",
                "optimize_parameters",
                "create_documentation",
            ],
            "copilot": [
                "deploy_expert_advisor",
                "monitor_deployment",
                "manage_risk_parameters",
                "validate_deployment",
                "get_account_info",
                "execute_trade",
                "close_position",
            ],
            "router": [
                # Router uses classification, no direct tools
            ],
            "executor": [
                "execute_trade",
                "close_position",
                "modify_position",
                "get_positions",
                "get_account_info",
            ],
        }
        
        return self.tools or default_tools.get(self.agent_type, [])
    
    @property
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        try:
            self._validate()
            return True
        except (ValueError, TypeError):
            return False


def load_agent_config(config_path: str, **overrides) -> AgentConfig:
    """
    Convenience function to load agent configuration with optional overrides.
    
    Args:
        config_path: Path to YAML configuration file
        **overrides: Optional runtime overrides
        
    Returns:
        AgentConfig instance
        
    Example:
        >>> config = load_agent_config("config/agents/analyst.yaml", temperature=0.5)
    """
    config = AgentConfig.from_yaml(config_path)
    if overrides:
        config = config.with_overrides(**overrides)
    return config
