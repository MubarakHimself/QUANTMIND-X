# src/agents/departments/schemas/development.py
"""
Development Department Schemas

Pydantic models for structured outputs in the Development department.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EALanguage(str, Enum):
    """EA/Bot programming languages."""
    MQL5 = "mql5"
    PINESCRIPT = "pinescript"
    PYTHON = "python"


class EAStatus(str, Enum):
    """EA lifecycle status."""
    DRAFT = "draft"
    CODE_GENERATED = "code_generated"
    TESTING = "testing"
    DEPLOYED = "deployed"
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    FAILED = "failed"


class EACreationRequest(BaseModel):
    """EA creation request."""
    name: str = Field(..., description="EA name", min_length=1)
    strategy_type: str = Field(..., description="Strategy type")
    language: EALanguage
    description: str = Field(default="", description="EA description")
    symbols: List[str] = Field(default_factory=list, description="Target symbols")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="EA parameters")

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "name": "TrendFollower",
                "strategy_type": "trend",
                "language": "mql5",
                "description": "SMA crossover trend follower",
                "symbols": ["EURUSD", "GBPUSD"]
            }
        }


class EACreationResponse(BaseModel):
    """EA creation response."""
    ea_id: str = Field(..., description="Unique EA identifier")
    name: str
    language: EALanguage
    status: EAStatus
    file_path: str = Field(..., description="Path to generated file")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    message: str = Field(default="", description="Status message")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "ea_id": "EA_001",
                "name": "TrendFollower",
                "language": "mql5",
                "status": "code_generated",
                "file_path": "/experts/TrendFollower.mq5"
            }
        }


class CodeGenerationRequest(BaseModel):
    """Code generation request."""
    file_type: str = Field(..., description="File type (.mq5, .py, .pine)")
    content: str = Field(..., description="Trading logic code")
    language: EALanguage
    metadata: Dict[str, Any] = Field(default_factory=dict)
    template: Optional[str] = Field(default=None, description="Code template to use")


class CodeGenerationResult(BaseModel):
    """Code generation result."""
    file_path: str = Field(..., description="Generated file path")
    code: str = Field(..., description="Generated code")
    language: EALanguage
    syntax_valid: bool = Field(default=True)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TestResult(BaseModel):
    """Test execution result."""
    test_id: str = Field(..., description="Test identifier")
    test_name: str = Field(..., description="Test name")
    ea_id: str = Field(..., description="EA being tested")
    symbol: str = Field(..., description="Test symbol")
    timeframe: str = Field(..., description="Test timeframe")
    start_date: datetime
    end_date: datetime
    status: str = Field(..., description="Test status")
    results: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    execution_time_ms: int = Field(default=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "test_id": "TEST_001",
                "test_name": "TrendFollower_Backtest",
                "ea_id": "EA_001",
                "symbol": "EURUSD",
                "timeframe": "H1",
                "status": "passed"
            }
        }


class DeploymentConfig(BaseModel):
    """EA deployment configuration."""
    ea_id: str = Field(..., description="EA to deploy")
    symbol: str = Field(..., description="Trading symbol")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    risk_mode: str = Field(default="conservative", description="Risk mode")
    max_spread: float = Field(default=3.0, description="Maximum spread")
    magic_number: Optional[int] = Field(default=None, description="Magic number")
    comment: Optional[str] = Field(default=None, description="Deployment comment")


class DeploymentResult(BaseModel):
    """Deployment result."""
    deployment_id: str = Field(..., description="Unique deployment identifier")
    ea_id: str
    symbol: str
    status: str = Field(..., description="Deployment status")
    instance_id: Optional[str] = Field(default=None, description="Running instance ID")
    message: str = Field(default="")
    error: Optional[str] = Field(default=None)
    deployed_at: datetime = Field(default_factory=datetime.utcnow)


class ParameterDefinition(BaseModel):
    """EA parameter definition."""
    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type")
    default: Any = Field(..., description="Default value")
    min_value: Optional[float] = Field(default=None)
    max_value: Optional[float] = Field(default=None)
    description: str = Field(default="")
    is_required: bool = Field(default=False)


class EADetails(BaseModel):
    """Complete EA details."""
    ea_id: str
    name: str
    language: EALanguage
    strategy_type: str
    description: str
    status: EAStatus
    file_path: str
    parameters: List[ParameterDefinition] = Field(default_factory=list)
    symbols: List[str] = Field(default_factory=list)
    deployment_config: Optional[DeploymentConfig] = None
    created_at: datetime
    updated_at: datetime
    last_test: Optional[TestResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CodeAnalysisResult(BaseModel):
    """Code analysis result."""
    file_path: str
    language: EALanguage
    issues: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    suggestions: List[Dict[str, Any]] = Field(default_factory=list)
    complexity_score: Optional[float] = None
    maintainability_score: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
