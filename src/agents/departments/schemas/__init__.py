# src/agents/departments/schemas/__init__.py
"""
Pydantic Schemas for Structured Outputs

This module provides Pydantic models for structured outputs across all departments.
Each department has its own schema file with proper validation and type hints.
"""
from src.agents.departments.schemas.common import (
    DepartmentOutput,
    TaskResult,
    ErrorResponse,
)
from src.agents.departments.schemas.research import (
    StrategyOutput,
    BacktestResult,
    AlphaFactor,
)
from src.agents.departments.schemas.trading import (
    OrderRequest,
    OrderResponse,
    FillInfo,
)
from src.agents.departments.schemas.risk import (
    PositionSizeRequest,
    PositionSizeResponse,
    DrawdownInfo,
    VaRResult,
)
from src.agents.departments.schemas.portfolio import (
    AllocationRequest,
    AllocationResult,
    RebalancePlan,
    PerformanceMetrics,
)
from src.agents.departments.schemas.development import (
    EACreationRequest,
    EACreationResponse,
    CodeGenerationRequest,
    TestResult,
    DeploymentConfig,
)

__all__ = [
    # Common
    "DepartmentOutput",
    "TaskResult",
    "ErrorResponse",
    # Research
    "StrategyOutput",
    "BacktestResult",
    "AlphaFactor",
    # Trading
    "OrderRequest",
    "OrderResponse",
    "FillInfo",
    # Risk
    "PositionSizeRequest",
    "PositionSizeResponse",
    "DrawdownInfo",
    "VaRResult",
    # Portfolio
    "AllocationRequest",
    "AllocationResult",
    "RebalancePlan",
    "PerformanceMetrics",
    # Development
    "EACreationRequest",
    "EACreationResponse",
    "CodeGenerationRequest",
    "TestResult",
    "DeploymentConfig",
]
