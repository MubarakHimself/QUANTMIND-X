"""
MQL5 Compiler Module

Provides compilation capabilities for MQL5 EA files using Docker-based
MetaTrader 5 compiler on Contabo.
"""
from src.mql5.compiler.docker_compiler import DockerMQL5Compiler, CompilationResult
from src.mql5.compiler.error_parser import MQL5ErrorParser, CompilationError
from src.mql5.compiler.autocorrect import AutoCorrector, CorrectionResult
from src.mql5.compiler.service import (
    MQL5CompilationService,
    AutoCorrectionResult,
    CompilationServiceResult,
    get_compilation_service,
)

__all__ = [
    "DockerMQL5Compiler",
    "CompilationResult",
    "MQL5ErrorParser",
    "CompilationError",
    "AutoCorrector",
    "CorrectionResult",
    "MQL5CompilationService",
    "AutoCorrectionResult",
    "CompilationServiceResult",
    "get_compilation_service",
]
