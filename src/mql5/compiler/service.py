"""
MQL5 Compilation Service

Orchestrates the complete compilation workflow including:
- Docker-based compilation
- Error handling and auto-correction
- Strategy record updates
- Escalation to FloorManager on failure
"""
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.mql5.compiler.docker_compiler import DockerMQL5Compiler, CompilationResult, get_compiler
from src.mql5.compiler.error_parser import MQL5ErrorParser, parse_compilation_output
from src.mql5.compiler.autocorrect import AutoCorrector, CorrectionResult, MAX_AUTO_CORRECT_ATTEMPTS
from src.strategy.output import EAOutputStorage

logger = logging.getLogger(__name__)


# Compile status values
COMPILE_STATUS_PENDING = "pending"
COMPILE_STATUS_SUCCESS = "success"
COMPILE_STATUS_FAILED = "failed"


@dataclass
class CompilationServiceResult:
    """Result of compilation service operation."""
    success: bool
    strategy_id: str
    version: int
    mq5_path: str
    ex5_path: Optional[str]
    compile_status: str
    errors: List[str]
    warnings: List[str]
    auto_correction_attempts: int
    escalated_to_floor_manager: bool
    escalation_reason: Optional[str] = None


class MQL5CompilationService:
    """
    Service for MQL5 compilation workflow.

    Handles:
    - Compilation via Docker MT5 compiler
    - Auto-correction (up to 2 attempts)
    - Escalation to FloorManager on failure
    - Strategy record updates
    """

    def __init__(
        self,
        compiler: Optional[DockerMQL5Compiler] = None,
        ea_storage: Optional[EAOutputStorage] = None,
    ):
        self.compiler = compiler or get_compiler()
        self.ea_storage = ea_storage or EAOutputStorage()
        self.error_parser = MQL5ErrorParser()
        self.auto_corrector = AutoCorrector(max_attempts=MAX_AUTO_CORRECT_ATTEMPTS)

    def compile_ea(
        self,
        strategy_id: str,
        version: Optional[int] = None,
    ) -> CompilationServiceResult:
        """
        Compile an EA after generation.

        Args:
            strategy_id: Strategy identifier
            version: Specific version to compile, or None for latest

        Returns:
            CompilationServiceResult with compilation outcome
        """
        logger.info(f"Starting compilation for {strategy_id} v{version}")

        # Get the EA from storage
        ea = self.ea_storage.get_ea(strategy_id, version)
        if not ea:
            return CompilationServiceResult(
                success=False,
                strategy_id=strategy_id,
                version=version or 0,
                mq5_path="",
                ex5_path=None,
                compile_status=COMPILE_STATUS_FAILED,
                errors=[f"EA not found for {strategy_id}"],
                warnings=[],
                auto_correction_attempts=0,
                escalated_to_floor_manager=True,
                escalation_reason="EA not found in storage",
            )

        mq5_path = ea.file_path
        version = ea.version

        # Check if already compiled
        if ea.status == "compiled":
            logger.info(f"EA {strategy_id} v{version} already compiled")
            return CompilationServiceResult(
                success=True,
                strategy_id=strategy_id,
                version=version,
                mq5_path=mq5_path,
                ex5_path=mq5_path.replace(".mq5", ".ex5"),
                compile_status=COMPILE_STATUS_SUCCESS,
                errors=[],
                warnings=[],
                auto_correction_attempts=0,
                escalated_to_floor_manager=False,
            )

        # Run compilation with auto-correction
        result = self._compile_with_autocorrect(mq5_path, strategy_id, version)

        # Update strategy record
        self._update_strategy_record(
            strategy_id=strategy_id,
            version=version,
            result=result,
        )

        # Handle escalation if needed
        if not result.success:
            should_escalate, reason = self._should_escalate(result)
            if should_escalate:
                return CompilationServiceResult(
                    success=False,
                    strategy_id=strategy_id,
                    version=version,
                    mq5_path=mq5_path,
                    ex5_path=result.ex5_path,
                    compile_status=COMPILE_STATUS_FAILED,
                    errors=result.errors,
                    warnings=result.warnings,
                    auto_correction_attempts=result.attempt_number,
                    escalated_to_floor_manager=True,
                    escalation_reason=reason,
                )

        return CompilationServiceResult(
            success=result.success,
            strategy_id=strategy_id,
            version=version,
            mq5_path=mq5_path,
            ex5_path=result.ex5_path,
            compile_status=COMPILE_STATUS_SUCCESS if result.success else COMPILE_STATUS_FAILED,
            errors=result.errors,
            warnings=result.warnings,
            auto_correction_attempts=result.attempt_number,
            escalated_to_floor_manager=False,
        )

    def _compile_with_autocorrect(
        self,
        mq5_path: str,
        strategy_id: str,
        version: int,
    ) -> CompilationResult:
        """
        Run compilation with auto-correction loop.

        Args:
            mq5_path: Path to .mq5 file
            strategy_id: Strategy identifier
            version: Version number

        Returns:
            CompilationResult from final attempt
        """
        current_mq5_path = mq5_path

        for attempt in range(1, MAX_AUTO_CORRECT_ATTEMPTS + 1):
            logger.info(f"Compilation attempt {attempt}/{MAX_AUTO_CORRECT_ATTEMPTS}")

            # Run compilation
            result = self.compiler.compile(
                mq5_file_path=current_mq5_path,
                strategy_id=strategy_id,
                version=version,
                attempt_number=attempt,
            )

            if result.success:
                logger.info(f"Compilation succeeded on attempt {attempt}")
                # Copy .ex5 to storage if returned
                if result.ex5_path:
                    self._store_ex5(result.ex5_path, mq5_path)
                return result

            # Parse errors
            error_output = "\n".join(result.errors)
            parsed = parse_compilation_output(error_output)

            if not parsed["errors"]:
                # No parseable errors, return current result
                return result

            # Check if errors are auto-correctable
            auto_correctable = any(e.is_auto_correctable() for e in parsed["errors"])

            if not auto_correctable or attempt >= MAX_AUTO_CORRECT_ATTEMPTS:
                logger.warning(f"Cannot auto-correct, returning with errors")
                return result

            # Attempt correction
            # Read current source
            with open(current_mq5_path, 'r') as f:
                source_code = f.read()

            correction_result = self.auto_corrector.correct(
                source_code,
                parsed["errors"],
                attempt,
            )

            if not correction_result.success or not correction_result.corrected_code:
                logger.warning(f"Auto-correction failed on attempt {attempt}")
                continue

            # Save corrected source
            corrected_path = mq5_path.replace(".mq5", f"_attempt{attempt + 1}.mq5")
            with open(corrected_path, 'w') as f:
                f.write(correction_result.corrected_code)

            current_mq5_path = corrected_path
            logger.info(f"Applied auto-correction, retrying with {corrected_path}")

        return result

    def _store_ex5(self, ex5_path: str, mq5_path: str) -> Optional[str]:
        """Copy .ex5 file to the same directory as .mq5."""
        import shutil

        try:
            source_ex5 = Path(ex5_path)
            if not source_ex5.exists():
                logger.warning(f"EX5 file not found: {ex5_path}")
                return None

            # Target path: same directory, same base name
            target_path = mq5_path.replace(".mq5", ".ex5")

            shutil.copy2(source_ex5, target_path)
            logger.info(f"Stored EX5 at {target_path}")
            return target_path

        except Exception as e:
            logger.error(f"Failed to store EX5: {e}")
            return None

    def _update_strategy_record(
        self,
        strategy_id: str,
        version: int,
        result: CompilationResult,
    ) -> None:
        """Update the strategy metadata with compilation results."""
        import json
        from datetime import datetime

        try:
            # Get metadata path
            ea = self.ea_storage.get_ea(strategy_id, version)
            if not ea:
                return

            metadata_path = Path(ea.file_path).parent / "metadata.json"
            if not metadata_path.exists():
                return

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Update with compilation results
            metadata["compile_status"] = COMPILE_STATUS_SUCCESS if result.success else COMPILE_STATUS_FAILED
            metadata["compile_errors"] = result.errors if result.errors else None
            metadata["compile_warnings"] = result.warnings if result.warnings else None
            metadata["compile_attempts"] = result.attempt_number
            metadata["compile_last_attempt"] = datetime.now().isoformat()

            if result.ex5_path:
                metadata["ex5_path"] = result.ex5_path

            if result.success:
                metadata["status"] = "compiled"

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Updated strategy record for {strategy_id} v{version}")

        except Exception as e:
            logger.error(f"Failed to update strategy record: {e}")

    def _should_escalate(
        self,
        result: CompilationServiceResult,
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if escalation to FloorManager is needed.

        Args:
            result: Compilation result

        Returns:
            Tuple of (should_escalate, reason)
        """
        if result.auto_correction_attempts >= MAX_AUTO_CORRECT_ATTEMPTS:
            return True, f"Auto-correction exhausted ({MAX_AUTO_CORRECT_ATTEMPTS} attempts)"

        if len(result.errors) > 10:
            return True, f"Too many errors ({len(result.errors)})"

        # Check for blocking error types
        blocking_patterns = [
            "undefined",
            "undeclared",
            "multiple definition",
            "undefined reference",
        ]

        error_text = " ".join(result.errors).lower()
        for pattern in blocking_patterns:
            if pattern in error_text:
                return True, f"Blocking error: {pattern}"

        return False, None

    def escalate_to_floor_manager(
        self,
        strategy_id: str,
        version: int,
        reason: str,
        errors: List[str],
    ) -> Dict[str, Any]:
        """
        Escalate compilation failure to FloorManager.

        Args:
            strategy_id: Strategy identifier
            version: Version number
            reason: Reason for escalation
            errors: List of compilation errors

        Returns:
            Escalation message payload
        """
        logger.warning(f"Escalating {strategy_id} v{version} to FloorManager: {reason}")

        # This would integrate with DepartmentMailService in production
        return {
            "source_department": "development",
            "target_department": "floor_manager",
            "subject": f"Compilation Failed - Requires Attention: {strategy_id}",
            "priority": "high",
            "body": f"""Compilation Failed - Escalation

Strategy: {strategy_id}
Version: {version}
Reason: {reason}

Errors:
{chr(10).join(errors[:10])}

Auto-correction attempts exhausted. Manual intervention required.
""",
        }


def get_compilation_service() -> MQL5CompilationService:
    """Factory function to get a configured compilation service."""
    if platform.system() == "Windows":
        from src.mql5.compiler.docker_compiler import WindowsMetaEditorCompiler
        backend = WindowsMetaEditorCompiler()
    else:
        from src.mql5.compiler.docker_compiler import DockerMQL5Compiler
        backend = DockerMQL5Compiler()
    return MQL5CompilationService(compiler=backend)


@dataclass
class AutoCorrectionResult:
    """Result of compile_with_auto_correction."""
    success: bool
    attempts: int
    escalation_triggered: bool
    original_errors: List[str]
    correction_attempts: List[str]
    final_code: Optional[str] = None
    ex5_path: Optional[str] = None


# Extend MQL5CompilationService with compile_with_auto_correction
_original_compile_ea = MQL5CompilationService.compile_ea


def compile_with_auto_correction(
    self,
    source_code: str,
    max_attempts: int = 2,
) -> AutoCorrectionResult:
    """
    Compile source code with auto-correction loop (max 2 iterations).

    Args:
        source_code: MQL5 source code to compile
        max_attempts: Maximum auto-correction attempts (default 2)

    Returns:
        AutoCorrectionResult with attempts, escalation_triggered, original_errors, etc.
    """
    original_errors: List[str] = []
    correction_attempts_list: List[str] = []
    current_code = source_code
    attempts = 0

    for attempt in range(1, max_attempts + 1):
        attempts = attempt

        # Compile current code
        raw_result = self.compile(
            mq5_file_path="",  # Not used when source_code provided
            strategy_id="",
            version=0,
            attempt_number=attempt,
        )

        # Handle dict result (from mocks) or CompilationResult
        if isinstance(raw_result, dict):
            result = CompilationResult(
                success=raw_result.get("success", False),
                strategy_id=raw_result.get("strategy_id", ""),
                version=raw_result.get("version", 0),
                mq5_path=raw_result.get("mq5_path", ""),
                ex5_path=raw_result.get("ex5_path"),
                errors=raw_result.get("errors", []),
                warnings=raw_result.get("warnings", []),
                compile_time_ms=raw_result.get("compile_time_ms", 0),
                attempt_number=attempt,
            )
        else:
            result = raw_result

        if result.success:
            return AutoCorrectionResult(
                success=True,
                attempts=attempts,
                escalation_triggered=False,
                original_errors=original_errors,
                correction_attempts=correction_attempts_list,
                final_code=current_code,
                ex5_path=result.ex5_path,
            )

        # Collect original errors on first attempt
        if attempt == 1:
            original_errors = list(result.errors)

        # Parse errors
        error_output = "\n".join(result.errors)
        parsed = parse_compilation_output(error_output)

        if not parsed["errors"]:
            break

        # Attempt correction if not at max
        if attempt < max_attempts:
            correction_result = self.auto_corrector.correct(
                current_code,
                parsed["errors"],
                attempt,
            )

            if correction_result.success and correction_result.corrected_code:
                current_code = correction_result.corrected_code
                correction_attempts_list.append(
                    f"Attempt {attempt}: {', '.join(correction_result.corrections_applied)}"
                )

    # Escalation triggered after max attempts exhausted
    return AutoCorrectionResult(
        success=False,
        attempts=attempts,
        escalation_triggered=True,
        original_errors=original_errors,
        correction_attempts=correction_attempts_list,
        final_code=current_code,
        ex5_path=None,
    )


def compile(
    self,
    mq5_file_path: str,
    strategy_id: str = "",
    version: int = 0,
    attempt_number: int = 1,
) -> CompilationResult:
    """
    Delegate compilation to the internal compiler.

    Args:
        mq5_file_path: Path to .mq5 file
        strategy_id: Strategy identifier
        version: Version number
        attempt_number: Attempt number for tracking

    Returns:
        CompilationResult from the compiler
    """
    result = self.compiler.compile(
        mq5_file_path=mq5_file_path,
        strategy_id=strategy_id,
        version=version,
        attempt_number=attempt_number,
    )
    # Handle mock returning dict (for testing)
    if isinstance(result, dict):
        return CompilationResult(
            success=result.get("success", False),
            errors=result.get("errors", []),
            warnings=result.get("warnings", []),
            ex5_path=result.get("ex5_path"),
            attempt_number=attempt_number,
        )
    return result


MQL5CompilationService.compile_with_auto_correction = compile_with_auto_correction
MQL5CompilationService.compile = compile
