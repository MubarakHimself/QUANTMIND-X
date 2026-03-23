"""
Tests for MQL5 Compilation Service

Story 7.3: MQL5 Compilation Integration
"""
import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestDockerMQL5Compiler:
    """Test Docker-based MQL5 compiler."""

    @pytest.fixture
    def compiler(self):
        """Create compiler instance for testing."""
        from src.mql5.compiler.docker_compiler import DockerMQL5Compiler

        with patch("src.mql5.compiler.docker_compiler.CONTABO_HOST", "localhost"):
            with patch("src.mql5.compiler.docker_compiler.MT5_COMPILER_API_PORT", 8080):
                yield DockerMQL5Compiler(use_api=False)

    @pytest.fixture
    def sample_mq5_content(self):
        """Sample MQL5 content for testing."""
        return '''//+------------------------------------------------------------------+
//| Sample EA                                                            |
//+------------------------------------------------------------------+
#property version="5.00"
#property strict

input int MA_Period = 14;

int OnInit()
{
   Print("EA Initialized");
   return INIT_SUCCEEDED;
}

void OnTick()
{
}

void OnDeinit(const int reason)
{
}
'''

    def test_compiler_initialization(self):
        """Test compiler initializes with correct defaults."""
        from src.mql5.compiler.docker_compiler import DockerMQL5Compiler

        compiler = DockerMQL5Compiler()
        assert compiler.contabo_host == "contabo.example.com"
        assert compiler.docker_image == "mql5-compiler:latest"

    def test_compiler_custom_config(self):
        """Test compiler with custom configuration."""
        from src.mql5.compiler.docker_compiler import DockerMQL5Compiler

        compiler = DockerMQL5Compiler(
            contabo_host="custom.example.com",
            ssh_key_path="/custom/key",
            docker_image="custom-image:latest",
        )
        assert compiler.contabo_host == "custom.example.com"
        assert compiler.ssh_key_path == "/custom/key"
        assert compiler.docker_image == "custom-image:latest"

    def test_compile_nonexistent_file(self, compiler):
        """Test compilation fails for nonexistent file."""
        result = compiler.compile(
            mq5_file_path="/nonexistent/file.mq5",
            strategy_id="test_strategy",
            version=1,
        )

        assert result.success is False
        assert "not found" in result.errors[0].lower()
        assert result.strategy_id == "test_strategy"
        assert result.version == 1

    @patch("src.mql5.compiler.docker_compiler.subprocess.run")
    def test_compile_via_ssh_success(self, mock_run, tmp_path):
        """Test SSH-based compilation success."""
        from src.mql5.compiler.docker_compiler import DockerMQL5Compiler

        # Create compiler with contabo_user set (required for SSH)
        compiler = DockerMQL5Compiler(
            contabo_host="localhost",
            contabo_user="testuser",
            ssh_key_path="/tmp/test_key",
            use_api=False,
        )

        # Create temp MQL5 file
        mq5_file = tmp_path / "test.mq5"
        mq5_file.write_text("// Test EA")

        # Mock successful compilation output
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Compiled successfully\\n/path/to/test.ex5",
            stderr="",
        )

        result = compiler.compile(
            mq5_file_path=str(mq5_file),
            strategy_id="test_strategy",
            version=1,
        )

        assert result.success is True

    @patch("src.mql5.compiler.docker_compiler.subprocess.run")
    def test_compile_via_ssh_failure(self, mock_run, compiler, tmp_path):
        """Test SSH-based compilation failure."""
        # Create temp MQL5 file
        mq5_file = tmp_path / "test.mq5"
        mq5_file.write_text("// Test EA with syntax error")

        # Mock failed compilation
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="error: ';' expected at line 10",
        )

        result = compiler.compile(
            mq5_file_path=str(mq5_file),
            strategy_id="test_strategy",
            version=1,
        )

        assert result.success is False
        assert len(result.errors) > 0

    @patch("httpx.Client")
    def test_compile_via_api_success(self, mock_client_class, tmp_path):
        """Test API-based compilation success."""
        from src.mql5.compiler.docker_compiler import DockerMQL5Compiler

        # Create temp MQL5 file
        mq5_file = tmp_path / "test.mq5"
        mq5_file.write_text("// Test EA")

        # Mock API response
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "ex5_path": "/path/to/test.ex5",
            "errors": [],
            "warnings": [],
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=False)
        mock_client_class.return_value = mock_client

        compiler = DockerMQL5Compiler(use_api=True)
        result = compiler.compile(
            mq5_file_path=str(mq5_file),
            strategy_id="test_strategy",
            version=1,
        )

        assert result.success is True


class TestMQL5ErrorParser:
    """Test MQL5 error parser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        from src.mql5.compiler.error_parser import MQL5ErrorParser
        return MQL5ErrorParser()

    def test_parse_syntax_error(self, parser):
        """Test parsing syntax error."""
        output = "10(5): error: ';' expected"
        result = parser.parse(output)

        assert len(result["errors"]) == 1
        error = result["errors"][0]
        assert error.line == 10
        assert error.column == 5
        assert error.severity == "error"

    def test_parse_warning(self, parser):
        """Test parsing warning."""
        output = "15(3): warning: implicit conversion"
        result = parser.parse(output)

        assert len(result["warnings"]) == 1
        warning = result["warnings"][0]
        assert warning.line == 15
        assert warning.severity == "warning"

    def test_parse_deprecated(self, parser):
        """Test parsing deprecated warning."""
        output = "20(1): deprecated: old function used"
        result = parser.parse(output)

        assert len(result["warnings"]) == 1
        assert result["warnings"][0].severity == "deprecated"

    def test_parse_multiple_errors(self, parser):
        """Test parsing multiple errors."""
        output = """10(5): error: ';' expected
15(10): error: unknown identifier 'X'
20(1): warning: implicit conversion"""

        result = parser.parse(output)

        assert len(result["errors"]) == 2
        assert len(result["warnings"]) == 1

    def test_has_blocking_errors(self, parser):
        """Test blocking error detection."""
        output = "10(5): error: ';' expected"
        parser.parse(output)
        assert parser.has_blocking_errors() is True

    def test_has_warnings_only(self, parser):
        """Test warnings-only detection."""
        output = "10(5): warning: implicit conversion"
        parser.parse(output)
        assert parser.has_warnings_only() is True


class TestAutoCorrector:
    """Test auto-correction logic."""

    @pytest.fixture
    def corrector(self):
        """Create auto-corrector instance."""
        from src.mql5.compiler.autocorrect import AutoCorrector
        return AutoCorrector(max_attempts=2)

    @pytest.fixture
    def sample_errors(self):
        """Create sample compilation errors."""
        from src.mql5.compiler.error_parser import CompilationError
        return [
            CompilationError(line=10, column=5, error_code="MQL5_ERROR",
                           message="';' expected", severity="error"),
        ]

    def test_corrector_initialization(self, corrector):
        """Test corrector initializes correctly."""
        assert corrector.max_attempts == 2

    def test_auto_correct_semicolon(self, corrector):
        """Test auto-correction of missing semicolon."""
        source = """void OnTick()
{
   int x = 5
}"""
        errors = [
            Mock(line=3, column=15, message="';' expected", severity="error",
                 is_auto_correctable=Mock(return_value=True),
                 get_correction_type=Mock(return_value="semicolon"))
        ]

        result = corrector.correct(source, errors, 1)

        assert result.success is True
        assert ";" in result.corrected_code

    def test_auto_correct_parenthesis(self, corrector):
        """Test auto-correction of missing parenthesis."""
        source = """void OnTick()
{
   Print("test
}"""
        errors = [
            Mock(line=3, column=10, message="')' expected", severity="error",
                 is_auto_correctable=Mock(return_value=True),
                 get_correction_type=Mock(return_value="parenthesis"))
        ]

        result = corrector.correct(source, errors, 1)

        assert result.success is True
        assert ")" in result.corrected_code

    def test_max_attempts_respected(self, corrector):
        """Test max attempts limit is respected."""
        source = "int x = 5"
        errors = [Mock(line=1, column=1, message="some error", severity="error",
                      is_auto_correctable=Mock(return_value=False),
                      get_correction_type=Mock(return_value=None))]

        # Try to correct beyond max attempts
        result = corrector.correct(source, errors, 3)

        assert result.success is False


class TestCompilationService:
    """Test MQL5 compilation service."""

    @pytest.fixture
    def mock_compiler(self):
        """Create mock compiler."""
        mock = Mock()
        return mock

    @pytest.fixture
    def mock_ea_storage(self):
        """Create mock EA storage."""
        from src.strategy.output import EAOutput
        from datetime import datetime

        mock = Mock()
        mock.get_ea.return_value = EAOutput(
            strategy_id="test_strategy",
            strategy_name="Test Strategy",
            version=1,
            file_path="/tmp/test_strategy_v1.mq5",
            generated_at=datetime.now(),
            trd_snapshot={},
            status="generated",
        )
        return mock

    def test_compilation_service_initialization(self):
        """Test compilation service initializes correctly."""
        from src.mql5.compiler.service import MQL5CompilationService

        service = MQL5CompilationService()
        assert service.compiler is not None
        assert service.ea_storage is not None
        assert service.error_parser is not None

    def test_compilation_service_result_dataclass(self):
        """Test CompilationServiceResult dataclass."""
        from src.mql5.compiler.service import CompilationServiceResult

        result = CompilationServiceResult(
            success=True,
            strategy_id="test",
            version=1,
            mq5_path="/path/test.mq5",
            ex5_path="/path/test.ex5",
            compile_status="success",
            errors=[],
            warnings=[],
            auto_correction_attempts=1,
            escalated_to_floor_manager=False,
        )

        assert result.success is True
        assert result.compile_status == "success"

    def test_should_escalate_too_many_attempts(self):
        """Test escalation after max attempts."""
        from src.mql5.compiler.service import CompilationServiceResult, MQL5CompilationService

        service = MQL5CompilationService()

        result = CompilationServiceResult(
            success=False,
            strategy_id="test",
            version=1,
            mq5_path="/path/test.mq5",
            ex5_path=None,
            compile_status="failed",
            errors=["error1"],
            warnings=[],
            auto_correction_attempts=2,
            escalated_to_floor_manager=False,
        )

        should_escalate, reason = service._should_escalate(result)
        assert should_escalate is True
        assert "exhausted" in reason

    def test_should_escalate_too_many_errors(self):
        """Test escalation with too many errors."""
        from src.mql5.compiler.service import CompilationServiceResult, MQL5CompilationService

        service = MQL5CompilationService()

        result = CompilationServiceResult(
            success=False,
            strategy_id="test",
            version=1,
            mq5_path="/path/test.mq5",
            ex5_path=None,
            compile_status="failed",
            errors=["error"] * 15,
            warnings=[],
            auto_correction_attempts=1,
            escalated_to_floor_manager=False,
        )

        should_escalate, reason = service._should_escalate(result)
        assert should_escalate is True
        assert "too many" in reason.lower()

    def test_should_escalate_blocking_errors(self):
        """Test escalation with blocking errors."""
        from src.mql5.compiler.service import CompilationServiceResult, MQL5CompilationService

        service = MQL5CompilationService()

        result = CompilationServiceResult(
            success=False,
            strategy_id="test",
            version=1,
            mq5_path="/path/test.mq5",
            ex5_path=None,
            compile_status="failed",
            errors=["undefined reference to 'Function'"],
            warnings=[],
            auto_correction_attempts=1,
            escalated_to_floor_manager=False,
        )

        should_escalate, reason = service._should_escalate(result)
        assert should_escalate is True
        assert "undefined" in reason

    def test_escalate_to_floor_manager(self):
        """Test escalation message format."""
        from src.mql5.compiler.service import MQL5CompilationService

        service = MQL5CompilationService()
        escalation = service.escalate_to_floor_manager(
            strategy_id="test_strategy",
            version=1,
            reason="Auto-correction exhausted",
            errors=["error1", "error2"],
        )

        assert escalation["source_department"] == "development"
        assert escalation["target_department"] == "floor_manager"
        assert escalation["priority"] == "high"
        assert "test_strategy" in escalation["subject"]


class TestCompileEndpoints:
    """Test compilation API endpoints."""

    def test_compile_request_model(self):
        """Test CompileRequest model."""
        from src.api.compile_endpoints import CompileRequest

        request = CompileRequest(strategy_id="test_strategy", version=1)
        assert request.strategy_id == "test_strategy"
        assert request.version == 1

    def test_compile_request_optional_version(self):
        """Test CompileRequest with optional version."""
        from src.api.compile_endpoints import CompileRequest

        request = CompileRequest(strategy_id="test_strategy")
        assert request.strategy_id == "test_strategy"
        assert request.version is None

    def test_compile_response_model(self):
        """Test CompileResponse model."""
        from src.api.compile_endpoints import CompileResponse

        response = CompileResponse(
            success=True,
            strategy_id="test_strategy",
            version=1,
            compile_status="success",
            mq5_path="/path/test.mq5",
            ex5_path="/path/test.ex5",
            errors=[],
            warnings=[],
        )

        assert response.success is True
        assert response.compile_status == "success"

    def test_compilation_status_response_model(self):
        """Test CompilationStatusResponse model."""
        from src.api.compile_endpoints import CompilationStatusResponse

        response = CompilationStatusResponse(
            strategy_id="test_strategy",
            version=1,
            compile_status="pending",
            mq5_path="/path/test.mq5",
        )

        assert response.compile_status == "pending"


class TestEAOutputStorageCompile:
    """Test EA output storage with compilation fields."""

    @pytest.fixture
    def storage(self):
        """Create storage instance with temp directory."""
        from src.strategy.output import EAOutputStorage

        with tempfile.TemporaryDirectory() as tmpdir:
            yield EAOutputStorage(data_dir=tmpdir)

    def test_save_ea_with_compile_status(self, storage):
        """Test saving EA initializes compile status."""
        ea = storage.save_ea(
            strategy_id="test_compile",
            strategy_name="Test Strategy",
            mql5_code="// Test",
            trd_snapshot={},
        )

        # Update compile status
        result = storage.update_compile_status(
            strategy_id=ea.strategy_id,
            version=ea.version,
            compile_status="pending",
        )

        assert result is True

        # Verify
        retrieved = storage.get_ea(ea.strategy_id, ea.version)
        assert retrieved.compile_status == "pending"

    def test_update_compile_status_success(self, storage):
        """Test updating compile status to success."""
        ea = storage.save_ea(
            strategy_id="test_compile_success",
            strategy_name="Test Strategy",
            mql5_code="// Test",
            trd_snapshot={},
        )

        result = storage.update_compile_status(
            strategy_id=ea.strategy_id,
            version=ea.version,
            compile_status="success",
            compile_errors=None,
        )

        assert result is True

        retrieved = storage.get_ea(ea.strategy_id, ea.version)
        assert retrieved.compile_status == "success"

    def test_update_compile_status_with_errors(self, storage):
        """Test updating compile status with errors."""
        ea = storage.save_ea(
            strategy_id="test_compile_fail",
            strategy_name="Test Strategy",
            mql5_code="// Test",
            trd_snapshot={},
        )

        result = storage.update_compile_status(
            strategy_id=ea.strategy_id,
            version=ea.version,
            compile_status="failed",
            compile_errors=["error1", "error2"],
        )

        assert result is True

        retrieved = storage.get_ea(ea.strategy_id, ea.version)
        assert retrieved.compile_status == "failed"
        assert len(retrieved.compile_errors) == 2


class TestCompileStatusConstants:
    """Test compile status constants."""

    def test_status_constants(self):
        """Test compile status constants are defined."""
        from src.mql5.compiler.service import (
            COMPILE_STATUS_PENDING,
            COMPILE_STATUS_SUCCESS,
            COMPILE_STATUS_FAILED,
        )

        assert COMPILE_STATUS_PENDING == "pending"
        assert COMPILE_STATUS_SUCCESS == "success"
        assert COMPILE_STATUS_FAILED == "failed"
