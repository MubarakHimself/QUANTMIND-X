"""
Docker-based MQL5 Compiler

Wraps the MetaTrader 5 compiler running in a Docker container on Contabo.
Compilation is triggered via SSH/API from the main application.
"""
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import httpx

logger = logging.getLogger(__name__)


# Configuration from environment
CONTABO_HOST = os.getenv("CONTABO_HOST", "contabo.example.com")
CONTABO_USER = os.getenv("CONTABO_USER", "mql5")
CONTABO_SSH_KEY = os.getenv("CONTABO_SSH_KEY", "/root/.ssh/contabo_mql5")
MT5_COMPILER_DOCKER_IMAGE = os.getenv("MT5_COMPILER_DOCKER_IMAGE", "mql5-compiler:latest")
MT5_COMPILER_API_PORT = int(os.getenv("MT5_COMPILER_API_PORT", "8080"))


@dataclass
class CompilationResult:
    """Result of MQL5 compilation."""
    success: bool
    strategy_id: str
    version: int
    mq5_path: str
    ex5_path: Optional[str]
    errors: List[str]
    warnings: List[str]
    compile_time_ms: int
    attempt_number: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "strategy_id": self.strategy_id,
            "version": self.version,
            "mq5_path": self.mq5_path,
            "ex5_path": self.ex5_path,
            "errors": self.errors,
            "warnings": self.warnings,
            "compile_time_ms": self.compile_time_ms,
            "attempt_number": self.attempt_number,
        }


class DockerMQL5Compiler:
    """
    Docker-based MQL5 compiler wrapper.

    Connects to Contabo server where MetaTrader 5 Docker compiler is running.
    Supports both SSH-based and HTTP API-based compilation.
    """

    def __init__(
        self,
        contabo_host: Optional[str] = None,
        contabo_user: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        docker_image: Optional[str] = None,
        use_api: bool = True,
    ):
        """
        Initialize the compiler wrapper.

        Args:
            contabo_host: Contabo server hostname/IP
            contabo_user: Contabo SSH username
            ssh_key_path: Path to SSH key for authentication
            docker_image: Docker image name for MT5 compiler
            use_api: Use HTTP API instead of SSH
        """
        self.contabo_host = contabo_host or CONTABO_HOST
        self.contabo_user = contabo_user or CONTABO_USER
        self.ssh_key_path = ssh_key_path or CONTABO_SSH_KEY
        self.docker_image = docker_image or MT5_COMPILER_DOCKER_IMAGE
        self.use_api = use_api
        self._api_base_url = f"http://{self.contabo_host}:{MT5_COMPILER_API_PORT}"

    def compile(
        self,
        mq5_file_path: str,
        strategy_id: str,
        version: int,
        attempt_number: int = 1,
    ) -> CompilationResult:
        """
        Compile a .mq5 file to .ex5.

        Args:
            mq5_file_path: Path to the .mq5 source file
            strategy_id: Strategy identifier
            version: Version number
            attempt_number: Current compilation attempt (for auto-correction)

        Returns:
            CompilationResult with success status and output
        """
        import time
        start_time = time.time()

        mq5_path = Path(mq5_file_path)
        if not mq5_path.exists():
            return CompilationResult(
                success=False,
                strategy_id=strategy_id,
                version=version,
                mq5_path=str(mq5_path),
                ex5_path=None,
                errors=[f"Source file not found: {mq5_file_path}"],
                warnings=[],
                compile_time_ms=0,
                attempt_number=attempt_number,
            )

        logger.info(f"Compiling {mq5_path} (attempt {attempt_number})")

        try:
            if self.use_api:
                result = self._compile_via_api(mq5_path, strategy_id, version, attempt_number)
            else:
                result = self._compile_via_ssh(mq5_path, strategy_id, version, attempt_number)

            result.compile_time_ms = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"Compilation failed: {e}")
            return CompilationResult(
                success=False,
                strategy_id=strategy_id,
                version=version,
                mq5_path=str(mq5_path),
                ex5_path=None,
                errors=[f"Compilation exception: {str(e)}"],
                warnings=[],
                compile_time_ms=int((time.time() - start_time) * 1000),
                attempt_number=attempt_number,
            )

    def _compile_via_api(
        self,
        mq5_path: Path,
        strategy_id: str,
        version: int,
        attempt_number: int,
    ) -> CompilationResult:
        """Compile using HTTP API."""
        # Read the MQL5 file content
        with open(mq5_path, 'r') as f:
            mql5_content = f.read()

        # Prepare request
        payload = {
            "source_code": mql5_content,
            "file_name": mq5_path.name,
            "strategy_id": strategy_id,
            "version": version,
        }

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self._api_base_url}/compile",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                return CompilationResult(
                    success=data.get("success", False),
                    strategy_id=strategy_id,
                    version=version,
                    mq5_path=str(mq5_path),
                    ex5_path=data.get("ex5_path"),
                    errors=data.get("errors", []),
                    warnings=data.get("warnings", []),
                    compile_time_ms=0,  # Will be set by caller
                    attempt_number=attempt_number,
                )

        except httpx.HTTPError as e:
            logger.warning(f"API request failed: {e}, falling back to SSH")
            return self._compile_via_ssh(mq5_path, strategy_id, version, attempt_number)

    def _compile_via_ssh(
        self,
        mq5_path: Path,
        strategy_id: str,
        version: int,
        attempt_number: int,
    ) -> CompilationResult:
        """Compile using SSH-based Docker command execution."""
        # Build the SSH command
        # The remote script will compile the MQL5 file in the Docker container
        remote_cmd = (
            f"docker run --rm -v /mnt/mql5/Experts:/mnt "
            f"{self.docker_image} /opt/mt5-compiler/compile.sh {mq5_path.name}"
        )

        ssh_cmd = [
            "ssh",
            "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            f"{self.contabo_user}@{self.contabo_host}",
            remote_cmd,
        ]

        # Copy file to remote first
        scp_cmd = [
            "scp",
            "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=no",
            str(mq5_path),
            f"{self.contabo_user}@{self.contabo_host}:/mnt/mql5/Experts/{mq5_path.name}",
        ]

        try:
            # Copy file to Contabo
            result = subprocess.run(
                scp_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.error(f"SCP failed: {result.stderr}")
                return CompilationResult(
                    success=False,
                    strategy_id=strategy_id,
                    version=version,
                    mq5_path=str(mq5_path),
                    ex5_path=None,
                    errors=[f"Failed to copy file to Contabo: {result.stderr}"],
                    warnings=[],
                    compile_time_ms=0,
                    attempt_number=attempt_number,
                )

            # Run compilation
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Parse output
            output = result.stdout + result.stderr

            if result.returncode == 0 and ".ex5" in output:
                # Extract .ex5 path from output
                ex5_path = self._extract_ex5_path(output, mq5_path)
                return CompilationResult(
                    success=True,
                    strategy_id=strategy_id,
                    version=version,
                    mq5_path=str(mq5_path),
                    ex5_path=ex5_path,
                    errors=[],
                    warnings=self._extract_warnings(output),
                    compile_time_ms=0,
                    attempt_number=attempt_number,
                )
            else:
                return CompilationResult(
                    success=False,
                    strategy_id=strategy_id,
                    version=version,
                    mq5_path=str(mq5_path),
                    ex5_path=None,
                    errors=self._extract_errors(output),
                    warnings=self._extract_warnings(output),
                    compile_time_ms=0,
                    attempt_number=attempt_number,
                )

        except subprocess.TimeoutExpired:
            return CompilationResult(
                success=False,
                strategy_id=strategy_id,
                version=version,
                mq5_path=str(mq5_path),
                ex5_path=None,
                errors=["Compilation timed out after 120 seconds"],
                warnings=[],
                compile_time_ms=0,
                attempt_number=attempt_number,
            )
        except Exception as e:
            return CompilationResult(
                success=False,
                strategy_id=strategy_id,
                version=version,
                mq5_path=str(mq5_path),
                ex5_path=None,
                errors=[f"SSH compilation failed: {str(e)}"],
                warnings=[],
                compile_time_ms=0,
                attempt_number=attempt_number,
            )

    def _extract_ex5_path(self, output: str, mq5_path: Path) -> Optional[str]:
        """Extract .ex5 file path from compilation output."""
        for line in output.split("\n"):
            if ".ex5" in line and "compiled" in line.lower():
                # Try to extract path
                parts = line.split()
                for part in parts:
                    if ".ex5" in part:
                        return part.strip()
        # Default path based on input
        return str(mq5_path.with_suffix(".ex5"))

    def _extract_errors(self, output: str) -> List[str]:
        """Extract error messages from compilation output."""
        errors = []
        in_error_section = False

        for line in output.split("\n"):
            if "error" in line.lower() or "failed" in line.lower():
                in_error_section = True

            if in_error_section and line.strip():
                errors.append(line.strip())

            # End of error section
            if in_error_section and "warning" in line.lower():
                in_error_section = False

        return errors[:50]  # Limit to 50 errors

    def _extract_warnings(self, output: str) -> List[str]:
        """Extract warning messages from compilation output."""
        warnings = []
        in_warning_section = False

        for line in output.split("\n"):
            if "warning" in line.lower():
                in_warning_section = True

            if in_warning_section and line.strip() and "warning" in line.lower():
                warnings.append(line.strip())

        return warnings[:50]  # Limit to 50 warnings


def get_compiler() -> DockerMQL5Compiler:
    """Get a configured compiler instance."""
    return DockerMQL5Compiler()
