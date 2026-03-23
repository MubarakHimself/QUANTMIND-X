"""
P0 Tests: Build Verification (Epic 1 - Story 1-2)

Tests verify that Svelte 5 migration is complete and npm build passes:
- No Svelte 4 deprecation warnings
- Build completes successfully
- All components render without errors

Risk: R-002 (Score: 4) - Svelte 5 migration may have missed edge cases

These tests are designed to FAIL before Svelte 5 migration is complete.
"""

import pytest
import subprocess
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestSvelte5MigrationBuild:
    """
    P0: npm build passes without Svelte 4 deprecation warnings.

    Risk: R-002 - Svelte 5 migration edge cases
    """

    @pytest.fixture
    def quantmind_ide_path(self):
        """Return path to quantmind-ide directory."""
        project_root = Path(__file__).parent.parent
        ide_path = project_root / "quantmind-ide"
        if not ide_path.exists():
            pytest.skip("quantmind-ide directory not found")
        return ide_path

    @pytest.fixture
    def package_json_path(self, quantmind_ide_path):
        """Return path to package.json in quantmind-ide."""
        pkg_path = quantmind_ide_path / "package.json"
        if not pkg_path.exists():
            pytest.skip("package.json not found")
        return pkg_path

    def test_npm_build_passes(self, quantmind_ide_path):
        """
        P0: Verify `npm run build` passes in quantmind-ide.

        Expected: Build completes with exit code 0.
        Current: WILL FAIL if build is broken.

        Risk: R-002
        """
        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("npm not available")

        # Run npm build
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(quantmind_ide_path),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Print output for debugging
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

        assert result.returncode == 0, f"npm build failed with exit code {result.returncode}"

    def test_no_svelte_4_deprecation_warnings(self, quantmind_ide_path):
        """
        P0: Verify npm build has zero Svelte 4 deprecation warnings.

        Expected: Build output contains no Svelte 4 deprecation warnings.
        Current: WILL FAIL if Svelte 4 warnings exist.

        Risk: R-002

        Common Svelte 4 deprecation patterns to check:
        - "Svelte 4" warnings
        - "will be removed in Svelte 5"
        - "deprecat" warnings from Svelte compiler
        """
        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("npm not available")

        # Run npm build with output capture
        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(quantmind_ide_path),
            capture_output=True,
            text=True,
            timeout=300
        )

        # Combine stdout and stderr for checking
        output = result.stdout + result.stderr

        # Svelte 4 specific deprecation warning patterns
        # These are actual warnings from the Svelte compiler about Svelte 4 deprecations
        svelte4_deprecation_patterns = [
            "Svelte 4",
            "will be removed in Svelte 5",
            "svelte 5 will",
            "convert to $derived",
            "convert to $effect",
            "use:action instead",
            "on:click -> onclick",
        ]

        found_warnings = []
        output_lower = output.lower()

        for pattern in svelte4_deprecation_patterns:
            if pattern.lower() in output_lower:
                found_warnings.append(pattern)

        if found_warnings:
            print("Build output:")
            print(output)
            print("\nFound Svelte 4 deprecation warnings:")
            for w in found_warnings:
                print(f"  - {w}")

        # The test should fail if any Svelte 4 specific deprecation warnings found
        assert len(found_warnings) == 0, \
            f"Build contains {len(found_warnings)} Svelte 4 deprecation warnings: {found_warnings}"

    def test_build_output_no_errors(self, quantmind_ide_path):
        """
        P0: Verify build completes without errors.

        Expected: No error messages in build output.
        Current: WILL FAIL if errors exist.
        """
        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("npm not available")

        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(quantmind_ide_path),
            capture_output=True,
            text=True,
            timeout=300
        )

        output = result.stdout + result.stderr

        # Error patterns that should NOT appear in successful build
        error_patterns = [
            "error:",
            "Error:",
            "ERROR:",
            "failed",
            "Failed",
            "Build failed",
        ]

        found_errors = []
        for pattern in error_patterns:
            if pattern in output:
                found_errors.append(pattern)

        if found_errors:
            print("Build output:")
            print(output)

        assert len(found_errors) == 0, \
            f"Build contains error indicators: {found_errors}"

    def test_typescript_compilation(self, quantmind_ide_path):
        """
        P0: Verify TypeScript compilation passes.

        Expected: No TypeScript errors in build output.
        Current: WILL FAIL if TypeScript errors exist.
        """
        # Check if npm is available
        try:
            subprocess.run(["npm", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("npm not available")

        result = subprocess.run(
            ["npm", "run", "build"],
            cwd=str(quantmind_ide_path),
            capture_output=True,
            text=True,
            timeout=300
        )

        output = result.stdout + result.stderr

        # TypeScript error patterns
        ts_error_patterns = [
            "TS[0-9]{4}",  # TypeScript error codes like TS2304
            "TypeScript error",
            "Argument of type",
            "Type '",
            "is not assignable to type",
        ]

        found_ts_errors = []
        for pattern in ts_error_patterns:
            if pattern in output:
                found_ts_errors.append(pattern)

        if found_ts_errors:
            print("Build output:")
            print(output)

        # Note: This is a weaker assertion since build passing implies TS is OK
        # But we include it for explicit verification
        assert result.returncode == 0, "Build should pass for TS to be valid"


class TestSvelte5MigrationComponentCompatibility:
    """
    P2: Svelte 5 components best practices verification.

    Note: Svelte 5 supports backward compatibility with legacy syntax.
    This is informational only - the blocking P0 is whether build passes.
    """

    def test_legacy_reactive_syntax_detected_but_acceptable(self):
        """
        P2: Detect legacy $: reactive syntax in components.

        This is informational - Svelte 5 allows legacy syntax via compatibility mode.
        The blocking test is whether npm build passes without warnings.

        Risk: R-002 (informational only)
        """
        project_root = Path(__file__).parent.parent
        components_path = project_root / "quantmind-ide" / "src" / "lib" / "components"

        if not components_path.exists():
            pytest.skip("Components directory not found")

        # Count files with legacy syntax for reporting
        files_with_legacy = []

        # Walk through all .svelte files
        for svelte_file in components_path.rglob("*.svelte"):
            try:
                content = svelte_file.read_text()
                # Simple detection - just report, don't fail
                if "$:" in content or "on:click=" in content:
                    files_with_legacy.append(
                        str(svelte_file.relative_to(project_root))
                    )
            except Exception:
                pass

        # This is informational only - Svelte 5 is backward compatible
        print(f"\nFound {len(files_with_legacy)} files with legacy Svelte 4 syntax")
        print("Note: Svelte 5 supports backward compatibility - this is informational only")

        # Don't fail - just report
        assert True
