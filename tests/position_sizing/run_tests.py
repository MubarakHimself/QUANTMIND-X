#!/usr/bin/env python3
"""
Enhanced Kelly Position Sizing - Test Runner

Runs all tests for the Enhanced Kelly system with detailed reporting.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --kelly            # Run Kelly tests only
    python run_tests.py --performance      # Run performance benchmarks
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --verbose          # Verbose output
"""

import sys
import subprocess
from pathlib import Path
from typing import List


def run_command(cmd: List[str], description: str) -> bool:
    """
    Run a command and report results.

    Args:
        cmd: Command to run
        description: Description of command

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print(f"\n✓ {description} - PASSED")
        return True
    else:
        print(f"\n✗ {description} - FAILED")
        return False


def main():
    """Main test runner."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    test_dir = Path(__file__).parent

    # Parse arguments
    args = sys.argv[1:]

    # Base pytest command
    pytest_cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes"
    ]

    # Add markers based on arguments
    if "--kelly" in args:
        pytest_cmd.extend(["-m", "kelly"])
    elif "--analyzer" in args:
        pytest_cmd.extend(["-m", "analyzer"])
    elif "--portfolio" in args:
        pytest_cmd.extend(["-m", "portfolio"])
    elif "--edge-case" in args:
        pytest_cmd.extend(["-m", "edge_case"])
    elif "--performance" in args:
        pytest_cmd.extend(["-m", "performance"])
    elif "--integration" in args:
        pytest_cmd.extend(["-m", "integration"])

    # Add coverage
    if "--coverage" in args:
        pytest_cmd.extend([
            "--cov=src/position_sizing",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Add verbose
    if "--verbose" in args or "-s" in args:
        pytest_cmd.append("-s")

    # Run tests
    success = run_command(pytest_cmd, "Enhanced Kelly Test Suite")

    if success:
        print(f"\n{'='*60}")
        print("All tests PASSED!")
        print(f"{'='*60}\n")

        # Show summary
        print("Test Summary:")
        print("  - Kelly Calculator: Core position sizing logic")
        print("  - Kelly Analyzer: Statistics and parameter extraction")
        print("  - Portfolio Scaler: Multi-bot risk management")
        print("  - Edge Cases: Exception handling and safety")
        print("  - Performance: Latency and memory benchmarks")
        print("  - Integration: End-to-end workflows")

        print("\nTest Coverage:")
        print("  - Unit tests: Individual components")
        print("  - Integration tests: Component interactions")
        print("  - Edge case tests: Boundary conditions")
        print("  - Performance tests: Latency targets")

        return 0
    else:
        print(f"\n{'='*60}")
        print("Some tests FAILED!")
        print(f"{'='*60}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
