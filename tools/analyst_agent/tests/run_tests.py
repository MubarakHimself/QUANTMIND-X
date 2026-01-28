#!/usr/bin/env python3
"""
Test runner for Analyst Agent.

Usage:
    python run_tests.py [options]

Options:
    -v, --verbose    Increase verbosity
    -k NAME          Only run tests matching the given substring
    -m MARKER        Only run tests with the given marker
    --cov=PATH       Measure coverage for module at PATH
    --cov-report=TYPE Report type (default: html)
    --html           Generate HTML coverage report
    --xml            Generate XML coverage report
"""

import sys
import argparse
import subprocess
import os
from pathlib import Path


def run_tests(verbose=False, keyword=None, marker=None, cov_module=None, cov_report="html"):
    """Run the test suite."""
    cmd = [sys.executable, "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    if keyword:
        cmd.extend(["-k", keyword])

    if marker:
        cmd.extend(["-m", marker])

    if cov_module:
        cmd.extend(["--cov", cov_module, "--cov-report", cov_report])

    print(f"Running tests with command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print("=" * 60)
        print("Tests completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        print(e.stdout, file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        print("=" * 60)
        print(f"Tests failed with return code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"Error running tests: {e}", file=sys.stderr)
        return 1


def main():
    """Parse command line arguments and run tests."""
    parser = argparse.ArgumentParser(description="Analyst Agent Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity")
    parser.add_argument("-k", "--keyword", help="Only run tests matching the given substring")
    parser.add_argument("-m", "--marker", help="Only run tests with the given marker")
    parser.add_argument("--cov", help="Measure coverage for module")
    parser.add_argument("--cov-report", default="html", help="Coverage report type (default: html)")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")
    parser.add_argument("--xml", action="store_true", help="Generate XML coverage report")

    args = parser.parse_args()

    # Determine coverage report type
    cov_report = "html"
    if args.xml:
        cov_report = "xml"
    elif args.html:
        cov_report = "html"

    return run_tests(
        verbose=args.verbose,
        keyword=args.keyword,
        marker=args.marker,
        cov_module=args.cov,
        cov_report=cov_report
    )


if __name__ == "__main__":
    sys.exit(main())