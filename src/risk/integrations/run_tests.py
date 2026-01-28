#!/usr/bin/env python3
"""
Run tests for MT5 integration.

Usage:
    python run_tests.py          # Run all tests
    python run_tests.py -v       # Verbose output
"""

import sys
import unittest
import logging

def main():
    """Run MT5 integration tests."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('src/risk/integrations', pattern='test_*.py')

    test_runner = unittest.TextTestRunner(verbosity=2 if '-v' in sys.argv else 1)
    result = test_runner.run(test_suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == "__main__":
    main()