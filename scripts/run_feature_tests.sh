#!/usr/bin/env bash
#
# Feature-Specific Test Runner for Task Group 9
#
# This script runs ONLY the tests related to the QuantMindX Trading System
# spec features, NOT the entire application test suite.
#
# Usage:
#   ./scripts/run_feature_tests.sh              # Run all feature tests
#   ./scripts/run_feature_tests.sh e2e          # Run only E2E tests
#   ./scripts/run_feature_tests.sh integration  # Run only integration tests
#   ./scripts/run_feature_tests.sh performance  # Run only performance benchmarks
#   ./scripts/run_feature_tests.sh --report     # Generate coverage report
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}QuantMindX Feature Test Runner${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""

# Default test paths
E2E_TESTS="tests/e2e/test_task_group_9_integration.py"
INTEGRATION_TESTS="tests/integration/test_v8_complete_workflow.py"
PERFORMANCE_TESTS="tests/benchmarks/test_*.py"

# Parse command line arguments
TEST_TYPE="${1:-all}"
GENERATE_REPORT="${2:-}"

# Determine which tests to run
case "$TEST_TYPE" in
    e2e)
        echo -e "${YELLOW}Running E2E tests only...${NC}"
        TEST_PATHS="$E2E_TESTS"
        MARKER="e2e"
        ;;
    integration)
        echo -e "${YELLOW}Running integration tests only...${NC}"
        TEST_PATHS="$INTEGRATION_TESTS"
        MARKER="integration"
        ;;
    performance)
        echo -e "${YELLOW}Running performance benchmarks only...${NC}"
        TEST_PATHS="$PERFORMANCE_TESTS"
        MARKER="benchmark"
        ;;
    all)
        echo -e "${YELLOW}Running all feature-specific tests...${NC}"
        TEST_PATHS="$E2E_TESTS $INTEGRATION_TESTS $PERFORMANCE_TESTS"
        MARKER="e2e or integration or benchmark"
        ;;
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Usage: $0 [e2e|integration|performance|all] [--report]"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}Test Configuration:${NC}"
echo "  Type: $TEST_TYPE"
echo "  Project Root: $PROJECT_ROOT"
echo ""

# Build pytest command
PYTEST_CMD="pytest -v"

# Add markers
if [ -n "$MARKER" ]; then
    PYTEST_CMD="$PYTEST_CMD -m '$MARKER'"
fi

# Add test paths
PYTEST_CMD="$PYTEST_CMD $TEST_PATHS"

# Add coverage if requested
if [ "$GENERATE_REPORT" == "--report" ]; then
    echo -e "${YELLOW}Generating coverage report...${NC}"
    PYTEST_CMD="$PYTEST_CMD --cov=src/backtesting --cov=src/router --cov=src/position_sizing --cov=src/data --cov-report=html --cov-report=term"
fi

# Add timeout marker for E2E tests
if [[ "$TEST_TYPE" == "e2e" || "$TEST_TYPE" == "all" ]]; then
    PYTEST_CMD="$PYTEST_CMD --timeout=120"
fi

# Show command
echo -e "${BLUE}Running command:${NC}"
echo "  $PYTEST_CMD"
echo ""
echo -e "${BLUE}====================================${NC}"
echo ""

# Run tests
eval $PYTEST_CMD
TEST_EXIT_CODE=$?

echo ""
echo -e "${BLUE}====================================${NC}"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All tests PASSED${NC}"
else
    echo -e "${RED}✗ Some tests FAILED${NC}"
fi

echo ""

# Show coverage report if generated
if [ "$GENERATE_REPORT" == "--report" ] && [ -f "htmlcov/index.html" ]; then
    echo -e "${BLUE}Coverage report generated:${NC}"
    echo "  HTML: file://$PROJECT_ROOT/htmlcov/index.html"
    echo ""
fi

# Show test count summary
echo -e "${BLUE}Test Summary:${NC}"
TEST_COUNT=$(pytest --collect-only -q $TEST_PATHS 2>/dev/null | grep "test session starts" -A 100 | grep "collected" || echo "N/A")
echo "  $TEST_COUNT"
echo ""

exit $TEST_EXIT_CODE
