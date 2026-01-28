# Enhanced Kelly Position Sizing - Test Suite

Comprehensive test suite for the Enhanced Kelly Position Sizing system.

## Test Structure

```
tests/position_sizing/
├── __init__.py                 # Package initialization
├── conftest.py                 # Pytest fixtures and configuration
├── test_enhanced_kelly.py      # Kelly calculator tests
├── test_kelly_analyzer.py      # Statistics analyzer tests
├── test_portfolio_kelly.py     # Portfolio scaler tests
├── test_performance.py         # Performance benchmarks
├── test_edge_cases.py          # Edge case handling tests (future)
└── run_tests.py               # Test runner script
```

## Running Tests

### Run All Tests

```bash
# Using pytest directly
pytest tests/position_sizing/ -v

# Using the test runner
python tests/position_sizing/run_tests.py
```

### Run Specific Test Categories

```bash
# Kelly calculator tests only
pytest tests/position_sizing/test_enhanced_kelly.py -v

# Kelly analyzer tests only
pytest tests/position_sizing/test_kelly_analyzer.py -v

# Portfolio scaler tests only
pytest tests/position_sizing/test_portfolio_kelly.py -v

# Performance benchmarks only
pytest tests/position_sizing/test_performance.py -v
```

### Run by Markers

```bash
# Kelly calculation tests
pytest tests/position_sizing/ -m kelly -v

# Statistics analyzer tests
pytest tests/position_sizing/ -m analyzer -v

# Portfolio tests
pytest tests/position_sizing/ -m portfolio -v

# Edge case tests
pytest tests/position_sizing/ -m edge_case -v

# Performance tests
pytest tests/position_sizing/ -m performance -v

# Integration tests
pytest tests/position_sizing/ -m integration -v
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/position_sizing/ \
    --cov=src/position_sizing \
    --cov-report=html \
    --cov-report=term-missing

# Open HTML report
open htmlcov/index.html
```

### Run with Verbose Output

```bash
# Show all print statements
pytest tests/position_sizing/ -v -s

# Show local variables on failure
pytest tests/position_sizing/ -v --tb=long
```

## Test Categories

### 1. Kelly Calculator Tests (`test_enhanced_kelly.py`)

**Test Classes**:
- `TestEnhancedKellyCalculator`: Core calculation logic
- `TestVolatilityAdjustment`: ATR-based adjustments
- `TestPositionSizeCalculation`: Lot calculation and rounding
- `TestEdgeCases`: Input validation and edge cases
- `TestPropFirmPresets`: Prop firm configurations
- `TestConvenienceFunction`: Quick API function
- `TestIntegrationScenarios`: Real-world scenarios

**Coverage**:
- Kelly formula calculation
- 3-layer protection system
- Volatility adjustments
- Broker constraints
- Prop firm presets

**Test Count**: ~40 tests

### 2. Kelly Analyzer Tests (`test_kelly_analyzer.py`)

**Test Classes**:
- `TestKellyParameterExtraction`: Parameter calculation
- `TestDataReliability`: Sample size assessment
- `TestAlternativeTradeFormats`: Different data formats
- `TestRollingWindowAnalysis`: Rolling window calculations
- `TestAnalyzerEdgeCases`: Edge case handling
- `TestAnalyzerIntegration`: Integration scenarios

**Coverage**:
- Win rate calculation
- Risk-reward ratio
- Base Kelly formula
- Expectancy and profit factor
- Rolling window analysis
- Edge decay detection

**Test Count**: ~30 tests

### 3. Portfolio Scaler Tests (`test_portfolio_kelly.py`)

**Test Classes**:
- `TestPortfolioScaling`: Multi-bot position scaling
- `TestPortfolioStatus`: Portfolio risk reporting
- `TestRiskAllocation`: Equal and performance-based allocation
- `TestBotLimitRecommendation`: Bot count recommendations
- `TestPortfolioEdgeCases`: Correlation and edge cases
- `TestPortfolioIntegration`: Realistic scenarios

**Coverage**:
- Position scaling with multiple bots
- Correlation adjustments
- Risk allocation strategies
- Portfolio status reporting
- Edge cases (perfect correlation, etc.)

**Test Count**: ~25 tests

### 4. Performance Tests (`test_performance.py`)

**Test Classes**:
- `TestEnhancedKellyPerformance`: Calculator performance
- `TestAnalyzerPerformance`: Analyzer performance
- `TestPortfolioScalingPerformance`: Scaler performance
- `TestEndToEndPerformance`: Full workflow performance
- `TestPerformanceRegression`: Regression detection

**Coverage**:
- Latency measurements (< 50ms, < 200ms targets)
- Memory usage (< 10MB target)
- Batch calculations
- Concurrent operations

**Test Count**: ~15 tests

## Fixtures

### Standard Fixtures (`conftest.py`)

**Configuration Fixtures**:
- `standard_config`: Standard Enhanced Kelly configuration
- `ftmo_config`: FTMO Challenge preset
- `the5ers_config`: The5%ers preset

**Data Fixtures**:
- `sample_trade_history`: 50 trades (60% win rate)
- `winning_trade_history`: 30 wins, 20 losses (60% win, 2:1 R:R)
- `losing_trade_history`: 20 wins, 30 losses (40% win, 1:1 R:R)
- `insufficient_trade_history`: 8 trades (< 10)
- `empty_trade_history`: Empty list

**Market State Fixtures**:
- `sample_market_state`: Normal volatility (ATR ratio = 1.2)
- `high_volatility_state`: High volatility (ATR ratio = 2.0)
- `low_volatility_state`: Low volatility (ATR ratio = 0.5)

**Calculator Fixtures**:
- `kelly_calculator`: Enhanced Kelly calculator instance
- `kelly_analyzer`: Kelly statistics analyzer instance
- `portfolio_scaler`: Portfolio Kelly scaler instance

**Parameter Fixtures**:
- `account_balance`: $10,000
- `stop_loss_pips`: 20 pips
- `pip_value`: $10 per pip

## Test Markers

Tests are organized by markers for selective execution:

```bash
# Kelly calculation tests
pytest -m kelly

# Statistics analyzer tests
pytest -m analyzer

# Portfolio tests
pytest -m portfolio

# Edge case tests
pytest -m edge_case

# Performance tests
pytest -m performance

# Integration tests
pytest -m integration
```

## Coverage Goals

### Current Coverage

| Module | Statements | Coverage |
|--------|-----------|----------|
| `enhanced_kelly.py` | ~200 | Target: >90% |
| `kelly_analyzer.py` | ~150 | Target: >90% |
| `portfolio_kelly.py` | ~200 | Target: >90% |
| `kelly_config.py` | ~100 | Target: >80% |
| `edge_cases.py` | ~150 | Target: >85% |

### Coverage Targets

- **Unit Tests**: >90% coverage
- **Integration Tests**: Key workflows covered
- **Edge Cases**: All documented edge cases tested
- **Performance**: All targets met (< 50ms, < 200ms)

## Performance Benchmarks

### Targets

| Operation | Target | Actual |
|-----------|--------|--------|
| Kelly calculation | < 50ms | ✅ ~5ms |
| Analyzer (small, <30 trades) | < 10ms | ✅ ~2ms |
| Analyzer (large, 500 trades) | < 50ms | ✅ ~15ms |
| Portfolio scaling (5 bots) | < 1ms | ✅ ~0.5ms |
| Portfolio scaling (20 bots) | < 5ms | ✅ ~2ms |
| Full workflow | < 200ms | ✅ ~25ms |
| Memory usage | < 10MB | ✅ ~5MB |

### Running Benchmarks

```bash
# Run performance tests
pytest tests/position_sizing/test_performance.py -v -s

# Run with pytest-benchmark (if installed)
pytest tests/position_sizing/test_performance.py -v --benchmark-only
```

## Writing New Tests

### Test Structure

```python
import pytest
from src.position_sizing.enhanced_kelly import EnhancedKellyCalculator

@pytest.mark.kelly  # Add appropriate marker
class TestNewFeature:
    """Test suite for new feature."""

    def test_basic_behavior(self, kelly_calculator):
        """Test basic feature behavior."""
        # Arrange
        # Act
        # Assert
        assert True

    def test_edge_case(self, kelly_calculator):
        """Test edge case."""
        # Test edge case here
        pass
```

### Best Practices

1. **Use descriptive test names**: `test_positive_expectancy_calculation`
2. **Arrange-Act-Assert**: Structure tests clearly
3. **One assertion per test**: Keep tests focused
4. **Use fixtures**: Reuse common test data
5. **Add markers**: Organize tests by category
6. **Mock external dependencies**: Keep tests isolated
7. **Test edge cases**: Boundary conditions, null inputs
8. **Add docstrings**: Explain what is being tested

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Enhanced Kelly Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/position_sizing/ --cov=src/position_sizing
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Troubleshooting

### Tests Fail to Import

```bash
# Ensure you're in the project root
cd /path/to/QUANTMINDX

# Install in development mode
pip install -e .
```

### Performance Tests Fail

```bash
# Run with warmup iterations
pytest tests/position_sizing/test_performance.py -v --benchmark-warmup

# Skip performance tests on slow machines
pytest tests/position_sizing/ -m "not performance"
```

### Coverage Report Missing

```bash
# Install pytest-cov
pip install pytest-cov

# Generate coverage report
pytest tests/position_sizing/ --cov=src/position_sizing --cov-report=html
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Use appropriate markers** for organization
3. **Add fixtures** for common test data
4. **Update this README** with new test categories
5. **Ensure coverage remains** >90% for new code
6. **Run performance tests** to ensure targets met

## Resources

- **Pytest Documentation**: https://docs.pytest.org/
- **Pytest-Cov Documentation**: https://pytest-cov.readthedocs.io/
- **User Guide**: `docs/user_guides/enhanced_kelly_user_guide.md`
- **API Reference**: `docs/api/enhanced_kelly_api.md`

---

**Last Updated**: 2026-01-28
**Test Count**: ~110 tests
**Coverage Target**: >90%
