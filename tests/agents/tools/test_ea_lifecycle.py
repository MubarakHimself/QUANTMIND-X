"""
Tests for EA Lifecycle Tools.

These tests verify that the EALifecycleTools class works correctly and that
deploy_live method does NOT exist (as live trading is prohibited).
"""

import unittest
from src.agents.tools.ea_lifecycle import EALifecycleTools, EALifecycleStatus, EACreationResult, EABacktestResult, EAMonteCarloResult


class TestEALifecycleTools(unittest.TestCase):
    """Test suite for EALifecycleTools"""

    def setUp(self):
        """Set up test environment"""
        self.tools = EALifecycleTools()
        self.test_strategy_code = """
def on_tick():
    # Simple moving average crossover strategy
    sma_short = calculate_sma(20)
    sma_long = calculate_sma(50)

    if sma_short > sma_long and not position_exists():
        buy()
    elif sma_short < sma_long and position_exists():
        sell()
        """

        self.test_parameters = {
            'stop_loss': 20.0,
            'take_profit': 40.0,
            'trailing_stop': True,
            'lot_size': 0.1
        }

        self.test_backtest_params = {
            'symbol': 'EURUSD',
            'timeframe': 'H1',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'initial_balance': 10000.0
        }

        self.test_stress_params = {
            'volatility_multiplier': 2.0,
            'liquidity_multiplier': 0.5
        }

        self.test_mc_params = {
            'num_simulations': 1000,
            'confidence_level': 0.95
        }

        self.test_paper_params = {
            'initial_balance': 10000.0,
            'leverage': 50
        }

        self.test_optimization_params = {
            'optimization_method': 'genetic',
            'population_size': 50,
            'generations': 20
        }

        # Create a test EA
        self.ea_id = self.tools.create_ea(self.test_strategy_code, self.test_parameters).ea_id

    def test_create_ea(self):
        """Test EA creation"""
        result = self.tools.create_ea(self.test_strategy_code, self.test_parameters)

        self.assertIsInstance(result, EACreationResult)
        self.assertTrue(result.ea_id.startswith("ea_"))
        self.assertEqual(result.status, EALifecycleStatus.CREATED)
        self.assertIsNotNone(result.created_at)

    def test_validate_ea(self):
        """Test EA validation"""
        result = self.tools.validate_ea(self.ea_id)

        self.assertIn('ea_id', result)
        self.assertIn('status', result)
        self.assertIn('validation_errors', result)
        self.assertIn('validated_at', result)

        # Check that validation works (should pass with our test code)
        self.assertEqual(result['status'], EALifecycleStatus.VALIDATED.value)

    def test_backtest_ea(self):
        """Test EA backtest"""
        # First validate the EA
        self.tools.validate_ea(self.ea_id)

        result = self.tools.backtest_ea(self.ea_id, self.test_backtest_params)

        self.assertIsInstance(result, EABacktestResult)
        self.assertEqual(result.ea_id, self.ea_id)
        self.assertIn('metrics', result.__dict__)
        self.assertIn('equity_curve', result.__dict__)
        self.assertIn('drawdown', result.__dict__)
        self.assertIn('sharpe_ratio', result.__dict__)
        self.assertIn('sortino_ratio', result.__dict__)
        self.assertEqual(result.status, EALifecycleStatus.BACKTESTED)

    def test_stress_test_ea(self):
        """Test EA stress test"""
        # First validate and backtest the EA
        self.tools.validate_ea(self.ea_id)
        self.tools.backtest_ea(self.ea_id, self.test_backtest_params)

        result = self.tools.stress_test_ea(self.ea_id, self.test_stress_params)

        self.assertIn('ea_id', result)
        self.assertIn('status', result)
        self.assertIn('stress_test_results', result)
        self.assertEqual(result['status'], EALifecycleStatus.STRESS_TESTED.value)

    def test_monte_carlo_sim(self):
        """Test Monte Carlo simulation"""
        # First validate and backtest the EA
        self.tools.validate_ea(self.ea_id)
        self.tools.backtest_ea(self.ea_id, self.test_backtest_params)

        result = self.tools.monte_carlo_sim(self.ea_id, self.test_mc_params)

        self.assertIsInstance(result, EAMonteCarloResult)
        self.assertEqual(result.ea_id, self.ea_id)
        self.assertIn('simulation_results', result.__dict__)
        self.assertIn('confidence_intervals', result.__dict__)
        self.assertIn('probability_of_ruin', result.__dict__)
        self.assertEqual(result.status, EALifecycleStatus.MONTE_CARLO_COMPLETE)

    def test_deploy_paper(self):
        """Test paper trading deployment"""
        # First validate and backtest the EA
        self.tools.validate_ea(self.ea_id)
        self.tools.backtest_ea(self.ea_id, self.test_backtest_params)

        result = self.tools.deploy_paper(self.ea_id, self.test_paper_params)

        self.assertIn('ea_id', result)
        self.assertIn('status', result)
        self.assertIn('paper_trading_id', result)
        self.assertIn('deployed_at', result)
        self.assertIn('initial_balance', result)
        self.assertEqual(result['status'], EALifecycleStatus.PAPER_DEPLOYED.value)

    def test_monitor_ea(self):
        """Test EA monitoring"""
        # First deploy to paper trading
        self.tools.validate_ea(self.ea_id)
        self.tools.backtest_ea(self.ea_id, self.test_backtest_params)
        self.tools.deploy_paper(self.ea_id, self.test_paper_params)

        result = self.tools.monitor_ea(self.ea_id)

        self.assertIn('ea_id', result)
        self.assertIn('status', result)
        self.assertIn('current_equity', result)
        self.assertIn('unrealized_pnl', result)
        self.assertIn('trades_executed', result)
        self.assertIn('last_update', result)
        self.assertEqual(result['status'], EALifecycleStatus.MONITORING.value)

    def test_optimize_ea(self):
        """Test EA optimization"""
        # First validate and backtest the EA
        self.tools.validate_ea(self.ea_id)
        self.tools.backtest_ea(self.ea_id, self.test_backtest_params)

        result = self.tools.optimize_ea(self.ea_id, self.test_optimization_params)

        self.assertIn('ea_id', result)
        self.assertIn('status', result)
        self.assertIn('optimization_results', result)
        self.assertIn('best_parameters', result['optimization_results'])
        self.assertIn('improvement', result['optimization_results'])
        self.assertIn('optimized_at', result['optimization_results'])

    def test_stop_ea(self):
        """Test stopping EA"""
        # First deploy to paper trading
        self.tools.validate_ea(self.ea_id)
        self.tools.backtest_ea(self.ea_id, self.test_backtest_params)
        self.tools.deploy_paper(self.ea_id, self.test_paper_params)

        result = self.tools.stop_ea(self.ea_id)

        self.assertIn('ea_id', result)
        self.assertIn('status', result)
        self.assertIn('stopped_at', result)
        self.assertIn('final_equity', result)
        self.assertIn('total_pnl', result)
        self.assertEqual(result['status'], EALifecycleStatus.STOPPED.value)

    def test_list_eas(self):
        """Test listing EAs"""
        eas = self.tools.list_eas()

        self.assertIsInstance(eas, list)
        self.assertGreater(len(eas), 0)

        ea = eas[0]
        self.assertIn('id', ea)
        self.assertIn('status', ea)
        self.assertIn('created_at', ea)
        self.assertIn('strategy_code', ea)
        self.assertIn('parameters', ea)

    def test_get_ea(self):
        """Test getting EA details"""
        ea = self.tools.get_ea(self.ea_id)

        self.assertIn('id', ea)
        self.assertIn('status', ea)
        self.assertIn('created_at', ea)
        self.assertIn('strategy_code', ea)
        self.assertIn('parameters', ea)

    def test_deploy_live_method_does_not_exist(self):
        """CRITICAL: Verify that deploy_live method does NOT exist"""
        # This is a critical test to ensure live trading is prohibited
        tools = EALifecycleTools()

        # Check that the class doesn't have a deploy_live method
        self.assertFalse(hasattr(tools, 'deploy_live'))

        # Check that we can't call a non-existent method
        with self.assertRaises(AttributeError):
            tools.deploy_live("test_ea_id", {})

    def test_invalid_ea_operations(self):
        """Test operations on non-existent EA"""
        non_existent_ea_id = "ea_non_existent"

        # Test validation on non-existent EA
        with self.assertRaises(ValueError):
            self.tools.validate_ea(non_existent_ea_id)

        # Test backtest on non-existent EA
        with self.assertRaises(ValueError):
            self.tools.backtest_ea(non_existent_ea_id, self.test_backtest_params)

        # Test paper deployment on non-existent EA
        with self.assertRaises(ValueError):
            self.tools.deploy_paper(non_existent_ea_id, self.test_paper_params)


if __name__ == '__main__':
    unittest.main()