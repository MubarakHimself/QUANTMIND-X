"""
Tests for EA API Endpoints.

These tests verify that the EA API endpoints work correctly and that
deploy-live endpoint does NOT exist (as live trading is prohibited).
"""

import unittest
from unittest.mock import Mock, patch
from fastapi import APIRouter
from pydantic import BaseModel
from src.api.ea_endpoints import router
from src.agents.tools.ea_lifecycle import EALifecycleTools, EALifecycleStatus
from src.api.ea_endpoints import (
    EACreationRequest, EAValidationRequest, EABacktestRequest, EAStressTestRequest,
    EAMonteCarloRequest, EADeployPaperRequest, EAOptimizationRequest,
    EACreationResponse, EAValidationResponse, EABacktestResponse, EAStressTestResponse,
    EAMonteCarloResponse, EADeployPaperResponse, EAMonitorResponse, EAOptimizationResponse,
    EAStopResponse, EAListResponse
)


class TestEAAPI(unittest.TestCase):
    """Test suite for EA API endpoints"""

    def setUp(self):
        """Set up test environment"""
        self.mock_tools = Mock(spec=EALifecycleTools)
        self.patcher = patch('src.api.ea_endpoints.get_ea_lifecycle_tools', return_value=self.mock_tools)
        self.patcher.start()

        # Create mock EA data
        self.ea_id = "ea_1"
        self.strategy_code = "def on_tick(): pass"
        self.parameters = {"stop_loss": 20.0, "take_profit": 40.0}

        # Create mock responses
        self.create_response = EACreationResponse(
            ea_id=self.ea_id,
            status=EALifecycleStatus.CREATED.value,
            created_at=1234567890.0
        )

        self.list_response = EAListResponse(
            eas=[
                {
                    'id': self.ea_id,
                    'status': EALifecycleStatus.CREATED.value,
                    'created_at': 1234567890.0,
                    'strategy_code': self.strategy_code,
                    'parameters': self.parameters
                }
            ]
        )

        self.validate_response = EAValidationResponse(
            ea_id=self.ea_id,
            status=EALifecycleStatus.VALIDATED.value,
            validation_errors=[],
            validated_at=1234567891.0
        )

        self.backtest_response = EABacktestResponse(
            ea_id=self.ea_id,
            metrics={
                'total_trades': 150,
                'winning_trades': 75,
                'losing_trades': 75,
                'win_rate': 0.5,
                'average_win': 25.5,
                'average_loss': -18.2,
                'max_drawdown': 12.3,
                'total_profit': 1250.0,
                'profit_factor': 1.4
            },
            equity_curve=[
                {'time': 0, 'equity': 10000.0},
                {'time': 1, 'equity': 10125.0}
            ],
            drawdown=12.3,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            status=EALifecycleStatus.BACKTESTED.value,
            completed_at=1234567892.0
        )

        self.stress_test_response = EAStressTestResponse(
            ea_id=self.ea_id,
            status=EALifecycleStatus.STRESS_TESTED.value,
            stress_test_results={
                'performance_under_high_volatility': 'PASS',
                'performance_under_low_liquidity': 'PASS',
                'parameter_sensitivity': 'MODERATE',
                'max_drawdown_under_stress': 18.5,
                'completed_at': 1234567893.0
            },
            completed_at=1234567893.0
        )

        self.monte_carlo_response = EAMonteCarloResponse(
            ea_id=self.ea_id,
            simulation_results=[
                {'run': 1, 'final_equity': 12500.0, 'max_drawdown': 15.2}
            ],
            confidence_intervals={
                '95%_lower': 10500.0,
                '95%_upper': 14500.0,
                'median': 12000.0
            },
            probability_of_ruin=0.05,
            status=EALifecycleStatus.MONTE_CARLO_COMPLETE.value,
            completed_at=1234567894.0
        )

        self.paper_deploy_response = EADeployPaperResponse(
            ea_id=self.ea_id,
            status=EALifecycleStatus.PAPER_DEPLOYED.value,
            paper_trading_id=f"paper_{self.ea_id}",
            deployed_at=1234567895.0,
            initial_balance=10000.0
        )

        self.monitor_response = EAMonitorResponse(
            ea_id=self.ea_id,
            status=EALifecycleStatus.MONITORING.value,
            current_equity=10250.0,
            unrealized_pnl=250.0,
            trades_executed=5,
            last_update=1234567896.0
        )

        self.optimize_response = EAOptimizationResponse(
            ea_id=self.ea_id,
            status=EALifecycleStatus.BACKTESTED.value,
            optimization_results={
                'best_parameters': {
                    'stop_loss': 15.0,
                    'take_profit': 30.0,
                    'trailing_stop': True
                },
                'improvement': 15.2,
                'optimized_at': 1234567897.0
            },
            optimized_at=1234567897.0
        )

        self.stop_response = EAStopResponse(
            ea_id=self.ea_id,
            status=EALifecycleStatus.STOPPED.value,
            stopped_at=1234567898.0,
            final_equity=10300.0,
            total_pnl=300.0
        )

    def tearDown(self):
        """Clean up"""
        self.patcher.stop()

    def test_create_ea_endpoint_exists(self):
        """Test that create EA endpoint exists"""
        # Check that the endpoint exists in the router
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/create", endpoints)

        # Configure mock
        self.mock_tools.create_ea.return_value = self.create_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import create_ea
        response = create_ea(
            request=EACreationRequest(strategy_code=self.strategy_code, parameters=self.parameters),
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_list_eas_endpoint_exists(self):
        """Test that list EAs endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/list", endpoints)

        # Configure mock
        self.mock_tools.list_eas.return_value = self.list_response.eas

        # Test the endpoint function directly
        from src.api.ea_endpoints import list_eas
        response = list_eas(tools=self.mock_tools)
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.eas, self.list_response['eas'])

    def test_validate_ea_endpoint_exists(self):
        """Test that validate EA endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/{ea_id}/validate", endpoints)

        # Configure mock
        self.mock_tools.validate_ea.return_value = self.validate_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import validate_ea
        response = validate_ea(
            ea_id=self.ea_id,
            request=EAValidationRequest(),
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_backtest_ea_endpoint_exists(self):
        """Test that backtest EA endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/{ea_id}/backtest", endpoints)

        # Configure mock
        self.mock_tools.validate_ea.return_value = self.validate_response
        self.mock_tools.backtest_ea.return_value = self.backtest_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import backtest_ea
        response = backtest_ea(
            ea_id=self.ea_id,
            request=EABacktestRequest(
                symbol="EURUSD",
                timeframe="H1",
                start_date="2023-01-01",
                end_date="2023-12-31",
                initial_balance=10000.0
            ),
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_stress_test_ea_endpoint_exists(self):
        """Test that stress test EA endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/{ea_id}/stress-test", endpoints)

        # Configure mock
        self.mock_tools.validate_ea.return_value = self.validate_response
        self.mock_tools.backtest_ea.return_value = self.backtest_response
        self.mock_tools.stress_test_ea.return_value = self.stress_test_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import stress_test_ea
        response = stress_test_ea(
            ea_id=self.ea_id,
            request=EAStressTestRequest(
                volatility_multiplier=2.0,
                liquidity_multiplier=0.5
            ),
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_monte_carlo_sim_endpoint_exists(self):
        """Test that Monte Carlo simulation endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/{ea_id}/monte-carlo", endpoints)

        # Configure mock
        self.mock_tools.validate_ea.return_value = self.validate_response
        self.mock_tools.backtest_ea.return_value = self.backtest_response
        self.mock_tools.monte_carlo_sim.return_value = self.monte_carlo_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import monte_carlo_sim
        response = monte_carlo_sim(
            ea_id=self.ea_id,
            request=EAMonteCarloRequest(
                num_simulations=1000,
                confidence_level=0.95
            ),
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_deploy_paper_endpoint_exists(self):
        """Test that deploy paper endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/{ea_id}/deploy-paper", endpoints)

        # Configure mock
        self.mock_tools.validate_ea.return_value = self.validate_response
        self.mock_tools.backtest_ea.return_value = self.backtest_response
        self.mock_tools.deploy_paper.return_value = self.paper_deploy_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import deploy_paper
        response = deploy_paper(
            ea_id=self.ea_id,
            request=EADeployPaperRequest(
                initial_balance=10000.0,
                leverage=50
            ),
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_monitor_ea_endpoint_exists(self):
        """Test that monitor EA endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/{ea_id}/monitor", endpoints)

        # Configure mock
        self.mock_tools.validate_ea.return_value = self.validate_response
        self.mock_tools.backtest_ea.return_value = self.backtest_response
        self.mock_tools.deploy_paper.return_value = self.paper_deploy_response
        self.mock_tools.monitor_ea.return_value = self.monitor_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import monitor_ea
        response = monitor_ea(
            ea_id=self.ea_id,
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_optimize_ea_endpoint_exists(self):
        """Test that optimize EA endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/{ea_id}/optimize", endpoints)

        # Configure mock
        self.mock_tools.validate_ea.return_value = self.validate_response
        self.mock_tools.backtest_ea.return_value = self.backtest_response
        self.mock_tools.optimize_ea.return_value = self.optimize_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import optimize_ea
        response = optimize_ea(
            ea_id=self.ea_id,
            request=EAOptimizationRequest(
                optimization_method="genetic",
                population_size=50,
                generations=20
            ),
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_stop_ea_endpoint_exists(self):
        """Test that stop EA endpoint exists"""
        endpoints = [route.path for route in router.routes]
        self.assertIn("/api/ea/{ea_id}/stop", endpoints)

        # Configure mock
        self.mock_tools.validate_ea.return_value = self.validate_response
        self.mock_tools.backtest_ea.return_value = self.backtest_response
        self.mock_tools.deploy_paper.return_value = self.paper_deploy_response
        self.mock_tools.stop_ea.return_value = self.stop_response

        # Test the endpoint function directly
        from src.api.ea_endpoints import stop_ea
        response = stop_ea(
            ea_id=self.ea_id,
            tools=self.mock_tools
        )
        # Await the coroutine
        response = response.__await__().__next__()
        self.assertEqual(response.ea_id, self.ea_id)

    def test_deploy_live_endpoint_does_not_exist(self):
        """CRITICAL: Verify that deploy-live endpoint does NOT exist"""
        # Check that the endpoint does NOT exist in the router
        from src.api.ea_endpoints import router
        endpoints = [route.path for route in router.routes]
        self.assertNotIn("/api/ea/{ea_id}/deploy-live", endpoints)

        # Test that accessing the non-existent endpoint would fail
        found = any(route.path == "/api/ea/{ea_id}/deploy-live" for route in router.routes)
        self.assertFalse(found)

    def test_invalid_ea_operations(self):
        """Test operations on non-existent EA"""
        non_existent_ea_id = "ea_non_existent"

        # Configure mock to raise ValueError for non-existent EA
        self.mock_tools.validate_ea.side_effect = ValueError(f"EA {non_existent_ea_id} not found")

        # Test validation on non-existent EA
        from src.api.ea_endpoints import validate_ea
        with self.assertRaises(ValueError):
            validate_ea(
                ea_id=non_existent_ea_id,
                request={},
                tools=self.mock_tools
            )

        # Test backtest on non-existent EA
        from src.api.ea_endpoints import backtest_ea
        with self.assertRaises(ValueError):
            backtest_ea(
                ea_id=non_existent_ea_id,
                request={
                    "symbol": "EURUSD",
                    "timeframe": "H1",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "initial_balance": 10000.0
                },
                tools=self.mock_tools
            )

        # Test paper deployment on non-existent EA
        from src.api.ea_endpoints import deploy_paper
        with self.assertRaises(ValueError):
            deploy_paper(
                ea_id=non_existent_ea_id,
                request={
                    "initial_balance": 10000.0,
                    "leverage": 50
                },
                tools=self.mock_tools
            )


if __name__ == '__main__':
    unittest.main()