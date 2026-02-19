"""
Monte Carlo Simulation WebSocket Endpoint

Provides real-time streaming of Monte Carlo simulation results.
Updates charts every 100 simulations as they complete.
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ============== Data Models ==============

class SimulationConfig(BaseModel):
    initial_capital: float = 10000.0
    num_simulations: int = 10000
    trading_days: int = 252
    run_id: Optional[str] = None
    # Optional: historical returns for bootstrap
    returns: Optional[List[float]] = None
    # Or use parametric approach
    mean_return: Optional[float] = None
    volatility: Optional[float] = None


class SimulationProgress(BaseModel):
    completed: int
    total: int
    progress: float


# ============== Simulation Engine ==============

class MonteCarloEngine:
    """Monte Carlo simulation engine with streaming results."""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.results: np.ndarray = None
        self.daily_paths: np.ndarray = None
        self.is_running = False
        self.is_paused = False
        self.completed = 0

    async def run_simulation(self, progress_callback, results_callback):
        """Run the Monte Carlo simulation with progress updates."""
        self.is_running = True
        self.completed = 0

        n_sim = self.config.num_simulations
        n_days = self.config.trading_days
        initial = self.config.initial_capital

        # Initialize arrays
        self.daily_paths = np.zeros((n_sim, n_days + 1))
        self.daily_paths[:, 0] = initial

        # Calculate daily returns
        if self.config.returns:
            # Bootstrap from historical returns
            returns = np.array(self.config.returns)
            daily_returns = returns / 100 if np.max(returns) > 1 else returns
        else:
            # Parametric approach
            mean = self.config.mean_return or 0.0002  # ~5% annual
            vol = self.config.volatility or 0.01  # ~16% annual volatility
            daily_returns = np.random.normal(mean, vol, (n_sim, n_days))

        # Run simulation in batches for streaming
        batch_size = 100
        update_interval = 100  # Update every 100 simulations

        for i in range(n_sim):
            if not self.is_running:
                break

            while self.is_paused:
                await asyncio.sleep(0.1)

            # Generate path
            path = np.zeros(n_days + 1)
            path[0] = initial

            for day in range(n_days):
                if self.config.returns:
                    # Bootstrap
                    ret = np.random.choice(daily_returns)
                else:
                    # Use pre-generated
                    ret = daily_returns[i, day] if i < len(daily_returns) else np.random.normal(
                        self.config.mean_return or 0.0002,
                        self.config.volatility or 0.01
                    )
                path[day + 1] = path[day] * (1 + ret)

            self.daily_paths[i] = path
            self.completed = i + 1

            # Send progress update
            if (i + 1) % update_interval == 0 or i == n_sim - 1:
                progress = (i + 1) / n_sim * 100
                await progress_callback(SimulationProgress(
                    completed=i + 1,
                    total=n_sim,
                    progress=progress
                ))

                # Send partial results
                if (i + 1) % 500 == 0:
                    await results_callback(self._compute_partial_results(i + 1))

        # Compute final results
        self.results = self.daily_paths[:, -1]
        final_results = self._compute_final_results()
        await results_callback(final_results)

        self.is_running = False
        return final_results

    def _compute_partial_results(self, n_completed: int) -> Dict[str, Any]:
        """Compute partial results for streaming."""
        paths = self.daily_paths[:n_completed]

        # Compute percentiles for each day
        percentiles = {
            'p10': np.percentile(paths, 10, axis=0).tolist(),
            'p25': np.percentile(paths, 25, axis=0).tolist(),
            'p50': np.percentile(paths, 50, axis=0).tolist(),
            'p75': np.percentile(paths, 75, axis=0).tolist(),
            'p90': np.percentile(paths, 90, axis=0).tolist(),
        }

        return {
            'type': 'partial_results',
            'payload': {
                'fan_chart': {
                    'percentiles': percentiles,
                    'days': list(range(self.config.trading_days + 1)),
                    'initial_value': self.config.initial_capital
                }
            }
        }

    def _compute_final_results(self) -> Dict[str, Any]:
        """Compute final simulation results."""
        final_values = self.daily_paths[:, -1]
        n_sim = len(final_values)

        # Fan chart data
        percentiles = {
            'p10': np.percentile(self.daily_paths, 10, axis=0).tolist(),
            'p25': np.percentile(self.daily_paths, 25, axis=0).tolist(),
            'p50': np.percentile(self.daily_paths, 50, axis=0).tolist(),
            'p75': np.percentile(self.daily_paths, 75, axis=0).tolist(),
            'p90': np.percentile(self.daily_paths, 90, axis=0).tolist(),
        }

        # Heatmap data (sample 100 runs for visualization)
        sample_size = min(100, n_sim)
        sample_indices = np.random.choice(n_sim, sample_size, replace=False)
        sampled_paths = self.daily_paths[sample_indices]

        # Subsample days for heatmap (max 50 days)
        day_indices = np.linspace(0, self.config.trading_days, min(50, self.config.trading_days + 1), dtype=int)

        heatmap_runs = sampled_paths[:, day_indices].tolist()
        heatmap_days = day_indices.tolist()

        # Distribution data
        n_bins = 50
        hist, bin_edges = np.histogram(final_values, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Statistics
        mean_val = np.mean(final_values)
        std_val = np.std(final_values)
        median_val = np.median(final_values)
        p5 = np.percentile(final_values, 5)
        p95 = np.percentile(final_values, 95)
        worst = np.min(final_values)
        best = np.max(final_values)

        # Risk of ruin (below initial capital or threshold)
        ruin_threshold = self.config.initial_capital * 0.5  # 50% loss
        risk_of_ruin = np.sum(final_values < ruin_threshold) / n_sim

        # Win probability
        win_prob = np.sum(final_values > self.config.initial_capital) / n_sim

        # Sharpe-like ratio (assuming risk-free rate of 0)
        daily_returns = (final_values - self.config.initial_capital) / self.config.initial_capital
        sharpe = np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0

        # Max drawdown calculation
        cumulative_max = np.maximum.accumulate(self.daily_paths, axis=1)
        drawdowns = (cumulative_max - self.daily_paths) / cumulative_max
        max_drawdowns = np.max(drawdowns, axis=1)
        avg_max_drawdown = np.mean(max_drawdowns)

        return {
            'type': 'results',
            'payload': {
                'fan_chart': {
                    'percentiles': percentiles,
                    'days': list(range(self.config.trading_days + 1)),
                    'initial_value': self.config.initial_capital
                },
                'heatmap': {
                    'runs': heatmap_runs,
                    'days': heatmap_days,
                    'min_value': float(worst),
                    'max_value': float(best)
                },
                'distribution': {
                    'values': final_values.tolist(),
                    'bins': bin_centers.tolist(),
                    'frequencies': hist.tolist(),
                    'statistics': {
                        'mean': float(mean_val),
                        'median': float(median_val),
                        'stdDev': float(std_val),
                        'percentile5': float(p5),
                        'percentile95': float(p95),
                        'riskOfRuin': float(risk_of_ruin)
                    }
                },
                'statistics': {
                    'expected_value': float(mean_val),
                    'worst_case': float(p5),
                    'best_case': float(p95),
                    'risk_of_ruin': float(risk_of_ruin),
                    'sharpe_ratio': float(sharpe),
                    'max_drawdown': float(avg_max_drawdown),
                    'win_probability': float(win_prob)
                }
            }
        }

    def pause(self):
        """Pause the simulation."""
        self.is_paused = True

    def resume(self):
        """Resume the simulation."""
        self.is_paused = False

    def stop(self):
        """Stop the simulation."""
        self.is_running = False


# ============== WebSocket Manager ==============

class MonteCarloWebSocketManager:
    """Manages Monte Carlo WebSocket connections."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.simulations: Dict[str, MonteCarloEngine] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Monte Carlo WebSocket connected: {client_id}")

    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.simulations:
            self.simulations[client_id].stop()
            del self.simulations[client_id]
        logger.info(f"Monte Carlo WebSocket disconnected: {client_id}")

    async def send_progress(self, client_id: str, progress: SimulationProgress):
        """Send progress update to client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps({
                'type': 'progress',
                'progress': progress.progress,
                'completed': progress.completed,
                'total': progress.total
            }))

    async def send_results(self, client_id: str, results: Dict[str, Any]):
        """Send results to client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps(
                results, default=str
            ))

    async def send_error(self, client_id: str, message: str):
        """Send error message to client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(json.dumps({
                'type': 'error',
                'message': message
            }))


# Global manager
ws_manager = MonteCarloWebSocketManager()


# ============== WebSocket Endpoint ==============

async def monte_carlo_websocket(websocket: WebSocket):
    """WebSocket endpoint for Monte Carlo simulation."""
    import uuid
    client_id = str(uuid.uuid4())

    await ws_manager.connect(websocket, client_id)

    async def progress_callback(progress: SimulationProgress):
        await ws_manager.send_progress(client_id, progress)

    async def results_callback(results: Dict[str, Any]):
        await ws_manager.send_results(client_id, results)

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            msg_type = message.get('type')

            if msg_type == 'start_simulation':
                config = SimulationConfig(**message.get('config', {}))
                engine = MonteCarloEngine(config)
                ws_manager.simulations[client_id] = engine

                # Run simulation in background
                asyncio.create_task(
                    engine.run_simulation(progress_callback, results_callback)
                )

            elif msg_type == 'pause':
                if client_id in ws_manager.simulations:
                    ws_manager.simulations[client_id].pause()

            elif msg_type == 'resume':
                if client_id in ws_manager.simulations:
                    ws_manager.simulations[client_id].resume()

            elif msg_type == 'get_results':
                # Return cached results if available
                if client_id in ws_manager.simulations:
                    engine = ws_manager.simulations[client_id]
                    if engine.results is not None:
                        await results_callback(engine._compute_final_results())

            elif msg_type == 'ping':
                await websocket.send_text(json.dumps({'type': 'pong'}))

    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.send_error(client_id, str(e))
        ws_manager.disconnect(client_id)


# Export for router registration
monte_carlo_ws_endpoint = monte_carlo_websocket
