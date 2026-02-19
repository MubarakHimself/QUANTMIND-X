"""
MCP Tools for Paper Trading Deployment.

Registers all paper trading deployment tools with FastMCP server.
"""

import logging
from typing import Optional, List

from fastmcp import FastMCP

from .deployer import PaperTradingDeployer
from .monitor import AgentHealthMonitor, AgentHealth
from .storage import PaperTradingStorage
from .models import (
    AgentDeploymentRequest,
    AgentDeploymentResult,
    PaperAgentStatus,
    AgentLogsResult,
    AgentPerformance,
)

logger = logging.getLogger(__name__)

# Singleton instances
_deployer: Optional[PaperTradingDeployer] = None
_monitor: Optional[AgentHealthMonitor] = None
_storage: Optional[PaperTradingStorage] = None


def _get_deployer() -> PaperTradingDeployer:
    """Get or create deployer instance."""
    global _deployer
    if _deployer is None:
        _deployer = PaperTradingDeployer()
    return _deployer


def _get_monitor() -> AgentHealthMonitor:
    """Get or create monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = AgentHealthMonitor()
        _monitor.start()
    return _monitor


def _get_storage() -> PaperTradingStorage:
    """Get or create storage instance."""
    global _storage
    if _storage is None:
        _storage = PaperTradingStorage()
    return _storage


def register_paper_trading_tools(mcp: FastMCP):
    """
    Register all paper trading deployment tools.

    Args:
        mcp: FastMCP server instance
    """

    @mcp.tool()
    def deploy_paper_agent(
        strategy_name: str,
        strategy_code: str,
        config: dict,
        mt5_credentials: dict,
        magic_number: int,
        agent_id: Optional[str] = None,
        image_tag: str = "latest",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        environment_vars: Optional[dict] = None,
        resource_limits: Optional[dict] = None,
    ) -> AgentDeploymentResult:
        """
        Deploy a paper trading agent as a Docker container.

        Creates a new containerized paper trading agent with the specified strategy,
        configuration, and MT5 credentials. The agent will connect to Redis for
        event publishing and health monitoring.

        Args:
            strategy_name: Name of the trading strategy (e.g., "RSI Reversal", "MACD Crossover")
            strategy_code: Strategy Python code or template reference (e.g., "template:rsi-reversal")
            config: Strategy configuration parameters
                Example: {"rsi_period": 14, "oversold": 30, "overbought": 70}
            mt5_credentials: MT5 account credentials
                Required keys: account (int), password (str), server (str)
            magic_number: Unique EA magic number for trade identification (0-2147483647)
            agent_id: Custom agent ID (auto-generated if not provided)
            image_tag: Docker image tag to use (default: "latest")
            redis_host: Redis host for event publishing (default: "localhost")
            redis_port: Redis port (default: 6379)
            redis_db: Redis database number (default: 0)
            environment_vars: Additional environment variables (optional)
            resource_limits: Container resource limits (optional)
                Example: {"memory": "512m", "cpus": "1.0"}

        Returns:
            AgentDeploymentResult with deployment details:
                - agent_id: Unique agent identifier
                - container_id: Docker container ID
                - container_name: Docker container name
                - status: Initial agent status (running, starting, error)
                - redis_channel: Redis channel for monitoring
                - logs_url: Command to view logs
                - message: Deployment status message

        Example:
            result = deploy_paper_agent(
                strategy_name="RSI Reversal",
                strategy_code="template:rsi-reversal",
                config={"rsi_period": 14, "symbols": ["EURUSD"]},
                mt5_credentials={
                    "account": 12345678,
                    "password": "demo_password",
                    "server": "MetaQuotes-Demo"
                },
                magic_number=98765432
            )
            print(f"Agent deployed: {result.agent_id}")
            print(f"Logs: docker logs -f {result.container_name}")
        """
        deployer = _get_deployer()

        return deployer.deploy_agent(
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            config=config,
            mt5_credentials=mt5_credentials,
            magic_number=magic_number,
            agent_id=agent_id,
            image_tag=image_tag,
            redis_host=redis_host,
            redis_port=redis_port,
            redis_db=redis_db,
            environment_vars=environment_vars,
            resource_limits=resource_limits,
        )

    @mcp.tool()
    def list_paper_agents(
        include_stopped: bool = False,
    ) -> List[PaperAgentStatus]:
        """
        List all paper trading agent containers.

        Returns a list of all deployed paper trading agents with their current
        status, health, and configuration.

        Args:
            include_stopped: Include stopped containers (default: False)

        Returns:
            List of PaperAgentStatus with:
                - agent_id: Unique agent identifier
                - container_id: Docker container ID
                - status: Agent lifecycle status (running, stopped, error, etc.)
                - health: Agent health based on heartbeat (healthy, stale, dead)
                - strategy_name: Name of the trading strategy
                - mt5_account: MT5 account number
                - uptime_seconds: Agent uptime in seconds
                - last_heartbeat: Last heartbeat timestamp
                - missed_heartbeats: Number of missed heartbeats

        Example:
            agents = list_paper_agents()
            for agent in agents:
                print(f"{agent.agent_id}: {agent.status} ({agent.health})")
        """
        deployer = _get_deployer()
        monitor = _get_monitor()

        agents = deployer.list_agents()

        # Update with health status from monitor
        agents_with_health = monitor.update_agent_statuses(agents)

        # Filter out stopped if requested
        if not include_stopped:
            agents_with_health = [
                a for a in agents_with_health
                if a.status.value in ["running", "starting"]
            ]

        return agents_with_health

    @mcp.tool()
    def get_paper_agent(
        agent_id: str,
    ) -> Optional[PaperAgentStatus]:
        """
        Get status of a specific paper trading agent.

        Returns detailed status and health information for a single agent.

        Args:
            agent_id: Agent identifier

        Returns:
            PaperAgentStatus if found, None otherwise

        Example:
            agent = get_paper_agent("strategy-rsi-eurusd-001")
            if agent:
                print(f"Status: {agent.status}, Health: {agent.health}")
        """
        deployer = _get_deployer()
        monitor = _get_monitor()

        agent_status = deployer.get_agent(agent_id)
        if agent_status:
            # Update with health status
            health = monitor.get_agent_health(agent_id)
            agent_status.health = health
            agent_status.last_heartbeat = monitor.get_last_heartbeat(agent_id)
            agent_status.missed_heartbeats = monitor.get_missed_heartbeat_count(agent_id)

        return agent_status

    @mcp.tool()
    def stop_paper_agent(
        agent_id: str,
        timeout: int = 30,
        force: bool = False,
    ) -> bool:
        """
        Stop a paper trading agent gracefully.

        Sends SIGTERM for graceful shutdown, allowing the agent to close
        open positions. If timeout is exceeded, uses SIGKILL.

        Args:
            agent_id: Agent identifier
            timeout: Graceful shutdown timeout in seconds (default: 30)
            force: If True, use SIGKILL immediately (default: False)

        Returns:
            True if stopped successfully

        Example:
            stop_paper_agent("strategy-rsi-eurusd-001")
        """
        deployer = _get_deployer()
        return deployer.stop_agent(agent_id=agent_id, timeout=timeout, force=force)

    @mcp.tool()
    def remove_paper_agent(
        agent_id: str,
        force: bool = False,
    ) -> bool:
        """
        Stop and remove a paper trading agent container.

        Permanently removes the agent container after stopping it.

        Args:
            agent_id: Agent identifier
            force: If True, force removal

        Returns:
            True if removed successfully

        Example:
            remove_paper_agent("strategy-rsi-eurusd-001")
        """
        deployer = _get_deployer()
        return deployer.remove_agent(agent_id=agent_id, force=force)

    @mcp.tool()
    def get_agent_logs(
        agent_id: str,
        tail_lines: int = 100,
    ) -> AgentLogsResult:
        """
        Retrieve logs from a paper trading agent container.

        Returns the most recent log lines from the agent's container.

        Args:
            agent_id: Agent identifier
            tail_lines: Number of lines to retrieve (default: 100, max: 10000)

        Returns:
            AgentLogsResult with:
                - agent_id: Agent identifier
                - logs: List of log lines
                - line_count: Number of log lines returned
                - tail_lines: Number of lines requested
                - has_more: True if more logs available

        Example:
            logs = get_agent_logs("strategy-rsi-eurusd-001", tail_lines=50)
            for line in logs.logs:
                print(line)
        """
        deployer = _get_deployer()
        return deployer.get_agent_logs(agent_id=agent_id, tail_lines=tail_lines)

    @mcp.tool()
    def get_agent_performance(
        agent_id: str,
    ) -> Optional[AgentPerformance]:
        """
        Get performance metrics for a paper trading agent.

        Calculates trading performance metrics from stored trade events.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentPerformance with:
                - total_trades: Total number of trades
                - winning_trades: Number of winning trades
                - losing_trades: Number of losing trades
                - win_rate: Win rate percentage
                - total_pnl: Total profit/loss
                - average_pnl: Average profit/loss per trade
                - max_drawdown: Maximum drawdown
                - profit_factor: Profit factor
                - sharpe_ratio: Sharpe ratio (if enough data)
                - symbols_traded: List of symbols traded
                - first_trade_at: First trade timestamp
                - last_trade_at: Last trade timestamp

        Example:
            perf = get_agent_performance("strategy-rsi-eurusd-001")
            print(f"Win Rate: {perf.win_rate:.2f}%")
            print(f"Total P&L: ${perf.total_pnl:.2f}")
        """
        storage = _get_storage()

        # Get trade events from storage
        trades = storage.get_agent_trades(agent_id, event_type="exit")

        if not trades:
            return None

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0.0
        max_drawdown = 0.0
        gross_profit = 0.0
        gross_loss = 0.0

        symbols = set()
        first_trade = None
        last_trade = None

        for trade in trades:
            metadata = trade["metadata"]
            pnl = float(metadata.get("pnl", 0))
            total_pnl += pnl

            if pnl > 0:
                winning_trades += 1
                gross_profit += pnl
            else:
                losing_trades += 1
                gross_loss += abs(pnl)

            symbol = metadata.get("symbol", "")
            if symbol:
                symbols.add(symbol)

            # Track timestamps
            timestamp_str = metadata.get("timestamp")
            if timestamp_str:
                try:
                    from datetime import datetime, UTC
                    ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    if first_trade is None or ts < first_trade:
                        first_trade = ts
                    if last_trade is None or ts > last_trade:
                        last_trade = ts
                except Exception:
                    pass

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        return AgentPerformance(
            agent_id=agent_id,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            average_pnl=average_pnl,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor,
            sharpe_ratio=None,  # TODO: Calculate if enough data
            symbols_traded=list(symbols),
            first_trade_at=first_trade,
            last_trade_at=last_trade,
        )

    @mcp.tool()
    def get_agent_health(
        agent_id: str,
    ) -> str:
        """
        Get health status of a paper trading agent.

        Returns the health status based on heartbeat monitoring.

        Args:
            agent_id: Agent identifier

        Returns:
            Health status: "healthy", "stale", or "dead"

        Health definitions:
            - healthy: Heartbeat received within last 5 minutes
            - stale: No heartbeat for 5-10 minutes
            - dead: No heartbeat for 10+ minutes

        Example:
            health = get_agent_health("strategy-rsi-eurusd-001")
            if health == "dead":
                print("Agent may need restart")
        """
        monitor = _get_monitor()
        health = monitor.get_agent_health(agent_id)
        return health.value

    @mcp.tool()
    def restart_agent(
        agent_id: str,
        timeout: int = 30,
    ) -> bool:
        """
        Restart a paper trading agent.

        Stops and restarts an agent container while preserving configuration.

        Args:
            agent_id: Agent identifier
            timeout: Graceful shutdown timeout (default: 30)

        Returns:
            True if restarted successfully

        Example:
            restart_agent("strategy-rsi-eurusd-001")
        """
        deployer = _get_deployer()

        # Get current agent status to preserve config
        agent = deployer.get_agent(agent_id)
        if not agent:
            return False

        # Stop the agent
        if not deployer.stop_agent(agent_id=agent_id, timeout=timeout):
            return False

        # Restart the container
        try:
            client = deployer._get_client()
            container_name = deployer._generate_container_name(agent_id)
            container = client.containers.get(container_name)
            container.start()
            return True
        except Exception as e:
            logger.error(f"Failed to restart agent {agent_id}: {e}")
            return False


# Cleanup function for graceful shutdown
def cleanup_paper_trading():
    """Cleanup resources on shutdown."""
    global _monitor, _deployer, _storage

    if _monitor:
        _monitor.stop()
        _monitor = None

    if _deployer:
        deployer.close()
        _deployer = None

    if _storage:
        storage.close()
        _storage = None

    logger.info("Paper trading resources cleaned up")
