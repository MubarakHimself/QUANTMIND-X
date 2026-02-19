"""
Paper Trading Agent Deployer.

Handles deployment, listing, stopping, and log retrieval for paper trading agents
running in Docker containers.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, UTC, timedelta
from typing import Optional, Literal

import docker
from docker.errors import DockerException, NotFound, APIError
from docker.models.containers import Container

from .models import (
    AgentStatus,
    AgentHealth,
    PaperAgentStatus,
    AgentDeploymentRequest,
    AgentDeploymentResult,
    AgentLogsResult,
    PaperTradingConfig,
)

logger = logging.getLogger(__name__)

# Container label for identifying QuantMindX paper trading agents
AGENT_LABEL = "quantmindx-paper-agent"
BASE_IMAGE = "quantmindx/strategy-agent"


class PaperTradingDeployer:
    """
    Deploys and manages paper trading agents as Docker containers.

    Features:
    - Deploy agents with custom strategy code and MT5 credentials
    - List all running agents with status
    - Stop agents gracefully (SIGTERM → SIGKILL)
    - Retrieve agent logs
    - Auto-generate unique agent IDs

    Example:
        ```python
        deployer = PaperTradingDeployer()

        # Deploy a new agent
        result = deployer.deploy_agent(
            strategy_name="RSI Reversal",
            strategy_code="template:rsi-reversal",
            config={"rsi_period": 14},
            mt5_credentials={"account": 123456, "password": "...", "server": "..."},
            magic_number=98765432
        )

        # List all agents
        agents = deployer.list_agents()

        # Stop an agent
        deployer.stop_agent(result.agent_id)

        # Get logs
        logs = deployer.get_agent_logs(result.agent_id, tail_lines=100)
        ```
    """

    # Graceful shutdown timeout in seconds
    GRACEFUL_SHUTDOWN_TIMEOUT = 30

    # Maximum number of log lines to retrieve
    MAX_LOG_LINES = 10000

    def __init__(self, docker_url: Optional[str] = None):
        """
        Initialize the deployer.

        Args:
            docker_url: Docker daemon URL (default: uses environment or unix://var/run/docker.sock)
        """
        self._docker_client: Optional[docker.DockerClient] = None
        self._docker_url = docker_url

    def _get_client(self) -> docker.DockerClient:
        """Get or create Docker client."""
        if self._docker_client is None:
            try:
                self._docker_client = docker.DockerClient(base_url=self._docker_url)
                # Test connection
                self._docker_client.ping()
                logger.info("Connected to Docker daemon")
            except DockerException as e:
                logger.error(f"Failed to connect to Docker: {e}")
                raise RuntimeError(f"Cannot connect to Docker daemon: {e}") from e
        return self._docker_client

    def _generate_agent_id(self, strategy_name: str, magic_number: int) -> str:
        """
        Generate a unique agent ID.

        Format: strategy-{slug}-{magic}-{timestamp}
        """
        # Create slug from strategy name
        slug = strategy_name.lower().replace(" ", "-").replace("_", "-")
        # Remove non-alphanumeric except dash
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        # Limit length
        slug = slug[:30]

        # Add timestamp suffix for uniqueness
        timestamp = int(time.time())
        return f"{slug}-{magic_number}-{timestamp}"

    def _generate_container_name(self, agent_id: str) -> str:
        """Generate a container name for the agent."""
        return f"quantmindx-agent-{agent_id}"

    # ========================================================================
    # Deploy Agent
    # ========================================================================

    def deploy_agent(
        self,
        strategy_name: str,
        strategy_code: str,
        config: dict,
        mt5_credentials: Optional[dict] = None,
        magic_number: int = 0,
        agent_id: Optional[str] = None,
        image_tag: str = "latest",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        environment_vars: Optional[dict] = None,
        resource_limits: Optional[dict] = None,
        paper_config: Optional[PaperTradingConfig] = None,
    ) -> AgentDeploymentResult:
        """
        Deploy a new paper trading agent as a Docker container.

        Args:
            strategy_name: Name of the trading strategy
            strategy_code: Strategy Python code or template reference
            config: Strategy configuration parameters
            mt5_credentials: MT5 account credentials (optional for pure demo mode)
            magic_number: Unique magic number for trade identification
            agent_id: Custom agent ID (auto-generated if not provided)
            image_tag: Docker image tag
            redis_host: Redis host for event publishing
            redis_port: Redis port
            redis_db: Redis database number
            environment_vars: Additional environment variables
            resource_limits: Container resource limits (memory, cpus)
            paper_config: Paper trading configuration. If provided with broker_connection=False,
                         enables pure demo mode without broker credentials.

        Returns:
            AgentDeploymentResult with deployment details

        Raises:
            RuntimeError: If deployment fails
            ValueError: If invalid parameters provided
        """
        try:
            client = self._get_client()

            # Generate agent ID if not provided
            if agent_id is None:
                agent_id = self._generate_agent_id(strategy_name, magic_number)

            container_name = self._generate_container_name(agent_id)

            # Build full image name
            image_name = f"{BASE_IMAGE}:{image_tag}"

            # Prepare environment variables
            env = {
                "PYTHONUNBUFFERED": "1",
                "AGENT_ID": agent_id,
                "STRATEGY_NAME": strategy_name,
                "STRATEGY_CODE": strategy_code,
                "STRATEGY_CONFIG": json.dumps(config),
                "MAGIC_NUMBER": str(magic_number),
                "REDIS_HOST": redis_host,
                "REDIS_PORT": str(redis_port),
                "REDIS_DB": str(redis_db),
                "LOG_LEVEL": "INFO",
            }

            # Handle paper trading mode
            # If paper_config.broker_connection is False, use pure demo mode without broker credentials
            use_pure_demo = False
            if paper_config is not None:
                use_pure_demo = not paper_config.broker_connection
                env["PAPER_VIRTUAL_BALANCE"] = str(paper_config.virtual_balance)
                env["PAPER_USE_LIVE_DATA"] = str(paper_config.use_live_data).lower()
                env["PAPER_SIMULATE_SLIPPAGE"] = str(paper_config.simulate_slippage).lower()
                env["PAPER_SIMULATE_FEES"] = str(paper_config.simulate_fees).lower()
            
            # Add MT5 credentials only if broker connection is required
            # For pure demo mode (broker_connection=False), omit MT5 credentials
            if mt5_credentials and not use_pure_demo:
                env["MT5_ACCOUNT"] = str(mt5_credentials["account"])
                env["MT5_PASSWORD"] = mt5_credentials["password"]
                env["MT5_SERVER"] = mt5_credentials["server"]
            elif use_pure_demo:
                # Pure demo mode: Use virtual balance with live tick data only
                env["MT5_ACCOUNT"] = ""  # Empty for demo mode
                env["MT5_PASSWORD"] = ""
                env["MT5_SERVER"] = "MetaQuotes-Demo"  # Default demo server
                logger.info(f"Deploying in pure demo mode with virtual balance: {paper_config.virtual_balance if paper_config else 10000.0}")

            # Add custom environment variables
            if environment_vars:
                env.update(environment_vars)

            # Prepare container labels
            labels = {
                AGENT_LABEL: "true",
                "agent-id": agent_id,
                "strategy-name": strategy_name,
                "magic-number": str(magic_number),
                "managed-by": "quantmindx",
            }

            # Prepare resource limits
            deploy_config = {}
            if resource_limits:
                resources = {}
                if "memory" in resource_limits:
                    resources["mem_limit"] = resource_limits["memory"]
                if "cpus" in resource_limits:
                    resources["cpu_quota"] = int(float(resource_limits["cpus"]) * 100000)
                    resources["cpu_period"] = 100000
                if resources:
                    deploy_config["resources"] = resources

            # Pull image if not exists
            try:
                client.images.get(image_name)
            except NotFound:
                logger.info(f"Pulling image {image_name}...")
                client.images.pull(BASE_IMAGE, tag=image_tag)

            # Create and start container
            logger.info(f"Deploying agent {agent_id}...")
            container: Container = client.containers.run(
                image_name,
                name=container_name,
                detach=True,
                environment=env,
                labels=labels,
                **deploy_config,
                # Security settings
                security_opt=["no-new-privileges:true"],
                # Auto-remove on exit (optional, can be disabled)
                # auto_remove=True,
            )

            # Wait a moment for container to start
            time.sleep(0.5)

            # Verify container is running
            container.reload()
            actual_status = container.status.lower()

            logger.info(f"Agent {agent_id} deployed. Container ID: {container.id[:12]}")
            logger.info(f"Container status: {actual_status}")

            # Determine agent status
            if actual_status == "running":
                agent_status = AgentStatus.RUNNING
            elif actual_status == "created":
                agent_status = AgentStatus.STARTING
            else:
                agent_status = AgentStatus.ERROR

            return AgentDeploymentResult(
                agent_id=agent_id,
                container_id=container.id,
                container_name=container_name,
                status=agent_status,
                redis_channel=f"agent:heartbeat:{agent_id}",
                logs_url=f"docker logs -f {container_name}",
                message=f"Agent deployed successfully (status: {actual_status})",
            )

        except APIError as e:
            logger.error(f"Docker API error during deployment: {e}")
            raise RuntimeError(f"Deployment failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during deployment: {e}")
            raise RuntimeError(f"Deployment failed: {e}") from e

    def deploy_demo_agent(
        self,
        strategy_name: str,
        strategy_code: str,
        config: dict,
        virtual_balance: float = 10000.0,
        magic_number: int = 0,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> AgentDeploymentResult:
        """
        Deploy a paper trading agent in pure demo mode (no broker credentials).
        
        This is a convenience method that sets up paper_config for demo-only mode.
        The agent will use virtual balance with live tick data.
        
        Args:
            strategy_name: Name of the trading strategy
            strategy_code: Strategy Python code or template reference
            config: Strategy configuration parameters
            virtual_balance: Starting virtual balance (default: $10,000)
            magic_number: Unique magic number for trade identification
            agent_id: Custom agent ID (auto-generated if not provided)
            **kwargs: Additional arguments passed to deploy_agent
            
        Returns:
            AgentDeploymentResult with deployment details
            
        Example:
            ```python
            deployer = PaperTradingDeployer()
            
            # Deploy in pure demo mode
            result = deployer.deploy_demo_agent(
                strategy_name="RSI Reversal",
                strategy_code="template:rsi-reversal",
                config={"rsi_period": 14},
                virtual_balance=10000.0,
                magic_number=98765432
            )
            ```
        """
        demo_config = PaperTradingConfig(
            broker_connection=False,  # Pure demo mode
            virtual_balance=virtual_balance,
            use_live_data=True,
            simulate_slippage=True,
            simulate_fees=False,
        )
        
        return self.deploy_agent(
            strategy_name=strategy_name,
            strategy_code=strategy_code,
            config=config,
            mt5_credentials=None,  # No credentials needed for demo mode
            magic_number=magic_number,
            agent_id=agent_id,
            paper_config=demo_config,
            **kwargs
        )

    # ========================================================================
    # List Agents
    # ========================================================================

    def list_agents(self) -> list[PaperAgentStatus]:
        """
        List all paper trading agent containers.

        Returns:
            List of PaperAgentStatus for all agents
        """
        try:
            client = self._get_client()

            # Get all containers with our label
            containers = client.containers.list(
                all=True,  # Include stopped containers
                filters={"label": AGENT_LABEL}
            )

            agents = []
            for container in containers:
                agent_status = self._container_to_status(container)
                agents.append(agent_status)

            logger.debug(f"Found {len(agents)} paper trading agents")
            return agents

        except DockerException as e:
            logger.error(f"Failed to list agents: {e}")
            return []

    def get_agent(self, agent_id: str) -> Optional[PaperAgentStatus]:
        """
        Get status of a specific agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            PaperAgentStatus if found, None otherwise
        """
        try:
            client = self._get_client()
            container_name = self._generate_container_name(agent_id)

            container = client.containers.get(container_name)
            return self._container_to_status(container)

        except NotFound:
            logger.warning(f"Agent {agent_id} not found")
            return None
        except DockerException as e:
            logger.error(f"Failed to get agent {agent_id}: {e}")
            return None

    def _container_to_status(self, container: Container) -> PaperAgentStatus:
        """Convert Docker container to PaperAgentStatus."""
        labels = container.labels or {}
        env = dict(container.attrs.get("Config", {}).get("Env", []))

        # Parse environment variables
        def get_env_value(key: str, default=""):
            for e in env:
                if e.startswith(f"{key}="):
                    return e[len(key):]
            return default

        agent_id = labels.get("agent-id", get_env_value("AGENT_ID"))
        strategy_name = labels.get("strategy-name", get_env_value("STRATEGY_NAME"))

        # Parse MT5 account from env
        mt5_account_str = get_env_value("MT5_ACCOUNT")
        mt5_account = int(mt5_account_str) if mt5_account_str and mt5_account_str.isdigit() else None

        # Parse magic number
        magic_str = labels.get("magic-number", get_env_value("MAGIC_NUMBER"))
        magic_number = int(magic_str) if magic_str and magic_str.isdigit() else None

        # Parse symbol and timeframe from STRATEGY_CONFIG
        symbol = None
        timeframe = None
        strategy_config_str = get_env_value("STRATEGY_CONFIG")
        if strategy_config_str:
            try:
                strategy_config = json.loads(strategy_config_str)
                symbol = strategy_config.get("symbol")
                timeframe = strategy_config.get("timeframe")
            except (json.JSONDecodeError, ValueError, AttributeError):
                pass

        # Determine agent status from container status
        container_status = container.status.lower()
        status_map = {
            "running": AgentStatus.RUNNING,
            "created": AgentStatus.STARTING,
            "paused": AgentStatus.STOPPED,
            "restarting": AgentStatus.STARTING,
            "exited": AgentStatus.STOPPED,
            "dead": AgentStatus.ERROR,
            "removing": AgentStatus.STOPPING,
        }
        status = status_map.get(container_status, AgentStatus.ERROR)

        # Calculate uptime
        uptime_seconds = None
        if status == AgentStatus.RUNNING and container.attrs.get("State", {}).get("StartedAt"):
            started_at = container.attrs["State"]["StartedAt"]
            # Parse ISO timestamp and calculate uptime
            try:
                start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                uptime_seconds = int((datetime.now(UTC) - start_time).total_seconds())
            except Exception:
                pass

        # Image name
        image_name = container.attrs.get("Config", {}).get("Image", "")

        # Created timestamp
        created_str = container.attrs.get("Created")
        created_at = None
        if created_str:
            try:
                created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except Exception:
                pass

        # Started timestamp
        started_at = None
        if status == AgentStatus.RUNNING:
            started_at_str = container.attrs.get("State", {}).get("StartedAt")
            if started_at_str:
                try:
                    started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
                except Exception:
                    pass

        return PaperAgentStatus(
            agent_id=agent_id or "unknown",
            container_id=container.id,
            container_name=container.name,
            status=status,
            health=AgentHealth.HEALTHY,  # Will be updated by monitor
            strategy_name=strategy_name or "Unknown",
            symbol=symbol,
            timeframe=timeframe,
            mt5_account=mt5_account,
            mt5_server=get_env_value("MT5_SERVER"),
            magic_number=magic_number,
            redis_channel=f"agent:heartbeat:{agent_id}" if agent_id else "",
            created_at=created_at or datetime.now(UTC),
            started_at=started_at,
            uptime_seconds=uptime_seconds,
            image_name=image_name,
        )

    # ========================================================================
    # Stop Agent
    # ========================================================================

    def stop_agent(
        self,
        agent_id: str,
        timeout: int = GRACEFUL_SHUTDOWN_TIMEOUT,
        force: bool = False,
    ) -> bool:
        """
        Stop a paper trading agent gracefully.

        Args:
            agent_id: Agent identifier
            timeout: Graceful shutdown timeout in seconds
            force: If True, use SIGKILL immediately

        Returns:
            True if stopped successfully
        """
        try:
            client = self._get_client()
            container_name = self._generate_container_name(agent_id)

            container = client.containers.get(container_name)

            logger.info(f"Stopping agent {agent_id} (force={force})...")

            if force:
                # Immediate kill
                container.kill()
                logger.info(f"Agent {agent_id} force-killed")
            else:
                # Graceful shutdown
                container.stop(timeout=timeout)
                logger.info(f"Agent {agent_id} stopped gracefully")

            return True

        except NotFound:
            logger.warning(f"Agent {agent_id} not found (may already be stopped)")
            return True  # Consider it success
        except APIError as e:
            logger.error(f"Failed to stop agent {agent_id}: {e}")
            return False

    def remove_agent(
        self,
        agent_id: str,
        force: bool = False,
    ) -> bool:
        """
        Stop and remove a paper trading agent container.

        Args:
            agent_id: Agent identifier
            force: If True, force removal

        Returns:
            True if removed successfully
        """
        try:
            client = self._get_client()
            container_name = self._generate_container_name(agent_id)

            container = client.containers.get(container_name)

            logger.info(f"Removing agent {agent_id}...")

            container.remove(force=force)
            logger.info(f"Agent {agent_id} removed")

            return True

        except NotFound:
            logger.warning(f"Agent {agent_id} not found (may already be removed)")
            return True
        except APIError as e:
            logger.error(f"Failed to remove agent {agent_id}: {e}")
            return False

    # ========================================================================
    # Get Logs
    # ========================================================================

    def get_agent_logs(
        self,
        agent_id: str,
        tail_lines: int = 100,
        follow: bool = False,
        since: Optional[datetime] = None,
    ) -> AgentLogsResult:
        """
        Retrieve logs from a paper trading agent container.

        Args:
            agent_id: Agent identifier
            tail_lines: Number of lines to retrieve from end
            follow: If True, stream new logs (not supported in MCP)
            since: Only return logs since this timestamp

        Returns:
            AgentLogsResult with log lines

        Raises:
            ValueError: If agent not found
        """
        try:
            client = self._get_client()
            container_name = self._generate_container_name(agent_id)

            container = client.containers.get(container_name)

            # Validate tail_lines
            if tail_lines > self.MAX_LOG_LINES:
                logger.warning(f"tail_lines {tail_lines} exceeds max {self.MAX_LOG_LINES}, truncating")
                tail_lines = self.MAX_LOG_LINES

            # Get logs
            logs_bytes = container.logs(
                tail=tail_lines,
                timestamps=False,  # We'll add our own timestamps
                follow=False,
                since=since,
            )

            # Decode and split into lines
            logs_text = logs_bytes.decode("utf-8", errors="replace")
            log_lines = logs_text.splitlines()

            return AgentLogsResult(
                agent_id=agent_id,
                logs=log_lines,
                line_count=len(log_lines),
                tail_lines=tail_lines,
                has_more=False,  # Docker doesn't tell us if there's more
            )

        except NotFound:
            logger.error(f"Agent {agent_id} not found")
            raise ValueError(f"Agent {agent_id} not found") from None
        except APIError as e:
            logger.error(f"Failed to get logs for agent {agent_id}: {e}")
            raise RuntimeError(f"Failed to get logs: {e}") from e

    # ========================================================================
    # Health Check
    # ========================================================================

    def is_agent_running(self, agent_id: str) -> bool:
        """
        Check if an agent container is running.

        Args:
            agent_id: Agent identifier

        Returns:
            True if container is running
        """
        try:
            client = self._get_client()
            container_name = self._generate_container_name(agent_id)
            container = client.containers.get(container_name)
            container.reload()
            return container.status.lower() == "running"
        except (NotFound, DockerException):
            return False

    def get_container_stats(self, agent_id: str) -> Optional[dict]:
        """
        Get container resource usage statistics.

        Args:
            agent_id: Agent identifier

        Returns:
            Dict with CPU, memory, etc. or None
        """
        try:
            client = self._get_client()
            container_name = self._generate_container_name(agent_id)
            container = client.containers.get(container_name)
            return container.stats(stream=False)
        except (NotFound, DockerException) as e:
            logger.error(f"Failed to get stats for {agent_id}: {e}")
            return None

    def close(self):
        """Close Docker client connection."""
        if self._docker_client:
            try:
                self._docker_client.close()
            except Exception:
                pass
            self._docker_client = None
