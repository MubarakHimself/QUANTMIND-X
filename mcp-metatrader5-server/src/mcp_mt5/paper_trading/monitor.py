"""
Agent Health Monitor for Paper Trading.

Monitors paper trading agents via Redis heartbeats and detects stale/dead agents.
Supports auto-restart and alerting.
"""

import json
import logging
import time
import threading
from collections import defaultdict
from datetime import datetime, UTC, timedelta
from typing import Optional, Callable, Dict, List

import redis
from redis.exceptions import RedisError

from .models import AgentHealth, PaperAgentStatus

logger = logging.getLogger(__name__)


class HeartbeatData:
    """Data extracted from heartbeat message."""

    def __init__(self, message_json: str):
        """Parse heartbeat JSON message."""
        data = json.loads(message_json)
        self.timestamp = data.get("timestamp", "")
        self.agent_id = data.get("agent_id", "")
        self.status = data.get("status", "unknown")
        self.uptime_seconds = data.get("uptime_seconds", 0)
        self.mt5_connected = data.get("mt5_connected", False)

    @property
    def parsed_timestamp(self) -> Optional[datetime]:
        """Parse timestamp string to datetime."""
        try:
            # Handle ISO 8601 with Z suffix
            ts = self.timestamp.replace("Z", "+00:00")
            return datetime.fromisoformat(ts)
        except Exception:
            return None


class AgentHealthMonitor:
    """
    Monitors paper trading agent health via Redis heartbeats.

    Features:
    - Subscribe to agent heartbeat channels
    - Detect stale agents (no heartbeat for 5 minutes)
    - Track missed heartbeats
    - Auto-restart dead agents (optional)
    - Callback for health status changes

    Expected heartbeat interval: 60 seconds
    Stale threshold: 5 minutes (5 missed heartbeats)
    Dead threshold: 10 minutes (10 missed heartbeats)

    Example:
        ```python
        def on_health_change(agent_id, old_health, new_health):
            print(f"Agent {agent_id}: {old_health} -> {new_health}")

        monitor = AgentHealthMonitor(
            redis_host="localhost",
            redis_port=6379,
            on_health_change=on_health_change
        )
        monitor.start()

        # Check agent health
        health = monitor.get_agent_health("strategy-rsi-001")
        ```
    """

    # Heartbeat interval in seconds
    HEARTBEAT_INTERVAL = 60

    # Stale threshold: 5 minutes without heartbeat
    STALE_THRESHOLD_SECONDS = 300

    # Dead threshold: 10 minutes without heartbeat
    DEAD_THRESHOLD_SECONDS = 600

    # Check interval for background monitoring
    MONITOR_CHECK_INTERVAL = 30

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        on_health_change: Optional[Callable[[str, AgentHealth, AgentHealth], None]] = None,
        on_agent_stale: Optional[Callable[[str], None]] = None,
        on_agent_dead: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the health monitor.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Optional Redis password
            on_health_change: Callback when agent health changes
            on_agent_stale: Callback when agent becomes stale
            on_agent_dead: Callback when agent becomes dead
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.on_health_change = on_health_change
        self.on_agent_stale = on_agent_stale
        self.on_agent_dead = on_agent_dead

        # Tracking data
        self._agent_health: Dict[str, AgentHealth] = {}
        self._last_heartbeat: Dict[str, datetime] = {}
        self._missed_heartbeats: Dict[str, int] = defaultdict(int)

        # Background thread
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._redis_client: Optional[redis.Redis] = None

    def _get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis_client is None:
            self._redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                password=self.redis_password,
                decode_responses=True,
            )
        return self._redis_client

    # ========================================================================
    # Health Status
    # ========================================================================

    def get_agent_health(self, agent_id: str) -> AgentHealth:
        """
        Get current health status of an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentHealth status
        """
        # Check if we have recent heartbeat data
        last_heartbeat = self._last_heartbeat.get(agent_id)
        if last_heartbeat is None:
            # No heartbeat recorded, check Redis
            health = self._check_redis_health(agent_id)
            return health

        # Calculate time since last heartbeat
        time_since = (datetime.now(UTC) - last_heartbeat).total_seconds()

        if time_since > self.DEAD_THRESHOLD_SECONDS:
            return AgentHealth.DEAD
        elif time_since > self.STALE_THRESHOLD_SECONDS:
            return AgentHealth.STALE
        else:
            return AgentHealth.HEALTHY

    def _check_redis_health(self, agent_id: str) -> AgentHealth:
        """
        Check agent health by querying Redis directly.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentHealth status
        """
        try:
            client = self._get_redis()

            # Check for latest heartbeat key
            key = f"agent:heartbeat:{agent_id}:latest"
            heartbeat_json = client.get(key)

            if heartbeat_json is None:
                # No heartbeat ever received
                return AgentHealth.DEAD

            # Parse heartbeat
            heartbeat = HeartbeatData(heartbeat_json)
            if heartbeat.parsed_timestamp is None:
                return AgentHealth.DEAD

            # Calculate age
            age = (datetime.now(UTC) - heartbeat.parsed_timestamp).total_seconds()

            # Update local tracking
            self._last_heartbeat[agent_id] = heartbeat.parsed_timestamp

            if age > self.DEAD_THRESHOLD_SECONDS:
                return AgentHealth.DEAD
            elif age > self.STALE_THRESHOLD_SECONDS:
                return AgentHealth.STALE
            else:
                return AgentHealth.HEALTHY

        except RedisError as e:
            logger.error(f"Redis error checking health for {agent_id}: {e}")
            return AgentHealth.DEAD
        except Exception as e:
            logger.error(f"Error checking health for {agent_id}: {e}")
            return AgentHealth.DEAD

    def get_last_heartbeat(self, agent_id: str) -> Optional[datetime]:
        """
        Get timestamp of last heartbeat for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Last heartbeat timestamp or None
        """
        if agent_id in self._last_heartbeat:
            return self._last_heartbeat[agent_id]

        # Check Redis
        try:
            client = self._get_redis()
            key = f"agent:heartbeat:{agent_id}:latest"
            heartbeat_json = client.get(key)

            if heartbeat_json:
                heartbeat = HeartbeatData(heartbeat_json)
                if heartbeat.parsed_timestamp:
                    self._last_heartbeat[agent_id] = heartbeat.parsed_timestamp
                    return heartbeat.parsed_timestamp

        except Exception as e:
            logger.error(f"Error getting last heartbeat for {agent_id}: {e}")

        return None

    def get_missed_heartbeat_count(self, agent_id: str) -> int:
        """
        Get number of consecutive missed heartbeats for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Number of missed heartbeats
        """
        last_heartbeat = self.get_last_heartbeat(agent_id)
        if last_heartbeat is None:
            return 999  # Effectively infinite

        time_since = (datetime.now(UTC) - last_heartbeat).total_seconds()
        return int(time_since // self.HEARTBEAT_INTERVAL)

    # ========================================================================
    # Update Health from Heartbeat
    # ========================================================================

    def record_heartbeat(self, agent_id: str, heartbeat_json: str) -> AgentHealth:
        """
        Record a heartbeat and update health status.

        Called when a heartbeat message is received.

        Args:
            agent_id: Agent identifier
            heartbeat_json: Heartbeat JSON message

        Returns:
            New health status
        """
        try:
            heartbeat = HeartbeatData(heartbeat_json)

            if heartbeat.parsed_timestamp:
                old_health = self._agent_health.get(agent_id, AgentHealth.HEALTHY)
                self._last_heartbeat[agent_id] = heartbeat.parsed_timestamp

                # Reset missed heartbeats
                self._missed_heartbeats[agent_id] = 0

                # Update health
                new_health = AgentHealth.HEALTHY

                if old_health != new_health:
                    self._agent_health[agent_id] = new_health
                    if self.on_health_change:
                        self.on_health_change(agent_id, old_health, new_health)

                return new_health

        except Exception as e:
            logger.error(f"Error recording heartbeat for {agent_id}: {e}")

        return AgentHealth.DEAD

    # ========================================================================
    # Background Monitoring
    # ========================================================================

    def start(self, check_interval: int = MONITOR_CHECK_INTERVAL):
        """
        Start background health monitoring thread.

        Args:
            check_interval: Check interval in seconds
        """
        if self._monitoring_thread is not None:
            logger.warning("Monitoring thread already running")
            return

        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            args=(check_interval,),
            daemon=True,
            name="AgentHealthMonitor",
        )
        self._monitoring_thread.start()
        logger.info(f"Started health monitoring (interval: {check_interval}s)")

    def stop(self):
        """Stop background monitoring thread."""
        if self._monitoring_thread is None:
            return

        self._stop_event.set()
        self._monitoring_thread.join(timeout=5)

        if self._monitoring_thread.is_alive():
            logger.warning("Monitoring thread did not stop gracefully")

        self._monitoring_thread = None
        logger.info("Stopped health monitoring")

    def _monitor_loop(self, check_interval: int):
        """Background monitoring loop."""
        logger.info("Health monitoring thread started")

        while not self._stop_event.is_set():
            try:
                self._check_all_agents()

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

            # Wait for next check or stop event
            self._stop_event.wait(check_interval)

        logger.info("Health monitoring thread stopped")

    def _check_all_agents(self):
        """Check health of all tracked agents."""
        now = datetime.now(UTC)

        # Get all agents from Redis keys
        try:
            client = self._get_redis()
            keys = client.keys("agent:heartbeat:*:latest")

            for key in keys:
                # Extract agent ID from key
                agent_id = key.replace("agent:heartbeat:", "").replace(":latest", "")

                # Get heartbeat
                heartbeat_json = client.get(key)
                if heartbeat_json:
                    try:
                        heartbeat = HeartbeatData(heartbeat_json)
                        if heartbeat.parsed_timestamp:
                            self._last_heartbeat[agent_id] = heartbeat.parsed_timestamp
                    except Exception:
                        pass

        except RedisError as e:
            logger.error(f"Redis error checking agents: {e}")
            return

        # Check each tracked agent
        agents_to_check = list(self._last_heartbeat.keys())

        for agent_id in agents_to_check:
            old_health = self._agent_health.get(agent_id, AgentHealth.HEALTHY)
            new_health = self.get_agent_health(agent_id)

            # Check for health changes
            if old_health != new_health:
                self._agent_health[agent_id] = new_health

                if self.on_health_change:
                    try:
                        self.on_health_change(agent_id, old_health, new_health)
                    except Exception as e:
                        logger.error(f"Error in health change callback: {e}")

                # Call specific callbacks
                if new_health == AgentHealth.STALE and self.on_agent_stale:
                    try:
                        self.on_agent_stale(agent_id)
                    except Exception as e:
                        logger.error(f"Error in stale callback: {e}")

                if new_health == AgentHealth.DEAD and self.on_agent_dead:
                    try:
                        self.on_agent_dead(agent_id)
                    except Exception as e:
                        logger.error(f"Error in dead callback: {e}")

    # ========================================================================
    # Batch Updates
    # ========================================================================

    def update_agent_statuses(self, agents: List[PaperAgentStatus]) -> List[PaperAgentStatus]:
        """
        Update health status for a list of agents.

        Args:
            agents: List of agent statuses

        Returns:
            Updated list with health status set
        """
        updated = []

        for agent in agents:
            health = self.get_agent_health(agent.agent_id)
            last_heartbeat = self.get_last_heartbeat(agent.agent_id)
            missed = self.get_missed_heartbeat_count(agent.agent_id)

            # Create updated copy
            agent_dict = agent.model_dump()
            agent_dict["health"] = health
            agent_dict["last_heartbeat"] = last_heartbeat
            agent_dict["missed_heartbeats"] = missed

            updated.append(PaperAgentStatus(**agent_dict))

        return updated

    # ========================================================================
    # Cleanup
    # ========================================================================

    def close(self):
        """Close monitor and cleanup resources."""
        self.stop()

        if self._redis_client:
            try:
                self._redis_client.close()
            except Exception:
                pass
            self._redis_client = None

        self._agent_health.clear()
        self._last_heartbeat.clear()
        self._missed_heartbeats.clear()
