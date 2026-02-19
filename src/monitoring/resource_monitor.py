"""
Resource Monitor

Monitors system resources and exposes metrics via Prometheus.
- CPU usage (per process)
- RAM usage (per process)
- Network I/O
- Disk I/O
- Tick stream rate

Alert thresholds:
- CPU: 70% warning, 85% critical, 95% emergency
- RAM: 75% warning, 90% critical, 95% emergency
- Tick latency: 10ms warning, 50ms critical
"""

import asyncio
import time
import logging
import psutil
from typing import Dict, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger(__name__)


class AlertLevel(IntEnum):
    """Resource alert levels."""
    NORMAL = 0
    WARNING = 1
    CRITICAL = 2
    EMERGENCY = 3


@dataclass
class ResourceMetrics:
    """Current resource metrics."""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_percent: float = 0.0
    cpu_count: int = 0
    ram_mb: float = 0.0
    ram_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    tick_stream_rate: float = 0.0
    tick_latency_ms: float = 0.0
    active_symbols: int = 0


@dataclass
class AlertThresholds:
    """Configurable alert thresholds."""
    cpu_warning: float = 70.0
    cpu_critical: float = 85.0
    cpu_emergency: float = 95.0
    ram_warning: float = 75.0
    ram_critical: float = 90.0
    ram_emergency: float = 95.0
    tick_latency_warning_ms: float = 10.0
    tick_latency_critical_ms: float = 50.0


class ResourceMonitor:
    """
    Monitors system resources and provides metrics for Prometheus.
    
    Features:
    - Real-time CPU, RAM, disk, network monitoring
    - Tick stream rate and latency tracking
    - Configurable alert thresholds
    - Prometheus-compatible metrics export
    """
    
    def __init__(self, process_name: Optional[str] = None):
        self._process_name = process_name
        self._thresholds = AlertThresholds()
        self._metrics_history: list[ResourceMetrics] = []
        self._max_history = 1000  # Keep last 1000 samples
        
        # Tick tracking
        self._tick_count = 0
        self._tick_count_start = time.time()
        self._last_tick_time: Dict[str, float] = {}
        
        # Current state
        self._current_level = AlertLevel.NORMAL
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._on_alert: Optional[callable] = None
        
        # Get process handle if name provided
        self._process = None
        if process_name:
            for proc in psutil.process_iter(['name']):
                try:
                    if proc.info['name'] == process_name:
                        self._process = proc
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    
    def set_alert_callback(self, callback: callable):
        """Set callback for alert events."""
        self._on_alert = callback
    
    def set_thresholds(self, thresholds: AlertThresholds):
        """Update alert thresholds."""
        self._thresholds = thresholds
    
    async def start(self):
        """Start the resource monitor."""
        if self._is_running:
            return
        
        self._is_running = True
        self._monitor_task = asyncio.create_task(self._run_monitor())
        logger.info("Resource monitor started")
    
    async def stop(self):
        """Stop the resource monitor."""
        self._is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Resource monitor stopped")
    
    async def _run_monitor(self):
        """Main monitoring loop."""
        while self._is_running:
            try:
                metrics = await self._collect_metrics()
                self._metrics_history.append(metrics)
                
                # Trim history
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history = self._metrics_history[-self._max_history:]
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Log every 30 seconds
                logger.debug(
                    f"Resources: CPU={metrics.cpu_percent:.1f}%, "
                    f"RAM={metrics.ram_mb:.0f}MB, "
                    f"TickRate={metrics.tick_stream_rate:.1f}/s"
                )
                
                await asyncio.sleep(10)  # Collect every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        ram = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        # Calculate tick stream rate
        current_time = time.time()
        elapsed = current_time - self._tick_count_start
        if elapsed > 0:
            tick_rate = self._tick_count / elapsed
        else:
            tick_rate = 0.0
        
        # Calculate average latency from history
        latency_ms = 0.0
        if len(self._metrics_history) > 0:
            recent = self._metrics_history[-10:]
            latencies = [m.tick_latency_ms for m in recent if m.tick_latency_ms > 0]
            if latencies:
                latency_ms = sum(latencies) / len(latencies)
        
        # Count active symbols
        active_symbols = len(self._last_tick_time)
        
        return ResourceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            cpu_count=cpu_count,
            ram_mb=ram.used / (1024 * 1024),
            ram_percent=ram.percent,
            disk_io_read_mb=disk_io.read_bytes / (1024 * 1024) if disk_io else 0,
            disk_io_write_mb=disk_io.write_bytes / (1024 * 1024) if disk_io else 0,
            network_sent_mb=net_io.bytes_sent / (1024 * 1024) if net_io else 0,
            network_recv_mb=net_io.bytes_recv / (1024 * 1024) if net_io else 0,
            tick_stream_rate=tick_rate,
            tick_latency_ms=latency_ms,
            active_symbols=active_symbols
        )
    
    async def _check_alerts(self, metrics: ResourceMetrics):
        """Check metrics against thresholds and trigger alerts."""
        new_level = AlertLevel.NORMAL
        
        # CPU check
        if metrics.cpu_percent >= self._thresholds.cpu_emergency:
            new_level = AlertLevel.EMERGENCY
        elif metrics.cpu_percent >= self._thresholds.cpu_critical:
            new_level = AlertLevel.CRITICAL
        elif metrics.cpu_percent >= self._thresholds.cpu_warning:
            new_level = AlertLevel.WARNING
        
        # RAM check (escalate if higher)
        if metrics.ram_percent >= self._thresholds.ram_emergency and new_level < AlertLevel.EMERGENCY:
            new_level = AlertLevel.EMERGENCY
        elif metrics.ram_percent >= self._thresholds.ram_critical and new_level < AlertLevel.CRITICAL:
            new_level = AlertLevel.CRITICAL
        elif metrics.ram_percent >= self._thresholds.ram_warning and new_level < AlertLevel.WARNING:
            new_level = AlertLevel.WARNING
        
        # Tick latency check
        if metrics.tick_latency_ms >= self._thresholds.tick_latency_critical and new_level < AlertLevel.CRITICAL:
            new_level = AlertLevel.CRITICAL
        elif metrics.tick_latency_ms >= self._thresholds.tick_latency_warning_ms and new_level < AlertLevel.WARNING:
            new_level = AlertLevel.WARNING
        
        # Alert if level changed
        if new_level != self._current_level:
            self._current_level = new_level
            if self._on_alert:
                await self._on_alert(new_level, metrics)
            
            if new_level >= AlertLevel.CRITICAL:
                logger.warning(
                    f"Resource alert: {new_level.name} - "
                    f"CPU={metrics.cpu_percent:.1f}%, RAM={metrics.ram_percent:.1f}%"
                )
    
    def record_tick(self, symbol: str):
        """Record a tick for rate calculation."""
        self._tick_count += 1
        self._last_tick_time[symbol] = time.time()
        
        # Clean up old symbols
        current_time = time.time()
        stale = [s for s, t in self._last_tick_time.items() if current_time - t > 60]
        for s in stale:
            self._last_tick_time.pop(s, None)
    
    def record_latency(self, latency_ms: float):
        """Record tick latency."""
        if len(self._metrics_history) > 0:
            self._metrics_history[-1].tick_latency_ms = latency_ms
    
    def get_current_metrics(self) -> ResourceMetrics:
        """Get the most recent metrics."""
        if self._metrics_history:
            return self._metrics_history[-1]
        return ResourceMetrics()
    
    def get_alert_level(self) -> AlertLevel:
        """Get current alert level."""
        return self._current_level
    
    def get_metrics_history(self, count: int = 100) -> list[ResourceMetrics]:
        """Get recent metrics history."""
        return self._metrics_history[-count:]
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output."""
        metrics = self.get_current_metrics()
        
        output = []
        output.append(f"# HELP quantmind_mt5_cpu_percent CPU usage percentage")
        output.append(f"# TYPE quantmind_mt5_cpu_percent gauge")
        output.append(f"quantmind_mt5_cpu_percent {metrics.cpu_percent}")
        
        output.append(f"# HELP quantmind_mt5_ram_mb RAM usage in MB")
        output.append(f"# TYPE quantmind_mt5_ram_mb gauge")
        output.append(f"quantmind_mt5_ram_mb {metrics.ram_mb}")
        
        output.append(f"# HELP quantmind_mt5_active_symbols Number of active symbols")
        output.append(f"# TYPE quantmind_mt5_active_symbols gauge")
        output.append(f"quantmind_mt5_active_symbols {metrics.active_symbols}")
        
        output.append(f"# HELP quantmind_tick_stream_rate Tick stream rate per second")
        output.append(f"# TYPE quantmind_tick_stream_rate gauge")
        output.append(f"quantmind_tick_stream_rate {metrics.tick_stream_rate}")
        
        output.append(f"# HELP quantmind_tick_stream_latency_ms Tick stream latency in milliseconds")
        output.append(f"# TYPE quantmind_tick_stream_latency_ms gauge")
        output.append(f"quantmind_tick_stream_latency_ms {metrics.tick_latency_ms}")
        
        output.append(f"# HELP quantmind_degradation_level Current degradation level")
        output.append(f"# TYPE quantmind_degradation_level gauge")
        output.append(f"quantmind_degradation_level {self._current_level}")
        
        return "\n".join(output)


# Global instance
_resource_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor(process_name: Optional[str] = None) -> ResourceMonitor:
    """Get or create the global resource monitor."""
    global _resource_monitor
    if _resource_monitor is None:
        _resource_monitor = ResourceMonitor(process_name)
    return _resource_monitor
