"""
Grafana Cloud Metrics Pusher

Pushes Prometheus metrics to Grafana Cloud's remote write endpoint.
This is used when self-hosted Prometheus is not available.

Grafana Cloud provides:
- Free tier: 10K active series, 14-day retention
- Remote write endpoint for Prometheus metrics
- Managed Prometheus and Loki

Implementation uses proper remote_write protocol:
- Protobuf WriteRequest messages
- Snappy compression
- Content-Type: application/x-protobuf
- X-Scope-OrgID header for multi-tenancy
"""

import logging
import os
import threading
import time
from typing import Optional, List, Tuple

import requests
from prometheus_client import CollectorRegistry, REGISTRY
from prometheus_client.metrics_core import CounterMetricFamily, GaugeMetricFamily, SummaryMetricFamily, HistogramMetricFamily, UntypedMetricFamily

logger = logging.getLogger(__name__)

# Try to import snappy for compression
try:
    import snappy
    SNAPPY_AVAILABLE = True
except ImportError:
    SNAPPY_AVAILABLE = False
    logger.warning("python-snappy not installed. Install with: pip install python-snappy")

# Try to import prometheus remote write protobuf
try:
    from prometheus_remote_write import WriteRequest, Label, Sample, TimeSeries
    PROTOBUF_AVAILABLE = True
except ImportError:
    # Fall back to manual protobuf encoding
    PROTOBUF_AVAILABLE = False
    try:
        # Try prometheus_client's internal protobuf support
        from prometheus_client.exposition import _protobuf
        PROTOBUF_AVAILABLE = True
    except ImportError:
        logger.warning("Protobuf support not available. Install prometheus-remote-write package.")


class RemoteWriteEncoder:
    """
    Encodes Prometheus metrics into remote_write protobuf format.
    
    This implements the Prometheus remote_write protocol:
    1. Collect metrics from registry
    2. Convert to protobuf TimeSeries
    3. Create WriteRequest
    4. Snappy compress
    """
    
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self.registry = registry
    
    def encode(self) -> bytes:
        """
        Encode all metrics from the registry into Snappy-compressed protobuf.
        
        Returns:
            Snappy-compressed WriteRequest protobuf bytes
        """
        # Build TimeSeries from registry
        timeseries_list = []
        
        for metric in self.registry.collect():
            timeseries_list.extend(self._metric_to_timeseries(metric))
        
        # Create WriteRequest protobuf
        write_request = self._create_write_request(timeseries_list)
        
        # Serialize and compress
        protobuf_data = write_request.SerializeToString()
        
        if SNAPPY_AVAILABLE:
            return snappy.compress(protobuf_data)
        else:
            # Without snappy, return uncompressed (may not work with Grafana Cloud)
            logger.warning("Snappy compression not available, sending uncompressed data")
            return protobuf_data
    
    def _metric_to_timeseries(self, metric) -> list:
        """Convert a Prometheus metric to TimeSeries protobuf messages."""
        timeseries_list = []
        
        if isinstance(metric, (CounterMetricFamily, GaugeMetricFamily, UntypedMetricFamily)):
            for sample in metric.samples:
                ts = self._create_timeseries(
                    name=sample.name,
                    labels=sample.labels,
                    value=sample.value,
                    timestamp=sample.timestamp
                )
                timeseries_list.append(ts)
        
        elif isinstance(metric, SummaryMetricFamily):
            for sample in metric.samples:
                ts = self._create_timeseries(
                    name=sample.name,
                    labels=sample.labels,
                    value=sample.value,
                    timestamp=sample.timestamp
                )
                timeseries_list.append(ts)
        
        elif isinstance(metric, HistogramMetricFamily):
            for sample in metric.samples:
                ts = self._create_timeseries(
                    name=sample.name,
                    labels=sample.labels,
                    value=sample.value,
                    timestamp=sample.timestamp
                )
                timeseries_list.append(ts)
        
        return timeseries_list
    
    def _create_timeseries(self, name: str, labels: dict, value: float, timestamp: Optional[float] = None):
        """Create a TimeSeries protobuf message."""
        # Use prometheus_remote_write package if available
        if PROTOBUF_AVAILABLE and 'TimeSeries' in dir():
            # Build labels
            label_protos = []
            # Add __name__ label first (convention)
            label_protos.append(Label(name='__name__', value=name))
            for label_name, label_value in sorted(labels.items()):
                label_protos.append(Label(name=label_name, value=str(label_value)))
            
            # Build sample with timestamp in milliseconds
            ts_ms = int(timestamp * 1000) if timestamp else int(time.time() * 1000)
            sample = Sample(value=float(value), timestamp=ts_ms)
            
            return TimeSeries(labels=label_protos, samples=[sample])
        else:
            # Return dict representation for manual protobuf encoding
            return {
                'labels': [('__name__', name)] + [(k, str(v)) for k, v in sorted(labels.items())],
                'samples': [(int(timestamp * 1000) if timestamp else int(time.time() * 1000), float(value))]
            }
    
    def _create_write_request(self, timeseries_list: list):
        """Create WriteRequest protobuf from TimeSeries list."""
        if PROTOBUF_AVAILABLE and 'WriteRequest' in dir():
            return WriteRequest(timeseries=timeseries_list)
        else:
            # Manual protobuf encoding using prometheus_client internals
            return self._manual_write_request(timeseries_list)
    
    def _manual_write_request(self, timeseries_list: list) -> bytes:
        """
        Manually encode WriteRequest protobuf.
        
        This is a fallback when prometheus_remote_write package is not available.
        Uses the prometheus_client internal protobuf support.
        """
        try:
            from prometheus_client.exposition import _protobuf as pb
            return pb.encode_write_request(timeseries_list)
        except Exception as e:
            logger.error(f"Failed to encode WriteRequest: {e}")
            raise


class GrafanaCloudPusher:
    """
    Pushes metrics to Grafana Cloud Prometheus remote write endpoint.
    
    Environment Variables:
        GRAFANA_PROMETHEUS_URL: Grafana Cloud Prometheus URL
        GRAFANA_LOKI_URL: Grafana Cloud Loki URL
        GRAFANA_INSTANCE_ID: Grafana Cloud instance ID
        GRAFANA_API_KEY: Grafana Cloud API key
    
    Usage:
        pusher = GrafanaCloudPusher()
        pusher.start_background_push(interval_seconds=15)
    """
    
    def __init__(
        self,
        prometheus_url: Optional[str] = None,
        instance_id: Optional[str] = None,
        api_key: Optional[str] = None,
        registry: CollectorRegistry = REGISTRY
    ):
        """
        Initialize Grafana Cloud pusher.
        
        Args:
            prometheus_url: Grafana Cloud Prometheus URL (default: from env)
            instance_id: Grafana Cloud instance ID (default: from env)
            api_key: Grafana Cloud API key (default: from env)
            registry: Prometheus registry to use
        """
        self.prometheus_url = prometheus_url or os.getenv('GRAFANA_PROMETHEUS_URL', '')
        self.instance_id = instance_id or os.getenv('GRAFANA_INSTANCE_ID', '')
        self.api_key = api_key or os.getenv('GRAFANA_API_KEY', '')
        self.registry = registry
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._encoder = RemoteWriteEncoder(registry)
        
        if not all([self.prometheus_url, self.instance_id, self.api_key]):
            logger.warning(
                "Grafana Cloud credentials not fully configured. "
                "Set GRAFANA_PROMETHEUS_URL, GRAFANA_INSTANCE_ID, and GRAFANA_API_KEY."
            )
        
        # Check dependencies
        if not SNAPPY_AVAILABLE:
            logger.error(
                "python-snappy is required for Grafana Cloud remote_write. "
                "Install with: pip install python-snappy"
            )
    
    def is_configured(self) -> bool:
        """Check if Grafana Cloud is properly configured."""
        return all([self.prometheus_url, self.instance_id, self.api_key])
    
    def push_metrics(self) -> bool:
        """
        Push current metrics to Grafana Cloud using proper remote_write protocol.
        
        Uses protobuf WriteRequest with Snappy compression as required by
        Grafana Cloud's remote_write endpoint.
        
        Returns:
            True if push successful, False otherwise
        """
        if not self.is_configured():
            logger.debug("Grafana Cloud not configured, skipping push")
            return False
        
        if not SNAPPY_AVAILABLE:
            logger.error("Cannot push metrics: python-snappy not installed")
            return False
        
        try:
            # Construct the remote write URL
            # Grafana Cloud uses /api/prom/push for Prometheus remote write
            url = f"{self.prometheus_url.rstrip('/')}/api/prom/push"
            
            # Encode metrics as Snappy-compressed protobuf WriteRequest
            metrics_data = self._encoder.encode()
            
            # Set up headers for remote_write protocol
            headers = {
                'Content-Type': 'application/x-protobuf',
                'Content-Encoding': 'snappy',
                'X-Scope-OrgID': self.instance_id,
                'User-Agent': 'QuantMind-GrafanaCloud-Pusher/1.0'
            }
            
            # Push to Grafana Cloud with HTTP basic authentication
            response = requests.post(
                url,
                data=metrics_data,
                headers=headers,
                auth=(self.instance_id, self.api_key),
                timeout=30
            )
            
            if response.status_code in (200, 204):
                logger.debug("Successfully pushed metrics to Grafana Cloud via remote_write")
                return True
            else:
                logger.warning(
                    f"Failed to push metrics to Grafana Cloud: "
                    f"status={response.status_code}, body={response.text[:200]}"
                )
                return False
                
        except requests.exceptions.Timeout:
            logger.warning("Timeout pushing metrics to Grafana Cloud")
            return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error pushing metrics to Grafana Cloud: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error pushing metrics: {e}")
            return False
    
    def start_background_push(self, interval_seconds: int = 15) -> None:
        """
        Start a background thread to push metrics periodically.
        
        Args:
            interval_seconds: Interval between pushes (default: 15)
        """
        if self._running:
            logger.warning("Background push already running")
            return
        
        if not self.is_configured():
            logger.warning("Grafana Cloud not configured, not starting background push")
            return
        
        if not SNAPPY_AVAILABLE:
            logger.error("Cannot start background push: python-snappy not installed")
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._push_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._thread.start()
        logger.info(f"Started Grafana Cloud metrics pusher (interval: {interval_seconds}s)")
    
    def stop_background_push(self) -> None:
        """Stop the background push thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Stopped Grafana Cloud metrics pusher")
    
    def _push_loop(self, interval_seconds: int) -> None:
        """Background loop for pushing metrics."""
        while self._running:
            try:
                self.push_metrics()
            except Exception as e:
                logger.error(f"Error in push loop: {e}")
            
            # Sleep in small increments to allow for clean shutdown
            for _ in range(interval_seconds):
                if not self._running:
                    break
                time.sleep(1)


# Global pusher instance
_global_pusher: Optional[GrafanaCloudPusher] = None


def get_grafana_cloud_pusher() -> GrafanaCloudPusher:
    """Get or create the global GrafanaCloudPusher instance."""
    global _global_pusher
    if _global_pusher is None:
        _global_pusher = GrafanaCloudPusher()
    return _global_pusher


def start_grafana_cloud_push(interval_seconds: int = 15) -> None:
    """
    Start pushing metrics to Grafana Cloud.
    
    This is a convenience function that uses the global pusher instance.
    
    Args:
        interval_seconds: Interval between pushes (default: 15)
    """
    pusher = get_grafana_cloud_pusher()
    pusher.start_background_push(interval_seconds)


def stop_grafana_cloud_push() -> None:
    """Stop pushing metrics to Grafana Cloud."""
    global _global_pusher
    if _global_pusher:
        _global_pusher.stop_background_push()
