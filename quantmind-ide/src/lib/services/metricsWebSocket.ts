/**
 * Metrics WebSocket Service
 *
 * Handles real-time metrics streaming from the backend monitoring system.
 * Connects to WebSocket endpoint and provides reconnection logic.
 */

import { writable, derived, get } from 'svelte/store';

// Types for metrics data
export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_in: number;
  network_out: number;
  uptime: number;
  chaos_score: number;
}

export interface TradingMetrics {
  tick_latency_ms: number;
  active_bots: number;
  active_positions: number;
  daily_pnl: number;
  total_trades: number;
  win_rate: number;
}

export interface DatabaseMetrics {
  query_latency_ms: number;
  connection_pool_size: number;
  active_connections: number;
  query_count: number;
}

export interface TickStreamMetrics {
  ticks_per_second: number;
  buffer_size: number;
  processing_time_ms: number;
  symbols_active: number;
}

export interface AlertData {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  source: string;
}

export interface MetricsUpdate {
  system: SystemMetrics;
  trading: TradingMetrics;
  database: DatabaseMetrics;
  tick_stream: TickStreamMetrics;
  alerts: AlertData[];
  timestamp: Date;
}

// Raw metrics payload from WebSocket (dates as strings)
interface RawMetricsPayload {
  system?: Partial<SystemMetrics>;
  trading?: Partial<TradingMetrics>;
  database?: Partial<DatabaseMetrics>;
  tick_stream?: Partial<TickStreamMetrics>;
}

// WebSocket message types
type MetricsMessageType = 'metrics' | 'alert' | 'pong';

interface BaseMetricsMessage {
  type: MetricsMessageType;
}

interface MetricsMessage extends BaseMetricsMessage {
  type: 'metrics';
  payload: RawMetricsPayload;
}

interface AlertMessage extends BaseMetricsMessage {
  type: 'alert';
  payload: AlertData & { timestamp: string };
}

interface PongMessage extends BaseMetricsMessage {
  type: 'pong';
}

type MetricsWebSocketMessage = MetricsMessage | AlertMessage | PongMessage;

// Connection state
type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

// Create stores for metrics
export const systemMetrics = writable<SystemMetrics>({
  cpu_usage: 0,
  memory_usage: 0,
  disk_usage: 0,
  network_in: 0,
  network_out: 0,
  uptime: 0,
  chaos_score: 0
});

export const tradingMetrics = writable<TradingMetrics>({
  tick_latency_ms: 0,
  active_bots: 0,
  active_positions: 0,
  daily_pnl: 0,
  total_trades: 0,
  win_rate: 0
});

export const databaseMetrics = writable<DatabaseMetrics>({
  query_latency_ms: 0,
  connection_pool_size: 0,
  active_connections: 0,
  query_count: 0
});

export const tickStreamMetrics = writable<TickStreamMetrics>({
  ticks_per_second: 0,
  buffer_size: 0,
  processing_time_ms: 0,
  symbols_active: 0
});

export const alerts = writable<AlertData[]>([]);

export const connectionState = writable<ConnectionState>('disconnected');
export const lastUpdate = writable<Date>(new Date());

// Time series data stores for charts
export const tickRateHistory = writable<Array<{ time: Date; value: number }>>([]);
export const cpuHistory = writable<Array<{ time: Date; value: number }>>([]);
export const memoryHistory = writable<Array<{ time: Date; value: number }>>([]);
export const latencyHistory = writable<Array<{ time: Date; value: number }>>([]);

// Maximum history length (1 hour at 1s intervals to support all time ranges)
const MAX_HISTORY_LENGTH = 3600;

class MetricsWebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private pongTimeout: ReturnType<typeof setTimeout> | null = null;
  private pongTimeoutDuration = 10000;
  private lastPongTime = 0;

  private wsUrl: string;

  constructor() {
    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = import.meta.env.VITE_API_PORT || '8000';
    this.wsUrl = `${protocol}//${host}:${port}/api/metrics/ws`;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    connectionState.set('connecting');

    try {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log('Metrics WebSocket connected');
        connectionState.set('connected');
        this.reconnectAttempts = 0;
        this.startPing();
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse metrics message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log('Metrics WebSocket closed:', event.code, event.reason);
        this.handleDisconnect();
      };

      this.ws.onerror = (error) => {
        console.error('Metrics WebSocket error:', error);
        this.handleDisconnect();
      };

    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      this.scheduleReconnect();
    }
  }

  private handleMessage(data: unknown): void {
    if (!data || typeof data !== 'object') {
      return;
    }

    const message = data as MetricsWebSocketMessage;
    const timestamp = new Date();

    switch (message.type) {
      case 'metrics':
        this.updateMetrics(message.payload, timestamp);
        break;

      case 'alert':
        this.handleAlert(message.payload);
        break;

      case 'pong':
        // Heartbeat response - update last pong time
        this.lastPongTime = Date.now();
        if (this.pongTimeout) {
          clearTimeout(this.pongTimeout);
          this.pongTimeout = null;
        }
        break;

      case 'ping':
        // Handle server-initiated ping - respond with pong
        if (this.ws?.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({ action: 'pong' }));
        }
        break;

      default:
        console.warn('Unknown message type:', (message as BaseMetricsMessage).type);
    }

    lastUpdate.set(timestamp);
  }

  private updateMetrics(payload: RawMetricsPayload, timestamp: Date): void {
    // Update individual metric stores
    if (payload.system) {
      systemMetrics.update(current => ({ ...current, ...payload.system }));

      // Update history
      cpuHistory.update(history => {
        const newHistory = [...history, { time: timestamp, value: payload.system.cpu_usage || 0 }];
        return newHistory.slice(-MAX_HISTORY_LENGTH);
      });

      memoryHistory.update(history => {
        const newHistory = [...history, { time: timestamp, value: payload.system.memory_usage || 0 }];
        return newHistory.slice(-MAX_HISTORY_LENGTH);
      });
    }

    if (payload.trading) {
      tradingMetrics.update(current => ({ ...current, ...payload.trading }));

      // Update latency history
      latencyHistory.update(history => {
        const newHistory = [...history, { time: timestamp, value: payload.trading.tick_latency_ms || 0 }];
        return newHistory.slice(-MAX_HISTORY_LENGTH);
      });
    }

    if (payload.database) {
      databaseMetrics.update(current => ({ ...current, ...payload.database }));
    }

    if (payload.tick_stream) {
      tickStreamMetrics.update(current => ({ ...current, ...payload.tick_stream }));

      // Update tick rate history
      tickRateHistory.update(history => {
        const newHistory = [...history, { time: timestamp, value: payload.tick_stream.ticks_per_second || 0 }];
        return newHistory.slice(-MAX_HISTORY_LENGTH);
      });
    }
  }

  private handleAlert(payload: AlertData & { timestamp: string }): void {
    alerts.update(current => {
      // Add new alert at the beginning
      const newAlerts = [{ ...payload, timestamp: new Date(payload.timestamp) }, ...current];
      // Keep only the last 100 alerts
      return newAlerts.slice(0, 100);
    });

    // Optionally trigger a notification
    if (payload.severity === 'critical') {
      this.showNotification({ ...payload, timestamp: new Date(payload.timestamp) });
    }
  }

  private showNotification(alert: AlertData): void {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification(`QuantMindX Alert: ${alert.severity.toUpperCase()}`, {
        body: alert.message,
        icon: '/favicon.ico'
      });
    }
  }

  private handleDisconnect(): void {
    connectionState.set('disconnected');
    this.stopPing();
    this.scheduleReconnect();
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnect attempts reached');
      return;
    }

    connectionState.set('reconnecting');
    this.reconnectAttempts++;

    // Exponential backoff with max of 30 seconds
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private startPing(): void {
    this.lastPongTime = Date.now();
    this.pingInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ action: 'ping' }));

        // Set timeout for pong response
        if (this.pongTimeout) {
          clearTimeout(this.pongTimeout);
        }

        this.pongTimeout = setTimeout(() => {
          console.warn('[MetricsWS] No pong received, connection may be dead');
          if (this.ws) {
            this.ws.close(1000, 'Heartbeat timeout');
          }
        }, this.pongTimeoutDuration);
      }
    }, 30000); // Ping every 30 seconds
  }

  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    if (this.pongTimeout) {
      clearTimeout(this.pongTimeout);
      this.pongTimeout = null;
    }
  }

  disconnect(): void {
    this.stopPing();

    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    connectionState.set('disconnected');
  }

  acknowledgeAlert(alertId: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'acknowledge_alert',
        alert_id: alertId
      }));
    }

    alerts.update(current =>
      current.map(a => a.id === alertId ? { ...a, acknowledged: true } : a)
    );
  }

  clearAlerts(): void {
    alerts.set([]);
  }

  removeAlert(alertId: string): void {
    alerts.update(current => current.filter(a => a.id !== alertId));
  }

  requestNotificationPermission(): void {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }
}

// Singleton instance
export const metricsWebSocket = new MetricsWebSocketService();

// Derived stores for convenience
export const isHealthy = derived(
  [systemMetrics, tradingMetrics],
  ([$system, $trading]) => {
    return $system.cpu_usage < 90 &&
      $system.memory_usage < 90 &&
      $trading.tick_latency_ms < 100;
  }
);

export const activeAlerts = derived(
  alerts,
  ($alerts) => $alerts.filter(a => !a.acknowledged)
);

export const criticalAlerts = derived(
  alerts,
  ($alerts) => $alerts.filter(a => a.severity === 'critical' && !a.acknowledged)
);
