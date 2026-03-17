// WebSocket base URL - use empty string for relative URLs
const WS_BASE = '';

/**
 * WebSocket Client for QuantMind IDE real-time updates.
 */

export interface WebSocketMessage {
  type: string;
  data?: Record<string, unknown>;
  topic?: string;
}

export interface BacktestStartMessage {
  backtest_id: string;
  variant: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  timestamp: string;
}

export interface BacktestProgressMessage {
  backtest_id: string;
  progress: number;
  status: string;
  bars_processed?: number;
  total_bars?: number;
  current_date?: string;
  trades_count?: number;
  current_pnl?: number;
  timestamp: string;
}

export interface BacktestCompleteMessage {
  backtest_id: string;
  final_balance: number;
  total_trades: number;
  win_rate?: number;
  sharpe_ratio?: number;
  drawdown?: number;
  return_pct?: number;
  duration_seconds?: number;
  timestamp: string;
  results?: Record<string, unknown>;
}

export interface BacktestErrorMessage {
  backtest_id: string;
  error: string;
  error_details?: string;
  timestamp: string;
}

export interface LogEntryMessage {
  backtest_id?: string;
  timestamp: string;
  level: string;
  message: string;
  module?: string;
  function?: string;
  line?: number;
  logger_name?: string;
}

export interface TickDataMessage {
  symbol: string;
  bid: number;
  ask: number;
  spread: number;
  timestamp: string;
}

export interface PaperTradingUpdateMessage {
  agent_id: string;
  status: string;
  uptime_seconds?: number;
  [key: string]: unknown;
}

// =============================================================================
// Chart Data Streaming Types (TradingView Integration)
// =============================================================================

export interface BarUpdateMessage {
  symbol: string;
  timeframe: string;
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: string;
}

export interface RegimeChangeMessage {
  symbol: string;
  regime: 'TRENDING' | 'RANGING' | 'VOLATILE' | 'CHOPPY';
  confidence: number;
  timestamp: string;
}

export interface TradeExecutionMessage {
  type: 'buy' | 'sell';
  symbol: string;
  price: number;
  volume: number;
  bot_id?: string;
  timestamp: string;
}

export interface ChartSubscriptionMessage {
  type: 'chart_subscription';
  symbol: string;
  timeframe: string;
}

export interface SubscriptionConfirmation {
  type: 'subscription_confirmed';
  topic: string;
}

export type MessageHandler = (message: WebSocketMessage) => void;

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000; // Start with 1 second
  private maxReconnectDelay = 30000; // Max 30 seconds
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private isConnecting = false;
  private isDisconnecting = false;
  private subscriptionTopics: Set<string> = new Set();
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private heartbeatInterval = 30000; // 30 seconds heartbeat
  private lastPongTime = 0;
  private pongTimeout: ReturnType<typeof setTimeout> | null = null;
  private pongTimeoutDuration = 10000; // 10 seconds to receive pong

  constructor(url: string) {
    this.url = url;
  }

  /**
   * Connect to WebSocket server.
   * @returns Promise that resolves when connected
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
        resolve();
        return;
      }

      this.isConnecting = true;

      try {
        this.ws = new WebSocket(this.url);
        this.ws.binaryType = 'arraybuffer';

        this.ws.onopen = () => {
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          this.lastPongTime = Date.now();
          console.log('[WS] Connected to', this.url);

          // Start heartbeat
          this._startHeartbeat();

          // Re-subscribe to topics
          for (const topic of this.subscriptionTopics) {
            this.send({ action: 'subscribe', topic });
          }

          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data) as WebSocketMessage;

            // Handle pong response
            if (message.type === 'pong') {
              this.lastPongTime = Date.now();
              if (this.pongTimeout) {
                clearTimeout(this.pongTimeout);
                this.pongTimeout = null;
              }
              return;
            }

            // Handle server-initiated ping - respond with pong
            if (message.type === 'ping') {
              this.send({ action: 'pong' });
              return;
            }

            this._handleMessage(message);
          } catch (e) {
            console.error('[WS] Failed to parse message:', e);
          }
        };

        this.ws.onerror = (error) => {
          console.error('[WS] Error:', error);
          this.isConnecting = false;
          reject(error);
        };

        this.ws.onclose = (event) => {
          this.isConnecting = false;
          this._stopHeartbeat();
          console.log('[WS] Disconnected:', event.code, event.reason);

          // Attempt reconnection if not intentionally disconnecting
          if (!this.isDisconnecting) {
            this._attemptReconnect();
          }
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  /**
   * Start heartbeat to keep connection alive.
   */
  private _startHeartbeat(): void {
    this._stopHeartbeat();

    this.heartbeatTimer = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        // Send ping
        this.send({ action: 'ping' });

        // Set timeout for pong response
        if (this.pongTimeout) {
          clearTimeout(this.pongTimeout);
        }

        this.pongTimeout = setTimeout(() => {
          // No pong received within timeout - force reconnect
          console.warn('[WS] No pong received, connection may be dead');
          if (this.ws) {
            this.ws.close(1000, 'Heartbeat timeout');
          }
        }, this.pongTimeoutDuration);
      }
    }, this.heartbeatInterval);
  }

  /**
   * Stop heartbeat timer.
   */
  private _stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
    if (this.pongTimeout) {
      clearTimeout(this.pongTimeout);
      this.pongTimeout = null;
    }
  }

  /**
   * Set heartbeat interval (in milliseconds).
   * @param interval - Interval in ms (default: 30000)
   */
  setHeartbeatInterval(interval: number): void {
    this.heartbeatInterval = interval;
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this._startHeartbeat();
    }
  }

  /**
   * Set pong timeout duration (in milliseconds).
   * @param duration - Duration in ms (default: 10000)
   */
  setPongTimeout(duration: number): void {
    this.pongTimeoutDuration = duration;
  }

  /**
   * Set maximum reconnection attempts.
   * @param max - Maximum attempts (default: 10)
   */
  setMaxReconnectAttempts(max: number): void {
    this.maxReconnectAttempts = max;
  }

  /**
   * Disconnect from WebSocket server.
   */
  disconnect(): void {
    this.isDisconnecting = true;
    this._stopHeartbeat();

    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.isDisconnecting = false;
  }

  /**
   * Send a message to the server.
   * @param data - JSON-serializable data to send
   */
  send(data: Record<string, unknown>): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('[WS] Cannot send message: not connected');
    }
  }

  /**
   * Subscribe to a topic.
   * @param topic - Topic name (e.g., "backtest", "trading", "logs")
   */
  subscribe(topic: string): void {
    this.subscriptionTopics.add(topic);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.send({ action: 'subscribe', topic });
    }
  }

  /**
   * Unsubscribe from a topic.
   * @param topic - Topic name
   */
  unsubscribe(topic: string): void {
    this.subscriptionTopics.delete(topic);

    // Note: Server doesn't have an unsubscribe action, but we track locally
  }

  /**
   * Register an event handler.
   * @param type - Event type (e.g., "backtest_start", "backtest_progress", "log")
   * @param handler - Handler function
   */
  on(type: string, handler: MessageHandler): void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);
  }

  /**
   * Remove an event handler.
   * @param type - Event type
   * @param handler - Handler function to remove
   */
  off(type: string, handler: MessageHandler): void {
    const handlers = this.handlers.get(type);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  /**
   * Check if connected.
   */
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection state.
   */
  getState(): number {
    return this.ws ? this.ws.readyState : WebSocket.CLOSED;
  }

  /**
   * Get connection state as string.
   */
  getStateString(): string {
    const state = this.getState();
    switch (state) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'connected';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'disconnected';
      default:
        return 'unknown';
    }
  }

  /**
   * Check if currently reconnecting.
   */
  isReconnecting(): boolean {
    return this.reconnectAttempts > 0 && this.reconnectAttempts < this.maxReconnectAttempts;
  }

  /**
   * Get current reconnect attempt count.
   */
  getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }

  /**
   * Manually trigger a reconnect.
   */
  reconnect(): void {
    this._stopHeartbeat();
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.reconnectAttempts = 0;
    this._attemptReconnect();
  }

  private _handleMessage(message: WebSocketMessage): void {
    const handlers = this.handlers.get(message.type);
    if (handlers) {
      for (const handler of handlers) {
        try {
          handler(message);
        } catch (e) {
          console.error(`[WS] Handler error for ${message.type}:`, e);
        }
      }
    }
  }

  private _attemptReconnect(): void {
    if (this.isConnecting || this.isDisconnecting) {
      return;
    }

    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WS] Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), this.maxReconnectDelay);

    console.log(`[WS] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

    setTimeout(() => {
      this.connect().catch((error) => {
        console.error('[WS] Reconnection failed:', error);
      });
    }, delay);
  }
}

/**
 * Create a WebSocket client connected to /ws endpoint.
 * @param baseUrl - Base URL (e.g., "http://localhost:8000")
 * @returns Connected WebSocket client
 */
export async function createWebSocketClient(baseUrl: string): Promise<WebSocketClient> {
  const wsUrl = `${baseUrl.replace('http://', 'ws://').replace('https://', 'wss://')}/ws`;
  const client = new WebSocketClient(wsUrl);
  await client.connect();
  return client;
}

/**
 * Create a backtest WebSocket client.
 * Connects to /ws and subscribes to "backtest" topic.
 * @param baseUrl - Base URL (e.g., "http://localhost:8000")
 * @returns Connected WebSocket client subscribed to backtest
 */
export async function createBacktestClient(baseUrl: string): Promise<WebSocketClient> {
  const client = await createWebSocketClient(baseUrl);
  client.subscribe('backtest');
  return client;
}

/**
 * Create a trading WebSocket client.
 * Connects to /ws and subscribes to "trading" topic.
 * @param baseUrl - Base URL (e.g., "http://localhost:8000")
 * @returns Connected WebSocket client subscribed to trading
 */
export async function createTradingClient(baseUrl: string): Promise<WebSocketClient> {
  const client = await createWebSocketClient(baseUrl);
  client.subscribe('trading');
  return client;
}

/**
 * Create a chart WebSocket client.
 * Connects to /ws and subscribes to "chart" topic for real-time bar updates.
 * @param baseUrl - Base URL (e.g., "http://localhost:8000")
 * @param symbol - Trading symbol to subscribe to (e.g., "EURUSD")
 * @param timeframe - Chart timeframe (e.g., "M1", "M5", "H1")
 * @returns Connected WebSocket client subscribed to chart data
 */
export async function createChartClient(
  baseUrl: string,
  symbol?: string,
  timeframe?: string
): Promise<WebSocketClient> {
  const client = await createWebSocketClient(baseUrl);
  client.subscribe('chart');

  // Send chart subscription with symbol and timeframe if provided
  if (symbol && timeframe) {
    client.send({
      action: 'subscribe',
      topic: 'chart',
      symbol,
      timeframe
    } as unknown as Record<string, unknown>);
  }

  return client;
}

export const wsClient = new WebSocketClient(`${WS_BASE}/ws`);
export default WebSocketClient;
