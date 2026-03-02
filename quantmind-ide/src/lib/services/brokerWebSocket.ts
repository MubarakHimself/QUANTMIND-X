/**
 * Broker WebSocket Service
 *
 * Handles real-time broker detection notifications and status updates.
 */

import { writable, derived, get } from 'svelte/store';

// Types
export interface BrokerInfo {
  id: string;
  account_id: string;
  server: string;
  broker_name: string;
  balance: number;
  equity: number;
  margin: number;
  leverage: number;
  currency: string;
  last_seen: Date;
  status: 'connected' | 'disconnected' | 'pending' | 'error';
  is_testnet?: boolean;
  type: 'mt5' | 'binance';
  detected_at?: Date;
}

// Raw broker data from WebSocket (before normalization)
interface RawBrokerData {
  id: string;
  account_id: string;
  server: string;
  broker_name: string;
  balance: number;
  equity: number;
  margin: number;
  leverage: number;
  currency: string;
  last_seen: string;
  status: BrokerInfo['status'];
  is_testnet?: boolean;
  type: BrokerInfo['type'];
  detected_at?: string;
}

// WebSocket message types
type BrokerMessageType =
  | 'broker_detected'
  | 'broker_status'
  | 'broker_disconnected'
  | 'broker_list'
  | 'broker_confirmed'
  | 'broker_synced'
  | 'pong';

interface BaseBrokerMessage {
  type: BrokerMessageType;
}

interface BrokerDetectedMessage extends BaseBrokerMessage {
  type: 'broker_detected';
  broker: RawBrokerData;
}

interface BrokerStatusMessage extends BaseBrokerMessage {
  type: 'broker_status';
  broker_id: string;
  status: BrokerInfo['status'];
  balance: number;
  equity: number;
  margin: number;
  last_seen: string;
}

interface BrokerDisconnectedMessage extends BaseBrokerMessage {
  type: 'broker_disconnected';
  broker_id: string;
}

interface BrokerListMessage extends BaseBrokerMessage {
  type: 'broker_list';
  brokers: RawBrokerData[];
}

interface BrokerConfirmedMessage extends BaseBrokerMessage {
  type: 'broker_confirmed';
  broker: RawBrokerData;
}

interface BrokerSyncedMessage extends BaseBrokerMessage {
  type: 'broker_synced';
  broker: RawBrokerData;
}

interface PongMessage extends BaseBrokerMessage {
  type: 'pong';
}

type BrokerWebSocketMessage =
  | BrokerDetectedMessage
  | BrokerStatusMessage
  | BrokerDisconnectedMessage
  | BrokerListMessage
  | BrokerConfirmedMessage
  | BrokerSyncedMessage
  | PongMessage;

export interface BrokerDetectionEvent {
  type: 'broker_detected';
  broker: BrokerInfo;
}

export interface BrokerStatusUpdate {
  type: 'broker_status';
  broker_id: string;
  status: BrokerInfo['status'];
  balance: number;
  equity: number;
  margin: number;
  last_seen: Date;
}

// Stores
export const brokers = writable<BrokerInfo[]>([]);
export const pendingBrokers = writable<BrokerInfo[]>([]);
export const connectionState = writable<'connecting' | 'connected' | 'disconnected' | 'reconnecting'>('disconnected');

// Derived stores
export const connectedBrokers = derived(brokers, ($brokers) =>
  $brokers.filter(b => b.status === 'connected')
);

export const mt5Brokers = derived(brokers, ($brokers) =>
  $brokers.filter(b => b.type === 'mt5')
);

export const binanceBrokers = derived(brokers, ($brokers) =>
  $brokers.filter(b => b.type === 'binance')
);

class BrokerWebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private wsUrl: string;

  constructor() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.hostname;
    const port = import.meta.env.VITE_API_PORT || '8000';
    this.wsUrl = `${protocol}//${host}:${port}/api/brokers/ws`;
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    connectionState.set('connecting');

    try {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        console.log('Broker WebSocket connected');
        connectionState.set('connected');
        this.reconnectAttempts = 0;
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleMessage(data);
        } catch (error) {
          console.error('Failed to parse broker message:', error);
        }
      };

      this.ws.onclose = () => {
        connectionState.set('disconnected');
        this.scheduleReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('Broker WebSocket error:', error);
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

    const message = data as BrokerWebSocketMessage;

    switch (message.type) {
      case 'broker_detected':
        this.handleBrokerDetected(message.broker);
        break;

      case 'broker_status':
        this.handleBrokerStatus(message);
        break;

      case 'broker_disconnected':
        this.handleBrokerDisconnected(message.broker_id);
        break;

      case 'broker_list':
        brokers.set(message.brokers.map(this.normalizeBroker));
        break;

      case 'broker_confirmed':
        this.handleBrokerConfirmed(message.broker);
        break;

      case 'broker_synced':
        this.handleBrokerSynced(message.broker);
        break;

      case 'pong':
        break;
    }
  }

  private normalizeBroker(b: RawBrokerData): BrokerInfo {
    return {
      id: b.id,
      account_id: b.account_id,
      server: b.server,
      broker_name: b.broker_name,
      balance: b.balance,
      equity: b.equity,
      margin: b.margin,
      leverage: b.leverage,
      currency: b.currency,
      last_seen: new Date(b.last_seen),
      status: b.status,
      is_testnet: b.is_testnet,
      type: b.type,
      detected_at: b.detected_at ? new Date(b.detected_at) : new Date()
    };
  }

  private handleBrokerDetected(broker: RawBrokerData): void {
    const normalized = this.normalizeBroker(broker);

    // Check if already registered
    brokers.update(current => {
      const existing = current.find(b => b.account_id === normalized.account_id);
      if (existing) {
        // Update existing
        return current.map(b =>
          b.account_id === normalized.account_id ? { ...b, ...normalized } : b
        );
      }
      // Add as pending
      pendingBrokers.update(pending => [...pending, normalized]);
      return current;
    });

    // Show notification
    this.showDetectionNotification(normalized);
  }

  private handleBrokerStatus(data: BrokerStatusMessage): void {
    brokers.update(current =>
      current.map(b =>
        b.id === data.broker_id || b.account_id === data.broker_id
          ? {
            ...b,
            status: data.status,
            balance: data.balance,
            equity: data.equity,
            margin: data.margin,
            last_seen: new Date(data.last_seen)
          }
          : b
      )
    );
  }

  private handleBrokerDisconnected(brokerId: string): void {
    brokers.update(current =>
      current.map(b =>
        b.id === brokerId ? { ...b, status: 'disconnected' } : b
      )
    );
  }

  private handleBrokerConfirmed(broker: RawBrokerData): void {
    const normalized = this.normalizeBroker(broker);
    
    // Remove from pending if present
    pendingBrokers.update(pending => 
      pending.filter(b => b.id !== normalized.id)
    );
    
    // Add to confirmed brokers or update existing
    brokers.update(current => {
      const existing = current.find(b => b.id === normalized.id || b.account_id === normalized.account_id);
      if (existing) {
        // Update existing broker with confirmed status
        return current.map(b =>
          b.id === normalized.id || b.account_id === normalized.account_id
            ? { ...b, ...normalized, status: 'connected' }
            : b
        );
      }
      // Add new confirmed broker
      return [...current, { ...normalized, status: 'connected' }];
    });
    
    // Show notification for confirmed broker
    this.showConfirmationNotification(normalized);
  }

  private handleBrokerSynced(broker: RawBrokerData): void {
    const normalized = this.normalizeBroker(broker);
    
    // Update broker data in the brokers list
    brokers.update(current =>
      current.map(b =>
        b.id === normalized.id || b.account_id === normalized.account_id
          ? { ...b, ...normalized, status: 'connected', last_seen: new Date() }
          : b
      )
    );
    
    // Show notification for synced broker
    this.showSyncNotification(normalized);
  }

  private showConfirmationNotification(broker: BrokerInfo): void {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Broker Confirmed', {
        body: `${broker.broker_name} (${broker.account_id}) has been confirmed`,
        icon: '/favicon.ico',
        tag: 'broker-confirmed'
      });
    }
  }

  private showSyncNotification(broker: BrokerInfo): void {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Broker Synced', {
        body: `${broker.broker_name} (${broker.account_id}) synced successfully`,
        icon: '/favicon.ico',
        tag: 'broker-synced'
      });
    }
  }

  private showDetectionNotification(broker: BrokerInfo): void {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('New Broker Detected', {
        body: `${broker.broker_name} (${broker.account_id})`,
        icon: '/favicon.ico',
        tag: 'broker-detection'
      });
    }
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

    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      30000
    );

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  // Public methods
  confirmBroker(brokerId: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'confirm_broker',
        broker_id: brokerId
      }));
    }

    // Move from pending to confirmed
    pendingBrokers.update(pending => {
      const broker = pending.find(b => b.id === brokerId);
      if (broker) {
        brokers.update(current => [...current, { ...broker, status: 'connected' }]);
        return pending.filter(b => b.id !== brokerId);
      }
      return pending;
    });
  }

  ignoreBroker(brokerId: string): void {
    pendingBrokers.update(pending =>
      pending.filter(b => b.id !== brokerId)
    );

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'ignore_broker',
        broker_id: brokerId
      }));
    }
  }

  disconnectBroker(brokerId: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'disconnect_broker',
        broker_id: brokerId
      }));
    }

    brokers.update(current =>
      current.map(b =>
        b.id === brokerId ? { ...b, status: 'disconnected' } : b
      )
    );
  }

  syncBroker(brokerId: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'sync_broker',
        broker_id: brokerId
      }));
    }
  }

  addManualBroker(broker: Partial<BrokerInfo>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'add_broker',
        broker
      }));
    }
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    connectionState.set('disconnected');
  }

  requestNotificationPermission(): void {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }

  /**
   * Subscribe to broker WebSocket messages.
   * Returns an unsubscribe function.
   */
  subscribe(callback: (message: BrokerWebSocketMessage) => void): () => void {
    const handler = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data && typeof data === 'object' && 'type' in data) {
          callback(data as BrokerWebSocketMessage);
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    if (this.ws) {
      this.ws.addEventListener('message', handler);
    }

    return () => {
      if (this.ws) {
        this.ws.removeEventListener('message', handler);
      }
    };
  }

  /**
   * Refresh broker list from server
   */
  refreshBrokers(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'get_brokers' }));
    }
  }
}

export const brokerWebSocket = new BrokerWebSocketService();
