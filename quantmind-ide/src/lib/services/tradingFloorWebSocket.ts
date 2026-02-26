/**
 * Trading Floor WebSocket Service
 *
 * Handles real-time updates for the Trading Floor visualization.
 */

import { writable, derived, get } from 'svelte/store';
import type { AgentState, MailMessage, DepartmentPosition } from '$lib/stores/tradingFloorStore';

export type TradingFloorEventType =
  | 'agent_spawned'
  | 'agent_state_changed'
  | 'mail_sent'
  | 'mail_delivered'
  | 'task_dispatched'
  | 'task_completed'
  | 'sub_agent_spawned'
  | 'sub_agent_completed';

export type TradingFloorEvent = {
  type: TradingFloorEventType;
  timestamp: string;
  data: Record<string, any>;
}

export type TradingFloorWSMessage = {
  topic: 'trading-floor';
  event: TradingFloorEvent;
}

// WebSocket connection state
export const wsConnected = writable(false);
export const wsError = writable<string | null>(null);
export const recentEvents = writable<TradingFloorEvent[]>([]);

// WebSocket connection
let ws: WebSocket | null = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 3000;

/**
 * Connect to Trading Floor WebSocket
 */
export function connectTradingFloorWS(url?: string) {
  const wsUrl = url || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/api/ws`;

  if (ws?.readyState === WebSocket.OPEN) {
    console.log('[TradingFloorWS] Already connected');
    return;
  }

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[TradingFloorWS] Connected');
      wsConnected.set(true);
      wsError.set(null);
      reconnectAttempts = 0;

      // Subscribe to trading-floor topic
      ws?.send(JSON.stringify({
        type: 'subscribe',
        topic: 'trading-floor',
      }));
    };

    ws.onmessage = (event) => {
      try {
        const message: TradingFloorWSMessage = JSON.parse(event.data);
        handleTradingFloorEvent(message.event);
      } catch (e) {
        console.error('[TradingFloorWS] Parse error:', e);
      }
    };

    ws.onerror = (error) => {
      console.error('[TradingFloorWS] Error:', error);
      wsError.set('WebSocket connection error');
    };

    ws.onclose = () => {
      console.log('[TradingFloorWS] Disconnected');
      wsConnected.set(false);
      ws = null;

      // Attempt reconnect
      if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
        reconnectAttempts++;
        console.log(`[TradingFloorWS] Reconnecting in ${RECONNECT_DELAY}ms (attempt ${reconnectAttempts})`);
        setTimeout(() => connectTradingFloorWS(url), RECONNECT_DELAY);
      }
    };
  } catch (e) {
    console.error('[TradingFloorWS] Connection failed:', e);
    wsError.set('Failed to connect');
  }
}

/**
 * Disconnect WebSocket
 */
export function disconnectTradingFloorWS() {
  if (ws) {
    ws.close();
    ws = null;
  }
  wsConnected.set(false);
}

/**
 * Handle incoming Trading Floor events
 */
function handleTradingFloorEvent(event: TradingFloorEvent) {
  // Add to recent events (keep last 100)
  recentEvents.update((events) => {
    const newEvents = [event, ...events].slice(0, 100);
    return newEvents;
  });

  // Update stores based on event type
  switch (event.type) {
    case 'agent_spawned':
      handleAgentSpawned(event.data);
      break;

    case 'agent_state_changed':
      handleAgentStateChanged(event.data);
      break;

    case 'mail_sent':
    case 'mail_delivered':
      handleMailEvent(event.data);
      break;

    case 'sub_agent_spawned':
      handleSubAgentSpawned(event.data);
      break;

    case 'task_dispatched':
    case 'task_completed':
      handleTaskEvent(event.data);
      break;
  }
}

/**
 * Handle agent spawned event
 */
function handleAgentSpawned(data: Record<string, any>) {
  const { updateAgentState } = require('$lib/stores/tradingFloorStore');

  const agent: AgentState = {
    id: data.id,
    name: data.name,
    department: data.department,
    status: 'idle',
    position: data.position || { x: 0, y: 0 },
    target: null,
    subAgents: [],
    isExpanded: false,
  };

  updateAgentState(data.id, agent);
}

/**
 * Handle agent state changed event
 */
function handleAgentStateChanged(data: Record<string, any>) {
  const { updateAgentState } = require('$lib/stores/tradingFloorStore');
  updateAgentState(data.agent_id, {
    status: data.new_state,
    speechBubble: data.speech_bubble,
  });
}

/**
 * Handle mail event
 */
function handleMailEvent(data: Record<string, any>) {
  const { sendMail, completeMailAnimation } = require('$lib/stores/tradingFloorStore');

  if (data.type === 'sent') {
    sendMail({
      id: data.message_id,
      fromDept: data.from_dept,
      toDept: data.to_dept,
      type: data.mail_type,
      subject: data.subject,
      startX: 0,
      startY: 0,
      progress: 0,
      duration: 1000,
    });
  } else if (data.type === 'delivered') {
    completeMailAnimation();
  }
}

/**
 * Handle sub-agent spawned event
 */
function handleSubAgentSpawned(data: Record<string, any>) {
  const { addSubAgent } = require('$lib/stores/tradingFloorStore');

  const subAgent: AgentState = {
    id: data.sub_agent_id,
    name: data.name,
    department: data.department,
    status: 'idle',
    position: data.position || { x: 0, y: 0 },
    target: null,
    subAgents: [],
    isExpanded: false,
  };

  addSubAgent(data.parent_id, subAgent);
}

/**
 * Handle task event
 */
function handleTaskEvent(data: Record<string, any>) {
  // Update floor stats based on task events
  const { floorStats } = require('$lib/stores/tradingFloorStore');

  floorStats.update((stats) => {
    if (data.type === 'dispatched') {
      return {
        ...stats,
        totalTasks: stats.totalTasks + 1,
        activeTasks: stats.activeTasks + 1,
      };
    } else if (data.type === 'completed') {
      return {
        ...stats,
        activeTasks: stats.activeTasks - 1,
        completedTasks: stats.completedTasks + 1,
      };
    }
    return stats;
  });
}

/**
 * Send event to Trading Floor
 */
export function sendTradingFloorEvent(event: TradingFloorEvent) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      topic: 'trading-floor',
      event,
    }));
  } else {
    console.warn('[TradingFloorWS] Cannot send - not connected');
  }
}

// Export for use in components
export const tradingFloorWS = {
  connect: connectTradingFloorWS,
  disconnect: disconnectTradingFloorWS,
  send: sendTradingFloorEvent,
  connected: wsConnected,
  error: wsError,
  recentEvents,
};
