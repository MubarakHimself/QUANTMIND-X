/**
 * Node Health Store
 *
 * Manages connectivity status for Contabo (agents/compute) and Cloudzy (live trading) nodes.
 * Implements degraded mode awareness and auto-recovery.
 */

import { writable, derived } from 'svelte/store';
import { apiFetch } from '$lib/api';

// Node status types
export type NodeStatus = 'connected' | 'disconnected' | 'reconnecting';

export interface NodeHealth {
  status: NodeStatus;
  latency_ms: number;
  lastChecked: string; // ISO timestamp with _utc suffix
  lastConnected: string | null; // ISO timestamp with _utc suffix
}

export interface NodeHealthState {
  contabo: NodeHealth;
  cloudzy: NodeHealth;
  lastKnownData: {
    agentActivity: any[];
    workflowState: any;
  };
  isDegraded: boolean;
}

// Default state
const defaultState: NodeHealthState = {
  contabo: {
    status: 'connected',
    latency_ms: 0,
    lastChecked: '',
    lastConnected: null
  },
  cloudzy: {
    status: 'connected',
    latency_ms: 0,
    lastChecked: '',
    lastConnected: null
  },
  lastKnownData: {
    agentActivity: [],
    workflowState: null
  },
  isDegraded: false
};

// Create the store
export const nodeHealthState = writable<NodeHealthState>(defaultState);

// Polling configuration
let healthCheckInterval: ReturnType<typeof setInterval> | null = null;
const DEFAULT_POLL_INTERVAL = 10000; // 10 seconds
const RECOVERY_TIMEOUT = 10000; // 10 seconds for auto-recovery

// Derived store for degraded mode status
export const isContaboDegraded = derived(nodeHealthState, ($state) =>
  $state.contabo.status !== 'connected'
);

// Derived store for cloudzy connectivity
export const isCloudzyConnected = derived(nodeHealthState, ($state) =>
  $state.cloudzy.status === 'connected'
);

// Derived store for cloudzy degraded mode
export const isCloudzyDegraded = derived(nodeHealthState, ($state) =>
  $state.cloudzy.status !== 'connected'
);

// Derived store for overall system health
export const systemDegraded = derived(nodeHealthState, ($state) =>
  $state.isDegraded || $state.contabo.status === 'disconnected'
);

/**
 * Check node health by calling the backend API
 */
export async function checkNodeHealth(): Promise<void> {
  try {
    const data = await apiFetch<any>('/v1/server/health/nodes');

    nodeHealthState.update((state) => {
      const now = new Date().toISOString() + '_utc';

      // Update Contabo status
      const contaboStatus = data.contabo?.status || 'disconnected';
      const wasContaboConnected = state.contabo.status === 'connected';

      state.contabo = {
        status: contaboStatus as NodeStatus,
        latency_ms: data.contabo?.latency_ms || 0,
        lastChecked: now,
        lastConnected: contaboStatus === 'connected' ? now : state.contabo.lastConnected
      };

      // Update Cloudzy status
      state.cloudzy = {
        status: (data.cloudzy?.status || 'connected') as NodeStatus,
        latency_ms: data.cloudzy?.latency_ms || 0,
        lastChecked: now,
        lastConnected: state.cloudzy.status === 'connected' ? now : state.cloudzy.lastConnected
      };

      // Determine degraded state
      state.isDegraded = contaboStatus === 'disconnected' || contaboStatus === 'reconnecting';

      // Handle reconnection - clear degraded indicators
      if (wasContaboConnected && contaboStatus === 'connected') {
        // Contabo reconnected, show toast notification
        console.log('[NodeHealth] Contabo reconnected');
      }

      return { ...state };
    });
  } catch (error) {
    console.error('[NodeHealth] Health check failed:', error);

    // Mark as disconnected on error
    nodeHealthState.update((state) => {
      const now = new Date().toISOString() + '_utc';
      state.contabo = {
        ...state.contabo,
        status: 'disconnected',
        lastChecked: now
      };
      state.isDegraded = true;
      return { ...state };
    });
  }
}

/**
 * Start periodic health checks
 */
export function startHealthMonitoring(intervalMs: number = DEFAULT_POLL_INTERVAL): void {
  // Clear existing interval if any
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
  }

  // Initial check
  checkNodeHealth();

  // Set up periodic checks
  healthCheckInterval = setInterval(() => {
    checkNodeHealth();
  }, intervalMs);

  console.log(`[NodeHealth] Started monitoring (interval: ${intervalMs}ms)`);
}

/**
 * Stop periodic health checks
 */
export function stopHealthMonitoring(): void {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
    healthCheckInterval = null;
    console.log('[NodeHealth] Stopped monitoring');
  }
}

/**
 * Update last known data (for degraded mode display)
 */
export function updateLastKnownData(agentActivity: any[], workflowState: any): void {
  nodeHealthState.update((state) => {
    state.lastKnownData = {
      agentActivity: agentActivity || state.lastKnownData.agentActivity,
      workflowState: workflowState || state.lastKnownData.workflowState
    };
    return { ...state };
  });
}

/**
 * Reset to default state
 */
export function resetNodeHealth(): void {
  stopHealthMonitoring();
  nodeHealthState.set(defaultState);
}
