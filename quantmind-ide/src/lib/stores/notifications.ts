/**
 * Notifications Store
 *
 * Manages in-app notifications with unread tracking,
 * canvas navigation links, and typed notification categories.
 */

import { writable, derived } from 'svelte/store';

export interface AppNotification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'trade' | 'agent';
  title: string;
  body?: string;
  timestamp: Date;
  read: boolean;
  canvasLink?: string;
}

// Seed demo notifications so the tray isn't empty on first load
const SEED_NOTIFICATIONS: AppNotification[] = [
  {
    id: 'seed-1',
    type: 'agent',
    title: 'Research Head',
    body: 'Hypothesis "BTC momentum regime" queued for review',
    timestamp: new Date(Date.now() - 2 * 60 * 1000),
    read: false,
    canvasLink: 'research',
  },
  {
    id: 'seed-2',
    type: 'trade',
    title: 'Kill Switch Armed',
    body: 'Tier 1 soft-stop activated by operator',
    timestamp: new Date(Date.now() - 18 * 60 * 1000),
    read: false,
    canvasLink: 'live-trading',
  },
  {
    id: 'seed-3',
    type: 'info',
    title: 'System',
    body: 'FlowForge: 2 workflows completed successfully',
    timestamp: new Date(Date.now() - 47 * 60 * 1000),
    read: false,
    canvasLink: 'flowforge',
  },
];

function createNotificationStore() {
  const { subscribe, set, update } = writable<AppNotification[]>(SEED_NOTIFICATIONS);

  return {
    subscribe,

    /** Add a new notification (id, timestamp and read:false are auto-assigned) */
    addNotification(n: Omit<AppNotification, 'id' | 'timestamp' | 'read'>) {
      const notification: AppNotification = {
        ...n,
        id: `notif-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        timestamp: new Date(),
        read: false,
      };
      update((list) => [notification, ...list]);
    },

    /** Mark a single notification as read */
    markRead(id: string) {
      update((list) =>
        list.map((n) => (n.id === id ? { ...n, read: true } : n))
      );
    },

    /** Mark all notifications as read */
    markAllRead() {
      update((list) => list.map((n) => ({ ...n, read: true })));
    },

    /** Remove all notifications */
    clearAll() {
      set([]);
    },
  };
}

export const notificationStore = createNotificationStore();

/** Derived count of unread notifications */
export const unreadCount = derived(
  notificationStore,
  ($notifications) => $notifications.filter((n) => !n.read).length
);

// Re-export helpers at module level for ergonomic imports
export const addNotification = notificationStore.addNotification.bind(notificationStore);
export const markRead = notificationStore.markRead.bind(notificationStore);
export const markAllRead = notificationStore.markAllRead.bind(notificationStore);
export const clearAll = notificationStore.clearAll.bind(notificationStore);
