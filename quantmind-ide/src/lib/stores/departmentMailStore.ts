/**
 * Department Mail Store
 *
 * State management for department mail in the Trading Floor.
 * Handles fetching, sending, and managing mail between departments.
 */

import { writable, derived, get } from 'svelte/store';

const API_BASE = 'http://localhost:8000/api';

// Types
export type MessageType = 'dispatch' | 'result' | 'question' | 'status' | 'error';

export type Priority = 'low' | 'normal' | 'high' | 'urgent';

export type DepartmentMailMessage = {
  id: string;
  from: string;
  to: string;
  type: MessageType;
  subject: string;
  body: string;
  priority: Priority;
  timestamp: string;
  read: boolean;
};

export type DelegationRequest = {
  from_department: string;
  task: string;
  suggested_department?: string | null;
  context?: Record<string, unknown>;
};

export type DelegationResponse = {
  status: 'success' | 'error';
  dispatch?: {
    status: string;
    message_id: string;
    from_department: string;
    to_department: string;
    priority: string;
  };
  error?: string;
};

// Available departments for delegation
export const DEPARTMENTS = [
  { id: 'analysis', name: 'Analysis', description: 'Market analysis and signals' },
  { id: 'research', name: 'Research', description: 'Strategy development and backtesting' },
  { id: 'risk', name: 'Risk', description: 'Position sizing and risk management' },
  { id: 'execution', name: 'Execution', description: 'Order execution and routing' },
  { id: 'portfolio', name: 'Portfolio', description: 'Portfolio management and allocation' },
] as const;

export type Department = typeof DEPARTMENTS[number]['id'];

// Store state
const mailMessages = writable<Map<string, DepartmentMailMessage[]>>(new Map());
const loading = writable(false);
const error = writable<string | null>(null);
const selectedDepartment = writable<Department | null>(null);
const unreadCounts = writable<Map<string, number>>(new Map());

// Derived stores
export const mailList = derived(
  [mailMessages, selectedDepartment],
  ([$mailMessages, $selectedDepartment]) => {
    if (!$selectedDepartment) return [];
    return $mailMessages.get($selectedDepartment) || [];
  }
);

export const hasUnread = derived(
  [unreadCounts, selectedDepartment],
  ([$unreadCounts, $selectedDepartment]) => {
    if (!$selectedDepartment) return false;
    return ($unreadCounts.get($selectedDepartment) || 0) > 0;
  }
);

export const totalUnread = derived(unreadCounts, ($unreadCounts) => {
  let total = 0;
  for (const count of $unreadCounts.values()) {
    total += count;
  }
  return total;
});

// Actions
export async function fetchDepartmentMail(department: Department): Promise<void> {
  loading.set(true);
  error.set(null);

  try {
    const response = await fetch(`${API_BASE}/trading-floor/mail/${department}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch mail: ${response.statusText}`);
    }

    const data = await response.json();
    const messages: DepartmentMailMessage[] = data.messages || [];

    mailMessages.update((state) => {
      const newState = new Map(state);
      newState.set(department, messages);
      return newState;
    });

    // Update unread count
    const unreadCount = messages.filter((m) => !m.read).length;
    unreadCounts.update((counts) => {
      const newCounts = new Map(counts);
      newCounts.set(department, unreadCount);
      return newCounts;
    });
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Unknown error';
    error.set(message);
    console.error('Failed to fetch department mail:', e);
  } finally {
    loading.set(false);
  }
}

export async function markMessageRead(messageId: string): Promise<void> {
  try {
    const response = await fetch(`${API_BASE}/trading-floor/mail/${messageId}/read`, {
      method: 'POST',
    });

    if (!response.ok) {
      throw new Error(`Failed to mark message as read: ${response.statusText}`);
    }

    // Update local state
    mailMessages.update((state) => {
      const newState = new Map();
      for (const [dept, messages] of state.entries()) {
        const updatedMessages = messages.map((m) =>
          m.id === messageId ? { ...m, read: true } : m
        );
        newState.set(dept, updatedMessages);
      }
      return newState;
    });

    // Update unread counts
    unreadCounts.update((counts) => {
      const newCounts = new Map();
      for (const [dept, count] of counts.entries()) {
        const deptMessages = get(mailMessages).get(dept) || [];
        const unreadInDept = deptMessages.filter(
          (m) => m.id === messageId ? true : !m.read
        ).length;
        newCounts.set(dept, Math.max(0, unreadInDept));
      }
      return newCounts;
    });
  } catch (e) {
    console.error('Failed to mark message as read:', e);
  }
}

export async function delegateToFloor(
  request: DelegationRequest
): Promise<DelegationResponse> {
  loading.set(true);
  error.set(null);

  try {
    const response = await fetch(`${API_BASE}/trading-floor/delegate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`Delegation failed: ${response.statusText}`);
    }

    const data: DelegationResponse = await response.json();

    // Refresh mail for the target department if delegation was successful
    if (data.status === 'success' && data.dispatch?.to_department) {
      await fetchDepartmentMail(data.dispatch.to_department as Department);
    }

    return data;
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Unknown error';
    error.set(message);
    return {
      status: 'error',
      error: message,
    };
  } finally {
    loading.set(false);
  }
}

export function selectDepartment(department: Department | null): void {
  selectedDepartment.set(department);
  if (department) {
    fetchDepartmentMail(department);
  }
}

export function clearError(): void {
  error.set(null);
}

// Utility functions
export function getDepartmentById(id: string) {
  return DEPARTMENTS.find((d) => d.id === id);
}

export function getDepartmentName(id: string): string {
  const dept = getDepartmentById(id);
  return dept?.name || id.charAt(0).toUpperCase() + id.slice(1);
}

// Initialize with default state
selectedDepartment.subscribe((dept) => {
  if (dept) {
    fetchDepartmentMail(dept);
  }
});
