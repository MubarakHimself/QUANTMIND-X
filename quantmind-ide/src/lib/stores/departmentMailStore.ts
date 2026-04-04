/**
 * Department Mail Store
 *
 * State management for department mail in the Trading Floor.
 * Handles fetching, sending, and managing mail between departments.
 */

import { writable, derived, get } from 'svelte/store';
import { apiFetch } from '$lib/api';

// Types
export type MessageType = 'dispatch' | 'result' | 'question' | 'status' | 'error' | 'approval_request' | 'approval_approved' | 'approval_rejected';

export type Priority = 'low' | 'normal' | 'high' | 'urgent';

// Alias for MessagePriority (used by DepartmentMailPanel)
export type MessagePriority = Priority;

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
  // Approval-related fields
  gate_id?: string;
  workflow_id?: string;
  from_stage?: string;
  to_stage?: string;
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
  { id: 'development', name: 'Development', description: 'Market analysis and signals', color: '#3b82f6' },
  { id: 'research', name: 'Research', description: 'Strategy development and backtesting', color: '#8b5cf6' },
  { id: 'risk', name: 'Risk', description: 'Position sizing and risk management', color: '#ef4444' },
  { id: 'trading', name: 'Trading', description: 'Order execution and routing', color: '#f97316' },
  { id: 'portfolio', name: 'Portfolio', description: 'Portfolio management and allocation', color: '#10b981' },
] as const;

// Department colors for UI
export const DEPARTMENT_COLORS: Record<string, string> = {
  development: '#3b82f6',
  research: '#8b5cf6',
  risk: '#ef4444',
  trading: '#f97316',
  portfolio: '#10b981',
};

// Priority colors for UI
export const PRIORITY_COLORS: Record<Priority, string> = {
  low: '#6b7280',
  normal: '#3b82f6',
  high: '#f59e0b',
  urgent: '#ef4444',
};

export type Department = typeof DEPARTMENTS[number]['id'];

// Store state
const mailMessages = writable<Map<string, DepartmentMailMessage[]>>(new Map());
const loading = writable(false);
const error = writable<string | null>(null);
const selectedDepartment = writable<Department | null>(null);
const unreadCounts = writable<Map<string, number>>(new Map());

// Export writable stores for component access
export { loading as mailLoading, error as mailError, selectedDepartment };

// Unread count for selected department
export const unreadCount = derived(
  [unreadCounts, selectedDepartment],
  ([$unreadCounts, $selectedDepartment]) => {
    if (!$selectedDepartment) return 0;
    return $unreadCounts.get($selectedDepartment) || 0;
  }
);

// All messages across departments (for inbox view)
export const allMessages = derived(mailMessages, ($mailMessages) => {
  const all: DepartmentMailMessage[] = [];
  for (const messages of $mailMessages.values()) {
    all.push(...messages);
  }
  return all.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
});

// Filter inbox by selected department
export const filteredInbox = derived(
  [allMessages, selectedDepartment],
  ([$allMessages, $selectedDepartment]) => {
    if (!$selectedDepartment) return $allMessages;
    return $allMessages.filter((m) => m.to === $selectedDepartment);
  }
);

// Messages sent by selected department
export const sentMessages = derived(
  [allMessages, selectedDepartment],
  ([$allMessages, $selectedDepartment]) => {
    if (!$selectedDepartment) return [];
    return $allMessages.filter((m) => m.from === $selectedDepartment);
  }
);

// Mail statistics
export const mailStats = derived(mailMessages, ($mailMessages) => {
  const stats = {
    total: 0,
    unread: 0,
    byDepartment: {} as Record<string, { total: number; unread: number }>,
  };

  for (const [dept, messages] of $mailMessages.entries()) {
    const unread = messages.filter((m) => !m.read).length;
    stats.total += messages.length;
    stats.unread += unread;
    stats.byDepartment[dept] = { total: messages.length, unread };
  }

  return stats;
});

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
    const data = await apiFetch<{ messages?: DepartmentMailMessage[] }>(`/departments/mail/inbox/${department}`);
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
    await apiFetch(`/departments/mail/${messageId}/read`, { method: 'POST' });

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
    const data = await apiFetch<DelegationResponse>('/trading-floor/delegate', {
      method: 'POST',
      body: JSON.stringify(request),
    });

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

// Fetch all messages for all departments
export async function fetchAllInbox(): Promise<void> {
  loading.set(true);
  error.set(null);

  try {
    const departments: Department[] = ['development', 'research', 'risk', 'trading', 'portfolio'];
    for (const dept of departments) {
      await fetchDepartmentMail(dept);
    }
  } catch (e) {
    const message = e instanceof Error ? e.message : 'Unknown error';
    error.set(message);
    console.error('Failed to fetch all mail:', e);
  } finally {
    loading.set(false);
  }
}

// Fetch sent messages for current department
export async function fetchSent(): Promise<void> {
  // Same as filteredInbox, the component will use the derived store
  const dept = get(selectedDepartment);
  if (dept) {
    await fetchDepartmentMail(dept);
  }
}

// Fetch statistics
export async function fetchStats(): Promise<void> {
  // Stats are derived from mailMessages, just trigger a refresh
  const dept = get(selectedDepartment);
  if (dept) {
    await fetchDepartmentMail(dept);
  }
}

// Fetch single message
export async function fetchMessage(messageId: string): Promise<DepartmentMailMessage | null> {
  const all = get(allMessages);
  return all.find((m) => m.id === messageId) || null;
}

// Set selected department wrapper
export { selectDepartment as setSelectedDepartment };

// Alias for markMessageRead (used by DepartmentMailPanel)
export { markMessageRead as markAsRead };

// Clear all mail
export function clearMail(): void {
  mailMessages.set(new Map());
  unreadCounts.set(new Map());
}

// Get selected message wrapper
export function getSelectedMessage(): DepartmentMailMessage | null {
  const dept = get(selectedDepartment);
  if (!dept) return null;
  const list = get(mailList);
  return list.length > 0 ? list[0] : null;
}

// Set selected message (placeholder for future use)
export function setSelectedMessage(message: DepartmentMailMessage | null): void {
  // Could be used to show message detail
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
