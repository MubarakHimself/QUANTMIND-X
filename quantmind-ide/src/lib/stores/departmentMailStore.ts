/**
 * Department Mail Store
 *
 * State management for cross-department messages in the Trading Floor.
 * Provides inbox, sent items, and unread count tracking.
 */

import { writable, derived, get } from 'svelte/store';

// Department colors for consistent styling
export const DEPARTMENT_COLORS: Record<string, string> = {
	analysis: '#3b82f6',    // blue
	research: '#8b5cf6',    // purple
	risk: '#ef4444',        // red
	execution: '#f97316',   // orange
	portfolio: '#10b981',   // green
};

// Priority levels with colors
export const PRIORITY_COLORS: Record<string, string> = {
	low: '#6b7280',      // gray
	normal: '#3b82f6',   // blue
	high: '#f59e0b',     // amber
	urgent: '#ef4444',   // red
};

export type MessagePriority = 'low' | 'normal' | 'high' | 'urgent';
export type MessageType = 'dispatch' | 'result' | 'question' | 'status' | 'escalation' | 'health_check';

export type DepartmentMailMessage = {
	id: string;
	from_department: string;
	to_department: string;
	from_agent?: string;
	to_agent?: string;
	subject: string;
	body: string;
	message_type: MessageType;
	priority: MessagePriority;
	is_read: boolean;
	created_at: string;
	read_at?: string;
	metadata?: Record<string, unknown>;
}

export type DepartmentMailStats = {
	total_unread: number;
	unread_by_department: Record<string, number>;
	total_inbox: number;
	total_sent: number;
	by_priority: Record<MessagePriority, number>;
}

export type DepartmentMailState = {
	inbox: DepartmentMailMessage[];
	sent: DepartmentMailMessage[];
	selectedMessage: DepartmentMailMessage | null;
	stats: DepartmentMailStats;
	loading: boolean;
	error: string | null;
	selectedDepartment: string | null;
}

// Initial state
const initialState: DepartmentMailState = {
	inbox: [],
	sent: [],
	selectedMessage: null,
	stats: {
		total_unread: 0,
		unread_by_department: {},
		total_inbox: 0,
		total_sent: 0,
		by_priority: { low: 0, normal: 0, high: 0, urgent: 0 },
	},
	loading: false,
	error: null,
	selectedDepartment: null,
};

// Create stores
const departmentMailStore = writable<DepartmentMailState>(initialState);

// Derived stores for convenience
export const inbox = derived(departmentMailStore, ($store) => $store.inbox);
export const sentMessages = derived(departmentMailStore, ($store) => $store.sent);
export const selectedMessage = derived(departmentMailStore, ($store) => $store.selectedMessage);
export const mailStats = derived(departmentMailStore, ($store) => $store.stats);
export const mailLoading = derived(departmentMailStore, ($store) => $store.loading);
export const mailError = derived(departmentMailStore, ($store) => $store.error);
export const selectedDepartment = derived(departmentMailStore, ($store) => $store.selectedDepartment);
export const unreadCount = derived(departmentMailStore, ($store) => $store.stats.total_unread);

// Derived store for unread messages
export const unreadMessages = derived(departmentMailStore, ($store) =>
	$store.inbox.filter((msg) => !msg.is_read)
);

// Derived store for filtered inbox by department
export const filteredInbox = derived(departmentMailStore, ($store) => {
	if (!$store.selectedDepartment) {
		return $store.inbox;
	}
	return $store.inbox.filter((msg) => msg.to_department === $store.selectedDepartment);
});

// API base URL
const API_BASE = 'http://localhost:8000/api';

/**
 * Set loading state
 */
export function setLoading(loading: boolean): void {
	departmentMailStore.update((state) => ({ ...state, loading }));
}

/**
 * Set error state
 */
export function setError(error: string | null): void {
	departmentMailStore.update((state) => ({ ...state, error }));
}

/**
 * Set selected department filter
 */
export function setSelectedDepartment(department: string | null): void {
	departmentMailStore.update((state) => ({ ...state, selectedDepartment: department }));
}

/**
 * Set selected message
 */
export function setSelectedMessage(message: DepartmentMailMessage | null): void {
	departmentMailStore.update((state) => ({ ...state, selectedMessage: message }));
}

/**
 * Update inbox messages
 */
export function setInbox(messages: DepartmentMailMessage[]): void {
	departmentMailStore.update((state) => {
		const stats = calculateStats(messages, state.sent);
		return { ...state, inbox: messages, stats };
	});
}

/**
 * Update sent messages
 */
export function setSent(messages: DepartmentMailMessage[]): void {
	departmentMailStore.update((state) => {
		const stats = calculateStats(state.inbox, messages);
		return { ...state, sent: messages, stats };
	});
}

/**
 * Add a new message to inbox
 */
export function addMessageToInbox(message: DepartmentMailMessage): void {
	departmentMailStore.update((state) => {
		const inbox = [message, ...state.inbox];
		const stats = calculateStats(inbox, state.sent);
		return { ...state, inbox, stats };
	});
}

// API response field mapping - transforms API response to store format
function mapApiMessageToStore(apiMsg: Record<string, unknown>): DepartmentMailMessage {
	return {
		id: apiMsg.id as string,
		from_department: (apiMsg.from_dept || apiMsg.from_department) as string,
		to_department: (apiMsg.to_dept || apiMsg.to_department) as string,
		from_agent: apiMsg.from_agent as string | undefined,
		to_agent: apiMsg.to_agent as string | undefined,
		subject: apiMsg.subject as string,
		body: apiMsg.body as string,
		message_type: (apiMsg.type || apiMsg.message_type) as MessageType,
		priority: (apiMsg.priority || 'normal') as MessagePriority,
		is_read: (apiMsg.read ?? apiMsg.is_read ?? false) as boolean,
		created_at: (apiMsg.timestamp || apiMsg.created_at) as string,
		read_at: apiMsg.read_at as string | undefined,
		metadata: apiMsg.metadata as Record<string, unknown> | undefined,
	};
}

/**
 * Calculate stats from messages
 */
function calculateStats(inbox: DepartmentMailMessage[], sent: DepartmentMailMessage[]): DepartmentMailStats {
	const unreadByDept: Record<string, number> = {};
	const byPriority: Record<MessagePriority, number> = { low: 0, normal: 0, high: 0, urgent: 0 };

	let totalUnread = 0;

	inbox.forEach((msg) => {
		if (!msg.is_read) {
			totalUnread++;
			unreadByDept[msg.to_department] = (unreadByDept[msg.to_department] || 0) + 1;
		}
		byPriority[msg.priority]++;
	});

	return {
		total_unread: totalUnread,
		unread_by_department: unreadByDept,
		total_inbox: inbox.length,
		total_sent: sent.length,
		by_priority: byPriority,
	};
}

/**
 * Fetch inbox for a specific department
 */
export async function fetchInbox(department?: string): Promise<void> {
	setLoading(true);
	setError(null);

	try {
		const dept = department || get(departmentMailStore).selectedDepartment;
		const url = dept
			? `${API_BASE}/departments/mail/inbox/${dept}`
			: `${API_BASE}/departments/mail/inbox`;

		const response = await fetch(url);

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		const data = await response.json();
		const messages = (data.messages || []).map(mapApiMessageToStore);
		setInbox(messages);
	} catch (e) {
		setError(e instanceof Error ? e.message : 'Failed to fetch inbox');
	} finally {
		setLoading(false);
	}
}

/**
 * Fetch all inbox messages (all departments)
 */
export async function fetchAllInbox(): Promise<void> {
	setLoading(true);
	setError(null);

	try {
		const response = await fetch(`${API_BASE}/departments/mail/inbox`);

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		const data = await response.json();
		const messages = (data.messages || []).map(mapApiMessageToStore);
		setInbox(messages);
	} catch (e) {
		setError(e instanceof Error ? e.message : 'Failed to fetch inbox');
	} finally {
		setLoading(false);
	}
}

/**
 * Fetch sent messages
 */
export async function fetchSent(): Promise<void> {
	setLoading(true);
	setError(null);

	try {
		const response = await fetch(`${API_BASE}/departments/mail/sent`);

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		const data = await response.json();
		const messages = (data.messages || []).map(mapApiMessageToStore);
		setSent(messages);
	} catch (e) {
		setError(e instanceof Error ? e.message : 'Failed to fetch sent messages');
	} finally {
		setLoading(false);
	}
}

/**
 * Fetch mail statistics
 */
export async function fetchStats(): Promise<void> {
	try {
		const response = await fetch(`${API_BASE}/departments/mail/stats`);

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		const apiStats = await response.json();
		departmentMailStore.update((state) => ({
			...state,
			stats: {
				total_unread: apiStats.unread_messages || apiStats.total_unread || 0,
				unread_by_department: apiStats.unread_by_department || {},
				total_inbox: apiStats.total_messages || apiStats.total_inbox || 0,
				total_sent: apiStats.total_sent || 0,
				by_priority: apiStats.by_priority || { low: 0, normal: 0, high: 0, urgent: 0 },
			},
		}));
	} catch (e) {
		console.error('Failed to fetch mail stats:', e);
	}
}

/**
 * Fetch a specific message by ID
 */
export async function fetchMessage(messageId: string): Promise<DepartmentMailMessage | null> {
	try {
		const response = await fetch(`${API_BASE}/departments/mail/${messageId}`);

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		const message = await response.json();
		const mappedMessage = mapApiMessageToStore(message);
		setSelectedMessage(mappedMessage);
		return mappedMessage;
	} catch (e) {
		setError(e instanceof Error ? e.message : 'Failed to fetch message');
		return null;
	}
}

/**
 * Mark a message as read
 */
export async function markAsRead(messageId: string): Promise<boolean> {
	try {
		const response = await fetch(`${API_BASE}/departments/mail/${messageId}/read`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
		});

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		// Update local state
		departmentMailStore.update((state) => {
			const inbox = state.inbox.map((msg) =>
				msg.id === messageId
					? { ...msg, is_read: true, read_at: new Date().toISOString() }
					: msg
			);
			const stats = calculateStats(inbox, state.sent);
			const selectedMessage =
				state.selectedMessage?.id === messageId
					? { ...state.selectedMessage, is_read: true, read_at: new Date().toISOString() }
					: state.selectedMessage;

			return { ...state, inbox, stats, selectedMessage };
		});

		return true;
	} catch (e) {
		setError(e instanceof Error ? e.message : 'Failed to mark message as read');
		return false;
	}
}

/**
 * Mark all messages as read for a department
 */
export async function markAllAsRead(department?: string): Promise<boolean> {
	try {
		const dept = department || get(departmentMailStore).selectedDepartment;
		const url = dept
			? `${API_BASE}/departments/mail/read-all/${dept}`
			: `${API_BASE}/departments/mail/read-all`;

		const response = await fetch(url, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
		});

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		// Refresh inbox
		await fetchAllInbox();
		return true;
	} catch (e) {
		setError(e instanceof Error ? e.message : 'Failed to mark all as read');
		return false;
	}
}

/**
 * Send a new message
 */
export async function sendMessage(message: {
	to_department: string;
	subject: string;
	body: string;
	message_type: MessageType;
	priority?: MessagePriority;
	from_department?: string;
	from_agent?: string;
	to_agent?: string;
	metadata?: Record<string, unknown>;
}): Promise<DepartmentMailMessage | null> {
	setLoading(true);
	setError(null);

	try {
		const response = await fetch(`${API_BASE}/departments/mail/send`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				...message,
				priority: message.priority || 'normal',
			}),
		});

		if (!response.ok) {
			throw new Error(`HTTP ${response.status}: ${response.statusText}`);
		}

		const sentMessage = await response.json();

		// Add to sent messages
		departmentMailStore.update((state) => {
			const sent = [sentMessage, ...state.sent];
			const stats = calculateStats(state.inbox, sent);
			return { ...state, sent, stats };
		});

		return sentMessage;
	} catch (e) {
		setError(e instanceof Error ? e.message : 'Failed to send message');
		return null;
	} finally {
		setLoading(false);
	}
}

/**
 * Clear all mail state
 */
export function clearMail(): void {
	departmentMailStore.set(initialState);
}

/**
 * Refresh all mail data
 */
export async function refreshMail(): Promise<void> {
	await Promise.all([fetchAllInbox(), fetchSent(), fetchStats()]);
}

// Export the main store
export { departmentMailStore };
