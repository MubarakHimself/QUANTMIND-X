/**
 * Department Chat Store
 *
 * State management for department-based chat in the Trading Floor.
 * Handles chat state per department, message history, and delegation.
 */

import { writable, derived, get } from 'svelte/store';
import type { Readable, Writable } from 'svelte/store';

// Department types
export type DepartmentId = 'development' | 'research' | 'risk' | 'trading' | 'portfolio';

export type DepartmentInfo = {
  id: DepartmentId;
  name: string;
  icon: string;
  color: string;
  description: string;
};

// Department definitions
export const DEPARTMENTS: Record<DepartmentId, DepartmentInfo> = {
  development: {
    id: 'development',
    name: 'Development',
    icon: 'BarChart2',
    color: '#3b82f6',
    description: 'Market analysis and signal generation',
  },
  research: {
    id: 'research',
    name: 'Research',
    icon: 'FlaskConical',
    color: '#8b5cf6',
    description: 'Strategy research and backtesting',
  },
  risk: {
    id: 'risk',
    name: 'Risk',
    icon: 'Shield',
    color: '#ef4444',
    description: 'Risk management and position sizing',
  },
  trading: {
    id: 'trading',
    name: 'Trading',
    icon: 'Zap',
    color: '#f97316',
    description: 'Order execution and routing',
  },
  portfolio: {
    id: 'portfolio',
    name: 'Portfolio',
    icon: 'Briefcase',
    color: '#10b981',
    description: 'Portfolio management and rebalancing',
  },
};

// Message types
export type DepartmentMessageRole = 'user' | 'department' | 'system';

export interface DepartmentMessage {
  id: string;
  role: DepartmentMessageRole;
  content: string;
  timestamp: Date;
  department?: DepartmentId;
  metadata?: {
    taskType?: 'analysis' | 'backtest' | 'risk_check' | 'order' | 'rebalance';
    status?: 'pending' | 'in_progress' | 'completed' | 'error';
    delegatedBy?: 'floor_manager' | 'user';
    duration?: number;
  };
}

// Chat history per department
export interface DepartmentChatHistory {
  departmentId: DepartmentId;
  messages: DepartmentMessage[];
  lastMessageAt: Date | null;
  unreadCount: number;
  isTyping: boolean;
}

// Task delegation status
export interface DelegatedTask {
  id: string;
  departmentId: DepartmentId;
  request: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  result?: string;
  createdAt: Date;
  completedAt?: Date;
}

// Store state interface
interface DepartmentChatStoreState {
  chats: Map<DepartmentId, DepartmentChatHistory>;
  activeDepartment: DepartmentId | null;
  delegatedTasks: Map<string, DelegatedTask>;
  isLoading: boolean;
  error: string | null;
}

// Initial state
const createInitialChats = (): Map<DepartmentId, DepartmentChatHistory> => {
  const chats = new Map<DepartmentId, DepartmentChatHistory>();
  Object.keys(DEPARTMENTS).forEach((deptId) => {
    chats.set(deptId as DepartmentId, {
      departmentId: deptId as DepartmentId,
      messages: [],
      lastMessageAt: null,
      unreadCount: 0,
      isTyping: false,
    });
  });
  return chats;
};

const initialState: DepartmentChatStoreState = {
  chats: createInitialChats(),
  activeDepartment: null,
  delegatedTasks: new Map(),
  isLoading: false,
  error: null,
};

// API base URL
const API_BASE = 'http://localhost:8000/api';

// Create the store
function createDepartmentChatStore() {
  const { subscribe, set, update }: Writable<DepartmentChatStoreState> = writable(initialState);

  // Generate unique message ID
  function generateMessageId(): string {
    return `dept_msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // Generate unique task ID
  function generateTaskId(): string {
    return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  return {
    subscribe,

    // Set active department for chat
    setActiveDepartment(departmentId: DepartmentId | null) {
      update((state) => {
        // Clear unread count when switching to department
        if (departmentId) {
          const chat = state.chats.get(departmentId);
          if (chat) {
            chat.unreadCount = 0;
          }
        }
        return { ...state, activeDepartment: departmentId };
      });
    },

    // Get chat history for a department
    getHistory(departmentId: DepartmentId): DepartmentMessage[] {
      const state = get({ subscribe });
      const chat = state.chats.get(departmentId);
      return chat ? chat.messages : [];
    },

    // Clear chat history for a department
    clearHistory(departmentId: DepartmentId) {
      update((state) => {
        const chat = state.chats.get(departmentId);
        if (chat) {
          chat.messages = [];
          chat.lastMessageAt = null;
        }
        return { ...state, chats: new Map(state.chats) };
      });
    },

    // Clear all chat histories
    clearAllHistory() {
      update((state) => {
        const newChats = createInitialChats();
        return { ...state, chats: newChats };
      });
    },

    // Set typing indicator for department
    setTyping(departmentId: DepartmentId, isTyping: boolean) {
      update((state) => {
        const chat = state.chats.get(departmentId);
        if (chat) {
          chat.isTyping = isTyping;
        }
        return { ...state, chats: new Map(state.chats) };
      });
    },

    // Send message to a department
    async sendMessage(
      departmentId: DepartmentId,
      content: string,
      options?: { delegatedBy?: 'floor_manager' | 'user'; taskType?: DelegatedTask['request'] extends any ? never : string }
    ): Promise<DepartmentMessage | null> {
      const userMessage: DepartmentMessage = {
        id: generateMessageId(),
        role: 'user',
        content,
        timestamp: new Date(),
        metadata: {
          delegatedBy: options?.delegatedBy || 'user',
        },
      };

      // Add user message to history
      update((state) => {
        const chat = state.chats.get(departmentId);
        if (chat) {
          chat.messages = [...chat.messages, userMessage];
          chat.lastMessageAt = new Date();
        }
        return { ...state, chats: new Map(state.chats), isLoading: true };
      });

      try {
        // Set typing indicator
        this.setTyping(departmentId, true);

        // Call API to send message to department
        const response = await fetch(`${API_BASE}/chat/departments/${departmentId}/message`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: content,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        const assistantMessage: DepartmentMessage = {
          id: generateMessageId(),
          role: 'department',
          content: data.reply || data.result || 'Task completed.',
          timestamp: new Date(),
          department: departmentId,
          metadata: {
            taskType: data.task_type,
            status: data.status || 'completed',
            duration: data.duration,
          },
        };

        // Add response to history
        update((state) => {
          const chat = state.chats.get(departmentId);
          if (chat) {
            chat.messages = [...chat.messages, assistantMessage];
            chat.lastMessageAt = new Date();
            chat.isTyping = false;
          }
          return { ...state, chats: new Map(state.chats), isLoading: false };
        });

        return assistantMessage;
      } catch (error) {
        const errorMessage: DepartmentMessage = {
          id: generateMessageId(),
          role: 'system',
          content: `Error: ${error instanceof Error ? error.message : 'Failed to send message'}`,
          timestamp: new Date(),
        };

        update((state) => {
          const chat = state.chats.get(departmentId);
          if (chat) {
            chat.messages = [...chat.messages, errorMessage];
            chat.isTyping = false;
          }
          return {
            ...state,
            chats: new Map(state.chats),
            isLoading: false,
            error: error instanceof Error ? error.message : 'Failed to send message',
          };
        });

        return null;
      }
    },

    // Delegate task to department via Floor Manager
    async delegateTask(
      departmentId: DepartmentId,
      task: string,
      taskType?: DelegatedTask['request'] extends any ? never : string
    ): Promise<DelegatedTask | null> {
      const delegatedTask: DelegatedTask = {
        id: generateTaskId(),
        departmentId,
        request: task,
        status: 'pending',
        createdAt: new Date(),
      };

      // Add task to tracking
      update((state) => {
        state.delegatedTasks.set(delegatedTask.id, delegatedTask);
        return { ...state, delegatedTasks: new Map(state.delegatedTasks) };
      });

      try {
        const response = await fetch(`${API_BASE}/trading-floor/delegate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            from_department: 'floor_manager',
            to_department: departmentId,
            task,
            task_type: taskType,
            suggested_department: departmentId,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        // Update task status
        update((state) => {
          const existingTask = state.delegatedTasks.get(delegatedTask.id);
          if (existingTask) {
            existingTask.status = data.status || 'completed';
            existingTask.result = data.result;
            existingTask.completedAt = new Date();
          }
          return { ...state, delegatedTasks: new Map(state.delegatedTasks) };
        });

        // Add messages to department chat
        const systemMessage: DepartmentMessage = {
          id: generateMessageId(),
          role: 'system',
          content: `Task delegated to ${DEPARTMENTS[departmentId].name}: ${task}`,
          timestamp: new Date(),
          department: departmentId,
          metadata: {
            delegatedBy: 'floor_manager',
            status: 'completed',
          },
        };

        const resultMessage: DepartmentMessage = {
          id: generateMessageId(),
          role: 'department',
          content: data.result || 'Task completed successfully.',
          timestamp: new Date(),
          department: departmentId,
          metadata: {
            taskType: taskType as any,
            status: data.status || 'completed',
          },
        };

        update((state) => {
          const chat = state.chats.get(departmentId);
          if (chat) {
            chat.messages = [...chat.messages, systemMessage, resultMessage];
            chat.lastMessageAt = new Date();
          }
          return { ...state, chats: new Map(state.chats) };
        });

        return get({ subscribe }).delegatedTasks.get(delegatedTask.id) || null;
      } catch (error) {
        update((state) => {
          const existingTask = state.delegatedTasks.get(delegatedTask.id);
          if (existingTask) {
            existingTask.status = 'error';
            existingTask.result = error instanceof Error ? error.message : 'Delegation failed';
            existingTask.completedAt = new Date();
          }
          return {
            ...state,
            delegatedTasks: new Map(state.delegatedTasks),
            error: error instanceof Error ? error.message : 'Delegation failed',
          };
        });

        return null;
      }
    },

    // Get active delegated tasks
    getActiveTasks(): DelegatedTask[] {
      const state = get({ subscribe });
      return Array.from(state.delegatedTasks.values()).filter(
        (task) => task.status === 'pending' || task.status === 'in_progress'
      );
    },

    // Get all delegated tasks
    getAllTasks(): DelegatedTask[] {
      const state = get({ subscribe });
      return Array.from(state.delegatedTasks.values());
    },

    // Clear completed tasks
    clearCompletedTasks() {
      update((state) => {
        const newTasks = new Map(state.delegatedTasks);
        newTasks.forEach((task, id) => {
          if (task.status === 'completed' || task.status === 'error') {
            newTasks.delete(id);
          }
        });
        return { ...state, delegatedTasks: newTasks };
      });
    },

    // Clear error
    clearError() {
      update((state) => ({ ...state, error: null }));
    },

    // Reset store
    reset() {
      set(initialState);
    },
  };
}

// Export the store instance
export const departmentChatStore = createDepartmentChatStore();

// Derived stores for convenience
export const activeDepartmentChat: Readable<DepartmentChatHistory | null> = derived(
  departmentChatStore,
  ($store) => {
    if (!$store.activeDepartment) return null;
    return $store.chats.get($store.activeDepartment) || null;
  }
);

export const activeDepartmentMessages: Readable<DepartmentMessage[]> = derived(
  activeDepartmentChat,
  ($chat) => $chat?.messages || []
);

export const totalUnreadCount: Readable<number> = derived(
  departmentChatStore,
  ($store) => {
    let count = 0;
    $store.chats.forEach((chat) => {
      count += chat.unreadCount;
    });
    return count;
  }
);

export const activeDelegatedTasks: Readable<DelegatedTask[]> = derived(
  departmentChatStore,
  ($store) => {
    return Array.from($store.delegatedTasks.values()).filter(
      (task) => task.status === 'pending' || task.status === 'in_progress'
    );
  }
);

// Department list for UI
export const departmentList = Object.values(DEPARTMENTS);
