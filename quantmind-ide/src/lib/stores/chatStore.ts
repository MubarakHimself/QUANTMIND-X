// Chat Store - Manages chat state with Svelte stores
import { writable, derived, get } from 'svelte/store';
import type { Readable, Writable } from 'svelte/store';
import { chatManager } from '../services/chatManager';

// Type definitions
export type AgentType = 'copilot' | 'quantcode' | 'analyst';

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  model?: string;
  tokenCount?: number;
  latency?: number;
  metadata?: Record<string, unknown>;
}

export interface FileReference {
  id: string;
  name: string;
  path: string;
  type: string;
  size?: number;
}

export interface StrategyReference {
  id: string;
  name: string;
  type: string;
}

export interface BrokerReference {
  id: string;
  name: string;
  status: 'connected' | 'disconnected';
}

export interface BacktestReference {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
}

export interface ChatContext {
  files: FileReference[];
  strategies: StrategyReference[];
  brokers: BrokerReference[];
  backtests: BacktestReference[];
}

export interface Chat {
  id: string;
  title: string;
  agent: AgentType;
  messages: Message[];
  context: ChatContext;
  createdAt: Date;
  lastMessageAt: Date;
  isPinned: boolean;
  tags: string[];
}

// Store state interface
interface ChatStoreState {
  chats: Chat[];
  activeChatId: string | null;
  activeAgent: AgentType;
  isLoading: boolean;
  error: string | null;
}

// Initial state
const initialState: ChatStoreState = {
  chats: [],
  activeChatId: null,
  activeAgent: 'copilot',
  isLoading: false,
  error: null
};

// Create the main store
function createChatStore() {
  const { subscribe, set, update }: Writable<ChatStoreState> = writable(initialState);

  return {
    subscribe,
    
    // Initialize store from localStorage
    async initialize() {
      update(state => ({ ...state, isLoading: true }));
      try {
        const chats = await chatManager.loadAllChats();
        const activeChatId = chats.length > 0 ? chats[0].id : null;
        set({
          ...initialState,
          chats,
          activeChatId,
          isLoading: false
        });
      } catch (error) {
        update(state => ({
          ...state,
          isLoading: false,
          error: 'Failed to load chats'
        }));
      }
    },

    // Create a new chat
    createChat(agent: AgentType): Chat {
      const newChat = chatManager.createChat(agent);
      update(state => ({
        ...state,
        chats: [newChat, ...state.chats],
        activeChatId: newChat.id
      }));
      return newChat;
    },

    // Select a chat
    selectChat(chatId: string) {
      update(state => ({
        ...state,
        activeChatId: chatId
      }));
    },

    // Delete a chat
    deleteChat(chatId: string) {
      update(state => {
        const chats = state.chats.filter(c => c.id !== chatId);
        const activeChatId = state.activeChatId === chatId
          ? (chats.length > 0 ? chats[0].id : null)
          : state.activeChatId;
        
        chatManager.deleteChat(chatId);
        
        return {
          ...state,
          chats,
          activeChatId
        };
      });
    },

    // Pin/unpin a chat
    togglePinChat(chatId: string) {
      update(state => {
        const chats = state.chats.map(chat => {
          if (chat.id === chatId) {
            const updated = { ...chat, isPinned: !chat.isPinned };
            chatManager.updateChat(updated);
            return updated;
          }
          return chat;
        });
        return { ...state, chats };
      });
    },

    // Add a message to the active chat (creates chat if none exists)
    addMessage(message: Omit<Message, 'id' | 'timestamp'>): Message | null {
      let state = get({ subscribe });
      
      // If no active chat exists, create one first
      if (!state.activeChatId) {
        const newChat = chatManager.createChat(state.activeAgent);
        update(s => ({
          ...s,
          chats: [newChat, ...s.chats],
          activeChatId: newChat.id
        }));
        state = get({ subscribe });
        
        // Log creation for user feedback
        console.log(`Created new chat for agent ${state.activeAgent} before adding message`);
      }

      const newMessage: Message = {
        ...message,
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date()
      };

      update(state => {
        const chats = state.chats.map(chat => {
          if (chat.id === state.activeChatId) {
            const updated = {
              ...chat,
              messages: [...chat.messages, newMessage],
              lastMessageAt: new Date(),
              title: chat.messages.length === 0 && message.role === 'user'
                ? message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '')
                : chat.title
            };
            chatManager.updateChat(updated);
            return updated;
          }
          return chat;
        });
        return { ...state, chats };
      });

      return newMessage;
    },

    // Update a message
    updateMessage(messageId: string, updates: Partial<Message>) {
      update(state => {
        const chats = state.chats.map(chat => {
          if (chat.id === state.activeChatId) {
            const updated = {
              ...chat,
              messages: chat.messages.map(msg =>
                msg.id === messageId ? { ...msg, ...updates } : msg
              )
            };
            chatManager.updateChat(updated);
            return updated;
          }
          return chat;
        });
        return { ...state, chats };
      });
    },

    // Update chat context
    updateContext(context: Partial<ChatContext>) {
      update(state => {
        const chats = state.chats.map(chat => {
          if (chat.id === state.activeChatId) {
            const updated = {
              ...chat,
              context: { ...chat.context, ...context }
            };
            chatManager.updateChat(updated);
            return updated;
          }
          return chat;
        });
        return { ...state, chats };
      });
    },

    // Add context item (creates chat if none exists)
    addContextItem(type: keyof ChatContext, item: FileReference | StrategyReference | BrokerReference | BacktestReference) {
      // Ensure a chat exists before adding context
      let state = get({ subscribe });
      if (!state.activeChatId) {
        const newChat = chatManager.createChat(state.activeAgent);
        update(s => ({
          ...s,
          chats: [newChat, ...s.chats],
          activeChatId: newChat.id
        }));
        console.log(`Created new chat for agent ${state.activeAgent} before adding context`);
      }
      
      update(state => {
        // If still no active chat, return unchanged state
        if (!state.activeChatId) {
          console.warn('Failed to create chat for context item');
          return state;
        }
        
        const chats = state.chats.map(chat => {
          if (chat.id === state.activeChatId) {
            const updated = {
              ...chat,
              context: {
                ...chat.context,
                [type]: [...chat.context[type], item]
              }
            };
            chatManager.updateChat(updated);
            return updated;
          }
          return chat;
        });
        return { ...state, chats };
      });
    },

    // Remove context item
    removeContextItem(type: keyof ChatContext, itemId: string) {
      update(state => {
        const chats = state.chats.map(chat => {
          if (chat.id === state.activeChatId) {
            const updated = {
              ...chat,
              context: {
                ...chat.context,
                [type]: chat.context[type].filter(item => item.id !== itemId)
              }
            };
            chatManager.updateChat(updated);
            return updated;
          }
          return chat;
        });
        return { ...state, chats };
      });
    },

    // Switch agent
    switchAgent(agent: AgentType) {
      update(state => {
        // Find or create a chat for this agent
        let activeChatId = state.activeChatId;
        const agentChats = state.chats.filter(c => c.agent === agent);
        
        if (agentChats.length === 0) {
          // Create a new chat for this agent
          const newChat = chatManager.createChat(agent);
          return {
            ...state,
            activeAgent: agent,
            chats: [newChat, ...state.chats],
            activeChatId: newChat.id
          };
        } else {
          // Select the most recent chat for this agent
          activeChatId = agentChats.sort((a, b) => 
            new Date(b.lastMessageAt).getTime() - new Date(a.lastMessageAt).getTime()
          )[0].id;
        }
        
        return {
          ...state,
          activeAgent: agent,
          activeChatId
        };
      });
    },

    // Search chats
    searchChats(query: string) {
      const state = get({ subscribe });
      if (!query.trim()) return state.chats;
      
      const lowerQuery = query.toLowerCase();
      return state.chats.filter(chat =>
        chat.title.toLowerCase().includes(lowerQuery) ||
        chat.messages.some(msg => msg.content.toLowerCase().includes(lowerQuery))
      );
    },

    // Clear error
    clearError() {
      update(state => ({ ...state, error: null }));
    },

    // Clear messages for a specific chat
    clearMessages(chatId: string) {
      update(state => {
        const chats = state.chats.map(chat => {
          if (chat.id === chatId) {
            const updated = {
              ...chat,
              messages: [],
              lastMessageAt: new Date()
            };
            chatManager.updateChat(updated);
            return updated;
          }
          return chat;
        });
        return { ...state, chats };
      });
    },

    // Export chats
    exportChats(): string {
      const state = get({ subscribe });
      return JSON.stringify(state.chats, null, 2);
    },

    // Import chats
    importChats(jsonData: string) {
      try {
        const chats = JSON.parse(jsonData) as Chat[];
        update(state => ({
          ...state,
          chats: [...chats, ...state.chats]
        }));
        chats.forEach(chat => chatManager.updateChat(chat));
      } catch (error) {
        update(state => ({
          ...state,
          error: 'Failed to import chats'
        }));
      }
    }
  };
}

// Export the store instance
export const chatStore = createChatStore();

// Derived stores for computed values
export const activeChat: Readable<Chat | null> = derived(
  chatStore,
  $store => $store.chats.find(c => c.id === $store.activeChatId) || null
);

export const activeMessages: Readable<Message[]> = derived(
  activeChat,
  $chat => $chat?.messages || []
);

export const activeContext: Readable<ChatContext> = derived(
  activeChat,
  $chat => $chat?.context || { files: [], strategies: [], brokers: [], backtests: [] }
);

export const pinnedChats: Readable<Chat[]> = derived(
  chatStore,
  $store => $store.chats.filter(c => c.isPinned).sort((a, b) => 
    new Date(b.lastMessageAt).getTime() - new Date(a.lastMessageAt).getTime()
  )
);

export const unpinnedChats: Readable<Chat[]> = derived(
  chatStore,
  $store => $store.chats.filter(c => !c.isPinned).sort((a, b) => 
    new Date(b.lastMessageAt).getTime() - new Date(a.lastMessageAt).getTime()
  )
);

export const agentChats: Readable<(agent: AgentType) => Chat[]> = derived(
  chatStore,
  $store => (agent: AgentType) => $store.chats.filter(c => c.agent === agent)
);

// Agent greetings
export const agentGreetings: Record<AgentType, string> = {
  copilot: "Hello! I'm the QuantMind Copilot. I can help analyze strategies, run backtests, and manage bots.",
  quantcode: "I'm QuantCode. I can help write MQ5 code, debug EAs, and optimize parameters.",
  analyst: "I'm the Analyst. I analyze NPRD outputs and help interpret trading patterns."
};
