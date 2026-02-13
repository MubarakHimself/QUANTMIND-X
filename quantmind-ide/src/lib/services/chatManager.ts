// Chat Manager Service - Handles chat persistence and operations
import type { Chat, Message, AgentType, ChatContext } from '../stores/chatStore';

const STORAGE_KEY = 'quantmind_chats';
const MAX_CHATS = 100;
const MAX_MESSAGES_PER_CHAT = 500;

// Generate unique ID
function generateId(): string {
  return `chat_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Chat Manager Service
export const chatManager = {
  // Load all chats from localStorage
  loadAllChats(): Chat[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) return [];
      
      const chats = JSON.parse(stored) as Chat[];
      
      // Convert date strings back to Date objects
      return chats.map(chat => ({
        ...chat,
        createdAt: new Date(chat.createdAt),
        lastMessageAt: new Date(chat.lastMessageAt),
        messages: chat.messages.map(msg => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
      }));
    } catch (error) {
      console.error('Failed to load chats from localStorage:', error);
      return [];
    }
  },

  // Create a new chat
  createChat(agent: AgentType): Chat {
    const now = new Date();
    const chat: Chat = {
      id: generateId(),
      title: 'New Chat',
      agent,
      messages: [],
      context: {
        files: [],
        strategies: [],
        brokers: [],
        backtests: []
      },
      createdAt: now,
      lastMessageAt: now,
      isPinned: false,
      tags: []
    };
    
    this.saveChat(chat);
    return chat;
  },

  // Save a single chat
  saveChat(chat: Chat): void {
    try {
      const chats = this.loadAllChats();
      const existingIndex = chats.findIndex(c => c.id === chat.id);
      
      if (existingIndex >= 0) {
        chats[existingIndex] = chat;
      } else {
        chats.unshift(chat);
      }
      
      // Enforce max chats limit
      const trimmedChats = this.enforceLimits(chats);
      
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmedChats));
    } catch (error) {
      console.error('Failed to save chat:', error);
    }
  },

  // Update an existing chat
  updateChat(chat: Chat): void {
    this.saveChat(chat);
  },

  // Delete a chat
  deleteChat(chatId: string): void {
    try {
      const chats = this.loadAllChats();
      const filtered = chats.filter(c => c.id !== chatId);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
    } catch (error) {
      console.error('Failed to delete chat:', error);
    }
  },

  // Get a single chat by ID
  getChat(chatId: string): Chat | null {
    const chats = this.loadAllChats();
    return chats.find(c => c.id === chatId) || null;
  },

  // Get chats by agent
  getChatsByAgent(agent: AgentType): Chat[] {
    const chats = this.loadAllChats();
    return chats.filter(c => c.agent === agent);
  },

  // Search chats by title or content
  searchChats(query: string): Chat[] {
    const chats = this.loadAllChats();
    const lowerQuery = query.toLowerCase();
    
    return chats.filter(chat =>
      chat.title.toLowerCase().includes(lowerQuery) ||
      chat.messages.some(msg => msg.content.toLowerCase().includes(lowerQuery)) ||
      chat.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
    );
  },

  // Auto-generate title from first message
  generateTitleFromMessage(content: string): string {
    // Take first 50 characters, clean up
    let title = content.trim().slice(0, 50);
    
    // Remove newlines and extra spaces
    title = title.replace(/\n/g, ' ').replace(/\s+/g, ' ');
    
    // Add ellipsis if truncated
    if (content.length > 50) {
      title += '...';
    }
    
    return title || 'New Chat';
  },

  // Add message to chat
  addMessage(chatId: string, message: Omit<Message, 'id' | 'timestamp'>): Chat | null {
    const chat = this.getChat(chatId);
    if (!chat) return null;
    
    const newMessage: Message = {
      ...message,
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    };
    
    // Update title if this is the first user message
    if (chat.messages.length === 0 && message.role === 'user') {
      chat.title = this.generateTitleFromMessage(message.content);
    }
    
    chat.messages.push(newMessage);
    chat.lastMessageAt = new Date();
    
    // Enforce message limit
    if (chat.messages.length > MAX_MESSAGES_PER_CHAT) {
      chat.messages = chat.messages.slice(-MAX_MESSAGES_PER_CHAT);
    }
    
    this.updateChat(chat);
    return chat;
  },

  // Clear all chats
  clearAllChats(): void {
    localStorage.removeItem(STORAGE_KEY);
  },

  // Export chats to JSON
  exportChats(): string {
    const chats = this.loadAllChats();
    return JSON.stringify(chats, null, 2);
  },

  // Import chats from JSON
  importChats(jsonData: string): boolean {
    try {
      const imported = JSON.parse(jsonData) as Chat[];
      
      // Validate structure
      if (!Array.isArray(imported)) {
        throw new Error('Invalid format: expected array');
      }
      
      // Convert dates and assign new IDs to avoid conflicts
      const chats = imported.map(chat => ({
        ...chat,
        id: generateId(), // New ID to avoid conflicts
        createdAt: new Date(chat.createdAt),
        lastMessageAt: new Date(chat.lastMessageAt),
        messages: chat.messages.map(msg => ({
          ...msg,
          id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          timestamp: new Date(msg.timestamp)
        }))
      }));
      
      // Merge with existing chats
      const existingChats = this.loadAllChats();
      const merged = [...chats, ...existingChats];
      const trimmed = this.enforceLimits(merged);
      
      localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
      return true;
    } catch (error) {
      console.error('Failed to import chats:', error);
      return false;
    }
  },

  // Enforce limits on chats
  enforceLimits(chats: Chat[]): Chat[] {
    // Sort by pinned first, then by lastMessageAt
    const sorted = chats.sort((a, b) => {
      if (a.isPinned && !b.isPinned) return -1;
      if (!a.isPinned && b.isPinned) return 1;
      return new Date(b.lastMessageAt).getTime() - new Date(a.lastMessageAt).getTime();
    });
    
    // Keep only MAX_CHATS
    return sorted.slice(0, MAX_CHATS);
  },

  // Get chat statistics
  getStats(): {
    totalChats: number;
    totalMessages: number;
    chatsByAgent: Record<AgentType, number>;
    oldestChat: Date | null;
    newestChat: Date | null;
  } {
    const chats = this.loadAllChats();
    
    const chatsByAgent: Record<AgentType, number> = {
      copilot: 0,
      quantcode: 0,
      analyst: 0
    };
    
    let totalMessages = 0;
    let oldestChat: Date | null = null;
    let newestChat: Date | null = null;
    
    chats.forEach(chat => {
      chatsByAgent[chat.agent]++;
      totalMessages += chat.messages.length;
      
      const createdAt = new Date(chat.createdAt);
      if (!oldestChat || createdAt < oldestChat) {
        oldestChat = createdAt;
      }
      if (!newestChat || createdAt > newestChat) {
        newestChat = createdAt;
      }
    });
    
    return {
      totalChats: chats.length,
      totalMessages,
      chatsByAgent,
      oldestChat,
      newestChat
    };
  },

  // Migrate old format chats (for backward compatibility)
  migrateOldFormat(oldChats: Record<string, Array<{id: string, title: string, date: string}>>): Chat[] {
    const migrated: Chat[] = [];
    
    Object.entries(oldChats).forEach(([agent, chats]) => {
      chats.forEach(oldChat => {
        const chat: Chat = {
          id: oldChat.id || generateId(),
          title: oldChat.title,
          agent: agent as AgentType,
          messages: [],
          context: { files: [], strategies: [], brokers: [], backtests: [] },
          createdAt: new Date(),
          lastMessageAt: new Date(),
          isPinned: false,
          tags: []
        };
        migrated.push(chat);
      });
    });
    
    return migrated;
  },

  // Duplicate a chat
  duplicateChat(chatId: string): Chat | null {
    const original = this.getChat(chatId);
    if (!original) return null;
    
    const duplicate: Chat = {
      ...original,
      id: generateId(),
      title: `${original.title} (Copy)`,
      createdAt: new Date(),
      lastMessageAt: new Date(),
      isPinned: false,
      messages: original.messages.map(msg => ({
        ...msg,
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      }))
    };
    
    this.saveChat(duplicate);
    return duplicate;
  },

  // Archive old chats (remove messages but keep metadata)
  archiveOldChats(olderThanDays: number = 30): number {
    const chats = this.loadAllChats();
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - olderThanDays);
    
    let archivedCount = 0;
    
    chats.forEach(chat => {
      if (!chat.isPinned && new Date(chat.lastMessageAt) < cutoff) {
        chat.messages = chat.messages.slice(-10); // Keep last 10 messages
        chat.tags = [...chat.tags, 'archived'];
        archivedCount++;
      }
    });
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
    return archivedCount;
  }
};

export default chatManager;
