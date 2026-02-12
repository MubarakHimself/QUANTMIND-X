import { v4 as uuidv4 } from 'uuid';

interface MemoryItem {
  id: string;
  namespace: string[];
  key: string;
  value: any;
  timestamp: number;
  embedding?: number[];
}

interface MemoryStore {
  get(namespace: string[], key: string): Promise<any>;
  set(namespace: string[], key: string, value: any): Promise<void>;
  delete(namespace: string[], key: string): Promise<void>;
  search(namespace: string[], query: string, limit?: number): Promise<MemoryItem[]>;
  clear(): Promise<void>;
}

export class HybridMemoryManager implements MemoryStore {
  private shortTerm: Map<string, MemoryItem> = new Map();
  private longTerm: Map<string, MemoryItem> = new Map();
  private maxShortTermSize = 100;
  private shortTermTTL = 30 * 60 * 1000; // 30 minutes

  private getNamespaceKey(namespace: string[], key: string): string {
    return [...namespace, key].join(':');
  }

  private createMemoryItem(namespace: string[], key: string, value: any): MemoryItem {
    return {
      id: uuidv4(),
      namespace,
      key,
      value,
      timestamp: Date.now()
    };
  }

  async get(namespace: string[], key: string): Promise<any> {
    const nsKey = this.getNamespaceKey(namespace, key);

    // Check short-term memory first
    const shortTermItem = this.shortTerm.get(nsKey);
    if (shortTermItem) {
      const age = Date.now() - shortTermItem.timestamp;
      if (age < this.shortTermTTL) {
        return shortTermItem.value;
      } else {
        this.shortTerm.delete(nsKey);
      }
    }

    // Check long-term memory
    const longTermItem = this.longTerm.get(nsKey);
    return longTermItem?.value;
  }

  async set(namespace: string[], key: string, value: any): Promise<void> {
    const nsKey = this.getNamespaceKey(namespace, key);
    const item = this.createMemoryItem(namespace, key, value);

    // Add to short-term memory
    this.shortTerm.set(nsKey, item);

    // Evict old items if short-term memory is full
    if (this.shortTerm.size > this.maxShortTermSize) {
      const oldestKey = Array.from(this.shortTerm.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp)[0][0];
      this.shortTerm.delete(oldestKey);
    }

    // Also store in long-term memory
    this.longTerm.set(nsKey, item);
  }

  async delete(namespace: string[], key: string): Promise<void> {
    const nsKey = this.getNamespaceKey(namespace, key);
    this.shortTerm.delete(nsKey);
    this.longTerm.delete(nsKey);
  }

  async search(namespace: string[], query: string, limit: number = 10): Promise<MemoryItem[]> {
    const results: MemoryItem[] = [];
    const nsPrefix = namespace.join(':');

    const queryLower = query.toLowerCase();

    // Search long-term memory
    for (const [key, item] of this.longTerm.entries()) {
      if (key.startsWith(nsPrefix)) {
        const strValue = JSON.stringify(item.value);
        if (strValue.toLowerCase().includes(queryLower)) {
          results.push(item);
          if (results.length >= limit) break;
        }
      }
    }

    return results;
  }

  async clear(): Promise<void> {
    this.shortTerm.clear();
    this.longTerm.clear();
  }

  // Conversation management
  async saveConversation(agentId: string, conversationId: string, messages: any[]): Promise<void> {
    await this.set(['quantmindx', agentId, 'conversations'], conversationId, {
      messages,
      timestamp: Date.now()
    });
  }

  async getConversationHistory(agentId: string): Promise<any[]> {
    const conversations = await this.search(['quantmindx', agentId, 'conversations'], '', 100);
    return conversations.flatMap(c => c.value.messages || []);
  }

  // Export/Import for persistence
  exportData(): string {
    const data = {
      shortTerm: Array.from(this.shortTerm.entries()),
      longTerm: Array.from(this.longTerm.entries())
    };
    return JSON.stringify(data);
  }

  importData(jsonData: string): void {
    try {
      const data = JSON.parse(jsonData);
      this.shortTerm = new Map(data.shortTerm);
      this.longTerm = new Map(data.longTerm);
    } catch (e) {
      console.error('Failed to import memory data:', e);
    }
  }
}

export const memoryManager = new HybridMemoryManager();
