// File History Manager Service - Tracks file operations by agents
import type { AgentType } from '../stores/chatStore';

// File version interface
export interface FileVersion {
  id: string;
  timestamp: Date;
  agent: AgentType;
  action: 'created' | 'modified' | 'deleted';
  content: string;
  diff?: string;
  summary?: string;
}

// File history interface
export interface FileHistory {
  fileId: string;
  fileName: string;
  filePath: string;
  versions: FileVersion[];
  createdAt: Date;
  updatedAt: Date;
}

// History statistics
export interface HistoryStats {
  totalFiles: number;
  totalVersions: number;
  byAgent: Record<AgentType, number>;
  byAction: Record<string, number>;
  oldestVersion: Date | null;
  newestVersion: Date | null;
}

const STORAGE_KEY = 'quantmind_file_history';
const MAX_VERSIONS_PER_FILE = 50;
const MAX_FILES = 100;

// Generate unique ID
function generateId(): string {
  return `ver_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Simple diff function (for more complex diffs, use a library like 'diff')
function computeDiff(oldContent: string, newContent: string): string {
  const oldLines = oldContent.split('\n');
  const newLines = newContent.split('\n');
  
  const changes: string[] = [];
  const maxLines = Math.max(oldLines.length, newLines.length);
  
  for (let i = 0; i < maxLines; i++) {
    const oldLine = oldLines[i];
    const newLine = newLines[i];
    
    if (oldLine !== newLine) {
      if (oldLine === undefined) {
        changes.push(`+ ${i + 1}: ${newLine}`);
      } else if (newLine === undefined) {
        changes.push(`- ${i + 1}: ${oldLine}`);
      } else {
        changes.push(`~ ${i + 1}: ${oldLine} -> ${newLine}`);
      }
    }
  }
  
  return changes.join('\n');
}

// File History Manager Service
export const fileHistoryManager = {
  // Load all file histories from storage
  loadAllHistories(): FileHistory[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (!stored) return [];
      
      const histories = JSON.parse(stored) as FileHistory[];
      
      // Convert date strings back to Date objects
      return histories.map(history => ({
        ...history,
        createdAt: new Date(history.createdAt),
        updatedAt: new Date(history.updatedAt),
        versions: history.versions.map(v => ({
          ...v,
          timestamp: new Date(v.timestamp)
        }))
      }));
    } catch (error) {
      console.error('Failed to load file histories:', error);
      return [];
    }
  },
  
  // Save all histories to storage
  saveHistories(histories: FileHistory[]): void {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(histories));
    } catch (error) {
      console.error('Failed to save file histories:', error);
    }
  },
  
  // Get history for a specific file
  getFileHistory(fileId: string): FileHistory | null {
    const histories = this.loadAllHistories();
    return histories.find(h => h.fileId === fileId) || null;
  },
  
  // Get history by file path
  getFileHistoryByPath(filePath: string): FileHistory | null {
    const histories = this.loadAllHistories();
    return histories.find(h => h.filePath === filePath) || null;
  },
  
  // Record a file operation
  recordOperation(
    fileId: string,
    fileName: string,
    filePath: string,
    agent: AgentType,
    action: FileVersion['action'],
    content: string,
    previousContent?: string
  ): FileVersion {
    const histories = this.loadAllHistories();
    let history = histories.find(h => h.fileId === fileId);
    
    const version: FileVersion = {
      id: generateId(),
      timestamp: new Date(),
      agent,
      action,
      content,
      diff: previousContent ? computeDiff(previousContent, content) : undefined,
      summary: this.generateSummary(action, content)
    };
    
    if (!history) {
      // Create new history entry
      history = {
        fileId,
        fileName,
        filePath,
        versions: [version],
        createdAt: new Date(),
        updatedAt: new Date()
      };
      histories.unshift(history);
    } else {
      // Add version to existing history
      history.versions.unshift(version);
      history.updatedAt = new Date();
      
      // Enforce version limit
      if (history.versions.length > MAX_VERSIONS_PER_FILE) {
        history.versions = history.versions.slice(0, MAX_VERSIONS_PER_FILE);
      }
    }
    
    // Enforce file limit
    const trimmedHistories = histories.slice(0, MAX_FILES);
    
    this.saveHistories(trimmedHistories);
    return version;
  },
  
  // Generate a summary of changes
  generateSummary(action: FileVersion['action'], content: string): string {
    const lines = content.split('\n').length;
    const chars = content.length;
    
    switch (action) {
      case 'created':
        return `Created file with ${lines} lines, ${chars} characters`;
      case 'modified':
        return `Modified file, now ${lines} lines, ${chars} characters`;
      case 'deleted':
        return 'Deleted file';
      default:
        return 'Unknown action';
    }
  },
  
  // Get a specific version
  getVersion(fileId: string, versionId: string): FileVersion | null {
    const history = this.getFileHistory(fileId);
    if (!history) return null;
    
    return history.versions.find(v => v.id === versionId) || null;
  },
  
  // Get the latest version
  getLatestVersion(fileId: string): FileVersion | null {
    const history = this.getFileHistory(fileId);
    if (!history || history.versions.length === 0) return null;
    
    return history.versions[0];
  },
  
  // Compare two versions
  compareVersions(fileId: string, versionId1: string, versionId2: string): {
    version1: FileVersion | null;
    version2: FileVersion | null;
    diff: string;
  } {
    const version1 = this.getVersion(fileId, versionId1);
    const version2 = this.getVersion(fileId, versionId2);
    
    const diff = (version1 && version2) 
      ? computeDiff(version1.content, version2.content)
      : '';
    
    return { version1, version2, diff };
  },
  
  // Revert to a specific version
  revertToVersion(fileId: string, versionId: string): FileVersion | null {
    const version = this.getVersion(fileId, versionId);
    if (!version) return null;
    
    // This would typically involve writing to the actual file system
    // For now, we just return the version content
    return version;
  },
  
  // Delete file history
  deleteFileHistory(fileId: string): void {
    const histories = this.loadAllHistories();
    const filtered = histories.filter(h => h.fileId !== fileId);
    this.saveHistories(filtered);
  },
  
  // Clear all histories
  clearAllHistories(): void {
    localStorage.removeItem(STORAGE_KEY);
  },
  
  // Get history statistics
  getStats(): HistoryStats {
    const histories = this.loadAllHistories();
    
    const stats: HistoryStats = {
      totalFiles: histories.length,
      totalVersions: 0,
      byAgent: { copilot: 0, quantcode: 0, analyst: 0 },
      byAction: { created: 0, modified: 0, deleted: 0 },
      oldestVersion: null,
      newestVersion: null
    };
    
    histories.forEach(history => {
      stats.totalVersions += history.versions.length;
      
      history.versions.forEach(version => {
        stats.byAgent[version.agent]++;
        stats.byAction[version.action]++;
        
        const timestamp = new Date(version.timestamp);
        if (!stats.oldestVersion || timestamp < stats.oldestVersion) {
          stats.oldestVersion = timestamp;
        }
        if (!stats.newestVersion || timestamp > stats.newestVersion) {
          stats.newestVersion = timestamp;
        }
      });
    });
    
    return stats;
  },
  
  // Search histories
  searchHistories(query: string): FileHistory[] {
    const histories = this.loadAllHistories();
    const lowerQuery = query.toLowerCase();
    
    return histories.filter(history =>
      history.fileName.toLowerCase().includes(lowerQuery) ||
      history.filePath.toLowerCase().includes(lowerQuery) ||
      history.versions.some(v => v.content.toLowerCase().includes(lowerQuery))
    );
  },
  
  // Get histories by agent
  getHistoriesByAgent(agent: AgentType): FileHistory[] {
    const histories = this.loadAllHistories();
    return histories.filter(h => h.versions.some(v => v.agent === agent));
  },
  
  // Get recent changes
  getRecentChanges(limit: number = 20): Array<{ history: FileHistory; version: FileVersion }> {
    const histories = this.loadAllHistories();
    const allVersions: Array<{ history: FileHistory; version: FileVersion }> = [];
    
    histories.forEach(history => {
      history.versions.forEach(version => {
        allVersions.push({ history, version });
      });
    });
    
    // Sort by timestamp descending
    allVersions.sort((a, b) => 
      new Date(b.version.timestamp).getTime() - new Date(a.version.timestamp).getTime()
    );
    
    return allVersions.slice(0, limit);
  },
  
  // Export history
  exportHistory(fileId: string): string | null {
    const history = this.getFileHistory(fileId);
    if (!history) return null;
    
    return JSON.stringify(history, null, 2);
  },
  
  // Import history
  importHistory(jsonData: string): boolean {
    try {
      const history = JSON.parse(jsonData) as FileHistory;
      
      // Validate structure
      if (!history.fileId || !history.fileName || !Array.isArray(history.versions)) {
        throw new Error('Invalid history structure');
      }
      
      // Convert dates
      history.createdAt = new Date(history.createdAt);
      history.updatedAt = new Date(history.updatedAt);
      history.versions = history.versions.map(v => ({
        ...v,
        timestamp: new Date(v.timestamp)
      }));
      
      const histories = this.loadAllHistories();
      
      // Check if history already exists
      const existingIndex = histories.findIndex(h => h.fileId === history.fileId);
      if (existingIndex >= 0) {
        // Merge versions
        const existing = histories[existingIndex];
        history.versions.forEach(version => {
          if (!existing.versions.some(v => v.id === version.id)) {
            existing.versions.push(version);
          }
        });
        existing.versions.sort((a, b) => 
          new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
      } else {
        histories.push(history);
      }
      
      this.saveHistories(histories);
      return true;
    } catch (error) {
      console.error('Failed to import history:', error);
      return false;
    }
  },
  
  // Prune old versions
  pruneOldVersions(olderThanDays: number = 30): number {
    const histories = this.loadAllHistories();
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - olderThanDays);
    
    let prunedCount = 0;
    
    histories.forEach(history => {
      const originalLength = history.versions.length;
      history.versions = history.versions.filter(v => 
        new Date(v.timestamp) >= cutoff || history.versions.indexOf(v) < 5 // Keep at least 5 versions
      );
      prunedCount += originalLength - history.versions.length;
    });
    
    this.saveHistories(histories);
    return prunedCount;
  }
};

export default fileHistoryManager;
