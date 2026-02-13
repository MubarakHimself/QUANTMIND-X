// Settings Manager Service - Handles settings persistence and encryption
import type { 
  SettingsStoreState, 
  GeneralSettings, 
  APIKeys, 
  MCPServer,
  AgentType,
  Rule,
  MemoryConfig,
  Workflow,
  AgentPermissions,
  Skill
} from '$lib/stores/settingsTypes';

const STORAGE_KEY = 'quantmind_settings';
const API_KEYS_KEY = 'quantmind_api_keys';
const SETTINGS_VERSION = 1;

// Simple encryption for API keys (in production, use proper encryption)
function encrypt(text: string): string {
  // This is a simple obfuscation - in production, use proper encryption
  return btoa(text);
}

function decrypt(encrypted: string): string {
  try {
    return atob(encrypted);
  } catch {
    return '';
  }
}

// Settings Manager Service
export const settingsManager = {
  // Load all settings from storage
  async loadSettings(): Promise<Partial<SettingsStoreState>> {
    try {
      // Load general settings
      const stored = localStorage.getItem(STORAGE_KEY);
      const settings = stored ? JSON.parse(stored) : {};
      
      // Load API keys separately (encrypted)
      const encryptedKeys = localStorage.getItem(API_KEYS_KEY);
      let apiKeys = {
        google: '',
        anthropic: '',
        openai: '',
        qwen: ''
      };
      
      if (encryptedKeys) {
        const decrypted = JSON.parse(decrypt(encryptedKeys));
        apiKeys = { ...apiKeys, ...decrypted };
      }
      
      // Migrate settings if needed
      const migratedSettings = this.migrateSettings(settings);
      
      return {
        ...migratedSettings,
        apiKeys
      };
    } catch (error) {
      console.error('Failed to load settings:', error);
      return {};
    }
  },

  // Save all settings to storage
  async saveSettings(settings: SettingsStoreState): Promise<void> {
    try {
      // Save general settings (without API keys)
      const settingsToSave = {
        version: SETTINGS_VERSION,
        general: settings.general,
        mcpServers: settings.mcpServers,
        skills: settings.skills,
        rules: settings.rules,
        memories: settings.memories,
        workflows: settings.workflows,
        permissions: settings.permissions
      };
      
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settingsToSave));
      
      // Save API keys separately (encrypted)
      const encryptedKeys = encrypt(JSON.stringify(settings.apiKeys));
      localStorage.setItem(API_KEYS_KEY, encryptedKeys);
    } catch (error) {
      console.error('Failed to save settings:', error);
      throw error;
    }
  },

  // Migrate settings from older versions
  migrateSettings(settings: any): Partial<SettingsStoreState> {
    const version = settings.version || 0;
    
    if (version < 1) {
      // Initial migration - add any missing fields
      return {
        ...settings,
        version: SETTINGS_VERSION
      };
    }
    
    return settings;
  },

  // Load general settings
  loadGeneralSettings(): GeneralSettings | null {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const settings = JSON.parse(stored);
        return settings.general || null;
      }
      return null;
    } catch {
      return null;
    }
  },

  // Save general settings
  saveGeneralSettings(general: GeneralSettings): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      const settings = stored ? JSON.parse(stored) : {};
      settings.general = general;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save general settings:', error);
    }
  },

  // Load API keys
  loadAPIKeys(): APIKeys {
    try {
      const encryptedKeys = localStorage.getItem(API_KEYS_KEY);
      if (encryptedKeys) {
        return JSON.parse(decrypt(encryptedKeys));
      }
    } catch {
      // Ignore errors
    }
    return {
      google: '',
      anthropic: '',
      openai: '',
      qwen: ''
    };
  },

  // Save API keys
  saveAPIKeys(apiKeys: APIKeys): void {
    try {
      const encryptedKeys = encrypt(JSON.stringify(apiKeys));
      localStorage.setItem(API_KEYS_KEY, encryptedKeys);
    } catch (error) {
      console.error('Failed to save API keys:', error);
    }
  },

  // Load MCP servers
  loadMCPServers(): MCPServer[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const settings = JSON.parse(stored);
        return settings.mcpServers || [];
      }
    } catch {
      // Ignore errors
    }
    return [];
  },

  // Save MCP servers
  saveMCPServers(servers: MCPServer[]): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      const settings = stored ? JSON.parse(stored) : {};
      settings.mcpServers = servers;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save MCP servers:', error);
    }
  },

  // Load rules
  loadRules(): Rule[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const settings = JSON.parse(stored);
        return settings.rules || [];
      }
    } catch {
      // Ignore errors
    }
    return [];
  },

  // Save rules
  saveRules(rules: Rule[]): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      const settings = stored ? JSON.parse(stored) : {};
      settings.rules = rules;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save rules:', error);
    }
  },

  // Load workflows
  loadWorkflows(): Workflow[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const settings = JSON.parse(stored);
        return settings.workflows || [];
      }
    } catch {
      // Ignore errors
    }
    return [];
  },

  // Save workflows
  saveWorkflows(workflows: Workflow[]): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      const settings = stored ? JSON.parse(stored) : {};
      settings.workflows = workflows;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save workflows:', error);
    }
  },

  // Export settings to JSON
  exportSettings(settings: SettingsStoreState): string {
    const exportData = {
      version: SETTINGS_VERSION,
      exportedAt: new Date().toISOString(),
      general: settings.general,
      mcpServers: settings.mcpServers,
      skills: settings.skills,
      rules: settings.rules,
      memories: settings.memories,
      workflows: settings.workflows,
      permissions: settings.permissions
      // Note: API keys excluded for security
    };
    
    return JSON.stringify(exportData, null, 2);
  },

  // Import settings from JSON
  importSettings(jsonData: string): Partial<SettingsStoreState> | null {
    try {
      const imported = JSON.parse(jsonData);
      
      // Validate structure
      if (!imported.version || imported.version > SETTINGS_VERSION) {
        throw new Error('Incompatible settings version');
      }
      
      // Migrate if needed
      return this.migrateSettings(imported);
    } catch (error) {
      console.error('Failed to import settings:', error);
      return null;
    }
  },

  // Reset all settings to defaults
  resetToDefaults(): void {
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(API_KEYS_KEY);
  },

  // Validate settings before save
  validateSettings(settings: Partial<SettingsStoreState>): string[] {
    const errors: string[] = [];
    
    // Validate general settings
    if (settings.general) {
      if (settings.general.autoSaveInterval < 10 || settings.general.autoSaveInterval > 300) {
        errors.push('Auto-save interval must be between 10 and 300 seconds');
      }
    }
    
    // Validate MCP servers
    if (settings.mcpServers) {
      settings.mcpServers.forEach(server => {
        if (!server.name || !server.url) {
          errors.push(`MCP server "${server.name || 'unnamed'}" is missing required fields`);
        }
      });
    }
    
    // Validate rules
    if (settings.rules) {
      settings.rules.forEach(rule => {
        if (!rule.name || !rule.content) {
          errors.push(`Rule "${rule.name || 'unnamed'}" is missing required fields`);
        }
      });
    }
    
    return errors;
  },

  // Get settings statistics
  getSettingsStats(): {
    hasAPIKeys: boolean;
    mcpServerCount: number;
    ruleCount: number;
    workflowCount: number;
  } {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      const settings = stored ? JSON.parse(stored) : {};
      
      const encryptedKeys = localStorage.getItem(API_KEYS_KEY);
      const apiKeys = encryptedKeys ? JSON.parse(decrypt(encryptedKeys)) : {};
      const hasAPIKeys = Object.values(apiKeys).some((key: any) => key && key.length > 0);
      
      return {
        hasAPIKeys,
        mcpServerCount: (settings.mcpServers || []).length,
        ruleCount: (settings.rules || []).length,
        workflowCount: (settings.workflows || []).length
      };
    } catch {
      return {
        hasAPIKeys: false,
        mcpServerCount: 0,
        ruleCount: 0,
        workflowCount: 0
      };
    }
  },

  // Backup settings
  backupSettings(): string {
    const settings = localStorage.getItem(STORAGE_KEY);
    const apiKeys = localStorage.getItem(API_KEYS_KEY);
    
    return JSON.stringify({
      settings: settings ? JSON.parse(settings) : {},
      apiKeys: apiKeys ? JSON.parse(decrypt(apiKeys)) : {},
      backupDate: new Date().toISOString()
    }, null, 2);
  },

  // Restore from backup
  restoreFromBackup(backupData: string): boolean {
    try {
      const backup = JSON.parse(backupData);
      
      if (backup.settings) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(backup.settings));
      }
      
      if (backup.apiKeys) {
        localStorage.setItem(API_KEYS_KEY, encrypt(JSON.stringify(backup.apiKeys)));
      }
      
      return true;
    } catch (error) {
      console.error('Failed to restore from backup:', error);
      return false;
    }
  }
};

export default settingsManager;
