<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { fade, slide } from 'svelte/transition';
  import { X, Save, RotateCcw, Download, Upload, AlertCircle } from 'lucide-svelte';
  
  // Import settings components
  import GeneralSettings from './GeneralSettings.svelte';
  import APIKeysSettings from './APIKeysSettings.svelte';
  import MCPSettings from './MCPSettings.svelte';
  import SkillsSettings from './SkillsSettings.svelte';
  import RulesSettings from './RulesSettings.svelte';
  import MemoriesSettings from './MemoriesSettings.svelte';
  import WorkflowsSettings from './WorkflowsSettings.svelte';
  import PermissionsSettings from './PermissionsSettings.svelte';
  
  // Import store
  import { settingsStore } from '../../../stores/settingsStore';
  
  const dispatch = createEventDispatcher();
  
  // State
  let activeTab = 'general';
  let isSaving = false;
  let showUnsavedWarning = false;
  
  // Tab configuration
  const tabs = [
    { id: 'general', label: 'General', icon: '⚙️' },
    { id: 'apiKeys', label: 'API Keys', icon: '🔑' },
    { id: 'mcp', label: 'MCP', icon: '🔌' },
    { id: 'skills', label: 'Skills', icon: '✨' },
    { id: 'rules', label: 'Rules', icon: '📋' },
    { id: 'memories', label: 'Memory', icon: '💾' },
    { id: 'workflows', label: 'Workflows', icon: '🔄' },
    { id: 'permissions', label: 'Permissions', icon: '🔒' }
  ];
  
  // Reactive state
  $: isDirty = $settingsStore.isDirty;
  $: error = $settingsStore.error;
  
  // Initialize
  onMount(() => {
    settingsStore.initialize();
  });
  
  // Handle tab change
  function handleTabChange(tabId: string) {
    if (isDirty) {
      showUnsavedWarning = true;
      return;
    }
    activeTab = tabId;
  }
  
  // Handle save
  async function handleSave() {
    isSaving = true;
    try {
      await settingsStore.save();
      showUnsavedWarning = false;
    } catch (error) {
      console.error('Failed to save settings:', error);
    } finally {
      isSaving = false;
    }
  }
  
  // Handle reset
  function handleReset() {
    if (confirm('Are you sure you want to reset all settings to defaults?')) {
      settingsStore.resetToDefaults();
    }
  }
  
  // Handle close
  function handleClose() {
    if (isDirty) {
      showUnsavedWarning = true;
      return;
    }
    dispatch('close');
  }
  
  // Handle export
  function handleExport() {
    const data = settingsStore.exportSettings();
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `quantmind-settings-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  // Handle import
  function handleImport(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const data = e.target?.result as string;
      if (settingsStore.importSettings(data)) {
        alert('Settings imported successfully!');
      } else {
        alert('Failed to import settings. Please check the file format.');
      }
    };
    reader.readAsText(file);
  }
  
  // Keyboard navigation
  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') {
      handleClose();
    }
    
    // Tab navigation with arrow keys
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
      const currentIndex = tabs.findIndex(t => t.id === activeTab);
      let newIndex: number;
      
      if (e.key === 'ArrowRight') {
        newIndex = (currentIndex + 1) % tabs.length;
      } else {
        newIndex = (currentIndex - 1 + tabs.length) % tabs.length;
      }
      
      handleTabChange(tabs[newIndex].id);
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions a11y-no-noninteractive-element-interactions -->
<div class="settings-panel" on:click|stopPropagation role="dialog" aria-label="Settings">
  <!-- Header -->
  <header class="panel-header">
    <h2>Settings</h2>
    <div class="header-actions">
      {#if isDirty}
        <span class="dirty-indicator">Unsaved changes</span>
      {/if}
      <button class="icon-btn" on:click={handleClose} aria-label="Close settings">
        <X size={18} />
      </button>
    </div>
  </header>
  
  <!-- Tab Navigation -->
  <!-- svelte-ignore a11y-no-interactive-element-to-noninteractive-role -->
  <div class="tab-nav" role="tablist">
    {#each tabs as tab}
      <button
        class="tab-btn"
        class:active={activeTab === tab.id}
        on:click={() => handleTabChange(tab.id)}
        role="tab"
        aria-selected={activeTab === tab.id}
        aria-controls="tab-panel-{tab.id}"
      >
        <span class="tab-icon">{tab.icon}</span>
        <span class="tab-label">{tab.label}</span>
      </button>
    {/each}
  </div>
  
  <!-- Tab Content -->
  <div class="tab-content" role="tabpanel" id="tab-panel-{activeTab}">
    {#if $settingsStore.isLoading}
      <div class="loading-state">
        <div class="spinner"></div>
        <span>Loading settings...</span>
      </div>
    {:else}
      {#if activeTab === 'general'}
        <GeneralSettings />
      {:else if activeTab === 'apiKeys'}
        <APIKeysSettings />
      {:else if activeTab === 'mcp'}
        <MCPSettings />
      {:else if activeTab === 'skills'}
        <SkillsSettings />
      {:else if activeTab === 'rules'}
        <RulesSettings />
      {:else if activeTab === 'memories'}
        <MemoriesSettings />
      {:else if activeTab === 'workflows'}
        <WorkflowsSettings />
      {:else if activeTab === 'permissions'}
        <PermissionsSettings />
      {/if}
    {/if}
  </div>
  
  <!-- Error Display -->
  {#if error}
    <div class="error-banner" transition:slide>
      <AlertCircle size={16} />
      <span>{error}</span>
      <button on:click={() => settingsStore.clearError()}>Dismiss</button>
    </div>
  {/if}
  
  <!-- Footer Actions -->
  <footer class="panel-footer">
    <div class="footer-left">
      <button class="btn secondary" on:click={handleReset} title="Reset to defaults">
        <RotateCcw size={14} />
        Reset
      </button>
      <div class="import-export">
        <button class="btn secondary" on:click={handleExport} title="Export settings">
          <Download size={14} />
        </button>
        <label class="btn secondary" title="Import settings">
          <Upload size={14} />
          <input type="file" accept=".json" on:change={handleImport} hidden />
        </label>
      </div>
    </div>
    <div class="footer-right">
      <button class="btn primary" on:click={handleSave} disabled={!isDirty || isSaving}>
        {#if isSaving}
          <span class="spinner small"></span>
        {:else}
          <Save size={14} />
        {/if}
        Save Changes
      </button>
    </div>
  </footer>
  
  <!-- Unsaved Warning Modal -->
  {#if showUnsavedWarning}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div class="modal-overlay" on:click={() => showUnsavedWarning = false} role="button" tabindex="-1" aria-label="Close dialog">
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions a11y-no-noninteractive-element-interactions -->
    <div class="warning-modal" on:click|stopPropagation transition:fade role="dialog" aria-modal="true" aria-labelledby="warning-title">
        <h3 id="warning-title">Unsaved Changes</h3>
        <p>You have unsaved changes. Do you want to save them before leaving?</p>
        <div class="modal-actions">
          <button class="btn secondary" on:click={() => { showUnsavedWarning = false; dispatch('close'); }}>
            Discard
          </button>
          <button class="btn secondary" on:click={() => showUnsavedWarning = false}>
            Cancel
          </button>
          <button class="btn primary" on:click={async () => { await handleSave(); showUnsavedWarning = false; dispatch('close'); }}>
            Save
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .settings-panel {
    display: flex;
    flex-direction: column;
    width: 640px;
    max-width: 90vw;
    max-height: 85vh;
    background: var(--bg-secondary);
    border-radius: 16px;
    border: 1px solid var(--border-subtle);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
    overflow: hidden;
  }
  
  .panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-tertiary);
  }
  
  .panel-header h2 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
  }
  
  .header-actions {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  
  .dirty-indicator {
    font-size: 11px;
    color: var(--accent-warning);
    padding: 4px 8px;
    background: rgba(245, 158, 11, 0.1);
    border-radius: 4px;
  }
  
  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .icon-btn:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  
  /* Tab Navigation */
  .tab-nav {
    display: flex;
    gap: 2px;
    padding: 8px 12px;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border-subtle);
    overflow-x: auto;
  }
  
  .tab-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-muted);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
  }
  
  .tab-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .tab-btn.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }
  
  .tab-icon {
    font-size: 14px;
  }
  
  .tab-label {
    font-weight: 500;
  }
  
  /* Tab Content */
  .tab-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }
  
  .loading-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: var(--text-muted);
    gap: 12px;
  }
  
  .spinner {
    width: 24px;
    height: 24px;
    border: 2px solid var(--border-subtle);
    border-top-color: var(--accent-primary);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }
  
  .spinner.small {
    width: 14px;
    height: 14px;
  }
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  
  /* Error Banner */
  .error-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: rgba(239, 68, 68, 0.1);
    border-top: 1px solid var(--accent-danger);
    color: var(--accent-danger);
    font-size: 12px;
  }
  
  .error-banner button {
    margin-left: auto;
    background: transparent;
    border: none;
    color: var(--accent-danger);
    cursor: pointer;
    font-size: 11px;
    text-decoration: underline;
  }
  
  /* Footer */
  .panel-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    border-top: 1px solid var(--border-subtle);
    background: var(--bg-tertiary);
  }
  
  .footer-left, .footer-right {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .import-export {
    display: flex;
    gap: 4px;
    margin-left: 8px;
  }
  
  /* Buttons */
  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    border: none;
  }
  
  .btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }
  
  .btn.primary:hover:not(:disabled) {
    background: var(--accent-secondary);
  }
  
  .btn.primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .btn.secondary {
    background: var(--bg-primary);
    color: var(--text-secondary);
    border: 1px solid var(--border-subtle);
  }
  
  .btn.secondary:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  /* Modal */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  
  .warning-modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 24px;
    max-width: 360px;
    border: 1px solid var(--border-subtle);
  }
  
  .warning-modal h3 {
    margin: 0 0 12px;
    font-size: 16px;
    color: var(--text-primary);
  }
  
  .warning-modal p {
    margin: 0 0 20px;
    font-size: 13px;
    color: var(--text-secondary);
  }
  
  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
  
  /* Responsive */
  @media (max-width: 640px) {
    .settings-panel {
      width: 100%;
      max-width: 100%;
      max-height: 100vh;
      border-radius: 0;
    }
    
    .tab-nav {
      flex-wrap: nowrap;
      overflow-x: auto;
      -webkit-overflow-scrolling: touch;
    }
    
    .tab-label {
      display: none;
    }
  }
</style>
