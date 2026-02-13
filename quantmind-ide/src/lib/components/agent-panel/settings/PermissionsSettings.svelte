<script lang="ts">
  import { Bot, Code, Wand2, Shield, AlertTriangle, Info } from 'lucide-svelte';
  import { settingsStore, permissionPresets } from '../../../stores/settingsStore';
  import type { AgentType, AgentPermissions } from '../../../stores/settingsStore';
  
  // State
  let selectedAgent: AgentType = 'copilot';
  let selectedPreset: 'restricted' | 'standard' | 'fullAccess' = 'standard';
  
  // Agent configuration
  const agents: Array<{ id: AgentType; name: string; icon: any }> = [
    { id: 'copilot', name: 'Copilot', icon: Bot },
    { id: 'quantcode', name: 'QuantCode', icon: Code },
    { id: 'analyst', name: 'Analyst', icon: Wand2 }
  ];
  
  // Permission types
  const permissionTypes = [
    { key: 'fileSystem', label: 'File System', description: 'Access to read/write files' },
    { key: 'broker', label: 'Broker', description: 'Access to trading accounts' },
    { key: 'database', label: 'Database', description: 'Access to data storage' }
  ];
  
  const accessLevels = [
    { value: 'none', label: 'None' },
    { value: 'read', label: 'Read' },
    { value: 'write', label: 'Write' },
    { value: 'full', label: 'Full' }
  ];
  
  const booleanPermissions = [
    { key: 'external', label: 'External APIs', description: 'Allow calls to external services' },
    { key: 'memory', label: 'Memory Access', description: 'Access to persistent memory' }
  ];
  
  // Reactive state
  $: permissions = $settingsStore.permissions[selectedAgent];
  
  // Apply preset
  function applyPreset(preset: 'restricted' | 'standard' | 'fullAccess') {
    selectedPreset = preset;
    settingsStore.updateAgentPermissions(selectedAgent, permissionPresets[preset]);
  }
  
  // Update permission
  function updatePermission(key: keyof AgentPermissions, value: any) {
    settingsStore.updateAgentPermissions(selectedAgent, { [key]: value });
    // Reset preset selection when manually changed
    selectedPreset = 'standard';
  }
  
  // Get access level color
  function getAccessColor(level: string): string {
    switch (level) {
      case 'full': return 'var(--accent-success)';
      case 'write': return 'var(--accent-warning)';
      case 'read': return 'var(--accent-primary)';
      default: return 'var(--text-muted)';
    }
  }
  
  // Check if current permissions match preset
  function matchesPreset(preset: 'restricted' | 'standard' | 'fullAccess'): boolean {
    const presetPerms = permissionPresets[preset];
    return JSON.stringify(permissions) === JSON.stringify(presetPerms);
  }

  // Helper to get permission value (avoid TypeScript in template)
  function getPermissionValue(key: string): string {
    return permissions[key as keyof AgentPermissions] as string;
  }

  // Helper to get boolean permission value
  function getBooleanPermission(key: string): boolean {
    return permissions[key as keyof AgentPermissions] as boolean;
  }

  // Helper to check if permission level is active
  function isPermissionActive(key: string, value: string): boolean {
    return permissions[key as keyof AgentPermissions] === value;
  }
</script>

<div class="permissions-settings">
  <h3>Agent Permissions</h3>
  <p class="description">Configure what each agent is allowed to access and modify.</p>
  
  <!-- Agent Selector -->
  <div class="agent-selector">
    {#each agents as agent}
      <button 
        class="agent-chip"
        class:active={selectedAgent === agent.id}
        on:click={() => selectedAgent = agent.id}
      >
        <svelte:component this={agent.icon} size={14} />
        {agent.name}
      </button>
    {/each}
  </div>
  
  <!-- Preset Selector -->
  <section class="preset-section">
    <h4>Permission Preset</h4>
    <div class="preset-grid">
      <button 
        class="preset-card"
        class:active={matchesPreset('restricted')}
        on:click={() => applyPreset('restricted')}
      >
        <Shield size={20} />
        <span class="preset-name">Restricted</span>
        <span class="preset-desc">Minimal access, safe for testing</span>
      </button>
      
      <button 
        class="preset-card"
        class:active={matchesPreset('standard')}
        on:click={() => applyPreset('standard')}
      >
        <Info size={20} />
        <span class="preset-name">Standard</span>
        <span class="preset-desc">Balanced access for normal use</span>
      </button>
      
      <button 
        class="preset-card"
        class:active={matchesPreset('fullAccess')}
        on:click={() => applyPreset('fullAccess')}
      >
        <AlertTriangle size={20} />
        <span class="preset-name">Full Access</span>
        <span class="preset-desc">Unrestricted, use with caution</span>
      </button>
    </div>
  </section>
  
  <!-- Permission Matrix -->
  <section class="permissions-section">
    <h4>Access Levels</h4>
    <div class="permission-matrix">
      {#each permissionTypes as perm}
        <div class="permission-row">
          <div class="permission-info">
            <span class="permission-label">{perm.label}</span>
            <span class="permission-desc">{perm.description}</span>
          </div>
          <div class="permission-options">
            {#each accessLevels as level}
              <button 
                class="level-btn"
                class:active={isPermissionActive(perm.key, level.value)}
                style="--level-color: {getAccessColor(level.value)}"
                on:click={() => updatePermission(perm.key, level.value)}
              >
                {level.label}
              </button>
            {/each}
          </div>
        </div>
      {/each}
    </div>
    
    <h4>Toggle Permissions</h4>
    <div class="permission-toggles">
      {#each booleanPermissions as perm}
        <div class="permission-row">
          <div class="permission-info">
            <span class="permission-label">{perm.label}</span>
            <span class="permission-desc">{perm.description}</span>
          </div>
          <label class="toggle">
            <input 
              type="checkbox" 
              checked={getBooleanPermission(perm.key)}
              on:change={() => updatePermission(perm.key, !getBooleanPermission(perm.key))}
            />
            <span class="toggle-slider"></span>
          </label>
        </div>
      {/each}
    </div>
  </section>
  
  <!-- Permission Summary -->
  <section class="summary-section">
    <h4>Current Configuration</h4>
    <div class="summary-grid">
      <div class="summary-item">
        <span class="summary-label">File System</span>
        <span class="summary-value" style="color: {getAccessColor(getPermissionValue('fileSystem'))}">
          {permissions.fileSystem}
        </span>
      </div>
      <div class="summary-item">
        <span class="summary-label">Broker</span>
        <span class="summary-value" style="color: {getAccessColor(getPermissionValue('broker'))}">
          {permissions.broker}
        </span>
      </div>
      <div class="summary-item">
        <span class="summary-label">Database</span>
        <span class="summary-value" style="color: {getAccessColor(getPermissionValue('database'))}">
          {permissions.database}
        </span>
      </div>
      <div class="summary-item">
        <span class="summary-label">External APIs</span>
        <span class="summary-value" style="color: {permissions.external ? 'var(--accent-success)' : 'var(--text-muted)'}">
          {permissions.external ? 'Enabled' : 'Disabled'}
        </span>
      </div>
      <div class="summary-item">
        <span class="summary-label">Memory</span>
        <span class="summary-value" style="color: {permissions.memory ? 'var(--accent-success)' : 'var(--text-muted)'}">
          {permissions.memory ? 'Enabled' : 'Disabled'}
        </span>
      </div>
    </div>
  </section>
  
  <!-- Warning -->
  {#if permissions.fileSystem === 'full' || permissions.broker === 'full'}
    <div class="warning-banner">
      <AlertTriangle size={14} />
      <span>Full access permissions granted. The agent can make significant changes to your system.</span>
    </div>
  {/if}
  
  <!-- Info Section -->
  <div class="info-section">
    <h4>About Permissions</h4>
    <p>Permissions control what actions each agent can perform. Restricting permissions can help prevent unintended changes. Permissions are enforced at the action level.</p>
  </div>
</div>

<style>
  .permissions-settings {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .description {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
  }
  
  /* Agent Selector */
  .agent-selector {
    display: flex;
    gap: 8px;
    padding: 4px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }
  
  .agent-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .agent-chip:hover {
    color: var(--text-primary);
  }
  
  .agent-chip.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }
  
  /* Preset Section */
  .preset-section h4,
  .permissions-section h4,
  .summary-section h4 {
    margin: 0 0 12px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
  }
  
  .preset-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  
  .preset-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 2px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
  }
  
  .preset-card:hover {
    border-color: var(--accent-primary);
  }
  
  .preset-card.active {
    border-color: var(--accent-primary);
    background: rgba(107, 200, 230, 0.1);
  }
  
  .preset-card svg {
    color: var(--text-muted);
  }
  
  .preset-card.active svg {
    color: var(--accent-primary);
  }
  
  .preset-name {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .preset-desc {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  /* Permission Matrix */
  .permission-matrix,
  .permission-toggles {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
  }
  
  .permission-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }
  
  .permission-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
  }
  
  .permission-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .permission-desc {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .permission-options {
    display: flex;
    gap: 4px;
  }
  
  .level-btn {
    padding: 6px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .level-btn:hover {
    border-color: var(--level-color);
    color: var(--text-primary);
  }
  
  .level-btn.active {
    background: var(--level-color);
    border-color: var(--level-color);
    color: var(--bg-primary);
  }
  
  /* Toggle Switch */
  .toggle {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
    flex-shrink: 0;
  }
  
  .toggle input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  
  .toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 22px;
    transition: all 0.2s;
  }
  
  .toggle-slider:before {
    position: absolute;
    content: "";
    height: 16px;
    width: 16px;
    left: 2px;
    bottom: 2px;
    background-color: var(--text-muted);
    border-radius: 50%;
    transition: all 0.2s;
  }
  
  .toggle input:checked + .toggle-slider {
    background-color: var(--accent-primary);
    border-color: var(--accent-primary);
  }
  
  .toggle input:checked + .toggle-slider:before {
    transform: translateX(18px);
    background-color: var(--bg-primary);
  }
  
  /* Summary Section */
  .summary-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 8px;
  }
  
  .summary-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    text-align: center;
  }
  
  .summary-label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
  }
  
  .summary-value {
    font-size: 12px;
    font-weight: 600;
    text-transform: capitalize;
  }
  
  /* Warning Banner */
  .warning-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid var(--accent-danger);
    border-radius: 8px;
    font-size: 12px;
    color: var(--accent-danger);
  }
  
  /* Info Section */
  .info-section {
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
  }
  
  .info-section h4 {
    margin: 0 0 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .info-section p {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
  }
  
  /* Responsive */
  @media (max-width: 600px) {
    .preset-grid {
      grid-template-columns: 1fr;
    }
    
    .permission-row {
      flex-direction: column;
      align-items: flex-start;
    }
    
    .permission-options {
      width: 100%;
      justify-content: flex-end;
      margin-top: 8px;
    }
    
    .summary-grid {
      grid-template-columns: repeat(2, 1fr);
    }
  }
</style>
