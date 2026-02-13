<script lang="ts">
  import { fade, slide } from 'svelte/transition';
  import { Plus, Trash2, Edit3, Save, X, GripVertical, FileText } from 'lucide-svelte';
  import { settingsStore } from '../../../stores/settingsStore';
  import type { Rule } from '../../../stores/settingsStore';
  
  // State
  let showAddModal = false;
  let editingRule: Rule | null = null;
  let newRule = { name: '', content: '', priority: 0, enabled: true };
  
  // Rule templates
  const ruleTemplates = [
    { 
      name: 'Code Style', 
      content: 'Always use consistent code formatting with proper indentation and naming conventions.' 
    },
    { 
      name: 'Safety Check', 
      content: 'Before executing any trading operation, verify all safety constraints and risk limits.' 
    },
    { 
      name: 'Documentation', 
      content: 'Include clear comments and documentation for all generated code.' 
    },
    { 
      name: 'Error Handling', 
      content: 'Always include proper error handling and logging in all operations.' 
    }
  ];
  
  // Reactive state
  $: rules = $settingsStore.rules;
  $: sortedRules = [...rules].sort((a, b) => b.priority - a.priority);
  
  // Add new rule
  function handleAddRule() {
    if (!newRule.name || !newRule.content) return;
    
    settingsStore.addRule({
      name: newRule.name,
      content: newRule.content,
      priority: newRule.priority,
      enabled: newRule.enabled
    });
    
    newRule = { name: '', content: '', priority: 0, enabled: true };
    showAddModal = false;
  }
  
  // Edit rule
  function startEdit(rule: Rule) {
    editingRule = { ...rule };
  }
  
  function saveEdit() {
    if (!editingRule) return;
    settingsStore.updateRule(editingRule.id, {
      name: editingRule.name,
      content: editingRule.content,
      priority: editingRule.priority,
      enabled: editingRule.enabled
    });
    editingRule = null;
  }
  
  function cancelEdit() {
    editingRule = null;
  }
  
  // Remove rule
  function handleRemoveRule(ruleId: string) {
    if (confirm('Are you sure you want to delete this rule?')) {
      settingsStore.removeRule(ruleId);
    }
  }
  
  // Toggle rule
  function toggleRule(ruleId: string, enabled: boolean) {
    settingsStore.updateRule(ruleId, { enabled });
  }
  
  // Apply template
  function applyTemplate(template: typeof ruleTemplates[0]) {
    newRule = {
      ...newRule,
      name: template.name,
      content: template.content
    };
  }
  
  // Move rule priority
  function moveRule(ruleId: string, direction: 'up' | 'down') {
    const rule = rules.find(r => r.id === ruleId);
    if (!rule) return;
    
    const newPriority = direction === 'up' ? rule.priority + 1 : Math.max(0, rule.priority - 1);
    settingsStore.updateRule(ruleId, { priority: newPriority });
  }
</script>

<div class="rules-settings">
  <div class="header">
    <div class="header-info">
      <h3>Custom Rules</h3>
      <p class="description">Define custom rules and guidelines for agent behavior.</p>
    </div>
    <button class="btn primary" on:click={() => showAddModal = true}>
      <Plus size={14} />
      Add Rule
    </button>
  </div>
  
  <!-- Rules List -->
  <div class="rules-list">
    {#if rules.length === 0}
      <div class="empty-state">
        <FileText size={32} />
        <h4>No Custom Rules</h4>
        <p>Add custom rules to guide agent behavior and responses.</p>
        <button class="btn primary" on:click={() => showAddModal = true}>
          <Plus size={14} />
          Add Your First Rule
        </button>
      </div>
    {:else}
      {#each sortedRules as rule (rule.id)}
        <div class="rule-card" class:disabled={!rule.enabled}>
          {#if editingRule?.id === rule.id}
            <!-- Edit Mode -->
            <div class="edit-form" transition:fade>
              <div class="form-group">
                <input 
                  type="text" 
                  bind:value={editingRule.name}
                  placeholder="Rule name"
                />
              </div>
              <div class="form-group">
                <textarea 
                  bind:value={editingRule.content}
                  placeholder="Rule content"
                  rows="3"
                ></textarea>
              </div>
              <div class="form-group">
                <label for="rule-priority">Priority: {editingRule.priority}</label>
                <input 
                  id="rule-priority"
                  type="range" 
                  min="0" 
                  max="10" 
                  bind:value={editingRule.priority}
                />
              </div>
              <div class="edit-actions">
                <button class="btn secondary" on:click={cancelEdit}>
                  <X size={12} /> Cancel
                </button>
                <button class="btn primary" on:click={saveEdit}>
                  <Save size={12} /> Save
                </button>
              </div>
            </div>
          {:else}
            <!-- View Mode -->
            <div class="rule-header">
              <div class="rule-info">
                <div class="rule-priority">
                  <GripVertical size={12} />
                  <span>{rule.priority}</span>
                </div>
                <span class="rule-name">{rule.name}</span>
              </div>
              <div class="rule-actions">
                <label class="toggle small">
                  <input 
                    type="checkbox" 
                    checked={rule.enabled}
                    on:change={() => toggleRule(rule.id, !rule.enabled)}
                  />
                  <span class="toggle-slider"></span>
                </label>
                <button 
                  class="icon-btn" 
                  on:click={() => startEdit(rule)}
                  title="Edit"
                >
                  <Edit3 size={14} />
                </button>
                <button 
                  class="icon-btn danger" 
                  on:click={() => handleRemoveRule(rule.id)}
                  title="Delete"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
            <div class="rule-content">
              {rule.content}
            </div>
            <div class="rule-meta">
              <span>Updated: {new Date(rule.updatedAt).toLocaleDateString()}</span>
            </div>
          {/if}
        </div>
      {/each}
    {/if}
  </div>
  
  <!-- Add Rule Modal -->
  {#if showAddModal}
    <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
    <div class="modal-overlay" on:click={() => showAddModal = false} transition:fade role="button" tabindex="-1" aria-label="Close dialog">
      <!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions a11y-no-noninteractive-element-interactions -->
      <div class="modal" on:click|stopPropagation transition:slide role="dialog" aria-modal="true" aria-labelledby="rule-modal-title">
        <h4 id="rule-modal-title">Add New Rule</h4>
        
        <!-- Templates -->
        <div class="templates-section">
          <span class="templates-label">Quick Templates:</span>
          <div class="templates-list">
            {#each ruleTemplates as template}
              <button 
                class="template-btn"
                on:click={() => applyTemplate(template)}
              >
                {template.name}
              </button>
            {/each}
          </div>
        </div>
        
        <div class="form-group">
          <label for="rule-name">Rule Name</label>
          <input 
            id="rule-name"
            type="text" 
            placeholder="e.g., Code Style"
            bind:value={newRule.name}
          />
        </div>
        
        <div class="form-group">
          <label for="rule-content">Rule Content</label>
          <textarea 
            id="rule-content"
            placeholder="Describe the rule or guideline..."
            bind:value={newRule.content}
            rows="4"
          ></textarea>
        </div>
        
        <div class="form-group">
          <label for="rule-priority">Priority: {newRule.priority}</label>
          <input 
            id="rule-priority"
            type="range" 
            min="0" 
            max="10" 
            bind:value={newRule.priority}
          />
          <span class="hint">Higher priority rules are applied first</span>
        </div>
        
        <div class="form-group">
          <label class="checkbox-label">
            <input type="checkbox" bind:checked={newRule.enabled} />
            Enable this rule
          </label>
        </div>
        
        <div class="modal-actions">
          <button class="btn secondary" on:click={() => showAddModal = false}>Cancel</button>
          <button 
            class="btn primary" 
            on:click={handleAddRule}
            disabled={!newRule.name || !newRule.content}
          >
            Add Rule
          </button>
        </div>
      </div>
    </div>
  {/if}
  
  <!-- Info Section -->
  <div class="info-section">
    <h4>About Rules</h4>
    <p>Rules are applied in priority order (highest first). They help guide agent behavior and ensure consistent responses.</p>
  </div>
</div>

<style>
  .rules-settings {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 16px;
  }
  
  .header-info h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .description {
    margin: 4px 0 0;
    font-size: 12px;
    color: var(--text-secondary);
  }
  
  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 6px;
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
  
  /* Rules List */
  .rules-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 32px;
    background: var(--bg-tertiary);
    border: 1px dashed var(--border-subtle);
    border-radius: 12px;
    color: var(--text-muted);
  }
  
  .empty-state h4 {
    margin: 12px 0 4px;
    font-size: 14px;
    color: var(--text-primary);
  }
  
  .empty-state p {
    margin: 0 0 16px;
    font-size: 12px;
  }
  
  .rule-card {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    transition: all 0.15s;
  }
  
  .rule-card.disabled {
    opacity: 0.6;
  }
  
  .rule-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .rule-info {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  
  .rule-priority {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: var(--bg-primary);
    border-radius: 4px;
    font-size: 11px;
    color: var(--text-muted);
  }
  
  .rule-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .rule-actions {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  
  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .icon-btn:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
  }
  
  .icon-btn.danger:hover {
    color: var(--accent-danger);
  }
  
  .rule-content {
    font-size: 12px;
    color: var(--text-secondary);
    line-height: 1.5;
    padding-left: 36px;
  }
  
  .rule-meta {
    font-size: 10px;
    color: var(--text-muted);
    padding-left: 36px;
  }
  
  /* Edit Form */
  .edit-form {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .form-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  
  .form-group label {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
  }
  
  .form-group input[type="text"],
  .form-group textarea {
    padding: 10px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
    font-family: inherit;
  }
  
  .form-group input:focus,
  .form-group textarea:focus {
    outline: none;
    border-color: var(--accent-primary);
  }
  
  .form-group textarea {
    resize: vertical;
    min-height: 80px;
  }
  
  .form-group input[type="range"] {
    width: 100%;
  }
  
  .hint {
    font-size: 10px;
    color: var(--text-muted);
  }
  
  .checkbox-label {
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    font-size: 12px;
    color: var(--text-secondary);
  }
  
  .edit-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
  }
  
  /* Toggle Switch */
  .toggle {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
  }
  
  .toggle.small {
    width: 32px;
    height: 18px;
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
  
  .toggle.small .toggle-slider:before {
    height: 12px;
    width: 12px;
  }
  
  .toggle input:checked + .toggle-slider {
    background-color: var(--accent-primary);
    border-color: var(--accent-primary);
  }
  
  .toggle input:checked + .toggle-slider:before {
    transform: translateX(18px);
    background-color: var(--bg-primary);
  }
  
  .toggle.small input:checked + .toggle-slider:before {
    transform: translateX(14px);
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
  
  .modal {
    background: var(--bg-secondary);
    border-radius: 12px;
    padding: 24px;
    width: 420px;
    max-width: 90%;
    max-height: 85vh;
    overflow-y: auto;
    border: 1px solid var(--border-subtle);
  }
  
  .modal h4 {
    margin: 0 0 20px;
    font-size: 16px;
    color: var(--text-primary);
  }
  
  .templates-section {
    margin-bottom: 16px;
  }
  
  .templates-label {
    font-size: 11px;
    color: var(--text-muted);
    display: block;
    margin-bottom: 8px;
  }
  
  .templates-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  
  .template-btn {
    padding: 6px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .template-btn:hover {
    background: var(--bg-primary);
    color: var(--text-primary);
    border-color: var(--accent-primary);
  }
  
  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 20px;
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
</style>
