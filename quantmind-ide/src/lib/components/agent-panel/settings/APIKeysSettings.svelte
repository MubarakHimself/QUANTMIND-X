<script lang="ts">
  import { Eye, EyeOff, Check, AlertCircle, Trash2 } from 'lucide-svelte';
  import { settingsStore } from '../../../stores/settingsStore';
  
  // State
  let visibleKeys: Record<string, boolean> = {
    google: false,
    anthropic: false,
    openai: false,
    qwen: false
  };
  
  let savedKeys: Record<string, boolean> = {};
  
  // Reactive state
  $: apiKeys = $settingsStore.apiKeys;
  
  // Provider configuration
  const providers = [
    { id: 'google', name: 'Google AI', placeholder: 'AIza...', docs: 'https://ai.google.dev/' },
    { id: 'anthropic', name: 'Anthropic', placeholder: 'sk-ant-...', docs: 'https://console.anthropic.com/' },
    { id: 'openai', name: 'OpenAI', placeholder: 'sk-...', docs: 'https://platform.openai.com/' },
    { id: 'qwen', name: 'Qwen', placeholder: 'sk-...', docs: 'https://dashscope.aliyun.com/' }
  ];
  
  // Toggle key visibility
  function toggleVisibility(provider: string) {
    visibleKeys[provider] = !visibleKeys[provider];
    visibleKeys = visibleKeys;
  }
  
  // Update API key
  function updateKey(provider: string, value: string) {
    settingsStore.updateAPIKeys({ [provider]: value });
    savedKeys[provider] = false;
    savedKeys = savedKeys;
  }
  
  // Clear API key
  function clearKey(provider: string) {
    settingsStore.updateAPIKeys({ [provider]: '' });
    savedKeys[provider] = false;
    savedKeys = savedKeys;
  }
  
  // Validate key format (basic)
  function validateKey(provider: string, key: string): boolean {
    if (!key) return true; // Empty is valid (not set)
    
    switch (provider) {
      case 'google':
        return key.startsWith('AIza');
      case 'anthropic':
        return key.startsWith('sk-ant-');
      case 'openai':
        return key.startsWith('sk-');
      case 'qwen':
        return key.length > 10;
      default:
        return true;
    }
  }
  
  // Get validation status
  function getValidationStatus(provider: string): 'valid' | 'invalid' | 'empty' {
    const key = apiKeys[provider as keyof typeof apiKeys];
    if (!key) return 'empty';
    return validateKey(provider, key) ? 'valid' : 'invalid';
  }
  
  // Mask key for display
  function maskKey(key: string): string {
    if (!key || key.length < 8) return key;
    return key.slice(0, 4) + '•'.repeat(Math.min(key.length - 8, 20)) + key.slice(-4);
  }

  // Get API key for provider (helper to avoid TypeScript in template)
  function getApiKey(providerId: string): string {
    return apiKeys[providerId as keyof typeof apiKeys] || '';
  }

  // Get input value helper
  function getInputValue(e: Event): string {
    return (e.target as HTMLInputElement).value;
  }
</script>

<div class="api-keys-settings">
  <h3>API Keys</h3>
  <p class="description">Configure your API keys for different AI providers. Keys are stored locally and encrypted.</p>
  
  <!-- Security Notice -->
  <div class="security-notice">
    <AlertCircle size={14} />
    <span>Your API keys are stored locally in your browser and encrypted. Never share your keys with others.</span>
  </div>
  
  <!-- Provider Keys -->
  <div class="keys-list">
    {#each providers as provider}
      {@const key = getApiKey(provider.id)}
      {@const status = getValidationStatus(provider.id)}
      
      <div class="key-item">
        <div class="key-header">
          <label for="key-{provider.id}">{provider.name}</label>
          <div class="key-status">
            {#if status === 'valid'}
              <span class="status valid"><Check size={12} /> Set</span>
            {:else if status === 'invalid'}
              <span class="status invalid"><AlertCircle size={12} /> Invalid format</span>
            {:else}
              <span class="status empty">Not set</span>
            {/if}
          </div>
        </div>
        
        <div class="key-input-group">
          <input
            id="key-{provider.id}"
            type={visibleKeys[provider.id] ? 'text' : 'password'}
            placeholder={provider.placeholder}
            value={key}
            on:input={(e) => updateKey(provider.id, getInputValue(e))}
            class:invalid={status === 'invalid'}
          />
          
          <button 
            class="input-btn" 
            on:click={() => toggleVisibility(provider.id)}
            title={visibleKeys[provider.id] ? 'Hide' : 'Show'}
            aria-label={visibleKeys[provider.id] ? 'Hide API key' : 'Show API key'}
          >
            {#if visibleKeys[provider.id]}
              <EyeOff size={14} />
            {:else}
              <Eye size={14} />
            {/if}
          </button>
          
          {#if key}
            <button 
              class="input-btn danger" 
              on:click={() => clearKey(provider.id)}
              title="Clear"
              aria-label="Clear API key"
            >
              <Trash2 size={14} />
            </button>
          {/if}
        </div>
        
        <a 
          href={provider.docs} 
          target="_blank" 
          rel="noopener noreferrer"
          class="docs-link"
        >
          Get API key →
        </a>
      </div>
    {/each}
  </div>
  
  <!-- Local Storage Info -->
  <div class="storage-info">
    <h4>Storage Information</h4>
    <p>API keys are stored in your browser's local storage with basic encryption. For enhanced security:</p>
    <ul>
      <li>Use environment variables on your backend server</li>
      <li>Rotate keys periodically</li>
      <li>Never commit keys to version control</li>
    </ul>
  </div>
</div>

<style>
  .api-keys-settings {
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
    font-size: 13px;
    color: var(--text-secondary);
  }
  
  .security-notice {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid var(--accent-warning);
    border-radius: 8px;
    font-size: 12px;
    color: var(--accent-warning);
  }
  
  .keys-list {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  
  .key-item {
    display: flex;
    flex-direction: column;
    gap: 8px;
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }
  
  .key-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  
  .key-header label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .key-status {
    display: flex;
    align-items: center;
  }
  
  .status {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
  }
  
  .status.valid {
    background: rgba(16, 185, 129, 0.1);
    color: var(--accent-success);
  }
  
  .status.invalid {
    background: rgba(239, 68, 68, 0.1);
    color: var(--accent-danger);
  }
  
  .status.empty {
    background: var(--bg-primary);
    color: var(--text-muted);
  }
  
  .key-input-group {
    display: flex;
    gap: 4px;
  }
  
  .key-input-group input {
    flex: 1;
    padding: 10px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: monospace;
  }
  
  .key-input-group input:focus {
    outline: none;
    border-color: var(--accent-primary);
  }
  
  .key-input-group input.invalid {
    border-color: var(--accent-danger);
  }
  
  .key-input-group input::placeholder {
    color: var(--text-muted);
    font-family: inherit;
  }
  
  .input-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .input-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .input-btn.danger:hover {
    color: var(--accent-danger);
    border-color: var(--accent-danger);
  }
  
  .docs-link {
    font-size: 11px;
    color: var(--accent-primary);
    text-decoration: none;
    align-self: flex-start;
  }
  
  .docs-link:hover {
    text-decoration: underline;
  }
  
  .storage-info {
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
  }
  
  .storage-info h4 {
    margin: 0 0 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .storage-info p {
    margin: 0 0 8px;
    font-size: 12px;
    color: var(--text-secondary);
  }
  
  .storage-info ul {
    margin: 0;
    padding-left: 20px;
    font-size: 11px;
    color: var(--text-muted);
  }
  
  .storage-info li {
    margin-bottom: 4px;
  }
</style>
