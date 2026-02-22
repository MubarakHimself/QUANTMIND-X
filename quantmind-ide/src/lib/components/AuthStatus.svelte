<script lang="ts">
  import { onMount } from 'svelte';
  import { settingsStore } from '$lib/stores/settingsStore';
  import { CheckCircle, XCircle, AlertCircle, Key, User, Wifi, WifiOff } from 'lucide-svelte';

  // Reactive state from settings store
  $: apiKeys = $settingsStore.apiKeys;

  // Auth status for each provider
  interface AuthStatus {
    connected: boolean;
    hasKey: boolean;
    keyValid: boolean;
    user?: string;
  }

  let geminiStatus: AuthStatus = {
    connected: false,
    hasKey: false,
    keyValid: false
  };

  let qwenStatus: AuthStatus = {
    connected: false,
    hasKey: false,
    keyValid: false
  };

  let checkingConnection = false;

  onMount(() => {
    updateAuthStatus();
    // Re-check when API keys change
    settingsStore.subscribe(() => {
      updateAuthStatus();
    });
  });

  async function updateAuthStatus() {
    // Check Gemini (Google AI) status
    const geminiKey = apiKeys.google;
    geminiStatus = {
      hasKey: !!geminiKey,
      keyValid: validateGoogleKey(geminiKey),
      connected: false // Will be updated by connection check
    };

    // Check Qwen status
    const qwenKey = apiKeys.qwen;
    qwenStatus = {
      hasKey: !!qwenKey,
      keyValid: validateQwenKey(qwenKey),
      connected: false
    };

    // Optionally test connection to API
    await testConnections();
  }

  function validateGoogleKey(key: string): boolean {
    return key?.startsWith('AIza') && key.length > 30;
  }

  function validateQwenKey(key: string): boolean {
    return key?.length > 10;
  }

  async function testConnections() {
    if (checkingConnection) return;
    checkingConnection = true;

    try {
      // Test Gemini connection if key is present
      if (geminiStatus.hasKey && geminiStatus.keyValid) {
        const geminiOk = await testGeminiConnection();
        geminiStatus = { ...geminiStatus, connected: geminiOk };
      }

      // Test Qwen connection if key is present
      if (qwenStatus.hasKey && qwenStatus.keyValid) {
        const qwenOk = await testQwenConnection();
        qwenStatus = { ...qwenStatus, connected: qwenOk };
      }
    } catch (e) {
      console.warn('Connection test failed:', e);
    } finally {
      checkingConnection = false;
    }
  }

  async function testGeminiConnection(): Promise<boolean> {
    try {
      const response = await fetch('http://localhost:8000/api/ai/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: 'google' })
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  async function testQwenConnection(): Promise<boolean> {
    try {
      const response = await fetch('http://localhost:8000/api/ai/test-connection', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider: 'qwen' })
      });
      return response.ok;
    } catch {
      return false;
    }
  }

  function getStatusIcon(status: AuthStatus) {
    if (!status.hasKey) {
      return XCircle;
    }
    if (!status.keyValid) {
      return AlertCircle;
    }
    if (status.connected) {
      return CheckCircle;
    }
    return AlertCircle;
  }

  function getStatusText(status: AuthStatus, providerName: string): string {
    if (!status.hasKey) return `${providerName}: No key`;
    if (!status.keyValid) return `${providerName}: Invalid key`;
    if (status.connected) return `${providerName}: Connected`;
    return `${providerName}: Testing...`;
  }

  function getStatusClass(status: AuthStatus): string {
    if (!status.hasKey) return 'no-key';
    if (!status.keyValid) return 'invalid';
    if (status.connected) return 'connected';
    return 'testing';
  }
</script>

<div class="auth-status">
  <!-- Gemini (Google AI) Status -->
  <div class="auth-item" class:connected={geminiStatus.connected}>
    <svelte:component this={getStatusIcon(geminiStatus)} size={12} />
    <span class="status-text">{getStatusText(geminiStatus, 'Gemini')}</span>
    {#if geminiStatus.hasKey && geminiStatus.keyValid}
      <Key size={10} class="key-indicator" />
    {/if}
  </div>

  <!-- Qwen Status -->
  <div class="auth-item" class:connected={qwenStatus.connected}>
    <svelte:component this={getStatusIcon(qwenStatus)} size={12} />
    <span class="status-text">{getStatusText(qwenStatus, 'Qwen')}</span>
    {#if qwenStatus.hasKey && qwenStatus.keyValid}
      <Key size={10} class="key-indicator" />
    {/if}
  </div>
</div>

<style>
  .auth-status {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 4px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    font-size: 11px;
  }

  .auth-item {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 2px 6px;
    border-radius: 4px;
    transition: all 0.15s ease;
  }

  .auth-item.connected {
    color: var(--accent-success);
    background: rgba(16, 185, 129, 0.1);
  }

  .auth-item:not(.connected) {
    color: var(--text-muted);
  }

  .auth-item.invalid {
    color: var(--accent-danger);
  }

  .status-text {
    white-space: nowrap;
  }

  .key-indicator {
    opacity: 0.7;
  }
</style>
