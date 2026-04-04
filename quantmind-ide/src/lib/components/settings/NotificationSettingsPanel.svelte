<script lang="ts">
  import { onMount } from 'svelte';
  import { Bell, BellOff, Mail, Webhook, Clock, Save, RefreshCw, Check, AlertCircle, Volume2, VolumeOff, Monitor } from 'lucide-svelte';
  import { settingsStore } from '$lib/stores/settingsStore';

  let isSaving = $state(false);
  let saveSuccess = $state(false);
  let saveError = $state<string | null>(null);

  // Local state bound to settingsStore
  let desktopNotifications = $state(true);
  let soundEffects = $state(false);
  let emailAlerts = $state(false);
  let webhookUrl = $state('');
  let notificationFrequency = $state<'immediate' | 'daily' | 'weekly'>('immediate');

  const frequencyOptions = [
    { value: 'immediate', label: 'Immediate' },
    { value: 'daily', label: 'Daily Digest' },
    { value: 'weekly', label: 'Weekly Summary' }
  ];

  // Load settings from store on mount
  onMount(() => {
    const unsub = settingsStore.subscribe(state => {
      desktopNotifications = state.general.notifications;
      soundEffects = state.general.soundEffects;
      emailAlerts = state.general.emailAlerts;
      webhookUrl = state.general.webhookUrl;
      notificationFrequency = state.general.notificationFrequency;
    });
    return unsub;
  });

  function updateNotification(key: 'notifications' | 'soundEffects' | 'emailAlerts', value: boolean) {
    if (key === 'notifications') desktopNotifications = value;
    else if (key === 'soundEffects') soundEffects = value;
    else if (key === 'emailAlerts') emailAlerts = value;
    settingsStore.updateGeneral({ [key]: value });
  }

  function updateWebhookUrl(value: string) {
    webhookUrl = value;
    settingsStore.updateGeneral({ webhookUrl: value });
  }

  function updateFrequency(value: 'immediate' | 'daily' | 'weekly') {
    notificationFrequency = value;
    settingsStore.updateGeneral({ notificationFrequency: value });
  }

  async function saveSettings() {
    isSaving = true;
    saveError = null;
    try {
      await settingsStore.save();
      saveSuccess = true;
      setTimeout(() => saveSuccess = false, 3000);
    } catch (e) {
      saveError = 'Failed to save notification settings';
      console.error(e);
    } finally {
      isSaving = false;
    }
  }
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Notifications</h3>
    <div class="header-actions">
      <button class="icon-btn" onclick={saveSettings} title="Save" disabled={isSaving}>
        {#if isSaving}
          <RefreshCw size={16} class="spinning" />
        {:else}
          <Save size={16} />
        {/if}
      </button>
    </div>
  </div>

  {#if saveError}
    <div class="alert error">
      <AlertCircle size={14} />
      <span>{saveError}</span>
    </div>
  {/if}

  {#if saveSuccess}
    <div class="alert success">
      <Check size={14} />
      <span>Notification settings saved</span>
    </div>
  {/if}

  <!-- Desktop Notifications -->
  <div class="settings-section">
    <div class="section-header">
      <Monitor size={16} />
      <h4>Desktop Notifications</h4>
    </div>
    <p class="section-description">
      Receive browser notifications for important events.
    </p>
    <div class="toggle-row">
      <span>Enable Desktop Notifications</span>
      <label class="switch">
        <input
          type="checkbox"
          checked={desktopNotifications}
          onchange={(e) => updateNotification('notifications', e.currentTarget.checked)}
        />
        <span class="slider"></span>
      </label>
    </div>
  </div>

  <!-- Sound Effects -->
  <div class="settings-section">
    <div class="section-header">
      {#if soundEffects}
        <Volume2 size={16} />
      {:else}
        <VolumeOff size={16} />
      {/if}
      <h4>Sound Effects</h4>
    </div>
    <p class="section-description">
      Play sounds for alerts and notifications.
    </p>
    <div class="toggle-row">
      <span>Enable Sound Effects</span>
      <label class="switch">
        <input
          type="checkbox"
          checked={soundEffects}
          onchange={(e) => updateNotification('soundEffects', e.currentTarget.checked)}
        />
        <span class="slider"></span>
      </label>
    </div>
  </div>

  <!-- Email Alerts -->
  <div class="settings-section">
    <div class="section-header">
      <Mail size={16} />
      <h4>Email Alerts</h4>
    </div>
    <p class="section-description">
      Send critical alerts to your email address.
    </p>
    <div class="toggle-row">
      <span>Enable Email Alerts</span>
      <label class="switch">
        <input
          type="checkbox"
          checked={emailAlerts}
          onchange={(e) => updateNotification('emailAlerts', e.currentTarget.checked)}
        />
        <span class="slider"></span>
      </label>
    </div>
  </div>

  <!-- Webhook URL -->
  <div class="settings-section">
    <div class="section-header">
      <Webhook size={16} />
      <h4>Slack / Discord Webhook</h4>
    </div>
    <p class="section-description">
      Send notifications to Slack or Discord via webhook URL.
    </p>
    <div class="input-row">
      <input
        type="text"
        class="text-input"
        placeholder="https://hooks.slack.com/services/... or Discord webhook URL"
        value={webhookUrl}
        oninput={(e) => updateWebhookUrl(e.currentTarget.value)}
      />
    </div>
  </div>

  <!-- Notification Frequency -->
  <div class="settings-section">
    <div class="section-header">
      <Clock size={16} />
      <h4>Notification Frequency</h4>
    </div>
    <p class="section-description">
      How often to receive non-critical notifications.
    </p>
    <div class="frequency-options">
      {#each frequencyOptions as option}
        <button
          class="frequency-btn"
          class:active={notificationFrequency === option.value}
          onclick={() => updateFrequency(option.value)}
        >
          {option.label}
        </button>
      {/each}
    </div>
  </div>

  <div class="action-row">
    <button class="btn primary" onclick={saveSettings} disabled={isSaving}>
      <Save size={14} />
      {isSaving ? 'Saving...' : 'Save Notification Settings'}
    </button>
  </div>
</div>

<style>
  .panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
  }

  .panel-header h3 {
    margin: 0;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary, #e8eaf0);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .header-actions { display: flex; gap: 8px; }

  .alert {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 12px;
    margin-bottom: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .alert.error {
    background: rgba(255, 59, 59, 0.1);
    border: 1px solid rgba(255, 59, 59, 0.25);
    color: #ff3b3b;
  }

  .alert.success {
    background: rgba(0, 200, 150, 0.08);
    border: 1px solid rgba(0, 200, 150, 0.2);
    color: #00c896;
  }

  .settings-section {
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    color: var(--text-primary, #e8eaf0);
  }

  .section-header h4 {
    margin: 0;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .section-description {
    font-size: 11px;
    color: rgba(255, 255, 255, 0.35);
    margin-bottom: 14px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  /* Toggle Row */
  .toggle-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.6);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  /* Toggle Switch */
  .switch {
    position: relative;
    display: inline-block;
    width: 40px;
    height: 22px;
  }

  .switch input { opacity: 0; width: 0; height: 0; }

  .slider {
    position: absolute;
    cursor: pointer;
    inset: 0;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 22px;
    transition: 0.2s;
  }

  .slider:before {
    position: absolute;
    content: "";
    height: 14px;
    width: 14px;
    left: 3px;
    bottom: 3px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 50%;
    transition: 0.2s;
  }

  input:checked + .slider {
    background: rgba(0, 212, 255, 0.25);
    border-color: rgba(0, 212, 255, 0.4);
  }

  input:checked + .slider:before {
    transform: translateX(18px);
    background: #00d4ff;
  }

  /* Text Input */
  .input-row {
    margin-top: 8px;
  }

  .text-input {
    width: 100%;
    padding: 8px 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: #e8eaf0;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    transition: border-color 0.15s, box-shadow 0.15s;
    box-sizing: border-box;
  }

  .text-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
  }

  .text-input::placeholder {
    color: rgba(255, 255, 255, 0.25);
  }

  /* Frequency Options */
  .frequency-options {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .frequency-btn {
    padding: 8px 14px;
    background: rgba(8, 13, 20, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    color: rgba(255, 255, 255, 0.5);
    font-size: 11px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.15s;
  }

  .frequency-btn:hover {
    border-color: rgba(255, 255, 255, 0.15);
    color: rgba(255, 255, 255, 0.7);
  }

  .frequency-btn.active {
    background: rgba(0, 212, 255, 0.08);
    border-color: rgba(0, 212, 255, 0.35);
    color: #00d4ff;
  }

  /* Action Row */
  .action-row {
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 4px;
  }

  .btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }

  .btn.primary {
    background: rgba(0, 212, 255, 0.15);
    border: 1px solid rgba(0, 212, 255, 0.4);
    color: #00d4ff;
  }

  .btn.primary:hover { background: rgba(0, 212, 255, 0.25); }
  .btn.primary:disabled { opacity: 0.45; cursor: not-allowed; }

  .icon-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border: none;
    border-radius: 6px;
    background: rgba(255, 255, 255, 0.04);
    color: rgba(255, 255, 255, 0.4);
    cursor: pointer;
    transition: all 0.15s;
  }

  .icon-btn:hover { background: rgba(255, 255, 255, 0.1); color: #e8eaf0; }
  .icon-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  .spinning { animation: spin 1s linear infinite; }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
</style>
