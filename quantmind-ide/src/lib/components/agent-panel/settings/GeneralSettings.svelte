<script lang="ts">
  import { settingsStore } from '../../../stores/settingsStore';
  
  // Reactive state
  $: general = $settingsStore.general;
  
  // Theme options
  const themeOptions = [
    { value: 'dark', label: 'Dark' },
    { value: 'light', label: 'Light' },
    { value: 'system', label: 'System' }
  ];
  
  // Font size options
  const fontSizeOptions = [
    { value: 'small', label: 'Small' },
    { value: 'medium', label: 'Medium' },
    { value: 'large', label: 'Large' }
  ];
  
  // Handle changes
  function handleThemeChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    settingsStore.updateGeneral({ theme: target.value as 'dark' | 'light' | 'system' });
  }
  
  function handleFontSizeChange(e: Event) {
    const target = e.target as HTMLSelectElement;
    settingsStore.updateGeneral({ fontSize: target.value as 'small' | 'medium' | 'large' });
  }
  
  function handleToggle(key: keyof typeof general) {
    settingsStore.updateGeneral({ [key]: !general[key] });
  }
  
  function handleNumberChange(key: keyof typeof general, value: number) {
    settingsStore.updateGeneral({ [key]: value });
  }

  function getInputValue(e: Event): number {
    const target = e.target as HTMLInputElement;
    return parseInt(target.value);
  }

  function getSelectValue(e: Event): string {
    const target = e.target as HTMLSelectElement;
    return target.value;
  }
</script>

<div class="general-settings">
  <h3>General Settings</h3>
  
  <!-- Appearance Section -->
  <section class="settings-section">
    <h4>Appearance</h4>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="theme">Theme</label>
        <span class="setting-description">Choose the application color scheme</span>
      </div>
      <select id="theme" value={general.theme} on:change={handleThemeChange}>
        {#each themeOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="fontSize">Font Size</label>
        <span class="setting-description">Adjust the text size throughout the application</span>
      </div>
      <select id="fontSize" value={general.fontSize} on:change={handleFontSizeChange}>
        {#each fontSizeOptions as option}
          <option value={option.value}>{option.label}</option>
        {/each}
      </select>
    </div>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="compactMode">Compact Mode</label>
        <span class="setting-description">Reduce spacing for a more compact layout</span>
      </div>
      <label class="toggle">
        <input 
          type="checkbox" 
          id="compactMode"
          checked={general.compactMode} 
          on:change={() => handleToggle('compactMode')} 
        />
        <span class="toggle-slider"></span>
      </label>
    </div>
  </section>
  
  <!-- Behavior Section -->
  <section class="settings-section">
    <h4>Behavior</h4>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="sendOnEnter">Send on Enter</label>
        <span class="setting-description">Press Enter to send messages (Shift+Enter for new line)</span>
      </div>
      <label class="toggle">
        <input 
          type="checkbox" 
          id="sendOnEnter"
          checked={general.sendOnEnter} 
          on:change={() => handleToggle('sendOnEnter')} 
        />
        <span class="toggle-slider"></span>
      </label>
    </div>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="showTimestamps">Show Timestamps</label>
        <span class="setting-description">Display time stamps on messages</span>
      </div>
      <label class="toggle">
        <input 
          type="checkbox" 
          id="showTimestamps"
          checked={general.showTimestamps} 
          on:change={() => handleToggle('showTimestamps')} 
        />
        <span class="toggle-slider"></span>
      </label>
    </div>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="autoSave">Auto-save</label>
        <span class="setting-description">Automatically save changes</span>
      </div>
      <label class="toggle">
        <input 
          type="checkbox" 
          id="autoSave"
          checked={general.autoSave} 
          on:change={() => handleToggle('autoSave')} 
        />
        <span class="toggle-slider"></span>
      </label>
    </div>
    
    {#if general.autoSave}
      <div class="setting-item">
        <div class="setting-info">
          <label for="autoSaveInterval">Auto-save Interval</label>
          <span class="setting-description">Seconds between auto-saves</span>
        </div>
        <input 
          type="number" 
          id="autoSaveInterval"
          value={general.autoSaveInterval}
          min="10"
          max="300"
          step="10"
          on:input={(e) => handleNumberChange('autoSaveInterval', getInputValue(e))}
        />
      </div>
    {/if}
  </section>
  
  <!-- Notifications Section -->
  <section class="settings-section">
    <h4>Notifications</h4>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="notifications">Desktop Notifications</label>
        <span class="setting-description">Show desktop notifications for important events</span>
      </div>
      <label class="toggle">
        <input 
          type="checkbox" 
          id="notifications"
          checked={general.notifications} 
          on:change={() => handleToggle('notifications')} 
        />
        <span class="toggle-slider"></span>
      </label>
    </div>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="soundEffects">Sound Effects</label>
        <span class="setting-description">Play sounds for notifications</span>
      </div>
      <label class="toggle">
        <input 
          type="checkbox" 
          id="soundEffects"
          checked={general.soundEffects} 
          on:change={() => handleToggle('soundEffects')} 
        />
        <span class="toggle-slider"></span>
      </label>
    </div>
  </section>
  
  <!-- Language Section -->
  <section class="settings-section">
    <h4>Language</h4>
    
    <div class="setting-item">
      <div class="setting-info">
        <label for="language">Display Language</label>
        <span class="setting-description">Choose your preferred language</span>
      </div>
      <select id="language" value={general.language} on:change={(e) => settingsStore.updateGeneral({ language: getSelectValue(e) })}>
        <option value="en">English</option>
        <option value="es">Español</option>
        <option value="fr">Français</option>
        <option value="de">Deutsch</option>
        <option value="zh">中文</option>
        <option value="ja">日本語</option>
      </select>
    </div>
  </section>
</div>

<style>
  .general-settings {
    display: flex;
    flex-direction: column;
    gap: 24px;
  }
  
  h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }
  
  .settings-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  
  .settings-section h4 {
    margin: 0 0 4px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
  }
  
  .setting-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    border: 1px solid var(--border-subtle);
  }
  
  .setting-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
  }
  
  .setting-info label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }
  
  .setting-description {
    font-size: 11px;
    color: var(--text-muted);
  }
  
  select, input[type="number"] {
    padding: 8px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
    min-width: 120px;
  }
  
  select:focus, input[type="number"]:focus {
    outline: none;
    border-color: var(--accent-primary);
  }
  
  input[type="number"] {
    width: 80px;
    text-align: center;
  }
  
  /* Toggle Switch */
  .toggle {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
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
    border-radius: 24px;
    transition: all 0.2s;
  }
  
  .toggle-slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
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
    transform: translateX(20px);
    background-color: var(--bg-primary);
  }
  
  .toggle input:focus + .toggle-slider {
    box-shadow: 0 0 0 2px rgba(107, 200, 230, 0.3);
  }
</style>
