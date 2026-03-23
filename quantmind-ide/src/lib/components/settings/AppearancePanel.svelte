<script lang="ts">
  import { theme, wallpaper, scanlines, THEME_PRESETS, getPreset, type ThemePreset } from '$lib/stores/theme';
  import { Palette, Image, Eye, EyeOff } from 'lucide-svelte';

  let wallpaperInput = $state('');
  let wallpaperError = $state('');

  function selectPreset(preset: ThemePreset) {
    theme.set(preset);
  }

  function handleWallpaperSubmit() {
    if (!wallpaperInput.trim()) {
      wallpaper.clear();
      return;
    }

    // Basic URL validation
    try {
      new URL(wallpaperInput);
      wallpaperError = '';
      wallpaper.set(wallpaperInput);
    } catch {
      wallpaperError = 'Please enter a valid URL';
    }
  }

  function clearWallpaper() {
    wallpaperInput = '';
    wallpaperError = '';
    wallpaper.clear();
  }

  // Get current preset for display
  const currentPreset = $derived(getPreset($theme));
</script>

<div class="appearance-panel">
  <div class="panel-header">
    <h3>Appearance</h3>
  </div>

  <!-- Theme Presets Section -->
  <div class="section">
    <div class="section-header">
      <Palette size={16} />
      <h4>Theme Presets</h4>
    </div>
    <p class="section-description">
      Choose a visual theme. Changes apply instantly without page reload.
    </p>

    <div class="preset-grid">
      {#each THEME_PRESETS as preset}
        <button
          class="preset-card"
          class:selected={$theme === preset.id}
          onclick={() => selectPreset(preset.id)}
        >
          <div class="preset-visual" data-theme={preset.id}>
            <div class="glass-preview"></div>
          </div>
          <div class="preset-info">
            <span class="preset-name">{preset.name}</span>
            <span class="preset-description">{preset.description}</span>
          </div>
          {#if $theme === preset.id}
            <div class="selected-indicator">
              <Eye size={14} />
            </div>
          {/if}
        </button>
      {/each}
    </div>
  </div>

  <!-- Wallpaper Section -->
  <div class="section">
    <div class="section-header">
      <Image size={16} />
      <h4>Wallpaper</h4>
    </div>
    <p class="section-description">
      Set a wallpaper image. For best results, use a dark image with the transparent window mode.
    </p>

    <div class="wallpaper-input-group">
      <input
        type="text"
        class="text-input"
        placeholder="Enter wallpaper URL (https://...)"
        bind:value={wallpaperInput}
        onkeydown={(e) => e.key === 'Enter' && handleWallpaperSubmit()}
      />
      <button class="btn secondary" onclick={handleWallpaperSubmit}>
        Apply
      </button>
    </div>

    {#if wallpaperError}
      <p class="error-message">{wallpaperError}</p>
    {/if}

    {#if $wallpaper}
      <div class="wallpaper-preview">
        <img src={$wallpaper} alt="Wallpaper preview" />
        <button class="remove-btn" onclick={clearWallpaper}>
          <EyeOff size={14} />
          Remove
        </button>
      </div>
    {/if}
  </div>

  <!-- Scan-Line Overlay Section -->
  <div class="section">
    <div class="section-header">
      <Eye size={16} />
      <h4>Scan-Line Overlay</h4>
    </div>
    <p class="section-description">
      Subtle CRT-style scan lines for retro aesthetic. Respects prefers-reduced-motion.
    </p>

    <div class="toggle-row">
      <span>Enable scan-lines</span>
      <label class="switch">
        <input
          type="checkbox"
          checked={$scanlines}
          onchange={() => scanlines.toggle()}
        />
        <span class="slider"></span>
      </label>
    </div>
  </div>

  <!-- Current Theme Info -->
  <div class="section info-section">
    <div class="info-box">
      <span class="info-label">Current Theme:</span>
      <span class="info-value">{currentPreset?.name || 'Frosted Terminal'}</span>
    </div>
    <div class="info-box">
      <span class="info-label">Color Scheme:</span>
      <span class="info-value">{currentPreset?.colorScheme || 'midnight-finance'}</span>
    </div>
  </div>
</div>

<style>
  .appearance-panel {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .panel-header h3 {
    margin: 0 0 20px;
    font-size: 15px;
    font-weight: 600;
    color: var(--text-primary, #e8eaf0);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .section {
    padding: 16px;
    background: rgba(8, 13, 20, 0.35);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
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
    margin-bottom: 16px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  /* Preset Grid */
  .preset-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .preset-card {
    display: flex;
    flex-direction: column;
    padding: 12px;
    background: var(--bg-tertiary);
    border: 2px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
    position: relative;
    text-align: left;
  }

  .preset-card:hover {
    border-color: var(--border-medium);
    transform: translateY(-2px);
  }

  .preset-card.selected {
    border-color: var(--accent-primary);
    background: var(--bg-secondary);
  }

  .preset-visual {
    height: 48px;
    border-radius: 4px;
    margin-bottom: 8px;
    overflow: hidden;
    position: relative;
  }

  .glass-preview {
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
      rgba(255, 255, 255, 0.1) 0%,
      rgba(255, 255, 255, 0.05) 100%
    );
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }

  /* Theme-specific previews */
  .preset-visual[data-theme="frosted-terminal"] {
    background: linear-gradient(135deg, #080d14 0%, #1a2332 100%);
  }

  .preset-visual[data-theme="ghost-panel"] {
    background: linear-gradient(135deg, #1a2332 0%, #2d4a5e 100%);
  }

  .preset-visual[data-theme="open-air"] {
    background: linear-gradient(135deg, #0f1729 0%, #1e293b 100%);
  }

  .preset-visual[data-theme="breathing-space"] {
    background: linear-gradient(135deg, #1e1e2e 0%, #313244 100%);
  }

  .preset-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .preset-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .preset-description {
    font-size: 11px;
    color: var(--text-muted);
  }

  .selected-indicator {
    position: absolute;
    top: 8px;
    right: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--accent-primary);
    border-radius: 50%;
    color: white;
  }

  /* Wallpaper Input */
  .wallpaper-input-group {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
  }

  .wallpaper-input-group .text-input {
    flex: 1;
  }

  .error-message {
    font-size: 12px;
    color: var(--accent-danger);
    margin-top: 4px;
  }

  .wallpaper-preview {
    position: relative;
    margin-top: 12px;
    border-radius: 8px;
    overflow: hidden;
    max-height: 200px;
  }

  .wallpaper-preview img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }

  .wallpaper-preview .remove-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 10px;
    background: rgba(0, 0, 0, 0.7);
    border: none;
    border-radius: 4px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
  }

  .wallpaper-preview .remove-btn:hover {
    background: rgba(239, 68, 68, 0.8);
  }

  /* Toggle Row */
  .toggle-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    font-size: 13px;
    color: var(--text-primary);
  }

  .toggle-row .switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
  }

  .toggle-row .switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }

  .toggle-row .slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: var(--border-subtle);
    transition: 0.3s;
    border-radius: 24px;
  }

  .toggle-row .slider:before {
    position: absolute;
    content: "";
    height: 18px;
    width: 18px;
    left: 3px;
    bottom: 3px;
    background-color: white;
    transition: 0.3s;
    border-radius: 50%;
  }

  .toggle-row input:checked + .slider {
    background-color: var(--accent-primary);
  }

  .toggle-row input:checked + .slider:before {
    transform: translateX(20px);
  }

  /* Info Section */
  .info-section {
    background: rgba(8, 13, 20, 0.5);
  }

  .info-box {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .info-box:last-child {
    border-bottom: none;
  }

  .info-label {
    color: rgba(255, 255, 255, 0.4);
  }

  .info-value {
    color: #e8eaf0;
    font-weight: 500;
  }

  /* Common styles */
  .text-input {
    padding: 8px 12px;
    background: rgba(8, 13, 20, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    color: #e8eaf0;
    font-size: 12px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  .text-input:focus {
    outline: none;
    border-color: rgba(0, 212, 255, 0.5);
    box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
  }

  .btn {
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .btn.secondary {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.65);
  }

  .btn.secondary:hover {
    background: rgba(255, 255, 255, 0.09);
    color: #fff;
  }
</style>
