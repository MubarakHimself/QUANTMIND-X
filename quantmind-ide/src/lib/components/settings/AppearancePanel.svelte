<script lang="ts">
  import {
    themes,
    applyTheme,
    currentTheme,
    customWallpaper,
    wallpaperEnabled,
    toggleWallpaper,
    wallpapers,
    fonts,
    currentFont,
    setFont,
    type ThemeName
  } from '$lib/stores/themeStore';
  import { scanlines } from '$lib/stores/theme';
  import { Palette, Image, Eye, EyeOff, Type, Grid, X } from 'lucide-svelte';

  // All 16 themes from themeStore
  const allThemes = Object.values(themes);

  let wallpaperInput = $state('');
  let wallpaperError = $state('');
  let showWallpaperGallery = $state(false);
  let fontPickerOpen = $state(false);

  function selectPreset(presetId: ThemeName) {
    applyTheme(presetId);
  }

  function handleWallpaperSubmit() {
    if (!wallpaperInput.trim()) {
      setCustomWallpaper('');
      return;
    }
    try {
      new URL(wallpaperInput);
      wallpaperError = '';
      setCustomWallpaper(wallpaperInput);
      showWallpaperGallery = false;
    } catch {
      wallpaperError = 'Please enter a valid URL';
    }
  }

  function clearWallpaper() {
    wallpaperInput = '';
    wallpaperError = '';
    setCustomWallpaper('');
  }

  function selectWallpaper(wallpaper: typeof wallpapers[0]) {
    if (wallpaper.type === 'image' && wallpaper.imageUrl) {
      setCustomWallpaper(wallpaper.imageUrl);
    } else if (wallpaper.gradient) {
      setCustomWallpaper(wallpaper.gradient);
    }
    showWallpaperGallery = false;
  }

  function selectFont(fontId: string) {
    setFont(fontId);
    fontPickerOpen = false;
  }

  // Get current preset from themes object
  const currentPreset = $derived(themes[$currentTheme]);

  // Theme visual colors for preview swatches
  function getThemeSwatch(themeId: ThemeName): string {
    const swatchColors: Record<string, string> = {
      'frosted-terminal': '#080d14',
      'trading-terminal': '#0a0a0a',
      'monokai': '#272822',
      'ambient': '#1a1a1a',
      'cyberpunk': '#0d0221',
      'matrix': '#000000',
      'dark-pro': '#1e1e1e',
      'bloomberg': '#0a0a0a',
      'crypto-quant': '#0d0d1a',
      'gold-futures': '#0c0a07',
      'forex-pro': '#0a0f0a',
      'midnight-quant': '#0a0f14',
      'ocean-blue': '#0a1628',
      'deep-space': '#050510',
      'nordic-frost': '#0c1220',
      'ghost-panel': '#1a2332',
      'open-air': '#0f1729',
      'breathing-space': '#1e1e2e'
    };
    return swatchColors[themeId] || '#080d14';
  }

  function getAccentColor(themeId: ThemeName): string {
    const accents: Record<string, string> = {
      'frosted-terminal': '#00d4ff',
      'trading-terminal': '#00ff00',
      'monokai': '#66d9ef',
      'ambient': '#ff9329',
      'cyberpunk': '#00ffff',
      'matrix': '#00ff00',
      'dark-pro': '#007acc',
      'bloomberg': '#ff9900',
      'crypto-quant': '#a855f7',
      'gold-futures': '#d4af37',
      'forex-pro': '#22c55e',
      'midnight-quant': '#14b8a6',
      'ocean-blue': '#38bdf8',
      'deep-space': '#a855f7',
      'nordic-frost': '#64748b',
      'ghost-panel': '#00d4ff',
      'open-air': '#00d4ff',
      'breathing-space': '#c084fc'
    };
    return accents[themeId] || '#00d4ff';
  }
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
      {#each allThemes as theme}
        <button
          class="preset-card"
          class:selected={$currentTheme === theme.name}
          onclick={() => selectPreset(theme.name)}
          title={theme.description}
        >
          <div class="preset-visual" style="background-color: {getThemeSwatch(theme.name)};">
            <div class="accent-dot" style="background-color: {getAccentColor(theme.name)};"></div>
            <div class="glass-preview"></div>
          </div>
          <div class="preset-info">
            <span class="preset-name">{theme.displayName}</span>
          </div>
          {#if $currentTheme === theme.name}
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
      <button class="gallery-toggle" onclick={() => showWallpaperGallery = !showWallpaperGallery}>
        {showWallpaperGallery ? 'Hide Gallery' : 'Browse Gallery'}
      </button>
    </div>
    <p class="section-description">
      Set a wallpaper image. For best results, use a dark image with transparent window mode.
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

    {#if $customWallpaper}
      <div class="wallpaper-preview">
        <img
          src={$customWallpaper.startsWith('http') ? $customWallpaper : `/api/wallpaper/${encodeURIComponent($customWallpaper)}`}
          alt="Wallpaper preview"
        />
        <button class="remove-btn" onclick={clearWallpaper}>
          <X size={14} />
          Remove
        </button>
      </div>
    {/if}

    {#if showWallpaperGallery}
      <div class="wallpaper-gallery">
        <div class="gallery-grid">
          {#each wallpapers as wallpaper}
            <button
              class="gallery-item"
              onclick={() => selectWallpaper(wallpaper)}
              title={wallpaper.name}
            >
              {#if wallpaper.type === 'image' && wallpaper.imageUrl}
                <img src={wallpaper.imageUrl} alt={wallpaper.name} />
              {:else if wallpaper.gradient}
                <div class="gradient-preview" style="background: {wallpaper.gradient};"></div>
              {:else}
                <div class="gradient-preview" style="background: #1a1a2e;"></div>
              {/if}
              <span class="gallery-item-name">{wallpaper.name}</span>
            </button>
          {/each}
        </div>
      </div>
    {/if}
  </div>

  <!-- Font Section -->
  <div class="section">
    <div class="section-header">
      <Type size={16} />
      <h4>Font</h4>
    </div>
    <p class="section-description">
      Choose a font for the interface. Monospace fonts work best for terminal aesthetics.
    </p>

    <div class="font-grid">
      {#each fonts as font}
        <button
          class="font-card"
          class:selected={$currentFont === font.id}
          onclick={() => selectFont(font.id)}
          style="font-family: {font.family};"
        >
          <span class="font-name">{font.name}</span>
          <span class="font-category">{font.category}</span>
        </button>
      {/each}
    </div>
  </div>

  <!-- Scan-Line Overlay Section -->
  <div class="section">
    <div class="section-header">
      <Grid size={16} />
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
      <span class="info-value">{currentPreset?.displayName || 'Frosted Terminal'}</span>
    </div>
    <div class="info-box">
      <span class="info-label">Description:</span>
      <span class="info-value">{currentPreset?.description || ''}</span>
    </div>
    <div class="info-box">
      <span class="info-label">Font:</span>
      <span class="info-value">{fonts.find(f => f.id === $currentFont)?.name || 'Inter'}</span>
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
    flex: 1;
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
    padding: 10px;
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
    height: 52px;
    border-radius: 4px;
    margin-bottom: 8px;
    overflow: hidden;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .accent-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    position: absolute;
    top: 6px;
    right: 6px;
  }

  .glass-preview {
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg,
      rgba(255, 255, 255, 0.08) 0%,
      rgba(255, 255, 255, 0.03) 100%
    );
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  .preset-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .preset-name {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .selected-indicator {
    position: absolute;
    top: 6px;
    left: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20px;
    height: 20px;
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

  .gallery-toggle {
    background: transparent;
    border: 1px solid var(--border-subtle);
    color: var(--text-secondary);
    font-size: 10px;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-family: 'JetBrains Mono', monospace;
  }

  .gallery-toggle:hover {
    border-color: var(--border-medium);
    color: var(--text-primary);
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

  /* Wallpaper Gallery */
  .wallpaper-gallery {
    margin-top: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 12px;
  }

  .gallery-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 8px;
    max-height: 240px;
    overflow-y: auto;
  }

  .gallery-item {
    display: flex;
    flex-direction: column;
    border-radius: 6px;
    overflow: hidden;
    cursor: pointer;
    border: 2px solid transparent;
    transition: all 0.15s;
    background: var(--bg-tertiary);
  }

  .gallery-item:hover {
    border-color: var(--border-medium);
    transform: scale(1.02);
  }

  .gallery-item img {
    width: 100%;
    height: 48px;
    object-fit: cover;
    display: block;
  }

  .gradient-preview {
    width: 100%;
    height: 48px;
  }

  .gallery-item-name {
    font-size: 9px;
    color: var(--text-muted);
    padding: 4px;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Font Grid */
  .font-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 8px;
  }

  .font-card {
    display: flex;
    flex-direction: column;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 2px solid var(--border-subtle);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
  }

  .font-card:hover {
    border-color: var(--border-medium);
  }

  .font-card.selected {
    border-color: var(--accent-primary);
    background: var(--bg-secondary);
  }

  .font-name {
    font-size: 13px;
    color: var(--text-primary);
  }

  .font-category {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 2px;
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
    text-align: right;
    max-width: 60%;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
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
