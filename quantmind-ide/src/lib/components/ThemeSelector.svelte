<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { currentTheme, applyTheme, themes, customWallpaper, setCustomWallpaper } from '../stores/themeStore';
  import { Palette, Sun, Moon, Sparkles, Image as ImageIcon, X, Check } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  let showCustomWallpaper = false;
  let customWallpaperUrl = '';

  function selectTheme(themeName: string) {
    applyTheme(themeName);
    dispatch('themeChanged', { theme: themeName });
  }

  function applyCustomWallpaper() {
    if (customWallpaperUrl) {
      setCustomWallpaper(customWallpaperUrl);
      dispatch('wallpaperChanged', { wallpaper: customWallpaperUrl });
      showCustomWallpaper = false;
      customWallpaperUrl = '';
    }
  }

  function resetToDefaultWallpaper() {
    setCustomWallpaper('');
    dispatch('wallpaperChanged', { wallpaper: '' });
  }
</script>

<div class="theme-selector">
  <div class="theme-header">
    <Palette size={20} />
    <h3>Theme & Appearance</h3>
  </div>

  <div class="theme-section">
    <h4>Color Themes</h4>
    <div class="theme-grid">
      {#each Object.entries(themes) as [name, theme]}
        <button
          class="theme-card"
          class:active={$currentTheme === name}
          on:click={() => selectTheme(name)}
          title={theme.description}
        >
          <div class="theme-preview" style="background: {theme.colors.bg.primary}; border-color: {theme.colors.border.accent};">
            <div class="preview-header" style="background: {theme.colors.bg.secondary};">
              <div class="preview-title" style="color: {theme.colors.text.primary};">Code</div>
              <div class="preview-actions">
                <div class="preview-dot" style="background: {theme.colors.accent.primary}"></div>
                <div class="preview-dot" style="background: {theme.colors.accent.secondary}"></div>
              </div>
            </div>
            <div class="preview-content" style="color: {theme.colors.text.secondary};">
              <div class="preview-line" style="color: {theme.colors.syntax.keyword};">function</div>
              <div class="preview-line" style="color: {theme.colors.syntax.function}">example</div>
              <div class="preview-line" style="color: {theme.colors.syntax.string}">"Hello World"</div>
            </div>
          </div>
          <div class="theme-info">
            <div class="theme-name">{theme.displayName}</div>
            {#if $currentTheme === name}
              <Check size={16} class="active-indicator" />
            {/if}
          </div>
        </button>
      {/each}
    </div>
  </div>

  <div class="theme-section">
    <h4>Wallpapers</h4>
    <div class="wallpaper-grid">
      <!-- Gradient Wallpapers -->
      {#each Object.values(themes).filter(t => t.wallpaper?.type === 'gradient') as wallpaper}
        <button
          class="wallpaper-card"
          class:active={customWallpaperUrl === wallpaper.gradient}
          on:click={() => {
            setCustomWallpaper(wallpaper.gradient || '');
            customWallpaperUrl = wallpaper.gradient || '';
          }}
          title={wallpaper.name}
        >
          <div class="wallpaper-preview" style="background: {wallpaper.gradient}"></div>
          <div class="wallpaper-info">
            <div class="wallpaper-name">{wallpaper.name}</div>
            {#if customWallpaperUrl === wallpaper.gradient}
              <Check size={16} class="active-indicator" />
            {/if}
          </div>
        </button>
      {/each}

      <!-- Pattern Wallpapers -->
      {#each Object.values(themes).filter(t => t.wallpaper?.type === 'pattern') as wallpaper}
        <button
          class="wallpaper-card"
          class:active={customWallpaperUrl === wallpaper.gradient}
          on:click={() => {
            setCustomWallpaper(wallpaper.gradient || '');
            customWallpaperUrl = wallpaper.gradient || '';
          }}
          title={wallpaper.name}
        >
          <div class="wallpaper-preview pattern" style="background: {wallpaper.gradient}"></div>
          <div class="wallpaper-info">
            <div class="wallpaper-name">{wallpaper.name}</div>
            {#if customWallpaperUrl === wallpaper.gradient}
              <Check size={16} class="active-indicator" />
            {/if}
          </div>
        </button>
      {/each}
    </div>

    <button 
      class="custom-wallpaper-btn"
      on:click={() => showCustomWallpaper = !showCustomWallpaper}
    >
      <ImageIcon size={16} />
      Custom Wallpaper
    </button>

    {#if customWallpaperUrl}
      <button 
        class="reset-wallpaper-btn"
        on:click={resetToDefaultWallpaper}
      >
        <X size={16} />
        Reset
      </button>
    {/if}
  </div>

  {#if showCustomWallpaper}
    <div class="custom-wallpaper-modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Custom Wallpaper</h3>
          <button class="close-btn" on:click={() => showCustomWallpaper = false}>
            <X size={18} />
          </button>
        </div>
        <div class="modal-body">
          <div class="form-group">
            <label for="wallpaper-url">Wallpaper URL</label>
            <input
              id="wallpaper-url"
              type="url"
              placeholder="https://example.com/image.jpg"
              bind:value={customWallpaperUrl}
            />
          </div>
          <div class="form-actions">
            <button class="btn secondary" on:click={() => showCustomWallpaper = false}>
              Cancel
            </button>
            <button class="btn primary" on:click={applyCustomWallpaper}>
              Apply
            </button>
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .theme-selector {
    padding: 24px;
    max-width: 800px;
    margin: 0 auto;
  }

  .theme-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .theme-header h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .theme-section {
    margin-bottom: 32px;
  }

  .theme-section h4 {
    margin: 0 0 16px 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .theme-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
  }

  .theme-card {
    position: relative;
    background: var(--bg-secondary);
    border: 2px solid var(--border-subtle);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    overflow: hidden;
  }

  .theme-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
  }

  .theme-card.active {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(var(--accent-primary), 0.3);
  }

  .theme-preview {
    height: 80px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .preview-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 10px;
  }

  .preview-title {
    font-weight: 600;
  }

  .preview-actions {
    display: flex;
    gap: 4px;
  }

  .preview-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }

  .preview-content {
    flex: 1;
    display: flex;
    gap: 4px;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 10px;
    line-height: 1.2;
  }

  .preview-line {
    white-space: nowrap;
  }

  .theme-info {
    padding: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .theme-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .active-indicator {
    color: var(--accent-primary);
  }

  .wallpaper-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }

  .wallpaper-card {
    position: relative;
    background: var(--bg-secondary);
    border: 2px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    overflow: hidden;
  }

  .wallpaper-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .wallpaper-card.active {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 2px rgba(var(--accent-primary), 0.3);
  }

  .wallpaper-preview {
    height: 60px;
    border-radius: 6px;
  }

  .wallpaper-preview.pattern {
    background-size: 20px 20px !important;
  }

  .wallpaper-info {
    padding: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .wallpaper-name {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .custom-wallpaper-btn,
  .reset-wallpaper-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
    font-size: 13px;
  }

  .custom-wallpaper-btn:hover,
  .reset-wallpaper-btn:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }

  .reset-wallpaper-btn {
    background: var(--bg-tertiary);
    border-color: var(--border-danger);
    color: var(--text-danger);
  }

  .reset-wallpaper-btn:hover {
    background: var(--bg-danger);
    color: white;
  }

  .custom-wallpaper-modal {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-content {
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    width: 90%;
    max-width: 500px;
    max-height: 90vh;
    overflow-y: auto;
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .modal-header h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .close-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    transition: all 0.15s;
  }

  .close-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .modal-body {
    padding: 24px;
  }

  .form-group {
    margin-bottom: 20px;
  }

  .form-group label {
    display: block;
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
  }

  .form-group input {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 14px;
    outline: none;
    transition: all 0.15s;
  }

  .form-group input:focus {
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 3px rgba(var(--accent-primary), 0.1);
  }

  .form-actions {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
    margin-top: 24px;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: white;
  }

  .btn.primary:hover {
    opacity: 0.9;
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    border-color: var(--border-subtle);
    color: var(--text-secondary);
  }

  .btn.secondary:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }
</style>
