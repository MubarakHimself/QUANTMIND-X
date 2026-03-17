<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { currentTheme, applyTheme, themes, customWallpaper, setCustomWallpaper, wallpapers, fonts, currentFont, setFont, wallpaperEnabled, toggleWallpaper, customWallpapers, fetchCustomWallpapers, uploadCustomWallpaper, deleteCustomWallpaper, activateCustomWallpaper, type CustomWallpaper } from '../stores/themeStore';
  import { Palette, Sun, Moon, Sparkles, Image as ImageIcon, X, Check, Type, Upload, Trash2 } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  let showCustomWallpaper = $state(false);
  let customWallpaperUrl = $state('');
  let uploading = $state(false);
  let uploadedWallpapers: CustomWallpaper[] = $state([]);

  // Load custom wallpapers on mount
  onMount(async () => {
    uploadedWallpapers = await fetchCustomWallpapers();
  });

  function selectTheme(themeName: string) {
    applyTheme(themeName);
    dispatch('themeChanged', { theme: themeName });
  }

  function selectWallpaper(wallpaper: typeof wallpapers[0]) {
    if (wallpaper.imageUrl) {
      // Image wallpaper
      setCustomWallpaper(wallpaper.imageUrl);
      customWallpaperUrl = wallpaper.imageUrl;
      dispatch('wallpaperChanged', { wallpaper: wallpaper.imageUrl });
    } else if (wallpaper.gradient) {
      // Gradient or pattern
      setCustomWallpaper(wallpaper.gradient);
      customWallpaperUrl = wallpaper.gradient;
      dispatch('wallpaperChanged', { wallpaper: wallpaper.gradient });
    }
  }

  function selectFont(fontId: string) {
    setFont(fontId);
    dispatch('fontChanged', { font: fontId });
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
    customWallpaperUrl = '';
    dispatch('wallpaperChanged', { wallpaper: '' });
  }

  // Custom wallpaper upload handlers
  let fileInput: HTMLInputElement = $state();

  async function handleFileSelect(event: Event) {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (!file) return;

    // Validate file
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      alert('Please select a valid image file (JPEG, PNG, GIF, or WebP)');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      alert('File size must be less than 10MB');
      return;
    }

    const name = file.name.replace(/\.[^/.]+$/, ''); // Remove extension
    uploading = true;

    const wallpaper = await uploadCustomWallpaper(file, name);
    uploading = false;

    if (wallpaper) {
      uploadedWallpapers = [...uploadedWallpapers, wallpaper];
      // Auto-activate the uploaded wallpaper
      await activateCustomWallpaper(wallpaper.id);
      customWallpaperUrl = wallpaper.urlPath;
    }
  }

  async function handleDeleteWallpaper(id: string) {
    if (confirm('Are you sure you want to delete this wallpaper?')) {
      const success = await deleteCustomWallpaper(id);
      if (success) {
        uploadedWallpapers = uploadedWallpapers.filter(w => w.id !== id);
      }
    }
  }

  async function handleActivateWallpaper(wallpaper: CustomWallpaper) {
    await activateCustomWallpaper(wallpaper.id);
    customWallpaperUrl = wallpaper.urlPath;
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
          onclick={() => selectTheme(name)}
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

  <!-- Wallpaper Toggle -->
  <div class="theme-section">
    <div class="wallpaper-toggle">
      <span>Show Wallpaper</span>
      <label class="switch">
        <input
          type="checkbox"
          checked={$wallpaperEnabled}
          onchange={(e) => toggleWallpaper(e.currentTarget.checked)}
        />
        <span class="slider"></span>
      </label>
    </div>
  </div>

  <!-- Wallpaper Selection -->
  <div class="theme-section">
    <h4>Wallpapers</h4>
    <div class="wallpaper-grid">
      {#each wallpapers as wallpaper}
        <button
          class="wallpaper-card"
          class:active={customWallpaperUrl === (wallpaper.gradient || wallpaper.imageUrl)}
          onclick={() => selectWallpaper(wallpaper)}
          title={wallpaper.name}
        >
          <div
            class="wallpaper-preview"
            class:pattern={wallpaper.type === 'pattern'}
            class:image-wall={wallpaper.type === 'image'}
            style="background: {wallpaper.type === 'image' ? `url(${wallpaper.imageUrl})` : wallpaper.gradient}; background-size: cover"
          ></div>
          <div class="wallpaper-info">
            <div class="wallpaper-name">{wallpaper.name}</div>
            {#if customWallpaperUrl === (wallpaper.gradient || wallpaper.imageUrl)}
              <Check size={16} class="active-indicator" />
            {/if}
          </div>
        </button>
      {/each}
    </div>
  </div>

  <!-- Custom Wallpaper Upload -->
  <div class="theme-section">
    <h4>
      <Upload size={16} />
      Your Images
    </h4>
    <p class="section-description">Upload your own images to use as wallpapers</p>

    <!-- Upload button -->
    <div class="upload-section">
      <input
        type="file"
        accept="image/jpeg,image/png,image/gif,image/webp"
        bind:this={fileInput}
        onchange={handleFileSelect}
        style="display: none;"
      />
      <button
        class="upload-btn"
        onclick={() => fileInput.click()}
        disabled={uploading}
      >
        {#if uploading}
          <span class="spinner"></span>
          Uploading...
        {:else}
          <Upload size={18} />
          Upload Image
        {/if}
      </button>
    </div>

    <!-- Uploaded wallpapers grid -->
    {#if uploadedWallpapers.length > 0}
      <div class="wallpaper-grid">
        {#each uploadedWallpapers as wallpaper}
          <div class="wallpaper-card" class:active={customWallpaperUrl === wallpaper.urlPath}>
            <div
              class="wallpaper-preview image-wall"
              style="background: url({wallpaper.urlPath}); background-size: cover; background-position: center;"
            ></div>
            <div class="wallpaper-info">
              <div class="wallpaper-name">{wallpaper.name}</div>
              <div class="wallpaper-actions">
                <button
                  class="action-btn activate"
                  onclick={() => handleActivateWallpaper(wallpaper)}
                  title="Set as wallpaper"
                >
                  <Check size={14} />
                </button>
                <button
                  class="action-btn delete"
                  onclick={() => handleDeleteWallpaper(wallpaper.id)}
                  title="Delete"
                >
                  <Trash2 size={14} />
                </button>
              </div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>

  <!-- Font Selection -->
  <div class="theme-section">
    <h4>
      <Type size={16} />
      Font
    </h4>
    <div class="font-grid">
      {#each fonts as font}
        <button
          class="font-card"
          class:active={$currentFont === font.id}
          onclick={() => selectFont(font.id)}
          style="font-family: {font.family}"
          title={font.category}
        >
          <span class="font-name">{font.name}</span>
          <span class="font-category">{font.category}</span>
          {#if $currentFont === font.id}
            <Check size={14} class="active-indicator" />
          {/if}
        </button>
      {/each}
    </div>
  </div>

  {#if showCustomWallpaper}
    <div class="custom-wallpaper-modal">
      <div class="modal-content">
        <div class="modal-header">
          <h3>Custom Wallpaper</h3>
          <button class="close-btn" onclick={() => showCustomWallpaper = false}>
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
            <button class="btn secondary" onclick={() => showCustomWallpaper = false}>
              Cancel
            </button>
            <button class="btn primary" onclick={applyCustomWallpaper}>
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

  .wallpaper-preview.image-wall {
    background-size: cover !important;
    background-position: center;
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

  /* Font Selection */
  .font-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 10px;
  }

  .font-card {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
    padding: 12px;
    background: var(--bg-secondary);
    border: 2px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
  }

  .font-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    border-color: var(--border-medium);
  }

  .font-card.active {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.1);
  }

  .font-name {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .font-category {
    font-size: 11px;
    color: var(--text-muted);
    text-transform: capitalize;
  }

  .font-card .active-indicator {
    position: absolute;
    top: 8px;
    right: 8px;
    color: var(--accent-primary);
  }

  /* Theme section h4 with icon */
  .theme-section h4 {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* Wallpaper Toggle */
  .wallpaper-toggle {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .wallpaper-toggle span {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  /* Upload Section */
  .section-description {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 16px;
  }

  .upload-section {
    margin-bottom: 16px;
  }

  .upload-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: var(--accent-primary);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
  }

  .upload-btn:hover:not(:disabled) {
    opacity: 0.9;
    transform: translateY(-1px);
  }

  .upload-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(255, 255, 255, 0.3);
    border-top-color: white;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .wallpaper-actions {
    display: flex;
    gap: 4px;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }

  .action-btn:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }

  .action-btn.activate:hover {
    background: var(--accent-success);
    border-color: var(--accent-success);
    color: white;
  }

  .action-btn.delete:hover {
    background: var(--accent-danger);
    border-color: var(--accent-danger);
    color: white;
  }

  .wallpaper-card.active .wallpaper-info {
    background: rgba(99, 102, 241, 0.1);
  }
</style>
