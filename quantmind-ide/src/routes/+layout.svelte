<script>
  import '../app.css';
  import { onMount } from 'svelte';
  import { theme, applyTheme, loadSavedTheme, setFont, loadSavedFont, setCustomWallpaper, loadSavedWallpaper } from '$lib/stores/themeStore';

  // Apply saved theme and font on mount
  onMount(() => {
    // Apply saved theme
    const savedTheme = loadSavedTheme();
    applyTheme(savedTheme);

    // Apply saved font
    const savedFont = loadSavedFont();
    setFont(savedFont);

    // Apply saved wallpaper
    const savedWallpaper = loadSavedWallpaper();
    if (savedWallpaper) {
      setCustomWallpaper(savedWallpaper);
    }
  });
</script>

<!-- Wallpaper Background -->
<div class="wallpaper-bg"></div>

<slot />

<style>
  .wallpaper-bg {
    position: fixed;
    inset: 0;
    z-index: -1;
    background: var(--wallpaper, none);
    background-position: center;
    background-repeat: no-repeat;
    pointer-events: none;
  }

  /* Different sizing based on wallpaper type */
  :global(:root) {
    --wallpaper: none;
    --wallpaper-type: none;
  }

  .wallpaper-bg {
    background-size: var(--wallpaper-type) === 'image' ? cover;
    background-size: var(--wallpaper-type) === 'gradient' ? cover;
    background-size: var(--wallpaper-type) === 'pattern' ? 20px 20px;
  }

  .wallpaper-bg::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
      180deg,
      rgba(0, 0, 0, 0.6) 0%,
      rgba(0, 0, 0, 0.4) 50%,
      rgba(0, 0, 0, 0.7) 100%
    );
    pointer-events: none;
  }
</style>
