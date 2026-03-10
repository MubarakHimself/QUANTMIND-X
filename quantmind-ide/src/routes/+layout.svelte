<script>
  import '../app.css';
  import { onMount } from 'svelte';
  import { theme, applyTheme, loadSavedTheme, setFont, loadSavedFont, setCustomWallpaper, loadSavedWallpaper, customWallpaper } from '$lib/stores/themeStore';

  // Reactive wallpaper style
  $: wallpaperStyle = $customWallpaper.startsWith('url')
    ? `url(${$customWallpaper})`
    : $customWallpaper || 'none';

  $: isImage = $customWallpaper.startsWith('url');
  $: isPattern = !$customWallpaper.startsWith('url') && $customWallpaper.includes('gradient') === false && $customWallpaper !== '';

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
<div
  class="wallpaper-bg"
  class:is-image={isImage}
  class:is-pattern={isPattern}
  style="background: {wallpaperStyle};"
></div>

<slot />

<style>
  .wallpaper-bg {
    position: fixed;
    inset: 0;
    z-index: -1;
    pointer-events: none;
    background-position: center;
    background-repeat: no-repeat;
    background-size: cover;
  }

  .wallpaper-bg.is-pattern {
    background-size: 20px 20px;
  }

  .wallpaper-bg::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
      180deg,
      rgba(0, 0, 0, 0.65) 0%,
      rgba(0, 0, 0, 0.45) 50%,
      rgba(0, 0, 0, 0.75) 100%
    );
    pointer-events: none;
  }
</style>
