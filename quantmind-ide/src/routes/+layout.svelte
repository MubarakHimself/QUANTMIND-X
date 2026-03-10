<script>
  import '../app.css';
  import { onMount } from 'svelte';
  import { theme, applyTheme, loadSavedTheme, setFont, loadSavedFont, currentFont, fonts, wallpapers, currentTheme } from '$lib/stores/themeStore';

  // Apply saved theme and font on mount
  onMount(() => {
    // Apply saved theme
    const savedTheme = loadSavedTheme();
    applyTheme(savedTheme);

    // Apply saved font
    const savedFont = loadSavedFont();
    setFont(savedFont);
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
    background-size: cover;
    pointer-events: none;
  }

  .wallpaper-bg.pattern-bg {
    background-size: 20px 20px;
  }

  .wallpaper-bg::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
      180deg,
      rgba(0, 0, 0, 0.5) 0%,
      rgba(0, 0, 0, 0.3) 50%,
      rgba(0, 0, 0, 0.6) 100%
    );
    pointer-events: none;
  }
</style>
