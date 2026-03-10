<script>
  import '../app.css';
  import { onMount } from 'svelte';
  import { theme, applyTheme, loadSavedTheme } from '$lib/stores/themeStore';

  // Apply saved theme on mount
  onMount(() => {
    const savedTheme = loadSavedTheme();
    applyTheme(savedTheme);
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
    pointer-events: none;
  }

  .wallpaper-bg::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
      180deg,
      rgba(0, 0, 0, 0.4) 0%,
      rgba(0, 0, 0, 0.2) 50%,
      rgba(0, 0, 0, 0.5) 100%
    );
    pointer-events: none;
  }
</style>
