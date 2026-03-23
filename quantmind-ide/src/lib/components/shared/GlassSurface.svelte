<script lang="ts">
  /**
   * GlassSurface — reusable glass container.
   * tier 1 = --glass-shell-bg (shell surfaces, 0.08 opacity)
   * tier 2 = --glass-content-bg (content tiles, 0.35 opacity)
   * Story 12-3
   */
  interface Props {
    tier?: 1 | 2;
    class?: string;
  }

  let { tier = 2, class: extraClass = '' }: Props = $props();
</script>

<div class="glass-surface glass-surface--tier-{tier} {extraClass}">
  <slot />
</div>

<style>
  .glass-surface {
    backdrop-filter: var(--glass-blur);
    -webkit-backdrop-filter: var(--glass-blur);
    border: 1px solid var(--color-border-subtle);
    border-radius: 8px;
  }

  .glass-surface--tier-1 {
    background: var(--glass-shell-bg);
  }

  .glass-surface--tier-2 {
    background: var(--glass-content-bg);
  }
</style>
