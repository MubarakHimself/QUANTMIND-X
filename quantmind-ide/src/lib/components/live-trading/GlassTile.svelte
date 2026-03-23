<script lang="ts">
  /**
   * GlassTile - Frosted Terminal Glass Container
   *
   * Tier 2 glass styling for Live Trading bot status tiles.
   * Background: rgba(8,13,20,0.35) with blur(16px)
   */

  export let clickable = false;
</script>

<div class="glass-tile" class:clickable>
  <slot />
</div>

<style>
  .glass-tile {
    /* Tier 2 Frosted Terminal Glass */
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px) saturate(120%);
    -webkit-backdrop-filter: blur(16px) saturate(120%);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    transition: transform 0.15s ease, border-color 0.2s ease, box-shadow 0.2s ease;
  }

  .clickable {
    cursor: pointer;
  }

  .clickable:hover {
    transform: translateY(-2px);
    border-color: rgba(0, 212, 255, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  }

  .clickable:active {
    transform: translateY(0);
  }
</style>
