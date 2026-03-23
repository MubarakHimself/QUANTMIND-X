<script lang="ts">
  /**
   * Breadcrumb — back-button navigation component.
   * NFR-PERF-2: onBack resets state synchronously (≤200ms).
   * Story 12-3
   */
  import { ChevronLeft } from 'lucide-svelte';

  interface Props {
    onBack: () => void;
    label?: string;
  }

  let { onBack, label = 'Back' }: Props = $props();
</script>

<button class="breadcrumb-back" onclick={onBack} aria-label="Go back">
  <ChevronLeft size={16} />
  <span>{label}</span>
</button>

<style>
  .breadcrumb-back {
    display: inline-flex;
    align-items: center;
    gap: var(--space-1);
    background: none;
    border: none;
    cursor: pointer;
    color: var(--color-text-muted);
    font-family: var(--font-ambient);
    font-size: var(--text-xs);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: var(--space-1) var(--space-2);
    border-radius: 4px;
    transition: color 0.15s ease, background 0.15s ease;
    flex-shrink: 0;
  }

  .breadcrumb-back:hover {
    color: var(--color-text-primary);
    background: rgba(255, 255, 255, 0.06);
  }

  .breadcrumb-back:active {
    background: rgba(255, 255, 255, 0.1);
  }
</style>
