<script lang="ts">
  /**
   * SkeletonLoader — pulsing placeholder using --color-bg-elevated.
   * Never shows white; uses CSS token for pulse background.
   * Story 12-3
   */
  interface Props {
    lines?: number;
    height?: string;
  }

  let { lines = 3, height = '12px' }: Props = $props();
</script>

<div class="skeleton-container">
  {#each Array(lines) as _, i}
    <div
      class="skeleton-line"
      style:height={height}
      style:width={i === lines - 1 ? '60%' : '100%'}
    ></div>
  {/each}
</div>

<style>
  .skeleton-container {
    display: flex;
    flex-direction: column;
    gap: var(--space-2);
  }

  .skeleton-line {
    background: var(--color-bg-elevated);
    border-radius: 4px;
    animation: skeleton-pulse 1.5s ease-in-out infinite;
  }

  @keyframes skeleton-pulse {
    0%, 100% { opacity: 0.4; }
    50% { opacity: 0.8; }
  }
</style>
