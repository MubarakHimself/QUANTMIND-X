<script lang="ts">
  import { ChevronRight, Home } from "lucide-svelte";

  interface Props {
    items?: Array<{ label: string; path?: string }>;
    onNavigate?: ((path: string) => void) | undefined;
    showHome?: boolean;
  }

  let { items = [], onNavigate = undefined, showHome = true }: Props = $props();

  function handleClick(
    item: { label: string; path?: string; fullPath?: string; id?: string },
    index: number,
  ) {
    const targetPath = item.path || item.fullPath || item.id;
    if (targetPath && onNavigate && index < items.length - 1) {
      onNavigate(targetPath);
    }
  }
</script>

<nav class="breadcrumbs" aria-label="Breadcrumb navigation">
  <ol class="breadcrumb-list">
    {#if showHome}
      <li class="breadcrumb-item">
        <button
          class="breadcrumb-link home"
          onclick={() => onNavigate?.("/")}
          aria-label="Go to home"
        >
          <Home size={14} />
        </button>
        <span class="separator-icon" aria-hidden="true">
          <ChevronRight size={12} />
        </span>
      </li>
    {/if}

    {#each items as item, index}
      <li class="breadcrumb-item">
        {#if index === items.length - 1}
          <span class="breadcrumb-current" aria-current="page">
            {item.label}
          </span>
        {:else}
          <button
            class="breadcrumb-link"
            onclick={() => handleClick(item, index)}
          >
            {item.label}
          </button>
          <span class="separator-icon" aria-hidden="true">
            <ChevronRight size={12} />
          </span>
        {/if}
      </li>
    {/each}
  </ol>
</nav>

<style>
  .breadcrumbs {
    display: flex;
    align-items: center;
    padding: 8px 12px;
    background: var(--color-bg-elevated);
    border-bottom: 1px solid var(--color-border-subtle);
    font-size: 12px;
  }

  .breadcrumb-list {
    display: flex;
    align-items: center;
    gap: 4px;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .breadcrumb-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .breadcrumb-link {
    background: transparent;
    border: none;
    color: var(--color-text-secondary);
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 12px;
    transition: all 0.15s ease;
  }

  .breadcrumb-link:hover {
    background: var(--color-bg-surface);
    color: var(--color-text-primary);
  }

  .breadcrumb-link.home {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px;
  }

  .breadcrumb-current {
    color: var(--color-text-primary);
    font-weight: 500;
    padding: 2px 6px;
  }

  .separator-icon {
    color: var(--color-text-muted);
    flex-shrink: 0;
    display: flex;
    align-items: center;
  }
</style>
