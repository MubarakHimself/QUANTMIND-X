<script lang="ts">
  import { onMount } from 'svelte';

  interface Hook {
    id: string;
    name: string;
    type: 'pre' | 'post';
    agent: string;
    event: string;
    enabled: boolean;
  }

  // Load from hook registry
  let hooks: Hook[] = [
    { id: 'pre_analyst', name: 'Pre Analyst', type: 'pre', agent: 'analyst', event: 'task_start', enabled: true },
    { id: 'post_analyst', name: 'Post Analyst', type: 'post', agent: 'analyst', event: 'task_complete', enabled: true },
    { id: 'copilot_log_trades', name: 'Copilot Log Trades', type: 'post', agent: 'copilot', event: 'trade', enabled: true },
  ];
</script>

<div class="hooks-viewer">
  <h3>Event Hooks</h3>

  <div class="hook-list">
    {#each hooks as hook}
      <div class="hook-item">
        <div class="hook-info">
          <span class="hook-type" class:pre={hook.type === 'pre'} class:post={hook.type === 'post'}>
            {hook.type}
          </span>
          <strong>{hook.name}</strong>
          <span class="hook-event">{hook.event}</span>
        </div>
        <label class="toggle">
          <input type="checkbox" checked={hook.enabled} />
          <span class="slider"></span>
        </label>
      </div>
    {/each}
  </div>
</div>

<style>
  .hooks-viewer { padding: 12px; }
  .hook-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px; border-bottom: 1px solid var(--border-color, #333);
  }
  .hook-type {
    padding: 2px 6px; border-radius: 4px; font-size: 10px; text-transform: uppercase;
    background: #444; color: white; margin-right: 8px;
  }
  .hook-type.pre { background: #f59e0b; }
  .hook-type.post { background: #22c55e; }
  .hook-event { color: var(--text-secondary, #888); font-size: 12px; margin-left: 8px; }
  .toggle { position: relative; width: 40px; height: 20px; }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .slider {
    position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0;
    background: #444; border-radius: 20px; transition: 0.3s;
  }
  .slider:before {
    position: absolute; content: ""; height: 16px; width: 16px; left: 2px; bottom: 2px;
    background: white; border-radius: 50%; transition: 0.3s;
  }
  .toggle input:checked + .slider { background: var(--accent-color, #4a9eff); }
  .toggle input:checked + .slider:before { transform: translateX(20px); }
</style>
