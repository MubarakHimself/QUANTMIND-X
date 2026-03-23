<script lang="ts">
  import { Clock } from 'lucide-svelte';

  interface Props {
    canvasName: string;
    epicNumber: number;
    epicName: string;
  }

  let { canvasName, epicNumber, epicName }: Props = $props();
</script>

<div class="canvas-placeholder">
  <div class="canvas-header">
    <h2 class="canvas-title">{canvasName}</h2>
    <span class="epic-badge">Epic {epicNumber}</span>
  </div>

  <div class="canvas-content">
    <div class="coming-soon">
      <Clock size={48} strokeWidth={1.5} />
      <h3>Coming in Epic {epicNumber}</h3>
      <p>{epicName}</p>
    </div>
  </div>

  <div class="canvas-footer">
    <span class="status">This canvas is under development</span>
  </div>
</div>

<style>
  .canvas-placeholder {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 24px;
    background: var(--color-bg-base);
  }

  .canvas-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 32px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--color-border-subtle);
  }

  .canvas-title {
    font-size: 24px;
    font-weight: 600;
    color: var(--color-text-primary);
    margin: 0;
  }

  .epic-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    background: var(--color-accent-cyan);
    color: var(--text-inverse);
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
  }

  .canvas-content {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .coming-soon {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    color: var(--color-text-muted);
  }

  .coming-soon h3 {
    font-size: 20px;
    font-weight: 500;
    margin: 0;
    color: var(--color-text-secondary);
  }

  .coming-soon p {
    font-size: 14px;
    margin: 0;
    max-width: 300px;
    text-align: center;
  }

  .canvas-footer {
    padding-top: 16px;
    border-top: 1px solid var(--color-border-subtle);
  }

  .status {
    font-size: 12px;
    color: var(--color-text-muted);
  }
</style>
