<script lang="ts">
  import { Brain } from 'lucide-svelte';

  export let hmmTraining: {
    isTraining: boolean;
    progress: number;
    message: string;
    lastJob: { jobId: string; status: string; message: string } | null;
  };
</script>

{#if hmmTraining.isTraining || hmmTraining.lastJob}
  <div class="hmm-training-status" class:training={hmmTraining.isTraining}>
    <div class="training-info">
      <Brain size={16} />
      <span class="training-label">HMM Training:</span>
      <span class="training-message">{hmmTraining.message}</span>
      {#if hmmTraining.isTraining && hmmTraining.progress > 0}
        <span class="training-progress-text">{hmmTraining.progress.toFixed(0)}%</span>
      {/if}
    </div>
    {#if hmmTraining.isTraining}
      <div class="training-bar">
        <div class="training-bar-fill" style="width: {hmmTraining.progress}%"></div>
      </div>
    {/if}
  </div>
{/if}

<style>
  .hmm-training-status {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0 24px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .hmm-training-status.training {
    border-color: #7c3aed;
    background: #1e1b2e;
  }

  .training-info {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #e2e8f0;
  }

  .training-label {
    font-weight: 500;
    color: #94a3b8;
  }

  .training-message {
    flex: 1;
  }

  .training-progress-text {
    font-weight: 600;
    color: #a78bfa;
  }

  .training-bar {
    height: 4px;
    background: #334155;
    border-radius: 2px;
    overflow: hidden;
  }

  .training-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #7c3aed, #a78bfa);
    transition: width 0.3s ease;
  }
</style>
