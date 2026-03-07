<script lang="ts">
  import { AlertTriangle, Play, X } from "lucide-svelte";

  export let killZoneSettings: {
    enabled: boolean;
    autoPause: boolean;
  };
  export let currentKillZone: any = null;

  function resumeAnyway() {
    killZoneSettings.autoPause = false;
  }

  function closeBanner() {
    currentKillZone = null;
  }
</script>

{#if currentKillZone && killZoneSettings.enabled}
  <div class="kill-zone-banner">
    <div class="banner-content">
      <div class="banner-left">
        <AlertTriangle size={20} class="warning-icon" />
        <div class="banner-text">
          <h3>Kill Zone Active</h3>
          <p>High-impact news event approaching. Trading is paused.</p>
        </div>
      </div>
      <div class="banner-right">
        <button class="btn" on:click={resumeAnyway}>
          <Play size={14} />
          <span>Resume Anyway</span>
        </button>
        <button class="btn close" on:click={closeBanner}>
          <X size={14} />
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .kill-zone-banner {
    background: linear-gradient(90deg, #dc2626 0%, #b91c1c 100%);
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 16px;
  }

  .banner-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .banner-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .banner-text h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
  }

  .banner-text p {
    margin: 2px 0 0;
    font-size: 12px;
    opacity: 0.9;
  }

  .banner-right {
    display: flex;
    gap: 8px;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    border: none;
    border-radius: 4px;
    font-size: 12px;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.2);
    color: white;
    transition: background 0.2s;
  }

  .btn:hover {
    background: rgba(255, 255, 255, 0.3);
  }

  .btn.close {
    padding: 6px;
  }
</style>
