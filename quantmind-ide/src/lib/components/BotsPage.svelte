<script lang="ts">
  import { createEventDispatcher } from "svelte";

  interface Props {
    bots?: Array<{
    id: string;
    name: string;
    state: string;
    symbol?: string;
  }>;
  }

  let { bots = [] }: Props = $props();

  const dispatch = createEventDispatcher();

  function getStatusColor(status: string): string {
    switch (status) {
      case "primal":
        return "#10b981";
      case "ready":
        return "#3b82f6";
      case "paused":
        return "#6b7280";
      case "quarantined":
        return "#ef4444";
      default:
        return "#6b7280";
    }
  }

  function handleTagChange(botId: string, newTag: string) {
    dispatch("tagChange", { botId, newTag });
  }

  const defaultBots = [
    { id: "ict-eu", name: "ICT_Scalper @EURUSD", state: "primal", symbol: "EURUSD" },
    { id: "ict-gb", name: "ICT_Scalper @GBPUSD", state: "primal", symbol: "GBPUSD" }
  ];
</script>

<div class="bots-page">
  <h2>Active Bots</h2>
  <div class="bot-cards">
    {#each bots.length > 0 ? bots : defaultBots as bot}
      <div class="bot-detail-card">
        <div
          class="bot-status-indicator"
          style="background: {getStatusColor(bot.state)}"
        ></div>
        <div class="bot-main">
          <h4>{bot.name}</h4>
          <p>{bot.symbol}</p>
        </div>
        <select
          class="tag-select"
          onchange={(e) => handleTagChange(bot.id, e.currentTarget.value)}
        >
          <option value="primal" selected={bot.state === "primal"}>Primal</option>
          <option value="ready" selected={bot.state === "ready"}>Ready</option>
          <option value="paused" selected={bot.state === "paused"}>Paused</option>
          <option value="quarantined" selected={bot.state === "quarantined"}>Quarantine</option>
        </select>
      </div>
    {/each}
  </div>
</div>

<style>
  .bots-page {
    padding: 20px;
  }

  .bots-page h2 {
    margin: 0 0 20px 0;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .bot-cards {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .bot-detail-card {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .bot-status-indicator {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }

  .bot-main {
    flex: 1;
  }

  .bot-main h4 {
    margin: 0;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .bot-main p {
    margin: 4px 0 0 0;
    font-size: 12px;
    color: var(--text-muted);
  }

  .tag-select {
    padding: 8px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 12px;
    cursor: pointer;
  }

  .tag-select:focus {
    outline: none;
    border-color: var(--accent-primary);
  }
</style>
