<script lang="ts">
  /**
   * BotStatusGrid - Grid Layout for Bot Status Cards
   *
   * Responsive grid displaying all active bot status cards.
   * Includes Close All button in header.
   * Shows degraded indicator when Contabo is unreachable.
   */
  import BotStatusCard from './BotStatusCard.svelte';
  import CloseAllModal from './CloseAllModal.svelte';
  import DegradedIndicator from './DegradedIndicator.svelte';
  import { activeBots, isLoading, type PositionInfo } from '$lib/stores/trading';
  import { isContaboDegraded } from '$lib/stores/node-health';
  import { Bot, XCircle } from 'lucide-svelte';

  // Responsive grid configuration
  const MIN_CARD_WIDTH = 280;
  const GAP = 16;

  // Close All modal state
  let showCloseAllModal = $state(false);

  // Mock positions for demo - in real app would come from API
  // This would be fetched based on open positions
  let positions: PositionInfo[] = $derived(
    $activeBots
      .filter((bot) => bot.open_positions > 0)
      .flatMap((bot) =>
        Array.from({ length: bot.open_positions }, (_, i) => ({
          ticket: 1000 + i,
          bot_id: bot.bot_id,
          symbol: bot.symbol,
          direction: 'buy' as const,
          lot: 0.01,
          current_pnl: bot.current_pnl / bot.open_positions
        }))
      )
  );

  function handleCloseAll() {
    if (positions.length > 0) {
      showCloseAllModal = true;
    }
  }

  function handleModalClose() {
    showCloseAllModal = false;
  }
</script>

<div class="grid-wrapper">
  <div class="grid-header">
    <div class="header-left">
      {#if $isContaboDegraded}
        <DegradedIndicator message="Agent data unavailable — retrying" />
      {/if}
    </div>
    <div class="header-right">
      {#if positions.length > 0}
        <button class="close-all-btn" onclick={handleCloseAll} title="Close all positions">
          <XCircle size={14} />
          <span>Close All ({positions.length})</span>
        </button>
      {/if}
    </div>
  </div>

  {#if $isLoading}
    <div class="skeleton-container">
      {#each Array(4) as _, i}
        <div class="skeleton-card">
          <div class="skeleton-line header-line"></div>
          <div class="skeleton-line pnl-line"></div>
          <div class="skeleton-line footer-line"></div>
        </div>
      {/each}
    </div>
  {:else if $activeBots.length === 0}
    <div class="empty-state">
      <Bot size={48} strokeWidth={1} />
      <p>No active bots</p>
      <span>Start a bot to see its status here</span>
    </div>
  {:else}
    {#each $activeBots as bot (bot.bot_id)}
      <BotStatusCard {bot} />
    {/each}
  {/if}
</div>

{#if showCloseAllModal}
  <CloseAllModal
    {positions}
    onClose={handleModalClose}
  />
{/if}

<style>
  .grid-wrapper {
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .grid-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0 16px;
  }

  .header-left,
  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .close-all-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 6px;
    color: #f59e0b;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .close-all-btn:hover {
    background: rgba(245, 158, 11, 0.25);
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.3);
  }

  .bot-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(var(--min-card-width), 1fr));
    gap: var(--gap);
    padding: 0 16px;
    width: 100%;
  }

  .skeleton-container {
    display: contents;
  }

  .skeleton-card {
    background: rgba(8, 13, 20, 0.35);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(0, 212, 255, 0.08);
    border-radius: 8px;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% {
      opacity: 0.4;
    }
    50% {
      opacity: 0.7;
    }
  }

  .skeleton-line {
    background: linear-gradient(90deg, rgba(0, 212, 255, 0.1) 25%, rgba(0, 212, 255, 0.2) 50%, rgba(0, 212, 255, 0.1) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 4px;
  }

  @keyframes shimmer {
    0% {
      background-position: 200% 0;
    }
    100% {
      background-position: -200% 0;
    }
  }

  .header-line {
    height: 20px;
    width: 70%;
  }

  .pnl-line {
    height: 24px;
    width: 50%;
  }

  .footer-line {
    height: 14px;
    width: 40%;
  }

  .empty-state {
    grid-column: 1 / -1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 48px;
    color: #555;
    text-align: center;
  }

  .empty-state p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 14px;
    margin-top: 16px;
    color: #888;
  }

  .empty-state span {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #555;
    margin-top: 8px;
  }
</style>
