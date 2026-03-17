<script lang="ts">
  import { Server, Clock, Award } from 'lucide-svelte';



  interface Props {
    auctionQueue: Array<{
    id: string;
    timestamp: Date;
    participants: string[];
    winner: string;
    winningScore: number;
    status: string;
  }>;
    bots: Array<{
    id: string;
    name: string;
    symbol: string;
    status: 'idle' | 'ready' | 'paused' | 'quarantined';
    signalStrength: number;
    conditions: string[];
    score: number;
    lastSignal: Date | null;
  }>;
    getScoreColor: (score: number) => string;
    getStatusColor: (status: string) => string;
    timeAgo: (date: Date | null) => string;
  }

  let {
    auctionQueue,
    bots,
    getScoreColor,
    getStatusColor,
    timeAgo
  }: Props = $props();
</script>

<div class="auction-section">
  <div class="section-header">
    <h3>Live Auctions</h3>
    <span class="count">{auctionQueue.length} auctions</span>
  </div>

  <div class="auction-list">
    {#each auctionQueue as auction}
      <div class="auction-card">
        <div class="auction-header">
          <div class="auction-time">
            <Clock size={12} />
            <span>{timeAgo(auction.timestamp)}</span>
          </div>
          <span class="auction-status {auction.status}">{auction.status}</span>
        </div>

        <div class="auction-participants">
          <span class="participants-label">Participants</span>
          <div class="participants-list">
            {#each auction.participants as participant}
              <div class="participant-badge" class:winner={participant === auction.winner}>
                <span>{participant}</span>
                {#if participant === auction.winner}
                  <Award size={12} />
                {/if}
              </div>
            {/each}
          </div>
        </div>

        <div class="auction-result">
          <span class="result-label">Winner</span>
          <div class="winner-display">
            <span class="winner-name">{auction.winner}</span>
            <span class="winner-score" style="color: {getScoreColor(auction.winningScore)}">
              Score: {auction.winningScore.toFixed(1)}
            </span>
          </div>
        </div>
      </div>
    {/each}

    {#if auctionQueue.length === 0}
      <div class="empty-state">
        <Server size={32} />
        <p>No auctions yet. Click "Run Auction" to start.</p>
      </div>
    {/if}
  </div>
</div>

<div class="signals-section">
  <div class="section-header">
    <h3>Bot Signals</h3>
    <span class="count">{bots.filter(b => b.status === 'ready').length} ready</span>
  </div>

  <div class="bots-grid">
    {#each bots as bot}
      <div class="bot-card" class:status-ready={bot.status === 'ready'} class:status-idle={bot.status === 'idle'}>
        <div class="bot-status" style="background: {getStatusColor(bot.status)}"></div>
        <div class="bot-header">
          <span class="bot-name">{bot.name}</span>
          <span class="bot-symbol">{bot.symbol}</span>
        </div>

        {#if bot.status === 'ready'}
          <div class="bot-signal">
            <span class="signal-strength" style="color: {getScoreColor(bot.score)}">
              {bot.score.toFixed(1)}
            </span>
            <span class="signal-label">Signal Strength</span>
          </div>

          <div class="bot-conditions">
            {#each bot.conditions as condition}
              <span class="condition-tag">{condition}</span>
            {/each}
          </div>

          <div class="bot-footer">
            <span class="last-signal">{timeAgo(bot.lastSignal)}</span>
            <span class="bot-status-text">{bot.status}</span>
          </div>
        {:else}
          <div class="bot-idle">
            <span class="idle-text">{bot.status}</span>
          </div>
        {/if}
      </div>
    {/each}
  </div>
</div>

<style>
  .auction-section {
    margin-bottom: 24px;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .section-header h3 {
    margin: 0;
    font-size: 14px;
    color: var(--text-primary);
  }

  .count {
    padding: 2px 8px;
    background: var(--bg-tertiary);
    border-radius: 10px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .auction-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .auction-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
  }

  .auction-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .auction-time {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .auction-status {
    padding: 4px 10px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
  }

  .auction-status.completed {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .auction-participants {
    margin-bottom: 12px;
  }

  .participants-label {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }

  .participants-list {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .participant-badge {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 11px;
    color: var(--text-secondary);
  }

  .participant-badge.winner {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .auction-result {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
  }

  .result-label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .winner-display {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .winner-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .winner-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: var(--text-muted);
    text-align: center;
  }

  .empty-state p {
    margin-top: 12px;
    font-size: 13px;
  }

  .signals-section {
    margin-bottom: 24px;
  }

  .bots-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
  }

  .bot-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
    position: relative;
  }

  .bot-status {
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    border-radius: 10px 0 0 10px;
  }

  .bot-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .bot-name {
    font-weight: 500;
    color: var(--text-primary);
    font-size: 14px;
  }

  .bot-symbol {
    font-size: 11px;
    color: var(--text-muted);
    background: var(--bg-tertiary);
    padding: 2px 8px;
    border-radius: 4px;
  }

  .bot-signal {
    margin-bottom: 12px;
  }

  .signal-strength {
    font-size: 24px;
    font-weight: 600;
  }

  .signal-label {
    display: block;
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .bot-conditions {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 12px;
  }

  .condition-tag {
    padding: 2px 6px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-size: 10px;
    color: var(--text-secondary);
  }

  .bot-footer {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
  }

  .bot-idle {
    padding: 20px 0;
    text-align: center;
  }

  .idle-text {
    color: var(--text-muted);
    font-size: 12px;
    text-transform: capitalize;
  }
</style>
