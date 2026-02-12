<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { Clock, Globe, TrendingUp, Sun, Moon, ArrowRight } from 'lucide-svelte';

  // Market sessions with their timezone offsets and open hours (UTC)
  interface MarketSession {
    name: string;
    timezone: string;
    openHour: number; // UTC hour when market opens
    closeHour: number; // UTC hour when market closes
    icon: any;
    color: string;
  }

  const markets: MarketSession[] = [
    { name: 'Tokyo', timezone: 'Asia/Tokyo', openHour: 0, closeHour: 6, icon: Sun, color: '#f59e0b' },
    { name: 'London', timezone: 'Europe/London', openHour: 8, closeHour: 16, icon: Globe, color: '#10b981' },
    { name: 'New York', timezone: 'America/New_York', openHour: 14, closeHour: 21, icon: TrendingUp, color: '#3b82f6' }
  ];

  let currentTime = new Date();
  let interval: number;

  // Calculate countdown to next market open
  function getTimeToNextOpen(): { hours: number; minutes: number; seconds: number; market: string | null } {
    const now = new Date();
    const currentHour = now.getUTCHours();
    const currentMinute = now.getUTCMinutes();
    const currentSecond = now.getUTCSeconds();
    const currentTotalSeconds = currentHour * 3600 + currentMinute * 60 + currentSecond;

    for (const market of markets) {
      if (currentTotalSeconds < market.openHour * 3600) {
        const secondsUntil = market.openHour * 3600 - currentTotalSeconds;
        return {
          hours: Math.floor(secondsUntil / 3600),
          minutes: Math.floor((secondsUntil % 3600) / 60),
          seconds: Math.floor(secondsUntil % 60),
          market: market.name
        };
      }
    }

    // Return next day's first market
    const firstMarket = markets[0];
    const secondsUntil = (24 * 3600 - currentTotalSeconds) + firstMarket.openHour * 3600;
    return {
      hours: Math.floor(secondsUntil / 3600),
      minutes: Math.floor((secondsUntil % 3600) / 60),
      seconds: Math.floor(secondsUntil % 60),
      market: firstMarket.name
    };
  }

  // Get status of a market
  function getMarketStatus(market: MarketSession): 'open' | 'closed' | 'soon' {
    const now = new Date();
    const currentHour = now.getUTCHours();

    if (currentHour >= market.openHour && currentHour < market.closeHour) {
      return 'open';
    }

    // Soon: within 1 hour of opening
    if (currentHour === market.openHour - 1 || (market.openHour === 0 && currentHour === 23)) {
      return 'soon';
    }

    return 'closed';
  }

  // Format time in specific timezone
  function formatTimeInZone(timezone: string): string {
    try {
      return currentTime.toLocaleTimeString('en-US', {
        timeZone: timezone,
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
      });
    } catch {
      return '--:--';
    }
  }

  onMount(() => {
    interval = window.setInterval(() => {
      currentTime = new Date();
    }, 1000);
  });

  onDestroy(() => {
    if (interval) clearInterval(interval);
  });

  $: timeToNextOpen = getTimeToNextOpen();
</script>

<div class="market-clock">
  <div class="clock-header">
    <Clock size={14} />
    <span>Market Hours</span>
  </div>

  {#if timeToNextOpen.market}
    <div class="countdown">
      <span class="countdown-label">Next: {timeToNextOpen.market}</span>
      <span class="countdown-time">
        {String(timeToNextOpen.hours).padStart(2, '0')}:
        {String(timeToNextOpen.minutes).padStart(2, '0')}:
        {String(timeToNextOpen.seconds).padStart(2, '0')}
      </span>
    </div>
  {/if}

  <div class="markets-grid">
    {#each markets as market}
      {@const status = getMarketStatus(market)}
      {@const localTime = formatTimeInZone(market.timezone)}
      <div class="market-item" class:open={status === 'open'} class:soon={status === 'soon'}>
        <div class="market-icon" style="color: {market.color}">
          <svelte:component this={market.icon} size={14} />
        </div>
        <div class="market-info">
          <span class="market-name">{market.name}</span>
          <span class="market-time">{localTime}</span>
        </div>
        <div class="market-status" class:open={status === 'open'} class:soon={status === 'soon'}>
          {#if status === 'open'}
            <span class="status-dot"></span>
            <span>Open</span>
          {:else if status === 'soon'}
            <ArrowRight size={10} />
            <span>Soon</span>
          {:else}
            <span>Closed</span>
          {/if}
        </div>
      </div>
    {/each}
  </div>

  <div class="market-hours-legend">
    <div class="legend-item">
      <span class="legend-dot" style="background: #f59e0b"></span>
      <span>Tokyo: 00:00-06:00 UTC</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background: #10b981"></span>
      <span>London: 08:00-16:00 UTC</span>
    </div>
    <div class="legend-item">
      <span class="legend-dot" style="background: #3b82f6"></span>
      <span>New York: 14:00-21:00 UTC</span>
    </div>
  </div>
</div>

<style>
  .market-clock {
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .clock-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .countdown {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 10px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 11px;
  }

  .countdown-label {
    color: var(--text-muted);
  }

  .countdown-time {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: var(--accent-primary);
  }

  .markets-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .market-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    border-left: 3px solid transparent;
  }

  .market-item.open {
    border-left-color: #10b981;
    background: rgba(16, 185, 129, 0.1);
  }

  .market-item.soon {
    border-left-color: #f59e0b;
    background: rgba(245, 158, 11, 0.1);
  }

  .market-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  .market-info {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .market-name {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .market-time {
    font-size: 10px;
    color: var(--text-muted);
  }

  .market-status {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-muted);
    background: rgba(107, 114, 128, 0.2);
  }

  .market-status.open {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  .market-status.soon {
    background: rgba(245, 158, 11, 0.2);
    color: #f59e0b;
  }

  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .market-hours-legend {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding-top: 8px;
    border-top: 1px solid var(--border-subtle);
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 10px;
    color: var(--text-muted);
  }

  .legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
  }
</style>
