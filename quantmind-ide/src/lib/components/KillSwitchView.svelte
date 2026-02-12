<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  // Simple toast notification system
  let toastMessage = '';
  let toastVisible = false;
  let toastType: 'success' | 'error' = 'success';

  function showToast(message: string, type: 'success' | 'error' = 'success') {
    toastMessage = message;
    toastType = type;
    toastVisible = true;
    setTimeout(() => toastVisible = false, 3000);
  }

  // Simple confirmation dialog
  let confirmDialog = {
    open: false,
    title: '',
    message: '',
    onConfirm: () => {}
  };

  function showConfirmDialog(title: string, message: string, onConfirm: () => void) {
    confirmDialog = { open: true, title, message, onConfirm };
  }

  // Types for our data structures
  type KillSwitchStrategy = 'immediate' | 'breakeven' | 'trailing' | 'scale_out';
  type BotStatus = 'active' | 'suspended' | 'quarantine';
  type TriggerReason = 'manual' | 'circuit_breaker' | 'news' | 'drawdown';

  interface BotKillSwitchConfig {
    bot_id: string;
    bot_name: string;
    status: BotStatus;
    strategy: KillSwitchStrategy;
    today_pnl: number;
    loss_count: number;
    max_losses: number;
    positions_open: number;
  }

  interface KillSwitchEvent {
    id: string;
    timestamp: string;
    trigger_reason: TriggerReason;
    strategy_used: string;
    positions_closed: number;
    total_pnl_preserved: number;
  }

  interface KillZoneConfig {
    enabled: boolean;
    timeRanges: Array<{
      start: string; // HH:mm format
      end: string; // HH:mm format
      days: number[]; // 0-6 (Sunday-Saturday)
    }>;
    newsImpactThreshold: 'high' | 'medium' | 'low';
  }

  // Reactive state
  let selectedStrategy: KillSwitchStrategy = 'immediate';
  let showHistory = false;
  let pollingInterval: number | null = null;

  // Mock data - replace with actual API calls
  let bots: BotKillSwitchConfig[] = [
    {
      bot_id: '1',
      bot_name: 'EURUSD_M5_Trend',
      status: 'active',
      strategy: 'immediate',
      today_pnl: 1250.50,
      loss_count: 2,
      max_losses: 5,
      positions_open: 3
    },
    {
      bot_id: '2',
      bot_name: 'GBPUSD_H1_Breakout',
      status: 'active',
      strategy: 'breakeven',
      today_pnl: -890.25,
      loss_count: 4,
      max_losses: 5,
      positions_open: 2
    },
    {
      bot_id: '3',
      bot_name: 'USDJPY_M15_Scalp',
      status: 'suspended',
      strategy: 'trailing',
      today_pnl: -1450.00,
      loss_count: 5,
      max_losses: 5,
      positions_open: 0
    },
    {
      bot_id: '4',
      bot_name: 'XAUUSD_H4_Swing',
      status: 'quarantine',
      strategy: 'immediate',
      today_pnl: -2100.00,
      loss_count: 6,
      max_losses: 5,
      positions_open: 1
    }
  ];

  let killSwitchHistory: KillSwitchEvent[] = [
    {
      id: '1',
      timestamp: '2025-01-15T14:32:00Z',
      trigger_reason: 'circuit_breaker',
      strategy_used: 'immediate',
      positions_closed: 2,
      total_pnl_preserved: -450.00
    },
    {
      id: '2',
      timestamp: '2025-01-15T10:15:00Z',
      trigger_reason: 'manual',
      strategy_used: 'breakeven',
      positions_closed: 3,
      total_pnl_preserved: 890.50
    },
    {
      id: '3',
      timestamp: '2025-01-14T16:45:00Z',
      trigger_reason: 'news',
      strategy_used: 'trailing',
      positions_closed: 1,
      total_pnl_preserved: 320.00
    }
  ];

  let killZoneConfig: KillZoneConfig = {
    enabled: true,
    timeRanges: [
      {
        start: '08:30',
        end: '10:00',
        days: [1, 2, 3, 4, 5] // Weekdays
      },
      {
        start: '13:30',
        end: '15:00',
        days: [1, 2, 3, 4, 5] // US market open
      }
    ],
    newsImpactThreshold: 'high'
  };

  // Computed metrics
  $: totalPositions = bots.reduce((sum, bot) => sum + bot.positions_open, 0);
  $: totalDailyPnL = bots.reduce((sum, bot) => sum + bot.today_pnl, 0);
  $: activeBots = bots.filter(b => b.status === 'active').length;
  $: circuitBreakerTripped = bots.some(b => b.loss_count >= b.max_losses);
  $: dailyDrawdown = Math.min(0, totalDailyPnL);
  $: dailyDrawdownPercent = dailyDrawdown < 0 ? Math.abs(dailyDrawdown) / 10000 * 100 : 0; // Assume 10k account

  // Strategy display names and descriptions
  const strategies = {
    immediate: {
      name: 'Immediate Exit',
      description: 'Close all positions immediately at market price',
      icon: '⚡',
      color: 'text-red-500'
    },
    breakeven: {
      name: 'Breakeven Exit',
      description: 'Move stops to breakeven, let profits run',
      icon: '🛡️',
      color: 'text-amber-500'
    },
    trailing: {
      name: 'Trailing Stop',
      description: 'Activate trailing stops on all positions',
      icon: '📈',
      color: 'text-blue-500'
    },
    scale_out: {
      name: 'Scale-Out',
      description: 'Partially close positions incrementally',
      icon: '📊',
      color: 'text-green-500'
    }
  };

  // Actions
  async function handleEmergencyStop() {
    showConfirmDialog(
      '🚨 EMERGENCY STOP ALL',
      `This will immediately activate the kill switch on ALL active bots!\n\nAffected Positions: ${totalPositions}\nActive Bots: ${activeBots}\nStrategy: ${strategies[selectedStrategy].name}\n\n${strategies[selectedStrategy].description}\n\nThis action cannot be undone. Please confirm you understand the consequences.`,
      async () => {
        await executeKillSwitch('all', selectedStrategy);
        showToast('Emergency stop activated for all bots');
      }
    );
  }

  async function executeKillSwitch(botId: string | 'all', strategy: KillSwitchStrategy) {
    // API call to execute kill switch
    // POST /api/router/kill-switch
    const payload = {
      bot_id: botId,
      strategy: strategy,
      trigger_reason: 'manual' as TriggerReason
    };

    console.log('Executing kill switch:', payload);

    // Update local state
    if (botId === 'all') {
      bots = bots.map(bot => ({
        ...bot,
        status: 'suspended' as BotStatus,
        positions_open: 0
      }));

      // Add to history
      const newEvent: KillSwitchEvent = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        trigger_reason: 'manual',
        strategy_used: strategy,
        positions_closed: totalPositions,
        total_pnl_preserved: totalDailyPnL
      };
      killSwitchHistory = [newEvent, ...killSwitchHistory];
    } else {
      const bot = bots.find(b => b.bot_id === botId);
      if (bot) {
        bot.status = 'suspended';
        bot.positions_open = 0;
      }
    }
  }

  async function suspendBot(botId: string) {
    const bot = bots.find(b => b.bot_id === botId);
    if (bot) {
      bot.status = 'suspended';
      showToast(`Suspended ${bot.bot_name}`);
    }
  }

  async function killBot(botId: string) {
    const bot = bots.find(b => b.bot_id === botId);
    if (!bot) return;

    showConfirmDialog(
      `Kill ${bot.bot_name}?`,
      `This will close all ${bot.positions_open} positions for ${bot.bot_name} using ${strategies[bot.strategy].name} strategy.`,
      async () => {
        await executeKillSwitch(botId, bot.strategy);
        showToast(`Kill switch activated for ${bot.bot_name}`);
      }
    );
  }

  function updateBotStrategy(botId: string, strategy: KillSwitchStrategy) {
    const bot = bots.find(b => b.bot_id === botId);
    if (bot) {
      bot.strategy = strategy;
      showToast(`Updated ${bot.bot_name} strategy to ${strategies[strategy].name}`);
    }
  }

  function toggleKillZone() {
    killZoneConfig.enabled = !killZoneConfig.enabled;
    showToast(`Kill zone ${killZoneConfig.enabled ? 'enabled' : 'disabled'}`);
  }

  function formatCurrency(value: number): string {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD'
    }).format(value);
  }

  function formatDate(isoString: string): string {
    return new Date(isoString).toLocaleString();
  }

  function getStatusColor(status: BotStatus): string {
    switch (status) {
      case 'active': return 'text-green-500';
      case 'suspended': return 'text-amber-500';
      case 'quarantine': return 'text-red-500';
      default: return 'text-gray-500';
    }
  }

  function getTriggerReasonIcon(reason: TriggerReason): string {
    switch (reason) {
      case 'manual': return '👤';
      case 'circuit_breaker': return '⚡';
      case 'news': return '📰';
      case 'drawdown': return '📉';
      default: return '❓';
    }
  }

  // Lifecycle
  onMount(() => {
    // Poll for updates every 5 seconds
    pollingInterval = window.setInterval(() => {
      // Refresh bot data from API
      // fetchBotData();
    }, 5000);
  });

  onDestroy(() => {
    if (pollingInterval) {
      clearInterval(pollingInterval);
    }
  });
</script>

<div class="kill-switch-view space-y-6">
  <!-- Header -->
  <div class="flex items-center justify-between">
    <div>
      <h1 class="text-2xl font-bold text-red-600 dark:text-red-400 flex items-center gap-2">
        <span>🚨</span>
        Kill Switch Control
      </h1>
      <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
        Emergency trading halt and risk management controls
      </p>
    </div>
    <button
      on:click={() => (showHistory = !showHistory)}
      class="px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
    >
      {showHistory ? 'Hide' : 'Show'} History
    </button>
  </div>

  <!-- Emergency Controls -->
  <div class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
    <div class="flex items-center justify-between">
      <div class="space-y-2">
        <h2 class="text-xl font-bold text-red-700 dark:text-red-300">
          Emergency Stop All
        </h2>
        <p class="text-sm text-red-600 dark:text-red-400">
          Immediately halt all trading and close positions
        </p>
        <div class="flex gap-4 text-sm">
          <div>
            <span class="font-semibold">Active Bots:</span>
            <span class="ml-1">{activeBots}</span>
          </div>
          <div>
            <span class="font-semibold">Open Positions:</span>
            <span class="ml-1">{totalPositions}</span>
          </div>
          <div>
            <span class="font-semibold">Exposure:</span>
            <span class="ml-1">{formatCurrency(totalDailyPnL)}</span>
          </div>
        </div>
      </div>
      <button
        on:click={handleEmergencyStop}
        class="px-6 py-3 text-lg font-bold bg-red-600 hover:bg-red-700 text-white rounded-lg animate-pulse-slow transition-colors"
      >
        🛑 EMERGENCY STOP ALL
      </button>
    </div>

    <!-- Strategy Selection -->
    <div class="mt-6">
      <h3 class="text-sm font-semibold text-red-700 dark:text-red-300 mb-3">
        Default Kill Switch Strategy:
      </h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {#each Object.entries(strategies) as [key, strategy]}
          <label class="cursor-pointer">
            <input
              type="radio"
              bind:group={selectedStrategy}
              value={key}
              class="hidden"
            />
            <div class="border-2 rounded-lg p-4 transition-all hover:border-red-400"
                 class:border-red-500={selectedStrategy === key}
                 class:bg-red-100={selectedStrategy === key}
                 class:dark:bg-red-900={selectedStrategy === key}
                 class:border-gray-300={selectedStrategy !== key}
                 class:dark:border-gray-700={selectedStrategy !== key}>
              <div class="flex items-center gap-2 mb-2">
                <span class="text-2xl">{strategy.icon}</span>
                <span class="font-semibold {strategy.color}">{strategy.name}</span>
              </div>
              <p class="text-xs text-gray-600 dark:text-gray-400">
                {strategy.description}
              </p>
            </div>
          </label>
        {/each}
      </div>
    </div>
  </div>

  <!-- Bleeding Metrics -->
  <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
    <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
      <span>📊</span>
      Bleeding Metrics
    </h2>
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <!-- Daily P&L -->
      <div>
        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Today's P&L
        </h3>
        <p
          class="text-3xl font-bold"
          class:text-green-600={totalDailyPnL >= 0}
          class:text-red-600={totalDailyPnL < 0}
        >
          {formatCurrency(totalDailyPnL)}
        </p>
        {#if totalDailyPnL < 0}
          <p class="text-sm text-red-600 mt-1">
            ⚠️ Down {Math.abs(dailyDrawdownPercent).toFixed(1)}% today
          </p>
        {/if}
      </div>

      <!-- Drawdown Gauge -->
      <div>
        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Daily Drawdown
        </h3>
        <div class="space-y-2">
          <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div
              class="h-full bg-gradient-to-r from-green-500 via-amber-500 to-red-500 transition-all"
              style="width: {Math.min(dailyDrawdownPercent, 100)}%"
            ></div>
          </div>
          <div class="flex justify-between text-xs text-gray-600 dark:text-gray-400">
            <span>0%</span>
            <span>{dailyDrawdownPercent.toFixed(1)}%</span>
            <span>Max: 10%</span>
          </div>
        </div>
      </div>

      <!-- Circuit Breaker Status -->
      <div>
        <h3 class="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
          Circuit Breaker
        </h3>
        <div class="space-y-2">
          {#if circuitBreakerTripped}
            <div class="flex items-center gap-2 text-red-600">
              <span class="animate-pulse">🔴</span>
              <span class="font-semibold">TRIPPED</span>
            </div>
            <p class="text-xs text-gray-600 dark:text-gray-400">
              One or more bots hit max loss limit
            </p>
          {:else}
            <div class="flex items-center gap-2 text-green-600">
              <span>🟢</span>
              <span class="font-semibold">Active</span>
            </div>
            <p class="text-xs text-gray-600 dark:text-gray-400">
              Monitoring all bots
            </p>
          {/if}
        </div>
      </div>
    </div>

    <!-- House Money Effect -->
    <div class="mt-6 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
      <div class="flex items-center gap-3">
        <span class="text-2xl">💰</span>
        <div>
          <h3 class="font-semibold text-amber-800 dark:text-amber-200">
            House Money Effect
          </h3>
          <p class="text-sm text-amber-700 dark:text-amber-300">
            {#if totalDailyPnL > 0}
              You're up {formatCurrency(totalDailyPnL)} today - Consider tightening stops to protect profits
            {:else}
              You're down {formatCurrency(Math.abs(totalDailyPnL))} - Reduce position sizes until recovery
            {/if}
          </p>
        </div>
      </div>
    </div>
  </div>

  <!-- Per-Bot Kill Switch Configuration -->
  <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
    <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
      <span>🤖</span>
      Per-Bot Configuration
    </h2>
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-gray-200 dark:border-gray-700">
            <th class="text-left py-3 px-4 font-semibold">Bot Name</th>
            <th class="text-left py-3 px-4 font-semibold">Status</th>
            <th class="text-left py-3 px-4 font-semibold">Strategy</th>
            <th class="text-right py-3 px-4 font-semibold">Today's P&L</th>
            <th class="text-center py-3 px-4 font-semibold">Losses</th>
            <th class="text-center py-3 px-4 font-semibold">Positions</th>
            <th class="text-center py-3 px-4 font-semibold">Actions</th>
          </tr>
        </thead>
        <tbody>
          {#each bots as bot (bot.bot_id)}
            <tr class="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-750">
              <td class="py-3 px-4">
                <div class="font-medium">{bot.bot_name}</div>
                <div class="text-xs text-gray-500 dark:text-gray-400">ID: {bot.bot_id}</div>
              </td>
              <td class="py-3 px-4">
                <div class="flex items-center gap-2 {getStatusColor(bot.status)}">
                  <span
                    class="w-2 h-2 rounded-full animate-pulse"
                    class:bg-green-500={bot.status === 'active'}
                    class:bg-amber-500={bot.status === 'suspended'}
                    class:bg-red-500={bot.status === 'quarantine'}
                  ></span>
                  <span class="capitalize">{bot.status}</span>
                </div>
              </td>
              <td class="py-3 px-4">
                <select
                  value={bot.strategy}
                  on:change={(e) => updateBotStrategy(bot.bot_id, e.currentTarget.value)}
                  class="px-3 py-2 rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-700 text-sm"
                >
                  {#each Object.entries(strategies) as [key, strategy]}
                    <option value={key}>{strategy.name}</option>
                  {/each}
                </select>
              </td>
              <td class="py-3 px-4 text-right font-mono"
                  class:text-green-600={bot.today_pnl >= 0}
                  class:text-red-600={bot.today_pnl < 0}>
                {formatCurrency(bot.today_pnl)}
              </td>
              <td class="py-3 px-4 text-center">
                <div class="inline-flex items-center gap-1 px-2 py-1 rounded"
                     class:bg-red-100={bot.loss_count >= bot.max_losses}
                     class:dark:bg-red-900={bot.loss_count >= bot.max_losses}
                     class:bg-amber-100={bot.loss_count < bot.max_losses && bot.loss_count > 0}
                     class:dark:bg-amber-900={bot.loss_count < bot.max_losses && bot.loss_count > 0}
                     class:bg-green-100={bot.loss_count === 0}
                     class:dark:bg-green-900={bot.loss_count === 0}>
                  <span>{bot.loss_count}</span>
                  <span class="text-gray-400">/</span>
                  <span>{bot.max_losses}</span>
                </div>
              </td>
              <td class="py-3 px-4 text-center">
                <span class="font-mono">{bot.positions_open}</span>
              </td>
              <td class="py-3 px-4">
                <div class="flex items-center gap-2">
                  {#if bot.status === 'active'}
                    <button
                      on:click={() => suspendBot(bot.bot_id)}
                      class="px-2 py-1 rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-sm"
                      title="Suspend bot"
                    >
                      ⏸️
                    </button>
                    <button
                      on:click={() => killBot(bot.bot_id)}
                      class="px-2 py-1 rounded bg-red-600 hover:bg-red-700 text-white text-sm"
                      title="Kill bot"
                    >
                      🛑
                    </button>
                  {:else}
                    <button
                      on:click={() => { bot.status = 'active'; showToast(`Resumed ${bot.bot_name}`); }}
                      class="px-2 py-1 rounded bg-green-600 hover:bg-green-700 text-white text-sm"
                      title="Resume bot"
                    >
                      ▶️
                    </button>
                  {/if}
                </div>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  </div>

  <!-- Kill Zone Configuration -->
  <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-bold flex items-center gap-2">
        <span>🕐</span>
        Kill Zone Configuration
      </h2>
      <label class="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          bind:checked={killZoneConfig.enabled}
          on:change={toggleKillZone}
          class="w-4 h-4 rounded border-gray-300 dark:border-gray-700"
        />
        <span class="text-sm font-medium">Enable Auto-Pause</span>
      </label>
    </div>

    <div class="space-y-4">
      {#if killZoneConfig.enabled}
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          {#each killZoneConfig.timeRanges as range, index}
            <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
              <h3 class="font-semibold mb-3">Time Range {index + 1}</h3>
              <div class="space-y-3">
                <div class="flex items-center gap-3">
                  <label class="text-sm w-16">Start:</label>
                  <input
                    type="time"
                    bind:value={range.start}
                    class="px-3 py-2 rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-700 text-sm w-full"
                  />
                </div>
                <div class="flex items-center gap-3">
                  <label class="text-sm w-16">End:</label>
                  <input
                    type="time"
                    bind:value={range.end}
                    class="px-3 py-2 rounded border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-700 text-sm w-full"
                  />
                </div>
                <div>
                  <label class="text-sm block mb-2">Active Days:</label>
                  <div class="flex flex-wrap gap-2">
                    {#each ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'] as day, dayIndex}
                      <label class="flex items-center gap-1 text-xs cursor-pointer">
                        <input
                          type="checkbox"
                          bind:group={range.days}
                          value={dayIndex}
                          class="w-3 h-3 rounded border-gray-300 dark:border-gray-700"
                        />
                        <span>{day}</span>
                      </label>
                    {/each}
                  </div>
                </div>
              </div>
            </div>
          {/each}
        </div>

        <div class="border-t border-gray-200 dark:border-gray-700 pt-4">
          <h3 class="font-semibold mb-3">News Impact Threshold</h3>
          <div class="flex flex-wrap gap-3">
            {#each ['high', 'medium', 'low'] as level}
              <label class="cursor-pointer">
                <input
                  type="radio"
                  bind:group={killZoneConfig.newsImpactThreshold}
                  value={level}
                  class="hidden"
                />
                <div
                  class="px-4 py-2 rounded-lg border-2 transition-all"
                  class:border-blue-500={killZoneConfig.newsImpactThreshold === level}
                  class:bg-blue-50={killZoneConfig.newsImpactThreshold === level}
                  class:dark:bg-blue-900={killZoneConfig.newsImpactThreshold === level}
                  class:border-gray-300={killZoneConfig.newsImpactThreshold !== level}
                  class:dark:border-gray-700={killZoneConfig.newsImpactThreshold !== level}>
                  <span class="capitalize font-medium">{level}</span>
                  <span class="text-xs text-gray-500 ml-2">+ impact</span>
                </div>
              </label>
            {/each}
          </div>
          <p class="text-xs text-gray-500 mt-2">
            Automatically pause trading before {killZoneConfig.newsImpactThreshold} impact news events
          </p>
        </div>
      {:else}
        <div class="text-center py-8 text-gray-500 dark:text-gray-400">
          <p class="text-lg mb-2">Kill zone is disabled</p>
          <p class="text-sm">Enable auto-pause to configure time ranges</p>
        </div>
      {/if}
    </div>
  </div>

  <!-- Kill Switch History -->
  {#if showHistory}
    <div class="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-6">
      <h2 class="text-lg font-bold mb-4 flex items-center gap-2">
        <span>📜</span>
        Kill Switch History
      </h2>
      <div class="space-y-3">
        {#each killSwitchHistory as event (event.id)}
          <div class="border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors">
            <div class="flex items-start justify-between">
              <div class="flex items-start gap-3">
                <span class="text-2xl">{getTriggerReasonIcon(event.trigger_reason)}</span>
                <div>
                  <h3 class="font-semibold capitalize">{event.trigger_reason.replace('_', ' ')} Trigger</h3>
                  <p class="text-sm text-gray-600 dark:text-gray-400 mt-1">
                    Strategy: {event.strategy_used}
                  </p>
                  <div class="flex gap-4 mt-2 text-sm">
                    <span>
                      <span class="text-gray-500">Positions Closed:</span>
                      <span class="font-medium ml-1">{event.positions_closed}</span>
                    </span>
                    <span>
                      <span class="text-gray-500">P&L Preserved:</span>
                      <span class="font-medium ml-1 {event.total_pnl_preserved >= 0 ? 'text-green-600' : 'text-red-600'}">
                        {formatCurrency(event.total_pnl_preserved)}
                      </span>
                    </span>
                  </div>
                </div>
              </div>
              <div class="text-right text-sm text-gray-500">
                <div>{formatDate(event.timestamp)}</div>
              </div>
            </div>
          </div>
        {/each}

        {#if killSwitchHistory.length === 0}
          <div class="text-center py-8 text-gray-500 dark:text-gray-400">
            <p class="text-lg mb-2">No kill switch events yet</p>
            <p class="text-sm">History will appear here after triggers</p>
          </div>
        {/if}
      </div>
    </div>
  {/if}

  <!-- Toast Notification -->
  {#if toastVisible}
    <div class="toast" class:success={toastType === 'success'} class:error={toastType === 'error'}>
      {toastMessage}
    </div>
  {/if}

  <!-- Confirmation Dialog -->
  {#if confirmDialog.open}
    <div
      class="dialog-overlay"
      on:click={() => confirmDialog.open = false}
      on:keydown={(e) => e.key === 'Escape' && (confirmDialog.open = false)}
      role="dialog"
      aria-modal="true"
      aria-labelledby="dialog-title"
    >
      <div
        class="dialog"
        on:click|stopPropagation
        on:keydown={(e) => e.key === 'Escape' && (confirmDialog.open = false)}
        role="document"
        tabindex="-1"
      >
        <h3 id="dialog-title">{confirmDialog.title}</h3>
        <p>{confirmDialog.message}</p>
        <div class="dialog-actions">
          <button
            on:click={() => confirmDialog.open = false}
            on:keydown={(e) => (e.key === 'Enter' || e.key === ' ') && (confirmDialog.open = false)}
          >
            Cancel
          </button>
          <button
            class="danger"
            on:click={() => { confirmDialog.onConfirm(); confirmDialog.open = false; }}
            on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') { confirmDialog.onConfirm(); confirmDialog.open = false; }}}
          >
            Confirm
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .kill-switch-view {
    animation: fadeIn 0.3s ease-in;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  @keyframes pulse-slow {
    0%, 100% {
      opacity: 1;
      transform: scale(1);
    }
    50% {
      opacity: 0.9;
      transform: scale(1.02);
    }
  }

  .animate-pulse-slow {
    animation: pulse-slow 2s ease-in-out infinite;
  }

  /* Toast Styles */
  .toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 12px 20px;
    background: var(--bg-secondary, #1f2937);
    border: 1px solid var(--border-subtle, #374151);
    border-radius: 8px;
    color: var(--text-primary, #f9fafb);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    z-index: 1000;
    animation: slideIn 0.3s ease;
  }

  .toast.success {
    border-left: 4px solid #10b981;
  }

  .toast.error {
    border-left: 4px solid #ef4444;
  }

  /* Dialog Styles */
  .dialog-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .dialog {
    background: var(--bg-secondary, #1f2937);
    border-radius: 12px;
    padding: 24px;
    max-width: 500px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    color: var(--text-primary, #f9fafb);
  }

  .dialog h3 {
    margin: 0 0 12px;
    color: var(--text-primary, #f9fafb);
  }

  .dialog p {
    color: var(--text-secondary, #d1d5db);
    margin-bottom: 20px;
    white-space: pre-line;
  }

  .dialog-actions {
    display: flex;
    gap: 12px;
    justify-content: flex-end;
  }

  .dialog button {
    padding: 8px 16px;
    border-radius: 6px;
    border: 1px solid var(--border-subtle, #374151);
    background: var(--bg-tertiary, #374151);
    color: var(--text-primary, #f9fafb);
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .dialog button:hover {
    background: var(--bg-hover, #4b5563);
  }

  .dialog button.danger {
    background: #ef4444;
    color: white;
    border-color: #ef4444;
  }

  .dialog button.danger:hover {
    background: #dc2626;
  }

  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
</style>
