<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import {
    FileText, Save, RefreshCw, Plus, X, ChevronRight, ChevronDown,
    Edit3, Eye, Code, TestTube, Settings as SettingsIcon, Sliders,
    Target, TrendingUp, AlertTriangle, CheckCircle, Clock,
    FolderOpen, Download, Upload as UploadIcon, Copy, Clipboard,
    Zap, Shield, DollarSign, Activity, BarChart3, Layers,
    ArrowLeft, ArrowRight, Maximize2, Minimize2, ToggleLeft, ToggleRight
  } from 'lucide-svelte';

  const dispatch = createEventDispatcher();

  // TRD state
  export let trdId: string | null = null;
  export let strategyName = 'ICT Scalper v2';
  export let readOnly = false;

  // TRD Data Structure
  let trdData = {
    // Header
    id: trdId || crypto.randomUUID(),
    name: strategyName,
    version: '1.0',
    status: 'draft' as 'draft' | 'review' | 'approved' | 'deployed',
    created: new Date().toISOString(),
    modified: new Date().toISOString(),

    // Strategy Overview
    overview: {
      description: 'London session scalper using ICT concepts',
      timeframe: 'M5',
      session: 'London',
      type: 'trend_following' as 'trend_following' | 'mean_reversion' | 'breakout' | 'scalping',
      edge: 0.52,
      expectancy: 1.2
    },

    // Entry Conditions (Vanilla vs Spiced)
    entry: {
      vanilla: {
        conditions: [
          {
            id: 'c1',
            name: 'FVG Setup',
            description: 'Fair Value Gap formed by 3-candle sequence',
            enabled: true,
            parameters: {
              gapSize: 5, // pips
              lookback: 50
            }
          },
          {
            id: 'c2',
            name: 'Order Block',
            description: 'Price returns to bearish order block',
            enabled: true,
            parameters: {
              lookback: 100,
              minSize: 20
            }
          },
          {
            id: 'c3',
            name: 'Session Filter',
            description: 'London session only (08:00-12:00 GMT)',
            enabled: true,
            parameters: {
              sessionStart: '08:00',
              sessionEnd: '12:00',
              timezone: 'GMT'
            }
          }
        ]
      },
      spiced: {
        // Enhanced with SharedAssets and Sentinel
        conditions: [
          {
            id: 's1',
            name: 'FVG Setup',
            description: 'Fair Value Gap with ATR_Filter confirmation',
            enabled: true,
            parameters: {
              gapSize: 5,
              lookback: 50,
              withAtrFilter: true,
              atrMultiplier: 1.5
            },
            sharedAsset: 'ATR_Filter.mqh'
          },
          {
            id: 's2',
            name: 'Order Block',
            description: 'Order block + RSI_Divergence confluence',
            enabled: true,
            parameters: {
              lookback: 100,
              minSize: 20,
              withRsiDivergence: true,
              rsiPeriod: 14
            },
            sharedAsset: 'RSI_Divergence.mqh'
          },
          {
            id: 's3',
            name: 'Regime Filter',
            description: 'Sentinel regime quality >= 0.7',
            enabled: true,
            parameters: {
              minRegimeQuality: 0.7,
              maxChaosScore: 30
            },
            sharedAsset: 'Sentinel'
          },
          {
            id: 's4',
            name: 'Session Filter',
            description: 'London session with spread filter',
            enabled: true,
            parameters: {
              sessionStart: '08:00',
              sessionEnd: '12:00',
              maxSpread: 3 // pips
            }
          }
        ]
      }
    },

    // Exit Management
    exit: {
      vanilla: {
        stopLoss: {
          type: 'fixed' as 'fixed' | 'atr' | 'swing',
          value: 10, // pips
          trail: false
        },
        takeProfit: {
          type: 'fixed' as 'fixed' | 'rr' as 'rr',
          value: 15, // pips
          ratio: 1.5 // RR:1
        }
      },
      spiced: {
        stopLoss: {
          type: 'atr',
          value: 1.5, // ATR multiplier
          trail: true,
          trailActivation: 1.0 // RR:1
        },
        takeProfit: {
          type: 'rr',
          ratio: 2.0 // 2:1
        }
      }
    },

    // Risk Management
    risk: {
      vanilla: {
        mode: 'fixed',
        lotSize: 0.01,
        maxPositions: 3
      },
      spiced: {
        mode: 'kelly',
        kellyConfig: {
          baseEdge: 0.52,
          baseOdds: 2.0,
          maxFraction: 0.02,
          regimeAdjustment: true,
          houseMoney: true
        },
        squadLimits: true,
        maxPositions: 4
      }
    },

    // Preferred Conditions (Router Integration)
    preferredConditions: {
      regime: {
        quality: { min: 0.7, max: 1.0 },
        chaos: { min: 0, max: 30 }
      },
      correlation: {
        maxCorrelatedPositions: 2,
        correlationThreshold: 0.7
      },
      houseMoney: {
        enabled: true,
        threshold: 0.5
      }
    },

    // Backtest Requirements
    backtest: {
      variants: ['vanilla', 'spiced', 'vanilla+kelly', 'spiced+kelly'],
      symbols: ['EURUSD', 'GBPUSD', 'USDJPY'],
      period: '6M',
      monteCarloRuns: 1000,
      walkForward: {
        enabled: true,
        inSample: 70, // %
        step: 1 // month
      }
    },

    // Shared Assets Dependencies
    sharedAssets: [
      {
        name: 'ATR_Filter.mqh',
        path: 'Include/QuantMind/Indicators/',
        version: '1.2'
      },
      {
        name: 'RSI_Divergence.mqh',
        path: 'Include/QuantMind/Indicators/',
        version: '1.0'
      },
      {
        name: 'RiskManager.mqh',
        path: 'Include/QuantMind/Risk/',
        version: '2.0'
      }
    ]
  };

  // View state
  let activeTab: 'vanilla' | 'spiced' | 'comparison' = 'comparison';
  let expandedSections: Record<string, boolean> = {
    overview: true,
    entry: true,
    exit: true,
    risk: true,
    preferred: true,
    backtest: false,
    assets: false
  };

  // Editing state
  let editingCondition: string | null = null;
  let tempCondition: any = null;

  // Load TRD data on mount
  onMount(async () => {
    if (trdId) {
      await loadTRD(trdId);
    }
  });

  async function loadTRD(id: string) {
    try {
      const res = await fetch(`http://localhost:8000/api/trd/${id}`);
      if (res.ok) {
        trdData = await res.json();
      }
    } catch (e) {
      console.error('Failed to load TRD:', e);
    }
  }

  async function saveTRD() {
    trdData.modified = new Date().toISOString();

    try {
      const url = trdId
        ? `http://localhost:8000/api/trd/${trdId}`
        : 'http://localhost:8000/api/trd';
      const method = trdId ? 'PUT' : 'POST';

      const res = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trdData)
      });

      if (res.ok) {
        dispatch('saved', { trd: trdData });
      }
    } catch (e) {
      console.error('Failed to save TRD:', e);
    }
  }

  function toggleSection(section: string) {
    expandedSections[section] = !expandedSections[section];
  }

  function startEditing(conditionId: string, variant: 'vanilla' | 'spiced') {
    const conditions = variant === 'vanilla'
      ? trdData.entry.vanilla.conditions
      : trdData.entry.spiced.conditions;
    const condition = conditions.find(c => c.id === conditionId);
    if (condition) {
      editingCondition = `${variant}-${conditionId}`;
      tempCondition = JSON.parse(JSON.stringify(condition));
    }
  }

  function saveCondition() {
    if (!tempCondition || !editingCondition) return;

    const [variant, conditionId] = editingCondition.split('-');
    const conditions = variant === 'vanilla'
      ? trdData.entry.vanilla.conditions
      : trdData.entry.spiced.conditions;

    const index = conditions.findIndex(c => c.id === conditionId);
    if (index >= 0) {
      conditions[index] = tempCondition;
    }

    editingCondition = null;
    tempCondition = null;
  }

  function cancelEdit() {
    editingCondition = null;
    tempCondition = null;
  }

  function toggleConditionEnabled(conditionId: string, variant: 'vanilla' | 'spiced') {
    const conditions = variant === 'vanilla'
      ? trdData.entry.vanilla.conditions
      : trdData.entry.spiced.conditions;
    const condition = conditions.find(c => c.id === conditionId);
    if (condition) {
      condition.enabled = !condition.enabled;
    }
  }

  function generateEA() {
    dispatch('generateEA', { trd: trdData });
  }

  function runBacktest() {
    dispatch('runBacktest', { trd: trdData });
  }

  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
  }

  function exportTRD() {
    const blob = new Blob([JSON.stringify(trdData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `TRD-${trdData.name.replace(/\s+/g, '_')}-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function getStatusColor(status: string) {
    const colors: Record<string, string> = {
      draft: '#6b7280',
      review: '#f59e0b',
      approved: '#3b82f6',
      deployed: '#10b981'
    };
    return colors[status] || '#6b7280';
  }

  function getSharedAssetIcon(asset: string) {
    if (asset.includes('ATR')) return Activity;
    if (asset.includes('RSI')) return TrendingUp;
    if (asset.includes('Risk')) return Shield;
    return Code;
  }
</script>

<div class="trd-editor">
  <!-- Header -->
  <div class="trd-header">
    <div class="header-left">
      <button class="icon-btn" on:click={() => dispatch('close')}>
        <ArrowLeft size={18} />
      </button>
      <FileText size={24} class="trd-icon" />
      <div class="header-info">
        <div class="trd-title">{trdData.name}</div>
        <div class="trd-meta">
          <span class="status-badge" style="background: {getStatusColor(trdData.status)}">
            {trdData.status}
          </span>
          <span>v{trdData.version}</span>
          <span>Modified: {new Date(trdData.modified).toLocaleDateString()}</span>
        </div>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn" on:click={exportTRD} title="Export TRD">
        <Download size={14} /> Export
      </button>
      <button class="btn" on:click={runBacktest} title="Run Backtest">
        <TestTube size={14} /> Backtest
      </button>
      <button class="btn primary" on:click={generateEA} title="Generate EA Code">
        <Code size={14} /> Generate EA
      </button>
      <button class="btn" class:save-btn={true} on:click={saveTRD} title="Save TRD">
        <Save size={14} /> Save
      </button>
    </div>
  </div>

  <!-- Tabs -->
  <div class="trd-tabs">
    <button
      class="tab"
      class:active={activeTab === 'vanilla'}
      on:click={() => activeTab = 'vanilla'}
    >
      <Layers size={14} />
      <span>Vanilla</span>
      <span class="tab-desc">Basic implementation</span>
    </button>
    <button
      class="tab"
      class:active={activeTab === 'spiced'}
      on:click={() => activeTab = 'spiced'}
    >
      <Zap size={14} />
      <span>Spiced</span>
      <span class="tab-desc">With SharedAssets + Sentinel</span>
    </button>
    <button
      class="tab"
      class:active={activeTab === 'comparison'}
      on:click={() => activeTab = 'comparison'}
    >
      <Copy size={14} />
      <span>Comparison</span>
      <span class="tab-desc">Side-by-side view</span>
    </button>
  </div>

  <!-- Content -->
  <div class="trd-content">
    {#if activeTab === 'comparison'}
      <!-- Comparison View -->
      <div class="comparison-view">
        <!-- Overview Section -->
        <div class="trd-section">
          <button class="section-header" on:click={() => toggleSection('overview')}>
            <FileText size={16} />
            <h3>Strategy Overview</h3>
            <span class:expanded={expandedSections.overview}><ChevronDown size={16} /></span>
          </button>

          {#if expandedSections.overview}
            <div class="section-content overview-grid">
              <div class="overview-item">
                <span class="label">Description</span>
                <span class="value">{trdData.overview.description}</span>
              </div>
              <div class="overview-item">
                <span class="label">Type</span>
                <span class="value badge">{trdData.overview.type}</span>
              </div>
              <div class="overview-item">
                <span class="label">Timeframe</span>
                <span class="value">{trdData.overview.timeframe}</span>
              </div>
              <div class="overview-item">
                <span class="label">Session</span>
                <span class="value">{trdData.overview.session}</span>
              </div>
              <div class="overview-item">
                <span class="label">Edge</span>
                <span class="value success">{(trdData.overview.edge * 100).toFixed(1)}%</span>
              </div>
              <div class="overview-item">
                <span class="label">Expectancy</span>
                <span class="value">{trdData.overview.expectancy.toFixed(2)}R</span>
              </div>
            </div>
          {/if}
        </div>

        <!-- Entry Conditions Comparison -->
        <div class="trd-section">
          <button class="section-header" on:click={() => toggleSection('entry')}>
            <Target size={16} />
            <h3>Entry Conditions</h3>
            <span class:expanded={expandedSections.entry}><ChevronDown size={16} /></span>
          </button>

          {#if expandedSections.entry}
            <div class="section-content comparison-grid">
              <!-- Vanilla -->
              <div class="comparison-column vanilla">
                <div class="column-header">
                  <Layers size={14} />
                  <h4>Vanilla</h4>
                </div>
                <div class="conditions-list">
                  {#each trdData.entry.vanilla.conditions as condition}
                    <div class="condition-item" class:disabled={!condition.enabled}>
                      <div class="condition-header">
                        <button
                          class="toggle-btn"
                          class:active={condition.enabled}
                          on:click|stopPropagation={() => toggleConditionEnabled(condition.id, 'vanilla')}
                        >
                          {#if condition.enabled}
                            <ToggleRight size={14} />
                          {:else}
                            <ToggleLeft size={14} />
                          {/if}
                        </button>
                        <span class="condition-name">{condition.name}</span>
                        <button class="icon-btn" on:click={() => startEditing(condition.id, 'vanilla')}>
                          <Edit3 size={12} />
                        </button>
                      </div>
                      <p class="condition-desc">{condition.description}</p>
                      <div class="condition-params">
                        {#each Object.entries(condition.parameters) as [key, value]}
                          <div class="param">
                            <span class="param-key">{key}:</span>
                            <span class="param-value">{String(value)}</span>
                          </div>
                        {/each}
                      </div>
                    </div>
                  {/each}
                </div>
              </div>

              <!-- Spiced -->
              <div class="comparison-column spiced">
                <div class="column-header">
                  <Zap size={14} />
                  <h4>Spiced (+SharedAssets)</h4>
                </div>
                <div class="conditions-list">
                  {#each trdData.entry.spiced.conditions as condition}
                    <div class="condition-item" class:disabled={!condition.enabled}>
                      <div class="condition-header">
                        <button
                          class="toggle-btn"
                          class:active={condition.enabled}
                          on:click|stopPropagation={() => toggleConditionEnabled(condition.id, 'spiced')}
                        >
                          {#if condition.enabled}
                            <ToggleRight size={14} />
                          {:else}
                            <ToggleLeft size={14} />
                          {/if}
                        </button>
                        <span class="condition-name">{condition.name}</span>
                        <button class="icon-btn" on:click={() => startEditing(condition.id, 'spiced')}>
                          <Edit3 size={12} />
                        </button>
                      </div>
                      <p class="condition-desc">{condition.description}</p>
                      {#if condition.sharedAsset}
                        <div class="shared-asset-badge">
                          <svelte:component this={getSharedAssetIcon(condition.sharedAsset)} size={10} />
                          <span>{condition.sharedAsset}</span>
                        </div>
                      {/if}
                      <div class="condition-params">
                        {#each Object.entries(condition.parameters) as [key, value]}
                          <div class="param">
                            <span class="param-key">{key}:</span>
                            <span class="param-value highlight={key.startsWith('with')}">{String(value)}</span>
                          </div>
                        {/each}
                      </div>
                    </div>
                  {/each}
                </div>
              </div>
            </div>
          {/if}
        </div>

        <!-- Exit Management Comparison -->
        <div class="trd-section">
          <button class="section-header" on:click={() => toggleSection('exit')}>
            <Activity size={16} />
            <h3>Exit Management</h3>
            <span class:expanded={expandedSections.exit}><ChevronDown size={16} /></span>
          </button>

          {#if expandedSections.exit}
            <div class="section-content comparison-grid">
              <!-- Vanilla Exit -->
              <div class="comparison-column vanilla">
                <div class="column-header">
                  <Layers size={14} />
                  <h4>Vanilla</h4>
                </div>
                <div class="exit-config">
                  <div class="exit-item">
                    <span class="exit-label">Stop Loss</span>
                    <div class="exit-value">
                      <span class="type">{trdData.exit.vanilla.stopLoss.type}</span>
                      <span class="value">{trdData.exit.vanilla.stopLoss.value}</span>
                      {#if trdData.exit.vanilla.stopLoss.type === 'fixed'}
                        <span class="unit">pips</span>
                      {:else}
                        <span class="unit">ATR</span>
                      {/if}
                    </div>
                  </div>
                  <div class="exit-item">
                    <span class="exit-label">Take Profit</span>
                    <div class="exit-value">
                      <span class="type">{trdData.exit.vanilla.takeProfit.type}</span>
                      <span class="value">{trdData.exit.vanilla.takeProfit.value}</span>
                      {#if trdData.exit.vanilla.takeProfit.type === 'fixed'}
                        <span class="unit">pips</span>
                      {:else}
                        <span class="unit">RR:1</span>
                      {/if}
                    </div>
                  </div>
                </div>
              </div>

              <!-- Spiced Exit -->
              <div class="comparison-column spiced">
                <div class="column-header">
                  <Zap size={14} />
                  <h4>Spiced (+Dynamic)</h4>
                </div>
                <div class="exit-config">
                  <div class="exit-item">
                    <span class="exit-label">Stop Loss</span>
                    <div class="exit-value">
                      <span class="type">{trdData.exit.spiced.stopLoss.type}</span>
                      <span class="value">{trdData.exit.spiced.stopLoss.value}</span>
                      <span class="unit">ATR</span>
                      {#if trdData.exit.spiced.stopLoss.trail}
                        <span class="badge">Trail @ {trdData.exit.spiced.stopLoss.trailActivation}R</span>
                      {/if}
                    </div>
                  </div>
                  <div class="exit-item">
                    <span class="exit-label">Take Profit</span>
                    <div class="exit-value">
                      <span class="type">{trdData.exit.spiced.takeProfit.type}</span>
                      <span class="value">{trdData.exit.spiced.takeProfit.ratio}</span>
                      <span class="unit">RR:1</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          {/if}
        </div>

        <!-- Risk Management Comparison -->
        <div class="trd-section">
          <button class="section-header" on:click={() => toggleSection('risk')}>
            <Shield size={16} />
            <h3>Risk Management</h3>
            <span class:expanded={expandedSections.risk}><ChevronDown size={16} /></span>
          </button>

          {#if expandedSections.risk}
            <div class="section-content comparison-grid">
              <!-- Vanilla Risk -->
              <div class="comparison-column vanilla">
                <div class="column-header">
                  <Layers size={14} />
                  <h4>Vanilla</h4>
                </div>
                <div class="risk-config">
                  <div class="risk-item">
                    <span class="risk-label">Mode</span>
                    <span class="risk-value">{trdData.risk.vanilla.mode}</span>
                  </div>
                  <div class="risk-item">
                    <span class="risk-label">Lot Size</span>
                    <span class="risk-value">{trdData.risk.vanilla.lotSize}</span>
                  </div>
                  <div class="risk-item">
                    <span class="risk-label">Max Positions</span>
                    <span class="risk-value">{trdData.risk.vanilla.maxPositions}</span>
                  </div>
                </div>
              </div>

              <!-- Spiced Risk -->
              <div class="comparison-column spiced">
                <div class="column-header">
                  <Zap size={14} />
                  <h4>Spiced (+Kelly)</h4>
                </div>
                <div class="risk-config">
                  <div class="risk-item">
                    <span class="risk-label">Mode</span>
                    <span class="risk-value highlight">{trdData.risk.spiced.mode}</span>
                  </div>
                  <div class="risk-details">
                    <div class="risk-detail">
                      <span>Base Edge: {(trdData.risk.spiced.kellyConfig.baseEdge * 100).toFixed(1)}%</span>
                    </div>
                    <div class="risk-detail">
                      <span>Base Odds: {trdData.risk.spiced.kellyConfig.baseOdds}:1</span>
                    </div>
                    <div class="risk-detail">
                      <span>Max Fraction: {(trdData.risk.spiced.kellyConfig.maxFraction * 100).toFixed(1)}%</span>
                    </div>
                    {#if trdData.risk.spiced.kellyConfig.regimeAdjustment}
                      <div class="risk-detail badge">
                        <TrendingUp size={10} />
                        <span>Regime Adjustment ON</span>
                      </div>
                    {/if}
                    {#if trdData.risk.spiced.kellyConfig.houseMoney}
                      <div class="risk-detail badge success">
                        <DollarSign size={10} />
                        <span>House Money Effect ON</span>
                      </div>
                    {/if}
                  </div>
                  <div class="risk-item">
                    <span class="risk-label">Squad Limits</span>
                    <span class="risk-value">{trdData.risk.spiced.squadLimits ? 'Enabled' : 'Disabled'}</span>
                  </div>
                </div>
              </div>
            </div>
          {/if}
        </div>

        <!-- Preferred Conditions (Router) -->
        <div class="trd-section">
          <button class="section-header" on:click={() => toggleSection('preferred')}>
            <BarChart3 size={16} />
            <h3>Preferred Conditions (Strategy Router)</h3>
            <span class:expanded={expandedSections.preferred}><ChevronDown size={16} /></span>
          </button>

          {#if expandedSections.preferred}
            <div class="section-content">
              <div class="info-box">
                <AlertTriangle size={14} />
                <span>These conditions tell the Strategy Router when this EA should participate in the auction</span>
              </div>

              <div class="preferred-grid">
                <div class="pref-card">
                  <TrendingUp size={16} />
                  <h4>Regime Quality</h4>
                  <div class="pref-range">
                    <span>Min: {trdData.preferredConditions.regime.quality.min}</span>
                    <span>Max: {trdData.preferredConditions.regime.quality.max}</span>
                  </div>
                </div>

                <div class="pref-card">
                  <Activity size={16} />
                  <h4>Chaos Score</h4>
                  <div class="pref-range">
                    <span>Min: {trdData.preferredConditions.regime.chaos.min}</span>
                    <span>Max: {trdData.preferredConditions.regime.chaos.max}</span>
                  </div>
                </div>

                <div class="pref-card">
                  <Layers size={16} />
                  <h4>Correlation Limit</h4>
                  <div class="pref-value">
                    <span>Max {trdData.preferredConditions.correlation.maxCorrelatedPositions} correlated positions</span>
                    <span>(threshold: {trdData.preferredConditions.correlation.correlationThreshold})</span>
                  </div>
                </div>

                <div class="pref-card">
                  <DollarSign size={16} />
                  <h4>House Money</h4>
                  <div class="pref-value">
                    {#if trdData.preferredConditions.houseMoney.enabled}
                      <span class="success">Enabled</span>
                      <span>Threshold: {(trdData.preferredConditions.houseMoney.threshold * 100).toFixed(0)}% of daily profit</span>
                    {:else}
                      <span>Disabled</span>
                    {/if}
                  </div>
                </div>
              </div>
            </div>
          {/if}
        </div>

        <!-- Backtest Requirements -->
        <div class="trd-section">
          <button class="section-header" on:click={() => toggleSection('backtest')}>
            <TestTube size={16} />
            <h3>Backtest Requirements</h3>
            <span class:expanded={expandedSections.backtest}><ChevronDown size={16} /></span>
          </button>

          {#if expandedSections.backtest}
            <div class="section-content">
              <div class="backtest-grid">
                <div class="bt-item">
                  <span class="bt-label">Variants</span>
                  <div class="bt-variants">
                    {#each trdData.backtest.variants as variant}
                      <span class="variant-badge">{variant}</span>
                    {/each}
                  </div>
                </div>
                <div class="bt-item">
                  <span class="bt-label">Symbols</span>
                  <div class="bt-symbols">
                    {#each trdData.backtest.symbols as symbol}
                      <span class="symbol-badge">{symbol}</span>
                    {/each}
                  </div>
                </div>
                <div class="bt-item">
                  <span class="bt-label">Period</span>
                  <span class="bt-value">{trdData.backtest.period}</span>
                </div>
                <div class="bt-item">
                  <span class="bt-label">Monte Carlo</span>
                  <span class="bt-value">{trdData.backtest.monteCarloRuns} runs</span>
                </div>
                {#if trdData.backtest.walkForward.enabled}
                  <div class="bt-item full-width">
                    <span class="bt-label">Walk-Forward</span>
                    <div class="walkforward-config">
                      <span>In-Sample: {trdData.backtest.walkForward.inSample}%</span>
                      <span>Step: {trdData.backtest.walkForward.step} month</span>
                    </div>
                  </div>
                {/if}
              </div>
            </div>
          {/if}
        </div>

        <!-- Shared Assets -->
        <div class="trd-section">
          <button class="section-header" on:click={() => toggleSection('assets')}>
            <FolderOpen size={16} />
            <h3>Shared Assets Dependencies</h3>
            <span class:expanded={expandedSections.assets}><ChevronDown size={16} /></span>
          </button>

          {#if expandedSections.assets}
            <div class="section-content">
              <div class="assets-list">
                {#each trdData.sharedAssets as asset}
                  <div class="asset-item">
                    <svelte:component this={getSharedAssetIcon(asset.name)} size={18} />
                    <div class="asset-info">
                      <span class="asset-name">{asset.name}</span>
                      <span class="asset-path">{asset.path}</span>
                    </div>
                    <span class="asset-version">v{asset.version}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      </div>
    {:else}
      <!-- Single View (Vanilla or Spiced) -->
      <div class="single-view">
        <div class="view-header">
          <h2>{activeTab === 'vanilla' ? 'Vanilla' : 'Spiced'} Configuration</h2>
          <p class="view-desc">
            {activeTab === 'vanilla'
              ? 'Basic strategy implementation with core conditions'
              : 'Enhanced implementation with SharedAssets and Sentinel integration'}
          </p>
        </div>
        <!-- Content similar to comparison but showing only one variant -->
        <div class="single-content">
          <p class="info-text">Edit mode for {activeTab} configuration</p>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .trd-editor {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
    overflow: hidden;
  }

  /* Header */
  .trd-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .trd-icon {
    color: var(--accent-primary);
  }

  .header-info {
    display: flex;
    flex-direction: column;
  }

  .trd-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .trd-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .status-badge {
    padding: 2px 8px;
    border-radius: 10px;
    color: white;
    font-size: 10px;
    font-weight: 500;
  }

  .header-actions {
    display: flex;
    gap: 8px;
  }

  /* Buttons */
  .btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:hover {
    background: var(--bg-surface);
    border-color: var(--border-default);
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.save-btn {
    background: #10b981;
    border-color: #10b981;
    color: white;
  }

  .icon-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }

  .icon-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  /* Tabs */
  .trd-tabs {
    display: flex;
    gap: 4px;
    padding: 12px 20px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .tab {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    background: transparent;
    border: none;
    border-radius: 8px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .tab-desc {
    font-size: 11px;
    opacity: 0.7;
  }

  /* Content */
  .trd-content {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
  }

  /* Sections */
  .trd-section {
    margin-bottom: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    overflow: hidden;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 14px 16px;
    background: transparent;
    border: none;
    border-bottom: 1px solid transparent;
    cursor: pointer;
    color: var(--text-primary);
  }

  .section-header:hover {
    background: var(--bg-tertiary);
  }

  .section-header h3 {
    flex: 1;
    margin: 0;
    font-size: 14px;
    font-weight: 500;
    text-align: left;
  }

  .section-header :global(svg) {
    transition: transform 0.2s;
  }

  .section-header :global(svg.expanded) {
    transform: rotate(180deg);
  }

  .section-content {
    padding: 16px;
    border-top: 1px solid var(--border-subtle);
  }

  /* Overview Grid */
  .overview-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }

  .overview-item {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .overview-item .label {
    font-size: 11px;
    color: var(--text-muted);
  }

  .overview-item .value {
    font-size: 14px;
    color: var(--text-primary);
  }

  .overview-item .value.success {
    color: #10b981;
  }

  /* Comparison Grid */
  .comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }

  .comparison-column {
    background: var(--bg-primary);
    border-radius: 8px;
    padding: 16px;
  }

  .comparison-column.vanilla {
    border-left: 3px solid #6b7280;
  }

  .comparison-column.spiced {
    border-left: 3px solid #f59e0b;
  }

  .column-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .column-header h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  /* Conditions List */
  .conditions-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .condition-item {
    padding: 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    transition: all 0.15s;
  }

  .condition-item:hover {
    border-color: var(--accent-primary);
  }

  .condition-item.disabled {
    opacity: 0.6;
  }

  .condition-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
  }

  .toggle-btn {
    display: flex;
    align-items: center;
    padding: 4px;
    background: var(--bg-tertiary);
    border: none;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
  }

  .toggle-btn.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .condition-name {
    flex: 1;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .condition-desc {
    font-size: 12px;
    color: var(--text-muted);
    margin-bottom: 8px;
  }

  .condition-params {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    font-size: 11px;
  }

  .param {
    display: flex;
    gap: 4px;
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
  }

  .param-key {
    color: var(--text-muted);
  }

  .param-value {
    color: var(--text-primary);
  }

  .param-value.highlight {
    color: var(--accent-primary);
  }

  .shared-asset-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 8px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 4px;
    font-size: 11px;
    color: #f59e0b;
    margin-bottom: 8px;
  }

  /* Exit Config */
  .exit-config {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .exit-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  .exit-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .exit-value {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .exit-value .type {
    padding: 2px 6px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    font-size: 11px;
    text-transform: capitalize;
  }

  .exit-value .value {
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
  }

  .exit-value .unit {
    font-size: 11px;
    color: var(--text-muted);
  }

  .exit-value .badge {
    padding: 2px 6px;
    background: var(--accent-primary);
    color: var(--bg-primary);
    border-radius: 4px;
    font-size: 10px;
  }

  /* Risk Config */
  .risk-config {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .risk-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  .risk-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .risk-value {
    font-size: 13px;
    color: var(--text-primary);
    font-weight: 500;
  }

  .risk-value.highlight {
    color: var(--accent-primary);
  }

  .risk-details {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 6px;
  }

  .risk-detail {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: var(--text-secondary);
    padding: 4px 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
  }

  .risk-detail.badge {
    background: rgba(59, 130, 246, 0.2);
    color: #3b82f6;
  }

  .risk-detail.badge.success {
    background: rgba(16, 185, 129, 0.2);
    color: #10b981;
  }

  /* Preferred Conditions */
  .info-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    margin-bottom: 16px;
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    font-size: 12px;
    color: var(--text-secondary);
  }

  .preferred-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
  }

  .pref-card {
    padding: 16px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
  }

  .pref-card h4 {
    display: flex;
    align-items: center;
    gap: 8px;
    margin: 0 0 12px;
    font-size: 12px;
    color: var(--text-muted);
  }

  .pref-range {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: var(--text-primary);
  }

  .pref-value {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-size: 13px;
    color: var(--text-primary);
  }

  .pref-value .success {
    color: #10b981;
  }

  /* Backtest Grid */
  .backtest-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
  }

  .bt-item {
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .bt-item.full-width {
    grid-column: 1 / -1;
  }

  .bt-label {
    display: block;
    font-size: 11px;
    color: var(--text-muted);
    margin-bottom: 6px;
  }

  .bt-value {
    font-size: 13px;
    color: var(--text-primary);
  }

  .bt-variants,
  .bt-symbols {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }

  .variant-badge,
  .symbol-badge {
    padding: 4px 8px;
    background: var(--bg-surface);
    color: var(--text-primary);
    border-radius: 4px;
    font-size: 11px;
  }

  .walkforward-config {
    display: flex;
    gap: 16px;
    font-size: 13px;
    color: var(--text-primary);
  }

  /* Assets List */
  .assets-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .asset-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .asset-item svg {
    color: var(--accent-primary);
  }

  .asset-info {
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  .asset-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .asset-path {
    font-size: 11px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
  }

  .asset-version {
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Single View */
  .single-view {
    height: 100%;
  }

  .view-header {
    text-align: center;
    padding: 40px 20px;
  }

  .view-header h2 {
    margin: 0 0 8px;
    font-size: 24px;
    color: var(--text-primary);
  }

  .view-desc {
    margin: 0;
    font-size: 14px;
    color: var(--text-muted);
  }

  .single-content {
    display: flex;
    align-items: center;
    justify-content: center;
    height: calc(100% - 120px);
  }

  .info-text {
    font-size: 14px;
    color: var(--text-muted);
  }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 2px 8px;
    background: var(--bg-tertiary);
    border-radius: 12px;
    font-size: 10px;
    text-transform: capitalize;
  }
</style>
