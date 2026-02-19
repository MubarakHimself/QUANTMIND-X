<script lang="ts">
  import { onMount } from 'svelte';
  import { Code, Copy, Check, AlertCircle, Loader, Send, ArrowRight, FileCode } from 'lucide-svelte';
  import { PUBLIC_API_BASE } from '$env/static/public';

  // State
  let strategyDescription = '';
  let pineScriptCode = '';
  let mql5Source = '';
  let mode: 'generate' | 'convert' = 'generate';
  let isGenerating = false;
  let status: 'idle' | 'generating' | 'validating' | 'complete' | 'error' = 'idle';
  let errors: string[] = [];
  let copiedToClipboard = false;
  let copyButtonText = 'Copy to TradingView';

  // API base URL
  const apiBase = PUBLIC_API_BASE || 'http://localhost:8000';

  // Generate Pine Script from natural language
  async function generatePineScript() {
    if (!strategyDescription.trim() && mode === 'generate') {
      errors = ['Please enter a strategy description'];
      return;
    }

    if (!mql5Source.trim() && mode === 'convert') {
      errors = ['Please enter MQL5 source code'];
      return;
    }

    isGenerating = true;
    status = 'generating';
    errors = [];
    pineScriptCode = '';

    try {
      const endpoint = mode === 'convert' 
        ? `${apiBase}/api/chat/pinescript/convert`
        : `${apiBase}/api/chat/pinescript`;

      const body = mode === 'convert'
        ? { mql5_code: mql5Source }
        : { query: strategyDescription };

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      pineScriptCode = data.pine_script || '';
      status = data.status || 'complete';
      errors = data.errors || [];

      if (status === 'error' && errors.length === 0) {
        errors = ['Failed to generate Pine Script'];
      }

    } catch (error) {
      status = 'error';
      errors = [error instanceof Error ? error.message : 'Unknown error occurred'];
    } finally {
      isGenerating = false;
    }
  }

  // Copy to clipboard
  async function copyToClipboard() {
    try {
      await navigator.clipboard.writeText(pineScriptCode);
      copiedToClipboard = true;
      copyButtonText = 'Copied!';
      setTimeout(() => {
        copiedToClipboard = false;
        copyButtonText = 'Copy to TradingView';
      }, 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }

  // Example strategies
  const exampleStrategies = [
    'Create an ICT Silver Bullet strategy for London and New York killzones',
    'RSI divergence strategy with MACD confirmation',
    'Bollinger Bands squeeze breakout strategy',
    'Moving average crossover with ATR-based stop loss',
    'Support/Resistance breakout with volume confirmation'
  ];

  // Insert example
  function insertExample(example: string) {
    strategyDescription = example;
    mode = 'generate';
  }
</script>

<div class="pinescript-panel">
  <!-- Header -->
  <div class="panel-header">
    <div class="header-title">
      <FileCode size={24} />
      <h2>Pine Script Generator</h2>
    </div>
    <p class="header-description">
      Generate TradingView Pine Script v5 code from natural language descriptions
    </p>
  </div>

  <!-- Mode Selector -->
  <div class="mode-selector">
    <button 
      class:active={mode === 'generate'}
      on:click={() => mode = 'generate'}
    >
      <Send size={16} />
      Generate from Description
    </button>
    <button 
      class:active={mode === 'convert'}
      on:click={() => mode = 'convert'}
    >
      <Code size={16} />
      Convert MQL5
    </button>
  </div>

  <!-- Input Section -->
  {#if mode === 'generate'}
    <div class="input-section">
      <label for="strategy-input">Strategy Description</label>
      <textarea
        id="strategy-input"
        bind:value={strategyDescription}
        placeholder="Describe your trading strategy in natural language...&#10;&#10;Example: Create an RSI reversal strategy that enters long when RSI crosses below 30 and exits when it crosses above 70"
        rows="5"
        disabled={isGenerating}
      ></textarea>
      
      <!-- Example Strategies -->
      <div class="examples">
        <span class="examples-label">Try an example:</span>
        <div class="example-chips">
          {#each exampleStrategies as example, i}
            <button 
              class="example-chip" 
              on:click={() => insertExample(example)}
              disabled={isGenerating}
            >
              {example.substring(0, 30)}...
            </button>
          {/each}
        </div>
      </div>
    </div>
  {:else}
    <div class="input-section">
      <label for="mql5-input">MQL5 Source Code</label>
      <textarea
        id="mql5-input"
        bind:value={mql5Source}
        placeholder="Paste your MQL5 code here..."
        rows="10"
        disabled={isGenerating}
        class="code-input"
      ></textarea>
    </div>
  {/if}

  <!-- Generate Button -->
  <button 
    class="generate-button"
    on:click={generatePineScript}
    disabled={isGenerating || (mode === 'generate' ? !strategyDescription.trim() : !mql5Source.trim())}
  >
    {#if isGenerating}
      <Loader size={20} class="spinning" />
      <span>Generating...</span>
    {:else}
      <ArrowRight size={20} />
      <span>Generate Pine Script</span>
    {/if}
  </button>

  <!-- Status Indicator -->
  {#if status !== 'idle'}
    <div class="status-indicator status-{status}">
      {#if status === 'generating'}
        <Loader size={16} class="spinning" />
        Generating Pine Script code...
      {:else if status === 'validating'}
        <Loader size={16} class="spinning" />
        Validating syntax...
      {:else if status === 'complete'}
        <Check size={16} />
        Pine Script generated successfully!
      {:else if status === 'error'}
        <AlertCircle size={16} />
        Error occurred
      {/if}
    </div>
  {/if}

  <!-- Errors -->
  {#if errors.length > 0}
    <div class="errors-section">
      <div class="errors-header">
        <AlertCircle size={16} />
        <span>{errors.length} error(s) found:</span>
      </div>
      <ul class="errors-list">
        {#each errors as error, i}
          <li>{error}</li>
        {/each}
      </ul>
    </div>
  {/if}

  <!-- Output Section -->
  {#if pineScriptCode}
    <div class="output-section">
      <div class="output-header">
        <h3>Generated Pine Script</h3>
        <button 
          class="copy-button"
          class:copied={copiedToClipboard}
          on:click={copyToClipboard}
        >
          {#if copiedToClipboard}
            <Check size={16} />
          {:else}
            <Copy size={16} />
          {/if}
          {copyButtonText}
        </button>
      </div>
      <div class="code-block">
        <pre><code>{pineScriptCode}</code></pre>
      </div>
      
      <!-- Instructions -->
      <div class="instructions">
        <h4>How to use in TradingView:</h4>
        <ol>
          <li>Copy the Pine Script code above</li>
          <li>Open TradingView and go to Pine Editor</li>
          <li>Paste the code and click "Add to Chart"</li>
          <li>Adjust input parameters as needed</li>
          <li>For alerts, configure alert conditions in TradingView settings</li>
        </ol>
      </div>
    </div>
  {/if}
</div>

<style>
  .pinescript-panel {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    padding: 1.5rem;
    background: var(--bg-secondary, #1a1a2e);
    border-radius: 12px;
    height: 100%;
    overflow-y: auto;
  }

  .panel-header {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    color: var(--text-primary, #ffffff);
  }

  .header-title h2 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 600;
  }

  .header-description {
    margin: 0;
    color: var(--text-secondary, #a0a0a0);
    font-size: 0.875rem;
  }

  .mode-selector {
    display: flex;
    gap: 0.5rem;
  }

  .mode-selector button {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    background: var(--bg-tertiary, #16213e);
    border: 1px solid var(--border-color, #2a2a4a);
    border-radius: 8px;
    color: var(--text-secondary, #a0a0a0);
    cursor: pointer;
    transition: all 0.2s;
  }

  .mode-selector button:hover {
    background: var(--bg-hover, #1f2b47);
    color: var(--text-primary, #ffffff);
  }

  .mode-selector button.active {
    background: var(--accent-primary, #4f46e5);
    border-color: var(--accent-primary, #4f46e5);
    color: white;
  }

  .input-section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .input-section label {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-primary, #ffffff);
  }

  .input-section textarea {
    width: 100%;
    padding: 1rem;
    background: var(--bg-tertiary, #16213e);
    border: 1px solid var(--border-color, #2a2a4a);
    border-radius: 8px;
    color: var(--text-primary, #ffffff);
    font-size: 0.875rem;
    resize: vertical;
    font-family: inherit;
  }

  .input-section textarea:focus {
    outline: none;
    border-color: var(--accent-primary, #4f46e5);
  }

  .input-section textarea.code-input {
    font-family: 'Fira Code', 'Monaco', monospace;
    font-size: 0.8rem;
  }

  .examples {
    margin-top: 0.5rem;
  }

  .examples-label {
    font-size: 0.75rem;
    color: var(--text-secondary, #a0a0a0);
  }

  .example-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
  }

  .example-chip {
    padding: 0.375rem 0.75rem;
    background: var(--bg-tertiary, #16213e);
    border: 1px solid var(--border-color, #2a2a4a);
    border-radius: 16px;
    color: var(--text-secondary, #a0a0a0);
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .example-chip:hover {
    background: var(--bg-hover, #1f2b47);
    color: var(--text-primary, #ffffff);
    border-color: var(--accent-primary, #4f46e5);
  }

  .generate-button {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.875rem 1.5rem;
    background: var(--accent-primary, #4f46e5);
    border: none;
    border-radius: 8px;
    color: white;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }

  .generate-button:hover:not(:disabled) {
    background: var(--accent-hover, #4338ca);
  }

  .generate-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-size: 0.875rem;
  }

  .status-generating,
  .status-validating {
    background: var(--status-info-bg, #1e3a5f);
    color: var(--status-info, #60a5fa);
  }

  .status-complete {
    background: var(--status-success-bg, #1e3f2f);
    color: var(--status-success, #22c55e);
  }

  .status-error {
    background: var(--status-error-bg, #3f1e1e);
    color: var(--status-error, #ef4444);
  }

  .errors-section {
    padding: 1rem;
    background: var(--status-error-bg, #3f1e1e);
    border-radius: 8px;
    border: 1px solid var(--status-error, #ef4444);
  }

  .errors-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--status-error, #ef4444);
    font-weight: 500;
    margin-bottom: 0.5rem;
  }

  .errors-list {
    margin: 0;
    padding-left: 1.5rem;
    color: var(--text-secondary, #fca5a5);
    font-size: 0.875rem;
  }

  .errors-list li {
    margin: 0.25rem 0;
  }

  .output-section {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .output-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .output-header h3 {
    margin: 0;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary, #ffffff);
  }

  .copy-button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: var(--bg-tertiary, #16213e);
    border: 1px solid var(--border-color, #2a2a4a);
    border-radius: 6px;
    color: var(--text-secondary, #a0a0a0);
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }

  .copy-button:hover {
    background: var(--bg-hover, #1f2b47);
    color: var(--text-primary, #ffffff);
  }

  .copy-button.copied {
    background: var(--status-success-bg, #1e3f2f);
    color: var(--status-success, #22c55e);
    border-color: var(--status-success, #22c55e);
  }

  .code-block {
    background: var(--bg-tertiary, #0d1117);
    border: 1px solid var(--border-color, #2a2a4a);
    border-radius: 8px;
    overflow: hidden;
  }

  .code-block pre {
    margin: 0;
    padding: 1rem;
    overflow-x: auto;
    max-height: 400px;
    overflow-y: auto;
  }

  .code-block code {
    font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
    font-size: 0.8rem;
    color: var(--text-primary, #e6e6e6);
    white-space: pre;
  }

  .instructions {
    padding: 1rem;
    background: var(--bg-tertiary, #16213e);
    border-radius: 8px;
  }

  .instructions h4 {
    margin: 0 0 0.75rem 0;
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary, #ffffff);
  }

  .instructions ol {
    margin: 0;
    padding-left: 1.25rem;
    color: var(--text-secondary, #a0a0a0);
    font-size: 0.8rem;
    line-height: 1.6;
  }

  .instructions li {
    margin: 0.25rem 0;
  }
</style>