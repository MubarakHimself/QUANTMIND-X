<script lang="ts">
  // RichRenderer — renders structured agent message content inline inside .ap-agent bubbles.
  // Handles: markdown tables, code fences, chart directives, plain text.

  interface Props {
    content: string;
  }

  let { content }: Props = $props();

  // ─── Block Types ──────────────────────────────────────────────────────────────

  type Block =
    | { kind: 'code'; lang: string; text: string }
    | { kind: 'table'; rows: string[][] }
    | { kind: 'chart'; spec: string }
    | { kind: 'text'; text: string };

  // ─── Parser ───────────────────────────────────────────────────────────────────

  function parseBlocks(raw: string): Block[] {
    const blocks: Block[] = [];
    const lines = raw.split('\n');
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // Code fence: ```lang
      const codeFenceMatch = line.match(/^```(\w*)$/);
      if (codeFenceMatch) {
        const lang = codeFenceMatch[1] || '';
        const codeLines: string[] = [];
        i++;
        while (i < lines.length && !lines[i].startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        i++; // skip closing ```
        blocks.push({ kind: 'code', lang, text: codeLines.join('\n') });
        continue;
      }

      // Chart directive: [CHART:type:spec]
      const chartMatch = line.match(/^\[CHART:(\w+):(.+)\]$/);
      if (chartMatch) {
        blocks.push({ kind: 'chart', spec: line });
        i++;
        continue;
      }

      // Markdown table: lines containing | characters (header row + separator)
      if (line.includes('|') && i + 1 < lines.length && lines[i + 1].match(/^\|?[\s\-:|]+\|/)) {
        const tableLines: string[] = [];
        while (i < lines.length && lines[i].includes('|')) {
          tableLines.push(lines[i]);
          i++;
        }
        const rows = tableLines
          .filter(l => !l.match(/^\|?[\s\-:|]+\|/)) // remove separator rows
          .map(l =>
            l
              .replace(/^\|/, '')
              .replace(/\|$/, '')
              .split('|')
              .map(cell => cell.trim())
          );
        if (rows.length > 0) {
          blocks.push({ kind: 'table', rows });
          continue;
        }
      }

      // Plain text accumulation
      const textLines: string[] = [];
      while (
        i < lines.length &&
        !lines[i].match(/^```\w*$/) &&
        !lines[i].match(/^\[CHART:\w+:.+\]$/) &&
        !(lines[i].includes('|') && i + 1 < lines.length && lines[i + 1].match(/^\|?[\s\-:|]+\|/))
      ) {
        textLines.push(lines[i]);
        i++;
      }
      const joined = textLines.join('\n').trim();
      if (joined) {
        blocks.push({ kind: 'text', text: joined });
      }
    }

    return blocks;
  }

  let blocks = $derived(parseBlocks(content));

  // ─── Helpers ──────────────────────────────────────────────────────────────────

  function isNumeric(cell: string): boolean {
    return /^-?[\d,]+\.?\d*%?$/.test(cell.trim());
  }
</script>

<div class="rich-renderer">
  {#each blocks as block, blockIdx (blockIdx)}
    {#if block.kind === 'code'}
      <pre class="rr-code"><code class="rr-code-inner lang-{block.lang}">{block.text}</code></pre>
    {:else if block.kind === 'table'}
      <div class="rr-table-wrap">
        <table class="rr-table">
          {#if block.rows.length > 0}
            <thead>
              <tr>
                {#each block.rows[0] as cell (cell)}
                  <th class="rr-th">{cell}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each block.rows.slice(1) as row, rowIdx (rowIdx)}
                <tr>
                  {#each row as cell, colIdx (colIdx)}
                    <td class="rr-td" class:numeric={isNumeric(cell)}>{cell}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          {/if}
        </table>
      </div>
    {:else if block.kind === 'chart'}
      <div class="rr-chart-placeholder" aria-label="Chart placeholder">
        <span class="rr-chart-label">Chart — rendered in Epic 5</span>
        <code class="rr-chart-spec">{block.spec}</code>
      </div>
    {:else if block.kind === 'text'}
      <p class="rr-text">{block.text}</p>
    {/if}
  {/each}
</div>

<style>
  .rich-renderer {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  /* Code blocks */
  .rr-code {
    background: var(--color-bg-elevated);
    border-radius: 4px;
    padding: 8px 10px;
    overflow-x: auto;
    margin: 0;
  }

  .rr-code-inner {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 11px;
    color: var(--color-text-primary);
    white-space: pre;
    display: block;
  }

  /* Tables */
  .rr-table-wrap {
    overflow-x: auto;
  }

  .rr-table {
    border-collapse: collapse;
    width: 100%;
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
  }

  .rr-th {
    text-align: left;
    padding: 4px 8px;
    border-bottom: 1px solid var(--color-border-subtle);
    color: var(--color-text-secondary);
    font-weight: 600;
    white-space: nowrap;
  }

  .rr-td {
    padding: 3px 8px;
    color: var(--color-text-primary);
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  }

  .rr-td.numeric {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }

  /* Chart placeholder */
  .rr-chart-placeholder {
    background: rgba(0, 170, 204, 0.04);
    border: 1px dashed rgba(0, 170, 204, 0.2);
    border-radius: 4px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
  }

  .rr-chart-label {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 10px;
    color: var(--color-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .rr-chart-spec {
    font-family: var(--font-mono, 'JetBrains Mono', monospace);
    font-size: 9px;
    color: var(--color-accent-cyan);
    opacity: 0.6;
  }

  /* Plain text */
  .rr-text {
    font-family: var(--font-family, 'Inter', system-ui, sans-serif);
    font-size: 13px;
    color: var(--color-text-primary);
    margin: 0;
    line-height: 1.5;
    white-space: pre-wrap;
  }
</style>
