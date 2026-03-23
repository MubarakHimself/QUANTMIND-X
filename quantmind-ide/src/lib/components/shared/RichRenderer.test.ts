/**
 * RichRenderer — Story 12-1 Tests
 *
 * Tests the parseBlocks() pure logic and structural content assertions
 * for RichRenderer.svelte (in shared/).
 *
 * Two test strategies used:
 * 1. File-content assertions — verify component structure, CSS tokens, imports
 * 2. Extracted pure-logic tests — replicate parseBlocks() logic for unit testing
 *    (Svelte 5 + @testing-library/svelte incompatibility workaround)
 *
 * AC 12-1-15: RichRenderer renders markdown tables, code fences, charts, plain text
 * AC 12-1-7:  Tool call rendering uses RichRenderer for agent message content
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'RichRenderer.svelte'), 'utf-8');

// ─── File-content assertions ──────────────────────────────────────────────

describe('RichRenderer.svelte — File structure (AC 12-1-15)', () => {
  it('component file exists and is non-empty', () => {
    expect(src.length).toBeGreaterThan(0);
  });

  it('uses $props() for content prop (Svelte 5 rune)', () => {
    expect(src).toContain('$props()');
  });

  it('uses $derived for blocks (reactive parse result)', () => {
    expect(src).toContain('$derived(parseBlocks(content))');
  });

  it('defines parseBlocks function', () => {
    expect(src).toContain('function parseBlocks');
  });

  it('defines Block type union with code | table | chart | text', () => {
    expect(src).toContain("'code'");
    expect(src).toContain("'table'");
    expect(src).toContain("'chart'");
    expect(src).toContain("'text'");
  });

  it('defines isNumeric helper function', () => {
    expect(src).toContain('function isNumeric');
  });

  it('has rich-renderer root class', () => {
    expect(src).toContain('class="rich-renderer"');
  });

  it('renders code blocks with rr-code class', () => {
    expect(src).toContain('rr-code');
  });

  it('renders tables with rr-table class', () => {
    expect(src).toContain('rr-table');
  });

  it('renders chart placeholder with rr-chart-placeholder', () => {
    expect(src).toContain('rr-chart-placeholder');
  });

  it('renders text blocks with rr-text class', () => {
    expect(src).toContain('rr-text');
  });
});

// ─── CSS token compliance (AC 12-2-4) ─────────────────────────────────────

describe('RichRenderer.svelte — CSS token compliance (AC 12-2-4)', () => {
  it('uses var(--color-bg-elevated) for code block background', () => {
    expect(src).toContain('var(--color-bg-elevated)');
  });

  it('uses var(--color-text-primary) for code inner text', () => {
    expect(src).toContain('var(--color-text-primary)');
  });

  it('uses var(--font-mono) for code font-family', () => {
    expect(src).toContain('var(--font-mono');
  });

  it('uses var(--color-border-subtle) for table header border', () => {
    expect(src).toContain('var(--color-border-subtle)');
  });

  it('uses var(--color-text-secondary) for table header text', () => {
    expect(src).toContain('var(--color-text-secondary)');
  });

  it('uses var(--color-text-muted) for chart label', () => {
    expect(src).toContain('var(--color-text-muted)');
  });

  it('uses var(--color-accent-cyan) for chart spec text', () => {
    expect(src).toContain('var(--color-accent-cyan)');
  });

  it('does NOT use var(--bg-primary) or var(--accent-primary) (legacy OKLCH tokens)', () => {
    expect(src).not.toContain('var(--bg-primary)');
    expect(src).not.toContain('var(--accent-primary)');
  });

  it('does NOT hardcode hex color values in CSS', () => {
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    // Allow rgba() — only check for pure hex (#rrggbb) in style
    expect(styleBlock).not.toMatch(/:\s*#[0-9a-fA-F]{6}/);
  });
});

// ─── Numeric cell detection logic ─────────────────────────────────────────

describe('RichRenderer — isNumeric() logic', () => {
  // Replicated from component for pure unit testing

  function isNumeric(cell: string): boolean {
    return /^-?[\d,]+\.?\d*%?$/.test(cell.trim());
  }

  it('detects plain integer as numeric', () => {
    expect(isNumeric('42')).toBe(true);
  });

  it('detects negative number as numeric', () => {
    expect(isNumeric('-12.5')).toBe(true);
  });

  it('detects comma-separated number as numeric', () => {
    expect(isNumeric('1,234.56')).toBe(true);
  });

  it('detects percentage as numeric', () => {
    expect(isNumeric('98.7%')).toBe(true);
  });

  it('detects plain string as non-numeric', () => {
    expect(isNumeric('RSI')).toBe(false);
  });

  it('detects mixed alphanumeric as non-numeric', () => {
    expect(isNumeric('EA_GBPUSD')).toBe(false);
  });

  it('detects empty string as non-numeric', () => {
    expect(isNumeric('')).toBe(false);
  });

  it('trims whitespace before checking', () => {
    expect(isNumeric('  42  ')).toBe(true);
  });
});

// ─── parseBlocks() logic — replicated for unit testing ────────────────────

describe('RichRenderer — parseBlocks() logic', () => {
  // Pure replication of the component's parseBlocks logic
  type Block =
    | { kind: 'code'; lang: string; text: string }
    | { kind: 'table'; rows: string[][] }
    | { kind: 'chart'; spec: string }
    | { kind: 'text'; text: string };

  function parseBlocks(raw: string): Block[] {
    const blocks: Block[] = [];
    const lines = raw.split('\n');
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      const codeFenceMatch = line.match(/^```(\w*)$/);
      if (codeFenceMatch) {
        const lang = codeFenceMatch[1] || '';
        const codeLines: string[] = [];
        i++;
        while (i < lines.length && !lines[i].startsWith('```')) {
          codeLines.push(lines[i]);
          i++;
        }
        i++;
        blocks.push({ kind: 'code', lang, text: codeLines.join('\n') });
        continue;
      }

      const chartMatch = line.match(/^\[CHART:(\w+):(.+)\]$/);
      if (chartMatch) {
        blocks.push({ kind: 'chart', spec: line });
        i++;
        continue;
      }

      if (line.includes('|') && i + 1 < lines.length && lines[i + 1].match(/^\|?[\s\-:|]+\|/)) {
        const tableLines: string[] = [];
        while (i < lines.length && lines[i].includes('|')) {
          tableLines.push(lines[i]);
          i++;
        }
        const rows = tableLines
          .filter(l => !l.match(/^\|?[\s\-:|]+\|/))
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

  // ─── Plain text ─────────────────────────────────────────────────────────

  it('parses plain text into a single text block', () => {
    const blocks = parseBlocks('Hello world');
    expect(blocks).toHaveLength(1);
    expect(blocks[0].kind).toBe('text');
    if (blocks[0].kind === 'text') {
      expect(blocks[0].text).toBe('Hello world');
    }
  });

  it('returns empty array for empty string', () => {
    const blocks = parseBlocks('');
    expect(blocks).toHaveLength(0);
  });

  it('trims whitespace-only text blocks', () => {
    const blocks = parseBlocks('   \n\n   ');
    expect(blocks).toHaveLength(0);
  });

  // ─── Code fence ─────────────────────────────────────────────────────────

  it('parses code fence with language into code block', () => {
    const input = '```python\nprint("hello")\n```';
    const blocks = parseBlocks(input);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].kind).toBe('code');
    if (blocks[0].kind === 'code') {
      expect(blocks[0].lang).toBe('python');
      expect(blocks[0].text).toBe('print("hello")');
    }
  });

  it('parses code fence without language (empty lang)', () => {
    const input = '```\nconst x = 1;\n```';
    const blocks = parseBlocks(input);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].kind).toBe('code');
    if (blocks[0].kind === 'code') {
      expect(blocks[0].lang).toBe('');
    }
  });

  it('parses multi-line code fence correctly', () => {
    const input = '```typescript\nconst a = 1;\nconst b = 2;\n```';
    const blocks = parseBlocks(input);
    expect(blocks[0].kind).toBe('code');
    if (blocks[0].kind === 'code') {
      expect(blocks[0].text).toBe('const a = 1;\nconst b = 2;');
    }
  });

  it('parses text before and after code fence as separate blocks', () => {
    const input = 'intro text\n```js\nconst x = 1;\n```\ntrailing text';
    const blocks = parseBlocks(input);
    expect(blocks).toHaveLength(3);
    expect(blocks[0].kind).toBe('text');
    expect(blocks[1].kind).toBe('code');
    expect(blocks[2].kind).toBe('text');
  });

  // ─── Chart directive ─────────────────────────────────────────────────────

  it('parses [CHART:line:spec] into a chart block', () => {
    const input = '[CHART:line:RSI_14_daily]';
    const blocks = parseBlocks(input);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].kind).toBe('chart');
    if (blocks[0].kind === 'chart') {
      expect(blocks[0].spec).toBe('[CHART:line:RSI_14_daily]');
    }
  });

  it('parses [CHART:bar:vol_profile] into a chart block', () => {
    const input = '[CHART:bar:vol_profile]';
    const blocks = parseBlocks(input);
    expect(blocks[0].kind).toBe('chart');
  });

  it('does NOT parse partial [CHART... pattern as chart (no closing bracket)', () => {
    const input = '[CHART:line without close';
    const blocks = parseBlocks(input);
    // Should be parsed as text, not chart (regex requires full [CHART:type:spec])
    expect(blocks[0].kind).toBe('text');
  });

  // ─── Markdown table ──────────────────────────────────────────────────────

  it('parses markdown table into table block', () => {
    const input = '| Symbol | PnL |\n|--------|-----|\n| EURUSD | +120 |';
    const blocks = parseBlocks(input);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].kind).toBe('table');
  });

  it('table block has header row as first row', () => {
    const input = '| Symbol | PnL |\n|--------|-----|\n| EURUSD | +120 |';
    const blocks = parseBlocks(input);
    if (blocks[0].kind === 'table') {
      expect(blocks[0].rows[0]).toEqual(['Symbol', 'PnL']);
    }
  });

  it('table block has data rows after header', () => {
    const input = '| Symbol | PnL |\n|--------|-----|\n| EURUSD | +120 |';
    const blocks = parseBlocks(input);
    if (blocks[0].kind === 'table') {
      expect(blocks[0].rows[1]).toEqual(['EURUSD', '+120']);
    }
  });

  it('table block strips separator row (dashes line)', () => {
    const input = '| Symbol | PnL |\n|--------|-----|\n| EURUSD | +120 |';
    const blocks = parseBlocks(input);
    if (blocks[0].kind === 'table') {
      // Separator row should be stripped — only 2 rows: header + data
      expect(blocks[0].rows).toHaveLength(2);
    }
  });

  it('parses multi-row table correctly', () => {
    const input =
      '| EA | Sharpe | DD |\n|-----|--------|----|\n| EA1 | 2.1 | 5% |\n| EA2 | 1.8 | 8% |';
    const blocks = parseBlocks(input);
    if (blocks[0].kind === 'table') {
      expect(blocks[0].rows).toHaveLength(3); // header + 2 data rows
    }
  });

  it('parses mixed content: text + table + code', () => {
    const input = [
      'Analysis results:',
      '| Symbol | PnL |',
      '|--------|-----|',
      '| EURUSD | 120 |',
      '```python',
      'result = 42',
      '```',
    ].join('\n');
    const blocks = parseBlocks(input);
    expect(blocks[0].kind).toBe('text');
    expect(blocks[1].kind).toBe('table');
    expect(blocks[2].kind).toBe('code');
  });

  // ─── Edge cases ──────────────────────────────────────────────────────────

  it('handles empty code fence (no lines between backticks)', () => {
    const input = '```\n```';
    const blocks = parseBlocks(input);
    expect(blocks).toHaveLength(1);
    expect(blocks[0].kind).toBe('code');
    if (blocks[0].kind === 'code') {
      expect(blocks[0].text).toBe('');
    }
  });

  it('handles multiple code fences in sequence', () => {
    const input = '```js\nconst a = 1;\n```\n```py\nprint(a)\n```';
    const blocks = parseBlocks(input);
    expect(blocks).toHaveLength(2);
    expect(blocks[0].kind).toBe('code');
    expect(blocks[1].kind).toBe('code');
  });

  it('handles multiple chart directives in sequence', () => {
    const input = '[CHART:line:spec1]\n[CHART:bar:spec2]';
    const blocks = parseBlocks(input);
    expect(blocks).toHaveLength(2);
    expect(blocks[0].kind).toBe('chart');
    expect(blocks[1].kind).toBe('chart');
  });
});

// ─── Chart placeholder content (AC 12-1-15 deferred) ─────────────────────

describe('RichRenderer.svelte — Chart placeholder (AC 12-1-15)', () => {
  it('chart placeholder shows "rendered in Epic 5" text (deferred)', () => {
    expect(src).toContain('rendered in Epic 5');
  });

  it('chart placeholder has aria-label for accessibility', () => {
    expect(src).toContain('aria-label="Chart placeholder"');
  });

  it('chart placeholder renders the spec as visible code element', () => {
    expect(src).toContain('rr-chart-spec');
  });
});

// ─── Numeric column class ─────────────────────────────────────────────────

describe('RichRenderer.svelte — Numeric column class (AC 12-1-15)', () => {
  it('applies numeric class to td cells', () => {
    expect(src).toContain('class:numeric={isNumeric(cell)}');
  });

  it('numeric class uses tabular-nums font-variant', () => {
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    expect(styleBlock).toContain('tabular-nums');
  });

  it('numeric cells have text-align: right', () => {
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    expect(styleBlock).toContain('text-align: right');
  });
});
