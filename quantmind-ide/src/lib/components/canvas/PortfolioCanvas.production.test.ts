import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const portfolioCanvasPath = resolve(process.cwd(), 'src/lib/components/canvas/PortfolioCanvas.svelte');

describe('PortfolioCanvas production data contract', () => {
  it('removes the Wiring Pending risk-exposure stub and demo-data account fallback', () => {
    const src = readFileSync(portfolioCanvasPath, 'utf8');
    const noComments = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');

    expect(src).not.toContain('Wiring Pending');
    expect(src).not.toContain('tile-value stub');
    expect(src).not.toContain("Fall back to the portfolio store's accounts as demo data");
    expect(noComments).not.toContain('accountsData = ($accounts as AccountRow[])');
  });

  it('does not ship hardcoded live portfolio metrics', () => {
    const src = readFileSync(portfolioCanvasPath, 'utf8');

    expect(src).not.toContain("value: '62.5%'");
    expect(src).not.toContain("value: '1.82'");
  });
});
