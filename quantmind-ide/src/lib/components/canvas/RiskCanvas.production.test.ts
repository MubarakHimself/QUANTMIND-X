import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const riskCanvasPath = resolve(process.cwd(), 'src/lib/components/canvas/RiskCanvas.svelte');

describe('RiskCanvas production placeholders', () => {
  it('does not ship static calendar placeholder tiles beside CalendarGateTile', () => {
    const src = readFileSync(riskCanvasPath, 'utf8');

    expect(src).toContain('<CalendarGateTile />');
    expect(src).not.toContain('Upcoming News Events');
    expect(src).not.toContain('Economic Calendar');
    expect(src).not.toContain('placeholder-tile');
  });
});
