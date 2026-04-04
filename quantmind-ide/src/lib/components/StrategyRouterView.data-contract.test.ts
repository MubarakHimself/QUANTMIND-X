import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'StrategyRouterView.svelte'), 'utf-8');

describe('StrategyRouterView.svelte — live router data contract', () => {
  it('loads router surfaces from live backend endpoints instead of seeded fake state', () => {
    expect(src).toContain("fetchJson<RouterStateShape>('/api/router/state')");
    expect(src).toContain("fetchJson<any>('/api/router/market')");
    expect(src).toContain("fetchJson<{ items?: any[] }>('/api/router/bots?limit=100')");
    expect(src).toContain("fetchJson<{ items?: any[] }>('/api/router/auctions?limit=20')");
    expect(src).toContain('/api/router/rankings?period=${period}');
    expect(src).toContain("fetchJson<any[]>('/api/router/correlations')");
    expect(src).toContain("fetchJson<any>('/api/router/house-money')");
    expect(src).not.toContain('ICT Scalper');
    expect(src).not.toContain('EURUSD/GBPUSD');
    expect(src).not.toContain('dailyProfit: 374.30');
  });

  it('uses the real auction payload shape and the router toggle query parameter', () => {
    expect(src).toContain("fetchJson(`/api/router/toggle?active=${nextActive}`");
    expect(src).toContain('if (result.auction)');
    expect(src).toContain('parseAuctionRecord(result.auction)');
  });
});
