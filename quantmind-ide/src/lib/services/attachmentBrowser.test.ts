import { describe, expect, it } from 'vitest';
import type { CanvasAttachableResource } from './canvasContextService';
import {
  buildAttachmentResourceGroups,
  getAttachmentCanvasLabel,
  getAttachmentParentGroupId,
  getAttachmentGroupResources,
} from './attachmentBrowser';

describe('attachmentBrowser', () => {
  it('builds canvas-aware folder groups for shared assets and research resources', () => {
    const resources: CanvasAttachableResource[] = [
      {
        id: 'shared-assets:doc:mql5.pdf',
        label: 'MQL5 Reference',
        canvas: 'shared-assets',
        resource_type: 'doc',
        path: 'docs/mql5.pdf',
      },
      {
        id: 'shared-assets:strategy:scalping-vwap',
        label: 'Scalping VWAP',
        canvas: 'shared-assets',
        resource_type: 'strategy',
        path: 'strategies/scalping/vwap',
      },
      {
        id: 'research:article:orb-intro',
        label: 'ORB Introduction',
        canvas: 'research',
        resource_type: 'article',
        metadata: { category: 'orb' },
      },
      {
        id: 'research:book:mql5',
        label: 'MQL5 Book',
        canvas: 'research',
        resource_type: 'book',
      },
    ];

    const groups = buildAttachmentResourceGroups(resources, 'shared-assets');

    expect(groups.map((group) => group.label)).toEqual(['Docs', 'Strategies']);
    expect(getAttachmentCanvasLabel('shared-assets')).toBe('Shared Assets');

    const researchGroups = buildAttachmentResourceGroups(resources, 'research');
    expect(researchGroups.map((group) => group.label)).toEqual(['Articles', 'Books']);
  });

  it('keeps resources addressable by group and sorts them by label', () => {
    const resources: CanvasAttachableResource[] = [
      {
        id: 'development:variant-editor',
        label: 'Variant Editor',
        canvas: 'development',
        resource_type: 'editor',
      },
      {
        id: 'development:department-kanban',
        label: 'Department Task Board',
        canvas: 'development',
        resource_type: 'kanban',
      },
      {
        id: 'development:active-eas',
        label: 'Active EAs',
        canvas: 'development',
        resource_type: 'active-tab',
      },
      {
        id: 'development:mql5-reference',
        label: 'MQL5 Reference',
        canvas: 'development',
        resource_type: 'reference-book',
      },
    ];

    const groups = buildAttachmentResourceGroups(resources, 'development');

    expect(groups.map((group) => group.label)).toEqual([
      'Active EAs',
      'Department Tasks',
      'Reference Books',
      'Variants',
    ]);

    expect(getAttachmentGroupResources(resources, groups[3].id)).toEqual([
      {
        id: 'development:variant-editor',
        label: 'Variant Editor',
        canvas: 'development',
        resource_type: 'editor',
      },
    ]);
  });

  it('supports nested Shared Assets docs folders before listing resources', () => {
    const resources: CanvasAttachableResource[] = [
      {
        id: 'shared-assets:doc:mql5.pdf',
        label: 'MQL5 Reference',
        canvas: 'shared-assets',
        resource_type: 'doc',
        path: 'docs/mql5.pdf',
      },
      {
        id: 'knowledge/books/mql5book.pdf',
        label: 'MQL5 Book',
        canvas: 'shared-assets',
        resource_type: 'book',
        path: 'knowledge/books/mql5book.pdf',
      },
      {
        id: 'scraped/trading_systems/volatility-breakout.md',
        label: 'Volatility Breakout',
        canvas: 'shared-assets',
        resource_type: 'article',
        path: 'scraped/trading_systems/volatility-breakout.md',
      },
      {
        id: 'scraped/trading/news-breakout.md',
        label: 'News Breakout',
        canvas: 'shared-assets',
        resource_type: 'article',
        path: 'scraped/trading/news-breakout.md',
      },
    ];

    const topLevel = buildAttachmentResourceGroups(resources, 'shared-assets');
    const docsGroup = topLevel.find((group) => group.label === 'Docs');

    expect(docsGroup).toBeTruthy();
    expect(docsGroup?.hasChildren).toBe(true);

    const docsChildren = buildAttachmentResourceGroups(resources, 'shared-assets', docsGroup?.id ?? null);
    expect(docsChildren.map((group) => group.label)).toEqual(['Articles', 'Books', 'Docs']);

    const articlesGroup = docsChildren.find((group) => group.label === 'Articles');
    expect(articlesGroup?.hasChildren).toBe(true);
    expect(getAttachmentParentGroupId(articlesGroup?.id ?? null)).toBe(docsGroup?.id ?? null);

    const articleCategories = buildAttachmentResourceGroups(resources, 'shared-assets', articlesGroup?.id ?? null);
    expect(articleCategories.map((group) => group.label)).toEqual(['Trading', 'Trading Systems']);

    const tradingSystems = articleCategories.find((group) => group.label === 'Trading Systems');
    expect(tradingSystems?.hasChildren).toBe(false);
    expect(getAttachmentGroupResources(resources, tradingSystems?.id ?? '')).toEqual([
      {
        id: 'scraped/trading_systems/volatility-breakout.md',
        label: 'Volatility Breakout',
        canvas: 'shared-assets',
        resource_type: 'article',
        path: 'scraped/trading_systems/volatility-breakout.md',
      },
    ]);
  });

  it('supports nested Shared Assets strategy folders by family and source bucket', () => {
    const resources: CanvasAttachableResource[] = [
      {
        id: 'strategies/scalping/single-videos/london_scalper',
        label: 'London Scalper',
        canvas: 'shared-assets',
        resource_type: 'strategies',
        path: 'data/shared_assets/strategies/scalping/single-videos/london_scalper',
        metadata: {
          strategy_family: 'scalping',
          source_bucket: 'single-videos',
        },
      },
      {
        id: 'strategies/scalping/playlists/london_scalper_playlist',
        label: 'London Scalper Playlist',
        canvas: 'shared-assets',
        resource_type: 'strategies',
        path: 'data/shared_assets/strategies/scalping/playlists/london_scalper_playlist',
        metadata: {
          strategy_family: 'scalping',
          source_bucket: 'playlists',
        },
      },
    ];

    const topLevel = buildAttachmentResourceGroups(resources, 'shared-assets');
    const strategiesGroup = topLevel.find((group) => group.label === 'Strategies');

    expect(strategiesGroup).toBeTruthy();
    expect(strategiesGroup?.hasChildren).toBe(true);

    const families = buildAttachmentResourceGroups(resources, 'shared-assets', strategiesGroup?.id ?? null);
    expect(families.map((group) => group.label)).toEqual(['Scalping']);

    const buckets = buildAttachmentResourceGroups(resources, 'shared-assets', families[0]?.id ?? null);
    expect(buckets.map((group) => group.label)).toEqual(['Playlists', 'Single Videos']);

    const singleVideos = buckets.find((group) => group.label === 'Single Videos');
    expect(getAttachmentGroupResources(resources, singleVideos?.id ?? '')).toEqual([
      {
        id: 'strategies/scalping/single-videos/london_scalper',
        label: 'London Scalper',
        canvas: 'shared-assets',
        resource_type: 'strategies',
        path: 'data/shared_assets/strategies/scalping/single-videos/london_scalper',
        metadata: {
          strategy_family: 'scalping',
          source_bucket: 'single-videos',
        },
      },
    ]);
  });
});
