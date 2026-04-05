import { afterEach, describe, expect, it, vi } from 'vitest';
import { canvasContextService } from './canvasContextService';

describe('canvasContextService.searchAttachableResources', () => {
  afterEach(() => {
    canvasContextService.clearCache();
    vi.restoreAllMocks();
  });

  it('returns no global search results for empty queries', async () => {
    const fetchMock = vi.fn();

    vi.stubGlobal('fetch', fetchMock);

    const resources = await canvasContextService.searchAttachableResources('   ', ['research', 'shared-assets'], 20);

    expect(resources).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('keeps backend search results scoped to the requested canvases', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        query: 'mql5',
        count: 3,
        resources: [
          {
            resource_id: 'research:books:mql5',
            canvas: 'research',
            tab: 'books',
            type: 'book',
            path: 'books/mql5.pdf',
            label: 'MQL5 Book',
            metadata: { description: 'Research-side reference' },
          },
          {
            resource_id: 'shared-assets:strategy:me_at_the_zoo_smoke',
            canvas: 'shared-assets',
            tab: 'strategies',
            type: 'file',
            path: 'strategies/scalping/single-videos/me_at_the_zoo_smoke',
            label: 'me_at_the_zoo_smoke',
            metadata: { description: 'Canonical WF1 root' },
          },
          {
            resource_id: 'development:tab:active-eas',
            canvas: 'development',
            tab: 'active-eas',
            type: 'active-tab',
            path: 'development/active-eas',
            label: 'Active EAs',
            metadata: { description: 'Leaked development-only tab' },
          },
        ],
      }),
    });

    vi.stubGlobal('fetch', fetchMock);

    const resources = await canvasContextService.searchAttachableResources('mql5', ['research', 'shared-assets'], 20);

    expect(resources.map((resource) => `${resource.canvas}:${resource.label}`)).toEqual([
      'research:MQL5 Book',
      'shared-assets:me_at_the_zoo_smoke',
    ]);
    expect(resources.some((resource) => resource.canvas === 'development')).toBe(false);
  });

  it('hydrates shared-assets menu resources from the backend when runtime state is empty', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          canvas: 'shared-assets',
          template: {
            canvas: 'shared-assets',
            canvas_display_name: 'Shared Assets',
            canvas_icon: 'Library',
            base_descriptor: 'Shared asset registry',
            memory_scope: [],
            workflow_namespaces: [],
            department_mailbox: null,
            shared_assets: [],
            skill_index: [],
            required_tools: [],
            max_identifiers: 0,
            department_head: null,
            suggestion_chips: [],
          },
          memory_identifiers: [],
          session_id: null,
          loaded_at: '2026-04-05T00:00:00.000Z',
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ([
          {
            id: 'docs/mql5.pdf',
            name: 'MQL5 Reference',
            category: 'doc',
            path: 'docs/mql5.pdf',
            description: 'Core reference PDF',
          },
        ]),
      });

    vi.stubGlobal('fetch', fetchMock);

    const resources = await canvasContextService.getMenuAttachableResources('shared-assets', 20);

    expect(resources).toEqual([
      expect.objectContaining({
        id: 'docs/mql5.pdf',
        label: 'MQL5 Reference',
        canvas: 'shared-assets',
        resource_type: 'doc',
        path: 'docs/mql5.pdf',
      }),
    ]);
    expect(canvasContextService.getAttachableResources('shared-assets')).toEqual(resources);
  });

  it('rehydrates shared-assets menu resources when cached runtime state only contains one loaded category', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          canvas: 'shared-assets',
          template: {
            canvas: 'shared-assets',
            canvas_display_name: 'Shared Assets',
            canvas_icon: 'Library',
            base_descriptor: 'Shared asset registry',
            memory_scope: [],
            workflow_namespaces: [],
            department_mailbox: null,
            shared_assets: [],
            skill_index: [],
            required_tools: [],
            max_identifiers: 0,
            department_head: null,
            suggestion_chips: [],
          },
          memory_identifiers: [],
          session_id: null,
          loaded_at: '2026-04-05T00:00:00.000Z',
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ([
          {
            id: 'knowledge/books/mql5.pdf',
            name: 'MQL5 Reference',
            category: 'Books',
            path: 'knowledge/books/mql5.pdf',
            description: 'Core reference PDF',
          },
          {
            id: 'scraped/trading-systems/example.md',
            name: 'Example Trading Systems Article',
            category: 'Articles',
            path: 'scraped/trading-systems/example.md',
            description: 'Categorized article',
          },
          {
            id: 'strategies/scalping/single-videos/me_at_the_zoo_smoke',
            name: 'me_at_the_zoo_smoke',
            category: 'Strategies',
            path: 'strategies/scalping/single-videos/me_at_the_zoo_smoke',
            description: 'Canonical WF1 root',
          },
        ]),
      });

    vi.stubGlobal('fetch', fetchMock);

    await canvasContextService.loadCanvasContext('shared-assets');
    canvasContextService.setRuntimeState('shared-assets', {
      counts: {
        docs: 1808,
        strategies: 1,
      },
      attachable_resources: [
        {
          id: 'strategies/scalping/single-videos/me_at_the_zoo_smoke',
          label: 'me_at_the_zoo_smoke',
          canvas: 'shared-assets',
          resource_type: 'strategies',
          path: 'strategies/scalping/single-videos/me_at_the_zoo_smoke',
        },
      ],
    });

    const resources = await canvasContextService.getMenuAttachableResources('shared-assets', 20);

    expect(resources.map((resource) => resource.label)).toEqual([
      'me_at_the_zoo_smoke',
      'MQL5 Reference',
      'Example Trading Systems Article',
    ]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
