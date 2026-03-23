/**
 * Story 12-3: Tile Grid Pattern — Tests
 *
 * AC 12-3-1 through 12-3-13 validation.
 * Uses file content assertions for Svelte 5 components (Svelte 5 + @testing-library/svelte
 * incompatibility workaround — consistent with existing test patterns in this project).
 */

import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const COMPONENTS_DIR = resolve(__dirname, '..');
const CANVAS_DIR = resolve(COMPONENTS_DIR, 'canvas');
const SHARED_DIR = resolve(COMPONENTS_DIR, 'shared');

function readComponent(dir: string, name: string): string {
  return readFileSync(resolve(dir, name), 'utf-8');
}

// ─── Task 7.1: CanvasTileGrid.svelte ────────────────────────────────────────

describe('CanvasTileGrid.svelte', () => {
  const src = readComponent(SHARED_DIR, 'CanvasTileGrid.svelte');

  it('uses CSS grid layout token for columns', () => {
    expect(src).toContain('repeat(auto-fill, minmax(var(--tile-min-width), 1fr))');
  });

  it('uses var(--tile-gap) for gap', () => {
    expect(src).toContain('gap: var(--tile-gap)');
  });

  it('uses $props() for title, subtitle, dept, showBackButton, onBack', () => {
    expect(src).toContain('$props()');
    expect(src).toContain('title');
    expect(src).toContain('showBackButton');
    expect(src).toContain('onBack');
  });

  it('title uses var(--font-heading) at font-weight 800 and var(--text-xl) — AC 12-3-2', () => {
    expect(src).toContain('var(--font-heading)');
    expect(src).toContain('font-weight: 800');
    expect(src).toContain('var(--text-xl)');
  });

  it('title color is var(--color-text-primary) — AC 12-3-2', () => {
    expect(src).toContain('var(--color-text-primary)');
  });

  it('conditionally renders Breadcrumb when showBackButton is true', () => {
    expect(src).toContain('showBackButton');
    expect(src).toContain('Breadcrumb');
    expect(src).toContain('onBack');
  });

  it('data-dept is set on root element from dept prop', () => {
    expect(src).toContain('data-dept={dept');
  });

  it('does not import from kill-switch — AC 12-3-13', () => {
    expect(src).not.toContain('kill-switch');
    expect(src).not.toContain('KillSwitch');
  });
});

// ─── Task 7.2: TileCard.svelte ───────────────────────────────────────────────

describe('TileCard.svelte', () => {
  const src = readComponent(SHARED_DIR, 'TileCard.svelte');

  it('uses glass-content-bg for background — AC 12-3-4', () => {
    expect(src).toContain('var(--glass-content-bg)');
  });

  it('hover border transitions to rgba(255,255,255,0.13) — AC 12-3-3', () => {
    expect(src).toContain('rgba(255, 255, 255, 0.13)');
  });

  it('uses SkeletonLoader when isLoading is true — AC 12-3-11', () => {
    expect(src).toContain('SkeletonLoader');
    expect(src).toContain('isLoading');
  });

  it('renders epicOwner badge when epicOwner prop is set — AC 12-3-11', () => {
    expect(src).toContain('epicOwner');
    expect(src).toContain('epic-badge');
  });

  it('tile-title uses var(--font-ambient) for CRM section label — AC 12-3-7', () => {
    expect(src).toContain('var(--font-ambient)');
  });

  it('shows → view detail hint for navigable tiles — AC 12-3-6', () => {
    expect(src).toContain('→ view detail');
    expect(src).toContain('navigable');
  });

  it('xl tile spans full grid width', () => {
    expect(src).toContain('grid-column: 1 / -1');
  });

  it('imports from shared/TileCard (import path self-reference) — AC 12-3-8', () => {
    // TileCard imports SkeletonLoader from ./SkeletonLoader (both in shared/)
    expect(src).toContain("import SkeletonLoader from './SkeletonLoader.svelte'");
  });

  it('does not import from kill-switch — AC 12-3-13', () => {
    expect(src).not.toContain('kill-switch');
  });
});

// ─── Task 7.3: SkeletonLoader.svelte ────────────────────────────────────────

describe('SkeletonLoader.svelte', () => {
  const src = readComponent(SHARED_DIR, 'SkeletonLoader.svelte');

  it('uses --color-bg-elevated as background — never white — AC 12-3-11', () => {
    expect(src).toContain('var(--color-bg-elevated)');
    expect(src).not.toMatch(/background:\s*white/);
    expect(src).not.toMatch(/background:\s*#fff/);
    expect(src).not.toMatch(/background:\s*#ffffff/);
  });

  it('animates with skeleton-pulse keyframe', () => {
    expect(src).toContain('@keyframes skeleton-pulse');
    expect(src).toContain('skeleton-pulse');
  });

  it('uses $props() for lines and height props', () => {
    expect(src).toContain('$props()');
    expect(src).toContain('lines');
    expect(src).toContain('height');
  });
});

// ─── Task 7.4: data-dept on all 9 canvas roots ──────────────────────────────

describe('Canvas data-dept attributes — AC 12-3-10', () => {
  const canvases: Array<[string, string]> = [
    ['LiveTradingCanvas.svelte', 'trading'],
    ['ResearchCanvas.svelte', 'research'],
    ['DevelopmentCanvas.svelte', 'development'],
    ['RiskCanvas.svelte', 'risk'],
    ['PortfolioCanvas.svelte', 'portfolio'],
    ['SharedAssetsCanvas.svelte', 'shared'],
    ['WorkshopCanvas.svelte', 'workshop'],
    ['FlowForgeCanvas.svelte', 'flowforge'],
  ];

  for (const [file, dept] of canvases) {
    it(`${file} has data-dept="${dept}"`, () => {
      const src = readComponent(CANVAS_DIR, file);
      expect(src).toContain(`data-dept="${dept}"`);
    });
  }
});

// ─── TradingCanvas data-dept — AC 12-3-10 (via CanvasTileGrid dept prop) ────

describe('TradingCanvas.svelte data-dept — AC 12-3-10', () => {
  const src = readComponent(CANVAS_DIR, 'TradingCanvas.svelte');

  it('passes dept="trading" to CanvasTileGrid which sets data-dept on root', () => {
    // TradingCanvas delegates data-dept to CanvasTileGrid via dept prop.
    // CanvasTileGrid places data-dept={dept} on its root .canvas-tile-grid div.
    expect(src).toContain('dept="trading"');
  });

  it('CanvasTileGrid renders data-dept from dept prop', () => {
    // Verify CanvasTileGrid actually renders data-dept attribute from the dept prop
    const ctgSrc = readComponent(SHARED_DIR, 'CanvasTileGrid.svelte');
    expect(ctgSrc).toContain('data-dept={dept');
  });
});

// ─── Task 7.5: TradingCanvas — no CanvasPlaceholder, no epicNumber={3} ──────
// Updated by Story 12-4: Skeleton tiles replaced with live tile components.

describe('TradingCanvas.svelte — AC 12-3-1 (updated for Story 12-4)', () => {
  const src = readComponent(CANVAS_DIR, 'TradingCanvas.svelte');

  it('does not import CanvasPlaceholder (in code — comments excluded)', () => {
    const noComments = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(noComments).not.toContain('CanvasPlaceholder');
  });

  it('does not reference epicNumber (in code — comments excluded)', () => {
    const noComments = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(noComments).not.toContain('epicNumber={3}');
    expect(noComments).not.toContain('epicNumber');
  });

  it('imports from shared/CanvasTileGrid', () => {
    expect(src).toContain("from '$lib/components/shared/CanvasTileGrid.svelte'");
  });

  it('Story 12-4: imports live tile components from trading/tiles/ (skeleton replaced)', () => {
    // Skeleton TileCard import replaced by Story 12-4 live tile component imports.
    expect(src).toContain("from '$lib/components/trading/tiles/");
  });

  it('Story 12-4: no skeleton epicOwner TileCards (live tiles implemented)', () => {
    // Story 12-3 skeleton TileCards replaced by Story 12-4 live tile components.
    const noComments = src.replace(/\/\/[^\n]*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    expect(noComments).not.toContain('epicOwner="Epic 12-4"');
    expect(noComments).not.toContain('isLoading={true}');
  });

  it('has data-dept="trading" via CanvasTileGrid dept prop', () => {
    expect(src).toContain('dept="trading"');
  });
});

// ─── Task 7.6: Kill switch compliance on all canvases ────────────────────────

describe('Kill switch compliance — AC 12-3-13', () => {
  const canvasFiles = [
    'LiveTradingCanvas.svelte',
    'ResearchCanvas.svelte',
    'DevelopmentCanvas.svelte',
    'RiskCanvas.svelte',
    'PortfolioCanvas.svelte',
    'SharedAssetsCanvas.svelte',
    'TradingCanvas.svelte',
  ];

  for (const file of canvasFiles) {
    it(`${file} has no kill-switch/ import`, () => {
      const src = readComponent(CANVAS_DIR, file);
      // Must not import from kill-switch/ directory (Trading Kill Switch)
      expect(src).not.toMatch(/from.*kill-switch\//);
    });
  }

  it('TopBar.svelte still has kill switch import (must not be removed)', () => {
    const topBarSrc = readFileSync(
      resolve(COMPONENTS_DIR, 'TopBar.svelte'),
      'utf-8'
    );
    expect(topBarSrc).toContain('kill-switch');
  });
});

// ─── Workshop canvas icon corrections — AC 12-3-12 ──────────────────────────

describe('WorkshopCanvas.svelte — AC 12-3-12', () => {
  const src = readComponent(CANVAS_DIR, 'WorkshopCanvas.svelte');

  it('imports MessageSquare (History — was Clock)', () => {
    expect(src).toContain('MessageSquare');
  });

  it('imports GitBranch (Projects — was FolderOpen)', () => {
    expect(src).toContain('GitBranch');
  });

  it('imports Brain (Memory — was Database)', () => {
    expect(src).toContain('Brain');
  });

  it('imports Zap (Skills — was Sparkles)', () => {
    expect(src).toContain('Zap');
  });

  it('does not list Clock as an imported identifier (old History icon)', () => {
    // Clock should not be a named import since MessageSquare replaced it.
    // We strip comments first before checking, as comments may mention 'Clock' for documentation.
    const stripComments = (code: string) =>
      code.replace(/\/\/.*/g, '').replace(/\/\*[\s\S]*?\*\//g, '');
    const importBlock =
      stripComments(src).match(
        /import\s*\{[^}]+\}\s*from\s*['"]lucide-svelte['"]/s
      )?.[0] ?? '';
    expect(importBlock).not.toContain('Clock');
  });

  it('does not use Sparkles icon in template', () => {
    expect(src).not.toContain('<Sparkles');
  });

  it('does not use FolderOpen or Database icons in sidebar items', () => {
    expect(src).not.toContain('FolderOpen,');
    expect(src).not.toContain('Database,');
  });

  it('has no emoji in template — AC 12-3-12', () => {
    // Emoji are off-putting and violate the Frosted Terminal aesthetic per memory feedback
    expect(src).not.toMatch(/[\u{1F300}-\u{1FFFF}]/u);
  });

  it('has data-dept="workshop"', () => {
    expect(src).toContain('data-dept="workshop"');
  });
});

// ─── Shared component file boundary rules ────────────────────────────────────

describe('Shared component file boundary rules', () => {
  it('TileCard does not import canvas-specific or API modules', () => {
    const src = readComponent(SHARED_DIR, 'TileCard.svelte');
    expect(src).not.toContain('$lib/api');
    expect(src).not.toContain('canvas/');
  });

  it('CanvasTileGrid does not import canvas-specific components or stores', () => {
    const src = readComponent(SHARED_DIR, 'CanvasTileGrid.svelte');
    expect(src).not.toContain('$lib/stores');
    expect(src).not.toContain('canvas/');
  });

  it('RichRenderer.svelte still exists and was not overwritten', () => {
    const richRendererSrc = readFileSync(
      resolve(SHARED_DIR, 'RichRenderer.svelte'),
      'utf-8'
    );
    // Should still be the RichRenderer (not overwritten)
    expect(richRendererSrc.length).toBeGreaterThan(0);
  });
});

// ─── AC 12-3-1: No raw "Coming Soon" text on any canvas ─────────────────────

describe('AC 12-3-1: No raw "Coming Soon" text on canvas faces', () => {
  const canvasFiles = [
    'LiveTradingCanvas.svelte',
    'ResearchCanvas.svelte',
    'DevelopmentCanvas.svelte',
    'RiskCanvas.svelte',
    'PortfolioCanvas.svelte',
    'SharedAssetsCanvas.svelte',
    'WorkshopCanvas.svelte',
    'FlowForgeCanvas.svelte',
    'TradingCanvas.svelte',
  ];

  for (const file of canvasFiles) {
    it(`${file} contains no raw "Coming Soon" text`, () => {
      const src = readComponent(CANVAS_DIR, file);
      // Exact text check — "Coming Soon" as visible user-facing string
      expect(src).not.toContain('Coming Soon');
    });
  }
});

// ─── DevelopmentCanvas CSS token compliance ───────────────────────────────────

describe('DevelopmentCanvas CSS token compliance — AC story anti-pattern #4', () => {
  const src = readComponent(CANVAS_DIR, 'DevelopmentCanvas.svelte');

  it('does not hardcode font-family with quotes (must use CSS tokens)', () => {
    // Check that CSS block doesn't contain raw quoted font families
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    expect(styleBlock).not.toContain("font-family: 'JetBrains Mono'");
    expect(styleBlock).not.toContain('font-family: "JetBrains Mono"');
  });

  it('does not hardcode color hex values (must use CSS tokens)', () => {
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    expect(styleBlock).not.toContain('#a855f7');
    expect(styleBlock).not.toContain('#e0e0e0');
  });

  it('uses var(--dept-accent) for accent color', () => {
    const styleBlock = src.match(/<style>([\s\S]*?)<\/style>/)?.[1] ?? '';
    expect(styleBlock).toContain('var(--dept-accent)');
  });
});
