/**
 * Story 12-2: Design Token Consistency Pass — Tests
 *
 * Validates that app.css:
 *   - Contains no oklch() in :root
 *   - Defines all canonical Frosted Terminal tokens
 *   - Has no legacy --color-danger token
 *   - All 4 theme preset blocks set --tile-min-width and --tile-gap
 *   - Typography font tokens are present
 *   - Full spacing scale --space-1 through --space-12 present
 *
 * These are file-content assertions — they verify the CSS source
 * file is in the correct state without requiring a browser render.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'fs';
import { resolve, join, extname } from 'path';

let appCss: string;

beforeAll(() => {
  // Load app.css relative to this test file location
  const cssPath = resolve(__dirname, '../../app.css');
  appCss = readFileSync(cssPath, 'utf-8');
});

// ─────────────────────────────────────────────────────────────
// AC 12-2-1: No oklch() in :root
// ─────────────────────────────────────────────────────────────
describe('AC 12-2-1: No OKLCH vars in :root', () => {
  it('should contain zero oklch( occurrences anywhere in app.css', () => {
    const matches = appCss.match(/oklch\(/g);
    expect(matches).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────
// AC 12-2-2: Canonical colour values
// ─────────────────────────────────────────────────────────────
describe('AC 12-2-2: Canonical colour token values', () => {
  it('should define --color-bg-base with #080d14', () => {
    expect(appCss).toContain('--color-bg-base:');
    expect(appCss).toContain('#080d14');
  });

  it('should define --color-accent-cyan as #00d4ff', () => {
    expect(appCss).toContain('--color-accent-cyan:    #00d4ff');
  });

  it('should define --color-accent-amber as #f0a500', () => {
    expect(appCss).toContain('--color-accent-amber:   #f0a500');
  });

  it('should define --color-accent-green as #00c896', () => {
    expect(appCss).toContain('--color-accent-green:   #00c896');
  });

  it('should define --color-accent-red as #ff3b3b', () => {
    expect(appCss).toContain('--color-accent-red:     #ff3b3b');
  });

  it('should define --color-text-primary as #e8edf5', () => {
    expect(appCss).toContain('--color-text-primary:   #e8edf5');
  });

  it('should define --color-text-muted as #5a6a80', () => {
    expect(appCss).toContain('--color-text-muted:     #5a6a80');
  });
});

// ─────────────────────────────────────────────────────────────
// AC 12-2-3: Dept accent overrides
// ─────────────────────────────────────────────────────────────
describe('AC 12-2-3: Dept accent system', () => {
  it('should define :root --dept-accent defaulting to cyan', () => {
    expect(appCss).toContain('--dept-accent: var(--color-accent-cyan)');
  });

  it('should define research dept accent as amber', () => {
    expect(appCss).toMatch(/\[data-dept="research"\].*--dept-accent.*var\(--color-accent-amber\)/s);
  });

  it('should define risk dept accent as red', () => {
    expect(appCss).toMatch(/\[data-dept="risk"\].*--dept-accent.*var\(--color-accent-red\)/s);
  });

  it('should define development dept accent as cyan', () => {
    expect(appCss).toMatch(/\[data-dept="development"\].*--dept-accent.*var\(--color-accent-cyan\)/s);
  });

  it('should define trading dept accent as green', () => {
    expect(appCss).toMatch(/\[data-dept="trading"\].*--dept-accent.*var\(--color-accent-green\)/s);
  });
});

// ─────────────────────────────────────────────────────────────
// AC 12-2-5 & 12-2-6: Theme preset blocks
// ─────────────────────────────────────────────────────────────
describe('AC 12-2-5 & 12-2-6: Theme preset blocks define tile density', () => {
  const themes = [
    'balanced-terminal',
    'ghost-panel',
    'open-air',
    'breathing-space',
  ];

  for (const theme of themes) {
    it(`[data-theme="${theme}"] defines --tile-min-width`, () => {
      const themeBlockMatch = appCss.match(
        new RegExp(`\\[data-theme="${theme}"\\]\\s*\\{[^}]+\\}`, 's')
      );
      expect(themeBlockMatch).not.toBeNull();
      expect(themeBlockMatch![0]).toContain('--tile-min-width');
    });

    it(`[data-theme="${theme}"] defines --tile-gap`, () => {
      const themeBlockMatch = appCss.match(
        new RegExp(`\\[data-theme="${theme}"\\]\\s*\\{[^}]+\\}`, 's')
      );
      expect(themeBlockMatch).not.toBeNull();
      expect(themeBlockMatch![0]).toContain('--tile-gap');
    });
  }

  it('Ghost Panel should have --tile-min-width: 220px', () => {
    const ghostBlock = appCss.match(
      /\[data-theme="ghost-panel"\]\s*\{[^}]+\}/s
    );
    expect(ghostBlock).not.toBeNull();
    expect(ghostBlock![0]).toContain('--tile-min-width:      220px');
  });

  it('Ghost Panel should have --tile-gap: 10px', () => {
    const ghostBlock = appCss.match(
      /\[data-theme="ghost-panel"\]\s*\{[^}]+\}/s
    );
    expect(ghostBlock).not.toBeNull();
    expect(ghostBlock![0]).toContain('--tile-gap:            10px');
  });

  it('Balanced Terminal default should have --tile-min-width: 280px in :root', () => {
    // The :root block is up to the first closing brace after :root {
    const rootBlock = appCss.match(/:root\s*\{[^}]+\}/s);
    expect(rootBlock).not.toBeNull();
    expect(rootBlock![0]).toContain('--tile-min-width:      280px');
    expect(rootBlock![0]).toContain('--tile-gap:            18px');
  });

  it('Breathing Space should set --sb-density: comfortable', () => {
    const breathingBlock = appCss.match(
      /\[data-theme="breathing-space"\]\s*\{[^}]+\}/s
    );
    expect(breathingBlock).not.toBeNull();
    expect(breathingBlock![0]).toContain('--sb-density:          comfortable');
  });
});

// ─────────────────────────────────────────────────────────────
// AC 12-2-7: No --color-danger token
// ─────────────────────────────────────────────────────────────
describe('AC 12-2-7: --color-danger must not be defined', () => {
  it('should not define --color-danger anywhere in app.css', () => {
    // Check for the property definition (not var() reference in other files)
    const defined = appCss.match(/--color-danger\s*:/);
    expect(defined).toBeNull();
  });
});

// ─────────────────────────────────────────────────────────────
// AC 12-2-8: Full spacing + typography scale
// ─────────────────────────────────────────────────────────────
describe('AC 12-2-8: Spacing scale tokens', () => {
  const spacingTokens = [
    '--space-1',
    '--space-2',
    '--space-3',
    '--space-4',
    '--space-5',
    '--space-6',
    '--space-8',
    '--space-10',
    '--space-12',
  ];

  for (const token of spacingTokens) {
    it(`should define ${token}`, () => {
      expect(appCss).toContain(`${token}:`);
    });
  }
});

describe('AC 12-2-8: Typography scale tokens', () => {
  const typeTokens = [
    '--text-xs',
    '--text-sm',
    '--text-base',
    '--text-md',
    '--text-lg',
    '--text-xl',
    '--text-2xl',
  ];

  for (const token of typeTokens) {
    it(`should define ${token}`, () => {
      expect(appCss).toContain(`${token}:`);
    });
  }
});

describe('AC 12-2-8: Font tokens', () => {
  it('should define --font-data', () => {
    expect(appCss).toContain('--font-data:');
  });

  it('should define --font-heading', () => {
    expect(appCss).toContain('--font-heading:');
  });

  it('should define --font-body', () => {
    expect(appCss).toContain('--font-body:');
  });

  it('should define --font-ambient', () => {
    expect(appCss).toContain('--font-ambient:');
  });
});

// ─────────────────────────────────────────────────────────────
// Glass tier tokens
// ─────────────────────────────────────────────────────────────
describe('Glass tier tokens defined', () => {
  it('should define --glass-shell-bg (Tier 1)', () => {
    expect(appCss).toContain('--glass-shell-bg:');
    expect(appCss).toContain('rgba(8, 13, 20, 0.08)');
  });

  it('should define --glass-content-bg (Tier 2)', () => {
    expect(appCss).toContain('--glass-content-bg:');
    expect(appCss).toContain('rgba(8, 13, 20, 0.35)');
  });

  it('should define --glass-blur', () => {
    expect(appCss).toContain('--glass-blur:');
    expect(appCss).toContain('blur(12px) saturate(160%)');
  });
});

// ─────────────────────────────────────────────────────────────
// AC 12-2-4: No banned legacy tokens in non-settings .svelte files
// ─────────────────────────────────────────────────────────────
describe('AC 12-2-4: No banned legacy tokens in .svelte files (non-settings)', () => {
  // Walk a directory recursively and collect .svelte file paths
  function collectSvelteFiles(dir: string, skip: string[]): string[] {
    const results: string[] = [];
    for (const entry of readdirSync(dir)) {
      const full = join(dir, entry);
      const stat = statSync(full);
      if (stat.isDirectory()) {
        if (!skip.includes(entry)) {
          results.push(...collectSvelteFiles(full, skip));
        }
      } else if (extname(entry) === '.svelte') {
        results.push(full);
      }
    }
    return results;
  }

  const srcRoot = resolve(__dirname, '../..');
  // Settings sub-panels are explicitly excluded per story mandate
  const svelteFiles = collectSvelteFiles(srcRoot, ['settings']);

  const bannedTokens = [
    'var(--bg-primary)',
    'var(--accent-primary)',
    'var(--text-primary)',
    'var(--color-danger)',
    'var(--accent-danger)',
  ];

  for (const token of bannedTokens) {
    it(`no non-settings .svelte file should reference ${token}`, () => {
      const violations = svelteFiles.filter((f) =>
        readFileSync(f, 'utf-8').includes(token)
      );
      expect(violations).toEqual([]);
    });
  }
});
