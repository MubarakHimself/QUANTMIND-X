import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const src = readFileSync(resolve(__dirname, 'flowforge.ts'), 'utf-8');

describe('flowforge.ts transport', () => {
  it('uses API_CONFIG for backend-prefixed workflow requests', () => {
    expect(src).toContain("from '$lib/config/api'");
    expect(src).toContain('function getFlowForgeApiUrl(path: string)');
  });

  it('does not use dev-server relative Prefect workflow paths', () => {
    expect(src).not.toContain("fetch('/api/prefect/workflows')");
    expect(src).toContain("fetch(getFlowForgeApiUrl('/api/prefect/workflows'))");
  });

  it('uses backend-prefixed workflow SSE events', () => {
    expect(src).toContain('new EventSource(');
    expect(src).toContain("getFlowForgeApiUrl(`/api/workflows/${workflowId}/events?run_id=${runResult.run_id}`)");
  });
});
