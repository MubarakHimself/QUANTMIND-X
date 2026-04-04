import { describe, expect, it } from 'vitest';
import {
  createCanvasPanelState,
  getCanvasCollapsed,
  getCanvasPanelState,
  setCanvasCollapsed,
  type CanvasPanelState,
} from './agentPanelState';

describe('agentPanelState', () => {
  it('returns isolated state buckets per canvas', () => {
    const buckets: Record<string, CanvasPanelState> = {};

    const research = getCanvasPanelState(buckets, 'research', createCanvasPanelState);
    research.activeSessionId = 'research-session';

    const development = getCanvasPanelState(buckets, 'development', createCanvasPanelState);

    expect(development.activeSessionId).toBeNull();
    expect(research).not.toBe(development);
  });

  it('keeps active session ids isolated per canvas bucket', () => {
    const buckets: Record<string, CanvasPanelState> = {};

    const research = getCanvasPanelState(buckets, 'research', createCanvasPanelState);
    research.activeSessionId = 'research-session';

    const development = getCanvasPanelState(buckets, 'development', createCanvasPanelState);
    development.activeSessionId = 'development-session';

    expect(research.activeSessionId).toBe('research-session');
    expect(development.activeSessionId).toBe('development-session');
  });

  it('stores collapse state independently per canvas', () => {
    let collapsedByCanvas: Record<string, boolean> = {};
    collapsedByCanvas = setCanvasCollapsed(collapsedByCanvas, 'research', true);
    collapsedByCanvas = setCanvasCollapsed(collapsedByCanvas, 'development', false);

    expect(getCanvasCollapsed(collapsedByCanvas, 'research')).toBe(true);
    expect(getCanvasCollapsed(collapsedByCanvas, 'development')).toBe(false);
    expect(getCanvasCollapsed(collapsedByCanvas, 'risk')).toBe(false);
  });

  it('restores collapse state for each canvas independently', () => {
    let collapsedByCanvas: Record<string, boolean> = {};
    collapsedByCanvas = setCanvasCollapsed(collapsedByCanvas, 'research', true);
    collapsedByCanvas = setCanvasCollapsed(collapsedByCanvas, 'development', false);

    expect(getCanvasCollapsed(collapsedByCanvas, 'research')).toBe(true);
    expect(getCanvasCollapsed(collapsedByCanvas, 'development')).toBe(false);
  });
});
