export interface CanvasPanelState {
  activeSessionId: string | null;
  sessions: Array<unknown>;
}

export function createCanvasPanelState(): CanvasPanelState {
  return {
    activeSessionId: null,
    sessions: [],
  };
}

export function getCanvasPanelState<T>(
  buckets: Record<string, T>,
  canvasId: string,
  createState: () => T
): T {
  if (!buckets[canvasId]) {
    buckets[canvasId] = createState();
  }
  return buckets[canvasId];
}

export function getCanvasCollapsed(
  collapsedByCanvas: Record<string, boolean>,
  canvasId: string
): boolean {
  return collapsedByCanvas[canvasId] ?? false;
}

export function setCanvasCollapsed(
  collapsedByCanvas: Record<string, boolean>,
  canvasId: string,
  collapsed: boolean
): Record<string, boolean> {
  return {
    ...collapsedByCanvas,
    [canvasId]: collapsed,
  };
}
