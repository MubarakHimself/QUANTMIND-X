// Canvas Context Service - loads CanvasContextTemplate for each canvas
import { API_CONFIG } from '$lib/config/api';
import { activeCanvasStore, CANVASES, type Canvas } from '$lib/stores/canvasStore';

export interface CanvasContextTemplate {
  canvas: string;
  canvas_display_name: string;
  canvas_icon: string;
  base_descriptor: string;
  memory_scope: string[];
  workflow_namespaces: string[];
  department_mailbox: string | null;
  shared_assets: string[];
  skill_index: Array<{
    id: string;
    path: string;
    trigger: string;
  }>;
  required_tools: string[];
  max_identifiers: number;
  department_head: string | null;
  suggestion_chips: CanvasSuggestionChip[];
}

export interface CanvasContextState {
  canvas: string;
  template: CanvasContextTemplate;
  memory_identifiers: string[];
  session_id: string | null;
  loaded_at: string;
}

export interface CanvasSuggestionChip {
  id: string;
  label: string;
  target_canvas: string;
  target_entity?: string;
  icon?: string;
}

class CanvasContextService {
  private cache: Map<string, CanvasContextTemplate> = new Map();
  private currentContext: CanvasContextState | null = null;

  /**
   * Get the current active canvas from the store
   */
  getActiveCanvas(): Canvas | undefined {
    let activeCanvas: string | undefined;

    // Subscribe to get current value
    activeCanvasStore.subscribe(value => {
      activeCanvas = value;
    })();

    return CANVASES.find(c => c.id === activeCanvas);
  }

  /**
   * Get canvas ID from canvas name (handles different formats)
   */
  private getCanvasId(canvasName: string): string {
    // Map common names to IDs
    const nameToId: Record<string, string> = {
      'live_trading': 'live-trading',
      'live-trading': 'live-trading',
      'trading': 'trading',
      'risk': 'risk',
      'portfolio': 'portfolio',
      'research': 'research',
      'development': 'development',
      'workshop': 'workshop',
      'flowforge': 'flowforge',
      'shared_assets': 'shared-assets',
      'shared-assets': 'shared-assets',
    };

    return nameToId[canvasName.toLowerCase()] || canvasName.toLowerCase();
  }

  /**
   * Get CanvasContextTemplate for a specific canvas
   */
  async getTemplate(canvasName: string): Promise<CanvasContextTemplate | null> {
    const canvasId = this.getCanvasId(canvasName);

    // Check cache first
    if (this.cache.has(canvasId)) {
      return this.cache.get(canvasId) || null;
    }

    try {
      const response = await fetch(
        `${API_CONFIG.API_BASE}/canvas-context/template/${canvasId}`
      );

      if (!response.ok) {
        console.warn(`Template not found for canvas: ${canvasId}`);
        return null;
      }

      const data = await response.json();
      const template = data.template as CanvasContextTemplate;

      // Cache the template
      this.cache.set(canvasId, template);

      return template;
    } catch (error) {
      console.error('Failed to load canvas template:', error);
      return null;
    }
  }

  /**
   * Load full canvas context including template and memory identifiers
   */
  async loadCanvasContext(
    canvasName: string,
    sessionId?: string,
    includeMemory: boolean = true
  ): Promise<CanvasContextState | null> {
    const canvasId = this.getCanvasId(canvasName);

    try {
      const response = await fetch(
        `${API_CONFIG.API_BASE}/canvas-context/load`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            canvas: canvasId,
            session_id: sessionId,
            include_memory_identifiers: includeMemory,
          }),
        }
      );

      if (!response.ok) {
        console.warn(`Failed to load canvas context for: ${canvasId}`);
        return null;
      }

      const data = await response.json();
      this.currentContext = data as CanvasContextState;

      return this.currentContext;
    } catch (error) {
      console.error('Failed to load canvas context:', error);
      return null;
    }
  }

  /**
   * Get list of all available canvases
   */
  async getAvailableCanvases(): Promise<Array<{
    id: string;
    name: string;
    icon?: string;
    department_head?: string;
  }>> {
    try {
      const response = await fetch(
        `${API_CONFIG.API_BASE}/canvas-context/canvases`
      );

      if (!response.ok) {
        return [];
      }

      const data = await response.json();
      return data.canvases || [];
    } catch (error) {
      console.error('Failed to get available canvases:', error);
      return [];
    }
  }

  /**
   * Get the current canvas context
   */
  getCurrentContext(): CanvasContextState | null {
    return this.currentContext;
  }

  /**
   * Clear the template cache
   */
  clearCache(): void {
    this.cache.clear();
    this.currentContext = null;
  }

  /**
   * Build canvas_context metadata for API calls
   */
  buildCanvasContextMetadata(sessionId?: string): {
    canvas_name: string;
    session_id: string;
    template_base_descriptor?: string;
  } {
    const activeCanvas = this.getActiveCanvas();
    const context = this.currentContext;

    return {
      canvas_name: activeCanvas?.id || 'workshop',
      session_id: sessionId || crypto.randomUUID(),
      template_base_descriptor: context?.template?.base_descriptor,
    };
  }

  /**
   * Get suggestion chips for the current canvas
   * These are canvas-aware navigation suggestions from the template
   */
  async getSuggestionChips(): Promise<CanvasSuggestionChip[]> {
    const activeCanvas = this.getActiveCanvas();
    if (!activeCanvas) {
      return this.getDefaultChips();
    }

    // Try to get chips from the cached template
    const template = this.cache.get(activeCanvas.id);
    if (template?.suggestion_chips && template.suggestion_chips.length > 0) {
      return template.suggestion_chips;
    }

    // Try to load template from API
    try {
      const loadedTemplate = await this.getTemplate(activeCanvas.id);
      if (loadedTemplate?.suggestion_chips && loadedTemplate.suggestion_chips.length > 0) {
        return loadedTemplate.suggestion_chips;
      }
    } catch (error) {
      console.warn('Failed to load template for chips:', error);
    }

    // Fall back to default chips
    return this.getDefaultChips();
  }

  /**
   * Get default suggestion chips when template doesn't provide any
   */
  private getDefaultChips(): CanvasSuggestionChip[] {
    const activeCanvas = this.getActiveCanvas();

    if (activeCanvas) {
      return [
        {
          id: 'goto-workshop',
          label: 'Go to Workshop',
          target_canvas: 'workshop',
          icon: 'Hammer',
        },
        {
          id: 'goto-risk',
          label: 'Risk Dashboard',
          target_canvas: 'risk',
          icon: 'Shield',
        },
        {
          id: 'goto-trading',
          label: 'Trading Floor',
          target_canvas: 'live-trading',
          icon: 'Activity',
        },
      ];
    }

    return [];
  }
}

// Singleton instance
export const canvasContextService = new CanvasContextService();