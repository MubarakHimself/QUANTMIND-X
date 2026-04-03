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
  runtime_state?: Record<string, unknown>;
}

export interface CanvasAttachableResource {
  id: string;
  label: string;
  canvas: string;
  resource_type: string;
  path?: string;
  description?: string;
  metadata?: Record<string, unknown>;
}

export interface WorkspaceResourceManifest {
  canvas: string;
  generated_at: string;
  total_resources: number;
  by_type: Record<string, number>;
  sample: Array<{
    id: string;
    label: string;
    resource_type: string;
    canvas: string;
    path?: string;
  }>;
}

interface WorkspaceResourceSearchResponse {
  query: string;
  count: number;
  resources: Array<{
    resource_id: string;
    canvas: string;
    tab: string;
    type: string;
    path: string;
    label: string;
    metadata?: Record<string, unknown>;
    version?: string;
    updated_at?: string;
    relevance?: number;
  }>;
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
  private contextByCanvas: Map<string, CanvasContextState> = new Map();
  private runtimeStateByCanvas: Map<string, Record<string, unknown>> = new Map();

  private truncateText(value: unknown, maxLength: number = 320): unknown {
    if (typeof value !== 'string') {
      return value;
    }
    if (value.length <= maxLength) {
      return value;
    }
    return `${value.slice(0, maxLength)}…`;
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .split(/[^a-z0-9]+/g)
      .map((token) => token.trim())
      .filter((token) => token.length >= 2);
  }

  private scoreResourceMatch(resource: CanvasAttachableResource, tokens: string[]): number {
    if (tokens.length === 0) {
      return 0;
    }

    const haystack = [
      resource.label,
      resource.path,
      resource.description,
      resource.resource_type,
      JSON.stringify(resource.metadata ?? {}),
    ]
      .filter((value): value is string => typeof value === 'string' && value.length > 0)
      .join(' ')
      .toLowerCase();

    if (!haystack) {
      return 0;
    }

    let score = 0;
    for (const token of tokens) {
      if (!haystack.includes(token)) {
        continue;
      }
      if (resource.label.toLowerCase().includes(token)) {
        score += 6;
      } else if ((resource.path ?? '').toLowerCase().includes(token)) {
        score += 4;
      } else {
        score += 2;
      }
    }
    return score;
  }

  private sanitizeRuntimeStateForChat(runtimeState: unknown): Record<string, unknown> | undefined {
    if (!runtimeState || typeof runtimeState !== 'object') {
      return undefined;
    }

    const source = runtimeState as Record<string, unknown>;
    const sanitized: Record<string, unknown> = {};

    const passThroughKeys = [
      'active_tab',
      'visible_tabs',
      'counts',
      'selected_item',
      'status',
      'summary',
      'health',
      'sessions',
      'workflows',
      'nodes',
      'last_updated',
    ];

    for (const key of passThroughKeys) {
      const value = source[key];
      if (value !== undefined) {
        sanitized[key] = value;
      }
    }

    const attachable = source['attachable_resources'];
    if (Array.isArray(attachable)) {
      const byType: Record<string, number> = {};
      const sample = attachable
        .slice(0, 20)
        .filter((entry): entry is Record<string, unknown> => !!entry && typeof entry === 'object')
        .map((entry) => {
          const resourceType = typeof entry.resource_type === 'string' ? entry.resource_type : 'unknown';
          byType[resourceType] = (byType[resourceType] ?? 0) + 1;
          return {
            id: typeof entry.id === 'string' ? entry.id : '',
            label: this.truncateText(entry.label, 120),
            canvas: typeof entry.canvas === 'string' ? this.getCanvasId(entry.canvas) : undefined,
            resource_type: resourceType,
          };
        });

      for (const entry of attachable.slice(20)) {
        if (!entry || typeof entry !== 'object') continue;
        const resourceType = typeof (entry as { resource_type?: unknown }).resource_type === 'string'
          ? (entry as { resource_type: string }).resource_type
          : 'unknown';
        byType[resourceType] = (byType[resourceType] ?? 0) + 1;
      }

      sanitized.attachable_resources_manifest = {
        total: attachable.length,
        by_type: byType,
        sample,
      };
    }

    for (const [key, value] of Object.entries(source)) {
      if (key in sanitized || key === 'attachable_resources') {
        continue;
      }
      if (key.endsWith('_summary') || key.endsWith('_status') || key.endsWith('_counts')) {
        sanitized[key] = value;
      }
    }

    return Object.keys(sanitized).length > 0 ? sanitized : undefined;
  }

  private sanitizeTemplateForChat(template: CanvasContextTemplate): CanvasContextTemplate {
    return {
      ...template,
      // Keep descriptor semantics but cap payload size for chat requests.
      base_descriptor: String(this.truncateText(template.base_descriptor, 900)),
      memory_scope: Array.isArray(template.memory_scope) ? template.memory_scope.slice(0, 24) : [],
      workflow_namespaces: Array.isArray(template.workflow_namespaces) ? template.workflow_namespaces.slice(0, 16) : [],
      shared_assets: Array.isArray(template.shared_assets) ? template.shared_assets.slice(0, 24) : [],
      skill_index: Array.isArray(template.skill_index) ? template.skill_index.slice(0, 40) : [],
      required_tools: Array.isArray(template.required_tools) ? template.required_tools.slice(0, 40) : [],
      suggestion_chips: Array.isArray(template.suggestion_chips) ? template.suggestion_chips.slice(0, 16) : [],
    };
  }

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

  private getApiCanvasName(canvasName: string): string {
    const canvasId = this.getCanvasId(canvasName);
    const idToApiName: Record<string, string> = {
      'live-trading': 'live_trading',
      'shared-assets': 'shared_assets',
    };

    return idToApiName[canvasId] || canvasId;
  }

  /**
   * Get CanvasContextTemplate for a specific canvas
   */
  async getTemplate(canvasName: string): Promise<CanvasContextTemplate | null> {
    const canvasId = this.getCanvasId(canvasName);
    const apiCanvasName = this.getApiCanvasName(canvasName);

    // Check cache first
    if (this.cache.has(canvasId)) {
      return this.cache.get(canvasId) || null;
    }

    try {
      const response = await fetch(
        `${API_CONFIG.API_BASE}/canvas-context/template/${apiCanvasName}`
      );

      if (!response.ok) {
        console.warn(`Template not found for canvas: ${canvasId}`);
        return null;
      }

      const data = await response.json();
      const template = {
        ...(data.template as CanvasContextTemplate),
        canvas: canvasId,
      };

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
    const apiCanvasName = this.getApiCanvasName(canvasName);

    try {
      const response = await fetch(
        `${API_CONFIG.API_BASE}/canvas-context/load`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            canvas: apiCanvasName,
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
      const runtimeState = this.runtimeStateByCanvas.get(canvasId);
      this.currentContext = {
        ...(data as CanvasContextState),
        canvas: canvasId,
        runtime_state: runtimeState,
      };
      this.contextByCanvas.set(canvasId, this.currentContext);

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

  getCanvasContext(canvasName: string): CanvasContextState | null {
    const canvasId = this.getCanvasId(canvasName);
    return this.contextByCanvas.get(canvasId) ?? null;
  }

  setRuntimeState(canvasName: string, runtimeState: Record<string, unknown>): void {
    const canvasId = this.getCanvasId(canvasName);
    this.runtimeStateByCanvas.set(canvasId, runtimeState);

    const existing = this.contextByCanvas.get(canvasId);
    if (!existing) {
      return;
    }

    const updatedContext = {
      ...existing,
      runtime_state: runtimeState,
    };
    this.contextByCanvas.set(canvasId, updatedContext);
    if (this.currentContext?.canvas === canvasId) {
      this.currentContext = updatedContext;
    }
  }

  async getEnrichedContext(
    canvasName: string,
    sessionId?: string,
    includeMemory: boolean = true
  ): Promise<CanvasContextState | null> {
    const canvasId = this.getCanvasId(canvasName);
    const cached = this.contextByCanvas.get(canvasId);
    if (cached) {
      return {
        ...cached,
        runtime_state: this.runtimeStateByCanvas.get(canvasId) ?? cached.runtime_state,
      };
    }

    return this.loadCanvasContext(canvasId, sessionId, includeMemory);
  }

  async getChatContext(
    canvasName: string,
    sessionId?: string,
    includeMemory: boolean = true
  ): Promise<CanvasContextState | null> {
    const context = await this.getEnrichedContext(canvasName, sessionId, includeMemory);
    if (!context) {
      return null;
    }

    const sanitizedRuntimeState = this.sanitizeRuntimeStateForChat(context.runtime_state);
    return {
      ...context,
      template: this.sanitizeTemplateForChat(context.template),
      runtime_state: sanitizedRuntimeState,
    };
  }

  getAttachableResources(canvasName: string): CanvasAttachableResource[] {
    const context = this.getCanvasContext(canvasName);
    const resources = context?.runtime_state?.['attachable_resources'];
    if (!Array.isArray(resources)) {
      return [];
    }

    return resources
      .filter((resource): resource is CanvasAttachableResource =>
        !!resource &&
        typeof resource === 'object' &&
        typeof (resource as CanvasAttachableResource).id === 'string' &&
        typeof (resource as CanvasAttachableResource).label === 'string',
      )
      .map((resource) => ({
        ...resource,
        canvas: this.getCanvasId(resource.canvas || canvasName),
      }));
  }

  buildResourceAttachmentContext(resource: CanvasAttachableResource): Record<string, unknown> {
    return {
      id: resource.id,
      label: this.truncateText(resource.label, 160),
      canvas: this.getCanvasId(resource.canvas),
      type: resource.resource_type,
      resource_type: resource.resource_type,
      path: resource.path,
      description: this.truncateText(resource.description, 500),
      metadata: resource.metadata ?? {},
    };
  }

  buildWorkspaceResourceManifest(
    canvasName: string,
    resources?: CanvasAttachableResource[],
    sampleLimit: number = 20,
  ): WorkspaceResourceManifest {
    const canvasId = this.getCanvasId(canvasName);
    const resourceList = resources ?? this.getAttachableResources(canvasId);
    const byType: Record<string, number> = {};
    for (const resource of resourceList) {
      const key = resource.resource_type || 'unknown';
      byType[key] = (byType[key] ?? 0) + 1;
    }

    const sample = resourceList.slice(0, sampleLimit).map((resource) => ({
      id: resource.id,
      label: String(this.truncateText(resource.label, 140)),
      resource_type: resource.resource_type,
      canvas: this.getCanvasId(resource.canvas || canvasId),
      path: resource.path,
    }));

    return {
      canvas: canvasId,
      generated_at: new Date().toISOString(),
      total_resources: resourceList.length,
      by_type: byType,
      sample,
    };
  }

  async buildCanvasAttachmentContract(
    canvasName: string,
    sessionId?: string,
  ): Promise<Record<string, unknown>> {
    const canvasId = this.getCanvasId(canvasName);
    const context = await this.getChatContext(canvasId, sessionId);
    const template = context?.template;
    const resources = this.getAttachableResources(canvasId);

    return {
      attachment_type: 'canvas',
      canvas: canvasId,
      template_manifest: template
        ? {
            canvas_display_name: template.canvas_display_name,
            department_head: template.department_head,
            required_tools: template.required_tools.slice(0, 16),
            workflow_namespaces: template.workflow_namespaces.slice(0, 12),
            shared_assets: template.shared_assets.slice(0, 16),
          }
        : null,
      runtime_manifest: this.buildWorkspaceResourceManifest(canvasId, resources),
      runtime_state: this.sanitizeRuntimeStateForChat(context?.runtime_state),
    };
  }

  async searchAttachableResources(
    query: string,
    canvasNames: string[],
    limit: number = 12,
  ): Promise<CanvasAttachableResource[]> {
    const canvasIds = Array.from(new Set(canvasNames.map((name) => this.getCanvasId(name))));
    try {
      const response = await fetch(`${API_CONFIG.API_BASE}/canvas-context/resources/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          canvases: canvasIds,
          limit: Math.max(1, Math.min(limit, 200)),
        }),
      });

      if (response.ok) {
        const payload = await response.json() as WorkspaceResourceSearchResponse;
        const mapped = (payload.resources ?? []).map((resource) => ({
          id: resource.resource_id,
          label: resource.label,
          canvas: this.getCanvasId(resource.canvas),
          resource_type: resource.type,
          path: resource.path,
          description: typeof resource.metadata?.description === 'string'
            ? resource.metadata.description
            : '',
          metadata: {
            ...(resource.metadata ?? {}),
            resource_id: resource.resource_id,
            tab: resource.tab,
            version: resource.version,
            updated_at: resource.updated_at,
            relevance: resource.relevance,
          },
        }));
        if (mapped.length > 0) {
          return mapped.slice(0, limit);
        }
      }
    } catch (error) {
      console.warn('Workspace resource search failed, falling back to local runtime state', error);
    }

    // Fallback to local runtime-state search
    for (const canvasId of canvasIds) {
      await this.getEnrichedContext(canvasId);
    }

    const allResources = canvasIds.flatMap((canvasId) => this.getAttachableResources(canvasId));
    if (allResources.length === 0) return [];

    const tokens = this.tokenize(query);
    const scored = allResources
      .map((resource) => ({ resource, score: this.scoreResourceMatch(resource, tokens) }))
      .filter((entry) => (tokens.length === 0 ? true : entry.score > 0))
      .sort((left, right) => right.score - left.score);

    const deduped = new Map<string, CanvasAttachableResource>();
    for (const entry of scored) {
      const key = `${entry.resource.canvas}:${entry.resource.id}`;
      if (deduped.has(key)) continue;
      deduped.set(key, entry.resource);
      if (deduped.size >= limit) break;
    }
    return Array.from(deduped.values());
  }

  /**
   * Clear the template cache
   */
  clearCache(): void {
    this.cache.clear();
    this.currentContext = null;
    this.contextByCanvas.clear();
    this.runtimeStateByCanvas.clear();
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
