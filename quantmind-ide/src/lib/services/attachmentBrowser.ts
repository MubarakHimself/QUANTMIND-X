import type { CanvasAttachableResource } from './canvasContextService';

export interface AttachmentResourceGroup {
  id: string;
  canvas: string;
  label: string;
  count: number;
  hasChildren: boolean;
}

const CANVAS_LABELS: Record<string, string> = {
  'live-trading': 'Live Trading',
  'research': 'Research',
  'development': 'Development',
  'trading': 'Trading',
  'risk': 'Risk',
  'portfolio': 'Portfolio',
  'shared-assets': 'Shared Assets',
  'flowforge': 'FlowForge',
  'workshop': 'Workshop',
};

function slugify(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

function titleCase(value: string): string {
  return value.replace(/\b\w/g, (match) => match.toUpperCase());
}

function formatCategoryLabel(value: string): string {
  return titleCase(value.replace(/[_-]+/g, ' ').trim());
}

function groupLabelForResource(resource: CanvasAttachableResource, canvas: string): string {
  const type = String(resource.resource_type || '').toLowerCase();
  const label = resource.label.toLowerCase();

  if (canvas === 'development') {
    if (type === 'reference-book') return 'Reference Books';
    if (type === 'kanban') return 'Department Tasks';
    if (type === 'active-tab' && label.includes('active ea')) return 'Active EAs';
    if (type === 'active-ea') return 'Active EAs';
    if (type === 'backtest-result') return 'Backtests';
    if (type === 'workflow-template' || (type === 'editor' && label.includes('workflow'))) return 'Workflows';
    if (type === 'browser' || (type === 'editor' && label.includes('variant'))) return 'Variants';
  }

  if (canvas === 'research') {
    if (type === 'article') return 'Articles';
    if (type === 'book') return 'Books';
    if (type === 'log') return 'Logs';
    if (type === 'personal-note') return 'Notes';
  }

  if (canvas === 'shared-assets') {
    if (['doc', 'docs', 'book', 'books', 'article', 'articles'].includes(type)) return 'Docs';
    if (['strategy', 'strategies'].includes(type)) return 'Strategies';
    if (type.includes('template')) return 'Strategy Templates';
    if (type.includes('indicator')) return 'Indicators';
    if (type.includes('skill')) return 'Skills';
    if (type.includes('flow-component')) return 'Flow Components';
    if (type.includes('mcp-config')) return 'MCP Configs';
  }

  if (canvas === 'portfolio') {
    if (type === 'active-tab') return 'Overview';
    if (type === 'broker-account') return 'Accounts';
    if (type === 'journal') return 'Journal';
    if (type === 'risk-metric') return 'Risk';
  }

  if (canvas === 'flowforge') {
    if (type === 'workflow-card') return 'Workflows';
    if (type === 'video-job') return 'Video Ingest';
  }

  if (canvas === 'trading' || canvas === 'live-trading') {
    if (type === 'kanban') return 'Department Tasks';
    if (type === 'active-bot') return 'Active Bots';
    if (type === 'tile') return 'Tiles';
  }

  if (canvas === 'risk') {
    if (type === 'risk-tab') return 'Risk Tabs';
    if (type === 'risk-metric') return 'Risk Metrics';
  }

  return titleCase(type.replace(/[-_]+/g, ' ').trim() || 'Resources');
}

function groupIdForResource(resource: CanvasAttachableResource, canvas: string): string {
  return `${canvas}:${slugify(groupLabelForResource(resource, canvas))}`;
}

interface AttachmentPathNode {
  segment: string;
  label: string;
}

function getSharedAssetDocSubgroup(resource: CanvasAttachableResource): AttachmentPathNode[] {
  const resourceType = String(resource.resource_type || '').toLowerCase();
  const resourceId = String(resource.id || '');
  const resourcePath = String(resource.path || '');

  if (resourceType === 'book' || resourceId.startsWith('knowledge/books/') || resourcePath.includes('/books/')) {
    return [{ segment: 'books', label: 'Books' }];
  }

  if (resourceType === 'article' || resourceId.startsWith('scraped/') || resourceId.startsWith('knowledge/articles/')) {
    const segments: AttachmentPathNode[] = [{ segment: 'articles', label: 'Articles' }];
    const articlePath = resourceId.startsWith('scraped/')
      ? resourceId.replace('scraped/', '')
      : resourcePath.startsWith('scraped/')
        ? resourcePath.replace('scraped/', '')
        : '';
    const category = articlePath.split('/')[0];
    if (category && category !== articlePath) {
      segments.push({ segment: slugify(category), label: formatCategoryLabel(category) });
    }
    return segments;
  }

  return [{ segment: 'docs-folder', label: 'Docs' }];
}

function getSharedAssetStrategySubgroup(resource: CanvasAttachableResource): AttachmentPathNode[] {
  const strategyFamily = String(resource.metadata?.strategy_family || '').trim();
  const sourceBucket = String(resource.metadata?.source_bucket || '').trim();
  const resourceId = String(resource.id || '').replace(/^strategies\//, '');
  const pathParts = resourceId.split('/').filter(Boolean);

  const family = strategyFamily || pathParts[0] || 'uncategorized';
  const bucket = sourceBucket || pathParts[1] || 'unknown';

  return [
    { segment: slugify(family), label: formatCategoryLabel(family) },
    { segment: slugify(bucket), label: formatCategoryLabel(bucket === 'single-videos' ? 'single videos' : bucket) },
  ];
}

function getAttachmentPathForResource(
  resource: CanvasAttachableResource,
  canvas: string,
): AttachmentPathNode[] {
  const topLevelLabel = groupLabelForResource(resource, canvas);
  const nodes: AttachmentPathNode[] = [{ segment: slugify(topLevelLabel), label: topLevelLabel }];

  if (canvas === 'shared-assets' && topLevelLabel === 'Docs') {
    nodes.push(...getSharedAssetDocSubgroup(resource));
  }

  if (canvas === 'shared-assets' && topLevelLabel === 'Strategies') {
    nodes.push(...getSharedAssetStrategySubgroup(resource));
  }

  return nodes;
}

function parseGroupId(groupId: string | null): { canvas: string; segments: string[] } | null {
  if (!groupId) return null;
  const separatorIndex = groupId.indexOf(':');
  if (separatorIndex === -1) return null;
  const canvas = groupId.slice(0, separatorIndex);
  const rawPath = groupId.slice(separatorIndex + 1);
  const segments = rawPath ? rawPath.split('/').filter(Boolean) : [];
  return { canvas, segments };
}

function buildGroupId(canvas: string, segments: string[]): string {
  return `${canvas}:${segments.join('/')}`;
}

function pathMatchesPrefix(path: AttachmentPathNode[], segments: string[]): boolean {
  if (segments.length > path.length) return false;
  return segments.every((segment, index) => path[index]?.segment === segment);
}

export function getAttachmentCanvasLabel(canvas: string): string {
  return CANVAS_LABELS[canvas] ?? titleCase(canvas.replace(/[-_]+/g, ' '));
}

export function buildAttachmentResourceGroups(
  resources: CanvasAttachableResource[],
  canvas: string,
  parentGroupId: string | null = null,
): AttachmentResourceGroup[] {
  const parsedParent = parseGroupId(parentGroupId);
  const parentSegments = parsedParent?.canvas === canvas ? parsedParent.segments : [];
  const groups = new Map<string, AttachmentResourceGroup>();
  for (const resource of resources) {
    if (resource.canvas !== canvas) continue;
    const path = getAttachmentPathForResource(resource, canvas);
    if (!pathMatchesPrefix(path, parentSegments)) continue;
    if (path.length <= parentSegments.length) continue;

    const nextNode = path[parentSegments.length];
    const nextSegments = [...parentSegments, nextNode.segment];
    const id = buildGroupId(canvas, nextSegments);
    const existing = groups.get(id);
    if (existing) {
      existing.count += 1;
      existing.hasChildren = existing.hasChildren || path.length > nextSegments.length;
      continue;
    }
    groups.set(id, {
      id,
      canvas,
      label: nextNode.label,
      count: 1,
      hasChildren: path.length > nextSegments.length,
    });
  }

  return Array.from(groups.values()).sort((left, right) => left.label.localeCompare(right.label));
}

export function getAttachmentGroupResources(
  resources: CanvasAttachableResource[],
  groupId: string,
): CanvasAttachableResource[] {
  const parsedGroup = parseGroupId(groupId);
  if (!parsedGroup) return [];
  const { canvas, segments } = parsedGroup;
  return resources
    .filter((resource) => resource.canvas === canvas)
    .filter((resource) => {
      const path = getAttachmentPathForResource(resource, canvas);
      return path.length === segments.length && pathMatchesPrefix(path, segments);
    })
    .sort((left, right) => left.label.localeCompare(right.label));
}

export function getAttachmentParentGroupId(groupId: string | null): string | null {
  const parsedGroup = parseGroupId(groupId);
  if (!parsedGroup || parsedGroup.segments.length === 0) return null;
  const parentSegments = parsedGroup.segments.slice(0, -1);
  return parentSegments.length > 0 ? buildGroupId(parsedGroup.canvas, parentSegments) : null;
}
