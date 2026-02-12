import { writable, derived } from 'svelte/store';

export interface Breadcrumb {
  id: string;
  label: string;
  name?: string; // Optional for backwards compatibility
  type: 'view' | 'folder' | 'file' | 'subpage';
  path?: string; // For Breadcrumbs component compatibility
  fullPath?: string; // Store the full path for nested folders (e.g., 'ict-scalper/nprd')
  level?: number; // Depth level in the hierarchy
}

export interface NavigationState {
  currentView: string;
  currentViewName: string;
  breadcrumbs: Breadcrumb[];
  currentFolder: string | null;
  currentFolderPath: string[]; // Array of folder names representing the path
  subPage: string | null;
  viewMode: 'grid' | 'list';
}

const initialState: NavigationState = {
  currentView: 'ea',
  currentViewName: 'EA Management',
  breadcrumbs: [],
  currentFolder: null,
  currentFolderPath: [],
  subPage: null,
  viewMode: 'grid'
};

// Helper function to format folder names (convert id-slug to Folder Name)
function formatFolderName(name: string): string {
  return name
    .split('-')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

// Helper function to parse folder path and build breadcrumbs
function buildFolderBreadcrumbs(folderId: string, folderName: string, basePath: Breadcrumb[] = []): Breadcrumb[] {
  const breadcrumbs = [...basePath];

  // If folderId contains /, it's a nested path
  if (folderId.includes('/')) {
    const pathParts = folderId.split('/').filter(p => p);

    // Build breadcrumbs for each level in the path
    let currentPath = '';
    pathParts.forEach((part, index) => {
      currentPath += (currentPath ? '/' : '') + part;
      breadcrumbs.push({
        id: currentPath,
        label: formatFolderName(part),
        name: formatFolderName(part),
        type: 'folder',
        fullPath: currentPath,
        level: index + 1
      });
    });
  } else {
    // Simple folder (not nested)
    breadcrumbs.push({
      id: folderId,
      label: folderName,
      name: folderName,
      type: 'folder',
      level: breadcrumbs.length
    });
  }

  return breadcrumbs;
}

function createNavigationStore() {
  const { subscribe, set, update } = writable<NavigationState>(initialState);

  return {
    subscribe,

    navigateToView: (viewId: string, viewName: string) => {
      update(state => ({
        ...state,
        currentView: viewId,
        currentViewName: viewName,
        breadcrumbs: [{ id: viewId, label: viewName, name: viewName, type: 'view', level: 0 }],
        currentFolder: null,
        currentFolderPath: [],
        subPage: null
      }));
    },

    navigateToFolder: (folderId: string, folderName: string, parentPath?: string) => {
      update(state => {
        // Start with view breadcrumb
        const viewBreadcrumb = { id: state.currentView, label: state.currentViewName, name: state.currentViewName, type: 'view' as const, level: 0 };

        // Build the full breadcrumb path
        let newBreadcrumbs: Breadcrumb[];

        if (parentPath) {
          // If we have a parent path, build from it
          const parentParts = parentPath.split('/').filter(p => p);
          const parentBreadcrumbs = parentParts.map((part, index) => ({
            id: parentParts.slice(0, index + 1).join('/'),
            label: formatFolderName(part),
            name: formatFolderName(part),
            type: 'folder' as const,
            fullPath: parentParts.slice(0, index + 1).join('/'),
            level: index + 1
          }));

          newBreadcrumbs = [viewBreadcrumb, ...parentBreadcrumbs];
        } else {
          newBreadcrumbs = [viewBreadcrumb];
        }

        // Add the current folder
        newBreadcrumbs = buildFolderBreadcrumbs(folderId, folderName, newBreadcrumbs);

        // Build the folder path array
        const folderPath = newBreadcrumbs
          .filter(b => b.type === 'folder')
          .map(b => b.label || b.name || '');

        return {
          ...state,
          breadcrumbs: newBreadcrumbs,
          currentFolder: folderId,
          currentFolderPath: folderPath,
          subPage: null
        };
      });
    },

    navigateToStrategy: (strategyId: string, strategyName: string) => {
      update(state => ({
        ...state,
        currentView: 'ea',
        currentViewName: 'EA Management',
        breadcrumbs: [
          { id: 'ea', label: 'EA Management', name: 'EA Management', type: 'view', level: 0 },
          { id: strategyId, label: strategyName, name: strategyName, type: 'folder', level: 1 }
        ],
        currentFolder: strategyId,
        currentFolderPath: [strategyName],
        subPage: null
      }));
    },

    navigateToSubPage: (subPageId: string, subPageName: string) => {
      update(state => ({
        ...state,
        breadcrumbs: [...state.breadcrumbs, { id: subPageId, label: subPageName, name: subPageName, type: 'subpage', level: state.breadcrumbs.length }],
        subPage: subPageId
      }));
    },

    navigateToBreadcrumb: (index: number) => {
      update(state => {
        const newBreadcrumbs = state.breadcrumbs.slice(0, index + 1);
        const clickedBreadcrumb = newBreadcrumbs[index];
        
        return {
          ...state,
          breadcrumbs: newBreadcrumbs,
          currentFolder: index === 0 ? null : clickedBreadcrumb.id,
          currentFolderPath: newBreadcrumbs
            .slice(1, index + 1)
            .filter(b => b.type === 'folder')
            .map(b => b.label || b.name || ''),
          subPage: null
        };
      });
    },

    resetNavigation: () => {
      update(state => ({
        ...state,
        breadcrumbs: [{ id: state.currentView, label: state.currentViewName, name: state.currentViewName, type: 'view', level: 0 }],
        currentFolder: null,
        currentFolderPath: [],
        subPage: null
      }));
    },

    navigateToPath: (path: string) => {
      update(state => {
        // Find the breadcrumb with the matching path and navigate to it
        const index = state.breadcrumbs.findIndex(b => b.fullPath === path || b.id === path);
        if (index === -1) return state;
        
        const newBreadcrumbs = state.breadcrumbs.slice(0, index + 1);
        const clickedBreadcrumb = newBreadcrumbs[index];
        
        return {
          ...state,
          breadcrumbs: newBreadcrumbs,
          currentFolder: index === 0 ? null : clickedBreadcrumb.id,
          currentFolderPath: newBreadcrumbs
            .slice(1, index + 1)
            .filter(b => b.type === 'folder')
            .map(b => b.label || b.name || ''),
          subPage: null
        };
      });
    },

    setViewMode: (mode: 'grid' | 'list') => {
      update(state => ({ ...state, viewMode: mode }));
    }
  };
}

export const navigationStore = createNavigationStore();
