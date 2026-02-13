<script lang="ts">
  import { createEventDispatcher, onMount, afterUpdate } from 'svelte';
  import hljs from 'highlight.js';
  import 'highlight.js/styles/github-dark.css';
  import {
    BookOpen, Boxes, Bot, TestTube, PlayCircle, Settings,
    ChevronRight, ChevronDown, File, Folder, FolderOpen,
    Upload, Search, RefreshCw, Plus, Database, BarChart3, Tag,
    Wallet, TrendingUp, Activity, Users, Edit3, Save, X,
    Home, ArrowLeft, Grid, List, Play, Pause, AlertTriangle,
    Router, Zap, Clock, Link as LinkIcon, FileText,
    MonitorPlay, BarChart2, PieChart, ExternalLink, Target, Loader,
    Library, ShieldAlert, Table, Newspaper, DollarSign
  } from 'lucide-svelte';
  import CodeEditor from './CodeEditor.svelte';

  import MonteCarloChart from './charts/MonteCarloChart.svelte';
  import SettingsView from './SettingsView.svelte';
  import TRDEditor from './TRDEditor.svelte';
  import StrategyRouterView from './StrategyRouterView.svelte';
  import TradeJournalView from './TradeJournalView.svelte';
  import SharedAssetsView from './SharedAssetsView.svelte';
  import KillSwitchView from './KillSwitchView.svelte';
  import DatabaseView from './DatabaseView.svelte';
  import NewsView from './NewsView.svelte';
  import LiveTradingView from './LiveTradingView.svelte';
  import ArticleViewer from './ArticleViewer.svelte';
  import PaperTradingPanel from './PaperTradingPanel.svelte';
  import { navigationStore } from '../stores/navigationStore';

  const dispatch = createEventDispatcher();

  export let activeView = 'ea';
  export let openFiles: Array<{id: string, name: string, content?: string, type?: string, path?: string}> = [];
  export let activeTabId = '';

  // Subscribe to navigation store
  let breadcrumbs = $navigationStore.breadcrumbs;
  let currentFolder = $navigationStore.currentFolder;
  let subPage = $navigationStore.subPage;
  let viewMode = $navigationStore.viewMode;

  // Local state (not navigation-related)
  let editingFileId: string | null = null;
  let editContent = '';
  let searchQuery = '';
  
  // NPRD modal state
  let nprdUrl = '';
  let nprdName = '';
  let nprdQueue: Array<{id: string, name: string, status: string, progress: number}> = [];
  
  // File Upload state
  let uploadModalOpen = false;
  let uploadDragOver = false;
  let uploadingFiles: Array<{name: string, progress: number, status: 'pending' | 'uploading' | 'done' | 'error'}> = [];
  let uploadCategory = 'books'; // 'books', 'articles', 'notes'
  let uploadType: 'book' | 'article' | 'note' = 'article';
  let uploadMetadata = {
    title: '',
    author: '',
    category: '',
    url: '',
    content: ''
  };
  let viewingArticle: any = null; // For article viewer modal
  let relatedArticles: Array<any> = []; // Related articles for the viewer
  
  // Article editor state
  let editMode = false;
  let articleEditContent = '';
  let previewContent = '';
  
  // Category filtering
  let selectedCategory = 'all';
  let categoryTree = {
    'expert_advisors': { count: 0, label: 'Expert Advisors' },
    'integration': { count: 0, label: 'Integration' },
    'trading': { count: 0, label: 'Trading Strategies' },
    'trading_systems': { count: 0, label: 'Trading Systems' }
  };
  
  // Backtest History (Sprint 6)
  let backtestHistory: Array<any> = [];
  
  // Mock EA files for Windows Explorer navigation
  let mockNprdFiles = [
    { id: 'nprd1', name: 'ICT_Scalper_NPRD.txt', size: '2.4 KB', path: '/ea/ict-scaler/nprd' },
    { id: 'nprd2', name: 'SMC_Reversal_NPRD.txt', size: '1.8 KB', path: '/ea/smc-reversal/nprd' },
    { id: 'nprd3', name: 'Order_Block_Hunter_NPRD.txt', size: '3.1 KB', path: '/ea/order-block/nprd' }
  ];
  
  let mockTrdFiles = [
    { id: 'trd1', name: 'ICT_Scalper_TRD.txt', size: '4.2 KB', path: '/ea/ict-scaler/trd' },
    { id: 'trd2', name: 'SMC_Reversal_TRD.txt', size: '3.7 KB', path: '/ea/smc-reversal/trd' },
    { id: 'trd3', name: 'Order_Block_Hunter_TRD.txt', size: '5.1 KB', path: '/ea/order-block/trd' }
  ];
  
  let mockEaFiles = [
    { id: 'ea1', name: 'ICT_Scalper.mq5', size: '12.8 KB', path: '/ea/ict-scaler/ICT_Scalper.mq5' },
    { id: 'ea2', name: 'SMC_Reversal.mq5', size: '15.3 KB', path: '/ea/smc-reversal/SMC_Reversal.mq5' },
    { id: 'ea3', name: 'Order_Block_Hunter.mq5', size: '18.7 KB', path: '/ea/order-block/Order_Block_Hunter.mq5' },
    { id: 'ea4', name: 'ICT_Scalper.mqh', size: '8.2 KB', path: '/ea/ict-scaler/ICT_Scalper.mqh' }
  ];
  
  let mockBacktestFiles = [
    { id: 'bt1', name: 'ICT_Scalper_Backtest_2024.txt', size: '6.4 KB', path: '/ea/ict-scaler/backtest' },
    { id: 'bt2', name: 'SMC_Reversal_Backtest_2024.txt', size: '7.1 KB', path: '/ea/smc-reversal/backtest' },
    { id: 'bt3', name: 'Order_Block_Hunter_Backtest_2024.txt', size: '8.9 KB', path: '/ea/order-block/backtest' }
  ];
  
  // Load backtests when view activates
  $: if ($navigationStore.currentView === 'backtest' && !$navigationStore.subPage) {
      loadBacktests();
  }

  // Subscribe to navigation store
  $: breadcrumbs = $navigationStore.breadcrumbs;
  $: currentFolder = $navigationStore.currentFolder;
  $: subPage = $navigationStore.subPage;
  $: viewMode = $navigationStore.viewMode;
  $: activeView = $navigationStore.currentView;

  async function loadBacktests() {
      try {
          const res = await fetch('http://localhost:8000/api/analytics/backtests?limit=20');
          if (res.ok) {
              backtestHistory = await res.json();
          }
      } catch (e) {
          console.error("Failed to load backtest history", e);
      }
  }

  async function loadRunDetails(runId: string) {
      try {
          const res = await fetch(`http://localhost:8000/api/analytics/trades?run_id=${runId}`);
          if (res.ok) {
              const trades = await res.json();
              // Calculate simple returns for MC visualization
              const returns = trades.map((t: any) => t.profit);
              
              if (returns.length === 0) return;

              const sorted = [...returns].sort((a: number, b: number) => a - b);
              const sum = returns.reduce((a: number, b: number) => a + b, 0);
              const mean = sum / returns.length;
              const median = sorted[Math.floor(sorted.length / 2)];
              const var95 = sorted[Math.floor(sorted.length * 0.05)]; // 5th percentile

              backtestResults = {
                  returns: returns,
                  confidence: var95,
                  worstCase: sorted[0],
                  bestCase: sorted[sorted.length - 1],
                  mean: mean,
                  median: median
              };
              // Switch to MC view
              navigationStore.navigateToSubPage('montecarlo', 'Monte Carlo');
          }
      } catch (e) {
          console.error("Failed to load run details", e);
      }
  }

  
  // Enhanced file upload handler with type support
  async function handleFileUpload(files: FileList | null) {
    if (!files || files.length === 0) return;

    // Validate required fields based on type
    if (uploadType === 'note') {
      // Notes should use handleNoteSubmit, not file upload
      alert('Please use the note form to create notes without files');
      return;
    }

    for (const file of Array.from(files)) {
      // Add to uploading list
      uploadingFiles = [...uploadingFiles, { name: file.name, progress: 0, status: 'pending' }];

      const formData = new FormData();
      formData.append('file', file);
      formData.append('category', uploadType + 's'); // books, articles, notes

      // Add metadata based on type
      if (uploadType === 'book') {
        formData.append('index', 'true');
        if (uploadMetadata.title) formData.append('title', uploadMetadata.title);
        if (uploadMetadata.author) formData.append('author', uploadMetadata.author);
        if (uploadMetadata.category) formData.append('subcategory', uploadMetadata.category);
      } else if (uploadType === 'article') {
        if (uploadMetadata.title) formData.append('title', uploadMetadata.title);
        if (uploadMetadata.author) formData.append('author', uploadMetadata.author);
        if (uploadMetadata.url) formData.append('url', uploadMetadata.url);
        if (uploadMetadata.category) formData.append('subcategory', uploadMetadata.category);
      }

      try {
        // Update status to uploading
        uploadingFiles = uploadingFiles.map(f =>
          f.name === file.name ? { ...f, status: 'uploading', progress: 30 } : f
        );

        // Single unified endpoint
        const res = await fetch('http://localhost:8000/api/ide/knowledge/upload', {
          method: 'POST',
          body: formData
        });

        uploadingFiles = uploadingFiles.map(f =>
          f.name === file.name ? { ...f, status: 'uploading', progress: 70 } : f
        );

        if (res.ok) {
          uploadingFiles = uploadingFiles.map(f =>
            f.name === file.name ? { ...f, status: 'done', progress: 100 } : f
          );
          // Refresh data
          await loadData();
          // Clear metadata after successful upload
          uploadMetadata = { title: '', author: '', category: '', url: '', content: '' };
        } else {
          const errorText = await res.text();
          console.error('Upload error:', errorText);
          uploadingFiles = uploadingFiles.map(f =>
            f.name === file.name ? { ...f, status: 'error' } : f
          );
        }
      } catch (e) {
        uploadingFiles = uploadingFiles.map(f =>
          f.name === file.name ? { ...f, status: 'error' } : f
        );
        console.error('Upload failed:', e);
      }
    }

    // Clear completed uploads after delay
    setTimeout(() => {
      uploadingFiles = uploadingFiles.filter(f => f.status !== 'done');
    }, 3000);
  }

  // Handle note submission (content-based, no file)
  async function handleNoteSubmit() {
    if (!uploadMetadata.title || !uploadMetadata.content) {
      alert('Please provide both title and content for the note');
      return;
    }

    uploadingFiles = [...uploadingFiles, {
      name: uploadMetadata.title,
      progress: 0,
      status: 'pending'
    }];

    try {
      uploadingFiles = uploadingFiles.map(f =>
        f.name === uploadMetadata.title ? { ...f, status: 'uploading', progress: 50 } : f
      );

      const formData = new FormData();
      formData.append('title', uploadMetadata.title);
      formData.append('content', uploadMetadata.content);
      formData.append('category', 'notes');

      const res = await fetch('http://localhost:8000/api/ide/knowledge/upload/note', {
        method: 'POST',
        body: formData
      });

      if (res.ok) {
        uploadingFiles = uploadingFiles.map(f =>
          f.name === uploadMetadata.title ? { ...f, status: 'done', progress: 100 } : f
        );
        await loadData();
        uploadMetadata = { title: '', author: '', category: '', url: '', content: '' };
      } else {
        const errorText = await res.text();
        console.error('Note creation error:', errorText);
        uploadingFiles = uploadingFiles.map(f =>
          f.name === uploadMetadata.title ? { ...f, status: 'error' } : f
        );
      }
    } catch (e) {
      uploadingFiles = uploadingFiles.map(f =>
        f.name === uploadMetadata.title ? { ...f, status: 'error' } : f
      );
      console.error('Note creation failed:', e);
    }

    setTimeout(() => {
      uploadingFiles = uploadingFiles.filter(f => f.status !== 'done');
    }, 3000);
  }

  // Open article viewer
  async function openArticleViewer(article: any) {
    viewingArticle = { ...article, loading: true };

    // Fetch article content from API
    try {
      const contentRes = await fetch(`http://localhost:8000/api/knowledge/${encodeURIComponent(article.id || article.path)}/content`);
      if (contentRes.ok) {
        const contentData = await contentRes.json();
        viewingArticle = { ...viewingArticle, content: contentData.content, loading: false };
      } else {
        viewingArticle = { ...viewingArticle, content: null, loading: false, error: 'Content not available' };
      }
    } catch (e) {
      console.error('Failed to load article content:', e);
      viewingArticle = { ...viewingArticle, content: null, loading: false, error: 'Failed to load content' };
    }

    // Load related articles based on category/tags
    try {
      const res = await fetch(`http://localhost:8000/api/knowledge/related?id=${article.id}&limit=5`);
      if (res.ok) {
        relatedArticles = await res.json();
      }
    } catch (e) {
      console.error('Failed to load related articles:', e);
    }
  }

  // Close article viewer
  function closeArticleViewer() {
    viewingArticle = null;
    relatedArticles = [];
  }

  // Copy code snippet to clipboard
  function copyCode(code: string, event: Event) {
    event.stopPropagation();
    navigator.clipboard.writeText(code).then(() => {
      // Could show a toast notification here
      alert('Code copied to clipboard!');
    });
  }
  
  function handleDrop(e: DragEvent) {
    e.preventDefault();
    uploadDragOver = false;
    if (e.dataTransfer?.files) {
      handleFileUpload(e.dataTransfer.files);
    }
  }
  
  // Data structures - would be loaded from API
  let articles: Array<any> = [];
  let filteredArticles: Array<any> = [];
  let strategies: Array<any> = [];
  let bots: Array<any> = [];
  let systemStatus: any = { regime: 'Unknown', kelly: 0, pnl: 0 };
  
  // Editor workspace state
  let editorFiles: Array<{id: string, name: string, content: string, language: string, path: string, source?: string, category?: string}> = [];
  let activeEditorFile: any = null;
  let editorContent = '';
  let editorLanguage = 'plaintext';
  let editorFilename = 'untitled';
  
  // Breadcrumb navigation for editor
  let editorBreadcrumb: Array<{name: string, path?: string}> = [];
  
  onMount(async () => {
    await loadData();
    // Initialize navigation with the current view
    navigationStore.navigateToView(activeView, viewConfig[activeView]?.title || 'Explorer');
  });
  
  async function loadData() {
    try {
      // Load articles for Knowledge Hub
      const articlesRes = await fetch('http://localhost:8000/api/knowledge');
      if (articlesRes.ok) {
        articles = await articlesRes.json();
        console.log('[MainContent] Loaded articles:', articles.length, articles);
        updateCategoryCounts();
        applyArticleFilter();
        console.log('[MainContent] After filter, filteredArticles:', filteredArticles.length);
      } else {
        console.error('[MainContent] Failed to load articles, status:', articlesRes.status);
      }
      
      // Load strategies for EA Management  
      const strategiesRes = await fetch('http://localhost:8000/api/strategies');
      if (strategiesRes.ok) strategies = await strategiesRes.json();
      
      // Load trading status
      const statusRes = await fetch('http://localhost:8000/api/trading/status');
      if (statusRes.ok) systemStatus = await statusRes.json();
      
      // Load bots
      const botsRes = await fetch('http://localhost:8000/api/trading/bots');
      if (botsRes.ok) bots = await botsRes.json();
    } catch (e) {
      console.error('Failed to load data:', e);
    }
  }
  
  // View configurations
  // Removed duplicate views: backtest-results, shared-assets, kill-switch, database-view, news
  // These are now accessible as sub-pages within their parent sections
  const viewConfig: Record<string, any> = {
    knowledge: {
      title: 'Knowledge Hub',
      icon: BookOpen,
      description: 'Articles, books, and indexed resources',
      hasUpload: true,
      categories: [
        { id: 'articles', name: 'Articles', count: 1806 },
        { id: 'books', name: 'Books', count: 12 },
        { id: 'notes', name: 'Strategy Notes', count: 24 }
      ]
    },
    assets: {
      title: 'Shared Assets & Database',
      icon: Boxes,
      description: 'Reusable components, indicators, and database visualization',
      tabs: [
        { id: 'indicators', name: 'Indicators', icon: BarChart3 },
        { id: 'libraries', name: 'Libraries', icon: Folder },
        { id: 'templates', name: 'Templates', icon: FileText },
        { id: 'database', name: 'Database', icon: Database, isPage: true }
      ]
    },
    ea: {
      title: 'EA Management',
      icon: Bot,
      description: 'Strategy folders with NPRD, TRD, EA code, and backtests'
    },
    backtest: {
      title: 'Backtests',
      icon: TestTube,
      description: 'Run and view backtest results with Monte Carlo analysis',
      tabs: [
        { id: 'results', name: 'Results', icon: BarChart2 },
        { id: 'montecarlo', name: 'Monte Carlo', icon: PieChart },
        { id: 'forward', name: 'Forward Test', icon: TrendingUp }
      ]
    },
    live: {
      title: 'Live Trading',
      icon: PlayCircle,
      description: 'Active bots, kill switch, news & kill zones, and trade journal',
      subTabs: [
        { id: 'dashboard', name: 'Dashboard', icon: Activity },
        { id: 'bots', name: 'Active Bots', icon: PlayCircle },
        { id: 'kill-switch', name: 'Kill Switch', icon: ShieldAlert },
        { id: 'news', name: 'News & Kill Zones', icon: Newspaper },
        { id: 'journal', name: 'Trade Journal', icon: FileText }
      ]
    },
    'paper-trading': {
      title: 'Paper Trading',
      icon: MonitorPlay,
      description: 'Backtest agents and paper trading simulations'
    },
    router: {
      title: 'Strategy Router',
      icon: Router,
      description: 'Auction-based signal selection and routing'
    },
    journal: {
      title: 'Trade Journal',
      icon: FileText,
      description: 'Complete trade history with context'
    },
    editor: {
      title: 'Editor Workspace',
      icon: Edit3,
      description: 'Code editor for articles, indicators, and agent workflows'
    },
    settings: {
      title: 'System Settings',
      icon: Settings,
      description: 'Configuration, API keys, and system preferences'
    }
  };
  
  // Mock strategy data for EA cards
  const strategyCards = [
    { id: 'ict-v2', name: 'ICT Scalper v2', status: 'primal', lastUpdated: '2 hours ago', backtests: 4, tag: 'Primal' },
    { id: 'smc-rev', name: 'SMC Reversal', status: 'ready', lastUpdated: '1 day ago', backtests: 2, tag: 'Ready' },
    { id: 'order-block', name: 'Order Block Hunter', status: 'pending', lastUpdated: '3 days ago', backtests: 0, tag: 'Pending' }
  ];
  
  // Mock Assets Data
  const mockAssets: Record<string, Array<{name: string, type: 'file'|'folder', size?: string}>> = {
    'indicators': [
      { name: 'ATR_Filter.mqh', type: 'file', size: '4KB' },
      { name: 'RSI_Divergence.mqh', type: 'file', size: '12KB' },
      { name: 'Volume_Profile.mqh', type: 'file', size: '24KB' },
    ],
    'libraries': [
      { name: 'RiskManager.mqh', type: 'file', size: '15KB' },
      { name: 'OrderManager.mqh', type: 'file', size: '18KB' },
      { name: 'TimeFilters.mqh', type: 'file', size: '6KB' },
    ],
    'templates': [
      { name: 'Scalping_M5.tpl', type: 'file', size: '2KB' },
      { name: 'Swing_H4.tpl', type: 'file', size: '2KB' },
    ]
  };
  
  function getStatusColor(status: string): string {
    const colors: Record<string, string> = {
      primal: '#10b981', ready: '#3b82f6', pending: '#f59e0b',
      quarantined: '#ef4444', paused: '#6b7280', processing: '#8b5cf6'
    };
    return colors[status] || '#6b7280';
  }
  
  function navigateTo(folderId: string, folderName: string) {
    navigationStore.navigateToFolder(folderId, folderName);
  }
  
  // Editor workspace functions
  async function openInEditor(item: any) {
    console.log('[Editor] Opening item in editor:', item);
    
    try {
      // Fetch content from API
      const contentRes = await fetch(`http://localhost:8000/api/knowledge/${encodeURIComponent(item.id || item.path)}/content`);
      let content = '';
      
      if (contentRes.ok) {
        const data = await contentRes.json();
        content = data.content || '';
      } else {
        // Try alternative endpoint for assets
        try {
          const assetRes = await fetch(`http://localhost:8000/api/assets/${item.id}/content`);
          if (assetRes.ok) {
            const assetData = await assetRes.json();
            content = assetData.content || '';
          }
        } catch {
          content = `// Could not load content for ${item.name}`;
        }
      }
      
      // Determine language from filename
      const filename = item.name || item.filename || 'untitled';
      const language = detectLanguage(filename);
      
      // Create breadcrumb navigation
      const category = item.category?.split('/')[1]?.replace('_', ' ') || 'general';
      const source = item.category?.split('/')[0] || 'unknown';
      
      // Map source to proper display names
      const sourceMap: Record<string, string> = {
        'scraped_articles': 'Knowledge',
        'assets': 'Assets',
        'ea': 'EA Management'
      };
      
      editorBreadcrumb = [
        { name: sourceMap[source] || source.charAt(0).toUpperCase() + source.slice(1) },
        { name: category.charAt(0).toUpperCase() + category.slice(1) },
        { name: filename }
      ];
      
      // Create or update file in editor
      const existingFile = editorFiles.find(f => f.id === item.id);
      if (existingFile) {
        existingFile.content = content;
        existingFile.language = language;
      } else {
        editorFiles = [...editorFiles, {
          id: item.id,
          name: filename,
          content: content,
          language: language,
          path: item.path || item.id,
          source: source,
          category: category
        }];
      }
      
      // Set as active file
      activeEditorFile = editorFiles.find(f => f.id === item.id);
      editorContent = content;
      editorLanguage = language;
      editorFilename = filename;
      
      // Switch to editor view
      dispatch('viewChange', { view: 'editor' });
      
    } catch (error) {
      console.error('[Editor] Failed to open file:', error);
      alert('Failed to open file in editor');
    }
  }
  
  function detectLanguage(filename: string): string {
    const ext = filename.split('.').pop()?.toLowerCase();
    const langMap: Record<string, string> = {
      'js': 'javascript',
      'ts': 'typescript',
      'py': 'python',
      'mql5': 'cpp',
      'mq5': 'cpp',
      'mqh': 'cpp',
      'cpp': 'cpp',
      'c': 'cpp',
      'h': 'cpp',
      'java': 'java',
      'cs': 'csharp',
      'php': 'php',
      'rb': 'ruby',
      'go': 'go',
      'rs': 'rust',
      'sql': 'sql',
      'json': 'json',
      'xml': 'xml',
      'html': 'html',
      'css': 'css',
      'scss': 'scss',
      'less': 'less',
      'md': 'markdown',
      'yaml': 'yaml',
      'yml': 'yaml',
      'sh': 'bash',
      'bash': 'bash',
      'zsh': 'bash'
    };
    return langMap[ext || ''] || 'plaintext';
  }
  
  function saveEditorFile() {
    if (!activeEditorFile) return;
    
    // Update file content
    activeEditorFile.content = editorContent;
    const fileIndex = editorFiles.findIndex(f => f.id === activeEditorFile.id);
    if (fileIndex !== -1) {
      editorFiles[fileIndex] = { ...activeEditorFile };
    }
    
    // TODO: Save to backend API
    console.log('[Editor] Saving file:', activeEditorFile.name);
    alert('File saved (backend save not implemented yet)');
  }
  
  function closeEditorFile(fileId: string) {
    editorFiles = editorFiles.filter(f => f.id !== fileId);
    if (activeEditorFile?.id === fileId) {
      activeEditorFile = editorFiles.length > 0 ? editorFiles[0] : null;
      if (activeEditorFile) {
        editorContent = activeEditorFile.content;
        editorLanguage = activeEditorFile.language;
        editorFilename = activeEditorFile.name;
      } else {
        editorContent = '';
        editorLanguage = 'plaintext';
        editorFilename = 'untitled';
      }
    }
  }

  function navigateBack() {
    if (breadcrumbs.length > 1) {
      navigationStore.navigateToBreadcrumb(breadcrumbs.length - 2);
    } else {
      navigationStore.resetNavigation();
    }
  }

  function navigateToBreadcrumb(index: number) {
    navigationStore.navigateToBreadcrumb(index);
  }

  function resetNavigation() {
    // Reset to current view root
    navigationStore.navigateToView(activeView, viewConfig[activeView]?.title || 'Explorer');
  }

  function openSubPage(page: string) {
    navigationStore.navigateToSubPage(page, page.charAt(0).toUpperCase() + page.slice(1).replace(/-/g, ' '));
  }
  
  function openFile(file: any) {
    const existingFile = openFiles.find(f => f.id === file.id);
    if (!existingFile) {
      dispatch('openFile', { ...file, content: `// Content of ${file.name}\n// Loading from backend...` });
    }
    activeTabId = file.id;
  }
  
  function startEditing(file: any) {
    editingFileId = file.id;
    editContent = file.content || '';
  }
  
  function saveFile() {
    // Save to backend
    editingFileId = null;
    editContent = '';
  }
  
  function cancelEdit() {
    editingFileId = null;
    editContent = '';
  }
  
  async function submitNPRD(): Promise<void> {
    if (!nprdUrl || !nprdName) return;
    
    try {
      const res = await fetch('http://localhost:8000/api/nprd/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: nprdUrl, strategy_name: nprdName })
      });
      const data = await res.json();
      nprdQueue = [...nprdQueue, { id: data.job_id, name: nprdName, status: 'processing', progress: 0 }];
      nprdUrl = '';
      nprdName = '';
      navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1);
    } catch (e) {
      console.error('NPRD submit failed:', e);
    }
  }
  
  async function handleTagChange(botId: string, newTag: string) {
    await fetch('http://localhost:8000/api/trading/bots/control', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bot_id: botId, action: newTag === 'quarantined' ? 'quarantine' : newTag === 'paused' ? 'pause' : 'resume' })
    });
  }
  
  async function triggerKillSwitch() {
    if (confirm('⚠️ KILL SWITCH: This will stop ALL trading immediately. Are you sure?')) {
      await fetch('http://localhost:8000/api/trading/kill', { method: 'POST' });
    }
  }
  
  // Backtest State
  let backtestRunning = false;
  let backtestConfig = { strategy: 'ict-v2', period: '1M', monteCarlo: true };
  let backtestResults: any = null;
  
  async function runBacktest() {
    backtestRunning = true;
    // Simulate backend delay
    setTimeout(() => {
      backtestRunning = false;
      navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1);

      // Navigate to Monte Carlo view
      navigationStore.navigateToSubPage('montecarlo', 'Monte Carlo');

      // Update data with "fresh" results
      backtestResults = {
        returns: Array.from({length: 1000}, () => (Math.random() - 0.45 + (Math.random() * 0.1)) * 50),
        confidence: 12.5 + Math.random() * 2,
        worstCase: -18.2 + Math.random() * 2,
        bestCase: 42.8 + Math.random() * 5,
        median: 15.2,
        mean: 14.8
      };
    }, 2500);
  }

  // Watch for view changes to reset navigation
  $: if (activeView && activeView !== $navigationStore.currentView) {
    navigationStore.navigateToView(activeView, viewConfig[activeView]?.title || 'Unknown');
  }

  // Settings visibility state
  let settingsVisible = false;
  let settingsViewComponent: SettingsView;

  // Listen for settings trigger
  $: if (activeView === 'settings' && settingsViewComponent) {
    settingsViewComponent.show();
  }

  function closeSettings() {
    settingsVisible = false;
    // Navigate back to EA view
    navigationStore.navigateToView('ea', 'EA Management');
    // Emit event to parent to update activeView
    dispatch('viewChange', { view: 'ea' });
  }

  // Enhanced markdown renderer with syntax highlighting
  function renderMarkdown(content: string): string {
    if (!content) return '';

    let html = content
      // Escape HTML
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    
    // Code blocks with syntax highlighting
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'plaintext';
      const langDisplay = language === 'mql5' ? 'MQL5' : language.toUpperCase();
      let highlighted;
      
      try {
        // Map MQL5 to C++ for highlighting
        const hlLang = language === 'mql5' ? 'cpp' : language;
        highlighted = hljs.highlight(code.trim(), { language: hlLang }).value;
      } catch {
        highlighted = code.trim();
      }
      
      return `<pre class="code-block"><div class="code-header"><span class="code-lang">${langDisplay}</span><button class="copy-btn" onclick="navigator.clipboard.writeText(decodeURIComponent('${encodeURIComponent(code.trim())}'))">Copy</button></div><code class="hljs language-${language}">${highlighted}</code></pre>`;
    });
    
    html = html
      // Inline code
      .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
      // Headers
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/^## (.*$)/gm, '<h2>$1</h2>')
      .replace(/^# (.*$)/gm, '<h1>$1</h1>')
      // Bold and italic
      .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      // Links
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
      // Lists
      .replace(/^\* (.+)$/gm, '<li>$1</li>')
      .replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
      // Paragraphs
      .replace(/\n\n/g, '</p><p>')
      // Line breaks
      .replace(/\n/g, '<br>');

    // Wrap in paragraph if not starting with block element
    if (!html.startsWith('<')) {
      html = '<p>' + html + '</p>';
    }

    // Fix list wrapping
    html = html.replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>');
    html = html.replace(/<\/ul><ul>/g, '');

    return html;
  }
  
  // Update category counts based on loaded articles
  function updateCategoryCounts() {
    const counts = {
      'expert_advisors': 0,
      'integration': 0,
      'trading': 0,
      'trading_systems': 0
    };
    
    articles.forEach(article => {
      const cat = article.category?.split('/')[1];
      if (cat && counts.hasOwnProperty(cat)) {
        counts[cat]++;
      }
    });
    
    Object.keys(categoryTree).forEach(key => {
      categoryTree[key].count = counts[key] || 0;
    });
  }
  
  // Apply category filter
  function applyArticleFilter() {
    console.log('[MainContent] Applying filter, selectedCategory:', selectedCategory, 'articles:', articles.length);
    if (selectedCategory === 'all') {
      filteredArticles = articles;
    } else {
      filteredArticles = articles.filter(a => a.category?.includes(selectedCategory));
    }
    console.log('[MainContent] Filtered to:', filteredArticles.length, 'articles');
  }
  
  // Toggle edit mode
  function toggleEditMode() {
    editMode = !editMode;
    if (editMode && viewingArticle) {
      articleEditContent = viewingArticle.content || '';
      updatePreview();
    }
  }
  
  // Update preview in real-time
  function updatePreview() {
    previewContent = renderMarkdown(articleEditContent);
  }
  
  // Save article changes
  async function saveArticle() {
    if (!viewingArticle) return;
    
    try {
      const res = await fetch(`http://localhost:8000/api/knowledge/${viewingArticle.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content: articleEditContent })
      });
      
      if (res.ok) {
        viewingArticle.content = articleEditContent;
        editMode = false;
        alert('Article saved successfully');
      } else {
        alert('Failed to save article');
      }
    } catch (e) {
      alert('Failed to save: ' + e.message);
    }
  }
  
  // Apply syntax highlighting after DOM updates
  afterUpdate(() => {
    document.querySelectorAll('pre code.hljs').forEach((block) => {
      if (!block.dataset.highlighted) {
        hljs.highlightElement(block);
        block.dataset.highlighted = 'yes';
      }
    });
  });
  
  // Watch category changes
  $: if (selectedCategory) {
    applyArticleFilter();
  }
</script>

<main class="main-content">
  <!-- Tab bar -->
  <div class="tab-bar">
    <button class="tab" class:active={!activeTabId || activeTabId === ''} on:click={() => { activeTabId = ''; resetNavigation(); }}>
      <svelte:component this={viewConfig[activeView]?.icon || BookOpen} size={14} />
      <span>{viewConfig[activeView]?.title || 'Explorer'}</span>
    </button>
    
    {#each openFiles as file}
      <button class="tab" class:active={activeTabId === file.id} on:click={() => activeTabId = file.id}>
        <File size={14} />
        <span>{file.name}</span>
        <button class="tab-close" on:click|stopPropagation={() => dispatch('closeTab', file.id)}>×</button>
      </button>
    {/each}
  </div>
  
  <div class="content-area">
    {#if activeTabId && openFiles.find(f => f.id === activeTabId)}
      <!-- File Editor View -->
      {@const file = openFiles.find(f => f.id === activeTabId)}
      <div class="file-editor">
        <div class="editor-toolbar">
          <div class="breadcrumb">
            <Home size={14} />
            <ChevronRight size={12} />
            <span>{file?.name}</span>
          </div>
          <div class="editor-actions">
            {#if editingFileId === file?.id}
              <button class="action-btn save" on:click={saveFile}><Save size={14} /> Save</button>
              <button class="action-btn" on:click={cancelEdit}><X size={14} /> Cancel</button>
            {:else}
              <button class="action-btn" on:click={() => startEditing(file)}><Edit3 size={14} /> Edit</button>
            {/if}
          </div>
        </div>
        {#if editingFileId === file?.id}
          <textarea class="file-textarea" bind:value={editContent}></textarea>
        {:else}
          <pre class="file-content">{file?.content || '// Loading...'}</pre>
        {/if}
      </div>
    {:else}
      <!-- View Content -->
      <div class="view-content">
        <!-- Header with breadcrumbs -->
        <div class="view-header">
          <div class="breadcrumb-nav">
            {#if breadcrumbs.length > 1 || subPage}
              <button class="back-btn" on:click={() => subPage ? navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1) : navigateBack()} title="Navigate back">
                <ArrowLeft size={16} />
              </button>
            {/if}
            {#each breadcrumbs as crumb, i}
              {#if i > 0}<ChevronRight size={14} />{/if}
              {#if i === breadcrumbs.length - 1 && !subPage}
                <span class="breadcrumb-current">{crumb.name}</span>
              {:else}
                <button class="breadcrumb-item" on:click={() => navigateToBreadcrumb(i)} title="Navigate to {crumb.name}">
                  {#if crumb.type === 'view'}
                    <svelte:component this={viewConfig[crumb.id]?.icon || BookOpen} size={16} />
                  {:else}
                    <Folder size={14} />
                  {/if}
                  <span>{crumb.name}</span>
                </button>
              {/if}
            {/each}
            {#if subPage}
              <ChevronRight size={14} />
              <span class="breadcrumb-current">{subPage}</span>
            {/if}
          </div>
          
          <div class="view-toolbar">
            <div class="search-box">
              <Search size={14} />
              <input type="text" placeholder="Search..." bind:value={searchQuery} />
            </div>
            <div class="view-toggle">
              <button class:active={viewMode === 'grid'} on:click={() => navigationStore.setViewMode('grid')}><Grid size={14} /></button>
              <button class:active={viewMode === 'list'} on:click={() => navigationStore.setViewMode('list')}><List size={14} /></button>
            </div>
            <button class="toolbar-btn" on:click={loadData}><RefreshCw size={14} /></button>
            {#if activeView === 'knowledge'}
              <button class="toolbar-btn primary" on:click={() => uploadModalOpen = true}><Upload size={14} /> Upload</button>
            {:else if activeView === 'ea'}
              <button class="toolbar-btn primary" on:click={() => openSubPage('nprd-modal')}>
                <Plus size={14} /> Process NPRD
              </button>
            {:else if activeView === 'backtest'}
              <button class="toolbar-btn primary" on:click={() => openSubPage('run-backtest')}>
                <Play size={14} /> Run Backtest
              </button>
            {/if}
          </div>
        </div>
        
        <!-- Sub-pages / Modals -->
        {#if subPage === 'nprd-modal'}
          <div class="modal-overlay">
            <div class="modal">
              <div class="modal-header">
                <h2>Process NPRD</h2>
                <button on:click={() => navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1)}><X size={20} /></button>
              </div>
              <div class="modal-body">
                <div class="form-group">
                  <label>YouTube URL or Playlist</label>
                  <input type="text" placeholder="https://youtube.com/watch?v=..." bind:value={nprdUrl} />
                </div>
                <div class="form-group">
                  <label>Strategy Name</label>
                  <input type="text" placeholder="ICT Scalper v3" bind:value={nprdName} />
                </div>
                {#if nprdQueue.length > 0}
                  <div class="queue-section">
                    <h4>Processing Queue</h4>
                    {#each nprdQueue as job}
                      <div class="queue-item">
                        <span>{job.name}</span>
                        <span class="status" style="color: {getStatusColor(job.status)}">{job.status}</span>
                        <div class="progress-bar"><div style="width: {job.progress}%"></div></div>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
              <div class="modal-footer">
                <button class="btn secondary" on:click={() => navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1)}>Cancel</button>
                <button class="btn primary" on:click={submitNPRD}>Start Processing</button>
              </div>
            </div>
          </div>
        
        {#if subPage === 'run-backtest'}
          <div class="modal-overlay" on:click|self={() => navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1)} role="button" tabindex="0" on:keydown={(e) => e.key === 'Enter' && navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1)}>
            <div class="modal">
              <div class="modal-header">
                <h2><Play size={20} /> Run Backtest</h2>
                <button on:click={() => navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1)}><X size={20} /></button>
              </div>
              <div class="modal-body">
                <div class="form-group">
                  <label>Strategy</label>
                  <select bind:value={backtestConfig.strategy}>
                    {#each strategyCards as strategy}
                      <option value={strategy.id}>{strategy.name}</option>
                    {/each}
                  </select>
                </div>
                <div class="form-group">
                  <label>Time Period</label>
                  <select bind:value={backtestConfig.period}>
                    <option value="1M">Last Month</option>
                    <option value="3M">Last 3 Months</option>
                    <option value="6M">Last 6 Months</option>
                    <option value="1Y">Last Year</option>
                    <option value="all">Max Available</option>
                  </select>
                </div>
                <div class="form-group checkbox">
                   <input type="checkbox" id="mc-sim" bind:checked={backtestConfig.monteCarlo} />
                   <label for="mc-sim">Run Monte Carlo Simulation (1000 runs)</label>
                </div>
                
                {#if backtestRunning}
                  <div class="progress-section">
                    <div class="spinner-row">
                      <Loader size={16} class="spinning" />
                      <span>Running simulation...</span>
                    </div>
                  </div>
                {/if}
              </div>
              <div class="modal-footer">
                <button class="btn secondary" on:click={() => navigationStore.navigateToBreadcrumb(breadcrumbs.length - 1)} disabled={backtestRunning}>Cancel</button>
                <button class="btn primary" on:click={runBacktest} disabled={backtestRunning}>
                  {#if backtestRunning}Running...{:else}Start Backtest{/if}
                </button>
              </div>
            </div>
          </div>
        {/if}
        
        {#if uploadModalOpen}
          <div class="modal-overlay" on:click|self={() => uploadModalOpen = false} role="button" tabindex="0" on:keydown={(e) => e.key === 'Enter' && (uploadModalOpen = false)}>
            <div class="modal upload-modal">
              <div class="modal-header">
                <h2><Upload size={20} /> Upload to Knowledge Hub</h2>
                <button on:click={() => uploadModalOpen = false}><X size={20} /></button>
              </div>
              <div class="modal-body">
                <!-- Upload Type Selector -->
                <div class="form-group">
                  <label for="upload-type-selector">Upload Type</label>
                  <div id="upload-type-selector" class="upload-type-selector" role="radiogroup">
                    <label class="type-option" class:selected={uploadType === 'article'}>
                      <input type="radio" bind:group={uploadType} value="article" />
                      <div class="type-content">
                        <FileText size={20} />
                        <div>
                          <span class="type-name">Article</span>
                          <span class="type-desc">Blog posts, research papers, tutorials</span>
                        </div>
                      </div>
                    </label>
                    <label class="type-option" class:selected={uploadType === 'book'}>
                      <input type="radio" bind:group={uploadType} value="book" />
                      <div class="type-content">
                        <Library size={20} />
                        <div>
                          <span class="type-name">Book (PDF)</span>
                          <span class="type-desc">PDF indexing with LangMem</span>
                        </div>
                      </div>
                    </label>
                    <label class="type-option" class:selected={uploadType === 'note'}>
                      <input type="radio" bind:group={uploadType} value="note" />
                      <div class="type-content">
                        <FileText size={20} />
                        <div>
                          <span class="type-name">Note</span>
                          <span class="type-desc">Quick notes without indexing</span>
                        </div>
                      </div>
                    </label>
                  </div>
                </div>

                <!-- Dynamic form fields based on type -->
                {#if uploadType === 'book'}
                  <div class="metadata-fields book-fields">
                    <div class="form-group">
                      <label for="book-title">Title <span class="optional">(optional)</span></label>
                      <input type="text" id="book-title" placeholder="Book title" bind:value={uploadMetadata.title} />
                    </div>
                    <div class="form-group">
                      <label for="book-author">Author <span class="optional">(optional)</span></label>
                      <input type="text" id="book-author" placeholder="Author name" bind:value={uploadMetadata.author} />
                    </div>
                    <div class="form-group">
                      <label for="book-category">Category <span class="optional">(optional)</span></label>
                      <select id="book-category" bind:value={uploadMetadata.category}>
                        <option value="">Select category...</option>
                        <option value="trading">Trading</option>
                        <option value="programming">Programming</option>
                        <option value="mathematics">Mathematics</option>
                        <option value="economics">Economics</option>
                        <option value="psychology">Psychology</option>
                        <option value="other">Other</option>
                      </select>
                    </div>
                  </div>
                {:else if uploadType === 'article'}
                  <div class="metadata-fields article-fields">
                    <div class="form-group">
                      <label>Title <span class="optional">(optional)</span></label>
                      <input type="text" placeholder="Article title" bind:value={uploadMetadata.title} />
                    </div>
                    <div class="form-group">
                      <label>Author <span class="optional">(optional)</span></label>
                      <input type="text" placeholder="Author name" bind:value={uploadMetadata.author} />
                    </div>
                    <div class="form-group">
                      <label>Source URL <span class="optional">(optional)</span></label>
                      <input type="url" placeholder="https://..." bind:value={uploadMetadata.url} />
                    </div>
                    <div class="form-group">
                      <label>Category <span class="optional">(optional)</span></label>
                      <select bind:value={uploadMetadata.category}>
                        <option value="">Select category...</option>
                        <option value="trading-strategies">Trading Strategies</option>
                        <option value="technical-analysis">Technical Analysis</option>
                        <option value="risk-management">Risk Management</option>
                        <option value="programming">Programming</option>
                        <option value="market-research">Market Research</option>
                        <option value="other">Other</option>
                      </select>
                    </div>
                  </div>
                {:else if uploadType === 'note'}
                  <div class="metadata-fields note-fields">
                    <div class="form-group">
                      <label>Title *</label>
                      <input type="text" placeholder="Note title" bind:value={uploadMetadata.title} required />
                    </div>
                    <div class="form-group">
                      <label>Content *</label>
                      <textarea
                        class="note-content"
                        placeholder="Write your note here... (Markdown supported)"
                        bind:value={uploadMetadata.content}
                        rows="6"
                      ></textarea>
                    </div>
                  </div>
                {/if}

                <!-- File upload area (not for notes) -->
                {#if uploadType !== 'note'}
                  <div
                    class="upload-dropzone"
                    class:drag-over={uploadDragOver}
                    on:dragover|preventDefault={() => uploadDragOver = true}
                    on:dragleave={() => uploadDragOver = false}
                    on:drop={handleDrop}
                  >
                    <Upload size={40} />
                    <p>Drag & drop {uploadType === 'book' ? 'PDF files' : 'files'} here</p>
                    <span>or</span>
                    <label class="file-input-label">
                      <input
                        type="file"
                        multiple
                        accept={uploadType === 'book' ? '.pdf' : '.pdf,.md,.txt,.csv,.json,.html'}
                        on:change={(e) => handleFileUpload(e.currentTarget.files)}
                      />
                      Browse Files
                    </label>
                    <p class="hint">
                      Supports: {uploadType === 'book' ? 'PDF (will be indexed)' : 'PDF, Markdown, TXT, CSV, JSON, HTML'}
                    </p>
                  </div>
                {:else}
                  <div class="note-actions">
                    <button class="btn primary" on:click={handleNoteSubmit}>
                      <Save size={14} /> Save Note
                    </button>
                  </div>
                {/if}

                <!-- Upload progress -->
                {#if uploadingFiles.length > 0}
                  <div class="upload-progress-list">
                    <h4>Uploading Files</h4>
                    {#each uploadingFiles as file}
                      <div class="upload-item" class:done={file.status === 'done'} class:error={file.status === 'error'}>
                        <File size={14} />
                        <span class="filename">{file.name}</span>
                        <span class="status-badge" class:uploading={file.status === 'uploading'} class:done={file.status === 'done'} class:error={file.status === 'error'}>
                          {file.status === 'uploading' ? `Uploading... ${file.progress}%` : file.status === 'done' ? 'Done' : file.status === 'error' ? 'Failed' : 'Pending'}
                        </span>
                        {#if file.status === 'uploading'}
                          <div class="progress-bar"><div style="width: {file.progress}%"></div></div>
                        {/if}
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
              <div class="modal-footer">
                <button class="btn secondary" on:click={() => { uploadModalOpen = false; uploadingFiles = []; uploadMetadata = { title: '', author: '', category: '', url: '', content: '' }; }}>Close</button>
              </div>
            </div>
          </div>
        {/if}

        {#if viewingArticle}
          <!-- Article Viewer Modal -->
      <div class="modal-overlay article-viewer-overlay" on:click={closeArticleViewer}>
        <div class="modal article-viewer-modal" on:click|stopPropagation>
          <ArticleViewer 
            article={viewingArticle} 
            onClose={closeArticleViewer}
            on:openInEditor={(e) => {
              closeArticleViewer();
              openInEditor(e.detail.article);
            }}
          />
        </div>
      </div>

        {:else if subPage === 'database'}
          <!-- Database Visualization Page -->
          <div class="database-page">
            <h2><Database size={20} /> Database Visualization</h2>
            <div class="db-stats">
              <div class="db-stat"><span class="label">Total Records</span><span class="value">12,450</span></div>
              <div class="db-stat"><span class="label">Strategies</span><span class="value">8</span></div>
              <div class="db-stat"><span class="label">Trades</span><span class="value">1,234</span></div>
              <div class="db-stat"><span class="label">Last Sync</span><span class="value">2 min ago</span></div>
            </div>
            <div class="db-tables">
              <div class="table-card">
                <h4>strategies</h4>
                <table>
                  <thead><tr><th>id</th><th>name</th><th>status</th><th>created</th></tr></thead>
                  <tbody>
                    <tr><td>1</td><td>ICT Scalper v2</td><td>primal</td><td>2025-01-15</td></tr>
                    <tr><td>2</td><td>SMC Reversal</td><td>ready</td><td>2025-02-01</td></tr>
                  </tbody>
                </table>
              </div>
              <div class="table-card">
                <h4>trades</h4>
                <table>
                  <thead><tr><th>id</th><th>symbol</th><th>type</th><th>pnl</th></tr></thead>
                  <tbody>
                    <tr><td>1024</td><td>EURUSD</td><td>BUY</td><td class="positive">+45.20</td></tr>
                    <tr><td>1023</td><td>GBPUSD</td><td>SELL</td><td class="negative">-12.50</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        
        {:else if subPage === 'bots-page'}
          <!-- All Bots Internal Page -->
          <div class="bots-page">
            <h2>Active Bots</h2>
            <div class="bot-cards">
              {#each bots.length > 0 ? bots : [{id: 'ict-eu', name: 'ICT_Scalper @EURUSD', state: 'primal', symbol: 'EURUSD'}, {id: 'ict-gb', name: 'ICT_Scalper @GBPUSD', state: 'primal', symbol: 'GBPUSD'}] as bot}
                <div class="bot-detail-card">
                  <div class="bot-status-indicator" style="background: {getStatusColor(bot.state)}"></div>
                  <div class="bot-main">
                    <h4>{bot.name}</h4>
                    <p>{bot.symbol}</p>
                  </div>
                  <select class="tag-select" on:change={(e) => handleTagChange(bot.id, e.currentTarget.value)}>
                    <option value="primal" selected={bot.state === 'primal'}>Primal</option>
                    <option value="ready" selected={bot.state === 'ready'}>Ready</option>
                    <option value="paused" selected={bot.state === 'paused'}>Paused</option>
                    <option value="quarantined" selected={bot.state === 'quarantined'}>Quarantine</option>
                  </select>
                </div>
              {/each}
            </div>
          </div>
        
        {:else if activeView === 'live'}
          <!-- Live Trading View with sub-tabs -->
          <LiveTradingView />

        {:else if activeView === 'paper-trading'}
          <!-- Paper Trading View -->
          <PaperTradingPanel baseUrl="http://localhost:8000" />

        {:else if activeView === 'router'}
          <!-- Strategy Router View -->
          <StrategyRouterView />

        {:else if activeView === 'journal'}
          <!-- Trade Journal View -->
          <TradeJournalView />

        {:else if activeView === 'shared-assets'}
          <!-- Shared Assets View -->
          <SharedAssetsView />

        {:else if activeView === 'kill-switch'}
          <!-- Kill Switch View -->
          <KillSwitchView />

        {:else if activeView === 'database-view'}
          <!-- Database View -->
          <DatabaseView />

        {:else if activeView === 'news'}
          <!-- News View -->
          <NewsView />

        {:else if activeView === 'assets'}
          <!-- Shared Assets with Database tab -->
          <div class="tab-nav">
            {#each viewConfig.assets.tabs as tab}
              <button 
                class="tab-item" 
                class:active={currentFolder === tab.id || (!currentFolder && tab.id === 'indicators')}
                on:click={() => tab.isPage ? openSubPage('database') : currentFolder = tab.id}
              >
                <svelte:component this={tab.icon} size={14} />
                {tab.name}
              </button>
            {/each}
          </div>
          
          <div class="asset-view">
            {#if !currentFolder}
               <div class="asset-grid">
                 {#each viewConfig.assets.tabs.filter(t => !t.isPage) as tab}
                   <div class="folder-item" role="button" tabindex="0" on:click={() => currentFolder = tab.id} on:keydown={(e) => e.key === 'Enter' && (currentFolder = tab.id)}>
                     <Folder size={40} class="text-accent" />
                     <span>{tab.name}</span>
                   </div>
                 {/each}
                 <div class="folder-item" role="button" tabindex="0" on:click={() => openSubPage('database')} on:keydown={(e) => e.key === 'Enter' && openSubPage('database')}>
                   <Database size={40} class="text-accent" />
                   <span>Database</span>
                 </div>
               </div>
            {:else}
              <!-- File List for Selected Folder -->
              <div class="file-list">
                 <div class="list-header">
                   <button class="back-link" on:click={() => currentFolder = null}>
                     <ArrowLeft size={14} /> Back to Assets
                   </button>
                   <h4>{viewConfig.assets.tabs.find(t => t.id === currentFolder)?.name || currentFolder}</h4>
                 </div>
                 
                 <div class="folder-grid">
                   {#each (mockAssets[currentFolder] || []) as file}
                     <div class="asset-item file">
                       <FileText size={32} />
                       <span>{file.name}</span>
                       <span class="file-size">{file.size}</span>
                     </div>
                   {/each}
                   {#if !(mockAssets[currentFolder] || []).length}
                     <p class="empty-msg">No files found in {currentFolder}</p>
                   {/if}
                 </div>
              </div>
            {/if}
          </div>
        
        {:else if activeView === 'ea'}
          <!-- EA Management Card View -->
          {#if nprdQueue.length > 0}
            <div class="queue-banner">
              <Clock size={14} /> {nprdQueue.length} NPRD job(s) in progress
            </div>
          {/if}
          
          <div class="ea-cards" class:list-view={viewMode === 'list'}>
            {#each strategyCards as strategy}
              <div class="ea-card" role="button" tabindex="0" on:click={() => navigateTo(strategy.id, strategy.name)} on:keydown={(e) => e.key === 'Enter' && navigateTo(strategy.id, strategy.name)}>
                <div class="card-status" style="background: {getStatusColor(strategy.status)}"></div>
                <div class="card-icon"><Bot size={32} /></div>
                <div class="card-info">
                  <h4>{strategy.name}</h4>
                  <p>Updated {strategy.lastUpdated}</p>
                  <div class="card-meta">
                    <span><TestTube size={12} /> {strategy.backtests} backtests</span>
                    <select class="card-tag" on:click|stopPropagation>
                      <option selected={strategy.tag === 'Primal'}>Primal</option>
                      <option selected={strategy.tag === 'Ready'}>Ready</option>
                      <option selected={strategy.tag === 'Paused'}>Paused</option>
                      <option selected={strategy.tag === 'Quarantine'}>Quarantine</option>
                    </select>
                  </div>
                </div>
                <ChevronRight size={16} class="card-arrow" />
              </div>
            {/each}
            <div class="ea-card add-new" role="button" tabindex="0" on:click={() => openSubPage('nprd-modal')} on:keydown={(e) => e.key === 'Enter' && openSubPage('nprd-modal')}>
              <Plus size={32} />
              <span>Process New NPRD</span>
            </div>
          </div>
          
          {#if currentFolder}
            <!-- Inside a strategy folder - Windows Explorer style -->
            <div class="folder-contents">
              <!-- Breadcrumb navigation -->
              <div class="folder-breadcrumb">
                <button class="breadcrumb-btn" on:click={() => currentFolder = ''}>
                  <Home size={14} />
                  <span>EA Management</span>
                </button>
                <ChevronRight size={12} class="breadcrumb-separator" />
                <span class="breadcrumb-current">{currentFolder}</span>
              </div>
              
              <div class="folder-grid">
                {#if currentFolder === 'nprd'}
                  <!-- NPRD Output files -->
                  {#each mockNprdFiles as file}
                    <div class="file-item" role="button" tabindex="0" on:click={() => openInEditor(file)} on:keydown={(e) => e.key === 'Enter' && openInEditor(file)}>
                      <FileText size={32} style="color: #f59e0b" />
                      <span>{file.name}</span>
                      <span class="file-size">{file.size}</span>
                    </div>
                  {/each}
                {:else if currentFolder === 'trd'}
                  <!-- TRD files -->
                  {#each mockTrdFiles as file}
                    <div class="file-item" role="button" tabindex="0" on:click={() => openInEditor(file)} on:keydown={(e) => e.key === 'Enter' && openInEditor(file)}>
                      <FileText size={32} style="color: #3b82f6" />
                      <span>{file.name}</span>
                      <span class="file-size">{file.size}</span>
                    </div>
                  {/each}
                {:else if currentFolder === 'ea'}
                  <!-- EA Code files -->
                  {#each mockEaFiles as file}
                    <div class="file-item" role="button" tabindex="0" on:click={() => openInEditor(file)} on:keydown={(e) => e.key === 'Enter' && openInEditor(file)}>
                      <FileText size={32} style="color: #10b981" />
                      <span>{file.name}</span>
                      <span class="file-size">{file.size}</span>
                    </div>
                  {/each}
                {:else if currentFolder === 'backtest'}
                  <!-- Backtest Report files -->
                  {#each mockBacktestFiles as file}
                    <div class="file-item" role="button" tabindex="0" on:click={() => openInEditor(file)} on:keydown={(e) => e.key === 'Enter' && openInEditor(file)}>
                      <FileText size={32} style="color: #8b5cf6" />
                      <span>{file.name}</span>
                      <span class="file-size">{file.size}</span>
                    </div>
                  {/each}
                {:else}
                  <!-- Default subfolders -->
                  <div class="folder-item" role="button" tabindex="0" on:click={() => currentFolder = 'nprd'} on:keydown={(e) => e.key === 'Enter' && (currentFolder = 'nprd')}>
                    <Folder size={40} style="color: #f59e0b" />
                    <span>NPRD Output</span>
                  </div>
                  <div class="folder-item" role="button" tabindex="0" on:click={() => currentFolder = 'trd'} on:keydown={(e) => e.key === 'Enter' && (currentFolder = 'trd')}>
                    <Folder size={40} style="color: #3b82f6" />
                    <span>TRD</span>
                  </div>
                  <div class="folder-item" role="button" tabindex="0" on:click={() => currentFolder = 'ea'} on:keydown={(e) => e.key === 'Enter' && (currentFolder = 'ea')}>
                    <Folder size={40} style="color: #10b981" />
                    <span>EA Code</span>
                  </div>
                  <div class="folder-item" role="button" tabindex="0" on:click={() => currentFolder = 'backtest'} on:keydown={(e) => e.key === 'Enter' && (currentFolder = 'backtest')}>
                    <Folder size={40} style="color: #8b5cf6" />
                    <span>Backtest Reports</span>
                  </div>
                {/if}
              </div>
            </div>
          {/if}
        
        {:else if activeView === 'backtest'}
          <!-- Backtest Results with Monte Carlo -->
          <div class="backtest-view">
            <div class="backtest-tabs">
                <button class="backtest-tab" class:active={!currentFolder || currentFolder === 'run'} on:click={() => currentFolder = 'run'}>
                  <Play size={14} /> Run Backtest
                </button>
                <button class="backtest-tab" class:active={currentFolder === 'results'} on:click={() => currentFolder = 'results'}>
                  <BarChart2 size={14} /> Results
                </button>
                <button class="backtest-tab" class:active={currentFolder === 'montecarlo'} on:click={() => currentFolder = 'montecarlo'}>
                  <PieChart size={14} /> Monte Carlo
                </button>
                <button class="backtest-tab" class:active={currentFolder === 'walkforward'} on:click={() => currentFolder = 'walkforward'}>
                  <TrendingUp size={14} /> Walk-Forward
                </button>
                <button class="backtest-tab" class:active={currentFolder === 'paper-trading'} on:click={() => currentFolder = 'paper-trading'}>
                  <DollarSign size={14} /> Paper Trading
                </button>
              </div>

            {#if currentFolder === 'run'}
              <!-- Run Backtest Configuration -->
              <div class="run-backtest-section">
                <div class="backtest-config">
                  <h3>Configure Backtest</h3>
                  <div class="config-grid">
                    <div class="form-group">
                      <label for="backtest-strategy">Strategy</label>
                      <select id="backtest-strategy" bind:value={backtestConfig.strategy}>
                        {#each strategyCards as strategy}
                          <option value={strategy.id}>{strategy.name}</option>
                        {/each}
                      </select>
                    </div>
                    <div class="form-group">
                      <label for="backtest-symbol">Symbol</label>
                      <select id="backtest-symbol">
                        <option value="EURUSD">EURUSD</option>
                        <option value="GBPUSD">GBPUSD</option>
                        <option value="USDJPY">USDJPY</option>
                        <option value="XAUUSD">XAUUSD</option>
                      </select>
                    </div>
                    <div class="form-group">
                      <label for="backtest-timeframe">Timeframe</label>
                      <select id="backtest-timeframe">
                        <option value="M5">M5</option>
                        <option value="M15">M15</option>
                        <option value="H1">H1</option>
                        <option value="H4">H4</option>
                      </select>
                    </div>
                    <div class="form-group">
                      <label for="backtest-period">Date Range</label>
                      <select id="backtest-period" bind:value={backtestConfig.period}>
                        <option value="1M">Last Month</option>
                        <option value="3M">Last 3 Months</option>
                        <option value="6M">Last 6 Months</option>
                        <option value="1Y">Last Year</option>
                        <option value="all">Max Available</option>
                      </select>
                    </div>
                  </div>

                  <div class="config-options">
                    <div class="checkbox-option">
                      <input type="checkbox" id="mc-sim" bind:checked={backtestConfig.monteCarlo} />
                      <label for="mc-sim">Run Monte Carlo Simulation (1000 runs)</label>
                    </div>
                    <div class="checkbox-option">
                      <input type="checkbox" id="walk-forward" />
                      <label for="walk-forward">Include Walk-Forward Analysis</label>
                    </div>
                  </div>

                  <div class="run-actions">
                    <button class="btn primary" on:click={runBacktest} disabled={backtestRunning}>
                      {#if backtestRunning}
                        <Loader size={16} class="spinning" />
                        Running...
                      {:else}
                        <Play size={16} />
                        Start Backtest
                      {/if}
                    </button>
                  </div>
                </div>
              </div>

            {:else if !currentFolder || currentFolder === 'results'}
              <!-- Backtest Results Summary -->
              <div class="backtest-summary">
                <div class="summary-cards">
                  <div class="summary-card">
                    <span class="label">Total Runs</span>
                    <span class="value">{backtestHistory.length}</span>
                  </div>
                  <div class="summary-card">
                    <span class="label">Total PnL (All)</span>
                    <span class="value" class:positive={backtestHistory.reduce((a,b) => a + (b.total_pnl || 0), 0) > 0} class:negative={backtestHistory.reduce((a,b) => a + (b.total_pnl || 0), 0) < 0}>
                       ${backtestHistory.reduce((a,b) => a + (b.total_pnl || 0), 0).toFixed(2)}
                    </span>
                  </div>
                  <div class="summary-card">
                    <span class="label">Best Run</span>
                    <span class="value positive">
                      ${Math.max(...backtestHistory.map(r => r.total_pnl || 0), 0).toFixed(2)}
                    </span>
                  </div>
                  <div class="summary-card">
                    <span class="label">Strategies</span>
                    <span class="value">{[...new Set(backtestHistory.map(r => r.strategy))].length}</span>
                  </div>
                </div>
                
                <div class="backtest-history">
                  <div class="history-header">
                    <h4>Recent Backtest Runs (DuckDB Analytics)</h4>
                    <button class="icon-btn" on:click={loadBacktests} title="Refresh"><RefreshCw size={14} /></button>
                  </div>
                  <div class="run-list">
                    {#if backtestHistory.length > 0}
                      {#each backtestHistory as run}
                        <div class="run-item" on:click={() => loadRunDetails(run.run_id)} role="button" tabindex="0" on:keydown={(e) => e.key === 'Enter' && loadRunDetails(run.run_id)}>
                          <div class="run-info">
                            <span class="run-name">{run.strategy}</span>
                            <span class="run-sub">{run.symbol} • {run.total_trades} trades</span>
                          </div>
                          <span class="run-date">{new Date(run.run_date).toLocaleDateString()} {new Date(run.run_date).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}</span>
                          <span class="run-result" class:positive={run.total_pnl > 0} class:negative={run.total_pnl < 0}>
                            {run.total_pnl > 0 ? '+' : ''}${run.total_pnl.toFixed(2)}
                          </span>
                        </div>
                      {/each}
                    {:else}
                      <div class="empty-state">
                        <AlertTriangle size={24} />
                        <span>No backtest data found. Run a backtest or check DB connection.</span>
                      </div>
                    {/if}
                  </div>
                </div>
              </div>
            
            {:else if currentFolder === 'montecarlo'}
              <!-- Monte Carlo Visualization -->
              <MonteCarloChart 
                data={backtestResults || {
                  returns: Array.from({length: 1000}, () => (Math.random() - 0.4) * 50),
                  confidence: 12.5,
                  worstCase: -18.2,
                  bestCase: 42.8,
                  median: 15.2,
                  mean: 14.8
                }}
              />
              
              <div class="mc-params">
                <h4>Simulation Parameters</h4>
                <div class="param-grid">
                  <div class="param-item">
                    <span class="param-label">Simulations</span>
                    <span class="param-value">1,000</span>
                  </div>
                  <div class="param-item">
                    <span class="param-label">Confidence Level</span>
                    <span class="param-value">95%</span>
                  </div>
                  <div class="param-item">
                    <span class="param-label">Time Period</span>
                    <span class="param-value">{backtestConfig.period}</span>
                  </div>
                  <div class="param-item">
                    <span class="param-label">Strategy</span>
                    <span class="param-value">{backtestConfig.strategy.toUpperCase()}</span>
                  </div>
                </div>
              </div>
            
            {:else if currentFolder === 'walkforward'}
              <!-- Walk-Forward Analysis -->
              <div class="walkforward-view">
                <div class="wf-summary">
                  <div class="wf-stat">
                    <span class="label">In-Sample Performance</span>
                    <span class="value positive">+28.4%</span>
                  </div>
                  <div class="wf-stat">
                    <span class="label">Out-of-Sample Performance</span>
                    <span class="value positive">+22.1%</span>
                  </div>
                  <div class="wf-stat">
                    <span class="label">Efficiency Ratio</span>
                    <span class="value">0.78</span>
                  </div>
                </div>
                
                <div class="wf-periods">
                  <h4>Walk-Forward Periods</h4>
                  <div class="period-list">
                    <div class="period-item">
                      <span class="period-label">Period 1</span>
                      <span class="period-train">Train: Jan-Jun 2025</span>
                      <span class="period-test">Test: Jul 2025</span>
                      <span class="period-result positive">+4.2%</span>
                    </div>
                    <div class="period-item">
                      <span class="period-label">Period 2</span>
                      <span class="period-train">Train: Feb-Jul 2025</span>
                      <span class="period-test">Test: Aug 2025</span>
                      <span class="period-result positive">+3.8%</span>
                    </div>
                    <div class="period-item">
                      <span class="period-label">Period 3</span>
                      <span class="period-train">Train: Mar-Aug 2025</span>
                      <span class="period-test">Test: Sep 2025</span>
                      <span class="period-result positive">+5.1%</span>
                    </div>
                  </div>
                </div>
              </div>
            {:else if currentFolder === 'paper-trading'}
              <!-- Paper Trading Panel -->
              <PaperTradingPanel baseUrl="http://localhost:8000" />
            {/if}
          </div>
        
        {:else if activeView === 'knowledge'}
          <!-- Knowledge Hub with Category Sidebar -->
          {#if !currentFolder}
            <div class="knowledge-hub-layout">
              <!-- Category Sidebar -->
              <div class="category-sidebar">
                <h3>Categories</h3>
                <button 
                  class="category-btn"
                  class:active={selectedCategory === 'all'}
                  on:click={() => selectedCategory = 'all'}
                >
                  <Folder size={16} />
                  <span>All Articles</span>
                  <span class="count">{articles.length}</span>
                </button>
                
                {#each Object.entries(categoryTree) as [key, cat]}
                  <button
                    class="category-btn"
                    class:active={selectedCategory === key}
                    on:click={() => selectedCategory = key}
                  >
                    <FolderOpen size={16} />
                    <span>{cat.label}</span>
                    <span class="count">{cat.count}</span>
                  </button>
                {/each}
              </div>
              
              <!-- Articles List -->
              <div class="articles-main">
                <div class="articles-header">
                  <h3>
                    {selectedCategory === 'all' ? 'All Articles' : categoryTree[selectedCategory]?.label || 'Articles'}
                  </h3>
                  <span class="article-count">{filteredArticles.length} articles</span>
                </div>
                
                <div class="articles-grid">
                  {#if filteredArticles.length === 0}
                    <div class="empty-state">
                      <Newspaper size={48} />
                      <p>No articles found in this category</p>
                      <span class="hint">Try selecting a different category or check if the backend is running</span>
                    </div>
                  {:else}
                    {#each filteredArticles as article}
                      <div class="article-card">
                        <div class="article-icon">
                          <Newspaper size={20} />
                        </div>
                        <div class="article-info">
                          <h4>{article.name}</h4>
                          <div class="article-meta">
                            <span class="category-tag" 
                              class:expert-advisors={article.category?.includes('expert_advisors')} 
                              class:integration={article.category?.includes('integration')} 
                              class:trading={article.category?.includes('trading')} 
                              class:trading-systems={article.category?.includes('trading-systems')}>
                              {article.category?.split('/')[1]?.replace('_', ' ').toUpperCase() || 'GENERAL'}
                            </span>
                          </div>
                        </div>
                        <div class="article-actions">
                          <button class="btn-icon" on:click={() => openArticleViewer(article)} title="View Article">
                            <FileText size={14} />
                          </button>
                          <button class="btn-icon" on:click={() => openInEditor(article)} title="Open in Editor">
                            <Edit3 size={14} />
                          </button>
                        </div>
                      </div>
                    {/each}
                  {/if}
                </div>
              </div>
            </div>
          {:else}
            <!-- Legacy Categories View -->
            <div class="category-cards">
              {#each viewConfig.knowledge.categories as cat}
                <div class="category-card" on:click={() => navigateTo(cat.id, cat.name)} role="button" tabindex="0" on:keydown={(e) => e.key === 'Enter' && navigateTo(cat.id, cat.name)}>
                  <Folder size={40} />
                  <span class="cat-name">{cat.name}</span>
                  <span class="cat-count">{cat.count} items</span>
                </div>
              {/each}
            </div>
            
            <div class="recent-section">
              <h3>Recent Articles</h3>
              <div class="recent-articles">
                {#each articles.slice(0, 5) as article}
                  <div class="recent-article-card" on:click={() => openArticleViewer(article)} role="button" tabindex="0" on:keydown={(e) => e.key === 'Enter' && openArticleViewer(article)}>
                    <div class="recent-article-icon">
                      {#if article.type === 'book'}
                        <Library size={18} />
                      {:else if article.type === 'note'}
                        <FileText size={18} />
                      {:else}
                        <Newspaper size={18} />
                      {/if}
                    </div>
                    <div class="recent-article-info">
                      <h6>{article.title || article.name}</h6>
                      <div class="recent-article-meta">
                        {#if article.author}<span>{article.author}</span>{/if}
                        {#if article.date}<span>{new Date(article.date).toLocaleDateString()}</span>{/if}
                      </div>
                    </div>
                    <ChevronRight size={14} class="recent-article-arrow" />
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        
        {:else if activeView === 'editor'}
          <!-- Editor Workspace -->
          <div class="editor-workspace">
            <!-- Breadcrumb Navigation -->
            {#if editorBreadcrumb.length > 0}
              <div class="editor-breadcrumb">
                <div class="breadcrumb-nav">
                  {#each editorBreadcrumb as crumb, index}
                    <button 
                      class="breadcrumb-item"
                      class:active={index === editorBreadcrumb.length - 1}
                      on:click={() => {
                        if (index < editorBreadcrumb.length - 1) {
                          // Navigate back to source with correct view mapping
                          const viewMap = {
                            'Knowledge': 'knowledge',
                            'Assets': 'assets', 
                            'EA Management': 'ea',
                            'Expert advisors': 'knowledge',
                            'Integration': 'knowledge',
                            'Trading': 'knowledge',
                            'Trading systems': 'knowledge'
                          };
                          const targetView = viewMap[crumb.name] || 'knowledge';
                          dispatch('viewChange', { view: targetView });
                        }  
                      }}
                    >
                      {crumb.name}
                    </button>
                    {#if index < editorBreadcrumb.length - 1}
                      <ChevronRight size={12} class="breadcrumb-separator" />
                    {/if}
                  {/each}
                </div>
              </div>
            {/if}
            
            <div class="editor-tabs">
              <div class="tab-list">
                {#each editorFiles as file}
                  <button 
                    class="tab-item"
                    class:active={activeEditorFile?.id === file.id}
                    on:click={() => {
                      activeEditorFile = file;
                      editorContent = file.content;
                      editorLanguage = file.language;
                      editorFilename = file.name;
                    }}
                  >
                    <FileText size={12} />
                    <span>{file.name}</span>
                    <button 
                      class="tab-close"
                      on:click|stopPropagation={() => closeEditorFile(file.id)}
                    >
                      <X size={10} />
                    </button>
                  </button>
                {/each}
                {#if editorFiles.length === 0}
                  <div class="empty-editor">
                    <Edit3 size={24} />
                    <p>No files open</p>
                    <span class="hint">Open articles, indicators, or other files from the Knowledge Hub or Assets sections</span>
                  </div>
                {/if}
              </div>
              <div class="tab-actions">
                <button class="toolbar-btn" on:click={() => dispatch('viewChange', { view: 'knowledge' })}>
                  <BookOpen size={14} /> Browse Files
                </button>
              </div>
            </div>
            
            {#if activeEditorFile}
              <div class="editor-container">
                <CodeEditor
                  bind:content={editorContent}
                  language={editorLanguage}
                  filename={editorFilename}
                  fileId={activeEditorFile?.id || ''}
                  filePath={activeEditorFile?.path || ''}
                  agent="copilot"
                  readOnly={false}
                  on:save={saveEditorFile}
                  on:change={(e) => {
                    if (activeEditorFile) {
                      activeEditorFile.content = e.detail.content;
                      const fileIndex = editorFiles.findIndex(f => f.id === activeEditorFile.id);
                      if (fileIndex !== -1) {
                        editorFiles[fileIndex] = { ...activeEditorFile };
                      }
                    }
                  }}
                />
              </div>
            {:else}
              <div class="editor-placeholder">
                <Edit3 size={48} />
                <h3>Editor Workspace</h3>
                <p>Open files from Knowledge Hub or Assets to start editing</p>
                <div class="placeholder-actions">
                  <button class="btn primary" on:click={() => dispatch('viewChange', { view: 'knowledge' })}>
                    <BookOpen size={14} /> Browse Knowledge Hub
                  </button>
                  <button class="btn secondary" on:click={() => dispatch('viewChange', { view: 'assets' })}>
                    <Boxes size={14} /> Browse Assets
                  </button>
                </div>
              </div>
            {/if}
          </div>
        
        {:else}
          <p class="placeholder">Select a view from the sidebar</p>
        {/if}
      </div>
    {/if}
</main>

<!-- Settings View (overlay) -->
<SettingsView bind:this={settingsViewComponent} on:close={closeSettings} />

<style>
  .main-content { grid-area: main; display: flex; flex-direction: column; background: var(--bg-primary); overflow: hidden; }
  
  .tab-bar { display: flex; align-items: center; height: 35px; background: var(--bg-secondary); border-bottom: 1px solid var(--border-subtle); overflow-x: auto; }
  .tab { display: flex; align-items: center; gap: 6px; padding: 0 12px; height: 100%; background: transparent; border: none; border-right: 1px solid var(--border-subtle); color: var(--text-secondary); font-size: 12px; cursor: pointer; white-space: nowrap; }
  .tab:hover { background: var(--bg-tertiary); }
  .tab.active { background: var(--bg-primary); color: var(--text-primary); }
  .tab-close { margin-left: 4px; padding: 0 4px; background: transparent; border: none; color: var(--text-muted); cursor: pointer; border-radius: 4px; }
  
  .content-area { flex: 1; overflow-y: auto; }
  .view-content { padding: 20px; }
  
  /* Header & Breadcrumbs - Windows Explorer Style */
  .view-header { margin-bottom: 20px; }
  .breadcrumb-nav {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 16px;
    padding: 8px 12px;
    background: var(--bg-glass);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    min-height: 40px;
    backdrop-filter: blur(var(--glass-blur));
  }
  .back-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-muted);
    cursor: pointer;
    margin-right: 8px;
    transition: all 0.2s ease;
  }
  .back-btn:hover {
    background: var(--bg-glass);
    color: var(--text-primary);
    border-color: var(--accent-primary);
    box-shadow: 0 0 0 1px var(--accent-primary);
  }
  .back-btn:active { transform: scale(0.95); }
  .breadcrumb-item {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    background: var(--bg-tertiary);
    border: 1px solid transparent;
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s ease;
  }
  .breadcrumb-item:hover {
    background: var(--bg-glass);
    color: var(--accent-primary);
    border-color: var(--border-subtle);
  }
  .breadcrumb-item:active { transform: scale(0.98); }
  .breadcrumb-current {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    color: var(--text-primary);
    font-weight: 500;
    font-size: 13px;
  }
  .breadcrumb-nav :global(svg) { flex-shrink: 0; }
  
  .view-toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .search-box { display: flex; align-items: center; gap: 8px; padding: 6px 12px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-muted); }
  .search-box input { background: transparent; border: none; color: var(--text-primary); font-size: 13px; outline: none; width: 180px; }
  .view-toggle { display: flex; border: 1px solid var(--border-subtle); border-radius: 6px; overflow: hidden; }
  .view-toggle button { padding: 6px 10px; background: var(--bg-secondary); border: none; color: var(--text-muted); cursor: pointer; }
  .view-toggle button.active { background: var(--accent-primary); color: var(--bg-primary); }
  .toolbar-btn { display: flex; align-items: center; gap: 6px; padding: 6px 12px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-secondary); font-size: 12px; cursor: pointer; }
  .toolbar-btn:hover { background: var(--bg-tertiary); }
  .toolbar-btn.primary { background: var(--accent-primary); border-color: var(--accent-primary); color: var(--bg-primary); }
  
  /* EA Cards */
  .ea-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
  .ea-cards.list-view { grid-template-columns: 1fr; }
  .ea-card { display: flex; align-items: center; gap: 16px; padding: 16px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 12px; cursor: pointer; transition: all 0.2s; position: relative; }
  .ea-card:hover { border-color: var(--accent-primary); transform: translateY(-2px); }
  .ea-card.add-new { justify-content: center; flex-direction: column; gap: 8px; border-style: dashed; color: var(--text-muted); }
  .card-status { position: absolute; top: 0; left: 0; width: 4px; height: 100%; border-radius: 12px 0 0 12px; }
  .card-icon { color: var(--accent-primary); }
  .card-info { flex: 1; }
  .card-info h4 { margin: 0 0 4px; font-size: 14px; color: var(--text-primary); }
  .card-info p { margin: 0; font-size: 11px; color: var(--text-muted); }
  .card-meta { display: flex; align-items: center; gap: 12px; margin-top: 8px; font-size: 11px; color: var(--text-muted); }
  .card-tag { padding: 2px 6px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 4px; color: var(--text-secondary); font-size: 10px; }
  :global(.card-arrow) { color: var(--text-muted); }
  
  /* Category cards */
  .category-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .category-card { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 24px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 12px; cursor: pointer; text-align: center; transition: all 0.2s; }
  .category-card:hover { border-color: var(--accent-primary); }
  .category-card :global(svg) { color: var(--accent-primary); }
  .cat-name { font-size: 14px; font-weight: 500; color: var(--text-primary); }
  .cat-count { font-size: 12px; color: var(--text-muted); }
  
  /* Folder contents */
  .folder-contents { margin-top: 20px; }
  .folder-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 16px; }
  .folder-item { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 20px; background: var(--bg-secondary); border-radius: 12px; cursor: pointer; text-align: center; }
  .folder-item:hover { background: var(--bg-tertiary); }
  .folder-item span { font-size: 12px; color: var(--text-primary); }
  
  /* Live Dashboard - removed unused selectors */

  /* Tab nav */
  .tab-nav { display: flex; gap: 4px; margin-bottom: 16px; }
  .tab-item { display: flex; align-items: center; gap: 6px; padding: 8px 14px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-muted); font-size: 12px; cursor: pointer; }
  .tab-item:hover { color: var(--text-primary); }
  .tab-item.active { background: var(--accent-primary); border-color: var(--accent-primary); color: var(--bg-primary); }
  
  /* Asset grid */
  .asset-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }
  .asset-item { display: flex; flex-direction: column; align-items: center; gap: 8px; padding: 20px; background: var(--bg-secondary); border-radius: 10px; cursor: pointer; }
  .asset-item:hover { background: var(--bg-tertiary); }
  .asset-item span { font-size: 11px; color: var(--text-primary); text-align: center; }
  
  /* Modal */
  .modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 100; }
  .modal { background: var(--bg-secondary); border-radius: 12px; width: 480px; max-width: 90%; }
  .modal-header { display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; border-bottom: 1px solid var(--border-subtle); }
  .modal-header h2 { margin: 0; font-size: 16px; color: var(--text-primary); }
  .modal-header button { background: none; border: none; color: var(--text-muted); cursor: pointer; }
  .modal-body { padding: 20px; }
  .form-group { margin-bottom: 16px; }
  .form-group label { display: block; margin-bottom: 6px; font-size: 12px; color: var(--text-muted); }
  .form-group input { width: 100%; padding: 10px 12px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 6px; color: var(--text-primary); font-size: 13px; }
  .modal-footer { display: flex; justify-content: flex-end; gap: 8px; padding: 16px 20px; border-top: 1px solid var(--border-subtle); }
  .btn { padding: 8px 16px; border-radius: 6px; font-size: 13px; cursor: pointer; }
  .btn.secondary { background: var(--bg-tertiary); border: 1px solid var(--border-subtle); color: var(--text-secondary); }
  .btn.primary { background: var(--accent-primary); border: none; color: var(--bg-primary); }
  
  /* Queue */
  .queue-banner { display: flex; align-items: center; gap: 8px; padding: 10px 16px; background: rgba(139,92,246,0.1); border: 1px solid #8b5cf6; border-radius: 8px; color: #8b5cf6; font-size: 12px; margin-bottom: 16px; }
  .queue-section { margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-subtle); }
  .queue-section h4 { margin: 0 0 12px; font-size: 13px; color: var(--text-primary); }
  .queue-item { display: flex; align-items: center; gap: 12px; padding: 8px; background: var(--bg-tertiary); border-radius: 6px; margin-bottom: 8px; }
  .queue-item .status { font-size: 11px; }
  .progress-bar { flex: 1; height: 4px; background: var(--border-subtle); border-radius: 2px; overflow: hidden; }
  .progress-bar div { height: 100%; background: var(--accent-primary); }
  
  /* Database page */
  .database-page h2 { display: flex; align-items: center; gap: 8px; margin: 0 0 20px; color: var(--accent-primary); }
  .db-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 24px; }
  .db-stat { padding: 16px; background: var(--bg-secondary); border-radius: 8px; text-align: center; }
  .db-stat .label { display: block; font-size: 11px; color: var(--text-muted); margin-bottom: 4px; }
  .db-stat .value { font-size: 18px; font-weight: 600; color: var(--text-primary); }
  .db-tables { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .table-card { padding: 16px; background: var(--bg-secondary); border-radius: 10px; }
  .table-card h4 { margin: 0 0 12px; font-size: 13px; color: var(--accent-primary); }
  .table-card table { width: 100%; font-size: 11px; border-collapse: collapse; }
  .table-card th, .table-card td { padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--border-subtle); }
  .table-card th { color: var(--text-muted); }
  .table-card td { color: var(--text-secondary); }
  .table-card .positive { color: #10b981; }
  .table-card .negative { color: #ef4444; }
  
  /* Bots page */
  .bots-page h2 { margin: 0 0 20px; color: var(--text-primary); }
  .bot-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }
  .bot-detail-card { display: flex; align-items: center; gap: 12px; padding: 16px; background: var(--bg-secondary); border: 1px solid var(--border-subtle); border-radius: 10px; }
  .bot-status-indicator { width: 8px; height: 8px; border-radius: 50%; }
  .bot-main { flex: 1; }
  .bot-main h4 { margin: 0; font-size: 13px; color: var(--text-primary); }
  .bot-main p { margin: 4px 0 0; font-size: 11px; color: var(--text-muted); }
  .tag-select { padding: 4px 8px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 4px; color: var(--text-primary); font-size: 11px; }
  
  /* Article list - removed unused selectors */

  /* File editor */
  .file-editor { display: flex; flex-direction: column; height: 100%; }
  .editor-toolbar { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: var(--bg-secondary); border-bottom: 1px solid var(--border-subtle); }
  .breadcrumb { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text-muted); }
  .editor-actions { display: flex; gap: 8px; }
  .action-btn { display: flex; align-items: center; gap: 4px; padding: 4px 10px; background: var(--bg-tertiary); border: 1px solid var(--border-subtle); border-radius: 4px; color: var(--text-secondary); font-size: 11px; cursor: pointer; }
  .action-btn.save { background: var(--accent-primary); border-color: var(--accent-primary); color: var(--bg-primary); }
  .file-textarea { flex: 1; padding: 16px; background: var(--bg-primary); border: none; color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-size: 13px; resize: none; outline: none; }
  .file-content { flex: 1; padding: 16px; margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 1.6; color: var(--text-secondary); overflow: auto; }
  
  .placeholder { color: var(--text-muted); text-align: center; padding: 40px; }
  
  /* Upload Modal Styles */
  .upload-modal { min-width: 450px; }
  .upload-dropzone {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 40px 20px;
    margin-top: 12px;
    border: 2px dashed var(--border-subtle);
    border-radius: 12px;
    background: var(--bg-primary);
    color: var(--text-muted);
    text-align: center;
    transition: all 0.2s ease;
    cursor: pointer;
  }
  .upload-dropzone:hover, .upload-dropzone.drag-over {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.05);
    color: var(--accent-primary);
  }
  .upload-dropzone p { margin: 0; font-size: 14px; }
  .upload-dropzone span { font-size: 12px; color: var(--text-muted); }
  .upload-dropzone .hint { font-size: 11px; color: var(--text-muted); margin-top: 8px; }
  
  .file-input-label {
    display: inline-block;
    padding: 8px 16px;
    background: var(--accent-primary);
    color: var(--bg-primary);
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.15s ease;
  }
  .file-input-label:hover { opacity: 0.9; }
  .file-input-label input { display: none; }
  
  .upload-progress-list {
    margin-top: 16px;
    padding: 12px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }
  .upload-progress-list h4 { margin: 0 0 8px; font-size: 12px; color: var(--text-secondary); }
  
  .upload-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
    font-size: 12px;
    border-bottom: 1px solid var(--border-subtle);
  }
  .upload-item:last-child { border-bottom: none; }
  .upload-item .filename { flex: 1; color: var(--text-primary); }
  .upload-item .status-badge {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
  }
  .upload-item .status-badge.uploading { background: rgba(99, 102, 241, 0.2); color: var(--accent-primary); }
  .upload-item .status-badge.done { background: rgba(16, 185, 129, 0.2); color: #10b981; }
  .upload-item .status-badge.error { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
  .upload-item.done { opacity: 0.6; }
  .upload-item.error .filename { color: #ef4444; }
  
  /* Backtest View Styles */
  .backtest-view { padding: 0; }
  .backtest-tabs {
    display: flex;
    gap: 4px;
    padding: 0 0 16px 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 16px;
  }
  .backtest-tab {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    background: var(--bg-tertiary);
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .backtest-tab:hover { background: var(--bg-secondary); color: var(--text-primary); }
  .backtest-tab.active {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  /* Run Backtest Section */
  .run-backtest-section {
    padding: 20px;
  }

  .backtest-config {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 24px;
    border: 1px solid var(--border-subtle);
  }

  .backtest-config h3 {
    margin: 0 0 20px;
    font-size: 16px;
    color: var(--text-primary);
  }

  .config-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-bottom: 20px;
  }

  .config-grid .form-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .config-grid .form-group label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
  }

  .config-grid .form-group select {
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    outline: none;
  }

  .config-grid .form-group select:focus {
    border-color: var(--accent-primary);
  }

  .config-options {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-bottom: 20px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .checkbox-option {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .checkbox-option input[type="checkbox"] {
    width: 18px;
    height: 18px;
    cursor: pointer;
  }

  .checkbox-option label {
    font-size: 13px;
    color: var(--text-primary);
    cursor: pointer;
  }

  .run-actions {
    display: flex;
    justify-content: flex-end;
  }

  .run-actions .btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    border: none;
    transition: all 0.15s;
  }

  .run-actions .btn.primary {
    background: var(--accent-primary);
    color: var(--bg-primary);
  }

  .run-actions .btn.primary:hover:not(:disabled) {
    opacity: 0.9;
  }

  .run-actions .btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  
  .backtest-summary {}
  .summary-cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }
  .summary-card {
    display: flex;
    flex-direction: column;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
  }
  .summary-card .label { font-size: 11px; color: var(--text-muted); margin-bottom: 6px; }
  .summary-card .value { font-size: 20px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  .summary-card .value.positive { color: #10b981; }
  .summary-card .value.negative { color: #ef4444; }
  
  .backtest-history { background: var(--bg-secondary); padding: 16px; border-radius: 10px; border: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: 12px; }
  
  .history-header { display: flex; justify-content: space-between; align-items: center; }
  .history-header h4 { margin: 0; font-size: 13px; color: var(--text-primary); }
  
  .icon-btn { background: none; border: none; color: var(--text-muted); cursor: pointer; padding: 4px; border-radius: 4px; transition: all 0.2s; }
  .icon-btn:hover { background: var(--bg-tertiary); color: var(--accent-primary); }

  .run-list { display: flex; flex-direction: column; gap: 8px; max-height: 400px; overflow-y: auto; }
  
  .run-item {
    display: flex;
    align-items: center;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
  }
  .run-item:hover { 
      background: var(--bg-surface); 
      transform: translateX(2px); 
      border-color: var(--border-subtle);
  }

  .run-info { flex: 1; display: flex; flex-direction: column; gap: 2px; }
  .run-item .run-name { font-size: 13px; color: var(--text-primary); font-weight: 500; }
  .run-sub { font-size: 11px; color: var(--text-muted); }
  
  .run-item .run-date { font-size: 10px; color: var(--text-muted); margin-right: 16px; font-family: 'JetBrains Mono', monospace; }
  .run-item .run-result { font-size: 13px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  .run-item .run-result.positive { color: #10b981; }
  .run-item .run-result.negative { color: #ef4444; }
  
  .mc-params {
    margin-top: 20px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
  }
  .mc-params h4 { margin: 0 0 12px; font-size: 13px; color: var(--text-primary); }
  .param-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
  .param-item { display: flex; flex-direction: column; gap: 4px; }
  .param-label { font-size: 10px; color: var(--text-muted); }
  .param-value { font-size: 13px; color: var(--text-primary); font-weight: 500; }
  
  .walkforward-view {}
  .wf-summary {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 20px;
  }
  .wf-stat {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
  }
  .wf-stat .label { font-size: 11px; color: var(--text-muted); }
  .wf-stat .value { font-size: 18px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  
  .wf-periods { background: var(--bg-secondary); padding: 16px; border-radius: 10px; border: 1px solid var(--border-subtle); }
  .wf-periods h4 { margin: 0 0 12px; font-size: 13px; color: var(--text-primary); }
  .period-list { display: flex; flex-direction: column; gap: 8px; }
  .period-item {
    display: grid;
    grid-template-columns: 80px 1fr 1fr 80px;
    align-items: center;
    padding: 10px 12px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    font-size: 12px;
  }
  .period-label { font-weight: 600; color: var(--text-primary); }
  .period-train, .period-test { color: var(--text-muted); }
  .period-result { text-align: right; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  .period-result.positive { color: #10b981; }
  .period-result.negative { color: #ef4444; }
  
  /* Loading Spinner */
  :global(.spinning) { animation: spin 1s linear infinite; }
  @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
  
  .progress-section { margin-top: 16px; padding: 12px; background: var(--bg-tertiary); border-radius: 8px; }
  .spinner-row { display: flex; align-items: center; gap: 8px; color: var(--text-secondary); font-size: 13px; }

  /* Upload Type Selector */
  .upload-type-selector {
    display: flex;
    gap: 12px;
    margin-top: 8px;
  }

  .type-option {
    flex: 1;
    display: flex;
    align-items: center;
    padding: 12px;
    background: var(--bg-tertiary);
    border: 2px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .type-option:hover {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.05);
  }

  .type-option.selected {
    border-color: var(--accent-primary);
    background: rgba(99, 102, 241, 0.1);
  }

  .type-option input {
    display: none;
  }

  .type-content {
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
  }

  .type-content :global(svg) {
    color: var(--accent-primary);
    flex-shrink: 0;
  }

  .type-content div {
    display: flex;
    flex-direction: column;
  }

  .type-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .type-desc {
    font-size: 11px;
    color: var(--text-muted);
  }

  /* Metadata Fields */
  .metadata-fields {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 16px;
    padding: 16px;
    background: var(--bg-tertiary);
    border-radius: 8px;
  }

  .metadata-fields .form-group {
    margin-bottom: 0;
  }

  .metadata-fields .form-group.full-width {
    grid-column: 1 / -1;
  }

  .optional {
    font-size: 10px;
    color: var(--text-muted);
    font-weight: normal;
  }

  .note-content {
    width: 100%;
    padding: 10px 12px;
    background: var(--bg-primary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-primary);
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    resize: vertical;
    outline: none;
    min-height: 120px;
  }

  .note-content:focus {
    border-color: var(--accent-primary);
  }

  .note-actions {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 16px;
  }

  /* Article Grid */
  .article-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }

  .article-card {
    display: flex;
    gap: 12px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .article-card:hover {
    border-color: var(--accent-primary);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  .article-type-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: var(--bg-tertiary);
    border-radius: 8px;
    color: var(--accent-primary);
    flex-shrink: 0;
  }

  .article-card-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 8px;
    min-width: 0;
  }

  .article-title {
    margin: 0;
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .article-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    font-size: 11px;
  }

  .meta-item {
    color: var(--text-muted);
  }

  .meta-label {
    font-weight: 500;
  }

  .meta-tag {
    padding: 2px 8px;
    background: var(--accent-primary);
    color: var(--bg-primary);
    border-radius: 4px;
    font-size: 10px;
    font-weight: 500;
  }

  .category-tag {
    padding: 2px 8px;
    background: var(--accent-primary);
    color: white;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
    text-transform: uppercase;
  }
  
  .category-tag.expert-advisors {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
  }
  
  .category-tag.integration {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
  }
  
  .category-tag.trading {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
  }
  
  .category-tag.trading-systems {
    background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
    color: white;
  }

  .article-excerpt {
    margin: 0;
    font-size: 12px;
    color: var(--text-secondary);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .article-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
  }

  .file-size {
    color: var(--text-muted);
  }

  .article-tags {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }

  .article-tags .tag {
    padding: 2px 6px;
    background: var(--bg-tertiary);
    color: var(--text-muted);
    border-radius: 3px;
    font-size: 10px;
  }

  .article-arrow {
    color: var(--text-muted);
    flex-shrink: 0;
  }

  /* Recent Articles */
  .recent-articles {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .recent-article-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .recent-article-card:hover {
    border-color: var(--accent-primary);
    background: var(--bg-tertiary);
  }

  .recent-article-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    color: var(--accent-primary);
    flex-shrink: 0;
  }

  .recent-article-info {
    flex: 1;
    min-width: 0;
  }

  .recent-article-info h6 {
    margin: 0 0 4px;
    font-size: 13px;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .recent-article-meta {
    display: flex;
    gap: 8px;
    font-size: 11px;
    color: var(--text-muted);
  }

  .recent-article-arrow {
    color: var(--text-muted);
    flex-shrink: 0;
  }

  /* Article Viewer Modal */
  .article-viewer-overlay {
    z-index: 150;
  }

  .article-viewer-modal {
    width: 90%;
    max-width: 1000px;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
  }

  .article-viewer-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-subtle);
  }

  .article-viewer-title h2 {
    margin: 0 0 8px;
    font-size: 18px;
    color: var(--text-primary);
  }

  .article-viewer-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 16px;
    font-size: 12px;
  }

  .article-viewer-meta .meta-item {
    color: var(--text-muted);
  }

  .article-viewer-meta .meta-label {
    font-weight: 500;
  }

  .article-viewer-meta .meta-tag {
    padding: 4px 10px;
    background: var(--accent-primary);
    color: var(--bg-primary);
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }

  .article-viewer-actions {
    display: flex;
    gap: 8px;
  }

  .article-viewer-body {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .article-content-wrapper {
    flex: 1;
    padding: 24px;
    overflow-y: auto;
  }

  .article-content {
    font-size: 14px;
    line-height: 1.7;
    color: var(--text-primary);
  }

  .article-content h1,
  .article-content h2,
  .article-content h3 {
    margin: 24px 0 12px;
    color: var(--text-primary);
  }

  .article-content h1 { font-size: 24px; }
  .article-content h2 { font-size: 20px; }
  .article-content h3 { font-size: 16px; }

  .article-content p {
    margin: 0 0 16px;
  }

  .article-content a {
    color: var(--accent-primary);
    text-decoration: none;
  }

  .article-content a:hover {
    text-decoration: underline;
  }

  .article-content ul,
  .article-content ol {
    margin: 0 0 16px;
    padding-left: 24px;
  }

  .article-content li {
    margin-bottom: 4px;
  }

  /* Enhanced Code Blocks with Syntax Highlighting */
  :global(.code-block) {
    position: relative;
    margin: 16px 0;
    border-radius: 8px;
    background: #1e1e1e;
    overflow: hidden;
    border: 1px solid #3d3d3d;
  }
  
  :global(.code-header) {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background: #2d2d2d;
    border-bottom: 1px solid #3d3d3d;
  }
  
  :global(.code-lang) {
    font-size: 11px;
    color: #888;
    text-transform: uppercase;
    font-weight: 600;
    letter-spacing: 0.5px;
  }
  
  :global(.copy-btn) {
    padding: 4px 10px;
    font-size: 11px;
    background: #3d3d3d;
    border: 1px solid #4d4d4d;
    color: #fff;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  :global(.copy-btn:hover) {
    background: #4d4d4d;
    border-color: #5d5d5d;
  }
  
  :global(.hljs) {
    padding: 16px;
    overflow-x: auto;
    font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
    font-size: 13px;
    line-height: 1.6;
    background: #1e1e1e !important;
  }
  
  :global(.inline-code) {
    padding: 2px 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 0.9em;
    color: #e06c75;
  }
  
  /* Split View Editor */
  .split-view-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    height: 100%;
    overflow: hidden;
  }
  
  .editor-pane, .preview-pane {
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border-right: 1px solid var(--border-subtle);
  }
  
  .preview-pane {
    border-right: none;
  }
  
  .pane-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }
  
  .pane-header h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  
  .markdown-editor {
    flex: 1;
    padding: 20px;
    font-family: 'Monaco', 'Menlo', monospace;
    font-size: 14px;
    line-height: 1.6;
    background: #1e1e1e;
    color: #d4d4d4;
    border: none;
    resize: none;
    outline: none;
  }
  
  .markdown-editor::placeholder {
    color: #666;
  }
  
  .preview-pane .article-content {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    background: var(--bg-primary);
  }
  
  /* Category Sidebar */
  .knowledge-hub-layout {
    display: grid;
    grid-template-columns: 250px 1fr;
    gap: 0;
    height: 100%;
    overflow: hidden;
  }
  
  .category-sidebar {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
    padding: 16px;
    overflow-y: auto;
  }
  
  .category-sidebar h3 {
    margin: 0 0 12px 0;
    font-size: 13px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  
  .category-btn {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    margin-bottom: 4px;
    background: transparent;
    border: none;
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
  }
  
  .category-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }
  
  .category-btn.active {
    background: var(--accent-primary);
    color: #fff;
  }
  
  .category-btn span:first-of-type {
    flex: 1;
    font-size: 13px;
  }
  
  .category-btn .count {
    padding: 2px 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }
  
  .category-btn.active .count {
    background: rgba(255, 255, 255, 0.2);
  }
  
  /* Articles Main Area */
  .articles-main {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  
  .articles-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-subtle);
    background: var(--bg-secondary);
  }
  
  .articles-header h3 {
    margin: 0;
    font-size: 18px;
    color: var(--text-primary);
  }
  
  .article-count {
    font-size: 13px;
    color: var(--text-muted);
  }
  
  .articles-grid {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    align-content: start;
  }
  
  .article-card {
    display: flex;
    gap: 12px;
    padding: 16px;
    background: var(--bg-secondary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .article-card:hover {
    background: var(--bg-tertiary);
    border-color: var(--accent-primary);
    transform: translateY(-2px);
  }
  
  .article-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: rgba(59, 130, 246, 0.1);
    border-radius: 8px;
    color: var(--accent-primary);
    flex-shrink: 0;
  }
  
  .article-info {
    flex: 1;
    min-width: 0;
  }
  
  .article-info h4 {
    margin: 0 0 8px 0;
    font-size: 14px;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }
  
  .article-meta {
    display: flex;
    gap: 8px;
    align-items: center;
    font-size: 12px;
  }
  
  .category-tag {
    padding: 2px 8px;
    background: rgba(59, 130, 246, 0.2);
    color: var(--accent-primary);
    border-radius: 4px;
    font-size: 11px;
    font-weight: 500;
  }
  
  .size {
    color: var(--text-muted);
  }
  
  .article-actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }
  
  .btn-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .btn-icon:hover {
    background: var(--accent-primary);
    color: white;
  }
  
  /* Article Viewer Modal Enhancements */
  .article-viewer-modal {
    max-width: 90vw;
    max-height: 90vh;
    width: 1200px;
    display: flex;
    flex-direction: column;
  }
  
  .article-viewer-modal.edit-mode {
    width: 1400px;
  }
  
  .btn-edit, .btn-save {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }
  
  .btn-edit:hover, .btn-save:hover {
    background: var(--bg-surface);
    color: var(--text-primary);
  }
  
  .btn-edit.active {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: #fff;
  }
  
  .btn-save {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: #fff;
  }
  
  .btn-save:hover {
    opacity: 0.9;
  }
  
  .article-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: var(--text-muted);
    gap: 16px;
  }
  
  .spinner {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .article-content .inline-code {
    padding: 2px 6px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: var(--accent-primary);
  }

  .article-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 48px;
    color: var(--text-muted);
    text-align: center;
  }

  .article-placeholder :global(svg) {
    opacity: 0.5;
  }

  .article-placeholder .file-info {
    font-size: 12px;
    color: var(--text-muted);
  }

  /* Related Articles Sidebar */
  .related-articles-sidebar {
    width: 280px;
    padding: 24px;
    background: var(--bg-tertiary);
    border-left: 1px solid var(--border-subtle);
    overflow-y: auto;
  }

  .related-articles-sidebar h4 {
    margin: 0 0 16px;
    font-size: 14px;
    color: var(--text-primary);
  }

  .related-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .related-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    background: var(--bg-secondary);
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .related-item:hover {
    background: var(--bg-surface);
    border-color: var(--accent-primary);
  }

  .related-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--bg-tertiary);
    border-radius: 6px;
    color: var(--accent-primary);
    flex-shrink: 0;
  }

  .related-info {
    flex: 1;
    min-width: 0;
  }

  .related-title {
    display: block;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .related-category {
    font-size: 10px;
    color: var(--text-muted);
  }

  /* Article Viewer Footer */
  .article-viewer-footer {
    padding: 16px 24px;
    border-top: 1px solid var(--border-subtle);
  }

  .article-tags {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }

  .article-tags .tags-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
  }

  .article-tags .tag {
    padding: 4px 10px;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    border-radius: 4px;
    font-size: 11px;
  }

  /* Editor Workspace Styles */
  .editor-workspace {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: var(--bg-primary);
  }

  .editor-tabs {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .tab-list {
    display: flex;
    gap: 0;
    align-items: center;
    overflow-x: auto;
  }

  .empty-editor {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 16px 24px;
    color: var(--text-muted);
    text-align: center;
  }

  .empty-editor p {
    margin: 0;
    font-size: 13px;
  }

  .empty-editor .hint {
    font-size: 11px;
    opacity: 0.8;
  }

  .editor-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .editor-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 24px;
    color: var(--text-muted);
    text-align: center;
  }

  .editor-placeholder h3 {
    margin: 0;
    font-size: 18px;
    color: var(--text-secondary);
  }

  .editor-placeholder p {
    margin: 0;
    font-size: 14px;
  }

  .placeholder-actions {
    display: flex;
    gap: 12px;
  }

  .btn {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 16px;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn.primary {
    background: var(--accent-primary);
    border-color: var(--accent-primary);
    color: var(--bg-primary);
  }

  .btn.secondary {
    background: var(--bg-secondary);
    color: var(--text-secondary);
  }

  .btn:hover {
    transform: translateY(-1px);
  }

  /* Editor Breadcrumb Navigation */
  .editor-breadcrumb {
    padding: 8px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
  }

  .breadcrumb-nav {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
  }

  .breadcrumb-item {
    display: flex;
    align-items: center;
    padding: 4px 8px;
    background: transparent;
    border: none;
    border-radius: 4px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
    font-size: 11px;
  }

  .breadcrumb-item:hover:not(.active) {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .breadcrumb-item.active {
    color: var(--accent-primary);
    font-weight: 600;
  }

  .breadcrumb-separator {
    color: var(--text-muted);
    opacity: 0.5;
  }

  /* Folder Breadcrumb Navigation */
  .folder-breadcrumb {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 16px;
  }

  .breadcrumb-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: transparent;
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s;
    font-size: 12px;
  }

  .breadcrumb-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border-color: var(--accent-primary);
  }

  .breadcrumb-current {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 13px;
  }

  /* File Items */
  .file-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 20px;
    background: var(--bg-secondary);
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-align: center;
  }

  .file-item:hover {
    background: var(--bg-tertiary);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }

  .file-item span:first-of-type {
    font-size: 12px;
    color: var(--text-primary);
    font-weight: 500;
    word-break: break-all;
  }

  .file-size {
    font-size: 10px;
    color: var(--text-muted);
  }
</style>
