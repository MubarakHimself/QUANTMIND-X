# QuantMind X - UI Data Models

## 1. Agent System
```typescript
type AgentId = 'analyst' | 'quant_code' | 'copilot';

interface AgentProfile {
    id: AgentId;
    name: string; // "Analyst Agent", "QuantCode", "QuantMind Copilot"
    role: string;
    avatar: string;
    systemPrompt: string; // Editable in Settings
    activeKnowledgeBaseId: string; // Linked KB
}
```

## 2. Rich Chat Interface
```typescript
interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: number;
    attachments?: FileAttachment[];
    toolCalls?: ToolCall[];
}

interface FileAttachment {
    name: string;
    path: string;
    type: 'image' | 'code' | 'text';
}

interface ChatContext {
    mode: 'autonomous' | 'interactive';
    activeAgent: AgentId;
    selectedSkills: string[]; // e.g. ["interface-design"]
}
```

## 3. The Library (Unified)
```typescript
type LibraryItemType = 'article' | 'asset' | 'nprd_storyline';

interface LibraryItem {
    id: string;
    type: LibraryItemType;
    title: string;
    content: string; // Markdown
    tags: string[];
    path: string; // File path
    metadata: {
        author?: string; // "NPRD", "Analyst"
        status?: 'new' | 'processed' | 'archived';
        linkedAgent?: AgentId; // For drag-and-drop association
        nprdData?: {
            speakers: Record<string, string>; // "SPEAKER_01": "Host"
            timeline: Array<{
                timestamp: string;
                transcript: string;
                visual_description: string;
                ocr_content: string[];
                slide_detected: boolean;
            }>;
        }
    };
}
```

## 4. Assets & Strategies (The Manifest)
```typescript
type EAStatus = 'incubation' | 'backtesting' | 'paper_trading' | 'live' | 'archived';

interface QuantMindManifest {
    id: string;
    name: string;
    version: string;
    status: EAStatus;
    paths: {
        trd: string;
        code: string;
        backtest: string;
    };
    performance: {
        pnl: number;
        drawdown: number;
        winRate: number;
    };
    configuration: {
        markets: string[]; // ["EURUSD", "XAUUSD"]
        timeframes: string[]; // ["M15", "H1"]
        vpsId?: string;
    };
}
```

## 5. Prop Firm Module
```typescript
interface PropFirmChallenge {
    id: string;
    firmName: string;
    challengeName: string;
    accountNumber: string;
    status: 'active' | 'passed' | 'failed';
    metrics: {
        balance: number;
        equity: number;
        dailyLoss: number;
        maxLoss: number;
        profitTarget: number;
    };
    rules: {
        maxDailyDrawdown: number; // Percent or Fixed
        maxTotalDrawdown: number;
    };
    assignedEAs: string[]; // List of Manifest IDs running here
}
```

## 6. Global State (Store)
```typescript
interface AppState {
    user: {
        balance: number;
        currency: string;
    };
    navigation: {
        sidebarActivity: 'library' | 'prop_firms' | 'nprd' | 'code';
        rightSidebarAgent: AgentId | null;
    };
    library: {
        items: LibraryItem[];
        filter: string;
    };
    activeChallenge: PropFirmChallenge | null;
}
```
