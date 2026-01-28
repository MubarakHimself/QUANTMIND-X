import { useState } from 'react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import {
    Files,
    Database,
    LineChart,
    Settings,
    BookOpen, // Knowledge
    FileText, // NPRD
    BrainCircuit, // AI/Copilot Toggle
    Cpu // EA Manager
} from 'lucide-react';
import clsx from 'clsx';

// Activity Bar Item Component
const ActivityBarItem = ({ icon: Icon, active, onClick, label, bottom }: any) => (
    <button
        onClick={onClick}
        className={clsx(
            "p-3 w-full flex justify-center items-center transition-all duration-200 border-l-2",
            active
                ? "border-primary text-primary bg-white/5"
                : "border-transparent text-text-muted hover:text-text-primary hover:bg-white/5",
            bottom && "mt-auto mb-2"
        )}
        title={label}
    >
        <Icon size={24} strokeWidth={1.5} />
    </button>
);

export default function Shell({
    children,
    activeActivity,
    onActivityChange,
    sidebarContent,
    rightSidebar, // NEW: Copilot Panel
    onToggleCopilot,
    isCopilotOpen
}: {
    children: React.ReactNode,
    activeActivity: string,
    onActivityChange: (activity: string) => void,
    sidebarContent?: React.ReactNode,
    rightSidebar?: React.ReactNode,
    onToggleCopilot: () => void,
    isCopilotOpen: boolean
}) {
    const [bottomPanelOpen, setBottomPanelOpen] = useState(true);

    return (
        <div className="h-screen w-screen bg-background flex text-text-primary overflow-hidden">
            {/* A. ACTIVITY BAR (Left) */}
            <div className="w-12 flex-none flex flex-col items-center bg-surface-start border-r border-border py-2 z-20">
                <div className="mb-4">
                    {/* LOGO */}
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-secondary to-primary flex items-center justify-center font-bold text-black">
                        Q
                    </div>
                </div>

                <ActivityBarItem
                    icon={Files}
                    label="Code"
                    active={activeActivity === 'code'}
                    onClick={() => onActivityChange('code')}
                />
                <ActivityBarItem
                    icon={BookOpen}
                    label="Knowledge Hub"
                    active={activeActivity === 'knowledge'}
                    onClick={() => onActivityChange('knowledge')}
                />
                <ActivityBarItem
                    icon={Database}
                    label="Assets"
                    active={activeActivity === 'assets'}
                    onClick={() => onActivityChange('assets')}
                />
                <ActivityBarItem
                    icon={Cpu}
                    label="EA Manager"
                    active={activeActivity === 'ea_manager'}
                    onClick={() => onActivityChange('ea_manager')}
                />
                <ActivityBarItem
                    icon={LineChart}
                    label="Backtests"
                    active={activeActivity === 'backtests'}
                    onClick={() => onActivityChange('backtests')}
                />
                <ActivityBarItem
                    icon={FileText}
                    label="NPRD Output"
                    active={activeActivity === 'nprd'}
                    onClick={() => onActivityChange('nprd')}
                />

                <div className="flex-grow" />

                <ActivityBarItem
                    icon={Settings}
                    label="Settings"
                    active={activeActivity === 'settings'}
                    onClick={() => onActivityChange('settings')}
                />
            </div>

            {/* WORKSPACE */}
            <div className="flex-grow flex flex-col h-full relative">

                {/* HEADER / COMMAND BAR */}
                <div className="h-10 bg-surface-start border-b border-border flex items-center px-4 justify-between">
                    <div className="text-xs font-mono text-text-muted">quantmind-ide-desktop</div>
                    <button
                        onClick={onToggleCopilot}
                        className={clsx(
                            "flex items-center gap-2 px-3 py-1 rounded border text-xs font-bold transition-all",
                            isCopilotOpen
                                ? "bg-secondary/20 border-secondary text-secondary"
                                : "bg-white/5 border-white/10 text-text-muted hover:text-text-primary"
                        )}
                    >
                        <BrainCircuit size={14} />
                        <span>COPILOT</span>
                    </button>
                </div>

                <PanelGroup direction="horizontal" className="flex-grow">

                    {/* B. LEFT SIDEBAR (Collapsible) */}
                    <Panel defaultSize={20} minSize={15} maxSize={30} collapsible className="bg-surface-start/50 backdrop-blur-sm border-r border-border">
                        <div className="p-2 h-full flex flex-col">
                            <div className="text-xs font-bold text-text-secondary uppercase tracking-wider mb-2 px-2 shrink-0">
                                {activeActivity.toUpperCase()} EXPLORER
                            </div>
                            <div className="flex-grow overflow-hidden relative">
                                {sidebarContent ? sidebarContent : (
                                    <div className="p-4 text-sm text-text-muted italic">
                                        {activeActivity === 'knowledge' && "Knowledge Base..."}
                                        {activeActivity === 'nprd' && "NPRD Reports..."}
                                        {activeActivity === 'assets' && "Assets Hub Connected"}
                                    </div>
                                )}
                            </div>
                        </div>
                    </Panel>

                    <PanelResizeHandle className="w-1 hover:bg-primary/50 transition-colors" />

                    {/* C. MAIN EDITOR */}
                    <Panel minSize={30}>
                        <PanelGroup direction="vertical">
                            <Panel defaultSize={70} className="relative bg-background/50">
                                <div className="h-full flex flex-col">
                                    {children}
                                </div>
                            </Panel>

                            <PanelResizeHandle className="h-1 hover:bg-primary/50 transition-colors" />

                            {/* E. BOTTOM PANEL */}
                            {bottomPanelOpen && (
                                <Panel defaultSize={30} minSize={10} className="bg-surface-end border-t border-border">
                                    <div className="flex items-center h-8 bg-surface-start border-b border-border px-2">
                                        <span className="text-xs uppercase text-text-secondary font-bold mr-4 cursor-pointer hover:text-primary">Terminal</span>
                                        <span className="text-xs uppercase text-text-muted font-bold mr-4 cursor-pointer hover:text-primary">Output</span>
                                        <div className="flex-grow" />
                                        <button onClick={() => setBottomPanelOpen(false)}>×</button>
                                    </div>
                                    <div className="p-2 font-mono text-xs text-text-secondary">
                                        &gt; QuantMind System v1.0 initialized...<br />
                                        &gt; Connected to MT5: OK<br />
                                        &gt; Assets Hub: LINKED
                                    </div>
                                </Panel>
                            )}
                        </PanelGroup>
                    </Panel>

                    {/* D. RIGHT SIDEBAR (COPILOT) */}
                    {isCopilotOpen && (
                        <>
                            <PanelResizeHandle className="w-1 hover:bg-secondary/50 transition-colors" />
                            <Panel defaultSize={25} minSize={20} maxSize={40} className="bg-surface-start/50 backdrop-blur-sm border-l border-border relative">
                                {rightSidebar}
                            </Panel>
                        </>
                    )}

                </PanelGroup>

                {/* STATUS BAR */}
                <div className="h-6 w-full bg-primary/10 border-t border-primary/20 flex items-center px-2 text-[10px] text-primary font-mono select-none">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                        <span>SYSTEM: ONLINE</span>
                    </div>
                </div>
            </div>
        </div>
    );
}
