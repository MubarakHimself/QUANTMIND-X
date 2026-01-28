import { useState } from 'react';
import { Send, Bot, Sparkles, BrainCircuit } from 'lucide-react';
import clsx from 'clsx';

const AgentTab = ({ active, label, icon: Icon, onClick }: any) => (
    <button
        onClick={onClick}
        className={clsx(
            "flex-1 flex flex-col items-center justify-center py-3 border-b-2 transition-all text-xs font-bold uppercase",
            active
                ? "border-secondary text-secondary bg-secondary/5"
                : "border-transparent text-text-muted hover:text-text-primary hover:bg-white/5"
        )}
    >
        <Icon size={16} className="mb-1" />
        {label}
    </button>
);

export default function CopilotPanel() {
    const [activeAgent, setActiveAgent] = useState('copilot');
    const [input, setInput] = useState('');

    return (
        <div className="h-full flex flex-col bg-surface-start">
            {/* AGENT NAVBAR */}
            <div className="flex w-full border-b border-white/10">
                <AgentTab
                    label="Copilot"
                    icon={Sparkles}
                    active={activeAgent === 'copilot'}
                    onClick={() => setActiveAgent('copilot')}
                />
                <AgentTab
                    label="Quant"
                    icon={BrainCircuit}
                    active={activeAgent === 'quant'}
                    onClick={() => setActiveAgent('quant')}
                />
                <AgentTab
                    label="Executor"
                    icon={Bot}
                    active={activeAgent === 'executor'}
                    onClick={() => setActiveAgent('executor')}
                />
            </div>

            {/* CHAT AREA */}
            <div className="flex-grow overflow-y-auto p-4 space-y-4">
                {activeAgent === 'copilot' && (
                    <div className="text-center mt-10 opacity-50">
                        <Sparkles size={48} className="mx-auto mb-4 text-secondary" />
                        <h3 className="font-bold">QuantMind Copilot</h3>
                        <p className="text-xs mt-2">Ask me anything about your code or strategy.</p>
                    </div>
                )}
                {activeAgent === 'quant' && (
                    <div className="text-center mt-10 opacity-50">
                        <BrainCircuit size={48} className="mx-auto mb-4 text-primary" />
                        <h3 className="font-bold">Quant Agent</h3>
                        <p className="text-xs mt-2">Specialized in mathematical modeling and verification.</p>
                    </div>
                )}
            </div>

            {/* INPUT AREA */}
            <div className="p-4 border-t border-white/10 bg-surface-end">
                <div className="relative">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder={`Message ${activeAgent.toUpperCase()}...`}
                        className="w-full bg-black/30 border border-white/10 rounded-lg pl-4 pr-10 py-3 text-sm focus:outline-none focus:border-secondary transition-colors"
                    />
                    <button className="absolute right-2 top-2 p-1 text-secondary hover:text-white transition-colors">
                        <Send size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
}
