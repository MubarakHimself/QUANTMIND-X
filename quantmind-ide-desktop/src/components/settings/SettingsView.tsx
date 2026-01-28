import { useState } from 'react';
import { Save, Key, Cpu, MessageSquare } from 'lucide-react';

const SectionHeader = ({ icon: Icon, title }: any) => (
    <div className="flex items-center gap-2 mb-4 text-primary border-b border-border pb-2">
        <Icon size={18} />
        <h3 className="font-bold uppercase tracking-wider text-sm">{title}</h3>
    </div>
);

const InputField = ({ label, type = "text", placeholder, value }: any) => (
    <div className="mb-4">
        <label className="block text-xs uppercase text-text-secondary mb-1 font-bold">{label}</label>
        <input
            type={type}
            placeholder={placeholder}
            defaultValue={value}
            className="w-full bg-surface-start border border-white/10 rounded p-2 text-sm text-text-primary focus:border-primary focus:outline-none transition-colors"
        />
    </div>
);

export default function SettingsView() {
    const [saved, setSaved] = useState(false);

    const handleSave = () => {
        // TODO: Persist to Tauri Store or local config file
        setSaved(true);
        setTimeout(() => setSaved(false), 2000);
    };

    return (
        <div className="p-8 max-w-4xl mx-auto h-full overflow-y-auto">
            <h1 className="text-2xl font-bold mb-8 flex items-center gap-3">
                <SettingsIcon size={32} className="text-secondary" />
                System Configuration
            </h1>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">

                {/* API KEYS */}
                <div className="glass-panel p-6 rounded-lg">
                    <SectionHeader icon={Key} title="API Credentials" />
                    <InputField label="OpenAI API Key" type="password" placeholder="sk-..." />
                    <InputField label="Anthropic API Key" type="password" placeholder="sk-ant-..." />
                    <InputField label="QuantMind Cloud Token" type="password" placeholder="qm_..." />
                </div>

                {/* REMOTE MT5 BRIDGE (VPS) */}
                <div className="glass-panel p-6 rounded-lg">
                    <SectionHeader icon={Cpu} title="Remote MT5 Bridge (VPS)" />
                    <div className="mb-4 text-xs text-text-muted">
                        Connect to the Windows VPS running MetaTrader 5.
                    </div>
                    <InputField label="VPS Host / IP" placeholder="e.g. 192.168.1.100" />
                    <InputField label="Bridge Port" placeholder="e.g. 5005" value="5005" />
                    <InputField label="Auth Token" type="password" placeholder="Bridge secret key" />
                </div>

                {/* MODEL CONFIG */}
                <div className="glass-panel p-6 rounded-lg">
                    <SectionHeader icon={Cpu} title="Model Inference" />
                    <div className="mb-4">
                        <label className="block text-xs uppercase text-text-secondary mb-1 font-bold">Planner Model</label>
                        <select className="w-full bg-surface-start border border-white/10 rounded p-2 text-sm text-text-primary focus:border-primary outline-none">
                            <option>gpt-4-turbo</option>
                            <option>claude-3-opus</option>
                            <option>local-mistral</option>
                        </select>
                    </div>
                    <div className="mb-4">
                        <label className="block text-xs uppercase text-text-secondary mb-1 font-bold">Coder Model</label>
                        <select className="w-full bg-surface-start border border-white/10 rounded p-2 text-sm text-text-primary focus:border-primary outline-none">
                            <option>claude-3.5-sonnet</option>
                            <option>gpt-4o</option>
                        </select>
                    </div>
                </div>

                {/* SYSTEM PROMPTS */}
                <div className="glass-panel p-6 rounded-lg md:col-span-2">
                    <SectionHeader icon={MessageSquare} title="Agent System Prompts" />
                    <div className="mb-4">
                        <label className="block text-xs uppercase text-text-secondary mb-1 font-bold">Core Persona (Global)</label>
                        <textarea
                            className="w-full h-32 bg-surface-start border border-white/10 rounded p-2 text-sm text-text-primary focus:border-primary focus:outline-none transition-colors font-mono"
                            defaultValue="You are QuantMind, an elite quantitative trading architect."
                        />
                    </div>
                </div>

            </div>

            <div className="mt-8 flex justify-end">
                <button
                    onClick={handleSave}
                    className="flex items-center gap-2 px-6 py-3 bg-primary/20 hover:bg-primary/30 text-primary border border-primary/50 rounded transition-all active:scale-95"
                >
                    <Save size={18} />
                    {saved ? "Saved!" : "Save Configuration"}
                </button>
            </div>
        </div>
    );
}

import { Settings as SettingsIcon } from 'lucide-react';
