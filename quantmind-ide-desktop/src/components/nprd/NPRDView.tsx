import React, { useState, useEffect, useRef } from 'react';
import { nprdApi, NPRDJob } from '../../services/nprdApi';
import { Play, RotateCw, Terminal, FileJson, AlertTriangle, CheckCircle, XCircle, Search } from 'lucide-react';

export const NPRDView: React.FC = () => {
    const [url, setUrl] = useState('');
    const [model, setModel] = useState<'gemini' | 'qwen'>('gemini');
    const [jobId, setJobId] = useState<string | null>(null);
    const [job, setJob] = useState<NPRDJob | null>(null);
    const [activeTab, setActiveTab] = useState<'output' | 'result'>('output');
    const logsEndRef = useRef<HTMLDivElement>(null);

    // Auto-scroll logs
    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [job?.logs]);

    // Polling Logic
    useEffect(() => {
        if (!jobId) return;
        const interval = setInterval(async () => {
            try {
                const status = await nprdApi.getJobStatus(jobId);
                setJob(status);
                if (status.status === 'COMPLETED') setActiveTab('result'); // Auto switch
                if (status.status === 'COMPLETED' || status.status === 'FAILED') clearInterval(interval);
            } catch (e) { console.error(e); }
        }, 1000);
        return () => clearInterval(interval);
    }, [jobId]);

    const handleRun = async () => {
        if (!url) return;
        setJob(null);
        setActiveTab('output');
        try {
            const res = await nprdApi.analyzeVideo(url, model);
            setJobId(res.jobId);
        } catch (e: any) { alert(e.message); }
    };

    return (
        <div className="flex flex-col h-full bg-[#1e1e1e] text-[#cccccc] font-sans selection:bg-[#264f78]">

            {/* Top Bar / Command Palette Style */}
            <div className="flex items-center px-4 py-2 bg-[#252526] border-b border-[#333333] shadow-sm">
                <span className="text-[#007acc] mr-3 font-semibold text-xs tracking-wider">NPRD</span>

                {/* Search / Command Input */}
                <div className="flex-1 max-w-3xl flex items-center bg-[#3c3c3c] rounded-sm border border-[#3c3c3c] focus-within:border-[#007acc] h-7">
                    <Search size={14} className="ml-2 text-[#cccccc]" />
                    <input
                        type="text"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        placeholder="Paste YouTube URL..."
                        className="flex-1 bg-transparent border-none text-sm px-2 outline-none placeholder-[#858585] h-full"
                        onKeyDown={(e) => e.key === 'Enter' && handleRun()}
                    />
                    <select
                        value={model}
                        onChange={(e) => setModel(e.target.value as any)}
                        className="bg-transparent border-l border-[#444444] text-xs px-2 h-full outline-none text-[#cccccc] focus:bg-[#2a2d2e]"
                    >
                        <option value="gemini">Gemini 1.5</option>
                        <option value="qwen">Qwen-VL</option>
                    </select>
                </div>

                <div className="ml-3 flex items-center gap-2">
                    <button
                        onClick={handleRun}
                        disabled={!!jobId && job?.status !== 'COMPLETED' && job?.status !== 'FAILED'}
                        className="bg-[#0e639c] hover:bg-[#1177bb] text-white p-1 rounded-sm disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Run Analysis"
                    >
                        <Play size={14} fill="currentColor" />
                    </button>
                </div>
            </div>

            {/* Split View Content */}
            <div className="flex-1 flex overflow-hidden">

                {/* Sidebar: Explorer / Details */}
                <div className="w-64 bg-[#252526] border-r border-[#333333] flex flex-col hidden md:flex">
                    <div className="px-4 py-2 text-xs font-bold text-[#bbbbbb] uppercase tracking-wider flex justify-between items-center bg-[#252526]">
                        <span>Explorer</span>
                        <span className="text-[#858585]">...</span>
                    </div>

                    <div className="flex-1 overflow-y-auto">
                        <div className="px-0">
                            <div className="px-4 py-1 text-xs text-[#007acc] font-medium flex items-center gap-1 bg-[#37373d]">
                                <span className="rotate-90">›</span> JOB_DETAILS
                            </div>
                            <div className="px-6 py-2 grid grid-cols-1 gap-2 text-xs font-mono text-[#cccccc]">
                                <div className="flex justify-between">
                                    <span className="text-[#858585]">ID:</span>
                                    <span className="truncate w-24 text-right" title={job?.id || '-'}>{job?.id || '-'}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-[#858585]">Status:</span>
                                    <span style={{ color: getStatusColorHex(job?.status) }}>{job?.status || 'IDLE'}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-[#858585]">Model:</span>
                                    <span>{model}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Main Editor Area */}
                <div className="flex-1 flex flex-col bg-[#1e1e1e]">

                    {/* Tabs / Breadcrumbs */}
                    <div className="flex bg-[#2d2d2d] h-9 items-center px-0 border-b border-[#252526]">
                        <Tab
                            active={activeTab === 'output'}
                            onClick={() => setActiveTab('output')}
                            icon={<Terminal size={14} className="text-[#cccccc]" />}
                            title="Terminal"
                        />
                        <Tab
                            active={activeTab === 'result'}
                            onClick={() => setActiveTab('result')}
                            icon={<FileJson size={14} className="text-[#cbcb41]" />}
                            title="nprd_output.json"
                        />
                    </div>

                    {/* Editor Content */}
                    <div className="flex-1 overflow-auto relative font-mono text-sm leading-5">
                        {activeTab === 'result' ? (
                            job?.result ? (
                                <div className="p-4">
                                    {/* Line Numbers + Content Mockup */}
                                    <div className="flex">
                                        <div className="text-[#858585] text-right pr-4 select-none opacity-50 text-xs leading-5">
                                            {Array.from({ length: 10 }).map((_, i) => <div key={i}>{i + 1}</div>)}
                                        </div>
                                        <pre className="text-[#ce9178] whitespace-pre-wrap">
                                            {JSON.stringify(job.result, null, 2)}
                                        </pre>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex h-full items-center justify-center text-[#3c3c3c]">
                                    <div className="text-center">
                                        <FileJson size={64} strokeWidth={0.5} className="mx-auto mb-4 opacity-20" />
                                        <p>No results generated.</p>
                                    </div>
                                </div>
                            )
                        ) : (
                            <div className="p-4 text-[#cccccc]">
                                <div className="text-[#858585] mb-2 selection:bg-[#264f78]">NPRD Integrated Terminal</div>
                                {jobId ? (
                                    <>
                                        <div className="mb-2">
                                            <span className="text-[#89d185]">user@quantmind</span>:<span className="text-[#3a96dd]">~/nprd</span>$ analyze --url "{url}" --model {model}
                                        </div>
                                        {job?.logs.map((log, i) => (
                                            <div key={i} className="whitespace-pre-wrap break-all">
                                                {formatLog(log)}
                                            </div>
                                        ))}
                                        {job?.status === 'WAITING_AUTH' && (
                                            <div className="mt-2 text-[#cca700] bg-[#333300] p-1 inline-block border border-[#665500]">
                                                [Auth Required] Please authenticate in your host browser.
                                            </div>
                                        )}
                                        <div ref={logsEndRef} className="h-4" />
                                    </>
                                ) : (
                                    <div className="text-[#858585]">
                                        Ready. Type a URL above to start.
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                </div>

            </div>

            {/* Status Bar */}
            <div className="h-6 bg-[#007acc] text-white flex items-center px-3 text-[11px] justify-between select-none">
                <div className="flex gap-4">
                    <div className="flex items-center gap-1 hover:bg-[#ffffff20] px-1 rounded-sm cursor-pointer">
                        <CheckCircle size={10} /> <span>Ready</span>
                    </div>
                    {jobId && (
                        <div className="flex items-center gap-1 hover:bg-[#ffffff20] px-1 rounded-sm cursor-pointer">
                            <span>{job?.progress || 0}%</span>
                        </div>
                    )}
                </div>
                <div className="flex gap-4">
                    <span className="hover:bg-[#ffffff20] px-1 cursor-pointer">Ln 1, Col 1</span>
                    <span className="hover:bg-[#ffffff20] px-1 cursor-pointer">UTF-8</span>
                    <span className="hover:bg-[#ffffff20] px-1 cursor-pointer">JSON</span>
                    <span className="hover:bg-[#ffffff20] px-1 cursor-pointer">NPRD</span>
                </div>
            </div>
        </div>
    );
};

// ... Subcomponents (Tab, Helpers) remain structurally similar but tuned for density ...

const Tab = ({ active, onClick, icon, title }: any) => (
    <div
        onClick={onClick}
        className={`
            flex items-center gap-2 px-3 h-full border-r border-[#252526] cursor-pointer select-none text-xs
            ${active ? 'bg-[#1e1e1e] text-[#ffffff] border-t-[2px] border-t-[#007acc]' : 'bg-[#2d2d2d] text-[#969696] hover:bg-[#2a2d2e]'}
        `}
    >
        {icon}
        <span>{title}</span>
        {active && <span className="ml-2 opacity-50 hover:opacity-100 hover:bg-[#444] rounded-sm p-0.5"><XCircle size={10} /></span>}
    </div>
);

function getStatusColorHex(status?: string) {
    if (status === 'COMPLETED') return '#89d185';
    if (status === 'FAILED') return '#f14c4c';
    if (status === 'WAITING_AUTH') return '#cca700';
    return '#3a96dd';
}

function formatLog(log: string) {
    // VS Code Terminal Colors
    if (log.includes('[Script]')) return <span className="text-[#cccccc]">{log.replace('[Script]', '')}</span>;
    if (log.includes('[Script Error]')) return <span className="text-[#f14c4c]">{log}</span>;
    if (log.includes('[Gemini]')) return <span className="text-[#4ec9b0]">{log}</span>;
    if (log.includes('WARN')) return <span className="text-[#cca700]">{log}</span>;
    return <span className="text-[#cccccc]">{log}</span>;
}
