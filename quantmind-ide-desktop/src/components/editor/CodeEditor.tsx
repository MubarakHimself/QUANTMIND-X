import { useRef } from "react";
import Editor, { OnMount } from "@monaco-editor/react";

interface CodeEditorProps {
    filePath: string;
    initialContent: string;
    onSave: (content: string) => void;
    onChange?: (content: string) => void;
}

const getLanguage = (path: string) => {
    if (path.endsWith('.py')) return 'python';
    if (path.endsWith('.js') || path.endsWith('.ts') || path.endsWith('.tsx')) return 'typescript';
    if (path.endsWith('.json')) return 'json';
    if (path.endsWith('.md')) return 'markdown';
    if (path.endsWith('.css')) return 'css';
    if (path.endsWith('.mq5') || path.endsWith('.mqh')) return 'cpp'; // Approximate MQL5 as CPP
    return 'plaintext';
};

export default function CodeEditor({ filePath, initialContent, onSave, onChange }: CodeEditorProps) {
    const editorRef = useRef<any>(null);

    const handleEditorDidMount: OnMount = (editor, monaco) => {
        editorRef.current = editor;

        // Custom Keybinding for Save (Cmd+S / Ctrl+S)
        editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
            const value = editor.getValue();
            onSave(value);
        });
    };

    const handleChange = (value: string | undefined) => {
        if (onChange && value !== undefined) {
            onChange(value);
        }
    };

    return (
        <div className="h-full w-full overflow-hidden pt-2 bg-[#0d0f14]">
            {/* HEADER */}
            <div className="flex items-center px-4 py-2 border-b border-white/10 text-xs font-mono text-text-secondary bg-[#0d0f14]">
                <span className="mr-2 opacity-50">FILE:</span>
                <span className="text-primary">{filePath}</span>
            </div>

            <Editor
                height="calc(100% - 32px)"
                language={getLanguage(filePath)}
                value={initialContent}
                theme="vs-dark"
                onMount={handleEditorDidMount}
                onChange={handleChange}
                options={{
                    minimap: { enabled: false },
                    fontSize: 13,
                    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
                    scrollBeyondLastLine: false,
                    automaticLayout: true,
                    padding: { top: 16 }
                }}
            />
        </div>
    );
}
