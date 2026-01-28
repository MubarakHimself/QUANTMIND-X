import { useState, useEffect } from "react";
import Shell from "./components/layout/Shell";
import FileTree from "./components/explorer/FileTree";
import SettingsView from "./components/settings/SettingsView";
import CodeEditor from "./components/editor/CodeEditor";
import CopilotPanel from "./components/copilot/CopilotPanel";
import { homeDir } from '@tauri-apps/api/path';
import { readTextFile, writeTextFile } from '@tauri-apps/plugin-fs';
import "./index.css";

function App() {
  const [activeActivity, setActiveActivity] = useState('code');
  const [currentPath, setCurrentPath] = useState<string>('');

  // Editor State
  const [activeFile, setActiveFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>("");
  const [isDirty, setIsDirty] = useState(false);

  // Layout State
  const [isCopilotOpen, setIsCopilotOpen] = useState(true);

  // Initialize with Home Directory
  useEffect(() => {
    const init = async () => {
      try {
        const home = await homeDir();
        setCurrentPath(home);
      } catch (e) {
        console.error("Failed to get home dir", e);
      }
    }
    init();
  }, []);

  const handleFileSelect = async (path: string) => {
    console.log("Opening file:", path);
    try {
      const content = await readTextFile(path);
      setActiveFile(path);
      setFileContent(content);
      setIsDirty(false);
    } catch (err) {
      console.error("Failed to read file:", err);
      alert("Error reading file. Check permissions.");
    }
  };

  const handleSave = async (content: string) => {
    if (!activeFile) return;
    try {
      await writeTextFile(activeFile, content);
      setIsDirty(false);
      console.log("File saved:", activeFile);
    } catch (err) {
      console.error("Failed to save file:", err);
      alert("Failed to save file.");
    }
  };

  const handleEditorChange = (val: string) => {
    if (!isDirty && val !== fileContent) {
      setIsDirty(true);
    }
  };

  // Switch Content based on Activity Bar
  const renderSidebarContent = () => {
    switch (activeActivity) {
      case 'code':
        return currentPath ? (
          <FileTree rootPath={currentPath} onFileSelect={handleFileSelect} />
        ) : null;
      case 'knowledge':
        return (
          <div className="p-4">
            <h3 className="text-sm font-bold text-text-primary mb-2">Knowledge Graph</h3>
            <div className="text-xs text-text-muted">Loading graph nodes...</div>
          </div>
        );
      case 'assets':
        return (
          <div className="p-4">
            <h3 className="text-sm font-bold text-text-primary mb-2">Assets Hub</h3>
            <div className="text-xs text-text-muted">Strategies: 5<br />Indicators: 12</div>
          </div>
        );
      case 'ea_manager':
        return (
          <div className="p-4">
            <h3 className="text-sm font-bold text-text-primary mb-2">EA Management</h3>
            <ul className="text-xs text-text-secondary space-y-2 mt-2">
              <li className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-green-500"></span> Kelly Criterion</li>
              <li className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-yellow-500"></span> Strategy Router</li>
              <li className="flex items-center gap-2"><span className="w-2 h-2 rounded-full bg-blue-500"></span> Bot Tagger</li>
            </ul>
          </div>
        );
      case 'backtests':
        return (
          <div className="p-4">
            <h3 className="text-sm font-bold text-text-primary mb-2">Backtest History</h3>
            <div className="text-xs text-text-muted">No recent runs.</div>
          </div>
        );
      case 'nprd':
        return (
          <div className="p-4">
            <h3 className="text-sm font-bold text-text-primary mb-2">NPRD Reports</h3>
            <div className="text-xs text-text-muted">No documents found.</div>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="dark">
      <Shell
        activeActivity={activeActivity}
        onActivityChange={setActiveActivity}
        sidebarContent={renderSidebarContent()}
        rightSidebar={<CopilotPanel />}
        isCopilotOpen={isCopilotOpen}
        onToggleCopilot={() => setIsCopilotOpen(!isCopilotOpen)}
      >
        <div className="h-full flex flex-col relative bg-background">
          {/* DIRTY INDICATOR */}
          {activeFile && isDirty && (
            <div className="absolute top-0 right-4 p-2 z-10 pointer-events-none">
              <span className="text-yellow-500 font-bold text-[10px] tracking-widest bg-black/80 px-2 py-1 rounded border border-yellow-500/20 shadow-[0_0_10px_rgba(234,179,8,0.2)]">● UNSAVED</span>
            </div>
          )}

          {activeActivity === 'settings' ? (
            <SettingsView />
          ) : (
            activeFile ? (
              <CodeEditor
                key={activeFile}
                filePath={activeFile}
                initialContent={fileContent}
                onSave={handleSave}
                onChange={handleEditorChange}
              />
            ) : (
              <div className="flex items-center justify-center h-full text-text-muted">
                <div className="text-center opacity-30">
                  {activeActivity === 'code' && (
                    <>
                      <div className="text-6xl mb-4">⌘</div>
                      <p>Select a file to edit</p>
                    </>
                  )}
                  {activeActivity === 'ea_manager' && (
                    <>
                      <div className="text-6xl mb-4">⚙️</div>
                      <p>Select an EA to configure</p>
                    </>
                  )}
                  {activeActivity === 'knowledge' && (
                    <>
                      <div className="text-6xl mb-4">📖</div>
                      <p>Knowledge Base</p>
                    </>
                  )}
                </div>
              </div>
            )
          )}
        </div>
      </Shell>
    </div>
  );
}

export default App;
